"""The Responses API: POST /v1/responses and friends."""

from __future__ import annotations

import base64
import copy
import time
from typing import Any

from starlette.requests import Request
from starlette.responses import Response

from ..common import SSE, cfg_of, endpoint, header_float, json_response, read_json, simulate_latency, sse_response
from ..engine import (
    analyze_responses_input,
    check_int,
    check_sampling,
    decide_tool_calls,
    fake_logprobs,
    generate_text,
    make_tool_call,
    parse_responses_tools,
    reasoning_token_count,
    require_model,
    resolve_tool_choice,
    response_format_kind,
)
from ..errors import APIError, invalid, not_found
from ..ids import hex_id, make_rng
from ..images import parse_size, render_cat_b64
from ..meow import Flavor, Meower
from ..store import ResponseStore, StoredResponse
from ..tokens import count_json_tokens, split_tokens

REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh")
SUMMARY_MODES = ("auto", "concise", "detailed")
TRUNCATION_MODES = ("auto", "disabled")


def _store(request: Request) -> ResponseStore:
    return request.app.state.response_store


@endpoint
async def create_response(request: Request) -> Response:
    cfg = cfg_of(request)
    body = await read_json(request)
    created_at = int(time.time())

    model = require_model(body)
    check_sampling(body)
    max_output_tokens = body.get("max_output_tokens")
    check_int(max_output_tokens, "max_output_tokens", 16)
    max_tool_calls = body.get("max_tool_calls")
    check_int(max_tool_calls, "max_tool_calls", 1)
    top_logprobs = body.get("top_logprobs")
    check_int(top_logprobs, "top_logprobs", 0, 20)
    stream = bool(body.get("stream"))
    store = body.get("store", True) is not False
    background = bool(body.get("background"))
    if background and not store:
        raise invalid("Background mode requires 'store' to be true.", "background")
    truncation = body.get("truncation") or "disabled"
    if truncation not in TRUNCATION_MODES:
        raise invalid(f"Invalid value: '{truncation}'. Supported values are: 'auto' and 'disabled'.", "truncation", "invalid_value")
    metadata = body.get("metadata") or {}
    if not isinstance(metadata, dict) or len(metadata) > 16:
        raise invalid("Invalid 'metadata': expected an object with at most 16 keys.", "metadata", "invalid_value")
    include = body.get("include") or []
    if not isinstance(include, list):
        raise invalid("Invalid type for 'include': expected an array of strings.", "include", "invalid_type")

    reasoning = body.get("reasoning") or {}
    if not isinstance(reasoning, dict):
        raise invalid("Invalid type for 'reasoning': expected an object.", "reasoning", "invalid_type")
    effort = reasoning.get("effort")
    if effort is not None and effort not in REASONING_EFFORTS:
        raise invalid(f"Invalid value: '{effort}'. Supported values are: {', '.join(repr(e) for e in REASONING_EFFORTS)}.", "reasoning.effort", "invalid_value")
    summary_mode = reasoning.get("summary") or reasoning.get("generate_summary")
    if summary_mode is not None and summary_mode not in SUMMARY_MODES:
        raise invalid(f"Invalid value: '{summary_mode}'. Supported values are: 'auto', 'concise', and 'detailed'.", "reasoning.summary", "invalid_value")

    text_cfg = body.get("text") or {}
    if not isinstance(text_cfg, dict):
        raise invalid("Invalid type for 'text': expected an object.", "text", "invalid_type")
    fmt_kind, schema, strict = response_format_kind(text_cfg.get("format"), param="text.format")

    # previous_response_id chaining
    prev_id = body.get("previous_response_id")
    prior: list[dict[str, Any]] = []
    if prev_id is not None:
        if body.get("conversation") is not None:
            raise invalid("Cannot specify both 'previous_response_id' and 'conversation'.", "previous_response_id")
        stored = _store(request).get(prev_id) if isinstance(prev_id, str) else None
        if stored is None:
            raise APIError(f"Previous response with id '{prev_id}' not found.", status=400, param="previous_response_id", code="previous_response_not_found")
        prior = stored.conversation

    prompt, input_items = analyze_responses_input(body.get("input"), body.get("instructions"), prior)
    specs = parse_responses_tools(body.get("tools"))
    policy = resolve_tool_choice(body.get("tool_choice"), specs, api="responses")
    parallel = body.get("parallel_tool_calls") is not False
    prompt.tokens += count_json_tokens(body.get("tools"))

    rng = make_rng(cfg.seed, body.get("seed"), body)
    tool_prob = header_float(request, "x-dummyllm-tool-probability")
    want_logprobs = "message.output_text.logprobs" in include
    want_encrypted = "reasoning.encrypted_content" in include
    meow = Meower(rng)

    # ---- "inference" -----------------------------------------------------
    output: list[dict[str, Any]] = []
    logprobs_by_item: dict[int, list[dict[str, Any]]] = {}
    tokens_by_item: dict[int, list[str]] = {}
    output_tokens = 0
    incomplete = False

    reasoning_tokens = reasoning_token_count(rng, effort)
    budget = None if max_output_tokens is None else max_output_tokens
    if reasoning_tokens or (effort not in (None, "none") and summary_mode):
        item: dict[str, Any] = {"id": hex_id("rs"), "type": "reasoning", "summary": []}
        if summary_mode:
            item["summary"] = [{"type": "summary_text", "text": meow.reasoning_summary()}]
        if want_encrypted:
            item["encrypted_content"] = base64.b64encode(rng.randbytes(48)).decode()
        output.append(item)
        if budget is not None and reasoning_tokens >= budget:
            reasoning_tokens = budget
            incomplete = True
        output_tokens += reasoning_tokens
        if budget is not None:
            budget -= reasoning_tokens

    if not incomplete:
        to_call = decide_tool_calls(rng, cfg, policy, prompt, parallel=parallel, probability_override=tool_prob)
        if max_tool_calls is not None:
            to_call = to_call[:max_tool_calls]
        drew = False
        for spec in to_call:
            call = make_tool_call(rng, spec, prompt)
            if spec.kind == "function":
                output.append(
                    {"id": hex_id("fc"), "type": "function_call", "status": "completed", "arguments": call.arguments, "call_id": call.id, "name": call.name}
                )
                output_tokens += len(split_tokens(call.arguments)) + 3
            elif spec.kind == "custom":
                output.append(
                    {"id": hex_id("ctc"), "type": "custom_tool_call", "status": "completed", "call_id": call.id, "input": call.arguments, "name": call.name}
                )
                output_tokens += len(split_tokens(call.arguments)) + 3
            else:
                output.append(_image_item(rng, spec.options, call.arguments))
                drew = True

        client_calls = any(it["type"] in ("function_call", "custom_tool_call") for it in output)
        if not client_calls:
            flavor = Flavor(saw_image=prompt.saw_image, after_tool_result=prompt.after_tool_result, drew_image=drew)
            tokens, finish = generate_text(
                rng, cfg, flavor=flavor, max_tokens=budget, fmt_kind=fmt_kind, schema=schema, strict=strict
            )
            incomplete = finish == "length"
            idx = len(output)
            tokens_by_item[idx] = tokens
            lps = fake_logprobs(rng, tokens, top_logprobs or 0) if want_logprobs else []
            logprobs_by_item[idx] = lps
            output.append(
                {
                    "id": hex_id("msg"),
                    "type": "message",
                    "status": "incomplete" if incomplete else "completed",
                    "content": [{"type": "output_text", "annotations": [], "logprobs": lps, "text": "".join(tokens)}],
                    "role": "assistant",
                }
            )
            output_tokens += len(tokens)

    usage = {
        "input_tokens": prompt.tokens,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
        "total_tokens": prompt.tokens + output_tokens,
    }

    rid = hex_id("resp")
    final = _response_object(
        rid,
        created_at,
        body,
        model=model,
        status="incomplete" if incomplete else "completed",
        output=output,
        usage=usage,
        store=store,
        background=background,
        effort=effort,
        summary_mode=summary_mode,
        truncation=truncation,
        metadata=metadata,
    )
    if incomplete:
        final["incomplete_details"] = {"reason": "max_output_tokens"}

    if store:
        stored_inputs = [_with_id(it) for it in input_items]
        # Instructions are not carried over between chained responses (matches OpenAI).
        conversation = [*prior, *stored_inputs, *output]
        _store(request).put(rid, StoredResponse(response=final, input_items=stored_inputs, conversation=conversation))

    if stream:
        events = _stream_events(final, tokens_by_item, logprobs_by_item)
        return sse_response(events, cfg, done_sentinel=False)

    if background:
        queued = copy.deepcopy(final)
        queued.update(status="queued", output=[], usage=None, incomplete_details=None, completed_at=None)
        return json_response(queued)

    await simulate_latency(cfg, output_tokens)
    return json_response(final)


def _with_id(item: dict[str, Any]) -> dict[str, Any]:
    out = dict(item)
    if "id" not in out:
        prefix = {"message": "msg", "function_call_output": "fco", "custom_tool_call_output": "ctco"}.get(out.get("type", ""), "item")
        out["id"] = hex_id(prefix)
    if out.get("type") == "message":
        out.setdefault("status", "completed")
    return out


def _image_item(rng, options: dict[str, Any], revised_prompt: str) -> dict[str, Any]:
    size_raw = options.get("size", "auto")
    size = parse_size(size_raw, default=(1024, 1024)) or (1024, 1024)
    return {
        "id": hex_id("ig"),
        "type": "image_generation_call",
        "status": "completed",
        "background": options.get("background", "opaque") if options.get("background") != "auto" else "opaque",
        "output_format": "png",
        "quality": options.get("quality", "medium") if options.get("quality") not in (None, "auto") else "medium",
        "result": render_cat_b64(size[0], size[1], rng),
        "revised_prompt": revised_prompt,
        "size": f"{size[0]}x{size[1]}",
    }


def _response_object(
    rid: str,
    created_at: int,
    body: dict[str, Any],
    *,
    model: str,
    status: str,
    output: list[dict[str, Any]],
    usage: dict[str, Any] | None,
    store: bool,
    background: bool,
    effort: str | None,
    summary_mode: str | None,
    truncation: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    text_cfg = dict(body.get("text") or {})
    text_cfg.setdefault("format", {"type": "text"})
    text_cfg.setdefault("verbosity", "medium")
    return {
        "id": rid,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "background": background,
        "completed_at": int(time.time()) if status in ("completed", "incomplete") else None,
        "error": None,
        "incomplete_details": None,
        "instructions": body.get("instructions"),
        "max_output_tokens": body.get("max_output_tokens"),
        "max_tool_calls": body.get("max_tool_calls"),
        "model": model,
        "output": [_public_item(it) for it in output],
        "parallel_tool_calls": body.get("parallel_tool_calls") is not False,
        "previous_response_id": body.get("previous_response_id"),
        "prompt_cache_key": body.get("prompt_cache_key"),
        "reasoning": {"effort": effort, "summary": summary_mode},
        "safety_identifier": body.get("safety_identifier"),
        "service_tier": "default",
        "store": store,
        "temperature": body.get("temperature", 1.0),
        "text": text_cfg,
        "tool_choice": body.get("tool_choice", "auto"),
        "tools": body.get("tools") or [],
        "top_logprobs": body.get("top_logprobs", 0),
        "top_p": body.get("top_p", 1.0),
        "truncation": truncation,
        "usage": usage,
        "user": body.get("user"),
        "metadata": metadata,
    }


def _public_item(item: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in item.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# Streaming


def _stream_events(
    final: dict[str, Any],
    tokens_by_item: dict[int, list[str]],
    logprobs_by_item: dict[int, list[dict[str, Any]]],
) -> list[SSE]:
    events: list[SSE] = []

    def emit(etype: str, tokens: int = 0, **fields: Any) -> None:
        events.append(SSE({"type": etype, "sequence_number": len(events), **fields}, event=etype, tokens=tokens))

    in_progress = copy.deepcopy(final)
    in_progress.update(status="in_progress", output=[], usage=None, incomplete_details=None, completed_at=None)
    emit("response.created", response=in_progress)
    emit("response.in_progress", response=in_progress)

    raw_items = final["output"]
    for oi, item in enumerate(raw_items):
        iid = item["id"]
        itype = item["type"]
        if itype == "message":
            emit("response.output_item.added", output_index=oi, item={**item, "status": "in_progress", "content": []})
            empty_part = {"type": "output_text", "annotations": [], "logprobs": [], "text": ""}
            emit("response.content_part.added", item_id=iid, output_index=oi, content_index=0, part=empty_part)
            toks = tokens_by_item.get(oi, [])
            lps = logprobs_by_item.get(oi, [])
            for j, tok in enumerate(toks):
                emit(
                    "response.output_text.delta",
                    tokens=1,
                    item_id=iid,
                    output_index=oi,
                    content_index=0,
                    delta=tok,
                    logprobs=[lps[j]] if j < len(lps) else [],
                )
            part = item["content"][0]
            emit("response.output_text.done", item_id=iid, output_index=oi, content_index=0, text=part["text"], logprobs=part["logprobs"])
            emit("response.content_part.done", item_id=iid, output_index=oi, content_index=0, part=part)
            emit("response.output_item.done", output_index=oi, item=item)
        elif itype == "function_call":
            emit("response.output_item.added", output_index=oi, item={**item, "status": "in_progress", "arguments": ""})
            for frag in split_tokens(item["arguments"]):
                emit("response.function_call_arguments.delta", tokens=1, item_id=iid, output_index=oi, delta=frag)
            emit("response.function_call_arguments.done", item_id=iid, output_index=oi, name=item["name"], arguments=item["arguments"])
            emit("response.output_item.done", output_index=oi, item=item)
        elif itype == "custom_tool_call":
            emit("response.output_item.added", output_index=oi, item={**item, "status": "in_progress", "input": ""})
            for frag in split_tokens(item["input"]):
                emit("response.custom_tool_call_input.delta", tokens=1, item_id=iid, output_index=oi, delta=frag)
            emit("response.custom_tool_call_input.done", item_id=iid, output_index=oi, input=item["input"])
            emit("response.output_item.done", output_index=oi, item=item)
        elif itype == "reasoning":
            emit("response.output_item.added", output_index=oi, item={**item, "summary": []})
            for si, summ in enumerate(item.get("summary", [])):
                emit("response.reasoning_summary_part.added", item_id=iid, output_index=oi, summary_index=si, part={"type": "summary_text", "text": ""})
                for frag in split_tokens(summ["text"]):
                    emit("response.reasoning_summary_text.delta", tokens=1, item_id=iid, output_index=oi, summary_index=si, delta=frag)
                emit("response.reasoning_summary_text.done", item_id=iid, output_index=oi, summary_index=si, text=summ["text"])
                emit("response.reasoning_summary_part.done", item_id=iid, output_index=oi, summary_index=si, part=summ)
            emit("response.output_item.done", output_index=oi, item=item)
        elif itype == "image_generation_call":
            pending = {k: v for k, v in item.items() if k != "result"}
            emit("response.output_item.added", output_index=oi, item={**pending, "status": "in_progress", "result": None})
            emit("response.image_generation_call.in_progress", item_id=iid, output_index=oi)
            emit("response.image_generation_call.generating", item_id=iid, output_index=oi, tokens=8)
            partials = 0
            for t in final.get("tools", []):
                if isinstance(t, dict) and t.get("type") == "image_generation":
                    partials = max(0, min(3, int(t.get("partial_images") or 0)))
            for pi in range(partials):
                emit(
                    "response.image_generation_call.partial_image",
                    tokens=4,
                    item_id=iid,
                    output_index=oi,
                    partial_image_index=pi,
                    partial_image_b64=item["result"],
                    output_format="png",
                    size=item.get("size"),
                    background=item.get("background"),
                    quality=item.get("quality"),
                )
            emit("response.image_generation_call.completed", item_id=iid, output_index=oi)
            emit("response.output_item.done", output_index=oi, item=item)

    done_type = "response.incomplete" if final["status"] == "incomplete" else "response.completed"
    emit(done_type, response=final)
    return events


# ---------------------------------------------------------------------------
# Retrieval / deletion


@endpoint
async def get_response(request: Request) -> Response:
    rid = request.path_params["response_id"]
    stored = _store(request).get(rid)
    if stored is None:
        raise not_found(f"Response with id '{rid}' not found.", "response_id")
    return json_response(stored.response)


@endpoint
async def delete_response(request: Request) -> Response:
    rid = request.path_params["response_id"]
    if not _store(request).delete(rid):
        raise not_found(f"Response with id '{rid}' not found.", "response_id")
    return json_response({"id": rid, "object": "response", "deleted": True})


@endpoint
async def cancel_response(request: Request) -> Response:
    rid = request.path_params["response_id"]
    stored = _store(request).get(rid)
    if stored is None:
        raise not_found(f"Response with id '{rid}' not found.", "response_id")
    if not stored.response.get("background"):
        raise invalid("Only responses created with background=true can be cancelled.", "response_id")
    # Background responses finish instantly in dummyllm, so there is nothing to cancel.
    return json_response(stored.response)


@endpoint
async def list_input_items(request: Request) -> Response:
    rid = request.path_params["response_id"]
    stored = _store(request).get(rid)
    if stored is None:
        raise not_found(f"Response with id '{rid}' not found.", "response_id")
    order = request.query_params.get("order", "desc")
    if order not in ("asc", "desc"):
        raise invalid(f"Invalid value: '{order}'. Supported values are: 'asc' and 'desc'.", "order", "invalid_value")
    try:
        limit = int(request.query_params.get("limit", "20"))
    except ValueError:
        raise invalid("Invalid 'limit': expected an integer.", "limit", "invalid_type") from None
    if not 1 <= limit <= 100:
        raise invalid("Invalid 'limit': expected a value between 1 and 100.", "limit", "invalid_value")
    items = list(stored.input_items)
    if order == "desc":
        items.reverse()
    after = request.query_params.get("after")
    if after:
        ids = [it.get("id") for it in items]
        items = items[ids.index(after) + 1 :] if after in ids else []
    page = items[:limit]
    return json_response(
        {
            "object": "list",
            "data": page,
            "first_id": page[0]["id"] if page else None,
            "last_id": page[-1]["id"] if page else None,
            "has_more": len(items) > limit,
        }
    )


@endpoint
async def count_input_tokens(request: Request) -> Response:
    body = await read_json(request)
    prior: list[dict[str, Any]] = []
    prev_id = body.get("previous_response_id")
    if prev_id is not None:
        stored = _store(request).get(prev_id)
        if stored is None:
            raise APIError(f"Previous response with id '{prev_id}' not found.", status=400, param="previous_response_id")
        prior = stored.conversation
    prompt, _ = analyze_responses_input(body.get("input"), body.get("instructions"), prior)
    prompt.tokens += count_json_tokens(body.get("tools"))
    return json_response({"object": "response.input_tokens", "input_tokens": prompt.tokens})
