"""POST /v1/chat/completions"""

from __future__ import annotations

from typing import Any

from starlette.requests import Request
from starlette.responses import Response

from ..common import SSE, cfg_of, endpoint, header_float, json_response, read_json, simulate_latency, sse_response
from ..engine import (
    Generation,
    analyze_chat_messages,
    check_int,
    check_sampling,
    decide_tool_calls,
    fake_logprobs,
    fit_tool_calls,
    generate_text,
    make_tool_call,
    normalize_stop,
    parse_chat_tools,
    generate_reasoning,
    resolve_effort,
    require_model,
    resolve_tool_choice,
    response_format_kind,
)
from ..errors import invalid
from ..ids import fingerprint, make_rng, new_id, now
from ..meow import Flavor
from ..tokens import count_json_tokens, split_tokens

REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


@endpoint
async def chat_completions(request: Request) -> Response:
    cfg = cfg_of(request)
    body = await read_json(request)

    model = require_model(body)
    check_sampling(body)
    n = body.get("n", 1)
    if n is None:
        n = 1
    check_int(n, "n", 1, 128)
    check_int(body.get("max_tokens"), "max_tokens", 1)
    check_int(body.get("max_completion_tokens"), "max_completion_tokens", 1)
    max_tokens = body.get("max_completion_tokens") or body.get("max_tokens")
    stops = normalize_stop(body.get("stop"))

    want_logprobs = bool(body.get("logprobs"))
    top_logprobs = body.get("top_logprobs")
    check_int(top_logprobs, "top_logprobs", 0, 20)
    if top_logprobs is not None and not want_logprobs:
        raise invalid("Invalid value for 'top_logprobs': 'logprobs' must be set to true when 'top_logprobs' is specified.", "top_logprobs")

    stream = bool(body.get("stream"))
    stream_options = body.get("stream_options")
    if stream_options is not None and not stream:
        raise invalid("The 'stream_options' parameter is only allowed when 'stream' is enabled.", "stream_options")
    include_usage = bool(isinstance(stream_options, dict) and stream_options.get("include_usage"))

    modalities = body.get("modalities")
    if isinstance(modalities, list) and "audio" in modalities:
        raise invalid("This model does not support audio output. The cat can only meow in text.", "modalities", "unsupported_value")

    effort = body.get("reasoning_effort")
    if effort is not None and effort not in REASONING_EFFORTS:
        raise invalid(
            f"Invalid value: '{effort}'. Supported values are: {', '.join(repr(e) for e in REASONING_EFFORTS)}.",
            "reasoning_effort",
            "invalid_value",
        )

    effort = resolve_effort(model, effort)
    prompt = analyze_chat_messages(body.get("messages"))
    legacy = body.get("functions") is not None
    specs = parse_chat_tools(body.get("tools"), body.get("functions"))
    if body.get("parallel_tool_calls") is not None and not body.get("tools"):
        raise invalid("Invalid value for 'parallel_tool_calls': 'parallel_tool_calls' is only allowed when 'tools' are specified.", "parallel_tool_calls")
    choice_param = body.get("function_call") if legacy else body.get("tool_choice")
    if legacy and body.get("tool_choice") is not None:
        raise invalid("Cannot specify both 'functions' and 'tool_choice'.", "tool_choice")
    policy = resolve_tool_choice(choice_param, specs, api="chat", legacy_function_call=legacy)
    parallel = (body.get("parallel_tool_calls") is not False) and not legacy
    fmt_kind, schema, strict = response_format_kind(body.get("response_format"), param="response_format")
    prompt.tokens += count_json_tokens(body.get("tools") or body.get("functions"))

    tool_prob = header_float(request, "x-dummyllm-tool-probability")
    rng = make_rng(cfg.seed, body.get("seed"), body)

    gens: list[Generation] = []
    for _ in range(n):
        gen = Generation()
        gen.reasoning = generate_reasoning(rng, effort)
        budget = None if max_tokens is None else max_tokens - gen.reasoning_tokens
        if budget is not None and budget <= 0:
            # Burned the whole budget thinking: empty content, finish_reason=length.
            gen.reasoning = gen.reasoning[:max_tokens]
            gen.finish_reason = "length"
            gens.append(gen)
            continue
        to_call = decide_tool_calls(rng, cfg, policy, prompt, parallel=parallel, probability_override=tool_prob)
        if to_call:
            gen.tool_calls = [make_tool_call(rng, s, prompt) for s in to_call]
            gen.finish_reason = "function_call" if legacy else "tool_calls"
            gen.tool_calls, cut = fit_tool_calls(gen.tool_calls, budget)
            if cut:
                gen.finish_reason = "length"
        else:
            flavor = Flavor(saw_image=prompt.saw_image, after_tool_result=prompt.after_tool_result)
            gen.tokens, gen.finish_reason = generate_text(
                rng, cfg, flavor=flavor, max_tokens=budget, stops=stops, fmt_kind=fmt_kind, schema=schema, strict=strict
            )
        gens.append(gen)

    top_n = top_logprobs or 0
    logprobs = [fake_logprobs(rng, g.tokens, top_n) if want_logprobs else None for g in gens]

    completion_tokens = sum(g.completion_tokens for g in gens)
    usage = {
        "prompt_tokens": prompt.tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt.tokens + completion_tokens,
        "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
        "completion_tokens_details": {
            "reasoning_tokens": sum(g.reasoning_tokens for g in gens),
            "audio_tokens": 0,
            "accepted_prediction_tokens": 0,
            "rejected_prediction_tokens": 0,
        },
    }
    meta = {
        "id": new_id("chatcmpl-", 29),
        "created": now(),
        "model": model,
        "service_tier": body.get("service_tier") if body.get("service_tier") in ("flex", "priority") else "default",
        "system_fingerprint": fingerprint(model),
    }

    if stream:
        events = _stream_events(meta, gens, logprobs, usage, legacy=legacy, include_usage=include_usage)
        return sse_response(events, cfg, done_sentinel=True)

    await simulate_latency(cfg, completion_tokens)
    choices = [
        {
            "index": i,
            "message": _message(g, legacy),
            "logprobs": {"content": lp, "refusal": None} if lp is not None else None,
            "finish_reason": g.finish_reason,
        }
        for i, (g, lp) in enumerate(zip(gens, logprobs))
    ]
    return json_response(
        {
            "id": meta["id"],
            "object": "chat.completion",
            "created": meta["created"],
            "model": model,
            "choices": choices,
            "usage": usage,
            "service_tier": meta["service_tier"],
            "system_fingerprint": meta["system_fingerprint"],
        }
    )


def _tool_call_obj(call) -> dict[str, Any]:
    if call.kind == "custom":
        return {"id": call.id, "type": "custom", "custom": {"name": call.name, "input": call.arguments}}
    return {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": call.arguments}}


def _message(gen: Generation, legacy: bool) -> dict[str, Any]:
    msg: dict[str, Any] = {
        "role": "assistant",
        "content": None if gen.tool_calls else gen.text,
        "refusal": None,
        "annotations": [],
    }
    if gen.reasoning:
        # DeepSeek/vLLM-style field used by OpenAI-compatible reasoning servers.
        msg["reasoning_content"] = gen.reasoning_text
    if gen.tool_calls:
        if legacy:
            call = gen.tool_calls[0]
            msg["function_call"] = {"name": call.name, "arguments": call.arguments}
        else:
            msg["tool_calls"] = [_tool_call_obj(c) for c in gen.tool_calls]
    return msg


def _stream_events(
    meta: dict[str, Any],
    gens: list[Generation],
    logprobs: list[list[dict[str, Any]] | None],
    usage: dict[str, Any],
    *,
    legacy: bool,
    include_usage: bool,
) -> list[SSE]:
    def chunk(choice: dict[str, Any] | None, tokens: int = 0) -> SSE:
        data: dict[str, Any] = {
            "id": meta["id"],
            "object": "chat.completion.chunk",
            "created": meta["created"],
            "model": meta["model"],
            "service_tier": meta["service_tier"],
            "system_fingerprint": meta["system_fingerprint"],
            "choices": [choice] if choice is not None else [],
        }
        if include_usage:
            data["usage"] = None
        return SSE(data, tokens=tokens)

    per_choice: list[list[SSE]] = []
    for i, (gen, lps) in enumerate(zip(gens, logprobs)):
        evs: list[SSE] = []

        def ch(delta: dict[str, Any], *, finish: str | None = None, lp: Any = None, tokens: int = 0, _i: int = i) -> SSE:
            return chunk({"index": _i, "delta": delta, "logprobs": lp, "finish_reason": finish}, tokens)

        def thinking() -> None:
            for tok in gen.reasoning:
                evs.append(ch({"reasoning_content": tok}, tokens=1))

        if gen.tool_calls:
            evs.append(ch({"role": "assistant", "content": None, "refusal": None}))
            thinking()
            for k, call in enumerate(gen.tool_calls):
                frags = split_tokens(call.arguments)
                if legacy:
                    evs.append(ch({"function_call": {"name": call.name, "arguments": ""}}, tokens=1))
                    for frag in frags:
                        evs.append(ch({"function_call": {"arguments": frag}}, tokens=1))
                    break
                if call.kind == "custom":
                    head = {"index": k, "id": call.id, "type": "custom", "custom": {"name": call.name, "input": ""}}
                    evs.append(ch({"tool_calls": [head]}, tokens=1))
                    for frag in frags:
                        evs.append(ch({"tool_calls": [{"index": k, "custom": {"input": frag}}]}, tokens=1))
                else:
                    head = {"index": k, "id": call.id, "type": "function", "function": {"name": call.name, "arguments": ""}}
                    evs.append(ch({"tool_calls": [head]}, tokens=1))
                    for frag in frags:
                        evs.append(ch({"tool_calls": [{"index": k, "function": {"arguments": frag}}]}, tokens=1))
        else:
            evs.append(ch({"role": "assistant", "content": "", "refusal": None}, lp={"content": [], "refusal": None} if lps is not None else None))
            thinking()
            for j, tok in enumerate(gen.tokens):
                lp = {"content": [lps[j]], "refusal": None} if lps is not None else None
                evs.append(ch({"content": tok}, lp=lp, tokens=1))
        evs.append(ch({}, finish=gen.finish_reason))
        per_choice.append(evs)

    # Interleave choices round-robin, as the real API does for n > 1.
    out: list[SSE] = []
    idx = [0] * len(per_choice)
    remaining = sum(len(e) for e in per_choice)
    while remaining:
        for c, evs in enumerate(per_choice):
            if idx[c] < len(evs):
                out.append(evs[idx[c]])
                idx[c] += 1
                remaining -= 1
    if include_usage:
        final = chunk(None)
        final.data["usage"] = usage
        out.append(final)
    return out
