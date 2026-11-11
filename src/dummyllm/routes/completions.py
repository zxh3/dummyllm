"""POST /v1/completions (legacy text completions)."""

from __future__ import annotations

from typing import Any

from starlette.requests import Request
from starlette.responses import Response

from ..common import SSE, cfg_of, endpoint, json_response, read_json, simulate_latency, sse_response
from ..engine import check_int, check_sampling, fake_logprobs, generate_text, normalize_stop, require_model
from ..errors import invalid
from ..ids import fingerprint, make_rng, new_id, now
from ..meow import Flavor
from ..tokens import count_tokens, split_tokens


def _prompts(raw: Any) -> list[str]:
    """Normalise ``prompt`` (str | [str] | [int] | [[int]]) into a list of strings."""
    if raw is None:
        raise invalid("Missing required parameter: 'prompt'.", "prompt", "missing_required_parameter")
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        if not raw:
            raise invalid("Invalid 'prompt': empty array.", "prompt", "empty_array")
        if all(isinstance(x, str) for x in raw):
            return list(raw)
        if all(isinstance(x, int) and not isinstance(x, bool) for x in raw):
            return [" ".join(f"<{t}>" for t in raw)]
        if all(isinstance(x, list) and all(isinstance(t, int) for t in x) for x in raw):
            return [" ".join(f"<{t}>" for t in x) for x in raw]
    raise invalid(
        "Invalid type for 'prompt': expected a string, array of strings, array of tokens, or array of token arrays.",
        "prompt",
        "invalid_type",
    )


@endpoint
async def completions(request: Request) -> Response:
    cfg = cfg_of(request)
    body = await read_json(request)

    model = require_model(body)
    check_sampling(body)
    prompts = _prompts(body.get("prompt"))
    n = body.get("n") or 1
    check_int(n, "n", 1, 128)
    best_of = body.get("best_of")
    check_int(best_of, "best_of", 1, 20)
    if best_of is not None and best_of < n:
        raise invalid("best_of must be greater than or equal to n.", "best_of")
    max_tokens = body.get("max_tokens", 16)
    if max_tokens is not None:
        check_int(max_tokens, "max_tokens", 0)
    logprobs_n = body.get("logprobs")
    check_int(logprobs_n, "logprobs", 0, 5)
    echo = bool(body.get("echo"))
    suffix = body.get("suffix")
    if suffix is not None and not isinstance(suffix, str):
        raise invalid("Invalid type for 'suffix': expected a string.", "suffix", "invalid_type")
    stops = normalize_stop(body.get("stop"))
    stream = bool(body.get("stream"))
    stream_options = body.get("stream_options")
    if stream_options is not None and not stream:
        raise invalid("The 'stream_options' parameter is only allowed when 'stream' is enabled.", "stream_options")
    include_usage = bool(isinstance(stream_options, dict) and stream_options.get("include_usage"))

    rng = make_rng(cfg.seed, body.get("seed"), body)
    prompt_tokens = sum(count_tokens(p) for p in prompts)

    choices: list[dict[str, Any]] = []
    completion_tokens = 0
    for p_idx, ptext in enumerate(prompts):
        for k in range(n):
            if max_tokens == 0:
                tokens, finish = [], "length"
            else:
                tokens, finish = generate_text(rng, cfg, flavor=Flavor(), max_tokens=max_tokens, stops=stops)
            completion_tokens += len(tokens)
            prefix_tokens = split_tokens(ptext) if echo else []
            all_tokens = prefix_tokens + tokens
            lp = None
            if logprobs_n is not None:
                entries = fake_logprobs(rng, all_tokens, logprobs_n)
                offsets, pos = [], 0
                for t in all_tokens:
                    offsets.append(pos)
                    pos += len(t)
                lp = {
                    "tokens": [e["token"] for e in entries],
                    "token_logprobs": [None if (echo and i == 0) else e["logprob"] for i, e in enumerate(entries)],
                    "top_logprobs": [
                        None if (echo and i == 0) else {a["token"]: a["logprob"] for a in e["top_logprobs"]} for i, e in enumerate(entries)
                    ],
                    "text_offset": offsets,
                }
            choices.append(
                {
                    "index": p_idx * n + k,
                    "text": "".join(all_tokens),
                    "logprobs": lp,
                    "finish_reason": finish,
                    "_tokens": tokens,
                    "_prefix": "".join(prefix_tokens),
                }
            )

    usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens}
    cid = new_id("cmpl-", 29)
    created = now()
    fp = fingerprint(model)

    if stream:
        return sse_response(_stream(cid, created, model, fp, choices, usage, include_usage), cfg, done_sentinel=True)

    await simulate_latency(cfg, completion_tokens)
    for c in choices:
        c.pop("_tokens")
        c.pop("_prefix")
    return json_response(
        {
            "id": cid,
            "object": "text_completion",
            "created": created,
            "model": model,
            "system_fingerprint": fp,
            "choices": choices,
            "usage": usage,
        }
    )


def _stream(cid, created, model, fp, choices, usage, include_usage) -> list[SSE]:
    def chunk(index: int, text: str, finish: str | None, tokens: int = 0) -> SSE:
        data: dict[str, Any] = {
            "id": cid,
            "object": "text_completion",
            "created": created,
            "model": model,
            "system_fingerprint": fp,
            "choices": [{"index": index, "text": text, "logprobs": None, "finish_reason": finish}],
        }
        if include_usage:
            data["usage"] = None
        return SSE(data, tokens=tokens)

    per_choice: list[list[SSE]] = []
    for c in choices:
        evs = []
        if c["_prefix"]:
            evs.append(chunk(c["index"], c["_prefix"], None))
        for tok in c["_tokens"]:
            evs.append(chunk(c["index"], tok, None, tokens=1))
        evs.append(chunk(c["index"], "", c["finish_reason"]))
        per_choice.append(evs)

    out: list[SSE] = []
    longest = max((len(e) for e in per_choice), default=0)
    for step in range(longest):
        for evs in per_choice:
            if step < len(evs):
                out.append(evs[step])
    if include_usage:
        out.append(
            SSE({"id": cid, "object": "text_completion", "created": created, "model": model, "system_fingerprint": fp, "choices": [], "usage": usage})
        )
    return out
