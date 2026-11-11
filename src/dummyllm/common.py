"""HTTP plumbing shared by all routes: JSON parsing, auth, fault injection,
latency simulation and Server-Sent Events."""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import random
import secrets
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse

from .config import Config
from .errors import APIError, injected_error, invalid

log = logging.getLogger("dummyllm")

Handler = Callable[[Request], Awaitable[Response]]

MAX_BODY_BYTES = 64 * 1024 * 1024


def cfg_of(request: Request) -> Config:
    return request.app.state.config


async def read_json(request: Request) -> dict[str, Any]:
    raw = await request.body()
    if len(raw) > MAX_BODY_BYTES:
        raise APIError("Request body too large.", status=413, code="request_too_large")
    if not raw.strip():
        raise invalid(
            "We could not parse the JSON body of your request. (HINT: This likely means you aren't using your HTTP "
            "library correctly. The OpenAI API expects a JSON payload, but what was sent was not valid JSON.)"
        )
    try:
        body = json.loads(raw, parse_constant=_reject_constant)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        raise invalid(
            "We could not parse the JSON body of your request. (HINT: This likely means you aren't using your HTTP "
            "library correctly. The OpenAI API expects a JSON payload, but what was sent was not valid JSON.)"
        ) from None
    if not isinstance(body, dict):
        raise invalid("Invalid request body: expected a JSON object.")
    return body


def _reject_constant(name: str) -> Any:
    # Python's json accepts NaN/Infinity; strict JSON (and OpenAI) does not.
    raise ValueError(f"invalid JSON constant {name}")


def json_response(payload: Any, status: int = 200, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse(payload, status_code=status, headers=headers)


def error_response(exc: APIError, request_id: str | None = None) -> JSONResponse:
    headers = dict(exc.headers)
    if request_id:
        headers["x-request-id"] = request_id
    return JSONResponse(exc.body(), status_code=exc.status, headers=headers)


def _check_auth(request: Request, cfg: Config) -> None:
    if not cfg.api_key:
        return
    auth = request.headers.get("authorization", "")
    scheme, _, token = auth.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        token = request.headers.get("api-key", "")  # Azure-style
        if not token:
            raise APIError(
                "You didn't provide an API key. You need to provide your API key in an Authorization header using "
                "Bearer auth (i.e. Authorization: Bearer YOUR_KEY).",
                status=401,
                code=None,
            )
    token = token.strip()
    if not secrets.compare_digest(token.encode(), cfg.api_key.encode()):
        shown = token[:3] + "*" * max(0, len(token) - 7) + token[-4:] if len(token) > 8 else "****"
        raise APIError(f"Incorrect API key provided: {shown}.", status=401, code="invalid_api_key")


def _maybe_inject_error(request: Request, cfg: Config) -> None:
    forced = request.headers.get("x-dummyllm-error")
    if forced:
        try:
            status = int(forced)
        except ValueError:
            raise invalid("Invalid 'x-dummyllm-error' header: expected an HTTP status code.") from None
        if not 400 <= status <= 599:
            raise invalid("Invalid 'x-dummyllm-error' header: expected a 4xx or 5xx status code.")
        raise injected_error(status)
    if cfg.error_rate > 0 and random.random() < cfg.error_rate:
        raise injected_error(random.choice((429, 500, 503)))


def endpoint(fn: Handler) -> Handler:
    """Wrap a route: auth, fault injection, OpenAI error envelope, standard headers."""

    @functools.wraps(fn)
    async def wrapper(request: Request) -> Response:
        cfg = cfg_of(request)
        request_id = "req_" + secrets.token_hex(16)
        request.state.request_id = request_id
        started = time.perf_counter()
        try:
            _check_auth(request, cfg)
            _maybe_inject_error(request, cfg)
            response = await fn(request)
        except APIError as exc:
            if exc.status >= 500:
                log.warning("%s %s -> %s %s", request.method, request.url.path, exc.status, exc.message)
            return error_response(exc, request_id)
        except Exception:  # pragma: no cover - defensive
            log.exception("unhandled error in %s", request.url.path)
            return error_response(APIError("The server had an error while processing your request. Meow.", status=500, type="server_error"), request_id)
        response.headers["x-request-id"] = request_id
        response.headers["openai-processing-ms"] = str(int((time.perf_counter() - started) * 1000))
        response.headers["openai-version"] = "2020-10-01"
        response.headers["x-ratelimit-limit-requests"] = "10000"
        response.headers["x-ratelimit-remaining-requests"] = "9999"
        response.headers["x-ratelimit-reset-requests"] = "6ms"
        response.headers["x-ratelimit-limit-tokens"] = "10000000"
        response.headers["x-ratelimit-remaining-tokens"] = "9999999"
        response.headers["x-ratelimit-reset-tokens"] = "0s"
        return response

    return wrapper


def header_float(request: Request, name: str, lo: float = 0.0, hi: float = 1.0) -> float | None:
    raw = request.headers.get(name)
    if raw is None:
        return None
    try:
        v = float(raw)
    except ValueError:
        raise invalid(f"Invalid '{name}' header: expected a number.") from None
    if not lo <= v <= hi:
        raise invalid(f"Invalid '{name}' header: expected a value between {lo} and {hi}.")
    return v


# ---------------------------------------------------------------------------
# Latency


async def simulate_latency(cfg: Config, tokens: int) -> None:
    delay = cfg.time_to_first_token_ms / 1000
    if cfg.tokens_per_second > 0:
        delay += tokens / cfg.tokens_per_second
    if delay > 0:
        await asyncio.sleep(delay)


# ---------------------------------------------------------------------------
# Server-Sent Events


@dataclass
class SSE:
    data: Any  # dict -> JSON; str -> sent verbatim
    event: str | None = None
    tokens: int = 0  # how many generated tokens this event "costs" (for pacing)


def _encode(ev: SSE) -> str:
    payload = ev.data if isinstance(ev.data, str) else json.dumps(ev.data, ensure_ascii=False, separators=(",", ":"))
    head = f"event: {ev.event}\n" if ev.event else ""
    return f"{head}data: {payload}\n\n"


async def _paced(events: Iterable[SSE], cfg: Config) -> AsyncIterator[str]:
    first = True
    per_token = 1 / cfg.tokens_per_second if cfg.tokens_per_second > 0 else 0.0
    for ev in events:
        if first:
            first = False
            if cfg.time_to_first_token_ms > 0:
                await asyncio.sleep(cfg.time_to_first_token_ms / 1000)
        elif per_token and ev.tokens:
            await asyncio.sleep(per_token * ev.tokens)
        yield _encode(ev)


def sse_response(events: Iterable[SSE], cfg: Config, *, done_sentinel: bool) -> StreamingResponse:
    async def body() -> AsyncIterator[str]:
        async for chunk in _paced(events, cfg):
            yield chunk
        if done_sentinel:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        body(),
        media_type="text/event-stream; charset=utf-8",
        headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
    )
