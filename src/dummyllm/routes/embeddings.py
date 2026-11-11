"""POST /v1/embeddings -- deterministic pseudo-random unit vectors."""

from __future__ import annotations

import base64
import hashlib
import math
import random
import struct
from typing import Any

from starlette.requests import Request
from starlette.responses import Response

from ..common import cfg_of, endpoint, json_response, read_json, simulate_latency
from ..engine import check_int, require_model
from ..errors import invalid
from ..tokens import count_tokens

MAX_INPUTS = 2048


def default_dimensions(model: str) -> int:
    return 3072 if "large" in model else 1536


def embed(text: str, model: str, dims: int) -> list[float]:
    """Same (model, text, dims) -> same unit vector, across processes."""
    seed = int.from_bytes(hashlib.sha256(f"{model}\x00{dims}\x00{text}".encode()).digest()[:16], "big")
    rng = random.Random(seed)
    vec = [rng.gauss(0.0, 1.0) for _ in range(dims)]
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _inputs(raw: Any) -> list[str]:
    if raw is None:
        raise invalid("Missing required parameter: 'input'.", "input", "missing_required_parameter")
    if isinstance(raw, str):
        items: list[Any] = [raw]
    elif isinstance(raw, list):
        if not raw:
            raise invalid("'$.input' is invalid. Please check the API reference: https://platform.openai.com/docs/api-reference.", "input")
        if all(isinstance(x, int) and not isinstance(x, bool) for x in raw):
            items = [" ".join(f"<{t}>" for t in raw)]
        else:
            items = raw
    else:
        raise invalid("'$.input' is invalid. Please check the API reference: https://platform.openai.com/docs/api-reference.", "input")
    if len(items) > MAX_INPUTS:
        raise invalid(f"Invalid 'input': maximum {MAX_INPUTS} inputs per request.", "input", "array_above_max_length")
    out = []
    for x in items:
        if isinstance(x, str):
            if x == "":
                raise invalid("'$.input' is invalid. Please check the API reference: https://platform.openai.com/docs/api-reference.", "input")
            out.append(x)
        elif isinstance(x, list) and x and all(isinstance(t, int) and not isinstance(t, bool) for t in x):
            out.append(" ".join(f"<{t}>" for t in x))
        else:
            raise invalid("'$.input' is invalid. Please check the API reference: https://platform.openai.com/docs/api-reference.", "input")
    return out


@endpoint
async def embeddings(request: Request) -> Response:
    cfg = cfg_of(request)
    body = await read_json(request)
    model = require_model(body)
    texts = _inputs(body.get("input"))
    dims = body.get("dimensions")
    check_int(dims, "dimensions", 1, 3072)
    dims = dims or default_dimensions(model)
    fmt = body.get("encoding_format") or "float"
    if fmt not in ("float", "base64"):
        raise invalid(f"Invalid value: '{fmt}'. Supported values are: 'float' and 'base64'.", "encoding_format", "invalid_value")

    data = []
    for i, text in enumerate(texts):
        vec = embed(text, model, dims)
        emb: Any = base64.b64encode(struct.pack(f"<{dims}f", *vec)).decode("ascii") if fmt == "base64" else vec
        data.append({"object": "embedding", "index": i, "embedding": emb})
    tokens = sum(count_tokens(t) for t in texts)
    await simulate_latency(cfg, 0)
    return json_response({"object": "list", "data": data, "model": model, "usage": {"prompt_tokens": tokens, "total_tokens": tokens}})
