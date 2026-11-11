"""Identifier and RNG helpers."""

from __future__ import annotations

import hashlib
import json
import random
import secrets
import string
import time
from typing import Any

_ALPHABET = string.ascii_letters + string.digits


def new_id(prefix: str, length: int = 24) -> str:
    """OpenAI-style id like ``chatcmpl-AbC...`` or ``call_AbC...``."""
    body = "".join(secrets.choice(_ALPHABET) for _ in range(length))
    sep = "" if prefix.endswith(("-", "_")) else "_"
    return f"{prefix}{sep}{body}"


def hex_id(prefix: str, nbytes: int = 24) -> str:
    """Responses-API style id like ``resp_68af...`` (hex body)."""
    return f"{prefix}_{secrets.token_hex(nbytes)}"


def now() -> int:
    return int(time.time())


# Request fields that must not affect generated content: they only change the
# transport (stream / stream_options) or bookkeeping.
_NON_SEMANTIC_KEYS = frozenset(
    {"stream", "stream_options", "user", "metadata", "store", "service_tier", "safety_identifier", "prompt_cache_key"}
)


def make_rng(server_seed: int | None, request_seed: Any, body: dict[str, Any]) -> random.Random:
    """Build the RNG for one request.

    If either the server or the request specifies a seed, the RNG is derived
    from a stable hash of (server seed, request seed, request body), so the same
    request always yields the same output -- streamed or not.
    """
    if server_seed is None and request_seed is None:
        return random.Random(secrets.randbits(128))
    semantic = {k: v for k, v in body.items() if k not in _NON_SEMANTIC_KEYS}
    canonical = json.dumps(semantic, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(f"{server_seed}|{request_seed}|{canonical}".encode()).digest()
    return random.Random(int.from_bytes(digest[:16], "big"))


def fingerprint(model: str) -> str:
    return "fp_" + hashlib.sha256(f"dummyllm:{model}".encode()).hexdigest()[:10]
