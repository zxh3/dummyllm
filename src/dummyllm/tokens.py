"""A tiny, dependency-free stand-in tokenizer.

It is not tiktoken, but it is consistent: ``"".join(split_tokens(s)) == s`` and
token counts are stable, which is what usage accounting and ``max_tokens``
truncation need.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

_TOKEN_RE = re.compile(r"\s*\w{1,8}|\s*[^\w\s]|\s+", re.UNICODE)


def split_tokens(text: str) -> list[str]:
    """Split text into tokens. Long words are chunked into <=8 char pieces."""
    if not text:
        return []
    return _TOKEN_RE.findall(text)


def count_tokens(text: str) -> int:
    return len(split_tokens(text))


def count_json_tokens(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return count_tokens(value)
    return count_tokens(json.dumps(value, separators=(",", ":"), ensure_ascii=False))


# ---------------------------------------------------------------------------
# Vision token accounting (mirrors OpenAI's published tile formula).

IMAGE_LOW_DETAIL_TOKENS = 85
IMAGE_TILE_TOKENS = 170
DEFAULT_IMAGE_SIZE = (1024, 1024)


def image_tokens(width: int | None, height: int | None, detail: str | None) -> int:
    if detail == "low":
        return IMAGE_LOW_DETAIL_TOKENS
    w, h = (width or DEFAULT_IMAGE_SIZE[0]), (height or DEFAULT_IMAGE_SIZE[1])
    w, h = max(w, 1), max(h, 1)
    # Fit within 2048x2048.
    scale = min(1.0, 2048 / max(w, h))
    w, h = w * scale, h * scale
    # Shortest side to 768.
    scale = 768 / min(w, h)
    if scale < 1:
        w, h = w * scale, h * scale
    tiles = math.ceil(w / 512) * math.ceil(h / 512)
    return IMAGE_LOW_DETAIL_TOKENS + IMAGE_TILE_TOKENS * tiles
