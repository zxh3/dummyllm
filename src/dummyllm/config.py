"""Runtime configuration for dummyllm.

Every field can be set from the CLI or from a ``DUMMYLLM_<FIELD>`` environment
variable (e.g. ``DUMMYLLM_TOKENS_PER_SECOND=40``).
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from typing import Any

DEFAULT_MODELS: tuple[str, ...] = (
    "kiana",  # fast chat
    "kiana-reasoning",  # thinks first (reasoning_content) by default
    "kiana-vision",  # chat that reacts to images
    "kiana-embed",  # embeddings
    "kiana-image",  # image generation (base64 output)
)


@dataclass
class Config:
    # Networking
    host: str = "127.0.0.1"
    port: int = 8000

    # If set, requests must carry ``Authorization: Bearer <api_key>``.
    api_key: str | None = None

    # Global seed. When set, every response is a deterministic function of the
    # request body (plus the request's own ``seed`` if given).
    seed: int | None = None

    # Model ids advertised by ``GET /v1/models``. Any model name is accepted
    # in requests regardless; this list is only what is advertised.
    models: tuple[str, ...] = DEFAULT_MODELS

    # Length of a natural (non-truncated) text reply, in tokens.
    min_reply_tokens: int = 4
    max_reply_tokens: int = 48

    # Probability of calling a tool when tools are offered and tool_choice is
    # "auto". A lower probability applies right after a tool result so agent
    # loops terminate.
    tool_call_probability: float = 0.5
    tool_call_probability_after_tool_result: float = 0.2

    # Latency simulation. 0 disables the corresponding delay.
    time_to_first_token_ms: float = 0.0
    tokens_per_second: float = 0.0

    # Fraction of requests that fail with a random retryable error (429/500/503).
    error_rate: float = 0.0

    # Max number of stored Responses API objects (LRU).
    response_store_size: int = 1000
    # Where Responses objects live: "memory" (per process) or
    # "modal-dict:<name>" (shared across containers on Modal).
    response_store: str = "memory"

    # Base URL used when building image URLs (``response_format="url"``).
    # Defaults to the URL the request came in on.
    public_url: str | None = None

    log_level: str = "info"

    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls, **overrides: Any) -> "Config":
        cfg = cls()
        for f in dataclasses.fields(cls):
            if f.name == "extra":
                continue
            raw = os.environ.get(f"DUMMYLLM_{f.name.upper()}")
            if raw is None:
                continue
            setattr(cfg, f.name, _coerce(f.name, raw, getattr(cfg, f.name)))
        for k, v in overrides.items():
            if v is not None:
                setattr(cfg, k, v)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.min_reply_tokens < 1:
            raise ValueError("min_reply_tokens must be >= 1")
        if self.max_reply_tokens < self.min_reply_tokens:
            raise ValueError("max_reply_tokens must be >= min_reply_tokens")
        for name in ("tool_call_probability", "tool_call_probability_after_tool_result", "error_rate"):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.time_to_first_token_ms < 0 or self.tokens_per_second < 0:
            raise ValueError("latency settings must be >= 0")
        if self.response_store_size < 0:
            raise ValueError("response_store_size must be >= 0")
        if not self.models:
            raise ValueError("at least one model must be advertised")


def _coerce(name: str, raw: str, current: Any) -> Any:
    if name == "models":
        return tuple(m.strip() for m in raw.split(",") if m.strip())
    if name in ("seed", "port"):
        return int(raw)
    if name in ("api_key", "public_url"):
        return raw or None
    if isinstance(current, bool):
        return raw.lower() in ("1", "true", "yes", "on")
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    return raw
