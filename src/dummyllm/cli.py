"""Command-line entry point: ``dummyllm`` / ``python -m dummyllm``."""

from __future__ import annotations

import argparse
import logging
import sys

from . import __version__
from .config import Config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dummyllm",
        description="An OpenAI-compatible LLM server that only speaks cat. Every option can also be set via "
        "a DUMMYLLM_<OPTION> environment variable (e.g. DUMMYLLM_PORT=9000).",
    )
    p.add_argument("--version", action="version", version=f"dummyllm {__version__}")
    p.add_argument("--host", help="bind address (default 127.0.0.1)")
    p.add_argument("--port", type=int, help="port (default 8000)")
    p.add_argument("--api-key", help="require this bearer token on every request")
    p.add_argument("--seed", type=int, help="global seed; makes every response deterministic")
    p.add_argument("--models", help="comma-separated model ids advertised by /v1/models")
    p.add_argument("--min-reply-tokens", type=int, help="shortest natural reply (default 4)")
    p.add_argument("--max-reply-tokens", type=int, help="longest natural reply (default 48)")
    p.add_argument("--tool-call-probability", type=float, help="P(call a tool) when tool_choice=auto (default 0.5)")
    p.add_argument(
        "--tool-call-probability-after-tool-result",
        type=float,
        help="P(call a tool) right after a tool result (default 0.2)",
    )
    p.add_argument("--time-to-first-token-ms", "--ttft-ms", type=float, dest="time_to_first_token_ms", help="simulated latency before the first token")
    p.add_argument("--tokens-per-second", "--tps", type=float, dest="tokens_per_second", help="simulated generation speed (0 = instant)")
    p.add_argument("--error-rate", type=float, help="fraction of requests failing with 429/500/503 (default 0)")
    p.add_argument("--response-store-size", type=int, help="max stored Responses API objects (default 1000)")
    p.add_argument("--response-store", help="'memory' (default) or 'modal-dict:<name>' to share Responses across containers")
    p.add_argument("--public-url", help="base URL used in image URLs (default: request host)")
    p.add_argument("--log-level", choices=["critical", "error", "warning", "info", "debug", "trace"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    overrides = {k: v for k, v in vars(args).items() if v is not None}
    if "models" in overrides:
        overrides["models"] = tuple(m.strip() for m in overrides["models"].split(",") if m.strip())
    try:
        cfg = Config.from_env(**overrides)
    except ValueError as exc:
        print(f"dummyllm: error: {exc}", file=sys.stderr)
        return 2

    import uvicorn

    from .app import create_app

    logging.basicConfig(level=cfg.log_level.upper() if cfg.log_level != "trace" else "DEBUG", format="%(levelname)s:     %(message)s")
    print(f"🐱 dummyllm {__version__} listening on http://{cfg.host}:{cfg.port}/v1", file=sys.stderr)
    uvicorn.run(create_app(cfg), host=cfg.host, port=cfg.port, log_level=cfg.log_level)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
