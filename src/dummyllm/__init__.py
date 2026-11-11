"""dummyllm: an OpenAI-compatible LLM server that only speaks cat."""

__version__ = "0.1.0"

from .config import Config  # noqa: E402

__all__ = ["Config", "__version__", "create_app"]


def create_app(config: Config | None = None):  # noqa: D401
    """Create the ASGI app (lazy import keeps ``import dummyllm`` cheap)."""
    from .app import create_app as _create_app

    return _create_app(config)
