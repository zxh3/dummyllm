"""Run dummyllm inside your own test suite.

    from dummyllm.testing import DummyLLMServer

    with DummyLLMServer(seed=42) as server:
        client = openai.OpenAI(base_url=server.base_url, api_key="sk-meow")
        ...

Or as a pytest fixture::

    @pytest.fixture(scope="session")
    def llm():
        with DummyLLMServer() as server:
            yield server
"""

from __future__ import annotations

import socket
import threading
import time
from typing import Any

from .config import Config


def _free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


class DummyLLMServer:
    """Runs a dummyllm server on a background thread (real HTTP, real sockets)."""

    def __init__(self, config: Config | None = None, **overrides: Any) -> None:
        import uvicorn

        from .app import create_app

        cfg = config or Config()
        for k, v in overrides.items():
            if not hasattr(cfg, k):
                raise TypeError(f"unknown config option: {k}")
            setattr(cfg, k, v)
        if cfg.port == 0 or (config is None and "port" not in overrides):
            cfg.port = _free_port(cfg.host)
        cfg.validate()
        self.config = cfg
        self.app = create_app(cfg)
        self._server = uvicorn.Server(
            uvicorn.Config(self.app, host=cfg.host, port=cfg.port, log_level="warning", lifespan="off")
        )
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.config.host}:{self.config.port}"

    @property
    def base_url(self) -> str:
        """The OpenAI ``base_url`` (ends in ``/v1``)."""
        return f"{self.url}/v1"

    def start(self, timeout: float = 10.0) -> "DummyLLMServer":
        self._thread = threading.Thread(target=self._server.run, name="dummyllm", daemon=True)
        self._thread.start()
        deadline = time.monotonic() + timeout
        while not self._server.started:
            if not self._thread.is_alive():
                raise RuntimeError("dummyllm server failed to start")
            if time.monotonic() > deadline:
                raise TimeoutError("dummyllm server did not start in time")
            time.sleep(0.01)
        return self

    def stop(self) -> None:
        self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __enter__(self) -> "DummyLLMServer":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()
