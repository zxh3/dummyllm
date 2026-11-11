"""ASGI application factory."""

from __future__ import annotations

from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from . import __version__
from .config import Config
from .errors import APIError
from .routes import chat, completions, embeddings, images, models, responses
from .store import ResponseStore

_API_ROUTES: list[tuple[str, object, list[str]]] = [
    ("/chat/completions", chat.chat_completions, ["POST"]),
    ("/completions", completions.completions, ["POST"]),
    ("/responses", responses.create_response, ["POST"]),
    ("/responses/input_tokens", responses.count_input_tokens, ["POST"]),
    ("/responses/{response_id}", responses.get_response, ["GET"]),
    ("/responses/{response_id}", responses.delete_response, ["DELETE"]),
    ("/responses/{response_id}/cancel", responses.cancel_response, ["POST"]),
    ("/responses/{response_id}/input_items", responses.list_input_items, ["GET"]),
    ("/embeddings", embeddings.embeddings, ["POST"]),
    ("/images/generations", images.generations, ["POST"]),
    ("/images/edits", images.edits, ["POST"]),
    ("/images/variations", images.variations, ["POST"]),
    ("/images/files/{image_id}", images.image_file, ["GET"]),
    ("/models", models.list_models, ["GET"]),
    ("/models/{model_id:path}", models.retrieve_model, ["GET"]),
]


async def health(request: Request) -> Response:
    return JSONResponse({"status": "ok", "meow": True})


async def index(request: Request) -> Response:
    return JSONResponse(
        {
            "name": "dummyllm",
            "version": __version__,
            "description": "An OpenAI-compatible LLM that only speaks cat.",
            "endpoints": [f"{methods[0]} /v1{path}" for path, _, methods in _API_ROUTES],
        }
    )


async def _http_exception(request: Request, exc: Exception) -> Response:
    assert isinstance(exc, HTTPException)
    if exc.status_code == 404:
        err = APIError(f"Invalid URL ({request.method} {request.url.path})", status=404)
    elif exc.status_code == 405:
        err = APIError(f"Method not allowed ({request.method} {request.url.path})", status=405)
    else:
        err = APIError(str(exc.detail), status=exc.status_code)
    return JSONResponse(err.body(), status_code=err.status, headers=getattr(exc, "headers", None))


def create_app(config: Config | None = None) -> Starlette:
    cfg = config or Config.from_env()
    cfg.validate()

    routes: list[Route] = [Route("/", index, methods=["GET"]), Route("/health", health, methods=["GET"])]
    # Serve under /v1 (canonical) and at the root (for clients configured without /v1).
    for prefix in ("/v1", ""):
        for path, handler, methods in _API_ROUTES:
            routes.append(Route(prefix + path, handler, methods=methods))  # type: ignore[arg-type]

    app = Starlette(
        routes=routes,
        exception_handlers={HTTPException: _http_exception},
        middleware=[
            Middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_methods=["*"],
                allow_headers=["*"],
                expose_headers=["x-request-id", "openai-processing-ms"],
            )
        ],
    )
    app.state.config = cfg
    app.state.response_store = ResponseStore(cfg.response_store_size)
    return app
