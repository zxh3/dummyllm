"""ASGI application factory."""

from __future__ import annotations

from pathlib import Path

from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from . import __version__
from .config import Config
from .errors import APIError
from .routes import chat, completions, embeddings, images, models, responses
from .store import make_store

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


# The React playground, built from web/ by `npm run build`.
PLAYGROUND_DIR = Path(__file__).parent / "static" / "playground"

_NOT_BUILT = """<!doctype html><meta charset="utf-8"><title>dummyllm</title>
<body style="font:15px/1.6 system-ui;max-width:560px;margin:15vh auto;padding:0 20px">
<h1 style="font-size:20px">The playground hasn't been built</h1>
<p>The API is up at <code>/v1</code>. To build the chat UI, run:</p>
<pre style="background:#f4f4f5;padding:12px;border-radius:8px">cd web &amp;&amp; npm install &amp;&amp; npm run build</pre></body>"""


async def playground(request: Request) -> Response:
    index = PLAYGROUND_DIR / "index.html"
    if not index.is_file():
        return HTMLResponse(_NOT_BUILT, status_code=503)
    # Asset names are content-hashed; only the HTML needs revalidating.
    return FileResponse(index, media_type="text/html; charset=utf-8", headers={"cache-control": "no-cache"})


async def index(request: Request) -> Response:
    # Browsers get the chat playground; API clients and curl get JSON.
    if "text/html" in request.headers.get("accept", ""):
        return await playground(request)
    return await info(request)


async def info(request: Request) -> Response:
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

    routes: list[Route | Mount] = [
        Route("/", index, methods=["GET"]),
        Route("/playground", playground, methods=["GET"]),
        Route("/info", info, methods=["GET"]),
        Route("/health", health, methods=["GET"]),
        Mount("/assets", StaticFiles(directory=PLAYGROUND_DIR / "assets", check_dir=False), name="assets"),
    ]
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
    app.state.response_store = make_store(cfg.response_store, cfg.response_store_size)
    return app
