"""GET /v1/models"""

from __future__ import annotations

from starlette.requests import Request
from starlette.responses import Response

from ..common import cfg_of, endpoint, json_response

# A fixed timestamp keeps model objects stable across restarts.
_CREATED = 1_700_000_000


def _model(model_id: str) -> dict:
    return {"id": model_id, "object": "model", "created": _CREATED, "owned_by": "dummyllm"}


@endpoint
async def list_models(request: Request) -> Response:
    return json_response({"object": "list", "data": [_model(m) for m in cfg_of(request).models]})


@endpoint
async def retrieve_model(request: Request) -> Response:
    # dummyllm accepts any model name, so every model "exists".
    return json_response(_model(request.path_params["model_id"]))
