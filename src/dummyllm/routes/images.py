"""Images API: generations, edits, variations -- every picture is a cat."""

from __future__ import annotations

import hashlib
import random
import re
import time
from typing import Any

from starlette.datastructures import UploadFile
from starlette.requests import Request
from starlette.responses import Response

from ..common import cfg_of, endpoint, error_response, json_response, read_json, simulate_latency
from ..errors import invalid, not_found
from ..images import image_size, parse_size, render_cat, render_cat_b64
from ..tokens import count_tokens

# Models that only ever return base64 (like gpt-image-1).
_B64_ONLY_RE = re.compile(r"^gpt-image")
_IMAGE_ID_RE = re.compile(r"^img_([0-9a-f]{32})_(\d{2,4})x(\d{2,4})\.png$")
MAX_IMAGE_BYTES = 50 * 1024 * 1024


def _b64_only(model: str) -> bool:
    return bool(_B64_ONLY_RE.match(model))


def _base_url(request: Request) -> str:
    public = cfg_of(request).public_url
    if public:
        return public.rstrip("/")
    # Behind a reverse proxy, prefer the client-facing host it reports.
    fwd_host = request.headers.get("x-forwarded-host")
    if fwd_host:
        proto = request.headers.get("x-forwarded-proto", request.url.scheme)
        return f"{proto.split(',')[0].strip()}://{fwd_host.split(',')[0].strip()}"
    return str(request.base_url).rstrip("/")


async def _build(request: Request, params: dict[str, Any], *, prompt: str | None, default_model: str) -> Response:
    cfg = cfg_of(request)
    model = params.get("model") or default_model
    if not isinstance(model, str):
        raise invalid("Invalid type for 'model': expected a string.", "model", "invalid_type")

    n_raw = params.get("n", 1)
    try:
        n = int(n_raw) if n_raw is not None else 1
    except (TypeError, ValueError):
        raise invalid("Invalid type for 'n': expected an integer.", "n", "invalid_type") from None
    if not 1 <= n <= 10:
        raise invalid(f"Invalid 'n': expected a value between 1 and 10, but got {n} instead.", "n", "invalid_value")

    size_raw = params.get("size")
    size = parse_size(size_raw if isinstance(size_raw, str) else None, default=(1024, 1024))
    if size is None:
        raise invalid(f"Invalid value: '{size_raw}'. Expected a size like '1024x1024' (16-4096 px per side) or 'auto'.", "size", "invalid_value")

    fmt = params.get("response_format")
    b64_only = _b64_only(model)
    if fmt is None:
        fmt = "b64_json" if b64_only else "url"
    if fmt not in ("url", "b64_json"):
        raise invalid(f"Invalid value: '{fmt}'. Supported values are: 'url' and 'b64_json'.", "response_format", "invalid_value")
    if b64_only and fmt == "url":
        raise invalid(f"Invalid value: 'url'. '{model}' only supports 'b64_json'.", "response_format", "invalid_value")
    out_fmt = params.get("output_format")
    if out_fmt not in (None, "png", "jpeg", "webp"):
        raise invalid(f"Invalid value: '{out_fmt}'. Supported values are: 'png', 'jpeg', and 'webp'.", "output_format", "invalid_value")

    seed_src = f"{cfg.seed}|{prompt}|{model}|{size}|{n}" if cfg.seed is not None else None
    rng = random.Random(seed_src) if seed_src is not None else random.Random()

    data = []
    w, h = size
    for _ in range(n):
        item: dict[str, Any] = {}
        if fmt == "b64_json":
            item["b64_json"] = render_cat_b64(w, h, rng)
        else:
            token = f"{rng.getrandbits(128):032x}"
            item["url"] = f"{_base_url(request)}/v1/images/files/img_{token}_{w}x{h}.png"
        if prompt is not None and model.startswith("dall-e-3"):
            item["revised_prompt"] = f"A cute cartoon cat face. The cat was asked to paint: {prompt[:500]}"
        data.append(item)

    payload: dict[str, Any] = {"created": int(time.time()), "data": data}
    if b64_only:
        payload.update(
            background="opaque",
            output_format="png",
            quality=params.get("quality") if params.get("quality") in ("low", "medium", "high") else "medium",
            size=f"{w}x{h}",
        )
        in_tokens = count_tokens(prompt or "")
        out_tokens = 272 * n
        payload["usage"] = {
            "input_tokens": in_tokens,
            "input_tokens_details": {"text_tokens": in_tokens, "image_tokens": 0},
            "output_tokens": out_tokens,
            "total_tokens": in_tokens + out_tokens,
        }
    await simulate_latency(cfg, 0)
    return json_response(payload)


@endpoint
async def generations(request: Request) -> Response:
    body = await read_json(request)
    prompt = body.get("prompt")
    if prompt is None:
        raise invalid("Missing required parameter: 'prompt'.", "prompt", "missing_required_parameter")
    if not isinstance(prompt, str) or not prompt.strip():
        raise invalid("Invalid 'prompt': expected a non-empty string.", "prompt", "invalid_value")
    if body.get("stream"):
        raise invalid("Streaming image generation is not supported by dummyllm.", "stream", "unsupported_value")
    return await _build(request, body, prompt=prompt, default_model="dall-e-2")


async def _form(request: Request) -> tuple[dict[str, Any], list[UploadFile]]:
    ctype = request.headers.get("content-type", "")
    if "multipart/form-data" not in ctype:
        raise invalid("Invalid content type. Image edits and variations require multipart/form-data.")
    form = await request.form(max_files=32, max_fields=64)
    params: dict[str, Any] = {}
    files: list[UploadFile] = []
    for key, value in form.multi_items():
        if isinstance(value, UploadFile) or hasattr(value, "read"):
            if key in ("image", "image[]"):
                files.append(value)  # type: ignore[arg-type]
            continue
        params[key] = value
    return params, files


async def _check_image(upload: UploadFile, param: str) -> None:
    data = await upload.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise invalid("Invalid input image - image is too large.", param, "image_too_large")
    if image_size(data) is None:
        raise invalid("Invalid input image - format must be in ['png', 'jpeg', 'webp', 'gif'].", param, "invalid_image_format")


@endpoint
async def edits(request: Request) -> Response:
    params, files = await _form(request)
    prompt = params.get("prompt")
    if not files:
        raise invalid("Missing required parameter: 'image'.", "image", "missing_required_parameter")
    if not prompt:
        raise invalid("Missing required parameter: 'prompt'.", "prompt", "missing_required_parameter")
    for f in files:
        await _check_image(f, "image")
    return await _build(request, params, prompt=prompt, default_model="dall-e-2")


@endpoint
async def variations(request: Request) -> Response:
    params, files = await _form(request)
    if not files:
        raise invalid("Missing required parameter: 'image'.", "image", "missing_required_parameter")
    await _check_image(files[0], "image")
    return await _build(request, params, prompt=None, default_model="dall-e-2")


async def image_file(request: Request) -> Response:
    """Serve an image URL produced by ``response_format=url``.

    URLs are stateless: the id encodes the seed and size, so the picture is
    re-rendered on demand and never expires.
    """
    m = _IMAGE_ID_RE.match(request.path_params["image_id"])
    if not m:
        return error_response(not_found("Image not found."))
    token, w, h = m.group(1), int(m.group(2)), int(m.group(3))
    if parse_size(f"{w}x{h}") is None:
        return error_response(not_found("Image not found."))
    rng = random.Random(int(hashlib.sha256(token.encode()).hexdigest(), 16))
    png = render_cat(w, h, rng)
    return Response(png, media_type="image/png", headers={"cache-control": "public, max-age=31536000, immutable"})
