"""Host dummyllm on Modal, served at https://dummyllm.com.

    uvx modal serve modal_app.py     # ephemeral, hot-reloads on code changes
    uvx modal deploy modal_app.py    # persistent deployment

Both use the active Modal profile's workspace and environment; pass ``-e <env>``
(and ``--profile`` / ``MODAL_PROFILE``) to target another one.

``DOMAIN`` must be registered in the workspace you deploy to. To use the default
``*.modal.run`` URL instead, drop ``custom_domains`` and ``DUMMYLLM_PUBLIC_URL``.

It's an ASGI web function rather than an ``app.server`` because only web
functions accept ``custom_domains``. The endpoint is public: any OpenAI client
works with just ``base_url``, and the playground opens in a browser. To require
a key, set ``DUMMYLLM_API_KEY`` (e.g. from a ``modal.Secret``).

Tune dummyllm through the ``DUMMYLLM_*`` variables in ``SERVER_ENV`` below.
"""

from __future__ import annotations

from pathlib import Path

import modal

DOMAIN = "dummyllm.com"
PACKAGE_DIR = Path(__file__).parent / "src" / "dummyllm"

SERVER_ENV = {
    "DUMMYLLM_LOG_LEVEL": "info",
    # Links dummyllm generates (image URLs) should point at the custom domain.
    "DUMMYLLM_PUBLIC_URL": f"https://{DOMAIN}",
    # Responses API objects live in a shared modal.Dict, so any container can
    # serve previous_response_id / GET /v1/responses/{id}.
    "DUMMYLLM_RESPONSE_STORE": "modal-dict:dummyllm-responses",
}

image = (
    modal.Image.debian_slim(python_version="3.13")
    # Runtime dependencies only, pinned by uv.lock (no dev group).
    .uv_sync(extra_options="--no-default-groups")
    .env(SERVER_ENV)
    # Copied by path (not import) so `uvx modal` works without dummyllm installed,
    # and so non-Python files like the playground bundle come along.
    .add_local_dir(PACKAGE_DIR, remote_path="/root/dummyllm", ignore=["**/__pycache__", "**/*.pyc"])
)

app = modal.App("dummyllm", image=image)


@app.function(
    cpu=1.0,
    memory=1024,
    min_containers=0,
    # Scale out under load; state is shared via modal.Dict, so any container works.
    max_containers=10,
    scaledown_window=600,
)
# dummyllm is fully async: each container serves many requests at once, and the
# autoscaler adds containers past ~100 in flight per container.
@modal.concurrent(max_inputs=500, target_inputs=100)
@modal.asgi_app(custom_domains=[DOMAIN])
def web():
    from dummyllm import Config, create_app

    return create_app(Config.from_env())
