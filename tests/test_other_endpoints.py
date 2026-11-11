"""Legacy completions, embeddings, images, models, and HTTP-level behaviour."""

from __future__ import annotations

import base64
import math
import struct

import httpx
import openai
import pytest
from conftest import png_file

from dummyllm.testing import DummyLLMServer

# ---------------------------------------------------------------------------
# Legacy completions


def test_completion(client):
    r = client.completions.create(model="meow-1", prompt="Once upon a time")
    assert r.object == "text_completion"
    assert r.choices[0].text
    assert r.usage.completion_tokens <= 16  # legacy default max_tokens


def test_completion_multi_prompt_and_n(client):
    r = client.completions.create(model="m", prompt=["a", "b"], n=2, max_tokens=5)
    assert [c.index for c in r.choices] == [0, 1, 2, 3]


def test_completion_echo_and_logprobs(client):
    r = client.completions.create(model="m", prompt="Hi cat", echo=True, logprobs=2, max_tokens=4)
    c = r.choices[0]
    assert c.text.startswith("Hi cat")
    assert "".join(c.logprobs.tokens) == c.text
    assert c.logprobs.text_offset[0] == 0


def test_completion_stream(client):
    full = client.completions.create(model="m", prompt="x", seed=4, max_tokens=30)
    text = "".join(ch.choices[0].text for ch in client.completions.create(model="m", prompt="x", seed=4, max_tokens=30, stream=True))
    assert text == full.choices[0].text


# ---------------------------------------------------------------------------
# Embeddings


def test_embeddings_default_sdk_base64(client):
    # The SDK requests base64 by default and decodes it client-side.
    r = client.embeddings.create(model="text-embedding-3-small", input=["meow", "purr"])
    assert len(r.data) == 2
    v = r.data[0].embedding
    assert len(v) == 1536
    assert math.isclose(sum(x * x for x in v), 1.0, rel_tol=1e-4)


def test_embeddings_deterministic_and_float(client):
    a = client.embeddings.create(model="e", input="meow", encoding_format="float").data[0].embedding
    b = client.embeddings.create(model="e", input="meow", encoding_format="float").data[0].embedding
    c = client.embeddings.create(model="e", input="purr", encoding_format="float").data[0].embedding
    assert a == b and a != c


def test_embeddings_dimensions_and_large(client):
    assert len(client.embeddings.create(model="text-embedding-3-large", input="x").data[0].embedding) == 3072
    assert len(client.embeddings.create(model="e", input="x", dimensions=256).data[0].embedding) == 256


def test_embeddings_raw_base64(http):
    r = http.post("/v1/embeddings", json={"model": "e", "input": "x", "encoding_format": "base64", "dimensions": 8})
    raw = base64.b64decode(r.json()["data"][0]["embedding"])
    assert len(struct.unpack("<8f", raw)) == 8


def test_embeddings_token_input(client):
    r = client.embeddings.create(model="e", input=[[1, 2, 3], [4, 5]])
    assert len(r.data) == 2


def test_embeddings_empty_string_rejected(client):
    with pytest.raises(openai.BadRequestError):
        client.embeddings.create(model="e", input="")


# ---------------------------------------------------------------------------
# Images


def _png_dims(data: bytes) -> tuple[int, int]:
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", data[16:24])


def test_image_generation_b64(client):
    r = client.images.generate(model="gpt-image-1", prompt="a cat", size="1536x1024", n=2)
    assert len(r.data) == 2
    for d in r.data:
        assert _png_dims(base64.b64decode(d.b64_json)) == (1536, 1024)
    assert r.usage.total_tokens > 0


def test_image_generation_url_is_fetchable(client, http):
    r = client.images.generate(model="dall-e-3", prompt="a cat", size="1024x1792")
    assert r.data[0].revised_prompt
    url = r.data[0].url
    resp = httpx.get(url)
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert _png_dims(resp.content) == (1024, 1792)
    # Stateless + deterministic: the same URL yields the same bytes.
    assert httpx.get(url).content == resp.content


def test_image_generation_validation(client):
    with pytest.raises(openai.BadRequestError):
        client.images.generate(model="dall-e-2", prompt="cat", size="huge")
    with pytest.raises(openai.BadRequestError):
        client.images.generate(model="dall-e-2", prompt="cat", n=11)
    with pytest.raises(openai.BadRequestError):
        client.images.generate(model="gpt-image-1", prompt="cat", response_format="url")


def test_image_edit_and_variation(client):
    r = client.images.edit(model="dall-e-2", image=png_file(), prompt="add a hat", size="256x256", response_format="b64_json")
    assert _png_dims(base64.b64decode(r.data[0].b64_json)) == (256, 256)
    v = client.images.create_variation(image=png_file(), n=2, size="512x512", response_format="b64_json")
    assert len(v.data) == 2


def test_image_edit_rejects_non_image(client):
    import io

    bogus = io.BytesIO(b"definitely not a png")
    bogus.name = "x.png"
    with pytest.raises(openai.BadRequestError):
        client.images.edit(image=bogus, prompt="x")


def test_unknown_image_url_404(http):
    assert http.get("/v1/images/files/nope.png").status_code == 404


# ---------------------------------------------------------------------------
# Models & HTTP behaviour


def test_models(client):
    ids = [m.id for m in client.models.list()]
    assert "meow-1" in ids
    m = client.models.retrieve("gpt-anything")
    assert m.id == "gpt-anything" and m.owned_by == "dummyllm"


def test_unknown_route_is_openai_shaped(http):
    r = http.post("/v1/nope", json={})
    assert r.status_code == 404
    assert r.json()["error"]["message"] == "Invalid URL (POST /v1/nope)"


def test_routes_without_v1_prefix(http):
    r = http.post("/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200


def test_standard_headers(http):
    r = http.get("/v1/models")
    assert r.headers["x-request-id"].startswith("req_")
    assert "openai-processing-ms" in r.headers


def test_health_and_index(http):
    assert http.get("/health").json()["status"] == "ok"
    assert "POST /v1/chat/completions" in http.get("/").json()["endpoints"]


@pytest.mark.parametrize("status, exc", [(429, openai.RateLimitError), (500, openai.InternalServerError), (401, openai.AuthenticationError)])
def test_injected_errors(client, status, exc):
    c = client.with_options(default_headers={"x-dummyllm-error": str(status)})
    with pytest.raises(exc) as e:
        c.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])
    assert e.value.status_code == status
    assert e.value.request_id.startswith("req_")


def test_injected_429_has_retry_after(http):
    r = http.get("/v1/models", headers={"x-dummyllm-error": "429"})
    assert r.status_code == 429
    assert r.headers["retry-after"] == "1"
    assert r.json()["error"]["code"] == "rate_limit_exceeded"


def test_api_key_auth():
    with DummyLLMServer(api_key="sk-secret") as s:
        good = openai.OpenAI(base_url=s.base_url, api_key="sk-secret", max_retries=0)
        assert good.models.list().data
        bad = openai.OpenAI(base_url=s.base_url, api_key="sk-wrongkey", max_retries=0)
        with pytest.raises(openai.AuthenticationError) as e:
            bad.models.list()
        assert e.value.code == "invalid_api_key"
        assert "sk-wrongkey" not in e.value.message


def test_global_seed_makes_everything_deterministic(seeded_client):
    msgs = [{"role": "user", "content": "hi"}]
    a = seeded_client.chat.completions.create(model="m", messages=msgs)
    b = seeded_client.chat.completions.create(model="m", messages=msgs)
    assert a.choices[0].message.content == b.choices[0].message.content


def test_error_rate_and_sdk_retries():
    with DummyLLMServer(error_rate=0.5) as s:
        c = openai.OpenAI(base_url=s.base_url, api_key="x", max_retries=8)
        c = c.with_options(timeout=10)
        # With retries the SDK should eventually succeed every time.
        for _ in range(5):
            assert c.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}]).choices


def test_latency_simulation():
    import time

    with DummyLLMServer(time_to_first_token_ms=200) as s:
        c = openai.OpenAI(base_url=s.base_url, api_key="x", max_retries=0)
        t = time.perf_counter()
        c.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])
        assert time.perf_counter() - t >= 0.2


def test_nan_json_rejected(http):
    r = http.post(
        "/v1/chat/completions",
        content=b'{"model": "m", "messages": [{"role": "user", "content": "hi"}], "temperature": NaN}',
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 400


def test_pathological_tool_schema_does_not_hang(client):
    tool = {"type": "function", "function": {"name": "evil", "parameters": {"type": "object", "properties": {"x": {"type": "string", "minLength": 10**12}}, "required": ["x"]}}}
    r = client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}], tools=[tool], tool_choice="required")
    assert r.choices[0].message.tool_calls


def test_playground_served_to_browsers(http):
    page = http.get("/playground")
    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    assert '<div id="root">' in page.text
    # The built bundle's assets are served too.
    import re

    assets = re.findall(r'(?:src|href)="\./(assets/[^"]+)"', page.text)
    assert any(a.endswith(".js") for a in assets) and any(a.endswith(".css") for a in assets)
    for asset in assets:
        r = http.get("/" + asset)
        assert r.status_code == 200, asset
    js = next(a for a in assets if a.endswith(".js"))
    assert "/chat/completions" in http.get("/" + js).text
    icon = re.search(r'rel="icon"[^>]*href="\./([^"]+)"', page.text).group(1)
    assert http.get("/" + icon).headers["content-type"].startswith("image/")
    # Browsers hitting / get the playground; API clients get JSON.
    assert http.get("/", headers={"accept": "text/html,application/xhtml+xml"}).headers["content-type"].startswith("text/html")
    assert http.get("/").json()["name"] == "dummyllm"
    assert http.get("/info").json()["name"] == "dummyllm"


def test_per_request_latency_headers(http):
    import time

    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10}
    t = time.perf_counter()
    r = http.post("/v1/chat/completions", json=body, headers={"x-dummyllm-ttft-ms": "250"})
    assert r.status_code == 200 and time.perf_counter() - t >= 0.25

    # Streaming is paced per token.
    t = time.perf_counter()
    with http.stream("POST", "/v1/chat/completions", json={**body, "stream": True}, headers={"x-dummyllm-tokens-per-second": "50"}) as s:
        token_chunks = sum('"content":"' in line and '"content":""' not in line for line in s.iter_lines())
    assert token_chunks >= 2
    assert time.perf_counter() - t >= (token_chunks - 1) / 50

    bad = http.post("/v1/chat/completions", json=body, headers={"x-dummyllm-ttft-ms": "-5"})
    assert bad.status_code == 400
    assert http.post("/v1/chat/completions", json=body, headers={"x-dummyllm-tokens-per-second": "fast"}).status_code == 400


def test_image_url_honors_forwarded_host(http):
    r = http.post(
        "/v1/images/generations",
        json={"model": "dall-e-2", "prompt": "cat", "size": "64x64", "response_format": "url"},
        headers={"x-forwarded-host": "llm.example.com", "x-forwarded-proto": "https"},
    )
    assert r.json()["data"][0]["url"].startswith("https://llm.example.com/v1/images/files/img_")
