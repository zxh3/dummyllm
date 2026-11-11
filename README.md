# dummyllm 🐱

An **OpenAI-compatible LLM server that only speaks cat.**

Point any OpenAI client at it and you get well-formed, spec-shaped responses — chat, streaming,
tool calls, structured outputs, vision input, image generation, embeddings — except every word is
`meow`, `mrrp`, `purr` or `*knocks your coffee off the desk*`.

It's built for **testing LLM applications without an LLM**: fast, free, offline, deterministic
when you want it, and strict about request validation the same way the real API is, so client
bugs surface locally instead of in production.

```text
$ curl -s localhost:8000/v1/chat/completions -d '{"model":"kiana","messages":[{"role":"user","content":"hi"}]}' | jq -r '.choices[0].message.content'
Mrrp rrrow. Prrr meow meow meow! *blinks slowly* Nya?
```

## Quick start

```bash
uv sync
uv run dummyllm                      # http://127.0.0.1:8000/v1
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="anything")
print(client.chat.completions.create(model="kiana", messages=[{"role": "user", "content": "hi"}]).choices[0].message.content)
```

Or with Docker:

```bash
docker build -t dummyllm .
docker run -p 8000:8000 dummyllm
```

Any model name is accepted (`gpt-4o`, `o3`, `whatever`) and echoed back, so you can usually swap
dummyllm in by changing only `base_url`.

## Playground

Open the server root in a browser (<http://127.0.0.1:8000/>) for a chat UI built with React,
Tailwind CSS and shadcn/ui. It has:

* a conversation history sidebar; chats are stored in your browser (IndexedDB, one record per chat), settings in localStorage
* light and dark themes
* a model picker and streaming replies with TTFT / tokens / tok/s stats
* tool calling, with demo tools that "run" in the browser and feed results back to the model
* image uploads (vision input) and image generation
* a reasoning-effort setting, JSON mode, system instructions, seed and max tokens
* a **Reply length** slider (sends `x-dummyllm-reply-tokens`; *auto* keeps the server's natural length)
* a **View code** dialog that turns the current conversation into Python, Node.js or curl

The playground asks the server to simulate a realistic typing speed via per-request headers
(see *Latency simulation*), so API clients are unaffected. curl and SDKs hitting `/` still get
JSON (the page is served to `Accept: text/html`). The UI is also at `/playground`, and the JSON
index at `/info`.

The UI source lives in [`web/`](web/). Its production build is committed to
`src/dummyllm/static/playground/`, so the Python package serves it without needing Node. To work
on it:

```bash
uv run dummyllm                 # API on :8000
cd web && npm install
npm run dev                     # Vite on :5173, proxies /v1 to :8000 (override with DUMMYLLM_URL)
npm run build                   # rebuild the bundle shipped in the Python package
```

## What's supported

| Endpoint | Notes |
|---|---|
| `POST /v1/chat/completions` | streaming (+`include_usage`), `n`, `max_tokens`/`max_completion_tokens`, `stop`, `logprobs`/`top_logprobs`, `seed`, **tools** (function + custom), `tool_choice` (`auto`/`none`/`required`/named/`allowed_tools`), `parallel_tool_calls`, legacy `functions`/`function_call`, `response_format` (`json_object`, `json_schema` → works with `client.chat.completions.parse(...)`), **image inputs** (`image_url`, data URLs and https), **reasoning**: thinking text in `message.reasoning_content` / streamed `delta.reasoning_content` (the DeepSeek/vLLM convention), sized by `reasoning_effort` and counted in `reasoning_tokens`; models named `*reason*` think by default |
| `POST /v1/responses` | string or item-list `input`, `instructions`, streaming with the full typed event protocol, function / custom tool calls, **`image_generation` tool** (incl. `partial_images`), `previous_response_id`, `store`, `background`, `text.format` (works with `client.responses.parse(...)`), `reasoning` (+ summaries), `max_output_tokens` → `incomplete`, `include=["message.output_text.logprobs", "reasoning.encrypted_content"]` |
| `GET/DELETE /v1/responses/{id}`, `GET /v1/responses/{id}/input_items`, `POST /v1/responses/{id}/cancel`, `POST /v1/responses/input_tokens` | in-memory LRU store |
| `POST /v1/completions` | legacy text completions: multi-prompt, `n`, `echo`, `logprobs`, streaming |
| `POST /v1/embeddings` | float **and base64** (the SDK default), `dimensions`; vectors are unit-length and deterministic per input |
| `POST /v1/images/generations`, `/edits`, `/variations` | real PNGs of procedurally drawn cats at the requested size; `b64_json` or `url` |
| `GET /v1/models`, `GET /v1/models/{id}` | |
| `GET /health` | |
| `GET /` (browsers), `GET /playground` | chat playground UI |

Routes are also served without the `/v1` prefix.

Not implemented: audio, files/uploads, batches, fine-tuning, assistants, vector stores, realtime.
Hosted Responses tools other than `image_generation` (web search, file search, code interpreter,
MCP...) are accepted but never called.

## How the cat behaves

**Text.** Replies are sentences of cat sounds with the occasional `*action*`. The cat reacts to
context: it sniffs images you send, bats at tool results, and proudly presents drawings.

**Tool calls.** When tools are offered:

* `tool_choice="required"` or a named tool → always calls.
* `tool_choice="none"` → never calls.
* `tool_choice="auto"` → calls with probability `tool_call_probability` (default 0.5), dropping to
  `tool_call_probability_after_tool_result` (default 0.2) right after a tool result, so agent loops
  terminate.
* With `parallel_tool_calls` (default on) it sometimes calls 2–3 distinct tools at once.
* Arguments are **random values that validate against the tool's JSON Schema**: types, `required`,
  `enum`, `const`, `anyOf`/`oneOf`/`allOf`, `$ref`/`$defs` (including recursive schemas), min/max,
  `multipleOf`, string `format`s (date-time, email, uuid, uri...), nullable types, `prefixItems`...
  Property names flavor the values (`city` → `"Meowtown"`, `name` → `"Whiskers"`).
* In the Responses API, if an `image_generation` tool is available and the user asks to
  draw/paint/picture something, the cat draws.

**Structured outputs.** `json_schema` formats produce JSON matching the schema; `json_object`
produces a small random object.

**Images.** Generated images are real PNGs of randomized cat faces at exactly the requested size.
`url` responses point back at the server (`/v1/images/files/...`) and are stateless: the URL encodes
the seed, so it never expires and always returns the same picture.

**Vision input.** Images are never fetched or looked at, but data-URL images are validated (must be a
real PNG/JPEG/GIF/WebP/BMP), and their dimensions feed OpenAI's tile formula for `prompt_tokens`
(85 for `detail: low`, `85 + 170 × tiles` otherwise).

**Usage accounting.** Token counts come from a simple word/punctuation tokenizer (not tiktoken) but are
self-consistent: streamed deltas add up to the content, `max_tokens` truncates exactly, and usage
matches what was produced. Reasoning efforts add `reasoning_tokens`, which count against
`max_completion_tokens`, so you can test the "the model thought so long the answer is empty
(`finish_reason="length"`)" case.

**Validation.** Requests are validated like the real API, with OpenAI-shaped errors
(`{"error": {"message", "type", "param", "code"}}`) that the SDK raises as `BadRequestError` etc. This includes:

* parameter ranges and types (`temperature`, `n`, `top_logprobs` without `logprobs`, `stream_options` without `stream`, ≤4 stop sequences...)
* message structure: roles, content parts, image data
* tool-call bookkeeping: a `tool` message must answer a preceding `tool_calls` id, and every tool call must be answered before the next turn; in Responses, `function_call_output` must match a known `call_id`
* tool definitions: function-name pattern, object-typed parameters, duplicate names, `tool_choice` naming a tool that doesn't exist
* `previous_response_id` that doesn't exist, malformed JSON (including `NaN`)

## Testing knobs

### Determinism

* Pass `seed` in a request: the same request body + seed → the same output. Streaming and
  non-streaming produce identical content for the same seed (`stream` is excluded from the hash).
* Start the server with `--seed N` to make **every** response a deterministic function of its request.

### Latency simulation

```bash
dummyllm --ttft-ms 300 --tps 50   # 300 ms to first token, then 50 tokens/s
```

Streaming responses are paced token by token; non-streaming responses sleep for the equivalent total.

The same knobs can be set per request with headers, overriding the server settings for just that
request: `x-dummyllm-ttft-ms: 300` and `x-dummyllm-tokens-per-second: 50`.

### Reply length

Natural replies are 4–48 tokens (`--min-reply-tokens` / `--max-reply-tokens`). Header
`x-dummyllm-reply-tokens: 200` asks for about 200 tokens for that request (0.8–1.2× the value,
1–4000) that still end naturally with `finish_reason: "stop"`. A smaller `max_tokens` still
truncates with `"length"`.

### Fault injection

* Header `x-dummyllm-error: 429` (or any 4xx/5xx) makes that request fail with an OpenAI-shaped
  error. 429/503 include `retry-after`, so SDK retry logic can be exercised.
* `--error-rate 0.1` fails 10% of requests at random with 429/500/503.

### Per-request tool probability

Header `x-dummyllm-tool-probability: 1` forces a tool call under `tool_choice="auto"`, and `0` prevents one.

### Auth

`--api-key sk-secret` requires `Authorization: Bearer sk-secret` (or Azure-style `api-key`).
Without it, any key is accepted.

## Configuration

Every option is a CLI flag and a `DUMMYLLM_*` environment variable:

| Flag | Env | Default |
|---|---|---|
| `--host` | `DUMMYLLM_HOST` | `127.0.0.1` |
| `--port` | `DUMMYLLM_PORT` | `8000` |
| `--api-key` | `DUMMYLLM_API_KEY` | none (no auth) |
| `--seed` | `DUMMYLLM_SEED` | none |
| `--models` | `DUMMYLLM_MODELS` | `kiana` |
| `--min-reply-tokens` / `--max-reply-tokens` | `DUMMYLLM_MIN_REPLY_TOKENS` / `DUMMYLLM_MAX_REPLY_TOKENS` | `4` / `48` |
| `--tool-call-probability` | `DUMMYLLM_TOOL_CALL_PROBABILITY` | `0.5` |
| `--tool-call-probability-after-tool-result` | `DUMMYLLM_TOOL_CALL_PROBABILITY_AFTER_TOOL_RESULT` | `0.2` |
| `--ttft-ms` | `DUMMYLLM_TIME_TO_FIRST_TOKEN_MS` | `0` |
| `--tps` | `DUMMYLLM_TOKENS_PER_SECOND` | `0` (instant) |
| `--error-rate` | `DUMMYLLM_ERROR_RATE` | `0` |
| `--response-store-size` | `DUMMYLLM_RESPONSE_STORE_SIZE` | `1000` |
| `--public-url` | `DUMMYLLM_PUBLIC_URL` | request host (used in image URLs) |
| `--log-level` | `DUMMYLLM_LOG_LEVEL` | `info` |

## Deploying to Modal

[`modal_app.py`](modal_app.py) serves dummyllm on Modal at **<https://dummyllm.com>** (the playground is
at the root; the API at `/v1`):

```bash
MODAL_PROFILE=modal-garden uvx modal deploy modal_app.py -e xiaohua-dev   # or `modal serve` for a hot-reloading dev run
```

Notes:

* **Custom domain.** `dummyllm.com` is registered in the `modal-garden` workspace and attached with
  `@modal.asgi_app(custom_domains=[...])`. (`app.server` doesn't take custom domains, so the app is
  an ASGI web function.)
* **No auth.** The endpoint is public: any OpenAI client works with just `base_url`, and the
  playground opens in a browser. Set `DUMMYLLM_API_KEY` to require a key.
* **Concurrency.** Each container serves up to 500 requests at once (`@modal.concurrent`), and
  Modal adds containers (up to 10) past ~100 in flight per container. The Responses store
  (`previous_response_id`, `GET /v1/responses/{id}`) lives in a shared `modal.Dict`
  (`--response-store modal-dict:<name>`), so every container sees every stored response. Locally
  the default `--response-store memory` keeps them in process.
* **Image URLs.** `DUMMYLLM_PUBLIC_URL` is set to the domain, so `response_format="url"` links point
  at `https://dummyllm.com`.

## Using it in your own tests

```python
import openai, pytest
from dummyllm.testing import DummyLLMServer

@pytest.fixture(scope="session")
def llm():
    with DummyLLMServer(seed=42, tool_call_probability=1.0) as server:   # free port, background thread
        yield openai.OpenAI(base_url=server.base_url, api_key="test")

def test_my_agent(llm):
    ...
```

`dummyllm.create_app(Config(...))` returns the ASGI app if you'd rather mount or serve it yourself.

## Development

```bash
uv sync
uv run pytest
```

The test suite drives the server through the official `openai` Python SDK (sync and async),
validates every generated tool argument and structured output with `jsonschema`, and checks
streaming/non-streaming equivalence, the Responses event protocol, error shapes and pathological
inputs. dummyllm has also been checked against the OpenAI Agents SDK and LangChain's `ChatOpenAI`.

Layout:

```
src/dummyllm/
  app.py            ASGI app + routing
  common.py         auth, fault injection, latency, SSE, error envelope
  engine.py         request validation + "inference": tool decisions, text/JSON generation
  meow.py           the cat language model
  schema_faker.py   random values that conform to a JSON Schema
  images.py         pure-Python PNG cat renderer + image header parsing
  tokens.py         stand-in tokenizer + vision token formula
  routes/           chat, completions, responses, embeddings, images, models
  testing.py        DummyLLMServer for use in other test suites
  static/playground the built chat UI (generated by `npm run build` in web/)
web/                chat UI source: React + Tailwind CSS + shadcn/ui (Vite)
modal_app.py        Modal deployment (app.server)
```
