# dummyllm 🐱

An **OpenAI-compatible LLM server that only speaks cat.**

Point any OpenAI client at it and you get well-formed, spec-shaped responses (chat, streaming,
tool calls, structured outputs, reasoning, vision input, image generation, embeddings), except
every word is `meow`, `mrrp`, `purr` or `*knocks your coffee off the desk*`.

It's built for **testing LLM applications without an LLM**: fast, free, offline, deterministic
when you want it, and as strict about request validation as the real API, so client bugs show up
locally instead of in production.

```text
$ curl -s https://dummyllm.com/v1/chat/completions \
    -d '{"model":"kiana","messages":[{"role":"user","content":"hi"}]}' | jq -r '.choices[0].message.content'
Mrrp rrrow. Prrr meow meow meow! *blinks slowly* Nya?
```

**Try it:** the playground is at **<https://dummyllm.com>**, and the API is at
`https://dummyllm.com/v1`. No key needed.

## Quick start

Use the hosted endpoint:

```python
from openai import OpenAI

client = OpenAI(base_url="https://dummyllm.com/v1", api_key="anything")
reply = client.chat.completions.create(model="kiana", messages=[{"role": "user", "content": "hi"}])
print(reply.choices[0].message.content)
```

Or run it locally (needs [uv](https://docs.astral.sh/uv/)):

```bash
uvx --from git+https://github.com/zxh3/dummyllm dummyllm   # http://127.0.0.1:8000/v1
```

Or with Docker:

```bash
git clone https://github.com/zxh3/dummyllm && cd dummyllm
docker build -t dummyllm .
docker run -p 8000:8000 dummyllm
```

Any model name is accepted (`gpt-4o`, `o3`, `whatever`) and echoed back, so swapping dummyllm in
usually means changing only `base_url`. The advertised model is `kiana`, which does chat,
reasoning, vision and tools.

The hosted endpoint is a shared demo. For load tests or CI, run your own copy.

## Playground

Open the server root in a browser (<http://127.0.0.1:8000/> locally) for a chat UI with:

* streaming replies with time-to-first-token and tokens/s stats
* a reasoning effort setting that shows the "thinking" as it streams
* tool calling, with demo tools that "run" in the browser and feed their results back
* image uploads (vision input) and image generation
* system instructions, temperature, reply length, max tokens, seed and JSON mode
* **View code**, which turns the current conversation into Python, Node.js or curl
* chat history kept in your browser (IndexedDB), light and dark themes, and a mobile layout

API clients aren't affected by the UI: requests to `/` without `Accept: text/html` get a JSON
index (also at `/info`), and the UI is also served at `/playground`.

## What's supported

| Endpoint | Highlights |
|---|---|
| `POST /v1/chat/completions` | streaming (+ `include_usage`), tools, structured outputs, reasoning, image inputs, logprobs (details below) |
| `POST /v1/responses` | the full streaming event protocol, function and custom tools, the `image_generation` tool, `previous_response_id`, `store`, `background`, `text.format`, `reasoning` with summaries |
| `GET`/`DELETE /v1/responses/{id}`, `GET …/input_items`, `POST …/cancel`, `POST /v1/responses/input_tokens` | stored responses |
| `POST /v1/completions` | legacy completions: multiple prompts, `n`, `echo`, `logprobs`, streaming |
| `POST /v1/embeddings` | float and base64 (the SDK default), `dimensions`; unit-length vectors, deterministic per input |
| `POST /v1/images/generations`, `/edits`, `/variations` | real PNGs of procedurally drawn cats at the requested size, as `b64_json` or `url` |
| `GET /v1/models`, `GET /v1/models/{id}`, `GET /health` | |

Every route also works without the `/v1` prefix.

**Chat Completions in detail:** `n`, `max_tokens` / `max_completion_tokens`, `stop`,
`logprobs` / `top_logprobs`, `seed`, function and custom tools, every `tool_choice` form (`auto`,
`none`, `required`, a named tool, `allowed_tools`), `parallel_tool_calls`, legacy `functions` /
`function_call`, `response_format` (`json_object`, and `json_schema`, which works with
`client.chat.completions.parse(...)`), and image inputs (data URLs and https).

**Reasoning:** `reasoning_effort` (`minimal` through `xhigh` and `max`) makes the model think
before answering. Chat Completions returns the thinking in `message.reasoning_content` (streamed as
`delta.reasoning_content`, the DeepSeek / vLLM convention); the Responses API returns reasoning
items. Thinking counts toward `reasoning_tokens`. Model names containing `reason` think by default.

**Not implemented:** audio, files, batches, fine-tuning, assistants, vector stores and realtime.
Hosted Responses tools other than `image_generation` (web search, file search, code interpreter,
MCP) are accepted but never called.

## How the cat behaves

**Text.** Replies are sentences of cat sounds with the occasional `*action*`. The cat reacts to
context: it sniffs images you send, bats at tool results, and proudly presents its drawings.

**Tool calls.** When tools are offered:

* `tool_choice="required"` or a named tool: always calls.
* `tool_choice="none"`: never calls.
* `tool_choice="auto"`: calls with probability 0.5, dropping to 0.2 right after a tool result so
  agent loops end.
* With `parallel_tool_calls` (on by default) it sometimes calls 2–3 different tools at once.
* Arguments are **random values that validate against the tool's JSON Schema**: types, `required`,
  `enum`, `const`, `anyOf` / `oneOf` / `allOf`, `$ref` / `$defs` (including recursive schemas),
  min/max, `multipleOf`, string formats (date-time, email, uuid, uri…), nullable types,
  `prefixItems` and more. Property names flavor the values (`city` → `"Meowtown"`,
  `name` → `"Whiskers"`).
* In the Responses API, if the `image_generation` tool is available and you ask it to draw
  something, the cat draws.

**Structured outputs.** `json_schema` formats produce JSON that matches the schema; `json_object`
produces a small random object.

**Images.** Generated images are real PNGs of randomized cat faces at exactly the requested size.
`url` responses point back at the server (`/v1/images/files/…`). The URL encodes the seed, so it
never expires and always returns the same picture.

**Vision input.** Images are never fetched or looked at, but data-URL images are validated (they
must be real PNG / JPEG / GIF / WebP / BMP), and their dimensions feed OpenAI's tile formula for
`prompt_tokens` (85 for `detail: "low"`, otherwise `85 + 170 × tiles`).

**Token accounting.** Counts come from a simple word-and-punctuation tokenizer (not tiktoken) but
are self-consistent: streamed deltas add up to the content, and usage matches what was produced.
`max_tokens` is a hard cap that tool calls and reasoning count against. A request that runs out
ends with `finish_reason: "length"` (`status: "incomplete"` in the Responses API), so you can test
the "it thought so long there's no answer" case. A small cap is reached the way a real model would
reach it.

**Validation.** Requests are validated like the real API, with OpenAI-shaped errors
(`{"error": {"message", "type", "param", "code"}}`) that the SDKs raise as `BadRequestError` and
so on. This covers:

* parameter ranges and types (`temperature`, `n`, `top_logprobs` without `logprobs`,
  `stream_options` without `stream`, at most 4 stop sequences…)
* message structure: roles, content parts, image data
* tool-call bookkeeping: a `tool` message must answer an earlier `tool_calls` id, and every tool
  call must be answered before the next turn; in Responses, `function_call_output` must match a
  known `call_id`
* tool definitions: the function-name pattern, object-typed parameters, duplicate names, and a
  `tool_choice` that names a missing tool
* unknown `previous_response_id`s and malformed JSON (including `NaN`)

## Testing knobs

**Determinism.** Pass `seed` in a request and the same body gives the same output; streaming and
non-streaming produce identical content. Start the server with `--seed N` to make every response
a deterministic function of its request.

**Per-request headers.** These override the server settings for a single request, so one shared
server can play many scenarios:

| Header | Effect |
|---|---|
| `x-dummyllm-ttft-ms: 300` | wait 300 ms before the first token |
| `x-dummyllm-tokens-per-second: 50` | stream at 50 tokens/s (non-streaming requests wait the equivalent time) |
| `x-dummyllm-reply-tokens: 200` | replies of about 200 tokens (0.8–1.2×, up to 4000) that still end with `"stop"` |
| `x-dummyllm-tool-probability: 1` | always call a tool under `tool_choice="auto"` (`0` never calls one) |
| `x-dummyllm-error: 429` | fail with that status (any 4xx / 5xx) and an OpenAI-shaped error; 429 and 503 include `retry-after` |

**Server-wide.** `--ttft-ms` and `--tps` set default latency, `--error-rate 0.1` fails 10% of
requests at random with 429 / 500 / 503, and `--api-key sk-secret` requires
`Authorization: Bearer sk-secret` (or Azure-style `api-key`). Without `--api-key`, any key is
accepted.

## Using it in your own tests

```python
import openai
import pytest
from dummyllm.testing import DummyLLMServer

@pytest.fixture(scope="session")
def llm():
    # Runs on a free port in a background thread.
    with DummyLLMServer(seed=42, tool_call_probability=1.0) as server:
        yield openai.OpenAI(base_url=server.base_url, api_key="test")

def test_my_agent(llm):
    ...
```

`DummyLLMServer` takes any configuration option as a keyword argument. To mount or serve the ASGI
app yourself, use `dummyllm.create_app(Config(...))`.

## Configuration

Every option is a CLI flag and a `DUMMYLLM_*` environment variable:

| Flag | Environment variable | Default |
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
| `--response-store` | `DUMMYLLM_RESPONSE_STORE` | `memory` (or `modal-dict:<name>`, see below) |
| `--response-store-size` | `DUMMYLLM_RESPONSE_STORE_SIZE` | `1000` |
| `--public-url` | `DUMMYLLM_PUBLIC_URL` | the request's host (used in image URLs) |
| `--log-level` | `DUMMYLLM_LOG_LEVEL` | `info` |

## Deploying to Modal

[`modal_app.py`](modal_app.py) is how <https://dummyllm.com> is hosted. To deploy your own copy:

```bash
uvx modal deploy modal_app.py   # or `uvx modal serve` for a hot-reloading dev run
```

* **Domain.** Set `DOMAIN` in `modal_app.py` to a domain registered in your Modal workspace, or
  drop `custom_domains` to use the default `*.modal.run` URL. Set `DUMMYLLM_PUBLIC_URL` to match,
  so image URLs point at it.
* **Scaling.** Each container serves up to 500 requests at once, and Modal adds containers (up to
  10) once more than ~100 are in flight per container. Stored responses live in a shared
  `modal.Dict` (`DUMMYLLM_RESPONSE_STORE=modal-dict:<name>`), so `previous_response_id` works
  whichever container picks up the request.
* **Auth.** The endpoint is public. Set `DUMMYLLM_API_KEY` (for example from a `modal.Secret`) to
  require a key.

## Development

```bash
git clone https://github.com/zxh3/dummyllm && cd dummyllm
uv sync
uv run pytest
uv run dummyllm
```

The tests drive the server through the official `openai` Python SDK (sync and async), validate
every generated tool argument and structured output with `jsonschema`, and check
streaming / non-streaming equivalence, the Responses event protocol, error shapes and pathological
inputs. dummyllm has also been tried with the OpenAI Agents SDK and LangChain's `ChatOpenAI`.

The playground source lives in [`web/`](web/) (React, Tailwind CSS, shadcn/ui, Vite). Its
production build is committed to `src/dummyllm/static/playground/`, so the Python package serves
it without Node. To work on it:

```bash
uv run dummyllm                 # API on :8000
cd web && npm install
npm run dev                     # Vite on :5173, proxying /v1 to :8000 (override with DUMMYLLM_URL)
npm run build                   # rebuild the bundle that ships in the Python package
```

Layout:

```text
src/dummyllm/
  app.py              ASGI app and routing
  common.py           auth, per-request overrides, fault injection, latency, SSE, errors
  engine.py           request validation and "inference": tool decisions, text/JSON generation
  meow.py             the cat language model
  schema_faker.py     random values that conform to a JSON Schema
  images.py           pure-Python PNG cat renderer and image header parsing
  tokens.py           stand-in tokenizer and vision token formula
  store.py            Responses API storage (in memory or a modal.Dict)
  routes/             chat, completions, responses, embeddings, images, models
  testing.py          DummyLLMServer for other test suites
  static/playground/  the built playground (from `npm run build` in web/)
web/                  playground source
modal_app.py          the Modal deployment behind dummyllm.com
```

## License

[MIT](LICENSE)
