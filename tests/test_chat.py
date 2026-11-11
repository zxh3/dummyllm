from __future__ import annotations

import asyncio
import json

import jsonschema
import openai
import pytest
from conftest import TIME_TOOL, WEATHER_TOOL, png_data_url
from pydantic import BaseModel

from dummyllm.meow import SOUNDS

USER = [{"role": "user", "content": "Hello there"}]


def test_basic_completion(client):
    r = client.chat.completions.create(model="kiana", messages=USER)
    assert r.object == "chat.completion"
    assert r.id.startswith("chatcmpl-")
    assert r.model == "kiana"
    choice = r.choices[0]
    assert choice.message.role == "assistant"
    assert choice.finish_reason == "stop"
    text = choice.message.content
    assert text and any(s in text.lower() for s in SOUNDS)
    assert r.usage.prompt_tokens > 0
    assert r.usage.completion_tokens > 0
    assert r.usage.total_tokens == r.usage.prompt_tokens + r.usage.completion_tokens
    assert r.system_fingerprint.startswith("fp_")


def test_any_model_name_is_accepted(client):
    r = client.chat.completions.create(model="gpt-4o-2024-08-06", messages=USER)
    assert r.model == "gpt-4o-2024-08-06"


def test_n_choices(client):
    r = client.chat.completions.create(model="kiana", messages=USER, n=3)
    assert [c.index for c in r.choices] == [0, 1, 2]


@pytest.mark.parametrize("param", ["max_tokens", "max_completion_tokens"])
def test_max_tokens_truncates(client, param):
    for _ in range(5):
        r = client.chat.completions.create(model="kiana", messages=USER, **{param: 3})
        assert r.usage.completion_tokens <= 3
        if r.usage.completion_tokens == 3:
            assert r.choices[0].finish_reason in ("length", "stop")


def test_max_tokens_one_is_length(client):
    r = client.chat.completions.create(model="kiana", messages=USER, max_tokens=1)
    assert r.choices[0].finish_reason == "length"
    assert r.usage.completion_tokens == 1


def test_stop_sequence(client):
    for _ in range(10):
        r = client.chat.completions.create(model="kiana", messages=USER, stop=[" ", "."])
        assert " " not in r.choices[0].message.content
        assert "." not in r.choices[0].message.content


def test_logprobs(client):
    r = client.chat.completions.create(model="kiana", messages=USER, logprobs=True, top_logprobs=3)
    lp = r.choices[0].logprobs.content
    assert "".join(t.token for t in lp) == r.choices[0].message.content
    for t in lp:
        assert t.logprob <= 0
        assert len(t.top_logprobs) == 3
        assert t.top_logprobs[0].token == t.token
        assert bytes(t.bytes).decode() == t.token


def test_seed_is_deterministic(client):
    a = client.chat.completions.create(model="kiana", messages=USER, seed=7)
    b = client.chat.completions.create(model="kiana", messages=USER, seed=7)
    c = client.chat.completions.create(model="kiana", messages=USER, seed=8)
    assert a.choices[0].message.content == b.choices[0].message.content
    assert a.choices[0].message.content != c.choices[0].message.content


def test_streaming_matches_non_streaming_with_seed(client):
    kwargs = dict(model="kiana", messages=USER, seed=99, n=2, logprobs=True)
    full = client.chat.completions.create(**kwargs)
    stream = client.chat.completions.create(**kwargs, stream=True, stream_options={"include_usage": True})
    texts = {0: "", 1: ""}
    finish = {}
    usage = None
    roles = set()
    for chunk in stream:
        assert chunk.object == "chat.completion.chunk"
        if not chunk.choices:
            usage = chunk.usage
            continue
        assert chunk.usage is None
        for ch in chunk.choices:
            if ch.delta.role:
                roles.add((ch.index, ch.delta.role))
            texts[ch.index] += ch.delta.content or ""
            if ch.finish_reason:
                finish[ch.index] = ch.finish_reason
    assert roles == {(0, "assistant"), (1, "assistant")}
    assert texts[0] == full.choices[0].message.content
    assert texts[1] == full.choices[1].message.content
    assert finish == {0: full.choices[0].finish_reason, 1: full.choices[1].finish_reason}
    assert usage is not None and usage.completion_tokens == full.usage.completion_tokens
    assert usage.prompt_tokens == full.usage.prompt_tokens


def test_streaming_without_usage(client):
    chunks = list(client.chat.completions.create(model="kiana", messages=USER, stream=True))
    assert chunks[-1].choices[0].finish_reason == "stop"
    assert all(c.usage is None for c in chunks)


def test_stream_raw_sse_format(http):
    with http.stream("POST", "/v1/chat/completions", json={"model": "m", "messages": USER, "stream": True}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        lines = [ln for ln in r.iter_lines() if ln]
    assert lines[-1] == "data: [DONE]"
    assert all(ln.startswith("data: ") for ln in lines)
    first = json.loads(lines[0][6:])
    assert first["choices"][0]["delta"] == {"role": "assistant", "content": "", "refusal": None}


def test_stream_helper(client):
    with client.chat.completions.stream(model="kiana", messages=USER) as stream:
        for _ in stream:
            pass
        final = stream.get_final_completion()
    assert final.choices[0].message.content


def test_async_client_streaming(server):
    async def go():
        client = openai.AsyncOpenAI(base_url=server.base_url, api_key="x", max_retries=0)
        stream = await client.chat.completions.create(model="kiana", messages=USER, stream=True)
        text = ""
        async for chunk in stream:
            if chunk.choices:
                text += chunk.choices[0].delta.content or ""
        await client.close()
        return text

    assert asyncio.run(go())


# ---------------------------------------------------------------------------
# Tool calling


def _validate_args(tool, call):
    args = json.loads(call.function.arguments)
    jsonschema.validate(args, tool["function"]["parameters"])
    return args


def test_tool_choice_required(client):
    for _ in range(10):
        r = client.chat.completions.create(model="kiana", messages=USER, tools=[WEATHER_TOOL], tool_choice="required")
        choice = r.choices[0]
        assert choice.finish_reason == "tool_calls"
        assert choice.message.content is None
        call = choice.message.tool_calls[0]
        assert call.id.startswith("call_")
        assert call.type == "function"
        assert call.function.name == "get_weather"
        args = _validate_args(WEATHER_TOOL, call)
        assert set(args) >= {"city", "unit"}


def test_tool_choice_named(client):
    r = client.chat.completions.create(
        model="kiana",
        messages=USER,
        tools=[WEATHER_TOOL, TIME_TOOL],
        tool_choice={"type": "function", "function": {"name": "get_time"}},
    )
    calls = r.choices[0].message.tool_calls
    assert len(calls) == 1 and calls[0].function.name == "get_time"


def test_tool_choice_none(client):
    for _ in range(5):
        r = client.chat.completions.create(model="kiana", messages=USER, tools=[WEATHER_TOOL], tool_choice="none")
        assert r.choices[0].message.tool_calls is None
        assert r.choices[0].finish_reason == "stop"


def test_tool_probability_header(client):
    always = client.with_options(default_headers={"x-dummyllm-tool-probability": "1"})
    never = client.with_options(default_headers={"x-dummyllm-tool-probability": "0"})
    for _ in range(5):
        assert always.chat.completions.create(model="m", messages=USER, tools=[WEATHER_TOOL]).choices[0].message.tool_calls
        assert never.chat.completions.create(model="m", messages=USER, tools=[WEATHER_TOOL]).choices[0].message.tool_calls is None


def test_auto_tool_calls_happen_sometimes(client):
    outcomes = {
        bool(client.chat.completions.create(model="m", messages=USER, tools=[WEATHER_TOOL]).choices[0].message.tool_calls)
        for _ in range(40)
    }
    assert outcomes == {True, False}


def test_parallel_tool_calls(client):
    seen_multi = False
    for _ in range(40):
        r = client.chat.completions.create(model="m", messages=USER, tools=[WEATHER_TOOL, TIME_TOOL], tool_choice="required")
        calls = r.choices[0].message.tool_calls
        names = [c.function.name for c in calls]
        assert len(names) == len(set(names))
        assert len({c.id for c in calls}) == len(calls)
        seen_multi |= len(calls) > 1
    assert seen_multi


def test_parallel_tool_calls_disabled(client):
    for _ in range(20):
        r = client.chat.completions.create(
            model="m", messages=USER, tools=[WEATHER_TOOL, TIME_TOOL], tool_choice="required", parallel_tool_calls=False
        )
        assert len(r.choices[0].message.tool_calls) == 1


def test_tool_call_streaming_reassembles(client):
    kwargs = dict(model="m", messages=USER, tools=[WEATHER_TOOL, TIME_TOOL], tool_choice="required", seed=3)
    full = client.chat.completions.create(**kwargs)
    calls: dict[int, dict] = {}
    finish = None
    for chunk in client.chat.completions.create(**kwargs, stream=True):
        ch = chunk.choices[0]
        for tc in ch.delta.tool_calls or []:
            entry = calls.setdefault(tc.index, {"id": None, "name": None, "arguments": ""})
            if tc.id:
                entry["id"] = tc.id
            if tc.function.name:
                entry["name"] = tc.function.name
            entry["arguments"] += tc.function.arguments or ""
        finish = ch.finish_reason or finish
    assert finish == "tool_calls"
    expected = full.choices[0].message.tool_calls
    assert [c["name"] for c in calls.values()] == [c.function.name for c in expected]
    assert [c["arguments"] for c in calls.values()] == [c.function.arguments for c in expected]
    assert all(c["id"].startswith("call_") for c in calls.values())


def test_agent_loop_terminates(client):
    messages = [{"role": "user", "content": "What's the weather in Paris?"}]
    for _turn in range(30):
        r = client.chat.completions.create(model="m", messages=messages, tools=[WEATHER_TOOL])
        msg = r.choices[0].message
        if not msg.tool_calls:
            assert msg.content
            break
        messages.append(msg.model_dump(exclude_none=True))
        for call in msg.tool_calls:
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps({"temp": 21})})
    else:
        pytest.fail("agent loop never terminated")


def test_tool_result_without_call_is_rejected(client):
    with pytest.raises(openai.BadRequestError) as e:
        client.chat.completions.create(
            model="m", messages=[*USER, {"role": "tool", "tool_call_id": "call_x", "content": "42"}]
        )
    assert "must be a response to a preceeding message with 'tool_calls'" in e.value.message


def test_missing_tool_result_is_rejected(client):
    with pytest.raises(openai.BadRequestError) as e:
        client.chat.completions.create(
            model="m",
            messages=[
                *USER,
                {
                    "role": "assistant",
                    "tool_calls": [{"id": "call_a", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}],
                },
                {"role": "user", "content": "well?"},
            ],
            tools=[WEATHER_TOOL],
        )
    assert "call_a" in e.value.message


def test_unknown_named_tool_is_rejected(client):
    with pytest.raises(openai.BadRequestError):
        client.chat.completions.create(
            model="m", messages=USER, tools=[WEATHER_TOOL], tool_choice={"type": "function", "function": {"name": "nope"}}
        )


def test_tool_choice_without_tools_is_rejected(client):
    with pytest.raises(openai.BadRequestError):
        client.chat.completions.create(model="m", messages=USER, tool_choice="required")


def test_invalid_function_name_is_rejected(client):
    bad = {"type": "function", "function": {"name": "has spaces", "parameters": {"type": "object"}}}
    with pytest.raises(openai.BadRequestError):
        client.chat.completions.create(model="m", messages=USER, tools=[bad])


def test_legacy_functions(client):
    r = client.chat.completions.create(
        model="m", messages=USER, functions=[WEATHER_TOOL["function"]], function_call={"name": "get_weather"}
    )
    msg = r.choices[0].message
    assert r.choices[0].finish_reason == "function_call"
    assert msg.function_call.name == "get_weather"
    jsonschema.validate(json.loads(msg.function_call.arguments), WEATHER_TOOL["function"]["parameters"])


def test_custom_tool(client):
    r = client.chat.completions.create(
        model="m",
        messages=USER,
        tools=[{"type": "custom", "custom": {"name": "shell", "description": "run a shell command"}}],
        tool_choice="required",
    )
    call = r.choices[0].message.tool_calls[0]
    assert call.type == "custom"
    assert call.custom.name == "shell"
    assert isinstance(call.custom.input, str) and call.custom.input


def test_strict_tool_with_refs_and_nesting(client):
    params = {
        "type": "object",
        "$defs": {
            "Address": {
                "type": "object",
                "properties": {"street": {"type": "string"}, "zip": {"type": ["string", "null"]}},
                "required": ["street", "zip"],
                "additionalProperties": False,
            }
        },
        "properties": {
            "name": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 4},
            "home": {"$ref": "#/$defs/Address"},
            "score": {"type": "number", "minimum": 0, "maximum": 1},
            "kind": {"anyOf": [{"type": "string", "enum": ["cat", "kitten"]}, {"type": "null"}]},
        },
        "required": ["name", "tags", "home", "score", "kind"],
        "additionalProperties": False,
    }
    tool = {"type": "function", "function": {"name": "save", "parameters": params, "strict": True}}
    for _ in range(25):
        r = client.chat.completions.create(model="m", messages=USER, tools=[tool], tool_choice="required")
        _validate_args(tool, r.choices[0].message.tool_calls[0])


# ---------------------------------------------------------------------------
# Structured outputs


class Pet(BaseModel):
    name: str
    age: int
    lives_left: int
    favorite_toys: list[str]
    is_grumpy: bool


def test_structured_output_parse(client):
    for _ in range(10):
        r = client.chat.completions.parse(model="m", messages=USER, response_format=Pet)
        pet = r.choices[0].message.parsed
        assert isinstance(pet, Pet)


def test_json_schema_response_format(client):
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}, "confidence": {"type": "number", "minimum": 0, "maximum": 1}},
        "required": ["answer", "confidence"],
        "additionalProperties": False,
    }
    r = client.chat.completions.create(
        model="m",
        messages=USER,
        response_format={"type": "json_schema", "json_schema": {"name": "reply", "schema": schema, "strict": True}},
    )
    jsonschema.validate(json.loads(r.choices[0].message.content), schema)


def test_json_object_response_format(client):
    r = client.chat.completions.create(model="m", messages=USER, response_format={"type": "json_object"})
    assert isinstance(json.loads(r.choices[0].message.content), dict)


# ---------------------------------------------------------------------------
# Vision / reasoning


def test_image_input(client):
    r = client.chat.completions.create(
        model="kiana-vision",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": png_data_url(), "detail": "low"}},
                ],
            }
        ],
        seed=1,
    )
    assert r.choices[0].message.content
    text_only = client.chat.completions.create(model="m", messages=[{"role": "user", "content": "What is in this image?"}])
    assert r.usage.prompt_tokens - text_only.usage.prompt_tokens == 85  # low-detail image cost


def test_image_tokens_scale_with_size(client):
    def cost(w, h):
        return client.chat.completions.create(
            model="m",
            messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": png_data_url(w, h), "detail": "high"}}]}],
        ).usage.prompt_tokens

    assert cost(512, 512) < cost(2048, 1024)


def test_remote_image_url_accepted(client):
    r = client.chat.completions.create(
        model="m",
        messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}}]}],
    )
    assert r.choices[0].message.content


def test_invalid_image_data_rejected(client):
    with pytest.raises(openai.BadRequestError):
        client.chat.completions.create(
            model="m",
            messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,bm90IGFuIGltYWdl"}}]}],
        )


def test_reasoning_effort(client):
    r = client.chat.completions.create(model="kiana-reasoning", messages=USER, reasoning_effort="high")
    assert r.usage.completion_tokens_details.reasoning_tokens > 0
    assert r.usage.completion_tokens > r.usage.completion_tokens_details.reasoning_tokens


def test_reasoning_can_exhaust_budget(client):
    r = client.chat.completions.create(model="kiana-reasoning", messages=USER, reasoning_effort="high", max_completion_tokens=5)
    assert r.choices[0].finish_reason == "length"
    assert r.choices[0].message.content == ""
    assert r.usage.completion_tokens == 5


# ---------------------------------------------------------------------------
# Validation


@pytest.mark.parametrize(
    "kwargs, param",
    [
        ({"temperature": 3}, "temperature"),
        ({"top_p": -1}, "top_p"),
        ({"n": 0}, "n"),
        ({"max_tokens": 0}, "max_tokens"),
        ({"top_logprobs": 3}, "top_logprobs"),
        ({"stop": ["a", "b", "c", "d", "e"]}, "stop"),
        ({"stream_options": {"include_usage": True}}, "stream_options"),
        ({"presence_penalty": 5}, "presence_penalty"),
    ],
)
def test_parameter_validation(client, kwargs, param):
    with pytest.raises(openai.BadRequestError) as e:
        client.chat.completions.create(model="m", messages=USER, **kwargs)
    assert e.value.body["param"] == param
    assert e.value.body["type"] == "invalid_request_error"


def test_empty_messages(client):
    with pytest.raises(openai.BadRequestError):
        client.chat.completions.create(model="m", messages=[])


def test_bad_role(http):
    r = http.post("/v1/chat/completions", json={"model": "m", "messages": [{"role": "cat", "content": "hi"}]})
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "messages[0].role"


def test_missing_model(http):
    r = http.post("/v1/chat/completions", json={"messages": USER})
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "model"


def test_invalid_json(http):
    r = http.post("/v1/chat/completions", content=b"{not json", headers={"content-type": "application/json"})
    assert r.status_code == 400
    assert "could not parse the JSON body" in r.json()["error"]["message"]


def test_multimodal_content_parts_and_developer_role(client):
    r = client.chat.completions.create(
        model="m",
        messages=[
            {"role": "developer", "content": [{"type": "text", "text": "Be a cat."}]},
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "content": "Meow."},
            {"role": "user", "content": "again"},
        ],
    )
    assert r.choices[0].message.content


# ---------------------------------------------------------------------------
# Reasoning (DeepSeek/vLLM-style `reasoning_content`)

from dummyllm.tokens import count_tokens  # noqa: E402


def test_reasoning_content_non_streaming(client):
    r = client.chat.completions.create(model="kiana", messages=USER, reasoning_effort="medium")
    msg = r.choices[0].message
    thinking = msg.reasoning_content  # extra field, exposed by the SDK's permissive models
    assert isinstance(thinking, str) and thinking
    assert msg.content and thinking != msg.content
    assert r.usage.completion_tokens_details.reasoning_tokens == count_tokens(thinking)
    assert r.usage.completion_tokens == count_tokens(thinking) + count_tokens(msg.content)


def test_reasoning_effort_scales_length(client):
    def n(effort):
        return client.chat.completions.create(model="m", messages=USER, reasoning_effort=effort, seed=1).usage.completion_tokens_details.reasoning_tokens

    assert 0 < n("low") < n("high")


def test_no_reasoning_without_effort(client):
    r = client.chat.completions.create(model="kiana", messages=USER)
    assert getattr(r.choices[0].message, "reasoning_content", None) is None
    assert r.usage.completion_tokens_details.reasoning_tokens == 0
    r = client.chat.completions.create(model="kiana-reasoning", messages=USER, reasoning_effort="none")
    assert getattr(r.choices[0].message, "reasoning_content", None) is None


def test_reasoning_model_thinks_by_default(client):
    r = client.chat.completions.create(model="kiana-reasoning", messages=USER)
    assert r.choices[0].message.reasoning_content
    assert r.usage.completion_tokens_details.reasoning_tokens > 0


def test_reasoning_streams_before_content(client):
    kwargs = dict(model="m", messages=USER, reasoning_effort="low", seed=21)
    full = client.chat.completions.create(**kwargs)
    phases, thinking, text = [], "", ""
    for chunk in client.chat.completions.create(**kwargs, stream=True, stream_options={"include_usage": True}):
        if not chunk.choices:
            assert chunk.usage.completion_tokens_details.reasoning_tokens == full.usage.completion_tokens_details.reasoning_tokens
            continue
        d = chunk.choices[0].delta
        r = getattr(d, "reasoning_content", None)
        if r:
            thinking += r
            phases.append("think")
        if d.content:
            text += d.content
            phases.append("answer")
    assert phases.index("answer") > phases.index("think")
    assert "think" not in phases[phases.index("answer") :]
    assert thinking == full.choices[0].message.reasoning_content
    assert text == full.choices[0].message.content


def test_reasoning_with_tool_calls(client):
    r = client.chat.completions.create(model="m", messages=USER, tools=[WEATHER_TOOL], tool_choice="required", reasoning_effort="low")
    msg = r.choices[0].message
    assert msg.reasoning_content and msg.tool_calls and msg.content is None


def test_reasoning_n_choices_are_independent(client):
    r = client.chat.completions.create(model="m", messages=USER, reasoning_effort="low", n=2)
    a, b = (c.message.reasoning_content for c in r.choices)
    assert a and b and a != b
    assert r.usage.completion_tokens_details.reasoning_tokens == count_tokens(a) + count_tokens(b)


def test_reasoning_effort_max(client):
    r = client.chat.completions.create(model="kiana", messages=USER, reasoning_effort="max", seed=3)
    high = client.chat.completions.create(model="kiana", messages=USER, reasoning_effort="high", seed=3)
    assert r.usage.completion_tokens_details.reasoning_tokens > high.usage.completion_tokens_details.reasoning_tokens
