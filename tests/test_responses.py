from __future__ import annotations

import base64
import json

import jsonschema
import openai
import pytest
from conftest import png_data_url
from pydantic import BaseModel

FN_TOOL = {
    "type": "function",
    "name": "get_weather",
    "description": "Get the weather",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}},
        "required": ["city", "unit"],
        "additionalProperties": False,
    },
    "strict": True,
}


def test_basic(client):
    r = client.responses.create(model="kiana", input="Hello")
    assert r.object == "response"
    assert r.id.startswith("resp_")
    assert r.status == "completed"
    assert r.output[0].type == "message"
    assert r.output[0].role == "assistant"
    assert r.output_text
    assert r.usage.input_tokens > 0 and r.usage.output_tokens > 0
    assert r.usage.total_tokens == r.usage.input_tokens + r.usage.output_tokens


def test_instructions_and_message_list(client):
    r = client.responses.create(
        model="m",
        instructions="You are a cat.",
        input=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Meow."},
            {"role": "user", "content": [{"type": "input_text", "text": "again"}]},
        ],
    )
    assert r.instructions == "You are a cat."
    assert r.output_text


def test_streaming_event_protocol(client):
    events = list(client.responses.create(model="m", input="hi", stream=True, extra_body={"seed": 5}))
    types = [e.type for e in events]
    assert types[0] == "response.created"
    assert types[1] == "response.in_progress"
    assert types[-1] == "response.completed"
    assert [e.sequence_number for e in events] == list(range(len(events)))
    deltas = "".join(e.delta for e in events if e.type == "response.output_text.delta")
    done = next(e for e in events if e.type == "response.output_text.done")
    assert deltas == done.text
    assert events[-1].response.output_text == deltas
    for t in ("response.output_item.added", "response.content_part.added", "response.content_part.done", "response.output_item.done"):
        assert t in types


def test_stream_helper(client):
    with client.responses.stream(model="m", input="hi") as stream:
        text = "".join(e.delta for e in stream if e.type == "response.output_text.delta")
        final = stream.get_final_response()
    assert text == final.output_text


def test_streaming_matches_non_streaming_with_seed(client):
    a = client.responses.create(model="m", input="hi", extra_body={"seed": 11})  # seed is a dummyllm extension here
    events = list(client.responses.create(model="m", input="hi", extra_body={"seed": 11}, stream=True))
    assert events[-1].response.output_text == a.output_text


def test_function_call_roundtrip_with_previous_response(client):
    r1 = client.responses.create(model="m", input="weather in Paris?", tools=[FN_TOOL], tool_choice="required")
    calls = [o for o in r1.output if o.type == "function_call"]
    assert calls and r1.output_text == ""
    call = calls[0]
    assert call.call_id.startswith("call_") and call.id.startswith("fc_")
    jsonschema.validate(json.loads(call.arguments), FN_TOOL["parameters"])

    r2 = client.responses.create(
        model="m",
        previous_response_id=r1.id,
        input=[{"type": "function_call_output", "call_id": c.call_id, "output": '{"temp": 20}'} for c in calls],
        tools=[FN_TOOL],
        tool_choice="none",
    )
    assert r2.previous_response_id == r1.id
    assert r2.output_text
    assert r2.usage.input_tokens > r1.usage.input_tokens


def test_function_call_roundtrip_stateless(client):
    r1 = client.responses.create(model="m", input="weather?", tools=[FN_TOOL], tool_choice="required", store=False)
    call = next(o for o in r1.output if o.type == "function_call")
    r2 = client.responses.create(
        model="m",
        input=[
            {"role": "user", "content": "weather?"},
            {"type": "function_call", "call_id": call.call_id, "name": call.name, "arguments": call.arguments},
            {"type": "function_call_output", "call_id": call.call_id, "output": "sunny"},
        ],
        tools=[FN_TOOL],
        tool_choice="none",
    )
    assert r2.output_text
    with pytest.raises(openai.NotFoundError):
        client.responses.retrieve(r1.id)  # store=False


def test_orphan_function_call_output_rejected(client):
    with pytest.raises(openai.BadRequestError) as e:
        client.responses.create(model="m", input=[{"type": "function_call_output", "call_id": "call_nope", "output": "x"}])
    assert "call_nope" in e.value.message


def test_unknown_previous_response(client):
    with pytest.raises(openai.BadRequestError) as e:
        client.responses.create(model="m", input="hi", previous_response_id="resp_doesnotexist")
    assert e.value.body["param"] == "previous_response_id"


def test_streaming_function_call(client):
    events = list(client.responses.create(model="m", input="hi", tools=[FN_TOOL], tool_choice="required", stream=True))
    args = "".join(e.delta for e in events if e.type == "response.function_call_arguments.delta")
    done = next(e for e in events if e.type == "response.function_call_arguments.done")
    assert args == done.arguments
    jsonschema.validate(json.loads(args), FN_TOOL["parameters"])
    added = next(e for e in events if e.type == "response.output_item.added")
    assert added.item.type == "function_call" and added.item.arguments == ""


def test_retrieve_delete_and_input_items(client):
    r = client.responses.create(model="m", input=[{"role": "user", "content": "a"}, {"role": "user", "content": "b"}])
    got = client.responses.retrieve(r.id)
    assert got.id == r.id and got.output_text == r.output_text
    items = client.responses.input_items.list(r.id, order="asc")
    assert [i.content[0].text for i in items.data] == ["a", "b"]
    assert all(i.id for i in items.data)
    deleted = client.responses.delete(r.id)
    assert deleted is None or getattr(deleted, "deleted", True)
    with pytest.raises(openai.NotFoundError):
        client.responses.retrieve(r.id)


class Cat(BaseModel):
    name: str
    lives: int
    likes: list[str]


def test_structured_output_parse(client):
    for _ in range(5):
        r = client.responses.parse(model="m", input="describe a cat", text_format=Cat)
        assert isinstance(r.output_parsed, Cat)


def test_image_generation_tool(client):
    r = client.responses.create(
        model="m", input="please draw a cat", tools=[{"type": "image_generation", "size": "512x512"}], tool_choice={"type": "image_generation"}
    )
    ig = next(o for o in r.output if o.type == "image_generation_call")
    png = base64.b64decode(ig.result)
    assert png.startswith(b"\x89PNG")
    import struct

    assert struct.unpack(">II", png[16:24]) == (512, 512)
    assert r.output_text  # the cat comments on its art


def test_image_generation_streaming_partials(client):
    events = list(
        client.responses.create(
            model="m",
            input="draw",
            tools=[{"type": "image_generation", "partial_images": 2, "size": "256x256"}],
            tool_choice={"type": "image_generation"},
            stream=True,
        )
    )
    types = [e.type for e in events]
    assert types.count("response.image_generation_call.partial_image") == 2
    assert "response.image_generation_call.completed" in types


def test_image_input(client):
    r = client.responses.create(
        model="m",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "what's this?"},
                    {"type": "input_image", "image_url": png_data_url(), "detail": "low"},
                ],
            }
        ],
    )
    assert r.output_text


def test_reasoning_summary(client):
    r = client.responses.create(model="kiana", input="think", reasoning={"effort": "medium", "summary": "auto"})
    rs = r.output[0]
    assert rs.type == "reasoning"
    assert rs.summary and rs.summary[0].text
    assert r.usage.output_tokens_details.reasoning_tokens > 0

    events = list(client.responses.create(model="m", input="think", reasoning={"effort": "low", "summary": "auto"}, stream=True))
    types = [e.type for e in events]
    assert "response.reasoning_summary_text.delta" in types


def test_max_output_tokens_incomplete(client):
    r = client.responses.create(model="m", input="hi", max_output_tokens=16, reasoning={"effort": "high"})
    assert r.status == "incomplete"
    assert r.incomplete_details.reason == "max_output_tokens"
    events = list(client.responses.create(model="m", input="hi", max_output_tokens=16, reasoning={"effort": "high"}, stream=True))
    assert events[-1].type == "response.incomplete"


def test_max_output_tokens_minimum(client):
    with pytest.raises(openai.BadRequestError):
        client.responses.create(model="m", input="hi", max_output_tokens=5)


def test_background_mode(client):
    r = client.responses.create(model="m", input="hi", background=True)
    assert r.status == "queued"
    polled = client.responses.retrieve(r.id)
    assert polled.status == "completed" and polled.output_text


def test_input_tokens_endpoint(http):
    r = http.post("/v1/responses/input_tokens", json={"model": "m", "input": "hello world"})
    assert r.status_code == 200
    assert r.json()["object"] == "response.input_tokens"
    assert r.json()["input_tokens"] > 0


def test_hosted_tools_are_ignored(client):
    r = client.responses.create(model="m", input="search the web", tools=[{"type": "web_search_preview"}])
    assert r.output_text


def test_missing_input(client):
    with pytest.raises(openai.BadRequestError):
        client.responses.create(model="m")


def test_reasoning_model_defaults_and_counts(client):
    r = client.responses.create(model="my-reasoning-model", input="think about it")
    assert r.usage.output_tokens_details.reasoning_tokens > 0
    assert r.output[0].type == "reasoning"
    assert r.reasoning.effort == "medium"
    plain = client.responses.create(model="kiana", input="hi")
    assert plain.usage.output_tokens_details.reasoning_tokens == 0
    assert all(o.type != "reasoning" for o in plain.output)


def test_function_call_over_budget_is_incomplete(client):
    big = {
        "type": "function",
        "name": "save_profile",
        "parameters": {
            "type": "object",
            "properties": {k: {"type": "string"} for k in ("first", "last", "city", "street", "email", "bio")},
            "required": ["first", "last", "city", "street", "email", "bio"],
            "additionalProperties": False,
        },
    }
    r = client.responses.create(model="kiana", input="save me", tools=[big], tool_choice="required", max_output_tokens=16)
    assert r.status == "incomplete"
    assert r.incomplete_details.reason == "max_output_tokens"
    assert r.usage.output_tokens <= 16
    calls = [o for o in r.output if o.type == "function_call"]
    assert calls and calls[-1].status == "incomplete"
