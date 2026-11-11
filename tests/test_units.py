"""Unit tests for the generators (no server)."""

from __future__ import annotations

import random

import jsonschema
import pytest

from dummyllm.config import Config
from dummyllm.images import image_size, render_cat
from dummyllm.meow import Meower, truncate_at_stop
from dummyllm.schema_faker import fake_from_schema
from dummyllm.tokens import image_tokens, split_tokens

SCHEMAS = [
    {"type": "object", "properties": {}},
    {"type": "object", "properties": {"a": {"type": "string", "minLength": 20, "maxLength": 25}}, "required": ["a"]},
    {"type": "object", "properties": {"n": {"type": "integer", "minimum": 5, "maximum": 6}}, "required": ["n"]},
    {"type": "object", "properties": {"n": {"type": "integer", "multipleOf": 7, "minimum": 1, "maximum": 50}}, "required": ["n"]},
    {"type": "object", "properties": {"x": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1}}, "required": ["x"]},
    {"type": "object", "properties": {"t": {"type": "array", "prefixItems": [{"type": "integer"}, {"type": "string"}], "items": False}}},
    {"type": "object", "properties": {"u": {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 1000}, "uniqueItems": True}}},
    {"type": "object", "properties": {"d": {"type": "string", "format": "date-time"}, "e": {"type": "string", "format": "email"}, "id": {"type": "string", "format": "uuid"}}, "required": ["d", "e", "id"]},
    {"type": "object", "additionalProperties": {"type": "integer"}, "minProperties": 1},
    {"type": "object", "properties": {"c": {"const": "meow"}, "e": {"enum": [1, "two", None]}}, "required": ["c", "e"]},
    {
        "type": "object",
        "properties": {"v": {"oneOf": [{"type": "object", "properties": {"k": {"const": "a"}}, "required": ["k"], "additionalProperties": False}, {"type": "integer"}]}},
        "required": ["v"],
    },
    {
        "type": "object",
        "allOf": [
            {"properties": {"a": {"type": "string"}}, "required": ["a"]},
            {"properties": {"b": {"type": "boolean"}}, "required": ["b"]},
        ],
    },
    {
        # Recursive tree via $ref
        "type": "object",
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {"value": {"type": "integer"}, "children": {"type": "array", "items": {"$ref": "#/$defs/Node"}}},
                "required": ["value", "children"],
                "additionalProperties": False,
            }
        },
        "properties": {"root": {"$ref": "#/$defs/Node"}},
        "required": ["root"],
    },
    {"type": "object", "properties": {"maybe": {"type": ["string", "null"]}, "list": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "boolean"}}}, "required": ["maybe", "list"]},
]


@pytest.mark.parametrize("schema", SCHEMAS)
def test_schema_faker_conforms(schema):
    for seed in range(60):
        value = fake_from_schema(schema, random.Random(seed))
        jsonschema.validate(value, schema, format_checker=jsonschema.FormatChecker())


def test_schema_faker_strict_includes_all_properties():
    schema = {"type": "object", "properties": {k: {"type": "string"} for k in "abcdef"}}
    for seed in range(20):
        assert set(fake_from_schema(schema, random.Random(seed), strict=True)) == set("abcdef")


def test_tokenizer_roundtrip():
    rng = random.Random(0)
    for s in ["", "hello", "  leading", "trailing  ", "a\n\nb", "supercalifragilistic!!", "emoji 🐱 ok", '{"a": [1, 2]}']:
        assert "".join(split_tokens(s)) == s
    for _ in range(200):
        s = "".join(rng.choice("ab  .,\n!🐱{}\"'") for _ in range(rng.randint(0, 40)))
        assert "".join(split_tokens(s)) == s


def test_meower_text_joins_cleanly():
    m = Meower(random.Random(1))
    for _ in range(50):
        toks = m.text(20)
        text = "".join(toks)
        assert len(toks) >= 20
        assert not text.startswith(" ") and "  " not in text


def test_truncate_at_stop():
    toks = ["Meow", " purr", ".", " Mrrp", "!"]
    assert truncate_at_stop(toks, ["."]) == (["Meow", " purr"], True)
    assert truncate_at_stop(toks, ["urr"]) == (["Meow", " p"], True)
    assert truncate_at_stop(toks, ["zzz"]) == (toks, False)
    assert truncate_at_stop(toks, ["Meow"]) == ([], True)


def test_image_tokens_formula():
    assert image_tokens(100, 100, "low") == 85
    assert image_tokens(1024, 1024, "high") == 765  # matches OpenAI docs
    assert image_tokens(2048, 4096, "high") == 1105  # matches OpenAI docs


@pytest.mark.parametrize("size", [(16, 16), (256, 256), (1792, 1024), (1024, 1792), (4096, 16)])
def test_render_cat_dims(size):
    png = render_cat(*size, random.Random(0))
    assert image_size(png) == size


def test_config_validation():
    with pytest.raises(ValueError):
        Config(tool_call_probability=2).validate()
    with pytest.raises(ValueError):
        Config(min_reply_tokens=10, max_reply_tokens=5).validate()


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("DUMMYLLM_SEED", "5")
    monkeypatch.setenv("DUMMYLLM_TOKENS_PER_SECOND", "12.5")
    monkeypatch.setenv("DUMMYLLM_MODELS", "a, b")
    cfg = Config.from_env()
    assert cfg.seed == 5 and cfg.tokens_per_second == 12.5 and cfg.models == ("a", "b")


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "string", "minLength": 10**12},
        {"type": "array", "minItems": 10**9, "items": {"type": "array", "minItems": 10**9, "items": {"type": "array", "minItems": 1000}}},
        {"type": "object", "minProperties": 10**9},
        {"allOf": [{"$ref": "#"}]},
        {"$ref": "#"},
        {"type": "object", "properties": {"a": {"$ref": "#"}, "b": {"$ref": "#"}, "c": {"$ref": "#"}}, "required": ["a", "b", "c"]},
        {"type": "number", "multipleOf": 1e-300, "maximum": 1e300},
        {"type": "integer", "minimum": "x", "maximum": None},
        {"allOf": [{"required": -5}, {"required": ["a"]}]},
    ],
)
def test_schema_faker_survives_pathological_schemas(schema):
    import time

    t = time.perf_counter()
    fake_from_schema(schema, random.Random(0))
    assert time.perf_counter() - t < 1.0
