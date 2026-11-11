from __future__ import annotations

import base64
import io
import struct
import zlib

import httpx
import openai
import pytest

from dummyllm.testing import DummyLLMServer


@pytest.fixture(scope="session")
def server():
    with DummyLLMServer() as s:
        yield s


@pytest.fixture(scope="session")
def seeded_server():
    with DummyLLMServer(seed=1234) as s:
        yield s


@pytest.fixture(scope="session")
def client(server) -> openai.OpenAI:
    return openai.OpenAI(base_url=server.base_url, api_key="sk-meow", max_retries=0)


@pytest.fixture(scope="session")
def seeded_client(seeded_server) -> openai.OpenAI:
    return openai.OpenAI(base_url=seeded_server.base_url, api_key="sk-meow", max_retries=0)


@pytest.fixture(scope="session")
def http(server):
    with httpx.Client(base_url=server.url, timeout=30) as c:
        yield c


def tiny_png(width: int = 4, height: int = 3) -> bytes:
    rows = b"".join(b"\x00" + b"\xff\x00\x00" * width for _ in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def png_data_url(width: int = 4, height: int = 3) -> str:
    return "data:image/png;base64," + base64.b64encode(tiny_png(width, height)).decode()


def png_file(width: int = 4, height: int = 3) -> io.BytesIO:
    f = io.BytesIO(tiny_png(width, height))
    f.name = "cat.png"
    return f


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                "days": {"type": "integer", "minimum": 1, "maximum": 7},
            },
            "required": ["city", "unit"],
            "additionalProperties": False,
        },
    },
}

TIME_TOOL = {
    "type": "function",
    "function": {
        "name": "get_time",
        "parameters": {"type": "object", "properties": {"tz": {"type": "string"}}, "required": ["tz"]},
    },
}
