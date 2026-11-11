"""The shared "inference" engine used by every text endpoint.

It decides *what* to say (cat noises, structured JSON, or tool calls) as plain
data; the route modules decide *how* to serialise it (chat completion, legacy
completion, Responses API; streamed or not).
"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass, field
from typing import Any

from .config import Config
from .errors import invalid
from .ids import new_id
from .images import decode_data_url, image_size
from .meow import SOUNDS, Flavor, Meower, truncate_at_stop
from .schema_faker import SchemaFaker
from .tokens import count_json_tokens, count_tokens, image_tokens, split_tokens

FUNCTION_NAME_RE = re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")
DRAW_HINT_RE = re.compile(r"\b(draw|drawing|image|picture|paint|painting|illustrat\w*|sketch|photo|render)\b", re.I)


# ---------------------------------------------------------------------------
# Data model


@dataclass
class ToolSpec:
    """A tool the model may call. ``kind`` is function | custom | image_generation."""

    kind: str
    name: str
    parameters: dict[str, Any] | None = None
    strict: bool = False
    options: dict[str, Any] = field(default_factory=dict)  # image_generation settings, custom format...


@dataclass
class ToolCall:
    id: str
    kind: str
    name: str
    arguments: str  # JSON for function tools; free text for custom tools; prompt for image gen


@dataclass
class Prompt:
    """What the engine needs to know about the input conversation."""

    tokens: int = 0
    saw_image: bool = False
    after_tool_result: bool = False
    last_user_text: str = ""


@dataclass
class Generation:
    tokens: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"  # stop | length | tool_calls
    reasoning: list[str] = field(default_factory=list)  # thinking tokens, emitted before the answer

    @property
    def reasoning_tokens(self) -> int:
        return len(self.reasoning)

    @property
    def reasoning_text(self) -> str:
        return "".join(self.reasoning)

    @property
    def text(self) -> str:
        return "".join(self.tokens)

    @property
    def completion_tokens(self) -> int:
        return len(self.tokens) + self.reasoning_tokens + sum(tool_call_cost(c) for c in self.tool_calls)


# Tokens a tool call costs beyond its arguments: its name plus framing.
TOOL_CALL_OVERHEAD = 3


def tool_call_cost(call: "ToolCall") -> int:
    return count_tokens(call.name) + count_tokens(call.arguments) + TOOL_CALL_OVERHEAD


def fit_tool_calls(calls: list["ToolCall"], budget: int | None) -> tuple[list["ToolCall"], bool]:
    """Fit tool calls into a token budget the way a model running out of tokens does.

    Calls are emitted in order; the one being written when the budget runs out
    gets truncated arguments and later ones are dropped. Usage computed with
    ``tool_call_cost`` never exceeds the budget. Returns (calls, truncated).
    """
    if budget is None:
        return calls, False
    kept: list[ToolCall] = []
    used = 0
    for call in calls:
        fixed = count_tokens(call.name) + TOOL_CALL_OVERHEAD
        args = split_tokens(call.arguments)
        if used + fixed + len(args) <= budget:
            kept.append(call)
            used += fixed + len(args)
            continue
        room = budget - used - fixed
        if room >= 0:  # the name fits: emit the call with partial arguments
            call.arguments = "".join(args[:room])
            kept.append(call)
        return kept, True
    return kept, False


# ---------------------------------------------------------------------------
# Common parameter validation


def check_range(value: Any, name: str, lo: float, hi: float) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or math.isnan(value):
        raise invalid(f"Invalid type for '{name}': expected a number, but got {type(value).__name__} instead.", name, "invalid_type")
    if not lo <= value <= hi:
        if value < lo:
            msg = f"Invalid '{name}': decimal below minimum value. Expected a value >= {lo}, but got {value} instead."
            code = "decimal_below_min_value"
        else:
            msg = f"Invalid '{name}': decimal above maximum value. Expected a value <= {hi}, but got {value} instead."
            code = "decimal_above_max_value"
        raise invalid(msg, name, code)


def check_int(value: Any, name: str, lo: int, hi: int | None = None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise invalid(f"Invalid type for '{name}': expected an integer, but got {type(value).__name__} instead.", name, "invalid_type")
    if value < lo:
        raise invalid(f"Invalid '{name}': integer below minimum value. Expected a value >= {lo}, but got {value} instead.", name, "integer_below_min_value")
    if hi is not None and value > hi:
        raise invalid(f"Invalid '{name}': integer above maximum value. Expected a value <= {hi}, but got {value} instead.", name, "integer_above_max_value")


def normalize_stop(stop: Any) -> list[str]:
    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop]
    if isinstance(stop, list) and all(isinstance(s, str) for s in stop):
        if len(stop) > 4:
            raise invalid(f"Invalid 'stop': array too long. Expected an array with maximum length 4, but got an array with length {len(stop)} instead.", "stop", "array_above_max_length")
        return stop
    raise invalid("Invalid type for 'stop': expected a string or an array of strings.", "stop", "invalid_type")


def check_sampling(body: dict[str, Any]) -> None:
    check_range(body.get("temperature"), "temperature", 0, 2)
    check_range(body.get("top_p"), "top_p", 0, 1)
    check_range(body.get("presence_penalty"), "presence_penalty", -2, 2)
    check_range(body.get("frequency_penalty"), "frequency_penalty", -2, 2)
    seed = body.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise invalid("Invalid type for 'seed': expected an integer.", "seed", "invalid_type")
    lb = body.get("logit_bias")
    if lb is not None:
        if not isinstance(lb, dict):
            raise invalid("Invalid type for 'logit_bias': expected an object.", "logit_bias", "invalid_type")
        for k, v in lb.items():
            if not str(k).lstrip("-").isdigit():
                raise invalid(f"Invalid key in 'logit_bias': {k}. Expected a token id.", "logit_bias", "invalid_value")
            check_range(v, f"logit_bias.{k}", -100, 100)


def require_model(body: dict[str, Any]) -> str:
    model = body.get("model")
    if model is None:
        raise invalid("you must provide a model parameter", "model", "missing_required_parameter")
    if not isinstance(model, str) or not model.strip():
        raise invalid("Invalid type for 'model': expected a non-empty string.", "model", "invalid_type")
    return model


# ---------------------------------------------------------------------------
# Tools


def _check_function_name(name: Any, param: str) -> str:
    if not isinstance(name, str):
        raise invalid(f"Missing required parameter: '{param}'.", param, "missing_required_parameter")
    if not FUNCTION_NAME_RE.match(name):
        raise invalid(
            f"Invalid '{param}': string does not match pattern. Expected a string that matches the pattern '^[a-zA-Z0-9_-]+$'.",
            param,
            "invalid_value",
        )
    return name


def _check_parameters(params: Any, param: str) -> dict[str, Any] | None:
    if params is None:
        return None
    if not isinstance(params, dict):
        raise invalid(f"Invalid type for '{param}': expected an object.", param, "invalid_type")
    t = params.get("type")
    if t is not None and t != "object":
        raise invalid(
            f"Invalid schema for function: schema must be a JSON Schema of 'type: \"object\"', got 'type: \"{t}\"'.",
            param,
            "invalid_function_parameters",
        )
    return params


def parse_chat_tools(tools: Any, functions: Any) -> list[ToolSpec]:
    """Chat Completions tools: ``{"type": "function", "function": {...}}`` or custom tools."""
    specs: list[ToolSpec] = []
    if tools is not None:
        if not isinstance(tools, list):
            raise invalid("Invalid type for 'tools': expected an array.", "tools", "invalid_type")
        if len(tools) > 128:
            raise invalid("Invalid 'tools': array too long. Expected an array with maximum length 128.", "tools", "array_above_max_length")
        for i, tool in enumerate(tools):
            p = f"tools[{i}]"
            if not isinstance(tool, dict):
                raise invalid(f"Invalid type for '{p}': expected an object.", p, "invalid_type")
            ttype = tool.get("type", "function")
            if ttype == "function":
                fn = tool.get("function")
                if not isinstance(fn, dict):
                    raise invalid(f"Missing required parameter: '{p}.function'.", f"{p}.function", "missing_required_parameter")
                name = _check_function_name(fn.get("name"), f"{p}.function.name")
                params = _check_parameters(fn.get("parameters"), f"{p}.function.parameters")
                specs.append(ToolSpec("function", name, params, bool(fn.get("strict"))))
            elif ttype == "custom":
                c = tool.get("custom")
                if not isinstance(c, dict):
                    raise invalid(f"Missing required parameter: '{p}.custom'.", f"{p}.custom", "missing_required_parameter")
                name = _check_function_name(c.get("name"), f"{p}.custom.name")
                specs.append(ToolSpec("custom", name, options={"format": c.get("format")}))
            else:
                raise invalid(
                    f"Invalid value: '{ttype}'. Supported values are: 'function' and 'custom'.", f"{p}.type", "invalid_value"
                )
    if functions is not None:
        if tools is not None:
            raise invalid("Cannot specify both 'functions' and 'tools'.", "functions")
        if not isinstance(functions, list):
            raise invalid("Invalid type for 'functions': expected an array.", "functions", "invalid_type")
        for i, fn in enumerate(functions):
            p = f"functions[{i}]"
            if not isinstance(fn, dict):
                raise invalid(f"Invalid type for '{p}': expected an object.", p, "invalid_type")
            name = _check_function_name(fn.get("name"), f"{p}.name")
            params = _check_parameters(fn.get("parameters"), f"{p}.parameters")
            specs.append(ToolSpec("function", name, params))
    names = [s.name for s in specs]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise invalid(f"Invalid 'tools': duplicate function names: {sorted(dupes)}.", "tools", "duplicate_function_names")
    return specs


def parse_responses_tools(tools: Any) -> list[ToolSpec]:
    """Responses API tools: flat ``{"type": "function", "name": ...}`` plus built-ins."""
    if tools is None:
        return []
    if not isinstance(tools, list):
        raise invalid("Invalid type for 'tools': expected an array.", "tools", "invalid_type")
    specs: list[ToolSpec] = []
    for i, tool in enumerate(tools):
        p = f"tools[{i}]"
        if not isinstance(tool, dict) or not isinstance(tool.get("type"), str):
            raise invalid(f"Missing required parameter: '{p}.type'.", f"{p}.type", "missing_required_parameter")
        ttype = tool["type"]
        if ttype == "function":
            name = _check_function_name(tool.get("name"), f"{p}.name")
            params = _check_parameters(tool.get("parameters"), f"{p}.parameters")
            strict = tool.get("strict")
            specs.append(ToolSpec("function", name, params, True if strict is None else bool(strict)))
        elif ttype == "custom":
            name = _check_function_name(tool.get("name"), f"{p}.name")
            specs.append(ToolSpec("custom", name, options={"format": tool.get("format")}))
        elif ttype == "image_generation":
            specs.append(ToolSpec("image_generation", "image_generation", options=dict(tool)))
        # Other hosted tools (web_search, file_search, code_interpreter, mcp,
        # computer_use...) are accepted but the cat never uses them.
    names = [s.name for s in specs if s.kind != "image_generation"]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise invalid(f"Invalid 'tools': duplicate function names: {sorted(dupes)}.", "tools", "duplicate_function_names")
    return specs


@dataclass
class ToolPolicy:
    mode: str  # none | auto | required
    allowed: list[ToolSpec]
    forced: ToolSpec | None = None


def resolve_tool_choice(choice: Any, specs: list[ToolSpec], *, api: str, legacy_function_call: bool = False) -> ToolPolicy:
    """Interpret ``tool_choice`` (or legacy ``function_call``) for either API."""
    param = "function_call" if legacy_function_call else "tool_choice"
    if choice is None:
        return ToolPolicy("auto" if specs else "none", specs)
    if isinstance(choice, str):
        if choice not in ("none", "auto", "required") or (legacy_function_call and choice == "required"):
            raise invalid(f"Invalid value: '{choice}'. Supported values are: 'none', 'auto', and 'required'.", param, "invalid_value")
        if choice != "none" and not specs:
            raise invalid(
                f"Invalid value for '{param}': '{param}' is only allowed when 'tools' are specified.", param
            )
        return ToolPolicy(choice, specs)
    if not isinstance(choice, dict):
        raise invalid(f"Invalid type for '{param}': expected a string or an object.", param, "invalid_type")

    if legacy_function_call:
        name = choice.get("name")
        return ToolPolicy("required", specs, _find_tool(specs, "function", name, param))

    ctype = choice.get("type")
    if ctype == "function":
        name = choice.get("function", {}).get("name") if api == "chat" else choice.get("name")
        return ToolPolicy("required", specs, _find_tool(specs, "function", name, param))
    if ctype == "custom":
        name = choice.get("custom", {}).get("name") if api == "chat" else choice.get("name")
        return ToolPolicy("required", specs, _find_tool(specs, "custom", name, param))
    if ctype == "image_generation" and api == "responses":
        return ToolPolicy("required", specs, _find_tool(specs, "image_generation", "image_generation", param))
    if ctype == "allowed_tools":
        spec = choice.get("allowed_tools", choice) if api == "chat" else choice
        mode = spec.get("mode", "auto")
        if mode not in ("auto", "required"):
            raise invalid(f"Invalid value: '{mode}'. Supported values are: 'auto' and 'required'.", f"{param}.mode", "invalid_value")
        allowed: list[ToolSpec] = []
        for t in spec.get("tools") or []:
            if not isinstance(t, dict):
                continue
            tt = t.get("type", "function")
            if tt == "function":
                nm = t.get("function", {}).get("name") if api == "chat" and "function" in t else t.get("name")
            elif tt == "custom":
                nm = t.get("custom", {}).get("name") if api == "chat" and "custom" in t else t.get("name")
            else:
                nm = tt
            match = next((s for s in specs if s.name == nm and (s.kind == tt or tt not in ("function", "custom"))), None)
            if match is not None:
                allowed.append(match)
        return ToolPolicy(mode if allowed else "none", allowed)
    if ctype in ("web_search_preview", "web_search", "file_search", "code_interpreter", "computer_use_preview", "mcp"):
        # Hosted tools the cat doesn't use: degrade to a plain answer.
        return ToolPolicy("none", [])
    raise invalid(f"Invalid value for '{param}.type': '{ctype}'.", f"{param}.type", "invalid_value")


def _find_tool(specs: list[ToolSpec], kind: str, name: Any, param: str) -> ToolSpec:
    for s in specs:
        if s.kind == kind and s.name == name:
            return s
    raise invalid(
        f"Invalid value for '{param}': tool '{name}' was not found in 'tools'." if name else f"Missing tool name in '{param}'.",
        param,
        "invalid_value",
    )


def decide_tool_calls(
    rng: random.Random,
    cfg: Config,
    policy: ToolPolicy,
    prompt: Prompt,
    *,
    parallel: bool,
    probability_override: float | None = None,
) -> list[ToolSpec]:
    """Return the tools to call this turn (possibly empty)."""
    if policy.mode == "none" or not policy.allowed:
        return []
    if policy.forced is not None:
        return [policy.forced]
    if policy.mode == "auto":
        if probability_override is not None:
            p = probability_override
        elif prompt.after_tool_result:
            p = cfg.tool_call_probability_after_tool_result
        else:
            p = cfg.tool_call_probability
        # A cat asked to draw, with a canvas available, will usually draw.
        if any(s.kind == "image_generation" for s in policy.allowed) and DRAW_HINT_RE.search(prompt.last_user_text):
            if not prompt.after_tool_result and probability_override is None:
                return [next(s for s in policy.allowed if s.kind == "image_generation")]
        if rng.random() >= p:
            return []
    # "required" or auto-and-decided-to-call: pick 1..3 distinct tools.
    count = 1
    if parallel and len(policy.allowed) > 1:
        r = rng.random()
        count = 1 if r < 0.6 else (2 if r < 0.9 else 3)
        count = min(count, len(policy.allowed))
    chosen = rng.sample(policy.allowed, count)
    # Image generation is hosted; keep at most one, and keep function calls ordered first.
    chosen.sort(key=lambda s: s.kind == "image_generation")
    return chosen


def make_tool_call(rng: random.Random, spec: ToolSpec, prompt: Prompt, *, id_prefix: str = "call_") -> ToolCall:
    if spec.kind == "function":
        faker = SchemaFaker(rng, strict=spec.strict)
        args = faker.generate(spec.parameters or {"type": "object", "properties": {}})
        if not isinstance(args, dict):
            args = {}
        arguments = json.dumps(args, ensure_ascii=False)
    elif spec.kind == "custom":
        arguments = Meower(rng).phrase(6)
    else:  # image_generation: the "argument" is the revised prompt
        subject = prompt.last_user_text.strip()[:200] or "a cat"
        arguments = f"A cute cartoon cat face, inspired by: {subject}"
    return ToolCall(id=new_id(id_prefix), kind=spec.kind, name=spec.name, arguments=arguments)


# ---------------------------------------------------------------------------
# Text / structured output generation


def response_format_kind(fmt: Any, *, param: str) -> tuple[str, dict[str, Any] | None, bool]:
    """Return (kind, schema, strict) where kind is text | json_object | json_schema."""
    if fmt is None:
        return "text", None, False
    if not isinstance(fmt, dict):
        raise invalid(f"Invalid type for '{param}': expected an object.", param, "invalid_type")
    ftype = fmt.get("type", "text")
    if ftype == "text":
        return "text", None, False
    if ftype == "json_object":
        return "json_object", None, False
    if ftype == "json_schema":
        # Chat: {"type":"json_schema","json_schema":{"name","schema","strict"}}
        # Responses: {"type":"json_schema","name","schema","strict"}
        spec = fmt.get("json_schema", fmt)
        if not isinstance(spec, dict):
            raise invalid(f"Missing required parameter: '{param}.json_schema'.", f"{param}.json_schema", "missing_required_parameter")
        name = spec.get("name")
        if not isinstance(name, str) or not FUNCTION_NAME_RE.match(name):
            sub = f"{param}.json_schema.name" if "json_schema" in fmt else f"{param}.name"
            raise invalid(f"Missing required parameter: '{sub}'." if name is None else f"Invalid '{sub}'.", sub, "missing_required_parameter")
        schema = spec.get("schema")
        if schema is not None and not isinstance(schema, dict):
            raise invalid(f"Invalid type for '{param}.schema': expected an object.", f"{param}.schema", "invalid_type")
        return "json_schema", schema or {"type": "object"}, bool(spec.get("strict"))
    if ftype == "grammar" or ftype == "python":
        return "text", None, False
    raise invalid(
        f"Invalid value: '{ftype}'. Supported values are: 'text', 'json_object', and 'json_schema'.", f"{param}.type", "invalid_value"
    )


def generate_text(
    rng: random.Random,
    cfg: Config,
    *,
    flavor: Flavor,
    max_tokens: int | None,
    stops: list[str] | None = None,
    fmt_kind: str = "text",
    schema: dict[str, Any] | None = None,
    strict: bool = False,
) -> tuple[list[str], str]:
    """Return (tokens, finish_reason) for a plain-text / JSON reply."""
    meow = Meower(rng)
    if fmt_kind == "json_schema":
        value = SchemaFaker(rng, strict=strict).generate(schema)
        tokens = split_tokens(json.dumps(value, ensure_ascii=False))
    elif fmt_kind == "json_object":
        obj: dict[str, Any] = {}
        for i in range(rng.randint(1, 3)):
            obj[meow.sound().lower() if i == 0 else f"{meow.sound().lower()}_{i}"] = (
                meow.phrase() if rng.random() < 0.7 else rng.randint(1, 9)
            )
        tokens = split_tokens(json.dumps(obj, ensure_ascii=False))
    else:
        target = meow.target_length(cfg.min_reply_tokens, cfg.max_reply_tokens)
        tokens = meow.text(target, flavor)

    finish = "stop"
    if stops and fmt_kind == "text":
        tokens, _hit = truncate_at_stop(tokens, stops)
    if max_tokens is not None and len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        finish = "length"
    return tokens, finish


REASONING_LENGTH = {"minimal": 0, "low": 16, "medium": 48, "high": 128, "xhigh": 256, "max": 384}


def resolve_effort(model: str, effort: str | None) -> str | None:
    """Explicit effort wins; reasoning-flavoured model names think by default."""
    if effort is not None:
        return effort
    return "medium" if "reason" in model.lower() else None


def generate_reasoning(rng: random.Random, effort: str | None) -> list[str]:
    """Thinking tokens for the given effort (empty for none/minimal)."""
    base = REASONING_LENGTH.get(effort or "none", 0)
    if not base:
        return []
    return Meower(rng).thoughts(base + rng.randint(0, base // 2))


# ---------------------------------------------------------------------------
# Logprobs


def fake_logprobs(rng: random.Random, tokens: list[str], top_n: int) -> list[dict[str, Any]]:
    out = []
    for tok in tokens:
        lp = -abs(rng.gauss(0, 0.6))
        entry: dict[str, Any] = {"token": tok, "logprob": round(lp, 6), "bytes": list(tok.encode("utf-8"))}
        alts: list[dict[str, Any]] = []
        if top_n > 0:
            alts.append({"token": tok, "logprob": entry["logprob"], "bytes": entry["bytes"]})
            seen = {tok}
            cur = lp
            lead = " " if tok.startswith(" ") else ""
            while len(alts) < top_n:
                cand = lead + rng.choice(SOUNDS)
                if cand in seen:
                    cand = cand + rng.choice("~!.")
                    if cand in seen:
                        continue
                seen.add(cand)
                cur -= abs(rng.gauss(1.5, 1.0)) + 0.01
                alts.append({"token": cand, "logprob": round(cur, 6), "bytes": list(cand.encode("utf-8"))})
        entry["top_logprobs"] = alts
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# Chat messages -> Prompt


CHAT_ROLES = ("system", "developer", "user", "assistant", "tool", "function")


def analyze_chat_messages(messages: Any) -> Prompt:
    """Validate chat messages (mirroring OpenAI's rules) and count prompt tokens."""
    if not isinstance(messages, list):
        if messages is None:
            raise invalid("Missing required parameter: 'messages'.", "messages", "missing_required_parameter")
        raise invalid("Invalid type for 'messages': expected an array.", "messages", "invalid_type")
    if not messages:
        raise invalid("Invalid 'messages': empty array. Expected an array with minimum length 1, but got an empty array instead.", "messages", "empty_array")

    prompt = Prompt(tokens=3)
    pending_calls: dict[str, int] = {}  # tool_call_id -> index of assistant msg
    pending_from: int | None = None
    for i, msg in enumerate(messages):
        p = f"messages[{i}]"
        if not isinstance(msg, dict):
            raise invalid(f"Invalid type for '{p}': expected an object.", p, "invalid_type")
        role = msg.get("role")
        if role is None:
            raise invalid(f"Missing required parameter: '{p}.role'.", f"{p}.role", "missing_required_parameter")
        if role not in CHAT_ROLES:
            raise invalid(
                f"Invalid value: '{role}'. Supported values are: 'system', 'assistant', 'user', 'function', 'tool', and 'developer'.",
                f"{p}.role",
                "invalid_value",
            )

        if pending_calls and role != "tool":
            missing = ", ".join(pending_calls)
            raise invalid(
                "An assistant message with 'tool_calls' must be followed by tool messages responding to each 'tool_call_id'. "
                f"The following tool_call_ids did not have response messages: {missing}",
                f"messages.[{pending_from}].role",
            )

        prompt.tokens += 4
        if isinstance(msg.get("name"), str):
            prompt.tokens += count_tokens(msg["name"])
        content = msg.get("content")

        if role == "tool":
            tcid = msg.get("tool_call_id")
            if not isinstance(tcid, str):
                raise invalid(f"Missing required parameter: '{p}.tool_call_id'.", f"{p}.tool_call_id", "missing_required_parameter")
            if tcid not in pending_calls:
                raise invalid(
                    "Invalid parameter: messages with role 'tool' must be a response to a preceeding message with 'tool_calls'.",
                    f"messages.[{i}].role",
                )
            del pending_calls[tcid]
        if role in ("user", "system", "developer", "tool", "function") and content is None:
            raise invalid(f"Missing required parameter: '{p}.content'.", f"{p}.content", "missing_required_parameter")

        text, images = _chat_content(content, f"{p}.content", role)
        prompt.tokens += count_tokens(text)
        for w, h, detail in images:
            prompt.tokens += image_tokens(w, h, detail)
        if role == "user":
            prompt.last_user_text = text
            prompt.saw_image = bool(images)

        if role == "assistant":
            tool_calls = msg.get("tool_calls")
            if tool_calls is not None:
                if not isinstance(tool_calls, list):
                    raise invalid(f"Invalid type for '{p}.tool_calls': expected an array.", f"{p}.tool_calls", "invalid_type")
                for j, tc in enumerate(tool_calls):
                    tp = f"{p}.tool_calls[{j}]"
                    if not isinstance(tc, dict) or not isinstance(tc.get("id"), str):
                        raise invalid(f"Missing required parameter: '{tp}.id'.", f"{tp}.id", "missing_required_parameter")
                    fn = tc.get("function") or tc.get("custom") or {}
                    prompt.tokens += count_json_tokens(fn) + 3
                    pending_calls[tc["id"]] = i
                if tool_calls:
                    pending_from = i
            if content is None and not tool_calls and msg.get("function_call") is None and msg.get("audio") is None:
                raise invalid(
                    f"Invalid value for '{p}.content': expected a string or array, got null (content is required unless 'tool_calls' or 'function_call' is specified).",
                    f"{p}.content",
                )
            if msg.get("function_call") is not None:
                prompt.tokens += count_json_tokens(msg["function_call"])

    if pending_calls:
        missing = ", ".join(pending_calls)
        raise invalid(
            "An assistant message with 'tool_calls' must be followed by tool messages responding to each 'tool_call_id'. "
            f"The following tool_call_ids did not have response messages: {missing}",
            f"messages.[{pending_from}].role",
        )

    last_role = messages[-1].get("role")
    prompt.after_tool_result = last_role in ("tool", "function")
    return prompt


def _chat_content(content: Any, p: str, role: str) -> tuple[str, list[tuple[int | None, int | None, str | None]]]:
    """Return (concatenated text, [(w, h, detail) for each image])."""
    if content is None:
        return "", []
    if isinstance(content, str):
        return content, []
    if not isinstance(content, list):
        raise invalid(f"Invalid type for '{p}': expected one of a string or array of objects, but got {type(content).__name__} instead.", p, "invalid_type")
    texts: list[str] = []
    images: list[tuple[int | None, int | None, str | None]] = []
    for k, part in enumerate(content):
        pp = f"{p}[{k}]"
        if not isinstance(part, dict):
            raise invalid(f"Invalid type for '{pp}': expected an object.", pp, "invalid_type")
        ptype = part.get("type")
        if ptype == "text":
            if not isinstance(part.get("text"), str):
                raise invalid(f"Missing required parameter: '{pp}.text'.", f"{pp}.text", "missing_required_parameter")
            texts.append(part["text"])
        elif ptype == "image_url":
            if role not in ("user",):
                raise invalid(f"Invalid '{pp}.type': image content is only supported for messages with role 'user'.", f"{pp}.type")
            iu = part.get("image_url")
            url = iu.get("url") if isinstance(iu, dict) else iu if isinstance(iu, str) else None
            if not isinstance(url, str) or not url:
                raise invalid(f"Missing required parameter: '{pp}.image_url.url'.", f"{pp}.image_url.url", "missing_required_parameter")
            detail = iu.get("detail") if isinstance(iu, dict) else None
            if detail not in (None, "auto", "low", "high"):
                raise invalid(f"Invalid value: '{detail}'. Supported values are: 'auto', 'low', and 'high'.", f"{pp}.image_url.detail", "invalid_value")
            images.append(_image_dims(url, detail, f"{pp}.image_url.url"))
        elif ptype == "input_audio":
            ia = part.get("input_audio") or {}
            data = ia.get("data") if isinstance(ia, dict) else None
            texts.append("x" * min(4000, len(data or "") // 40))  # ~ token-ish cost of audio
        elif ptype == "file":
            texts.append("[file]")
        elif ptype == "refusal":
            texts.append(str(part.get("refusal", "")))
        else:
            raise invalid(
                f"Invalid value: '{ptype}'. Supported values are: 'text', 'image_url', 'input_audio', 'refusal', and 'file'.",
                f"{pp}.type",
                "invalid_value",
            )
    return "\n".join(texts), images


def _image_dims(url: str, detail: str | None, param: str) -> tuple[int | None, int | None, str | None]:
    if url.startswith("data:"):
        if not re.match(r"^data:image/[a-zA-Z0-9.+-]+(;[^,]*)?,", url):
            raise invalid(
                "Invalid 'image_url'. Expected a base64-encoded data URL with an image MIME type (e.g. 'data:image/png;base64,...').",
                param,
                "invalid_image_url",
            )
        raw = decode_data_url(url)
        size = image_size(raw) if raw else None
        if size is None:
            raise invalid("You uploaded an unsupported image. Please make sure your image is valid.", param, "invalid_image_format")
        return size[0], size[1], detail
    if not re.match(r"^https?://", url):
        raise invalid("Invalid 'image_url': expected a URL or a base64 data URL.", param, "invalid_image_url")
    # Remote URLs are never fetched (the cat is offline); assume 1024x1024.
    return None, None, detail


# ---------------------------------------------------------------------------
# Responses API input -> Prompt


RESPONSES_ROLES = ("user", "assistant", "system", "developer")


def analyze_responses_input(
    input_value: Any, instructions: Any, prior_items: list[dict[str, Any]] | None = None
) -> tuple[Prompt, list[dict[str, Any]]]:
    """Validate Responses ``input`` and return (Prompt, normalized input items)."""
    prior_items = prior_items or []
    prompt = Prompt(tokens=3)
    if isinstance(instructions, str):
        prompt.tokens += count_tokens(instructions) + 4
    elif instructions is not None:
        raise invalid("Invalid type for 'instructions': expected a string.", "instructions", "invalid_type")

    if input_value is None:
        items: list[dict[str, Any]] = []
    elif isinstance(input_value, str):
        items = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": input_value}]}]
    elif isinstance(input_value, list):
        items = []
        for i, item in enumerate(input_value):
            p = f"input[{i}]"
            if not isinstance(item, dict):
                raise invalid(f"Invalid type for '{p}': expected an object.", p, "invalid_type")
            items.append(_normalize_responses_item(item, p))
    else:
        raise invalid("Invalid type for 'input': expected a string or an array of input items.", "input", "invalid_type")

    if not items and not prior_items:
        raise invalid("Missing required parameter: 'input'.", "input", "missing_required_parameter")

    known_calls = {
        it.get("call_id")
        for it in (*prior_items, *items)
        if it.get("type") in ("function_call", "custom_tool_call")
    }
    for idx, item in enumerate(prior_items + items):
        t = item.get("type")
        if t == "message":
            prompt.tokens += 4
            text, imgs = _responses_content(item.get("content"), "input")
            prompt.tokens += count_tokens(text)
            for w, h, d in imgs:
                prompt.tokens += image_tokens(w, h, d)
            if item.get("role") == "user":
                prompt.last_user_text = text
                prompt.saw_image = bool(imgs)
        elif t in ("function_call_output", "custom_tool_call_output"):
            if item.get("call_id") not in known_calls:
                raise invalid(
                    f"No tool call found for function call output with call_id {item.get('call_id')}.",
                    "input",
                )
            out = item.get("output")
            prompt.tokens += count_json_tokens(out) + 4
        elif t in ("function_call", "custom_tool_call"):
            prompt.tokens += count_tokens(str(item.get("name", ""))) + count_tokens(str(item.get("arguments", item.get("input", "")))) + 4
        elif t == "reasoning":
            prompt.tokens += 4
        else:
            prompt.tokens += count_json_tokens(item) // 4 + 1

    seq = prior_items + items
    if seq:
        last = seq[-1].get("type")
        prompt.after_tool_result = last in ("function_call_output", "custom_tool_call_output")
    return prompt, items


def _normalize_responses_item(item: dict[str, Any], p: str) -> dict[str, Any]:
    t = item.get("type")
    if t is None and "role" in item:
        t = "message"
    if t == "message":
        role = item.get("role")
        if role not in RESPONSES_ROLES:
            raise invalid(
                f"Invalid value: '{role}'. Supported values are: 'assistant', 'system', 'developer', and 'user'.",
                f"{p}.role",
                "invalid_value",
            )
        content = item.get("content")
        if isinstance(content, str):
            ctype = "output_text" if role == "assistant" else "input_text"
            content = [{"type": ctype, "text": content}]
        elif not isinstance(content, list):
            raise invalid(f"Missing required parameter: '{p}.content'.", f"{p}.content", "missing_required_parameter")
        _responses_content(content, f"{p}.content")  # validate
        out = {"type": "message", "role": role, "content": content}
        if "id" in item:
            out["id"] = item["id"]
        if "status" in item:
            out["status"] = item["status"]
        return out
    if t in ("function_call_output", "custom_tool_call_output"):
        if not isinstance(item.get("call_id"), str):
            raise invalid(f"Missing required parameter: '{p}.call_id'.", f"{p}.call_id", "missing_required_parameter")
        if "output" not in item:
            raise invalid(f"Missing required parameter: '{p}.output'.", f"{p}.output", "missing_required_parameter")
        return dict(item)
    if t in ("function_call", "custom_tool_call"):
        for k in ("call_id", "name"):
            if not isinstance(item.get(k), str):
                raise invalid(f"Missing required parameter: '{p}.{k}'.", f"{p}.{k}", "missing_required_parameter")
        return dict(item)
    if isinstance(t, str):
        return dict(item)  # reasoning, image_generation_call, item_reference, ... passed through
    raise invalid(f"Missing required parameter: '{p}.type'.", f"{p}.type", "missing_required_parameter")


def _responses_content(content: Any, p: str) -> tuple[str, list[tuple[int | None, int | None, str | None]]]:
    if isinstance(content, str):
        return content, []
    texts: list[str] = []
    images: list[tuple[int | None, int | None, str | None]] = []
    for k, part in enumerate(content or []):
        pp = f"{p}[{k}]"
        if not isinstance(part, dict):
            raise invalid(f"Invalid type for '{pp}': expected an object.", pp, "invalid_type")
        ptype = part.get("type")
        if ptype in ("input_text", "output_text", "text", "summary_text", "reasoning_text"):
            texts.append(str(part.get("text", "")))
        elif ptype == "refusal":
            texts.append(str(part.get("refusal", "")))
        elif ptype == "input_image":
            detail = part.get("detail", "auto")
            url = part.get("image_url")
            if isinstance(url, str) and url:
                images.append(_image_dims(url, detail, f"{pp}.image_url"))
            elif isinstance(part.get("file_id"), str):
                images.append((None, None, detail))
            else:
                raise invalid(f"Missing required parameter: '{pp}.image_url'.", f"{pp}.image_url", "missing_required_parameter")
        elif ptype in ("input_file", "input_audio"):
            texts.append("[file]")
        else:
            raise invalid(
                f"Invalid value: '{ptype}'. Supported values are: 'input_text', 'input_image', 'input_file', and 'output_text'.",
                f"{pp}.type",
                "invalid_value",
            )
    return "\n".join(texts), images
