"""Generate random values that conform to a JSON Schema.

Used for tool-call arguments and structured outputs. Covers the subset of JSON
Schema that OpenAI function / structured-output schemas use in practice:
types (incl. type unions), properties/required, additionalProperties, items,
prefixItems, enum, const, anyOf/oneOf/allOf, $ref with $defs/definitions,
min/max constraints, string formats and nullable fields.
"""

from __future__ import annotations

import math
import random
import string
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from .meow import CAT_NAMES, PLACES, Meower

MAX_DEPTH = 6
# Hard caps so pathological schemas (minItems: 1e9, minLength: 1e12, deeply
# recursive $refs) can't hang or exhaust the server.
MAX_ARRAY_ITEMS = 64
MAX_STRING_LENGTH = 10_000
NODE_BUDGET = 5_000
HARD_DEPTH = 3 * MAX_DEPTH  # absolute recursion limit (self-referencing allOf/$ref chains)
_TYPES = ("string", "number", "integer", "boolean", "object", "array", "null")


class SchemaFaker:
    def __init__(self, rng: random.Random, *, strict: bool = False) -> None:
        self.rng = rng
        self.meow = Meower(rng)
        # In strict mode every declared property is emitted (OpenAI strict
        # schemas require all properties to be listed in "required" anyway).
        self.strict = strict
        self.budget = NODE_BUDGET

    def generate(self, schema: Any) -> Any:
        if schema is None:
            schema = {}
        return self._gen(schema, root=schema if isinstance(schema, dict) else {}, depth=0, name=None)

    # ------------------------------------------------------------------

    def _gen(self, schema: Any, *, root: dict[str, Any], depth: int, name: str | None) -> Any:
        self.budget -= 1
        if self.budget <= 0 or depth > HARD_DEPTH:
            # Pathologically large schema: stop generating (sane schemas never get here).
            return None
        if schema is True or schema == {}:
            return self._any(depth)
        if schema is False or not isinstance(schema, dict):
            return None

        if "$ref" in schema:
            target = self._resolve(schema["$ref"], root)
            if target is None or depth > MAX_DEPTH + 2:
                return None
            merged = {**target, **{k: v for k, v in schema.items() if k != "$ref"}}
            return self._gen(merged, root=root, depth=depth + 1, name=name)

        if "const" in schema:
            return schema["const"]
        if isinstance(schema.get("enum"), list) and schema["enum"]:
            return self.rng.choice(schema["enum"])

        if isinstance(schema.get("allOf"), list) and schema["allOf"]:
            merged = self._merge_all_of(schema, root)
            return self._gen(merged, root=root, depth=depth + 1, name=name)

        for key in ("anyOf", "oneOf"):
            options = schema.get(key)
            if isinstance(options, list) and options:
                base = {k: v for k, v in schema.items() if k != key}
                choice = self._pick_variant(options, root, depth)
                if isinstance(choice, dict):
                    choice = {**base, **choice}
                return self._gen(choice, root=root, depth=depth, name=name)

        stype = schema.get("type")
        if isinstance(stype, list):
            types = [t for t in stype if t in _TYPES] or ["string"]
            non_null = [t for t in types if t != "null"]
            if non_null and ("null" not in types or self.rng.random() < 0.85 or depth >= MAX_DEPTH):
                stype = self.rng.choice(non_null)
            else:
                stype = "null"
        if stype is None:
            stype = self._infer_type(schema)

        if stype == "object":
            return self._object(schema, root, depth)
        if stype == "array":
            return self._array(schema, root, depth, name)
        if stype == "string":
            return self._string(schema, name)
        if stype == "integer":
            return self._integer(schema, name)
        if stype == "number":
            return self._number(schema)
        if stype == "boolean":
            return self.rng.random() < 0.5
        if stype == "null":
            return None
        return self._string(schema, name)

    # -- helpers --------------------------------------------------------

    @staticmethod
    def _resolve(ref: Any, root: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(ref, str):
            return None
        if ref == "#":
            return root
        if not ref.startswith("#/"):
            return None
        node: Any = root
        for part in ref[2:].split("/"):
            part = part.replace("~1", "/").replace("~0", "~")
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return None
        return node if isinstance(node, dict) else None

    def _merge_all_of(self, schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
        merged: dict[str, Any] = {k: v for k, v in schema.items() if k != "allOf"}
        for sub in schema["allOf"]:
            if isinstance(sub, dict) and "$ref" in sub:
                sub = self._resolve(sub["$ref"], root) or {}
            if not isinstance(sub, dict):
                continue
            for k, v in sub.items():
                if k == "properties" and isinstance(v, dict):
                    prev = merged.get("properties")
                    merged["properties"] = {**(prev if isinstance(prev, dict) else {}), **v}
                elif k == "required" and isinstance(v, list):
                    prev = merged.get("required")
                    prev = prev if isinstance(prev, list) else []
                    merged["required"] = list(dict.fromkeys(x for x in [*prev, *v] if isinstance(x, str)))
                else:
                    merged.setdefault(k, v)
        return merged

    def _pick_variant(self, options: list[Any], root: dict[str, Any], depth: int) -> Any:
        if depth >= MAX_DEPTH:
            # Prefer something that terminates: null or a non-ref, non-container schema.
            for opt in options:
                if isinstance(opt, dict) and opt.get("type") == "null":
                    return opt
            for opt in options:
                if isinstance(opt, dict) and "$ref" not in opt and opt.get("type") not in ("object", "array"):
                    return opt
        non_null = [o for o in options if not (isinstance(o, dict) and o.get("type") == "null")]
        if non_null and (len(non_null) == len(options) or self.rng.random() < 0.85):
            return self.rng.choice(non_null)
        return self.rng.choice(options)

    @staticmethod
    def _infer_type(schema: dict[str, Any]) -> str:
        if "properties" in schema or "additionalProperties" in schema or "required" in schema:
            return "object"
        if "items" in schema or "prefixItems" in schema or "minItems" in schema:
            return "array"
        if any(k in schema for k in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf")):
            return "number"
        return "string"

    def _any(self, depth: int) -> Any:
        r = self.rng.random()
        if r < 0.6 or depth >= MAX_DEPTH:
            return self.meow.phrase()
        if r < 0.8:
            return self.rng.randint(0, 9)
        return self.rng.random() < 0.5

    def _object(self, schema: dict[str, Any], root: dict[str, Any], depth: int) -> dict[str, Any]:
        props = schema.get("properties") or {}
        if not isinstance(props, dict):
            props = {}
        req_raw = schema.get("required")
        required = [r for r in req_raw if isinstance(r, str)] if isinstance(req_raw, list) else []
        out: dict[str, Any] = {}
        for key, sub in props.items():
            include = key in required or self.strict or (depth < MAX_DEPTH and self.rng.random() < 0.6)
            if include:
                out[key] = self._gen(sub, root=root, depth=depth + 1, name=key)
        # Required keys not described in properties.
        for key in required:
            if key not in out:
                out[key] = self._any(depth + 1)

        min_props = schema.get("minProperties")
        min_props = min(min_props, MAX_ARRAY_ITEMS) if isinstance(min_props, int) and not isinstance(min_props, bool) else 0
        addl = schema.get("additionalProperties", None if props else True)
        if addl not in (None, False) and depth < MAX_DEPTH:
            extra = self.rng.randint(0 if props else 1, 2)
            extra = max(extra, min_props - len(out))
            sub = addl if isinstance(addl, dict) else {}
            for i in range(extra):
                key = f"{self.meow.sound().lower()}_{i}"
                while key in out:
                    key += "_"
                out[key] = self._gen(sub, root=root, depth=depth + 1, name=key)
        max_props = schema.get("maxProperties")
        if isinstance(max_props, int) and len(out) > max_props:
            keep = [k for k in out if k in required] + [k for k in out if k not in required]
            out = {k: out[k] for k in keep[: max(max_props, len(required))]}
        return out

    def _array(self, schema: dict[str, Any], root: dict[str, Any], depth: int, name: str | None) -> list[Any]:
        prefix = schema.get("prefixItems")
        items = schema.get("items", {})
        if isinstance(items, list):  # draft-04 tuple form
            prefix, items = items, schema.get("additionalItems", False)
        prefix = prefix if isinstance(prefix, list) else []

        lo = schema.get("minItems")
        if not isinstance(lo, int) or isinstance(lo, bool):
            lo = 0 if depth >= MAX_DEPTH else 1
        hi = schema.get("maxItems")
        if not isinstance(hi, int) or isinstance(hi, bool):
            hi = max(lo, 3)
        lo = min(max(lo, 0), MAX_ARRAY_ITEMS)
        hi = min(max(hi, lo), MAX_ARRAY_ITEMS)
        if depth >= MAX_DEPTH:
            hi = lo
        n = self.rng.randint(lo, hi)

        out: list[Any] = []
        for i in range(n):
            if i < len(prefix):
                out.append(self._gen(prefix[i], root=root, depth=depth + 1, name=name))
            elif items is False:
                break
            else:
                out.append(self._gen(items, root=root, depth=depth + 1, name=name))
        if schema.get("uniqueItems"):
            seen: list[Any] = []
            for v in out:
                if v not in seen:
                    seen.append(v)
            out = seen
        return out

    def _string(self, schema: dict[str, Any], name: str | None) -> str:
        fmt = schema.get("format")
        value = self._formatted(fmt) if isinstance(fmt, str) else None
        if value is None:
            value = self._named_string(name)
        lo = schema.get("minLength")
        hi = schema.get("maxLength")
        if isinstance(lo, int) and len(value) < lo:
            lo = min(lo, MAX_STRING_LENGTH)
            value = (value + " meow" * (lo // 5 + 1))[: max(lo, len(value))]
            if isinstance(hi, int):
                value = value[: max(hi, lo)]
        if isinstance(hi, int) and hi >= 0 and len(value) > hi:
            value = value[:hi]
        return value

    def _formatted(self, fmt: str) -> str | None:
        rng = self.rng
        base = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=rng.randrange(0, 3 * 365 * 86400))
        if fmt == "date-time":
            return base.strftime("%Y-%m-%dT%H:%M:%SZ")
        if fmt == "date":
            return base.strftime("%Y-%m-%d")
        if fmt == "time":
            return base.strftime("%H:%M:%S")
        if fmt == "duration":
            return f"PT{rng.randint(1, 9)}H{rng.randint(0, 59)}M"
        if fmt in ("email", "idn-email"):
            return f"{rng.choice(CAT_NAMES).lower()}@meow.cat"
        if fmt in ("uri", "url", "iri", "uri-reference"):
            return f"https://meow.cat/{self.meow.sound().lower()}/{rng.randint(1, 999)}"
        if fmt == "uuid":
            return str(uuid.UUID(int=rng.getrandbits(128), version=4))
        if fmt == "ipv4":
            return ".".join(str(rng.randint(1, 254)) for _ in range(4))
        if fmt == "ipv6":
            return ":".join(f"{rng.randint(0, 0xFFFF):x}" for _ in range(8))
        if fmt in ("hostname", "idn-hostname"):
            return f"{self.meow.sound().lower()}.meow.cat"
        return None

    def _named_string(self, name: str | None) -> str:
        n = (name or "").lower()
        rng = self.rng
        if any(k in n for k in ("city", "location", "place", "address", "country", "town", "destination")):
            return rng.choice(PLACES)
        if n in ("name", "first_name", "firstname", "username", "user", "author", "owner", "pet_name") or n.endswith("_name"):
            return rng.choice(CAT_NAMES)
        if "email" in n:
            return f"{rng.choice(CAT_NAMES).lower()}@meow.cat"
        if "url" in n or "link" in n or "href" in n:
            return f"https://meow.cat/{self.meow.sound().lower()}"
        if n in ("id",) or n.endswith("_id"):
            return "".join(rng.choice(string.ascii_lowercase + string.digits) for _ in range(8))
        if "date" in n:
            return self._formatted("date") or ""
        if any(k in n for k in ("unit", "units")):
            return rng.choice(("meows", "purrs", "whiskers", "paws"))
        if n == "q" or "query" in n or "search" in n:
            return f"{self.meow.phrase(2)} {rng.choice(('tuna', 'catnip', 'laser pointer', 'cardboard box'))}"
        if "lang" in n:
            return "cat"
        return self.meow.phrase()

    def _integer(self, schema: dict[str, Any], name: str | None) -> int:
        lo, hi = self._bounds(schema, integer=True)
        mult = schema.get("multipleOf")
        if isinstance(mult, int) and not isinstance(mult, bool) and mult > 0:
            first = -(-lo // mult) * mult
            if first > hi:
                return first
            return first + mult * self.rng.randint(0, (hi - first) // mult)
        if lo <= 9 <= hi and self.rng.random() < 0.15:
            return 9  # nine lives
        return self.rng.randint(lo, hi)

    def _number(self, schema: dict[str, Any]) -> float:
        lo, hi = self._bounds(schema, integer=False)
        mult = schema.get("multipleOf")
        if isinstance(mult, (int, float)) and not isinstance(mult, bool) and mult > 0:
            try:
                k_lo = int(-(-lo // mult))
                k_hi = int(hi // mult)
                if k_hi >= k_lo:
                    return round(self.rng.randint(k_lo, k_hi) * mult, 10)
            except (OverflowError, ValueError):
                pass  # absurd multipleOf; fall back to a plain number
        value = round(self.rng.uniform(lo, hi), 2)
        # Rounding may push us out of an exclusive bound; clamp back inside.
        if "exclusiveMinimum" in schema and isinstance(schema["exclusiveMinimum"], (int, float)) and value <= schema["exclusiveMinimum"]:
            value = lo
        if "exclusiveMaximum" in schema and isinstance(schema["exclusiveMaximum"], (int, float)) and value >= schema["exclusiveMaximum"]:
            value = hi
        return value

    def _bounds(self, schema: dict[str, Any], *, integer: bool) -> tuple[Any, Any]:
        def num(key: str, allow_bool: bool = False) -> Any:
            v = schema.get(key)
            if isinstance(v, bool):
                return v if allow_bool else None
            if isinstance(v, (int, float)) and math.isfinite(v):
                return v
            return None

        lo = num("minimum")
        hi = num("maximum")
        emin = num("exclusiveMinimum", allow_bool=True)
        emax = num("exclusiveMaximum", allow_bool=True)
        step = 1 if integer else 0.01
        if isinstance(emin, bool):  # draft-04 boolean form
            emin = lo if emin else None
            lo = None if emin is not None else lo
        if isinstance(emax, bool):
            emax = hi if emax else None
            hi = None if emax is not None else hi
        if isinstance(emin, (int, float)):
            lo = emin + step if lo is None else max(lo, emin + step)
        if isinstance(emax, (int, float)):
            hi = emax - step if hi is None else min(hi, emax - step)
        if lo is None and hi is None:
            lo, hi = 0, 100
        elif lo is None:
            lo = hi - 100
        elif hi is None:
            hi = lo + 100
        if integer:
            lo, hi = math.ceil(lo), math.floor(hi)
        if hi < lo:
            hi = lo
        return lo, hi


def fake_from_schema(schema: Any, rng: random.Random, *, strict: bool = False) -> Any:
    return SchemaFaker(rng, strict=strict).generate(schema)
