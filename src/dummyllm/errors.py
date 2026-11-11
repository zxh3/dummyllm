"""OpenAI-shaped errors."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError


class APIError(Exception):
    """An error rendered as ``{"error": {...}}`` with the given HTTP status."""

    def __init__(
        self,
        message: str,
        *,
        status: int = 400,
        type: str = "invalid_request_error",
        param: str | None = None,
        code: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.type = type
        self.param = param
        self.code = code
        self.headers = headers or {}

    def body(self) -> dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": self.type,
                "param": self.param,
                "code": self.code,
            }
        }


def invalid(message: str, param: str | None = None, code: str | None = None) -> APIError:
    return APIError(message, status=400, param=param, code=code)


def not_found(message: str, param: str | None = None) -> APIError:
    return APIError(message, status=404, param=param, code="not_found")


def from_validation_error(exc: ValidationError) -> APIError:
    """Convert the first pydantic error into an OpenAI-style 400."""
    err = exc.errors()[0]
    loc = [p for p in err.get("loc", ()) if not (isinstance(p, str) and p.startswith("function-"))]
    param = _format_loc(loc) if loc else None
    msg = err.get("msg", "Invalid request")
    if msg.startswith("Value error, "):
        msg = msg[len("Value error, ") :]
    etype = err.get("type")
    if etype == "missing":
        message = f"Missing required parameter: '{param}'."
        return APIError(message, param=param, code="missing_required_parameter")
    if etype == "extra_forbidden":
        message = f"Unrecognized request argument supplied: {param}"
        return APIError(message, param=param, code="unknown_parameter")
    if param:
        message = f"Invalid value for '{param}': {msg}"
    else:
        message = msg
    return APIError(message, param=param, code="invalid_value")


def _format_loc(loc: list[Any]) -> str:
    out = ""
    for part in loc:
        if isinstance(part, int):
            out += f"[{part}]"
        else:
            # pydantic adds union-member tags like "str" / "list[...]" to locs; drop them.
            if out and _looks_like_type_tag(str(part)):
                continue
            out += ("." if out else "") + str(part)
    return out


def _looks_like_type_tag(part: str) -> bool:
    return (
        part in ("str", "int", "float", "bool", "dict", "list", "none")
        or part.startswith(("list[", "dict[", "tagged-union", "literal[", "function-"))
        or part[:1].isupper()
    )


# status -> (type, code, message)
INJECTABLE_ERRORS: dict[int, tuple[str, str | None, str]] = {
    400: ("invalid_request_error", None, "Meow? (injected bad request)"),
    401: ("invalid_request_error", "invalid_api_key", "Incorrect API key provided. Hiss! (injected)"),
    403: ("permission_denied", "unsupported_country_region_territory", "The cat refuses. (injected)"),
    404: ("invalid_request_error", "model_not_found", "The model does not exist. (injected)"),
    409: ("invalid_request_error", "conflict", "Another cat is sitting here. (injected)"),
    422: ("invalid_request_error", None, "Unprocessable meow. (injected)"),
    429: ("requests", "rate_limit_exceeded", "Rate limit reached: too many meows. Please try again in 1s. (injected)"),
    500: ("server_error", None, "The cat knocked the server off the table. (injected)"),
    502: ("server_error", None, "Bad gateway: the cat chewed the cable. (injected)"),
    503: ("server_error", "service_unavailable", "The cat is napping. Please retry. (injected)"),
    504: ("server_error", None, "Gateway timeout: the cat got distracted. (injected)"),
}


def injected_error(status: int) -> APIError:
    etype, code, message = INJECTABLE_ERRORS.get(
        status, ("server_error" if status >= 500 else "invalid_request_error", None, f"Injected error {status}.")
    )
    headers = {"retry-after": "1", "retry-after-ms": "1000"} if status in (429, 503) else {}
    return APIError(message, status=status, type=etype, code=code, headers=headers)
