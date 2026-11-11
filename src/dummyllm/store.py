"""Storage for Responses API objects (``store=true``).

Two backends with the same async interface:

* ``MemoryStore`` (default): an in-process LRU. Fine for one server process.
* ``ModalDictStore``: a shared ``modal.Dict``, so several containers on Modal
  see the same responses (``previous_response_id`` keeps working when the
  deployment scales out). Selected with ``response_store="modal-dict:<name>"``.
"""

from __future__ import annotations

import dataclasses
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class StoredResponse:
    response: dict[str, Any]
    input_items: list[dict[str, Any]]
    # Full conversation (prior turns + this input + this output), used to
    # resolve ``previous_response_id`` chains.
    conversation: list[dict[str, Any]] = field(default_factory=list)


class ResponseStore(Protocol):
    async def put(self, rid: str, value: StoredResponse) -> None: ...
    async def get(self, rid: str) -> StoredResponse | None: ...
    async def delete(self, rid: str) -> bool: ...


class MemoryStore:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._items: OrderedDict[str, StoredResponse] = OrderedDict()
        self._lock = threading.Lock()

    async def put(self, rid: str, value: StoredResponse) -> None:
        if self.capacity <= 0:
            return
        with self._lock:
            self._items[rid] = value
            self._items.move_to_end(rid)
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)

    async def get(self, rid: str) -> StoredResponse | None:
        with self._lock:
            value = self._items.get(rid)
            if value is not None:
                self._items.move_to_end(rid)
            return value

    async def delete(self, rid: str) -> bool:
        with self._lock:
            return self._items.pop(rid, None) is not None

    def __len__(self) -> int:
        return len(self._items)


class ModalDictStore:
    """Responses in a named ``modal.Dict`` shared by every container."""

    def __init__(self, name: str, dict_obj: Any = None) -> None:
        if dict_obj is None:
            import modal  # available inside Modal containers

            dict_obj = modal.Dict.from_name(name, create_if_missing=True)
        self._d = dict_obj

    async def put(self, rid: str, value: StoredResponse) -> None:
        await self._d.put.aio(rid, dataclasses.asdict(value))

    async def get(self, rid: str) -> StoredResponse | None:
        raw = await self._d.get.aio(rid)
        return StoredResponse(**raw) if raw else None

    async def delete(self, rid: str) -> bool:
        if not await self._d.contains.aio(rid):
            return False
        await self._d.pop.aio(rid)
        return True


def make_store(spec: str, capacity: int) -> ResponseStore:
    """``"memory"`` or ``"modal-dict:<name>"``."""
    if spec == "memory":
        return MemoryStore(capacity)
    if spec.startswith("modal-dict:") and spec.split(":", 1)[1]:
        return ModalDictStore(spec.split(":", 1)[1])
    raise ValueError(f"unknown response_store {spec!r}: use 'memory' or 'modal-dict:<name>'")
