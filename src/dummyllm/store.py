"""In-memory LRU store for Responses API objects (``store=true``)."""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StoredResponse:
    response: dict[str, Any]
    input_items: list[dict[str, Any]]
    # Full conversation (prior turns + this input + this output), used to
    # resolve ``previous_response_id`` chains.
    conversation: list[dict[str, Any]] = field(default_factory=list)


class ResponseStore:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._items: OrderedDict[str, StoredResponse] = OrderedDict()
        self._lock = threading.Lock()

    def put(self, rid: str, value: StoredResponse) -> None:
        if self.capacity <= 0:
            return
        with self._lock:
            self._items[rid] = value
            self._items.move_to_end(rid)
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)

    def get(self, rid: str) -> StoredResponse | None:
        with self._lock:
            value = self._items.get(rid)
            if value is not None:
                self._items.move_to_end(rid)
            return value

    def delete(self, rid: str) -> bool:
        with self._lock:
            return self._items.pop(rid, None) is not None

    def __len__(self) -> int:
        return len(self._items)
