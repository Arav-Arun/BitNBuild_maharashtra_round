"""Versioned agent state with read/write provenance."""

from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from copy import deepcopy
from typing import Any, Protocol


class StateObserver(Protocol):
    def state_read(self, key: str, value: Any, version: int, producer: str | None) -> None: ...

    def state_write(self, key: str, value: Any, version: int) -> str | None: ...

    def snapshot(self, state: dict[str, Any]) -> str: ...


class State(MutableMapping[str, Any]):
    def __init__(self, observer: StateObserver, initial: dict[str, Any] | None = None) -> None:
        self._observer = observer
        self._data = deepcopy(initial or {})
        self._versions = {key: 1 for key in self._data}
        self._producers: dict[str, str | None] = {key: None for key in self._data}

    def __getitem__(self, key: str) -> Any:
        value = self._data[key]
        self._observer.state_read(
            key,
            value,
            self._versions.get(key, 0),
            self._producers.get(key),
        )
        return value

    def __setitem__(self, key: str, value: Any) -> None:
        version = self._versions.get(key, 0) + 1
        self._data[key] = deepcopy(value)
        self._versions[key] = version
        self._producers[key] = self._observer.state_write(key, value, version)

    def __delitem__(self, key: str) -> None:
        version = self._versions.get(key, 0) + 1
        del self._data[key]
        self._versions[key] = version
        self._producers[key] = self._observer.state_write(key, None, version)

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def as_dict(self) -> dict[str, Any]:
        return deepcopy(self._data)

    def checkpoint(self) -> str:
        return self._observer.snapshot(self.as_dict())
