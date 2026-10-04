"""Read-only access to a Recorder directory, for consumers that must never write to it."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from blackbox.recorder import Store


class ReadOnlyDatabase:
    """The Recorder's SQLite file opened with ``mode=ro``; the same query/one helpers."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"no Recorder database at {self.path}")
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(
            f"{self.path.resolve().as_uri()}?mode=ro", uri=True, check_same_thread=False
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA query_only = ON")

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(row) for row in self._connection.execute(sql, params).fetchall()]

    def one(self, sql: str, params: Sequence[Any] = ()) -> dict[str, Any] | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> ReadOnlyDatabase:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class ReadOnlyStore(Store):
    """The content store without the constructor's directory creation or any write path."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        if not (self.root / "blobs").is_dir():
            raise FileNotFoundError(f"no content store at {self.root}")

    def _save_bytes(self, directory: str, payload: bytes) -> str:
        raise PermissionError("the content store is open read-only")


class RecorderView:
    """One Recorder directory (``blackbox.db`` plus ``content/``) opened read-only."""

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self.database = ReadOnlyDatabase(self.data_dir / "blackbox.db")
        self.store = ReadOnlyStore(self.data_dir / "content")

    def run(self, run_id: str) -> dict[str, Any] | None:
        return self.database.one("SELECT * FROM runs WHERE run_id = ?", (run_id,))

    def steps(self, run_id: str) -> list[dict[str, Any]]:
        """The run's steps in execution order."""
        return self.database.query("SELECT * FROM steps WHERE run_id = ? ORDER BY seq", (run_id,))

    def step(self, run_id: str, addr: str) -> dict[str, Any] | None:
        return self.database.one("SELECT * FROM steps WHERE run_id = ? AND addr = ?", (run_id, addr))

    def edges(self, run_id: str) -> list[dict[str, Any]]:
        return self.database.query("SELECT * FROM edges WHERE run_id = ?", (run_id,))

    def load(self, ref: str | None) -> Any:
        """A recorded blob (a step's input, output or reasoning) by its content hash."""
        return None if ref is None else self.store.load_json(ref)

    def state(self, ref: str | None) -> dict[str, Any] | None:
        """A state snapshot (``state_before`` / ``state_after``) by its checkpoint hash."""
        return None if ref is None else self.store.load_checkpoint(ref)

    def close(self) -> None:
        self.database.close()

    def __enter__(self) -> RecorderView:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
