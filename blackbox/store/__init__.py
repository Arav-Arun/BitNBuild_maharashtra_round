"""SQLite schema and serialized database access."""

from __future__ import annotations

import argparse
import queue
import sqlite3
import threading
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import Future
from pathlib import Path
from typing import Any, TypeVar

from blackbox.config import Settings

T = TypeVar("T")


class SQLiteDatabase:
    """Own one SQLite connection on a worker thread and serialize all access."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._jobs: queue.Queue[tuple[Callable[[sqlite3.Connection], Any], Future[Any]] | None] = (
            queue.Queue()
        )
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        schema = Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")
        self.call(lambda connection: connection.executescript(schema))

    def _worker(self) -> None:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            while True:
                job = self._jobs.get()
                if job is None:
                    return
                operation, future = job
                try:
                    result = operation(connection)
                    connection.commit()
                except BaseException as error:
                    connection.rollback()
                    future.set_exception(error)
                else:
                    future.set_result(result)
        finally:
            connection.close()

    def call(self, operation: Callable[[sqlite3.Connection], T]) -> T:
        if not self._thread.is_alive():
            raise RuntimeError("database is closed")
        future: Future[T] = Future()
        self._jobs.put((operation, future))
        return future.result()

    def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        def operation(connection: sqlite3.Connection) -> int:
            cursor = connection.execute(sql, params)
            return cursor.lastrowid

        return self.call(operation)

    def executemany(self, sql: str, rows: Iterable[Sequence[Any]]) -> None:
        materialized = list(rows)
        self.call(lambda connection: connection.executemany(sql, materialized))

    def transaction(self, statements: Iterable[tuple[str, Sequence[Any]]]) -> None:
        materialized = list(statements)

        def operation(connection: sqlite3.Connection) -> None:
            for sql, params in materialized:
                connection.execute(sql, params)

        self.call(operation)

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        def operation(connection: sqlite3.Connection) -> list[dict[str, Any]]:
            return [dict(row) for row in connection.execute(sql, params).fetchall()]

        return self.call(operation)

    def one(self, sql: str, params: Sequence[Any] = ()) -> dict[str, Any] | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def close(self) -> None:
        if self._thread.is_alive():
            self._jobs.put(None)
            self._thread.join()

    def __enter__(self) -> SQLiteDatabase:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize the Black Box SQLite database")
    parser.add_argument("--path", type=Path)
    args = parser.parse_args()
    path = args.path or Settings.load().data_dir / "blackbox.db"
    with SQLiteDatabase(path):
        pass
    print(path)


if __name__ == "__main__":
    main()
