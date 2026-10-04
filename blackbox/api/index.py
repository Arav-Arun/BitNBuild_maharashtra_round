"""The run index: per-run metadata the Runs table filters and sorts on, kept in SQLite.

Recorded runs never change, so each run is indexed once. Model-derived columns (risk,
top suspect, failure signature) are refreshed whenever the diagnoser version changes.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Any

from blackbox.api.agents import AgentProfile
from blackbox.eval.splits import task_bucket
from blackbox.forge.operators import all_operators, held_out_operators
from blackbox.recorder import Store

FAULT_SPECS = {op.spec.name: op.spec for op in [*all_operators(), *held_out_operators()]}

INDEX_COLUMNS = (
    "run_id",
    "task",
    "task_text",
    "origin",
    "split",
    "steps",
    "duration_ms",
    "llm_calls",
    "tokens_in",
    "tokens_out",
    "tokens_cached",
)


def origin_of(run: dict[str, Any], label: dict[str, Any] | None) -> str:
    if label and label["source"] == "injected":
        return "injected"
    if run.get("fork_id"):
        return "fork"
    return "natural"


def split_of(run: dict[str, Any], label: dict[str, Any] | None) -> str | None:
    """The evaluation split a labelled failure belongs to, matching ``make_splits``."""
    if label is None or label["recovered"]:
        return None
    if label["source"] == "natural_auto":
        return "S4"
    if label["source"] != "injected":
        return None
    spec = FAULT_SPECS.get(label["fault_type"])
    bucket = task_bucket(run["task_id"])
    if spec is not None and spec.held_out:
        return "S1" if bucket != "train" else None
    return "S0" if bucket == "test" else None


def duration_ms(run: dict[str, Any]) -> float:
    if not run.get("ended_at"):
        return 0.0
    started = datetime.fromisoformat(run["started_at"])
    ended = datetime.fromisoformat(run["ended_at"])
    return max(0.0, (ended - started).total_seconds() * 1000)


def build_rows(
    database: Any,
    store: Store,
    profile: AgentProfile,
    runs: Iterable[dict[str, Any]],
) -> list[tuple[Any, ...]]:
    """Index rows for ``runs``, using two aggregate queries instead of one per run."""
    runs = list(runs)
    if not runs:
        return []
    wanted = {run["run_id"] for run in runs}
    totals = {
        row["run_id"]: row
        for row in database.query(
            """
            SELECT run_id, COUNT(*) AS steps,
                   SUM(kind = 'llm' AND request_key IS NOT NULL) AS llm_calls,
                   COALESCE(SUM(tokens_in), 0) AS tokens_in,
                   COALESCE(SUM(tokens_out), 0) AS tokens_out,
                   COALESCE(SUM(tokens_cached), 0) AS tokens_cached
            FROM steps GROUP BY run_id
            """
        )
        if row["run_id"] in wanted
    }
    first_inputs = {
        row["run_id"]: row["input_hash"]
        for row in database.query(
            """
            SELECT s.run_id, s.input_hash FROM steps s
            JOIN (SELECT run_id, MIN(seq) AS seq FROM steps
                  WHERE input_hash IS NOT NULL GROUP BY run_id) f
              ON f.run_id = s.run_id AND f.seq = s.seq
            """
        )
        if row["run_id"] in wanted
    }
    labels = {
        row["run_id"]: row
        for row in database.query("SELECT * FROM labels")
        if row["run_id"] in wanted
    }
    task_cache: dict[str, tuple[str, str | None]] = {}
    rows = []
    for run in runs:
        run_id = run["run_id"]
        total = totals.get(run_id) or {}
        ref = first_inputs.get(run_id)
        if ref not in task_cache:
            try:
                first_input = store.load_json(ref) if ref else None
            except (OSError, ValueError):
                first_input = None
            task_cache[ref] = profile.task(run["task_id"], first_input)
        task, task_text = task_cache[ref]
        label = labels.get(run_id)
        rows.append(
            (
                run_id,
                task,
                task_text,
                origin_of(run, label),
                split_of(run, label),
                int(total.get("steps") or 0),
                round(duration_ms(run), 3),
                int(total.get("llm_calls") or 0),
                int(total.get("tokens_in") or 0),
                int(total.get("tokens_out") or 0),
                int(total.get("tokens_cached") or 0),
            )
        )
    return rows


def insert_rows(database: Any, rows: list[tuple[Any, ...]]) -> None:
    if not rows:
        return
    placeholders = ", ".join("?" for _ in INDEX_COLUMNS)
    database.executemany(
        f"INSERT OR REPLACE INTO run_index({', '.join(INDEX_COLUMNS)}) VALUES ({placeholders})",
        rows,
    )
