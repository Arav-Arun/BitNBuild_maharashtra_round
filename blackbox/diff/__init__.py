"""Compare two recorded runs step by step, and find the nearest passing twin of a run (Task 9).

Alignment is by address when both runs visited the same addresses in the same order.
When control flow changed, a longest common subsequence over ``(kind, name, args_hash)``
anchors the steps that clearly correspond; unmatched steps between two anchors pair up
by address (``changed``) and anything left over is ``new`` or ``removed``.

Statuses: ``same`` (identical input and output), ``cached`` (identical, and the right
run served it from the recording), ``changed``, ``new`` and ``removed``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from blackbox.recorder import Store, canonical_json, content_hash
from blackbox.store import SQLiteDatabase

DIVERGENT = {"changed", "new", "removed"}
_CHUNK = 500


@dataclass(slots=True)
class _Step:
    addr: str
    seq: int
    kind: str
    name: str
    input_hash: str | None
    output_hash: str | None
    state_before: str | None
    state_after: str | None
    cache_status: str
    args_hash: str | None


def _load_blob(store: Store, digest: str | None) -> Any:
    if not digest:
        return None
    try:
        return store.load_json(digest)
    except (OSError, ValueError):
        return None


def _steps(database: SQLiteDatabase, store: Store, run_id: str) -> list[_Step]:
    steps = []
    for row in database.query("SELECT * FROM steps WHERE run_id = ? ORDER BY seq", (run_id,)):
        args_hash = None
        if row["kind"] == "tool":
            request = _load_blob(store, row["input_hash"])
            if isinstance(request, dict):
                args_hash = content_hash(request.get("args"))
        steps.append(
            _Step(
                addr=row["addr"],
                seq=row["seq"],
                kind=row["kind"],
                name=row["name"],
                input_hash=row["input_hash"],
                output_hash=row["output_hash"],
                state_before=row["state_before"],
                state_after=row["state_after"],
                cache_status=row["cache_status"],
                args_hash=args_hash,
            )
        )
    return steps


def _lcs(left: list[tuple], right: list[tuple]) -> list[tuple[int, int]]:
    n, m = len(left), len(right)
    table = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            table[i][j] = (
                table[i + 1][j + 1] + 1
                if left[i] == right[j]
                else max(table[i + 1][j], table[i][j + 1])
            )
    pairs, i, j = [], 0, 0
    while i < n and j < m:
        if left[i] == right[j]:
            pairs.append((i, j))
            i += 1
            j += 1
        elif table[i + 1][j] >= table[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


def align(left: list[_Step], right: list[_Step]) -> list[tuple[_Step | None, _Step | None]]:
    """Pairs of corresponding steps; ``None`` on one side means new or removed."""
    if [s.addr for s in left] == [s.addr for s in right]:
        return list(zip(left, right))

    def key(step: _Step) -> tuple:
        return (step.kind, step.name, step.args_hash)

    anchors = _lcs([key(s) for s in left], [key(s) for s in right])
    pairs: list[tuple[_Step | None, _Step | None]] = []
    previous = (-1, -1)
    for anchor in [*anchors, (len(left), len(right))]:
        gap_left = left[previous[0] + 1 : anchor[0]]
        gap_right = right[previous[1] + 1 : anchor[1]]
        by_addr = {s.addr: s for s in gap_right}
        matched = set()
        for step in gap_left:
            twin = by_addr.get(step.addr)
            if twin is not None:
                pairs.append((step, twin))
                matched.add(twin.addr)
            else:
                pairs.append((step, None))
        pairs.extend((None, s) for s in gap_right if s.addr not in matched)
        if anchor[0] < len(left):
            pairs.append((left[anchor[0]], right[anchor[1]]))
        previous = anchor
    return pairs


def _status(left: _Step | None, right: _Step | None, same_run: bool) -> str:
    if left is None:
        return "new"
    if right is None:
        return "removed"
    if same_run:
        return "same"
    if right.cache_status == "edited":
        return "changed"
    if left.input_hash == right.input_hash and left.output_hash == right.output_hash:
        return "cached" if right.cache_status == "cached" else "same"
    return "changed"


def _payload(value: Any) -> Any:
    """The semantic part of a stored output: an LLM reply's parsed content."""
    try:
        content = value["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return value
    if isinstance(content, str):
        try:
            return json.loads(content)
        except ValueError:
            return content
    return content


def _escape(key: str) -> str:
    return str(key).replace("~", "~0").replace("/", "~1")


def leaves(value: Any, pointer: str = "") -> dict[str, Any]:
    """Scalar leaves keyed by RFC 6901 JSON Pointer."""
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            out.update(leaves(item, f"{pointer}/{_escape(key)}"))
        return out
    if isinstance(value, list):
        out = {}
        for index, item in enumerate(value):
            out.update(leaves(item, f"{pointer}/{index}"))
        return out
    return {pointer: value}


def pointer_changes(left: Any, right: Any, limit: int = 20) -> list[dict[str, Any]]:
    """Field-level differences between two payloads, by JSON Pointer."""
    a, b = leaves(_payload(left)), leaves(_payload(right))
    changes = []
    for pointer in sorted(set(a) | set(b)):
        if (
            pointer in a
            and pointer in b
            and canonical_json(a[pointer]) == canonical_json(b[pointer])
        ):
            continue
        changes.append({"pointer": pointer, "left": a.get(pointer), "right": b.get(pointer)})
        if len(changes) >= limit:
            break
    return changes


def _final_state(store: Store, steps: list[_Step]) -> dict[str, str]:
    for step in reversed(steps):
        if step.state_after:
            try:
                return store.checkpoint_entries(step.state_after)
            except (OSError, ValueError):
                return {}
    return {}


def compare(
    database: SQLiteDatabase, store: Store, left_run_id: str, right_run_id: str
) -> dict[str, Any]:
    """A ``DiffResponse``: rows, first divergence, per-key state diff and outcome diff."""
    runs = {}
    for run_id in (left_run_id, right_run_id):
        run = database.one("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        if run is None:
            raise KeyError(f"unknown run: {run_id}")
        runs[run_id] = run
    left_steps = _steps(database, store, left_run_id)
    right_steps = _steps(database, store, right_run_id)
    same_run = left_run_id == right_run_id
    rows = []
    for left, right in align(left_steps, right_steps):
        status = _status(left, right, same_run)
        left_value = _load_blob(store, left.output_hash) if left else None
        right_value = _load_blob(store, right.output_hash) if right else None
        row: dict[str, Any] = {
            "addr": (right or left).addr,
            "status": status,
            "left": _payload(left_value),
            "right": _payload(right_value),
        }
        if status == "changed":
            row["changes"] = pointer_changes(left_value, right_value)
        rows.append(row)
    first = next((row["addr"] for row in rows if row["status"] in DIVERGENT), None)

    left_state = _final_state(store, left_steps)
    right_state = _final_state(store, right_steps)
    state_diff = []
    for key in sorted(set(left_state) | set(right_state)):
        if key not in right_state:
            status = "removed"
        elif key not in left_state:
            status = "new"
        else:
            status = "same" if left_state[key] == right_state[key] else "changed"
        entry: dict[str, Any] = {"key": key, "status": status}
        if status != "same":
            entry["left"] = _load_blob(store, left_state.get(key))
            entry["right"] = _load_blob(store, right_state.get(key))
        state_diff.append(entry)

    def outcome(run: dict[str, Any]) -> dict[str, Any]:
        return {"outcome": run["outcome"], "score": run["score"], "reason": run["checker_reason"]}

    left_outcome, right_outcome = outcome(runs[left_run_id]), outcome(runs[right_run_id])
    return {
        "left_run_id": left_run_id,
        "right_run_id": right_run_id,
        "first_divergence": first,
        "rows": rows,
        "state_diff": state_diff,
        "outcome": {
            "left": left_outcome,
            "right": right_outcome,
            "flipped": left_outcome["outcome"] != right_outcome["outcome"],
        },
    }


# ---------------------------------------------------------------------------
# Nearest passing twin
# ---------------------------------------------------------------------------


def nearest_passing(database: SQLiteDatabase, run_id: str) -> str | None:
    """The passing (non-fork) run most like ``run_id``.

    Prefer the same task (ideally the same seed and model, so requests match byte for
    byte); otherwise the same agent with the most similar step graph (Jaccard over
    addresses). A same-task twin is a source of known-good values; any twin is a
    side-by-side reference.
    """
    run = database.one("SELECT * FROM runs WHERE run_id = ?", (run_id,))
    if run is None:
        raise KeyError(f"unknown run: {run_id}")
    candidates = database.query(
        "SELECT run_id, task_id, seed, model FROM runs "
        "WHERE outcome = 'passed' AND fork_id IS NULL AND agent = ? AND run_id != ?",
        (run["agent"], run_id),
    )
    same_task = [c for c in candidates if c["task_id"] == run["task_id"]]
    pool = same_task or candidates
    if not pool:
        return None
    addrs = {
        r["addr"] for r in database.query("SELECT addr FROM steps WHERE run_id = ?", (run_id,))
    }
    shapes: dict[str, set[str]] = {c["run_id"]: set() for c in pool}
    ids = list(shapes)
    for start in range(0, len(ids), _CHUNK):
        chunk = ids[start : start + _CHUNK]
        marks = ",".join("?" * len(chunk))
        for row in database.query(
            f"SELECT run_id, addr FROM steps WHERE run_id IN ({marks})", chunk
        ):
            shapes[row["run_id"]].add(row["addr"])

    def score(candidate: dict[str, Any]) -> tuple:
        shape = shapes[candidate["run_id"]]
        union = addrs | shape
        jaccard = len(addrs & shape) / len(union) if union else 0.0
        return (
            jaccard,
            candidate["seed"] == run["seed"],
            candidate["model"] == run["model"],
            candidate["run_id"] == run["parent_run_id"],
        )

    return max(pool, key=lambda c: (score(c), c["run_id"]))["run_id"]


__all__ = ["align", "compare", "leaves", "nearest_passing", "pointer_changes"]
