"""Per-step feature rows for the blame ranker (Task 7).

Each step of a run becomes one row of numbers, in groups that can be switched off:

A structure   where the step sits (position, kind, reads/writes)
B telemetry   error/retry/finish signals the runtime reports (never replay timings)
C validity    deterministic schema checks: error payloads, unparsable JSON, missing keys
D grounding   values that contradict the step's own input or earlier steps
F novelty     distance from the same address in healthy runs (robust z-scores, staleness)
G context     the same signals one step before/after, and "first anomaly in the run"
H lineage     how far the step's output travelled through the provenance graph

Novelty is measured against a :class:`Reference` fit only on passing runs, so "unusual"
always means "unlike healthy runs of the same step", never "unlike the test set".

Views degrade a trace to what a weaker recorder would have captured before any feature
is computed, so the recorder ablations compare like with like:

- ``full``          everything Black Box records
- ``otel``          span name/order/status/telemetry only: no payloads, state or provenance
- ``outputs_only``  ordered step names and outputs only: no inputs, state, telemetry, edges

Edge views (``full``, ``protocol``, ``none``) separately vary only the graph used by H.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Any

import numpy as np

from blackbox.ml.dataset import Trace, TraceStep

KINDS = ("llm", "tool", "retrieval", "state")
DATE = re.compile(r"\d{4}-\d{2}-\d{2}")

# name -> group. The order here is the column order of every matrix.
FEATURE_GROUPS: dict[str, str] = {
    # A. structure
    "rel_pos": "A",
    "seq": "A",
    "n_steps": "A",
    "steps_after": "A",
    "is_last": "A",
    **{f"kind_{kind}": "A" for kind in KINDS},
    "n_reads": "A",
    "n_writes": "A",
    # B. telemetry
    "has_error_type": "B",
    "retries": "B",
    "finish_length": "B",
    "status_error": "B",
    # C. validity
    "parse_failed": "C",
    "empty_output": "C",
    "bad_numbers": "C",
    "dup_list_items": "C",
    "keys_unseen_frac": "C",
    "keys_missing_frac": "C",
    "type_mismatch_frac": "C",
    "empty_arg": "C",
    # D. grounding
    "ungrounded_num_frac": "D",
    "io_conflicts": "D",
    "earlier_conflicts": "D",
    # F. novelty
    "num_max_absz": "F",
    "num_mean_absz": "F",
    "date_days_older": "F",
    "date_older_z": "F",
    "unseen_string_frac": "F",
    "leaf_count_z": "F",
    "text_len_z": "F",
    "anomaly": "F",
    # G. context
    "prev_anomaly": "G",
    "next_anomaly": "G",
    "earlier_anomalies": "G",
    "later_anomalies": "G",
    "is_first_anomaly": "G",
    "later_status_errors": "G",
    "anomaly_pct": "G",
    "num_absz_pct": "G",
    "conflict_pct": "G",
    # H. lineage
    "n_parents": "H",
    "n_children": "H",
    "n_descendants": "H",
    "frac_descendants": "H",
    "reaches_last": "H",
    "depth": "H",
    "later_overwritten": "H",
}
FEATURE_NAMES: tuple[str, ...] = tuple(FEATURE_GROUPS)
GROUPS = tuple(sorted(set(FEATURE_GROUPS.values())))

# Identifiers and replay bookkeeping that would hand the model the answer.
FORBIDDEN_FEATURES = frozenset(
    {
        "run_id",
        "task_id",
        "fork_id",
        "fault_type",
        "root_addr",
        "source",
        "label",
        "relevance",
        "cache_status",
        "latency_ms",
        "held_out",
        "distractor_addr",
    }
)

OBSERVABILITY_VIEWS = ("full", "otel", "outputs_only")
EDGE_VIEWS = ("full", "protocol", "none")


def assert_no_leakage(names: Iterable[str]) -> None:
    leaked = set(names) & FORBIDDEN_FEATURES
    if leaked:
        raise AssertionError(f"label or replay columns in the feature matrix: {sorted(leaked)}")


# ---------------------------------------------------------------------------
# Payload helpers
# ---------------------------------------------------------------------------


def _maybe_json(value: Any) -> Any:
    if isinstance(value, str) and value[:1] in "{[":
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _llm_content(output: Any) -> str | None:
    try:
        content = output["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None
    return content if isinstance(content, str) else None


def output_payload(step: TraceStep) -> tuple[Any, bool]:
    """The step's semantic output and whether it parsed (False only for broken LLM JSON)."""
    if step.kind == "llm":
        content = _llm_content(step.output)
        if content is None:
            return None, step.output is None
        try:
            return json.loads(content), True
        except ValueError:
            return content, False
    return step.output, True


def input_payload(step: TraceStep) -> Any:
    if not isinstance(step.input, dict):
        return None
    if step.kind == "llm":
        messages = step.input.get("messages") or []
        users = [_maybe_json(m.get("content")) for m in messages if m.get("role") == "user"]
        return users[-1] if users else None
    return step.input.get("args", step.input)


def flatten(value: Any, path: str = "") -> list[tuple[str, Any]]:
    """Scalar leaves as (JSON-pointer-like path, value); list indices become ``*``."""
    if isinstance(value, dict):
        leaves = []
        for key, item in value.items():
            leaves.extend(flatten(item, f"{path}/{key}"))
        return leaves
    if isinstance(value, list):
        leaves = []
        for item in value:
            leaves.extend(flatten(item, f"{path}/*"))
        return leaves
    return [(path or "/", value)]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _date(value: Any) -> int | None:
    if isinstance(value, str) and DATE.fullmatch(value[:10]):
        try:
            return date.fromisoformat(value[:10]).toordinal()
        except ValueError:
            return None
    return None


def _leaf_name(path: str) -> str | None:
    """Last key of a path outside any list; list members are not single facts."""
    if "/*" in path:
        return None
    return path.rsplit("/", 1)[-1] or None


def is_error_payload(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    status = value.get("status")
    return "error" in value or (_is_number(status) and status >= 400)


def _conflicts(trace: Trace) -> dict[str, tuple[int, int]]:
    """Per step: (outputs contradicting the step's own input, outputs contradicting a
    value every earlier step agreed on). Matched by leaf name, so some collisions are
    structural; features use the excess over the healthy baseline of the same address.
    """
    counts: dict[str, tuple[int, int]] = {}
    earlier_values: dict[str, set[str]] = defaultdict(set)
    for step in sorted(trace.steps, key=lambda s: s.seq):
        payload, _ = output_payload(step)
        leaves = flatten(payload) if payload is not None else []
        inputs = input_payload(step)
        input_named = {
            name: value
            for path, value in (flatten(inputs) if inputs is not None else [])
            if (name := _leaf_name(path))
        }
        io = sum(
            1
            for path, value in leaves
            if (name := _leaf_name(path)) in input_named and input_named[name] != value
        )
        earlier = 0
        for path, value in leaves:
            name = _leaf_name(path)
            if name is None:
                continue
            previous = earlier_values.get(name)
            if previous and len(previous) == 1 and json.dumps(value, default=str) not in previous:
                earlier += 1
        for path, value in leaves:
            if (name := _leaf_name(path)) is not None:
                earlier_values[name].add(json.dumps(value, default=str))
        counts[step.addr] = (io, earlier)
    return counts


def _robust(values: Sequence[float], constant_scale: float | None = None) -> tuple[float, float]:
    """Median and a robust spread (scaled MAD). A field that never varies in healthy runs
    gets ``constant_scale`` (one day for dates) or 5% of its magnitude (amounts)."""
    median = statistics.median(values)
    mad = statistics.median(abs(v - median) for v in values)
    scale = 1.4826 * mad
    if scale == 0:
        spread = max(values) - min(values)
        if spread:
            scale = spread / 4
        elif constant_scale is not None:
            scale = constant_scale
        else:
            scale = max(abs(median) * 0.05, 1e-6)
    return median, scale


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------


def _protocol_edges(trace: Trace) -> list[tuple[str, str, str]]:
    """Call order and same-agent hand-offs only: what a span tree without data provenance knows."""
    edges = set()
    ordered = sorted(trace.steps, key=lambda s: s.seq)
    for before, after in zip(ordered, ordered[1:]):
        edges.add((before.addr, after.addr, "protocol"))
    last_by_role: dict[str, str] = {}
    for step in ordered:
        role = step.role or step.addr.split("/")[0]
        if role in last_by_role:
            edges.add((last_by_role[role], step.addr, "protocol"))
        last_by_role[role] = step.addr
    return sorted(edges)


def materialize(trace: Trace, view: str = "full", edges: str = "full") -> Trace:
    """Degrade a trace to what a weaker recorder would have captured."""
    if view not in OBSERVABILITY_VIEWS:
        raise ValueError(f"unknown observability view: {view}")
    if edges not in EDGE_VIEWS:
        raise ValueError(f"unknown edge view: {edges}")
    if view == "otel":
        edges = "protocol" if edges == "full" else edges
        steps = [
            replace(
                step,
                input=None,
                output={"error": True} if is_error_payload(output_payload(step)[0]) else None,
                reads=(),
                writes=(),
            )
            for step in trace.steps
        ]
    elif view == "outputs_only":
        edges = "none"
        steps = [
            replace(step, input=None, reads=(), writes=(), error_type=None, retries=0)
            for step in trace.steps
        ]
    else:
        steps = list(trace.steps)
    degraded = replace(trace, steps=steps)
    if edges == "protocol":
        degraded.edges = _protocol_edges(trace)
    elif edges == "none":
        degraded.edges = []
    return degraded


# ---------------------------------------------------------------------------
# Reference profile of healthy runs
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _AddrProfile:
    runs: int = 0
    paths: Counter = field(default_factory=Counter)
    types: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    numbers: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    dates: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))
    strings: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    leaf_counts: list[int] = field(default_factory=list)
    text_lengths: list[int] = field(default_factory=list)
    io_conflicts: list[int] = field(default_factory=list)
    earlier_conflicts: list[int] = field(default_factory=list)
    stats: dict[str, tuple[float, float]] = field(default_factory=dict)
    date_stats: dict[str, tuple[float, float]] = field(default_factory=dict)
    baseline_io_conflicts: float = 0.0
    baseline_earlier_conflicts: float = 0.0


class Reference:
    """Per-address statistics of healthy (passing) runs."""

    def __init__(self) -> None:
        self.profiles: dict[str, _AddrProfile] = {}

    @classmethod
    def fit(cls, traces: Iterable[Trace]) -> Reference:
        reference = cls()
        for trace in traces:
            conflicts = _conflicts(trace)
            for step in trace.steps:
                payload, _ = output_payload(step)
                profile = reference.profiles.setdefault(step.addr, _AddrProfile())
                profile.runs += 1
                io, earlier = conflicts[step.addr]
                profile.io_conflicts.append(io)
                profile.earlier_conflicts.append(earlier)
                leaves = flatten(payload) if payload is not None else []
                profile.leaf_counts.append(len(leaves))
                profile.text_lengths.append(len(json.dumps(payload, default=str)))
                for path in {p for p, _ in leaves}:
                    profile.paths[path] += 1
                for path, value in leaves:
                    profile.types[path][type(value).__name__] += 1
                    if _is_number(value):
                        profile.numbers[path].append(float(value))
                    elif (day := _date(value)) is not None:
                        profile.dates[path].append(day)
                    elif isinstance(value, str):
                        profile.strings[path].add(value)
        for profile in reference.profiles.values():
            profile.stats = {p: _robust(v) for p, v in profile.numbers.items() if len(v) >= 3}
            profile.date_stats = {
                p: _robust(v, constant_scale=1.0) for p, v in profile.dates.items() if len(v) >= 3
            }
            profile.stats["__leaves__"] = _robust(profile.leaf_counts)
            profile.stats["__text__"] = _robust(profile.text_lengths)
            profile.baseline_io_conflicts = statistics.median(profile.io_conflicts)
            profile.baseline_earlier_conflicts = statistics.median(profile.earlier_conflicts)
        return reference

    def to_dict(self) -> dict[str, Any]:
        """Only the statistics features read; raw value lists are not kept."""
        return {
            addr: {
                "runs": p.runs,
                "paths": dict(p.paths),
                "types": {path: dict(c) for path, c in p.types.items()},
                "strings": {path: sorted(v) for path, v in p.strings.items()},
                "stats": {path: list(v) for path, v in p.stats.items()},
                "date_stats": {path: list(v) for path, v in p.date_stats.items()},
                "baseline_io_conflicts": p.baseline_io_conflicts,
                "baseline_earlier_conflicts": p.baseline_earlier_conflicts,
            }
            for addr, p in self.profiles.items()
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Reference:
        reference = cls()
        for addr, item in data.items():
            profile = _AddrProfile(runs=item["runs"], paths=Counter(item["paths"]))
            profile.types = defaultdict(Counter, {p: Counter(c) for p, c in item["types"].items()})
            profile.strings = defaultdict(set, {p: set(v) for p, v in item["strings"].items()})
            profile.stats = {p: tuple(v) for p, v in item["stats"].items()}
            profile.date_stats = {p: tuple(v) for p, v in item["date_stats"].items()}
            profile.baseline_io_conflicts = item["baseline_io_conflicts"]
            profile.baseline_earlier_conflicts = item["baseline_earlier_conflicts"]
            reference.profiles[addr] = profile
        return reference


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------


def _graph(trace: Trace) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    children: dict[str, set[str]] = defaultdict(set)
    parents: dict[str, set[str]] = defaultdict(set)
    known = set(trace.addrs)
    for src, dst, _ in trace.edges:
        if src in known and dst in known:
            children[src].add(dst)
            parents[dst].add(src)
    return children, parents


def _descendants(addr: str, children: dict[str, set[str]]) -> set[str]:
    seen: set[str] = set()
    stack = [addr]
    while stack:
        for child in children.get(stack.pop(), ()):
            if child not in seen:
                seen.add(child)
                stack.append(child)
    return seen


def _pct_rank(values: list[float]) -> list[float]:
    """Within-run percentile, so features transfer between agents with different scales."""
    if len(values) <= 1:
        return [0.0 for _ in values]
    clean = [(-math.inf if v is None or math.isnan(v) else v) for v in values]
    order = sorted(clean)
    return [order.index(v) / (len(values) - 1) for v in clean]


def _step_signals(step: TraceStep, profile: _AddrProfile | None) -> dict[str, float]:
    payload, parsed = output_payload(step)
    leaves = flatten(payload) if payload is not None else []
    inputs = input_payload(step)
    input_leaves = flatten(inputs) if inputs is not None else []
    nan = math.nan
    row: dict[str, float] = {
        "has_error_type": float(step.error_type is not None),
        "retries": float(step.retries),
        "finish_length": float(step.finish_reason == "length"),
        "status_error": float(is_error_payload(payload)),
        "parse_failed": float(not parsed),
        "empty_output": float(
            step.kind != "state" and (payload is None or payload in ({}, [], ""))
        ),
        "bad_numbers": float(
            sum(
                1
                for _, v in leaves
                if _is_number(v) and (not math.isfinite(v) or (v < 0 and not isinstance(v, bool)))
            )
        ),
        "dup_list_items": 0.0,
        "empty_arg": float(any(v == "" for _, v in input_leaves)),
    }
    if isinstance(payload, (dict, list)):
        lists = [payload] if isinstance(payload, list) else []
        if isinstance(payload, dict):
            lists = [v for v in payload.values() if isinstance(v, list)]
        duplicates = 0
        for items in lists:
            encoded = [json.dumps(i, sort_keys=True, default=str) for i in items]
            duplicates = max(duplicates, len(encoded) - len(set(encoded)))
        row["dup_list_items"] = float(duplicates)

    # D. grounding: outputs that disagree with the step's own inputs.
    input_numbers = {float(v) for _, v in input_leaves if _is_number(v)}
    output_numbers = [float(v) for _, v in leaves if _is_number(v)]
    if input_leaves and output_numbers:
        row["ungrounded_num_frac"] = sum(n not in input_numbers for n in output_numbers) / len(
            output_numbers
        )
    else:
        row["ungrounded_num_frac"] = nan

    # C/F. validity and novelty against the healthy profile of this address.
    if profile is None or profile.runs == 0:
        for name in (
            "keys_unseen_frac",
            "keys_missing_frac",
            "type_mismatch_frac",
            "num_max_absz",
            "num_mean_absz",
            "date_days_older",
            "date_older_z",
            "unseen_string_frac",
            "leaf_count_z",
            "text_len_z",
        ):
            row[name] = nan
        return row
    paths = {p for p, _ in leaves}
    common = {p for p, c in profile.paths.items() if c >= 0.9 * profile.runs}
    row["keys_unseen_frac"] = (
        sum(p not in profile.paths for p in paths) / len(paths) if paths else 0.0
    )
    row["keys_missing_frac"] = len(common - paths) / len(common) if common else 0.0
    mismatched = 0
    zs: list[float] = []
    older: list[float] = []
    older_z: list[float] = []
    unseen_strings = []
    for path, value in leaves:
        types = profile.types.get(path)
        if types and type(value).__name__ != types.most_common(1)[0][0]:
            mismatched += 1
        if _is_number(value) and path in profile.stats:
            median, scale = profile.stats[path]
            zs.append(abs(float(value) - median) / scale)
        elif (day := _date(value)) is not None and path in profile.date_stats:
            median, scale = profile.date_stats[path]
            older.append(median - day)
            older_z.append((median - day) / scale)
        elif isinstance(value, str) and path in profile.strings:
            unseen_strings.append(value not in profile.strings[path])
    row["type_mismatch_frac"] = mismatched / len(leaves) if leaves else 0.0
    row["num_max_absz"] = max(zs) if zs else 0.0
    row["num_mean_absz"] = sum(zs) / len(zs) if zs else 0.0
    row["date_days_older"] = max(older) if older else 0.0
    row["date_older_z"] = max(older_z) if older_z else 0.0
    row["unseen_string_frac"] = sum(unseen_strings) / len(unseen_strings) if unseen_strings else 0.0
    median, scale = profile.stats["__leaves__"]
    row["leaf_count_z"] = (len(leaves) - median) / scale
    median, scale = profile.stats["__text__"]
    row["text_len_z"] = (len(json.dumps(payload, default=str)) - median) / scale
    return row


def _anomaly(row: dict[str, float]) -> float:
    """One unsupervised suspicion score; also the 'anomaly maximum' baseline."""

    def clip(value: float, scale: float) -> float:
        return 0.0 if value is None or math.isnan(value) else min(abs(value) / scale, 1.0)

    return (
        row["status_error"]
        + row["parse_failed"]
        + row["has_error_type"]
        + row["finish_length"]
        + clip(row["keys_unseen_frac"], 0.5)
        + clip(row["keys_missing_frac"], 0.5)
        + clip(row["type_mismatch_frac"], 0.5)
        + clip(row["num_max_absz"], 6.0)
        + clip(max(row["date_older_z"], 0.0), 6.0)
        + clip(max(row["io_conflicts"], 0.0), 1.0)
        + clip(max(row["earlier_conflicts"], 0.0), 1.0)
        + clip(row["dup_list_items"], 1.0)
        + clip(row["bad_numbers"], 1.0)
    )


class FeatureBuilder:
    """Turn traces into a step × feature matrix. Accepts traces only, never labels."""

    def __init__(self, reference: Reference) -> None:
        self.reference = reference

    def trace_rows(self, trace: Trace) -> list[dict[str, float]]:
        if not isinstance(trace, Trace):
            raise TypeError("FeatureBuilder only accepts Trace objects")
        steps = sorted(trace.steps, key=lambda s: s.seq)
        n = len(steps)
        children, parents = _graph(trace)
        last = steps[-1].addr if steps else None
        rows: list[dict[str, float]] = []
        conflicts = _conflicts(trace)
        depth: dict[str, int] = {}
        for index, step in enumerate(steps):
            profile = self.reference.profiles.get(step.addr)
            row = _step_signals(step, profile)
            io, earlier = conflicts[step.addr]
            row["io_conflicts"] = io - (profile.baseline_io_conflicts if profile else 0.0)
            row["earlier_conflicts"] = earlier - (
                profile.baseline_earlier_conflicts if profile else 0.0
            )
            row["anomaly"] = _anomaly(row)

            descendants = _descendants(step.addr, children)
            depth[step.addr] = 1 + max(
                (depth.get(p, 0) for p in parents.get(step.addr, ())), default=-1
            )
            written = {k.rsplit("@v", 1)[0] for k in step.writes}
            row.update(
                rel_pos=index / (n - 1) if n > 1 else 0.0,
                seq=float(index),
                n_steps=float(n),
                steps_after=float(n - index - 1),
                is_last=float(index == n - 1),
                n_reads=float(len(step.reads)),
                n_writes=float(len(step.writes)),
                n_parents=float(len(parents.get(step.addr, ()))),
                n_children=float(len(children.get(step.addr, ()))),
                n_descendants=float(len(descendants)),
                frac_descendants=len(descendants) / (n - 1) if n > 1 else 0.0,
                reaches_last=float(last in descendants or step.addr == last),
                depth=float(depth[step.addr]),
                later_overwritten=float(
                    any(
                        written & {k.rsplit("@v", 1)[0] for k in later.writes}
                        for later in steps[index + 1 :]
                    )
                ),
                **{f"kind_{kind}": float(step.kind == kind) for kind in KINDS},
            )
            rows.append(row)

        anomalies = [row["anomaly"] for row in rows]
        first = next((i for i, a in enumerate(anomalies) if a >= 1.0), None)
        anomaly_pct = _pct_rank(anomalies)
        absz_pct = _pct_rank([row["num_max_absz"] for row in rows])
        conflict_pct = _pct_rank([row["io_conflicts"] + row["earlier_conflicts"] for row in rows])
        for i, row in enumerate(rows):
            row.update(
                prev_anomaly=anomalies[i - 1] if i > 0 else 0.0,
                next_anomaly=anomalies[i + 1] if i + 1 < n else 0.0,
                earlier_anomalies=float(sum(a >= 1.0 for a in anomalies[:i])),
                later_anomalies=float(sum(a >= 1.0 for a in anomalies[i + 1 :])),
                is_first_anomaly=float(first == i),
                later_status_errors=float(sum(r["status_error"] for r in rows[i + 1 :])),
                anomaly_pct=anomaly_pct[i],
                num_absz_pct=absz_pct[i],
                conflict_pct=conflict_pct[i],
            )
        return rows


@dataclass(slots=True)
class Matrix:
    """Stacked step rows; ``groups[i]`` is the slice of rows belonging to run ``run_ids[i]``."""

    X: np.ndarray
    feature_names: tuple[str, ...]
    run_ids: list[str]
    addrs: list[list[str]]
    offsets: list[int]

    def run_slice(self, index: int) -> slice:
        return slice(self.offsets[index], self.offsets[index + 1])

    @property
    def group_sizes(self) -> list[int]:
        return [b - a for a, b in zip(self.offsets, self.offsets[1:])]


def build_matrix(
    traces: Sequence[Trace],
    reference: Reference,
    *,
    view: str = "full",
    edges: str = "full",
    groups: Iterable[str] = GROUPS,
) -> Matrix:
    """Feature matrix for ``traces`` under one observability/edge view.

    Disabled groups are blanked to NaN rather than dropped, so every matrix has the
    same columns and a model trained on one ablation can be inspected like the others.
    """
    enabled = set(groups)
    names = FEATURE_NAMES
    assert_no_leakage(names)
    builder = FeatureBuilder(reference)
    rows: list[list[float]] = []
    run_ids, addrs, offsets = [], [], [0]
    for trace in traces:
        degraded = materialize(trace, view, edges)
        trace_rows = builder.trace_rows(degraded)
        ordered = sorted(degraded.steps, key=lambda s: s.seq)
        for row in trace_rows:
            rows.append(
                [row[name] if FEATURE_GROUPS[name] in enabled else math.nan for name in names]
            )
        run_ids.append(trace.run_id)
        addrs.append([step.addr for step in ordered])
        offsets.append(len(rows))
    X = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, len(names)))
    return Matrix(X=X, feature_names=names, run_ids=run_ids, addrs=addrs, offsets=offsets)
