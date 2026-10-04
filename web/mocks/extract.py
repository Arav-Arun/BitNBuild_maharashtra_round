"""Read recorded runs back as contract models, plus the fixture-side derivations.

The functions that take contract models (provenance, damage path, diff, prediction) are small
reference implementations of what the backend will serve. They exist so the fixtures are
internally consistent and so the backend wave has an executable specification to compare to.
"""

from __future__ import annotations

import json
import re
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Literal, get_args

from blackbox.recorder import content_hash
from blackbox.sdk import Recorder
from server import models as m

STALE_AFTER_DAYS = 30
MAX_PROVENANCE_DEPTH = 8


# ---------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------


def indian_money(value: float) -> str:
    """Format 120000 as 'Rs 1,20,000' using the lakh grouping, with the rupee sign."""
    digits = f"{round(value):d}"
    head, tail = digits[:-3], digits[-3:]
    groups: list[str] = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return "₹" + ",".join([*groups, tail])


REASON_CLASSES: tuple[tuple[str, str], ...] = (
    (r"Incorrect INR total", "total_mismatch"),
    (r"Budget exceeded", "budget_exceeded"),
    (r"constraint violated", "constraint_violated"),
    (r"unknown ID|Unknown flight or hotel", "invalid_selection"),
    (
        r"invalid structured output|Invalid final plan|Extra inputs|validation error",
        "invalid_output",
    ),
    (r"No answer produced", "no_answer"),
    (r"Token F1", "wrong_answer"),
)


def reason_class(reason: str | None) -> str:
    for pattern, name in REASON_CLASSES:
        if reason and re.search(pattern, reason):
            return name
    return "crash" if reason else "other"


# ---------------------------------------------------------------------------
# JSON pointers
# ---------------------------------------------------------------------------


def unescape(token: str) -> str:
    return token.replace("~1", "/").replace("~0", "~")


def escape(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def resolve_pointer(document: Any, pointer: str) -> tuple[bool, Any]:
    """Resolve an RFC 6901 pointer. Returns (found, value)."""
    if pointer == "":
        return True, document
    if not pointer.startswith("/"):
        return False, None
    node = document
    for raw in pointer[1:].split("/"):
        token = unescape(raw)
        if isinstance(node, dict) and token in node:
            node = node[token]
        elif isinstance(node, list) and token.isdigit() and int(token) < len(node):
            node = node[int(token)]
        else:
            return False, None
    return True, node


def step_document(step: m.StepDetail, side: Literal["src", "dst"]) -> dict[str, Any]:
    """The document that citation and edge pointers address (see `Citation`)."""
    return {
        "input": step.input,
        "output": step.output,
        "reasoning": step.reasoning,
        "state_before": step.state_before,
        "state_after": step.state_after,
        "state": step.state_after if side == "src" else step.state_before,
    }


def _is_prefix(prefix: str, pointer: str) -> bool:
    return pointer == prefix or pointer.startswith(prefix.rstrip("/") + "/")


# ---------------------------------------------------------------------------
# Deterministic schema rules (evidence, not model output)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RawStep:
    row: dict[str, Any]
    input: Any
    output: Any
    reasoning: list[str]
    state_before: dict[str, Any]
    state_after: dict[str, Any]


def _as_of(output: Any) -> date | None:
    if isinstance(output, dict) and isinstance(output.get("as_of"), str):
        try:
            return date.fromisoformat(output["as_of"])
        except ValueError:
            return None
    return None


def evaluate_rules(steps: Sequence[RawStep]) -> dict[str, list[m.RuleViolation]]:
    """Run the schema/protocol rules over a run. Every violation cites a field."""
    found: dict[str, list[m.RuleViolation]] = {}

    def add(addr: str, rule: m.SchemaRule, severity: m.Severity, message: str, pointer: str, **ev):
        found.setdefault(addr, []).append(
            m.RuleViolation(
                rule=rule,
                severity=severity,
                message=message,
                citation=m.Citation(addr=addr, json_pointer=pointer),
                evidence=ev or None,
            )
        )

    dated = [(step, _as_of(step.output)) for step in steps if step.row["kind"] == "tool"]
    newest = max((when for _, when in dated if when), default=None)
    seen_keys: dict[str, str] = {}
    for step in steps:
        addr, kind = step.row["addr"], step.row["kind"]
        when = _as_of(step.output) if kind == "tool" else None
        if when and newest and (newest - when).days > STALE_AFTER_DAYS:
            age = (newest - when).days
            add(
                addr,
                "stale_as_of",
                "error",
                f"`as_of` is {age} days older than the freshest data in this run.",
                "/output/as_of",
                as_of=when.isoformat(),
                reference_as_of=newest.isoformat(),
                age_days=age,
            )
        out = step.output
        if kind in {"tool", "retrieval"}:
            if isinstance(out, dict) and ("error" in out or out.get("status") in range(400, 600)):
                add(
                    addr,
                    "error_status_in_output",
                    "error",
                    "The tool returned an error payload instead of data.",
                    "/output",
                    error=out.get("error"),
                    status=out.get("status"),
                )
            elif out == [] or (isinstance(out, dict) and out.get("options") == []):
                add(addr, "empty_result", "warning", "The call returned no results.", "/output")
        key = step.row.get("request_key")
        if key and kind in {"tool", "retrieval"}:
            if key in seen_keys:
                add(
                    addr,
                    "repeated_call",
                    "warning",
                    f"Identical request already made by {seen_keys[key]}.",
                    "/input",
                    first_addr=seen_keys[key],
                )
            seen_keys.setdefault(key, addr)
    return found


# ---------------------------------------------------------------------------
# Reading a recorder
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RunMeta:
    """What the fixture builder knows about a run beyond what the recorder stored."""

    task: str
    task_text: str | None
    origin: m.RunOrigin
    split: m.Split | None
    started_at: datetime
    risk: float | None
    top_suspect: m.SuspectBrief | None
    failure_signature: str | None
    final_answer_key: str
    label_confidence: Literal["high", "low"] | None = None


def _parse(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp)


def display_name(row: Mapping[str, Any], request: Any) -> str:
    """Tool steps show the tool function; llm and state steps show '<role> chat/state'."""
    if row["kind"] in {"tool", "retrieval"} and isinstance(request, dict) and request.get("name"):
        return str(request["name"])
    if row["kind"] == "llm" and row["agent_role"]:
        return f"{row['agent_role']} chat"
    if row["kind"] == "state" and row["agent_role"]:
        return f"{row['agent_role']} state"
    return str(row["name"])


class RunReader:
    """Adapt one recorder's SQLite rows and blobs to the contract models."""

    def __init__(self, recorder: Recorder, fault_specs: Mapping[str, Any]) -> None:
        self.recorder = recorder
        self.db = recorder.database
        self.store = recorder.store
        self.fault_specs = fault_specs

    def run_row(self, run_id: str) -> dict[str, Any]:
        row = self.db.one("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        if row is None:
            raise KeyError(run_id)
        return row

    def raw_steps(self, run_id: str) -> list[RawStep]:
        rows = self.db.query("SELECT * FROM steps WHERE run_id = ? ORDER BY seq", (run_id,))
        steps = []
        for row in rows:
            reasoning = self.store.load_json(row["reasoning_hash"]) if row["reasoning_hash"] else []
            steps.append(
                RawStep(
                    row=row,
                    input=self.store.load_json(row["input_hash"]) if row["input_hash"] else None,
                    output=self.store.load_json(row["output_hash"]) if row["output_hash"] else None,
                    reasoning=[str(item) for item in reasoning],
                    state_before=self.store.load_checkpoint(row["state_before"]),
                    state_after=self.store.load_checkpoint(row["state_after"]),
                )
            )
        return steps

    def edges(self, run_id: str) -> list[m.Edge]:
        rows = self.db.query("SELECT * FROM edges WHERE run_id = ? ORDER BY edge_id", (run_id,))
        return [
            m.Edge(
                src_addr=row["src_addr"],
                src_pointer=row["src_pointer"],
                dst_addr=row["dst_addr"],
                dst_pointer=row["dst_pointer"],
                kind=row["kind"],
                key=row["key"],
                version=row["version"],
                value_hash=row["value_hash"],
            )
            for row in rows
        ]

    def label(self, run_id: str, confidence: str | None) -> m.LabelInfo | None:
        row = self.db.one("SELECT * FROM labels WHERE run_id = ?", (run_id,))
        if row is None:
            return None
        spec = self.fault_specs.get(row["fault_type"])
        return m.LabelInfo(
            source=row["source"],
            root_addr=row["root_addr"],
            fault_code=spec.code if spec else None,
            fault_type=row["fault_type"],
            fault_family=spec.family if spec else None,
            held_out=spec.held_out if spec else None,
            recovered=bool(row["recovered"]),
            manifest_addr=row["manifest_addr"],
            verified=bool(row["verified"]),
            confidence=confidence if row["source"] == "injected" else None,  # type: ignore[arg-type]
        )

    def step_details(
        self, run_id: str, status_override: Mapping[str, m.CacheStatus] | None = None
    ) -> list[m.StepDetail]:
        raw = self.raw_steps(run_id)
        run = self.run_row(run_id)
        violations = evaluate_rules(raw)
        details = []
        for index, step in enumerate(raw):
            row = step.row
            addr = row["addr"]
            error = None
            if row["error_type"]:
                is_last = index == len(raw) - 1
                error = m.StepError(
                    type=row["error_type"],
                    message=run["checker_reason"]
                    if is_last and run["outcome"] == "failed"
                    else None,
                    recovered=not is_last,
                )
            tokens = None
            if row["tokens_in"] is not None or row["tokens_out"] is not None:
                tokens = m.TokenUsage(
                    input=row["tokens_in"] or 0,
                    output=row["tokens_out"] or 0,
                    cached=row["tokens_cached"] or 0,
                )
            details.append(
                m.StepDetail(
                    addr=addr,
                    seq=row["seq"],
                    kind=row["kind"],
                    name=display_name(row, step.input),
                    agent_role=row["agent_role"],
                    input=step.input,
                    output=step.output,
                    reasoning=step.reasoning,
                    state_before=step.state_before,
                    state_after=step.state_after,
                    reads=json.loads(row["reads_json"]),
                    writes=json.loads(row["writes_json"]),
                    has_call=row["request_key"] is not None,
                    cache_status=(status_override or {}).get(addr, row["cache_status"]),
                    tokens=tokens,
                    latency_ms=round(row["latency_ms"], 3),
                    finish_reason=row["finish_reason"],
                    retries=row["retries"],
                    error=error,
                    rule_violations=violations.get(addr, []),
                    hashes=m.StepHashes(
                        request_key=row["request_key"],
                        input=row["input_hash"],
                        output=row["output_hash"],
                        reasoning=row["reasoning_hash"],
                        state_before=row["state_before"],
                        state_after=row["state_after"],
                    ),
                )
            )
        return details

    def summary(
        self, run_id: str, meta: RunMeta, steps: Sequence[m.StepDetail], label_hidden: bool = False
    ) -> m.RunSummary:
        run = self.run_row(run_id)
        started = _parse(run["started_at"])
        ended = _parse(run["ended_at"]) if run["ended_at"] else None
        duration = (ended - started).total_seconds() * 1000 if ended else 0.0
        tokens = [step.tokens for step in steps if step.tokens]
        return m.RunSummary(
            run_id=run_id,
            status={"passed": "passed", "failed": "failed"}.get(run["outcome"], "running"),
            agent=run["agent"],
            task_id=run["task_id"],
            task=meta.task,
            origin=meta.origin,
            parent_run_id=run["parent_run_id"],
            fork_id=run["fork_id"],
            mode=run["mode"],
            model=run["model"],
            client="deterministic_standin",
            seed=run["seed"],
            steps=len(steps),
            duration_ms=round(duration, 3),
            cost=m.Cost(
                llm_calls=sum(step.kind == "llm" and step.has_call for step in steps),
                tokens_in=sum(t.input for t in tokens),
                tokens_out=sum(t.output for t in tokens),
                tokens_cached=sum(t.cached for t in tokens),
                usd=None,
            ),
            started_at=meta.started_at,
            ended_at=meta.started_at + timedelta(milliseconds=duration) if ended else None,
            score=run["score"],
            checker_reason=run["checker_reason"],
            split=meta.split,
            risk=meta.risk,
            top_suspect=meta.top_suspect,
            failure_signature=meta.failure_signature,
            label=None if label_hidden else self.label(run_id, meta.label_confidence),
        )

    def detail(
        self,
        run_id: str,
        meta: RunMeta,
        *,
        status_override: Mapping[str, m.CacheStatus] | None = None,
        fixture: bool,
    ) -> m.RunDetail:
        steps = self.step_details(run_id, status_override)
        final = steps[-1].state_after.get(meta.final_answer_key) if steps else None
        return m.RunDetail(
            run=self.summary(run_id, meta, steps),
            task_text=meta.task_text,
            steps=steps,
            edges=self.edges(run_id),
            final_answer=final,
            fixture=fixture,
        )


# ---------------------------------------------------------------------------
# Graph derivations over contract models
# ---------------------------------------------------------------------------


def descendants(edges: Iterable[m.Edge], roots: Iterable[str]) -> set[str]:
    adjacency: dict[str, set[str]] = {}
    for edge in edges:
        adjacency.setdefault(edge.src_addr, set()).add(edge.dst_addr)
    seen = set(roots)
    queue = deque(seen)
    while queue:
        for nxt in adjacency.get(queue.popleft(), ()):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen


def normalise_statuses(
    steps: Sequence[m.StepDetail], invalidated: set[str], edited: set[str]
) -> dict[str, m.CacheStatus]:
    """Contract semantics for a fork run's cache status.

    The replay engine reports `live` for any step without a recorded call. A pure state step
    outside the dependency cone is not re-executed in any meaningful sense, so it reports
    `cached`; inside the cone it re-evaluates and reports `live`.
    """
    result: dict[str, m.CacheStatus] = {}
    for step in steps:
        status = step.cache_status
        if not step.has_call and step.addr not in invalidated:
            status = "cached"
        if step.addr in edited and step.has_call and status == "live":
            status = "edited"
        result[step.addr] = status
    return result


def damage_path(
    steps: Sequence[m.StepDetail],
    edges: Sequence[m.Edge],
    root_addr: str,
    visible_failure_addr: str | None,
) -> m.DamagePath:
    cone = descendants(edges, [root_addr])
    nodes = [
        m.DamageNode(
            addr=step.addr,
            name=step.name,
            kind=step.kind,
            tag="root"
            if step.addr == root_addr
            else "symptom"
            if step.addr in cone
            else "unaffected",
            is_visible_failure=step.addr == visible_failure_addr,
        )
        for step in steps
    ]
    chosen: dict[tuple[str, str], m.Edge] = {}
    for edge in edges:
        if edge.src_addr in cone and edge.dst_addr in cone:
            key = (edge.src_addr, edge.dst_addr)
            if key not in chosen or (edge.kind == "state" and chosen[key].kind != "state"):
                chosen[key] = edge
    last = max(steps, key=lambda step: step.seq).addr
    return m.DamagePath(
        root_addr=root_addr,
        nodes=nodes,
        edges=list(chosen.values()),
        reaches_final_answer=last in cone,
    )


def _is_produced(step: m.StepDetail, pointer: str) -> bool:
    """True when the step itself produced the value at `pointer` (as opposed to consuming it)."""
    if pointer.startswith("/output"):
        return True
    if pointer.startswith("/state/"):
        key = unescape(pointer.split("/")[2])
        return any(write.split("@")[0] == key for write in step.writes)
    return False


def _carries(edge: m.Edge, pointer: str) -> bool:
    """Whether an outgoing edge can carry the produced value at `pointer`."""
    source = edge.src_pointer
    if source is None:
        return True
    related = _is_prefix(source, pointer) or _is_prefix(pointer, source)
    if pointer.startswith("/output"):
        # A state edge forwards the step's whole output, so it carries any output field.
        return related or source.startswith("/state/")
    return related if source.startswith("/state/") else False


def provenance(detail: m.RunDetail, addr: str, pointer: str) -> m.ValueProvenance:
    """Producer candidates and the forward consumer path for one clicked value."""
    steps = {step.addr: step for step in detail.steps}
    target_step = steps[addr]
    produced = _is_produced(target_step, pointer)
    found, value = resolve_pointer(
        step_document(target_step, "src" if produced else "dst"), pointer
    )
    if not found:
        raise KeyError(f"{addr} has no value at {pointer}")
    candidates: list[m.ProducerCandidate] = []
    for edge in detail.edges:
        if produced or edge.dst_addr != addr or edge.dst_pointer is None:
            continue
        if not _is_prefix(edge.dst_pointer, pointer):
            continue
        suffix = pointer[len(edge.dst_pointer) :]
        candidates.append(
            m.ProducerCandidate(
                addr=edge.src_addr,
                name=steps[edge.src_addr].name,
                json_pointer=(edge.src_pointer or "") + suffix if edge.src_pointer else None,
                edge_kind=edge.kind,
                key=edge.key,
                version=edge.version,
                value_hash=edge.value_hash,
            )
        )
    order = {"state": 0, "message": 1, "inferred": 2}
    candidates.sort(key=lambda c: (order[c.edge_kind], c.addr, c.json_pointer or ""))

    outgoing: dict[str, list[m.Edge]] = {}
    for edge in detail.edges:
        outgoing.setdefault(edge.src_addr, []).append(edge)
    hops: list[m.ConsumerHop] = []
    visited = {addr}
    queue: deque[tuple[str, int]] = deque([(addr, 0)])
    while queue:
        current, depth = queue.popleft()
        if depth >= MAX_PROVENANCE_DEPTH:
            continue
        for edge in outgoing.get(current, []):
            if produced and current == addr and not _carries(edge, pointer):
                continue
            if edge.dst_addr in visited:
                continue
            visited.add(edge.dst_addr)
            hops.append(m.ConsumerHop(**edge.model_dump(), depth=depth + 1))
            queue.append((edge.dst_addr, depth + 1))
    final = max(detail.steps, key=lambda step: step.seq).addr
    return m.ValueProvenance(
        run_id=detail.run.run_id,
        target=m.ValueRef(
            addr=addr,
            json_pointer=pointer,
            value=value,
            value_hash=content_hash(value) if value is not None else None,
        ),
        producer=candidates[0] if len(candidates) == 1 else None,
        candidates=candidates,
        ambiguous=len(candidates) > 1,
        consumers=hops,
        reaches_final_answer=final in visited,
    )


# ---------------------------------------------------------------------------
# Replay prediction
# ---------------------------------------------------------------------------


def format_duration(ms: float) -> str:
    return f"{ms:.0f} ms" if ms < 1000 else f"{ms / 1000:.1f} s"


def predict_replay(
    detail: m.RunDetail, request: m.ForkRequest, mode: m.ReplayMode | None = None
) -> m.ReplayPrediction:
    mode = mode or request.mode
    steps = sorted(detail.steps, key=lambda step: step.seq)
    edited = {edit.addr for edit in request.edits} or ({request.resample_from} - {None})
    cone = descendants(detail.edges, edited)
    first_seq = min((step.seq for step in steps if step.addr in edited), default=0)
    expected: dict[str, m.CacheStatus] = {}
    for step in steps:
        if step.addr in edited and request.edits:
            status: m.CacheStatus = "edited"
        elif mode == "full" or (mode == "prefix" and step.seq >= first_seq):
            status = "invalidated"
        elif mode == "cone" and step.addr in cone:
            status = "invalidated"
        else:
            status = "cached"
        expected[step.addr] = status
    rerun = [step for step in steps if expected[step.addr] != "cached"]
    runs = request.samples * (2 if request.control else 1)
    per_run_ms = sum(step.latency_ms for step in rerun)
    per_run_tokens = sum(
        (step.tokens.input + step.tokens.output) for step in rerun if step.tokens is not None
    )
    calls_total = sum(step.has_call for step in steps)
    calls_rerun = sum(step.has_call for step in rerun)
    estimated_ms = per_run_ms * runs
    return m.ReplayPrediction(
        base_run_id=detail.run.run_id,
        mode=mode,
        samples=request.samples,
        total_steps=len(steps),
        reexecute_steps=len(rerun),
        total_calls=calls_total,
        reexecute_calls=calls_rerun,
        per_step=[
            m.StepPrediction(addr=step.addr, expected=expected[step.addr], has_call=step.has_call)
            for step in steps
        ],
        estimated_ms=round(estimated_ms, 3),
        estimated_tokens=per_run_tokens * runs,
        upper_bound=mode == "cone",
        summary=(
            f"Will re-run {calls_rerun} of {calls_total} calls ({len(rerun)} of {len(steps)} "
            f"steps), about {format_duration(estimated_ms)}."
        ),
        blocked=None,
    )


# ---------------------------------------------------------------------------
# Trace comparison
# ---------------------------------------------------------------------------


def _final_state(detail: m.RunDetail) -> dict[str, Any]:
    return max(detail.steps, key=lambda step: step.seq).state_after if detail.steps else {}


def _own_state_changed(a: m.StepDetail, b: m.StepDetail) -> bool:
    """Whether the keys this step wrote differ. Later steps inherit earlier changes in the
    global state, so comparing whole snapshots would flag every step downstream of an edit."""
    keys = {write.split("@")[0] for write in [*a.writes, *b.writes]}
    return any(
        content_hash(a.state_after.get(k)) != content_hash(b.state_after.get(k)) for k in keys
    )


def build_diff(
    left: m.RunDetail,
    right: m.RunDetail,
    *,
    invalidated: set[str],
    nearest: m.NearestPassingLink | None,
) -> m.DiffResponse:
    lefts = {step.addr: step for step in left.steps}
    rights = {step.addr: step for step in right.steps}
    order = [step.addr for step in sorted(left.steps, key=lambda s: s.seq)]
    order += [
        step.addr for step in sorted(right.steps, key=lambda s: s.seq) if step.addr not in lefts
    ]

    def side(step: m.StepDetail, *, payload: bool) -> m.DiffSide:
        return m.DiffSide(
            seq=step.seq,
            kind=step.kind,
            name=step.name,
            cache_status=step.cache_status,
            input=step.input if payload else None,
            output=step.output if payload else None,
            input_hash=step.hashes.input,
            output_hash=step.hashes.output,
            state_after_hash=step.hashes.state_after,
        )

    rows: list[m.DiffRow] = []
    for addr in order:
        a, b = lefts.get(addr), rights.get(addr)
        if a is None or b is None:
            status: m.DiffStatus = "new" if a is None else "removed"
            changed: list[Any] = []
        else:
            changed = [
                name
                for name, differs in (
                    ("input", a.hashes.input != b.hashes.input),
                    ("output", a.hashes.output != b.hashes.output),
                    ("state", _own_state_changed(a, b)),
                )
                if differs
            ]
            status = "cached" if b.cache_status == "cached" else "changed" if changed else "same"
        payload = status in {"changed", "new", "removed"}
        rows.append(
            m.DiffRow(
                addr=addr,
                status=status,
                in_cone=addr in invalidated,
                changed=changed,
                left=side(a, payload=payload) if a else None,
                right=side(b, payload=payload) if b else None,
            )
        )

    stats = {status: 0 for status in get_args(m.DiffStatus)}
    for row in rows:
        stats[row.status] += 1
    first = next((row.addr for row in rows if row.status in {"changed", "new", "removed"}), None)

    left_state, right_state = _final_state(left), _final_state(right)
    state_diff = []
    for key in sorted(set(left_state) | set(right_state)):
        in_left, in_right = key in left_state, key in right_state
        lv, rv = left_state.get(key), right_state.get(key)
        status_key: m.StateKeyStatus = (
            "added"
            if not in_left
            else "removed"
            if not in_right
            else "unchanged"
            if content_hash(lv) == content_hash(rv)
            else "changed"
        )
        state_diff.append(
            m.StateKeyDiff(
                key=key,
                status=status_key,
                left=lv if in_left else None,
                right=rv if in_right else None,
                left_hash=content_hash(lv) if in_left else None,
                right_hash=content_hash(rv) if in_right else None,
            )
        )

    def brief(run: m.RunSummary) -> m.OutcomeBrief:
        return m.OutcomeBrief(status=run.status, score=run.score, reason=run.checker_reason)

    flipped = left.run.status != right.run.status
    direction = (
        "fail_to_pass"
        if left.run.status == "failed" and right.run.status == "passed"
        else "pass_to_fail"
        if left.run.status == "passed" and right.run.status == "failed"
        else "unchanged"
    )
    return m.DiffResponse(
        left_run_id=left.run.run_id,
        right_run_id=right.run.run_id,
        alignment="address",
        first_divergence=first,
        stats=m.DiffStats(**stats),
        rows=rows,
        state_diff=state_diff,
        outcome=m.OutcomeDiff(
            left=brief(left.run), right=brief(right.run), flipped=flipped, direction=direction
        ),
        nearest_passing=nearest,
    )
