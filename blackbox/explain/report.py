"""Assemble a full diagnosis for one failed run: ranking, evidence, damage path, precedents.

The trained :class:`~blackbox.ml.model.Diagnoser` decides the ranking. Everything else here
is deterministic bookkeeping over the recorded trace, so each claim can be traced back to
a step and a field.
"""

from __future__ import annotations

import json
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from blackbox.explain.evidence import Finding, step_findings
from blackbox.explain.reasons import Reason, contributions, top_reasons
from blackbox.ml.dataset import Corpus, Trace
from blackbox.ml.features import FEATURE_NAMES, output_payload
from blackbox.ml.model import Diagnoser

EVALUATION_ROLES = frozenset({"checker", "grader", "judge"})
TOP_SUSPECTS = 5


def is_evaluation_step(addr: str, role: str | None = None) -> bool:
    """Steps that grade a run rather than belong to the agent are never suspects."""
    return (role or addr.split("/", 1)[0]).lower() in EVALUATION_ROLES


@dataclass(slots=True)
class Suspect:
    addr: str
    probability: float
    rank: int
    reasons: list[Reason] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "addr": self.addr,
            "probability": round(self.probability, 4),
            "rank": self.rank,
            "reasons": [r.as_dict() for r in self.reasons],
            "findings": [f.as_dict() for f in self.findings],
        }


@dataclass(slots=True)
class DiagnosisReport:
    run_id: str
    agent: str
    task_id: str
    model_version: str
    ranking: list[tuple[str, float]]
    suspects: list[Suspect]
    conformal_set: list[str]
    abstain: bool
    abstain_reason: str | None
    visible_failure: str | None
    damage_path: list[dict[str, str]]
    precedents: dict[str, Any] | None
    twin_run_id: str | None
    findings_by_step: dict[str, list[Finding]]

    @property
    def top(self) -> Suspect | None:
        return self.suspects[0] if self.suspects else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "agent": self.agent,
            "task_id": self.task_id,
            "model_version": self.model_version,
            "ranking": [{"addr": a, "probability": round(p, 4)} for a, p in self.ranking],
            "suspects": [s.as_dict() for s in self.suspects],
            "conformal_set": self.conformal_set,
            "abstain": self.abstain,
            "abstain_reason": self.abstain_reason,
            "visible_failure": self.visible_failure,
            "damage_path": self.damage_path,
            "precedents": self.precedents,
            "twin_run_id": self.twin_run_id,
            "findings_by_step": {
                addr: [f.as_dict() for f in findings]
                for addr, findings in self.findings_by_step.items()
                if findings
            },
        }


def visible_failure(trace: Trace) -> str | None:
    """Where the failure became visible: the first crash or error, else the final step.

    This is a symptom marker, deliberately separate from the blamed root cause.
    """
    agent_steps = [
        s
        for s in sorted(trace.steps, key=lambda s: s.seq)
        if not is_evaluation_step(s.addr, s.role)
    ]
    for step in agent_steps:
        payload, parsed = output_payload(step)
        if step.error_type or not parsed or (isinstance(payload, dict) and ("error" in payload)):
            return step.addr
    return agent_steps[-1].addr if agent_steps else None


def damage_path(trace: Trace, root: str, failure: str | None) -> list[dict[str, str]]:
    """Every step tagged root / symptom / unaffected, in execution order.

    Symptoms are the steps reachable from the root through recorded provenance; the
    shortest provenance path to the visible failure is marked ``on_path``.
    """
    children: dict[str, set[str]] = {}
    for src, dst, _kind in trace.edges:
        children.setdefault(src, set()).add(dst)
    reachable: set[str] = set()
    parent: dict[str, str] = {}
    queue = deque([root])
    while queue:
        node = queue.popleft()
        for child in sorted(children.get(node, ())):
            if child not in reachable and child != root:
                reachable.add(child)
                parent[child] = node
                queue.append(child)
    on_path: set[str] = {root}
    if failure and failure in reachable:
        node = failure
        while node != root:
            on_path.add(node)
            node = parent[node]
    result = []
    for step in sorted(trace.steps, key=lambda s: s.seq):
        if is_evaluation_step(step.addr, step.role):
            continue
        tag = "root" if step.addr == root else "symptom" if step.addr in reachable else "unaffected"
        result.append(
            {"addr": step.addr, "tag": tag, "on_path": "true" if step.addr in on_path else "false"}
        )
    return result


class PrecedentLibrary:
    """Root steps of labelled training failures, for "seen this before" lookups."""

    def __init__(
        self, rows: np.ndarray, meta: list[dict[str, Any]], scale: np.ndarray, center: np.ndarray
    ) -> None:
        self.rows = rows
        self.meta = meta
        self.scale = scale
        self.center = center

    @classmethod
    def build(cls, diagnoser: Diagnoser, corpus: Corpus, run_ids: list[str]) -> PrecedentLibrary:
        traces = [corpus.traces[r] for r in run_ids if r in corpus.traces and r in corpus.labels]
        matrix = diagnoser.matrix(traces)
        rows, meta = [], []
        for index, trace in enumerate(traces):
            label = corpus.labels[trace.run_id]
            addrs = matrix.addrs[index]
            if label.root_addr not in addrs:
                continue
            row = matrix.X[matrix.run_slice(index)][addrs.index(label.root_addr)]
            rows.append(row)
            meta.append(
                {
                    "run_id": trace.run_id,
                    "agent": trace.agent,
                    "addr": label.root_addr,
                    "fault_type": label.fault_type,
                }
            )
        data = (
            np.nan_to_num(np.asarray(rows, dtype=float))
            if rows
            else np.zeros((0, len(FEATURE_NAMES)))
        )
        center = data.mean(axis=0) if len(data) else np.zeros(len(FEATURE_NAMES))
        scale = data.std(axis=0) if len(data) else np.ones(len(FEATURE_NAMES))
        scale[scale == 0] = 1.0
        return cls((data - center) / scale, meta, scale, center)

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "rows": self.rows.round(5).tolist(),
                    "meta": self.meta,
                    "scale": self.scale.tolist(),
                    "center": self.center.tolist(),
                }
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> PrecedentLibrary | None:
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            np.asarray(data["rows"], dtype=float),
            data["meta"],
            np.asarray(data["scale"], dtype=float),
            np.asarray(data["center"], dtype=float),
        )

    def lookup(
        self, row: np.ndarray, *, k: int = 12, exclude: str | None = None
    ) -> dict[str, Any] | None:
        if len(self.rows) == 0:
            return None
        query = (np.nan_to_num(row) - self.center) / self.scale
        norms = np.linalg.norm(self.rows, axis=1) * (np.linalg.norm(query) or 1.0)
        norms[norms == 0] = 1.0
        similarity = self.rows @ query / norms
        order = [i for i in np.argsort(-similarity) if self.meta[i]["run_id"] != exclude][:k]
        if not order:
            return None
        counts = Counter(self.meta[i]["fault_type"] for i in order)
        fault, count = counts.most_common(1)[0]
        return {
            "k": len(order),
            "dominant_fault_type": fault,
            "dominant_count": count,
            "fault_type_counts": dict(counts.most_common()),
            "neighbours": [
                {**self.meta[i], "similarity": round(float(similarity[i]), 4)} for i in order[:5]
            ],
        }


def nearest_passing_twin(
    trace: Trace, corpus: Corpus | None, base_run_id: str | None = None
) -> str | None:
    """The healthy run to compare against: the fork's own passing base run when there is
    one, else the passing run of the same agent with the most similar step layout."""
    if corpus is None:
        return base_run_id
    if base_run_id and base_run_id in corpus.passing:
        return base_run_id
    addrs = set(trace.addrs)
    best: tuple[float, str] | None = None
    for run_id in corpus.passing:
        other = corpus.traces.get(run_id)
        if other is None or other.agent != trace.agent or run_id == trace.run_id:
            continue
        overlap = len(addrs & set(other.addrs)) / max(len(addrs | set(other.addrs)), 1)
        score = overlap + (0.5 if other.task_id == trace.task_id else 0.0)
        if best is None or score > best[0] or (score == best[0] and run_id < best[1]):
            best = (score, run_id)
    return best[1] if best else None


def build_report(
    diagnoser: Diagnoser,
    trace: Trace,
    *,
    precedents: PrecedentLibrary | None = None,
    corpus: Corpus | None = None,
    base_run_id: str | None = None,
) -> DiagnosisReport:
    matrix = diagnoser.matrix([trace])
    diagnosis = diagnoser.diagnose_many([trace])[0] if matrix.X.shape[0] else None
    roles = {s.addr: s.role for s in trace.steps}
    agent_ranking = [
        (addr, p)
        for addr, p in (diagnosis.ranking if diagnosis else [])
        if not is_evaluation_step(addr, roles.get(addr))
    ]
    mass = sum(p for _, p in agent_ranking) or 1.0
    ranking = [(addr, p / mass) for addr, p in agent_ranking]
    conformal = [
        a
        for a in (diagnosis.conformal_set if diagnosis else [])
        if not is_evaluation_step(a, roles.get(a))
    ]
    if not conformal and ranking:
        conformal = [ranking[0][0]]

    findings = {step.addr: step_findings(trace, step, diagnoser.reference) for step in trace.steps}
    contrib = (
        contributions(diagnoser.booster, matrix, 0, diagnoser.best_iteration)
        if diagnosis
        else np.zeros((0, len(FEATURE_NAMES) + 1))
    )
    addr_index = {addr: i for i, addr in enumerate(matrix.addrs[0])} if matrix.addrs else {}
    rows = matrix.X[matrix.run_slice(0)] if matrix.addrs else np.zeros((0, len(FEATURE_NAMES)))

    suspects = []
    for rank, (addr, probability) in enumerate(ranking[:TOP_SUSPECTS], start=1):
        i = addr_index[addr]
        reasons = top_reasons(contrib[i], rows[i], findings.get(addr, []))
        suspects.append(Suspect(addr, probability, rank, reasons, findings.get(addr, [])[:6]))

    abstain_reason = None
    abstain = bool(diagnosis.abstain) if diagnosis else True
    if diagnosis and len(conformal) > diagnoser.config.max_set:
        abstain = True
    if abstain:
        abstain_reason = (
            f"No confident culprit: the calibrated set needs {len(conformal)} steps "
            f"(more than {diagnoser.config.max_set}) to reach "
            f"{round((1 - diagnoser.config.alpha) * 100)}% coverage."
        )
    failure = visible_failure(trace)
    root = ranking[0][0] if ranking else None
    precedent = None
    if precedents is not None and root is not None:
        precedent = precedents.lookup(rows[addr_index[root]], exclude=trace.run_id)
    return DiagnosisReport(
        run_id=trace.run_id,
        agent=trace.agent,
        task_id=trace.task_id,
        model_version=diagnosis.model_version if diagnosis else "unavailable",
        ranking=ranking,
        suspects=suspects,
        conformal_set=conformal,
        abstain=abstain,
        abstain_reason=abstain_reason,
        visible_failure=failure,
        damage_path=damage_path(trace, root, failure) if root else [],
        precedents=precedent,
        twin_run_id=nearest_passing_twin(trace, corpus, base_run_id),
        findings_by_step=findings,
    )
