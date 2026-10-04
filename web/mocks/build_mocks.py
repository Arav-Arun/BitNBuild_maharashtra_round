"""Regenerate the web/mocks fixtures from freshly recorded offline runs.

    uv run --locked --extra dev --extra ml --extra server python -m web.mocks.build_mocks

Recorded from the deterministic stand-ins (never a language model): real step addresses,
hashes, state, edges, replay statuses, Wilson intervals and verdicts. Authored: model outputs
(rankings, precedents, evaluation), always flagged `fixture: true`. Timestamps are moved onto a
fixed clock and fork ids are renamed, so only measured latencies differ between rebuilds.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from agents.hoprag.data import load_examples
from blackbox.forge.operators import all_operators, held_out_operators
from blackbox.recorder import canonical_json
from server import models as m
from web.mocks import authored
from web.mocks import extract as x
from web.mocks.recording import HOPRAG_DATASET, TRIPCREW_MODEL, ForkRecord, World, build_world

MOCKS = Path(__file__).resolve().parent
BASE_TIME = datetime(2026, 10, 4, 9, 0, tzinfo=UTC)
DEMO_RUN = "tc-0001"
TWIN_RUN = "tc-0007"
MD_NAME = "crash-report.md"
ABSTAIN_RUN = "hr-0003"
HOPRAG_BASE = [
    "hr-0001",
    "hr-0002",
    "hr-0003",
    "hr-0004",
    "hr-0034",
    "hr-0044",
    "hr-0007",
    "hr-0010",
]


@dataclass(slots=True)
class Entry:
    """One row of the Runs list."""

    run_id: str
    agent: str
    origin: m.RunOrigin


def indian_floor(value: float) -> str:
    return x.indian_money(math.floor(value))


class Assembler:
    def __init__(self, world: World, dataset: Path | None) -> None:
        self.world = world
        specs = {op.spec.name: op.spec for op in [*all_operators(), *held_out_operators()]}
        self.readers = {
            "tripcrew": x.RunReader(world.tripcrew, specs),
            "hoprag": x.RunReader(world.hoprag, specs),
        }
        self.questions = {
            e.question.question_id: e.question.text
            for e in (load_examples(dataset) if dataset and dataset.exists() else [])
        }
        self.ids: dict[str, str] = {}
        for number, run_id in enumerate(world.injected, 1):
            self.ids[run_id.removesuffix("-fix-0")] = f"ij-{number:04d}"
        for number, name in enumerate(world.demo, 1):
            self.ids[world.demo[name].fork_id] = f"fk-{number:04d}"
        self.entries = self._entries()
        self.clock = {
            entry.run_id: BASE_TIME + timedelta(minutes=11 * (len(self.entries) - index))
            for index, entry in enumerate(self.entries)
        }
        self._steps: dict[str, list[m.StepDetail]] = {}
        self.files: dict[str, str] = {}
        self.provenance_notes: dict[str, str] = {}

    # -- run selection -----------------------------------------------------------------

    def _entries(self) -> list[Entry]:
        w = self.world
        tripcrew = [Entry(w.tripcrew_ids[n], "tripcrew", "natural") for n in sorted(w.tripcrew_ids)]
        injected = [
            Entry(run_id, "tripcrew" if run_id in self._tc_ids() else "hoprag", "injected")
            for run_id in w.injected
        ]
        hoprag = [
            Entry(run_id, "hoprag", "natural") for run_id in HOPRAG_BASE if run_id in w.hoprag_ids
        ]
        merged: list[Entry] = []
        queues = [tripcrew, injected, hoprag]
        while any(queues):
            for queue in queues:
                if queue:
                    merged.append(queue.pop(0))
        return merged

    def _tc_ids(self) -> set[str]:
        return {
            row["run_id"] for row in self.world.tripcrew.database.query("SELECT run_id FROM runs")
        }

    def reader(self, run_id: str) -> x.RunReader:
        return self.readers["tripcrew" if run_id in self._tc_ids() else "hoprag"]

    def steps(self, run_id: str) -> list[m.StepDetail]:
        if run_id not in self._steps:
            self._steps[run_id] = self.reader(run_id).step_details(run_id)
        return self._steps[run_id]

    def when(self, run_id: str) -> datetime:
        return self.clock.get(run_id, BASE_TIME + timedelta(hours=3))

    # -- authored model fields for the list ------------------------------------------------

    def task(self, run_id: str) -> tuple[str, str | None, str]:
        row = self.reader(run_id).run_row(run_id)
        if row["agent"] == "tripcrew":
            scenario = self.world.scenarios[row["task_id"]]
            text = f"{scenario.origin} → {scenario.destination} under {indian_floor(scenario.budget_inr)}"
            return text, scenario.request, "final_plan"
        question = self.questions.get(row["task_id"], row["task_id"])
        short = question if len(question) <= 88 else question[:85].rstrip() + "..."
        return short, question, "final_answer"

    def model_fields(
        self, entry: Entry, slot: int, label: m.LabelInfo | None
    ) -> tuple[float | None, m.SuspectBrief | None, str | None, m.Split | None]:
        row = self.reader(entry.run_id).run_row(entry.run_id)
        steps = {step.addr: step for step in self.steps(entry.run_id)}
        kind = x.reason_class(row["checker_reason"])
        split: m.Split | None = None
        if entry.origin == "injected" and label:
            split = "S3" if entry.agent == "hoprag" else "S1" if label.held_out else "S0"
        elif entry.origin == "natural" and row["outcome"] == "failed":
            split = "S4"
        if row["outcome"] == "passed":
            risk = 0.21 if label and label.recovered else round(0.03 + 0.01 * (slot % 7), 2)
            return risk, None, None, split
        if label is None:
            return 0.72, None, f"{entry.agent}:unlocalised:{kind}", split
        if entry.run_id == ABSTAIN_RUN:
            return 0.88, None, f"{entry.agent}:unlocalised:{kind}", split
        addr = label.root_addr
        if slot % 5 == 4 and label.manifest_addr in steps:
            addr = label.manifest_addr  # the model blames the symptom, not the cause
        probability = round(0.86 - 0.04 * (slot % 9), 2)
        if entry.run_id == DEMO_RUN:
            probability = 0.82
        suspect = m.SuspectBrief(addr=addr, name=steps[addr].name, probability=probability)
        return (
            round(0.95 - 0.02 * (slot % 5), 2),
            suspect,
            f"{entry.agent}:{suspect.name}:{kind}",
            split,
        )

    def meta(self, entry: Entry, slot: int, label: m.LabelInfo | None) -> x.RunMeta:
        risk, suspect, signature, split = self.model_fields(entry, slot, label)
        task, text, answer_key = self.task(entry.run_id)
        return x.RunMeta(
            task=task,
            task_text=text,
            origin=entry.origin,
            split=split,
            started_at=self.when(entry.run_id),
            risk=risk,
            top_suspect=suspect,
            failure_signature=signature,
            final_answer_key=answer_key,
            label_confidence="low" if entry.origin == "injected" else None,
        )

    def summary(self, entry: Entry, slot: int) -> m.RunSummary:
        reader = self.reader(entry.run_id)
        label = reader.label(entry.run_id, "low")
        return reader.summary(entry.run_id, self.meta(entry, slot, label), self.steps(entry.run_id))

    def detail(self, entry: Entry, slot: int) -> m.RunDetail:
        reader = self.reader(entry.run_id)
        label = reader.label(entry.run_id, "low")
        return reader.detail(entry.run_id, self.meta(entry, slot, label), fixture=True)

    def entry(self, run_id: str) -> tuple[int, Entry]:
        for slot, entry in enumerate(self.entries):
            if entry.run_id == run_id:
                return slot, entry
        raise KeyError(run_id)

    # -- forks -------------------------------------------------------------------------

    def fork_run_ids(self, record: ForkRecord) -> tuple[list[str], list[str]]:
        return [r.run_id for r in record.batch.edited], [r.run_id for r in record.batch.controls]

    def fork_steps(self, record: ForkRecord, run_id: str) -> list[m.StepDetail]:
        edited = {edit.addr for edit in record.edits}
        reader = self.readers["tripcrew"]
        steps = reader.step_details(run_id)
        statuses = x.normalise_statuses(steps, set(record.batch.invalidated), edited)
        return reader.step_details(run_id, statuses)

    def savings(self, record: ForkRecord) -> m.Savings:
        steps = self.fork_steps(record, record.batch.edited[0].run_id)
        live = {"live", "edited"}
        return m.Savings(
            reexecuted=sum(s.cache_status in live for s in steps),
            cached=sum(s.cache_status == "cached" for s in steps),
            invalidated=len(record.batch.invalidated),
            total_steps=len(steps),
            calls_reexecuted=sum(s.has_call and s.cache_status in live for s in steps),
            calls_cached=sum(s.has_call and s.cache_status == "cached" for s in steps),
            tokens_saved=record.batch.tokens_saved,
            ms_saved=round(record.batch.ms_saved, 3),
        )

    def fidelity(self, record: ForkRecord) -> float:
        """Share of pre-edit steps whose replayed state hash equals the recording."""
        base = {s.addr: s for s in self.steps(record.batch.base_run_id)}
        first = min(base[edit.addr].seq for edit in record.edits)
        replayed = self.readers["tripcrew"].step_details(record.batch.edited[0].run_id)
        before = [s for s in replayed if s.seq < first]
        same = sum(base[s.addr].hashes.state_before == s.hashes.state_before for s in before)
        return same / len(before) if before else 1.0

    def fork_events(self, record: ForkRecord) -> list[m.ForkEvent]:
        """Engine events, grouped by run, with sample/branch, timings and a running phase added."""
        events: list[Any] = []
        pending: list[dict[str, Any]] = []
        counter = 0
        reader = self.readers["tripcrew"]
        edited = {edit.addr for edit in record.edits}
        invalidated = set(record.batch.invalidated)

        def next_id() -> int:
            nonlocal counter
            counter += 1
            return counter

        for raw in record.raw_events:
            data = raw["data"]
            if raw["event"] == "step":
                pending.append(data)
                continue
            if raw["event"] == "outcome":
                branch, sample = data["branch"], data["sample"]
                steps = {s.addr: s for s in reader.step_details(data["run_id"])}
                statuses = x.normalise_statuses(list(steps.values()), invalidated, edited)
                for step_data in pending:
                    step = steps[step_data["addr"]]
                    status = statuses[step_data["addr"]]
                    if step_data["phase"] == "queued":
                        events.append(
                            m.StepEvent(
                                id=next_id(),
                                event="step",
                                data=m.StepEventData(
                                    addr=step.addr,
                                    phase="queued",
                                    cache_status="invalidated",
                                    ms=None,
                                    tokens=None,
                                    sample=sample,
                                    branch=branch,
                                ),
                            )
                        )
                        continue
                    if status in {"live", "edited"}:
                        events.append(
                            m.StepEvent(
                                id=next_id(),
                                event="step",
                                data=m.StepEventData(
                                    addr=step.addr,
                                    phase="running",
                                    cache_status=None,
                                    ms=None,
                                    tokens=None,
                                    sample=sample,
                                    branch=branch,
                                ),
                            )
                        )
                    tokens = (step.tokens.input + step.tokens.output) if step.tokens else None
                    events.append(
                        m.StepEvent(
                            id=next_id(),
                            event="step",
                            data=m.StepEventData(
                                addr=step.addr,
                                phase="done",
                                cache_status=status,
                                ms=step.latency_ms,
                                tokens=0 if status == "cached" else tokens,
                                sample=sample,
                                branch=branch,
                            ),
                        )
                    )
                pending = []
                events.append(
                    m.OutcomeEvent(
                        id=next_id(),
                        event="outcome",
                        data=m.OutcomeEventData(
                            run_id=data["run_id"],
                            sample=sample,
                            branch=branch,
                            passed=data["passed"],
                            reason=data["reason"],
                            is_control=branch == "control",
                        ),
                    )
                )
        batch = record.batch
        savings = self.savings(record)
        events.append(
            m.SummaryEvent(
                id=next_id(),
                event="summary",
                data=m.SummaryEventData(
                    fork_id=record.fork_id,
                    reexecuted=savings.reexecuted,
                    cached=savings.cached,
                    invalidated=savings.invalidated,
                    total_steps=savings.total_steps,
                    calls_reexecuted=savings.calls_reexecuted,
                    calls_cached=savings.calls_cached,
                    tokens_saved=savings.tokens_saved,
                    ms_saved=savings.ms_saved,
                    fix_rate=batch.fix_pass_rate,
                    fix_ci=batch.fix_interval,
                    control_rate=batch.control_pass_rate,
                    control_ci=batch.control_interval,
                    k=len(batch.edited),
                    verdict=batch.verdict,
                    preview=not batch.controls,
                ),
            )
        )
        problems = m.check_event_stream(events)
        if problems:
            raise AssertionError(f"recorded stream violates the contract: {problems}")
        return events

    def run_refs(self, record: ForkRecord) -> tuple[list[m.ForkRunRef], list[m.ForkRunRef]]:
        def refs(runs: list[Any], branch: m.Branch) -> list[m.ForkRunRef]:
            return [
                m.ForkRunRef(
                    run_id=run.run_id,
                    sample=index,
                    branch=branch,
                    passed=run.outcome == "passed",
                    score=run.score,
                    reason=run.reason,
                )
                for index, run in enumerate(runs)
            ]

        return refs(record.batch.edited, "fix"), refs(record.batch.controls, "control")

    def fork_edit(self, edit: Any) -> m.ForkEdit:
        return m.ForkEdit(
            addr=edit.addr, kind=edit.kind, value=edit.value, known_good=edit.known_good
        )

    def fork_summary(self, name: str, record: ForkRecord) -> m.ForkSummary:
        batch = record.batch
        fix = m.pass_rate(sum(r.outcome == "passed" for r in batch.edited), len(batch.edited))
        control = (
            m.pass_rate(sum(r.outcome == "passed" for r in batch.controls), len(batch.controls))
            if batch.controls
            else None
        )
        number = list(self.world.demo).index(name)
        created = BASE_TIME + timedelta(hours=3, minutes=5 * number)
        edited_runs, control_runs = self.run_refs(record)
        return m.ForkSummary(
            fork_id=record.fork_id,
            base_run_id=batch.base_run_id,
            branch_name=record.branch_name,
            status="complete",
            hypothesis=record.hypothesis,
            edits=[self.fork_edit(e) for e in record.edits],
            mode=batch.mode,
            samples=len(batch.edited),
            control=bool(batch.controls),
            created_at=created,
            completed_at=created + timedelta(seconds=2),
            savings=self.savings(record),
            fix=fix,
            control_result=control,
            verdict=batch.verdict,
            preview=control is None,
            replay_fidelity=self.fidelity(record),
            edited_runs=edited_runs,
            control_runs=control_runs,
            export_available=batch.verdict == "VERIFIED",
            error=None,
        )

    # -- the stale-FX story ------------------------------------------------------------------

    def twin(self) -> m.NearestTwin:
        slot, entry = self.entry(TWIN_RUN)
        twin_steps = {s.addr: s for s in self.steps(TWIN_RUN)}
        demo_steps = {s.addr: s for s in self.steps(DEMO_RUN)}
        fx_twin, fx_demo = twin_steps["fx/tool#1"], demo_steps["fx/tool#1"]
        return m.NearestTwin(
            run=self.summary(entry, slot),
            similarity=0.91,
            same_task=False,
            basis=["agent", "graph_shape"],
            known_good_values=[
                m.KnownGoodValue(
                    citation=m.Citation(addr="fx/tool#1", json_pointer="/output/rate"),
                    value=fx_twin.output["rate"],
                    failing_value=fx_demo.output["rate"],
                ),
                m.KnownGoodValue(
                    citation=m.Citation(addr="fx/tool#1", json_pointer="/output/as_of"),
                    value=fx_twin.output["as_of"],
                    failing_value=fx_demo.output["as_of"],
                ),
            ],
        )

    def precedents(self) -> m.Precedents:
        cases = []
        for run_id, similarity in [("tc-0004", 0.94), ("tc-0009", 0.89)]:
            cases.append(
                m.PrecedentCase(
                    run_id=run_id,
                    root_addr="fx/tool#1",
                    fault_type="natural",
                    similarity=similarity,
                    fix_worked=True,
                    label_source="natural_auto",
                )
            )
        injected = next(r for r, res in self.world.injected.items() if res.fault_code == "T2")
        cases.append(
            m.PrecedentCase(
                run_id=injected,
                root_addr="fx/tool#1",
                fault_type="stale_data",
                similarity=0.83,
                fix_worked=None,
                label_source="injected",
            )
        )
        return m.Precedents(
            fault_type="stale_data",
            neighbours=12,
            same_fault=9,
            fix_summary="fresh=true",
            fix_tried=9,
            fix_worked=7,
            summary="Looks like 9 of 12 past stale-data failures; fresh=true fixed 7 of them.",
            cases=cases,
        )

    def verification(self, record: ForkRecord) -> m.Verification:
        summary = self.fork_summary("fresh_fx", record)
        assert summary.fix and summary.control_result
        return m.Verification(
            verdict=summary.verdict,
            status="complete",
            fork_id=record.fork_id,
            edit_addr="fx/tool#1",
            k=summary.samples,
            fix=summary.fix,
            control=summary.control_result,
            replay_fidelity=summary.replay_fidelity,
            preview=False,
            explanation=(
                f"Fix lower bound {summary.fix.ci_low:.3f} is above control upper bound "
                f"{summary.control_result.ci_high:.3f} at K={summary.samples}."
            ),
        )

    def verify_job(self) -> m.VerifyJob:
        plan = [
            ("fresh_fx", "precedent", "Re-fetched with fresh=true."),
            (
                "correct_budget",
                "oracle",
                "Fixing the downstream budget also repairs the run, but it "
                "is a symptom of the stale rate, not its cause.",
            ),
            (
                "writer_hint",
                "llm",
                "A prompt hint changed nothing: the stand-in ignores prompts. "
                "A speculative patch cannot refute a suspect.",
            ),
        ]
        candidates = []
        for name, source, note in plan:
            record = self.world.demo[name]
            summary = self.fork_summary(name, record)
            candidates.append(
                m.VerifyCandidate(
                    addr=record.edits[0].addr,
                    status="complete",
                    fork_id=record.fork_id,
                    edit=self.fork_edit(record.edits[0]),
                    edit_source=source,  # type: ignore[arg-type]
                    fix=summary.fix,
                    control_result=summary.control_result,
                    verdict=summary.verdict,
                    note=note,
                )
            )
        return m.VerifyJob(
            job_id="job-0001",
            run_id=DEMO_RUN,
            status="complete",
            samples=5,
            candidates=candidates,
            best_addr="fx/tool#1",
            created_at=BASE_TIME + timedelta(hours=3),
            completed_at=BASE_TIME + timedelta(hours=3, seconds=9),
            error=None,
        )

    def timeline(self) -> m.ForkTimeline:
        slot, entry = self.entry(DEMO_RUN)
        original = self.summary(entry, slot)
        entries = [
            m.TimelineEntry(
                kind="original",
                label="Original run",
                run_id=DEMO_RUN,
                fork_id=None,
                parent_fork_id=None,
                outcome=original.status,
                score=original.score,
                checker_reason=original.checker_reason,
                pass_rate=None,
                verdict=None,
                preview=False,
                savings=None,
                edit_summary=None,
            )
        ]
        forks = []
        for name in ["fresh_fx", "correct_budget", "writer_hint"]:
            record = self.world.demo[name]
            summary = self.fork_summary(name, record)
            forks.append(summary)
            edit = record.edits[0]
            control_run = record.batch.controls[0]
            entries.append(
                m.TimelineEntry(
                    kind="control",
                    label=f"Unchanged control ({summary.samples} samples)",
                    run_id=control_run.run_id,
                    fork_id=None,
                    parent_fork_id=record.fork_id,
                    outcome="passed" if control_run.outcome == "passed" else "failed",
                    score=control_run.score,
                    checker_reason=control_run.reason,
                    pass_rate=summary.control_result,
                    verdict=None,
                    preview=False,
                    savings=None,
                    edit_summary=None,
                )
            )
            fix_run = record.batch.edited[0]
            entries.append(
                m.TimelineEntry(
                    kind="fork",
                    label=record.branch_name,
                    run_id=fix_run.run_id,
                    fork_id=record.fork_id,
                    parent_fork_id=None,
                    outcome="passed" if fix_run.outcome == "passed" else "failed",
                    score=fix_run.score,
                    checker_reason=fix_run.reason,
                    pass_rate=summary.fix,
                    verdict=summary.verdict,
                    preview=False,
                    savings=summary.savings,
                    edit_summary=f"{edit.addr} {edit.kind} {json.dumps(edit.value, sort_keys=True)}",
                )
            )
        return m.ForkTimeline(base_run_id=DEMO_RUN, entries=entries, forks=forks)

    # -- everything else ------------------------------------------------------------------------

    def hoprag_diagnosis(self, detail: m.RunDetail) -> m.Diagnosis:
        steps = {s.addr: s for s in detail.steps}
        searches = [s for s in detail.steps if s.addr.endswith("/search#1")]
        scores = {s.addr: s.output[0]["score"] for s in searches if s.output}
        lowest = min(scores, key=lambda addr: scores[addr])
        hop = int(re.match(r"hop(\d)", lowest).group(1))  # type: ignore[union-attr]
        reasons = [
            m.Reason(
                suspect_addr=lowest,
                text=f"The best retrieved paragraph at hop {hop} scored {scores[lowest]:.1f}, the "
                "lowest of this run's searches.",
                citation=m.Citation(addr=lowest, json_pointer="/output/0/score"),
                feature=m.FeatureAttribution(
                    name="retrieval_top_score",
                    group="Grounding",
                    value=round(scores[lowest], 2),
                    contribution=0.41,
                ),
            )
        ]
        second = next((s for s in detail.steps if s.addr == "hop2/search#1"), None)
        if second is not None and "hop2_query" in second.state_after:
            reasons.append(
                m.Reason(
                    suspect_addr="hop2/search#1",
                    text="The hop-2 query is built from the hop-1 answer, so an early mistake "
                    "carries forward.",
                    citation=m.Citation(
                        addr="hop2/search#1", json_pointer="/state_after/hop2_query"
                    ),
                    feature=m.FeatureAttribution(
                        name="hop_dependency_depth", group="Lineage", value=2.0, contribution=0.33
                    ),
                )
            )
        candidates = [lowest, "hop1/answer#1", "hop2/answer#1", "decompose/chat#1", "final/chat#1"]
        candidates = [addr for addr in dict.fromkeys(candidates) if addr in steps]
        weights = {addr: 0.24 - 0.02 * i for i, addr in enumerate(candidates)}
        return authored.abstaining_diagnosis(detail, weights=weights, reasons=reasons)

    def provenance_files(self, detail: m.RunDetail) -> dict[str, m.ValueProvenance]:
        return {
            "provenance.json": x.provenance(detail, "budget/tool#1", "/input/args/fx/rate"),
            "provenance-ambiguous.json": x.provenance(
                detail, "budget/tool#1", "/input/args/fx/currency"
            ),
            "provenance-state.json": x.provenance(
                detail, "writer/chat#1", "/state/budget/total_inr"
            ),
        }

    def export_test(self, record: ForkRecord, base_detail: m.RunDetail) -> m.ExportTestResponse:
        run_id = record.batch.edited[0].run_id
        steps = self.fork_steps(record, run_id)
        cone = [
            s.addr for s in sorted(steps, key=lambda s: s.seq) if s.addr in record.batch.invalidated
        ]
        cassette = [
            {"addr": s.addr, "request_key": s.hashes.request_key, "response": s.output}
            for s in steps
            if s.has_call and s.cache_status in {"live", "edited"}
        ]
        cassette_bytes = canonical_json(cassette)
        slug = re.sub(r"[^a-z0-9]+", "_", f"{DEMO_RUN}_{record.edits[0].addr}".lower()).strip("_")
        fixture_dir = f"tests/regressions/fixtures/{slug}"
        test_path = f"tests/regressions/test_{slug}.py"
        test_text = (
            f"# Source run {DEMO_RUN}, fork {record.fork_id}, patch: fx/tool#1 args += fresh=true\n"
            f"# Invalidation cone: {', '.join(cone)}\n"
            "# Offline regression: the original recording fails, the saved patch passes.\n"
        ).encode()
        fixture_sha = hashlib.sha256(cassette_bytes).hexdigest()
        manifest = hashlib.sha256(
            f"{fixture_dir}/cassette.json\t{fixture_sha}".encode()
        ).hexdigest()
        return m.ExportTestResponse(
            export_id="exp-0001",
            fork_id=record.fork_id,
            verdict="VERIFIED",
            test_path=test_path,
            fixture_dir=fixture_dir,
            test_sha256=hashlib.sha256(test_text).hexdigest(),
            fixture_sha256=manifest,
            files=[
                m.ExportedFile(
                    path=test_path,
                    sha256=hashlib.sha256(test_text).hexdigest(),
                    bytes=len(test_text),
                ),
                m.ExportedFile(
                    path=f"{fixture_dir}/cassette.json",
                    sha256=fixture_sha,
                    bytes=len(cassette_bytes),
                ),
            ],
            invalidation_cone=cone,
            run_command=f"python -m pytest {test_path} -q",
            created_at=BASE_TIME + timedelta(hours=4),
        )

    def crash_report(self, detail: m.RunDetail, diagnosis: m.Diagnosis) -> m.CrashReport:
        steps = {s.addr: s for s in detail.steps}
        tags = {
            n.addr: n.tag for n in (diagnosis.damage_path.nodes if diagnosis.damage_path else [])
        }
        verification = diagnosis.verification
        assert verification and verification.fix and verification.control
        stale = next(v for v in steps["fx/tool#1"].rule_violations if v.rule == "stale_as_of")
        narrative = [
            ("planner/chat#1", "The planner parsed the request into constraints."),
            (
                "fx/tool#1",
                f"The FX tool returned a rate dated {steps['fx/tool#1'].output['as_of']}.",
            ),
            ("budget/tool#1", "The budget step multiplied by that rate."),
            ("writer/chat#1", "The writer put the budget total into the plan."),
            ("verifier/chat#1", "The verifier passed the plan through unchanged."),
            ("checker/state#1", f"The checker rejected the plan: {detail.run.checker_reason}"),
        ]
        return m.CrashReport(
            run_id=detail.run.run_id,
            title=f"{detail.run.task} failed on a stale exchange rate",
            generated_at=authored.FIXTURE_TIME,
            synopsis=f"{detail.run.run_id} failed the checker ({detail.run.checker_reason}). The "
            f"responsible step is fx/tool#1, confirmed by replay: {verification.fix.passed}/"
            f"{verification.fix.total} edited runs passed against {verification.control.passed}/"
            f"{verification.control.total} unchanged controls.",
            sequence_of_events=[
                m.ReportEvent(seq=steps[a].seq, addr=a, text=t, tag=tags.get(a))
                for a, t in narrative
            ],
            probable_cause=m.ReportClaim(
                text="The SGD to INR rate came from a snapshot months older than the rest of the "
                "run's data.",
                cites=[stale.citation],
            ),
            contributing_factors=[
                m.ReportClaim(
                    text="Nothing downstream checks the age of the rate before using it.",
                    cites=[m.Citation(addr="budget/tool#1", json_pointer="/input/args/fx/as_of")],
                )
            ],
            findings=[m.ReportClaim(text=r.text, cites=[r.citation]) for r in diagnosis.reasons],
            recommended_fix=m.ReportClaim(
                text="Re-fetch the rate with fresh=true.",
                cites=[m.Citation(addr="fx/tool#1", json_pointer="/input/args")],
            ),
            verification=m.ReportClaim(
                text=f"VERIFIED at K={verification.k}: fix {verification.fix.passed}/"
                f"{verification.fix.total} (95% interval {verification.fix.ci_low:.3f} to "
                f"{verification.fix.ci_high:.3f}) against control {verification.control.passed}/"
                f"{verification.control.total} ({verification.control.ci_low:.3f} to "
                f"{verification.control.ci_high:.3f}).",
                cites=[m.Citation(addr="fx/tool#1", json_pointer="/output")],
            ),
            fixture=True,
        )

    # -- assembly ------------------------------------------------------------------------------

    def put(self, name: str, model: Any) -> None:
        if isinstance(model, str):
            self.files[name] = model
        elif isinstance(model, list):
            payload = [
                i.model_dump(mode="json") if isinstance(i, m.ContractModel) else i for i in model
            ]
            self.files[name] = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        elif isinstance(model, m.ContractModel):
            self.files[name] = (
                json.dumps(model.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n"
            )
        else:
            self.files[name] = json.dumps(model, indent=2, ensure_ascii=False) + "\n"

    def assemble(self) -> dict[str, str]:
        summaries = [self.summary(entry, slot) for slot, entry in enumerate(self.entries)]
        summaries.sort(key=lambda s: s.started_at, reverse=True)
        facets = m.RunFacets(
            agents=dict(Counter(s.agent for s in summaries)),
            outcomes=dict(Counter(s.status for s in summaries)),
            origins=dict(Counter(s.origin for s in summaries)),
            splits=dict(Counter(s.split for s in summaries if s.split)),
        )
        total = len(summaries)
        self.put(
            "runs.json",
            m.RunList(
                total=total,
                limit=50,
                offset=0,
                next_offset=None,
                items=summaries,
                facets=facets,
                fixture=True,
            ),
        )
        self.put("failure-groups.json", self.failure_groups(summaries))

        slot, demo_entry = self.entry(DEMO_RUN)
        detail = self.detail(demo_entry, slot)
        self.put("run-detail.json", detail)
        for name, model in self.provenance_files(detail).items():
            self.put(name, model)

        fresh = self.world.demo["fresh_fx"]
        fix_run = fresh.batch.edited[0].run_id
        fix_reader = self.readers["tripcrew"]
        fix_steps = self.fork_steps(fresh, fix_run)
        fix_meta = x.RunMeta(
            task=detail.run.task,
            task_text=detail.task_text,
            origin="fork",
            split=None,
            started_at=BASE_TIME + timedelta(hours=3, seconds=1),
            risk=None,
            top_suspect=None,
            failure_signature=None,
            final_answer_key="final_plan",
        )
        fix_detail = m.RunDetail(
            run=fix_reader.summary(fix_run, fix_meta, fix_steps),
            task_text=detail.task_text,
            steps=fix_steps,
            edges=fix_reader.edges(fix_run),
            final_answer=fix_steps[-1].state_after.get("final_plan"),
            fixture=True,
        )
        self.put("run-detail-fix.json", fix_detail)

        slot, hr_entry = self.entry(ABSTAIN_RUN)
        hr_detail = self.detail(hr_entry, slot)
        self.put("run-detail-hoprag.json", hr_detail)

        twin = self.twin()
        damage = x.damage_path(detail.steps, detail.edges, "fx/tool#1", "checker/state#1")
        downstream = sum(n.tag == "symptom" for n in damage.nodes)
        fx_demo = next(s for s in detail.steps if s.addr == "fx/tool#1")
        drift = round(1 - fx_demo.output["rate"] / twin.known_good_values[0].value, 3)
        diagnosis = authored.stale_fx_diagnosis(
            detail,
            twin=twin,
            precedents=self.precedents(),
            verification=self.verification(fresh),
            damage=damage,
            downstream=downstream,
            drift=drift,
        )
        self.put("diagnosis.json", diagnosis)
        self.put("diagnosis-abstain.json", self.hoprag_diagnosis(hr_detail))
        report = self.crash_report(detail, diagnosis)
        self.put("crash-report.json", report)
        self.put("crash-report.md", render_markdown(report))

        request = m.ForkRequest(
            base_run_id=DEMO_RUN,
            edits=[self.fork_edit(e) for e in fresh.edits],
            mode="cone",
            samples=5,
            control=True,
            hypothesis=fresh.hypothesis,
            branch_name=fresh.branch_name,
        )
        self.put("fork-request.json", request)
        prediction = x.predict_replay(detail, request)
        for mode in ("cone", "prefix", "full"):
            suffix = "" if mode == "cone" else f"-{mode}"
            self.put(f"replay-prediction{suffix}.json", x.predict_replay(detail, request, mode))  # type: ignore[arg-type]
        self.put(
            "fork-created.json",
            m.ForkCreated(
                fork_id=fresh.fork_id,
                base_run_id=DEMO_RUN,
                branch_name=fresh.branch_name,
                status="queued",
                stream_url=f"/forks/{fresh.fork_id}/stream",
                prediction=prediction,
                created_at=BASE_TIME + timedelta(hours=3),
            ),
        )
        self.put("fork-summary.json", self.fork_summary("fresh_fx", fresh))
        self.put("forks.json", self.timeline())
        self.put("verify-job.json", self.verify_job())
        for name, filename, text in [
            ("fresh_fx", "fork-events-verified.json", "Edited 5/5 against control 0/5: VERIFIED."),
            ("noisy", "fork-events-inconclusive.json", "A noisy writer step: the bounds overlap."),
        ]:
            record = self.world.demo[name]
            self.put(
                filename,
                m.ForkStreamRecording(
                    fork_id=record.fork_id,
                    description=f"{text} Engine events with sample/branch, timings and a "
                    "running phase added.",
                    provenance="engine_recording"
                    if name == "fresh_fx"
                    else "engine_recording_noisy_standin",
                    events=self.fork_events(record),
                ),
            )
        noisy_summary = self.fork_summary("noisy", self.world.demo["noisy"])
        self.put("fork-summary-inconclusive.json", noisy_summary)

        base_left = detail
        nearest = m.NearestPassingLink(
            run_id=TWIN_RUN,
            similarity=0.91,
            reason="Same agent and graph shape; also a Singapore trip.",
        )
        self.put(
            "diff.json",
            x.build_diff(
                base_left, fix_detail, invalidated=set(fresh.batch.invalidated), nearest=nearest
            ),
        )
        self.put("export-test.json", self.export_test(fresh, detail))
        self.put("eval.json", authored.evaluation())
        self.put("health.json", self.health(summaries))
        self.put("agents.json", self.agents(summaries))
        self.put("label-queue.json", self.label_queue(summaries))
        self.put(
            "label-request.json",
            m.LabelRequest(
                run_id=ABSTAIN_RUN,
                annotator="annotator-a",
                root_addr="hop1/answer#1",
                certainty="unsure",
                notes="Hop 1 returned a name from the wrong paragraph.",
            ),
        )
        self.put(
            "label-response.json",
            m.LabelResponse(
                label_id="lbl-0001",
                run_id=ABSTAIN_RUN,
                annotator="annotator-a",
                root_addr="hop1/answer#1",
                created_at=BASE_TIME + timedelta(hours=5),
                progress=m.LabelProgress(labelled=1, target=50, remaining=49),
                agreement=m.KappaStatus(kappa=None, n=0, annotators=1),
            ),
        )
        self.put(
            "otlp-ack.json",
            m.OtlpIngestAck(
                partialSuccess=m.OtlpPartialSuccess(rejectedSpans=0, errorMessage=""),
                accepted_spans=14,
                runs=[m.OtlpRunRef(run_id="otlp-0001", spans=14)],
                replayable=False,
                note="Imported traces support diagnosis and comparison only; replay needs the SDK.",
            ),
        )
        self.put(
            "error-live-call-refused.json",
            m.ErrorResponse(
                error=m.ErrorBody(
                    code="live_call_refused",
                    status=409,
                    message="This fork needs a live LLM or tool call, and recorded mode refuses them.",
                    hint="Run the API with MODE=offline or MODE=live, or fork a step whose "
                    "dependency cone is already in the recording.",
                    issues=[],
                    context={"base_run_id": DEMO_RUN, "addr": "fx/tool#1", "mode": "recorded"},
                    request_id="req-0001",
                )
            ),
        )
        self.put("manifest.json", self.manifest())
        return self.files

    def failure_groups(self, summaries: list[m.RunSummary]) -> m.FailureGroupList:
        failed = [s for s in summaries if s.status == "failed" and s.failure_signature]
        groups = []
        for signature, count in Counter(s.failure_signature for s in failed).most_common():
            members = [s for s in failed if s.failure_signature == signature]
            suspect = members[0].top_suspect.name if members[0].top_suspect else None
            kind = signature.split(":")[-1].replace("_", " ")
            fault = Counter(s.label.fault_type for s in members if s.label and s.label.fault_type)
            groups.append(
                m.FailureGroup(
                    signature=signature,
                    label=f"{suspect or 'unlocalised'} · {kind}",
                    agent=members[0].agent,
                    count=count,
                    top_suspect_name=suspect,
                    fault_type=fault.most_common(1)[0][0] if fault else None,
                    example_run_ids=[s.run_id for s in members[:3]],
                )
            )
        return m.FailureGroupList(total_failed=len(failed), items=groups, fixture=True)

    def health(self, summaries: list[m.RunSummary]) -> m.AppHealth:
        steps = sum(s.steps for s in summaries)
        labels = sum(s.label is not None for s in summaries)
        return m.AppHealth(
            status="ok",
            api_version=m.CONTRACT_VERSION,
            mode="recorded",
            static_bundle=False,
            model_version=None,
            dataset_version=None,
            capabilities=m.Capabilities(
                diagnose=True,
                fork=True,
                live_calls=False,
                verify=True,
                export_test=True,
                label=True,
                ingest_otlp=True,
                eval=True,
            ),
            counts=m.HealthCounts(
                runs=len(summaries), steps=steps, forks=len(self.world.demo), labels=labels
            ),
            started_at=BASE_TIME,
            notes=[
                "Fixture response: counts describe the mock run list, not a real database.",
                "No diagnoser is trained yet; diagnoses in the mocks are authored fixtures.",
            ],
        )

    def agents(self, summaries: list[m.RunSummary]) -> m.AgentList:
        items = []
        for agent, name, description, model, typical in [
            (
                "tripcrew",
                "TripCrew",
                "Multi-agent trip planner over deterministic travel tools.",
                TRIPCREW_MODEL,
                16,
            ),
            (
                "hoprag",
                "HopRAG",
                "Multi-hop retrieval QA over MuSiQue-Ans paragraphs.",
                "hoprag-lexical-v1-h3",
                12,
            ),
        ]:
            mine = [s for s in summaries if s.agent == agent]
            items.append(
                m.AgentInfo(
                    agent_id=agent,
                    display_name=name,
                    description=description,
                    client="deterministic_standin",
                    models=[model],
                    runs=len(mine),
                    passed=sum(s.status == "passed" for s in mine),
                    failed=sum(s.status == "failed" for s in mine),
                    typical_steps=typical,
                )
            )
        return m.AgentList(items=items)

    def label_queue(self, summaries: list[m.RunSummary]) -> m.LabelQueue:
        failed = [s for s in summaries if s.status == "failed" and s.origin == "natural"]
        items = [
            m.LabelTask(
                run_id=s.run_id,
                agent=s.agent,
                task=s.task,
                checker_reason=s.checker_reason,
                steps=s.steps,
                labelled_by_you=s.run_id == ABSTAIN_RUN,
            )
            for s in failed
        ]
        return m.LabelQueue(
            progress=m.LabelProgress(labelled=1, target=50, remaining=49), items=items
        )

    def manifest(self) -> dict[str, Any]:
        recorded = "recorded"
        mixed = "recorded+authored"
        authored_only = "authored"
        files = {
            "runs.json": ("RunList", "GET /runs", mixed),
            "failure-groups.json": ("FailureGroupList", "GET /failure-groups", mixed),
            "run-detail.json": ("RunDetail", "GET /runs/tc-0001", mixed),
            "run-detail-fix.json": ("RunDetail", "GET /runs/{fork run}", recorded),
            "run-detail-hoprag.json": ("RunDetail", "GET /runs/hr-0003", mixed),
            "provenance.json": ("ValueProvenance", "GET /runs/tc-0001/provenance", recorded),
            "provenance-ambiguous.json": (
                "ValueProvenance",
                "GET /runs/tc-0001/provenance",
                recorded,
            ),
            "provenance-state.json": ("ValueProvenance", "GET /runs/tc-0001/provenance", recorded),
            "diagnosis.json": ("Diagnosis", "GET /runs/tc-0001/diagnosis", mixed),
            "diagnosis-abstain.json": ("Diagnosis", "GET /runs/hr-0003/diagnosis", mixed),
            "crash-report.json": ("CrashReport", "GET /runs/tc-0001/report", mixed),
            "fork-request.json": ("ForkRequest", "POST /forks", recorded),
            "fork-created.json": ("ForkCreated", "POST /forks", recorded),
            "replay-prediction.json": ("ReplayPrediction", "POST /replay/predict", recorded),
            "replay-prediction-prefix.json": ("ReplayPrediction", "POST /replay/predict", recorded),
            "replay-prediction-full.json": ("ReplayPrediction", "POST /replay/predict", recorded),
            "fork-summary.json": ("ForkSummary", "GET /forks/{id}", recorded),
            "fork-summary-inconclusive.json": ("ForkSummary", "GET /forks/{id}", recorded),
            "forks.json": ("ForkTimeline", "GET /runs/tc-0001/forks", recorded),
            "verify-job.json": ("VerifyJob", "POST /runs/tc-0001/verify", recorded),
            "fork-events-verified.json": (
                "ForkStreamRecording",
                "GET /forks/{id}/stream",
                recorded,
            ),
            "fork-events-inconclusive.json": (
                "ForkStreamRecording",
                "GET /forks/{id}/stream",
                recorded,
            ),
            "diff.json": ("DiffResponse", "GET /diff", recorded),
            "export-test.json": (
                "ExportTestResponse",
                "POST /forks/{id}/export-test",
                authored_only,
            ),
            "eval.json": ("EvalResponse", "GET /eval", authored_only),
            "health.json": ("AppHealth", "GET /health", mixed),
            "agents.json": ("AgentList", "GET /agents", recorded),
            "label-queue.json": ("LabelQueue", "GET /labels/queue", mixed),
            "label-request.json": ("LabelRequest", "POST /labels", authored_only),
            "label-response.json": ("LabelResponse", "POST /labels", authored_only),
            "otlp-ack.json": ("OtlpIngestAck", "POST /v1/traces", authored_only),
            "error-live-call-refused.json": ("ErrorResponse", "POST /forks (409)", authored_only),
        }
        return {
            "contract_version": m.CONTRACT_VERSION,
            "note": "Generated by web/mocks/build_mocks.py. Run data is recorded from deterministic "
            "stand-ins, not language models. 'authored' content is illustrative.",
            "sources": {
                "hoprag_questions": "MuSiQue-Ans dev, CC BY 4.0, https://github.com/StonyBrookNLP/musique",
            },
            "files": {
                name: {"model": model, "endpoint": endpoint, "provenance": kind}
                for name, (model, endpoint, kind) in files.items()
            },
            "text_files": {MD_NAME: "Markdown rendering of crash-report.json"},
        }


def render_markdown(report: m.CrashReport) -> str:
    def cites(claim: m.ReportClaim) -> str:
        chips = ", ".join(f"`{c.addr}{c.json_pointer or ''}`" for c in claim.cites)
        return f"{claim.text} ({chips})"

    lines = [f"# {report.title}", "", f"Run `{report.run_id}`. Illustrative fixture.", ""]
    lines += ["## Synopsis", "", report.synopsis, "", "## Sequence of events", ""]
    lines += [f"{e.seq}. `{e.addr}`: {e.text}" for e in report.sequence_of_events]
    sections = [
        ("Probable cause", [report.probable_cause] if report.probable_cause else []),
        ("Contributing factors", report.contributing_factors),
        ("Findings", report.findings),
        ("Recommended fix", [report.recommended_fix] if report.recommended_fix else []),
        ("Verification", [report.verification] if report.verification else []),
    ]
    for title, claims in sections:
        lines += ["", f"## {title}", ""]
        lines += [f"- {cites(c)}" for c in claims]
    return "\n".join(lines) + "\n"


def rename_ids(text: str, ids: dict[str, str]) -> str:
    for raw in sorted(ids, key=len, reverse=True):
        text = text.replace(raw, ids[raw])
    return text


async def build(out: Path, dataset: Path | None = HOPRAG_DATASET) -> list[str]:
    with tempfile.TemporaryDirectory() as workdir:
        world = await build_world(Path(workdir), dataset)
        try:
            assembler = Assembler(world, dataset)
            files = assembler.assemble()
        finally:
            world.tripcrew.close()
            world.hoprag.close()
    out.mkdir(parents=True, exist_ok=True)
    for name, text in sorted(files.items()):
        (out / name).write_text(rename_ids(text, assembler.ids), encoding="utf-8")
    return sorted(files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=MOCKS, help="output directory")
    parser.add_argument("--dataset", type=Path, default=HOPRAG_DATASET)
    args = parser.parse_args()
    for name in asyncio.run(build(args.out, args.dataset)):
        print(f"wrote {args.out / name}")


if __name__ == "__main__":
    main()
