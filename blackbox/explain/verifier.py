"""Turn "the model suspects X" into "an intervention at X does/does not fix the run".

For each of the top suspects the verifier picks an intervention, then runs K paired
``cone`` replays with the fix and K no-edit controls with identical seeds:

1. **Masking check.** If the suspect's *input* already differs from the same-task
   passing twin, damage reached it from upstream. Replacing its output could flip the
   outcome while hiding the real root, so it is not tested: INCONCLUSIVE, citing the
   earliest upstream step that diverged on its own.
2. **Known-good value**, preferred: an oracle fix, else the twin's output for the same
   input. Only these can REFUTE a suspect.
3. **Fix template** from similar past cases or the step's own evidence: retry the call
   (error payloads), re-fetch with ``fresh=true`` (stale data). Never known-good.

Verdicts follow the replay engine: VERIFIED when the fix's Wilson 95% lower bound
exceeds the control's upper bound; REFUTED when a known-good fix reproducibly fails to
improve the outcome; INCONCLUSIVE otherwise, with what would resolve it.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from blackbox.explain.reasons import same_input
from blackbox.ml.dataset import Trace, TraceStep
from blackbox.ml.features import output_payload
from blackbox.recorder import canonical_json
from blackbox.replay import (
    Edit,
    ReplayEngine,
    override_output,
    patch_tool_args,
    rerun,
    wilson_interval,
)
from blackbox.sdk import Recorder, RunSession

logger = logging.getLogger(__name__)

AgentFactory = Callable[[str], Callable[[RunSession], Awaitable[Any] | Any]]
Oracle = Callable[[str], Mapping[str, Any]]
EDITABLE = {"llm", "tool", "retrieval"}


@dataclass(slots=True)
class Intervention:
    edit: Edit | None
    source: str | None  # oracle | twin | template
    note: str
    masked_by: str | None = None
    no_op: bool = False  # the known-good value equals what the step already produced


def _same_output(a: TraceStep, b: TraceStep) -> bool:
    try:
        return canonical_json(output_payload(a)[0]) == canonical_json(output_payload(b)[0])
    except (TypeError, ValueError):
        return False


def _ancestors(trace: Trace, addr: str) -> set[str]:
    parents: dict[str, set[str]] = {}
    for src, dst, _ in trace.edges:
        parents.setdefault(dst, set()).add(src)
    seen: set[str] = set()
    stack = [addr]
    while stack:
        for parent in parents.get(stack.pop(), ()):
            if parent not in seen:
                seen.add(parent)
                stack.append(parent)
    return seen


def first_self_divergence(trace: Trace, twin: Trace, among: set[str] | None = None) -> str | None:
    """Earliest step that got the twin's input but produced a different output."""
    twin_steps = {s.addr: s for s in twin.steps}
    for step in sorted(trace.steps, key=lambda s: s.seq):
        if among is not None and step.addr not in among:
            continue
        other = twin_steps.get(step.addr)
        if other is not None and same_input(step, other) and not _same_output(step, other):
            return step.addr
    return None


def samples_needed(fix_rate: float, control_rate: float, limit: int = 100) -> int | None:
    """Smallest K at which these pass rates would separate (None if they never would)."""
    if fix_rate <= control_rate:
        return None
    for n in range(3, limit + 1):
        low = wilson_interval(round(fix_rate * n), n)[0]
        high = wilson_interval(round(control_rate * n), n)[1]
        if low > high:
            return n
    return None


class Verifier:
    def __init__(
        self,
        recorder: Recorder,
        agent_factory: AgentFactory,
        *,
        oracle: Oracle | None = None,
        samples: int = 5,
        top: int = 3,
    ) -> None:
        self.engine = ReplayEngine(recorder)
        self.agent_factory = agent_factory
        self.oracle = oracle
        self.samples = samples
        self.top = top

    def plan(
        self,
        trace: Trace,
        addr: str,
        *,
        twin: Trace | None,
        fix_template: Mapping[str, Any] | None = None,
        evidence: list[dict[str, Any]] = (),
        oracle_fixes: Mapping[str, Any] | None = None,
    ) -> Intervention:
        """Choose the intervention for one suspect. ``twin`` must be a same-task twin."""
        step = next((s for s in trace.steps if s.addr == addr), None)
        if step is None:
            return Intervention(None, None, f"{addr} is not a step of this run.")
        if step.kind not in EDITABLE:
            return Intervention(
                None,
                None,
                "State updates make no call that replay can edit; inspect the steps "
                "that produced the values it wrote.",
            )
        twin_step = next((s for s in twin.steps if s.addr == addr), None) if twin else None
        if twin_step is not None and not same_input(step, twin_step):
            upstream = first_self_divergence(trace, twin, _ancestors(trace, addr))
            where = f" (first diverged at {upstream})" if upstream else ""
            return Intervention(
                None,
                None,
                f"Its input already differs from passing run {twin.run_id}{where}: the "
                "damage began upstream, and editing this step would only hide it.",
                masked_by=upstream,
            )
        if oracle_fixes and addr in oracle_fixes:
            return Intervention(
                override_output(addr, oracle_fixes[addr], known_good=True),
                "oracle",
                "Replaced its output with the oracle's correct value.",
                no_op=_same_output(step, replace(step, output=oracle_fixes[addr])),
            )
        if twin_step is not None:
            return Intervention(
                override_output(addr, twin_step.output, known_good=True),
                "twin",
                f"Replaced its output with passing run {twin.run_id}'s output for the same input.",
                no_op=_same_output(step, twin_step),
            )
        kinds = {line["kind"] for line in evidence}
        template = (fix_template or {}).get("kind")
        if "error_payload" in kinds or "empty_output" in kinds or template == "rerun":
            return Intervention(rerun(addr), "template", "Retried the call with the same request.")
        if step.kind == "tool" and ("stale_date" in kinds or template == "fresh"):
            return Intervention(
                patch_tool_args(addr, {"fresh": True}),
                "template",
                "Re-fetched with fresh=true.",
            )
        return Intervention(
            None, None, "No oracle value, same-task passing twin or applicable fix template."
        )

    async def verify(
        self,
        trace: Trace,
        suspects: list[tuple[str, float]],
        *,
        twin: Trace | None = None,
        templates: Mapping[str, Mapping[str, Any]] | None = None,
        evidence: Mapping[str, list[dict[str, Any]]] | None = None,
        extra: list[tuple[str, float]] = (),
    ) -> list[dict[str, Any]]:
        """Verify the model's top suspects, then any ``extra`` candidates.

        ``extra`` holds candidates proposed by something other than the model (the
        first divergence from the twin); results say who proposed each one, so the
        model's own verification rate is never inflated by them. ``twin`` must share
        the run's task.
        """
        if twin is not None and twin.task_id != trace.task_id:
            twin = None
        oracle_fixes: Mapping[str, Any] | None = None
        if self.oracle is not None:
            try:
                oracle_fixes = self.oracle(trace.run_id)
            except (KeyError, ValueError, IndexError) as error:
                logger.info("no oracle for %s: %s", trace.run_id, error)
        agent_fn = self.agent_factory(trace.run_id)
        results = []
        candidates = [(a, p, "model") for a, p in suspects[: self.top]]
        listed = {a for a, _, _ in candidates}
        candidates += [(a, p, "twin_diff") for a, p in extra if a not in listed]
        for addr, probability, proposed_by in candidates:
            intervention = self.plan(
                trace,
                addr,
                twin=twin,
                fix_template=(templates or {}).get(addr),
                evidence=(evidence or {}).get(addr, []),
                oracle_fixes=oracle_fixes,
            )
            result: dict[str, Any] = {
                "addr": addr,
                "probability": round(probability, 4),
                "proposed_by": proposed_by,
                "source": intervention.source,
                "edit": intervention.edit.as_dict() if intervention.edit else None,
                "known_good": bool(intervention.edit and intervention.edit.known_good),
                "masked_by": intervention.masked_by,
                "samples": 0,
                "verdict": "INCONCLUSIVE",
                "note": intervention.note,
            }
            if intervention.edit is not None:
                result.update(await self._replay(trace.run_id, agent_fn, intervention))
            results.append(result)
        return results

    async def _replay(
        self, run_id: str, agent_fn: Any, intervention: Intervention
    ) -> dict[str, Any]:
        edit = intervention.edit
        try:
            batch = await self.engine.replay(
                run_id,
                agent_fn,
                edits=[edit],
                mode="cone",
                samples=self.samples,
                control=True,
                control_scope="downstream",
                branch_name=f"verify-{edit.addr}",
            )
        except Exception as error:  # a broken replay is evidence of nothing
            logger.warning("verification replay of %s at %s failed: %s", run_id, edit.addr, error)
            return {"note": f"{intervention.note} The replay failed: {error}"}
        verdict = batch.verdict or "INCONCLUSIVE"
        note = intervention.note
        if verdict == "VERIFIED":
            note += " The fix reproducibly made the run pass where the unedited replay failed."
        elif verdict == "REFUTED":
            note += " A known-good value here did not make the run pass."
            if intervention.no_op:
                note += " (Its output already matched that value: the step behaved correctly.)"
        elif batch.fix_pass_rate <= (batch.control_pass_rate or 0.0):
            note += (
                " The edit did not improve the pass rate."
                if edit.known_good
                else " The edit did not help, but it was not a known-good value, so this does "
                "not refute the step."
            )
        else:
            needed = samples_needed(batch.fix_pass_rate, batch.control_pass_rate or 0.0)
            note += (
                f" The pass rates differ but the intervals overlap at K={self.samples}; about "
                f"K={needed} samples at these rates would separate them."
                if needed
                else " The pass rates differ too little to separate with a practical K."
            )
        return {
            "verdict": verdict,
            "note": note,
            "samples": self.samples,
            "fork_id": batch.fork_id,
            "fix_run_id": batch.edited[0].run_id if batch.edited else None,
            "fix_rate": batch.fix_pass_rate,
            "fix_ci": list(batch.fix_interval),
            "control_rate": batch.control_pass_rate,
            "control_ci": list(batch.control_interval) if batch.control_interval else None,
            "cone": sorted(batch.invalidated),
        }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """One verdict for the diagnosis: VERIFIED if any suspect is; REFUTED if all are."""
    verified = next((r["addr"] for r in results if r["verdict"] == "VERIFIED"), None)
    if verified:
        verdict = "VERIFIED"
    elif results and all(r["verdict"] == "REFUTED" for r in results):
        verdict = "REFUTED"
    else:
        verdict = "INCONCLUSIVE"
    return {"verdict": verdict, "verified_addr": verified}


def record_verified(database: Any, run_id: str, addr: str, fault_type: str | None = None) -> bool:
    """Retraining loop: a VERIFIED root becomes a ``verified`` label (training split only).

    An existing label for the run is marked verified when it names the same root and is
    otherwise left alone; returns whether anything was written.
    """
    existing = database.one("SELECT root_addr FROM labels WHERE run_id = ?", (run_id,))
    if existing is not None:
        if existing["root_addr"] != addr:
            return False
        database.execute("UPDATE labels SET verified = 1 WHERE run_id = ?", (run_id,))
        return True
    database.execute(
        "INSERT INTO labels(run_id, root_addr, fault_type, source, verified) "
        "VALUES (?, ?, ?, 'verified', 1)",
        (run_id, addr, fault_type),
    )
    return True
