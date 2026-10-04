"""Fork-and-perturb: inject a fault into a passing run and replay it."""

from __future__ import annotations

import logging
import random as random_module
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from blackbox.forge.label import ForkResult, Labeler
from blackbox.forge.operators import FaultOperator
from blackbox.replay import Edit, ReplayEngine, ghost_hint
from blackbox.sdk import Recorder
from blackbox.sdk.runtime import RunSession

logger = logging.getLogger(__name__)

# Deterministic agent test doubles ignore prompts, so a ghost hint would change nothing.
TEST_DOUBLE_MODEL_PREFIXES = ("tripcrew-fixture-", "hoprag-lexical-")

# How many steps are tried as a distractor before an attempt falls back to a plain fork.
DISTRACTOR_TRIES = 5


def is_test_double(model: str | None) -> bool:
    return bool(model) and model.startswith(TEST_DOUBLE_MODEL_PREFIXES)


@dataclass(slots=True)
class InjectionPlan:
    """A planned fault injection ready for execution."""

    base_run_id: str
    target_step: dict[str, Any]
    operator: FaultOperator
    edit: Edit
    original_output: Any


class FaultInjector:
    """Orchestrate fault injection: pick a step, apply an operator, replay, and label."""

    def __init__(
        self,
        recorder: Recorder,
        agent_fn_factory: Callable[[str], Callable[[RunSession], Awaitable[Any] | Any]],
        *,
        default_samples: int = 1,
        default_control: bool = True,
        ghost_hints: bool = True,
    ) -> None:
        self.recorder = recorder
        self.engine = ReplayEngine(recorder)
        self.labeler = Labeler(recorder.database)
        self.agent_fn_factory = agent_fn_factory
        self.default_samples = default_samples
        self.default_control = default_control
        self.ghost_hints = ghost_hints
        # A recorded run never changes, so its steps and outputs are read from disk once.
        self._loaded: dict[str, list[tuple[dict[str, Any], Any]]] = {}

    def _uses_ghost_hint(self, run_id: str, step: dict[str, Any], hint: str) -> bool:
        if not self.ghost_hints or not hint or step.get("kind") != "llm":
            return False
        run = self.recorder.database.one("SELECT model FROM runs WHERE run_id = ?", (run_id,))
        return run is not None and not is_test_double(run.get("model"))

    def _passing_runs(self, agent: str | None = None) -> list[dict[str, Any]]:
        """Query all passing base runs (not forks)."""
        query = "SELECT * FROM runs WHERE outcome = 'passed' AND fork_id IS NULL"
        params: list[Any] = []
        if agent:
            query += " AND agent = ?"
            params.append(agent)
        return self.recorder.database.query(query, params)

    def _steps_for_run(self, run_id: str) -> list[dict[str, Any]]:
        """Return all steps for a run, ordered by sequence."""
        return self.recorder.database.query(
            "SELECT * FROM steps WHERE run_id = ? ORDER BY seq",
            (run_id,),
        )

    def _load_output(self, step: dict[str, Any]) -> Any:
        """Load the recorded output for a step from the content store."""
        output_hash = step.get("output_hash")
        if not output_hash:
            return None
        try:
            return self.recorder.store.load_json(output_hash)
        except (OSError, ValueError):
            return None

    def _steps_with_outputs(self, run_id: str) -> list[tuple[dict[str, Any], Any]]:
        """Steps of a run that recorded an output, with that output, in sequence order."""
        if run_id not in self._loaded:
            pairs = []
            for step in self._steps_for_run(run_id):
                output = self._load_output(step)
                if output is not None:
                    pairs.append((step, output))
            self._loaded[run_id] = pairs
        return self._loaded[run_id]

    def candidate_sites(
        self, operator: FaultOperator, agent: str | None = None
    ) -> list[tuple[str, str]]:
        """Every (passing base run, step address) the operator applies to, in a stable order."""
        sites = []
        for run in sorted(self._passing_runs(agent), key=lambda row: row["run_id"]):
            for step, output in self._steps_with_outputs(run["run_id"]):
                if operator.applicable(step, output):
                    sites.append((run["run_id"], step["addr"]))
        return sites

    def plan_injection(
        self,
        run_id: str,
        operator: FaultOperator,
        rng: random_module.Random,
        *,
        target_addr: str | None = None,
    ) -> InjectionPlan | None:
        """Create an injection plan for a specific run and operator.

        If target_addr is None, a random applicable step is chosen. Returns None if no
        applicable step is found, or if every candidate's edit would leave the output as
        it was: an edit that changes nothing is not a fault, and replaying it would only
        manufacture a "recovered" example.
        """
        candidates = [
            (step, output)
            for step, output in self._steps_with_outputs(run_id)
            if (not target_addr or step["addr"] == target_addr)
            and operator.applicable(step, output)
        ]
        rng.shuffle(candidates)

        for step, output in candidates:
            # Decision/coordination faults on a real model: let the LLM write the mistake
            # itself from a hidden instruction. Otherwise apply the operator's own edit.
            hint = operator.ghost_hint(step, rng)
            if self._uses_ghost_hint(run_id, step, hint):
                edit = ghost_hint(step["addr"], hint)
            else:
                edit = operator.apply(step, output, rng)
                if edit.value == output:
                    continue
            return InjectionPlan(
                base_run_id=run_id,
                target_step=step,
                operator=operator,
                edit=edit,
                original_output=output,
            )
        return None

    async def execute(
        self,
        plan: InjectionPlan,
        *,
        samples: int | None = None,
        control: bool | None = None,
        distractor_edit: Edit | None = None,
        distractor_addr: str | None = None,
        distractor_fault_code: str | None = None,
    ) -> ForkResult:
        """Execute a fault injection plan: replay with the fault and label the result."""
        samples = samples or self.default_samples
        control = control if control is not None else self.default_control

        edits = [plan.edit]
        if distractor_edit is not None:
            edits.append(distractor_edit)

        # Build agent function for replay
        agent_fn = self.agent_fn_factory(plan.base_run_id)

        batch = await self.engine.replay(
            plan.base_run_id,
            agent_fn,
            edits=edits,
            mode="cone",
            samples=samples,
            control=control,
            branch_name=f"forge-{plan.operator.spec.code}",
        )

        # Verify anti-cheating: only declared fields changed. A ghost-hint edit has
        # no precomputed output; the model writes it during replay.
        if (
            plan.edit.kind != "ghost_hint"
            and plan.original_output is not None
            and plan.edit.value is not None
        ):
            if not plan.operator.verify_change(plan.original_output, plan.edit.value):
                logger.warning(
                    "Anti-cheating check failed for %s at %s: undeclared fields changed",
                    plan.operator.spec.code,
                    plan.target_step["addr"],
                )

        result = self.labeler.classify(
            batch,
            target_addr=plan.target_step["addr"],
            fault_code=plan.operator.spec.code,
            fault_type=plan.operator.spec.name,
            held_out=plan.operator.spec.held_out,
            distractor_addr=distractor_addr,
            distractor_fault_code=distractor_fault_code,
        )

        logger.info(
            "Forge %s at %s in %s → %s (fix=%.2f, control=%s)",
            plan.operator.spec.code,
            plan.target_step["addr"],
            plan.base_run_id,
            result.label.value,
            result.fix_pass_rate,
            f"{result.control_pass_rate:.2f}" if result.control_pass_rate is not None else "N/A",
        )

        return result

    async def _passes_alone(self, run_id: str, edit: Edit, branch_name: str) -> bool:
        """Replay one edit by itself, without a control, and report whether the run still passes."""
        batch = await self.engine.replay(
            run_id,
            self.agent_fn_factory(run_id),
            edits=[edit],
            mode="cone",
            samples=1,
            control=False,
            branch_name=branch_name,
        )
        return batch.fix_pass_rate == 1.0

    async def plan_distractor(
        self,
        plan: InjectionPlan,
        distractor_operator: FaultOperator,
        rng: random_module.Random,
    ) -> tuple[Edit, str] | None:
        """Pick a recoverable fault at another step of the same run, or None if there is none.

        The root of a distractor fork is the primary fault, so two things must hold before the
        distractor is used: the primary fault alone fails the run (otherwise the combined failure
        could be an interaction and the root would be wrong), and the distractor alone leaves it
        passing (a separate control showing the distractor is suspicious but not causal).
        """
        run_id = plan.base_run_id
        if await self._passes_alone(run_id, plan.edit, "primary-check"):
            return None

        candidates = [
            (step, output)
            for step, output in self._steps_with_outputs(run_id)
            if step["addr"] != plan.target_step["addr"]
            and distractor_operator.applicable(step, output)
        ]
        rng.shuffle(candidates)
        for step, output in candidates[:DISTRACTOR_TRIES]:
            edit = distractor_operator.apply(step, output, rng)
            if edit.value == output:
                continue
            try:
                recoverable = await self._passes_alone(run_id, edit, "distractor-check")
            except Exception:
                logger.debug("Distractor check failed at %s", step["addr"], exc_info=True)
                continue
            if recoverable:
                return edit, step["addr"]
        return None

    async def inject_site(
        self,
        operator: FaultOperator,
        run_id: str,
        addr: str,
        rng: random_module.Random,
        *,
        samples: int | None = None,
        control: bool | None = None,
        distractor_operator: FaultOperator | None = None,
    ) -> ForkResult | None:
        """Inject at one chosen step, with a recoverable distractor when one can be found.

        The outcome depends only on the site, the operator and ``rng``, so a caller that seeds
        ``rng`` from the site gets the same fork however the attempts are scheduled. Returns None
        when the operator has no effective edit at that step.
        """
        plan = self.plan_injection(run_id, operator, rng, target_addr=addr)
        if plan is None:
            return None
        distractor = None
        if distractor_operator is not None:
            distractor = await self.plan_distractor(plan, distractor_operator, rng)
        if distractor is None:
            return await self.execute(plan, samples=samples, control=control)
        edit, distractor_addr = distractor
        return await self.execute(
            plan,
            samples=samples,
            control=control,
            distractor_edit=edit,
            distractor_addr=distractor_addr,
            distractor_fault_code=distractor_operator.spec.code,
        )

    async def inject_random(
        self,
        operator: FaultOperator,
        rng: random_module.Random,
        *,
        agent: str | None = None,
        samples: int | None = None,
        control: bool | None = None,
    ) -> ForkResult | None:
        """Pick a random passing run, inject a fault, replay, and label."""
        runs = self._passing_runs(agent)
        if not runs:
            logger.warning("No passing runs found for agent=%s", agent)
            return None

        rng.shuffle(runs)
        for run in runs:
            plan = self.plan_injection(run["run_id"], operator, rng)
            if plan is not None:
                return await self.execute(plan, samples=samples, control=control)

        logger.warning("No applicable steps found for %s", operator.spec.code)
        return None

    async def inject_with_distractor(
        self,
        primary_operator: FaultOperator,
        distractor_operator: FaultOperator,
        rng: random_module.Random,
        *,
        agent: str | None = None,
        samples: int | None = None,
        control: bool | None = None,
    ) -> ForkResult | None:
        """Inject a primary fault plus a recoverable distractor.

        The distractor should be a fault that the agent can recover from
        (T4 retried or C4 loop that recovers), teaching the model the
        difference between 'suspicious' and 'causal'.
        """
        runs = self._passing_runs(agent)
        if not runs:
            return None

        rng.shuffle(runs)
        for run in runs:
            primary_plan = self.plan_injection(run["run_id"], primary_operator, rng)
            if primary_plan is None:
                continue
            distractor = await self.plan_distractor(primary_plan, distractor_operator, rng)
            if distractor is None:
                continue
            edit, distractor_addr = distractor
            return await self.execute(
                primary_plan,
                samples=samples,
                control=control,
                distractor_edit=edit,
                distractor_addr=distractor_addr,
                distractor_fault_code=distractor_operator.spec.code,
            )

        return None
