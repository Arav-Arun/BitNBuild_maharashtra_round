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

    def plan_injection(
        self,
        run_id: str,
        operator: FaultOperator,
        rng: random_module.Random,
        *,
        target_addr: str | None = None,
    ) -> InjectionPlan | None:
        """Create an injection plan for a specific run and operator.

        If target_addr is None, a random applicable step is chosen.
        Returns None if no applicable step is found.
        """
        steps = self._steps_for_run(run_id)
        if not steps:
            return None

        # Filter to applicable steps
        candidates = []
        for step in steps:
            if target_addr and step["addr"] != target_addr:
                continue
            output = self._load_output(step)
            if output is not None and operator.applicable(step, output):
                candidates.append((step, output))

        if not candidates:
            return None

        step, output = rng.choice(candidates)
        # Decision/coordination faults on a real model: let the LLM write the mistake
        # itself from a hidden instruction. Otherwise apply the operator's own edit.
        hint = operator.ghost_hint(step, rng)
        if self._uses_ghost_hint(run_id, step, hint):
            edit = ghost_hint(step["addr"], hint)
        else:
            edit = operator.apply(step, output, rng)

        return InjectionPlan(
            base_run_id=run_id,
            target_step=step,
            operator=operator,
            edit=edit,
            original_output=output,
        )

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
            # Plan primary fault
            primary_plan = self.plan_injection(run["run_id"], primary_operator, rng)
            if primary_plan is None:
                continue

            # Find a different step for the distractor
            steps = self._steps_for_run(run["run_id"])
            distractor_candidates = []
            for step in steps:
                if step["addr"] == primary_plan.target_step["addr"]:
                    continue
                output = self._load_output(step)
                if output is not None and distractor_operator.applicable(step, output):
                    distractor_candidates.append((step, output))

            if not distractor_candidates:
                continue

            dist_step, dist_output = rng.choice(distractor_candidates)
            dist_edit = distractor_operator.apply(dist_step, dist_output, rng)

            # Distractor control: verify distractor alone does not cause failure
            try:
                agent_fn = self.agent_fn_factory(run["run_id"])
                dist_batch = await self.engine.replay(
                    run["run_id"],
                    agent_fn,
                    edits=[dist_edit],
                    mode="cone",
                    samples=1,
                    control=False,
                    branch_name="distractor-check",
                )
                if dist_batch.fix_pass_rate < 1.0:
                    continue
            except Exception:
                continue

            return await self.execute(
                primary_plan,
                samples=samples,
                control=control,
                distractor_edit=dist_edit,
                distractor_addr=dist_step["addr"],
                distractor_fault_code=distractor_operator.spec.code,
            )

        return None
