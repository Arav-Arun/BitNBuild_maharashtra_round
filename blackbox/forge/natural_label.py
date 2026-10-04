"""Natural failure labeling: attribute failures in uninjected runs.

These runs are test-only — they never enter training data.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from blackbox.replay import ReplayBatch, ReplayEngine, override_output
from blackbox.sdk import Recorder

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class NaturalLabel:
    """Attribution result for a naturally failed run.

    ``verdict`` is "attributed" (the edited lower bound beat the control's upper bound),
    "natural_inconclusive" (a candidate whose bounds still overlap, never truth) or
    "unattributable" (no oracle fix flipped the run). ``samples`` is the K of the deciding
    paired replay (0 when no step was promising enough to run it).
    """

    run_id: str
    candidate_addr: str | None
    verdict: str
    fix_pass_rate: float | None
    control_pass_rate: float | None
    fix_ci: tuple[float, float] | None
    control_ci: tuple[float, float] | None
    attempts: int
    created_at: str
    samples: int = 0

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> NaturalLabel:
        data = dict(data)
        for key in ("fix_ci", "control_ci"):
            if data.get(key) is not None:
                data[key] = tuple(data[key])
        return cls(**data)


class NaturalLedger:
    """Append-only record of every natural run examined, so a restart skips finished runs."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.labels: dict[str, NaturalLabel] = {}
        if path.exists():
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
                if not line.strip():
                    continue
                try:
                    label = NaturalLabel.from_json(json.loads(line))
                except (json.JSONDecodeError, TypeError, ValueError):
                    # A crash mid-write can truncate only the final line.
                    logger.warning(
                        "Skipping unreadable ledger line %d in %s", line_number + 1, path
                    )
                    continue
                self.labels[label.run_id] = label

    def append(self, label: NaturalLabel) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(asdict(label)) + "\n")
        self.labels[label.run_id] = label


class NaturalLabeler:
    """Attempt to find the root cause of a naturally failed run.

    Strategy (from the plan):
    1. Get gold answers (solver output for TripCrew, sub-answers for HopRAG).
    2. Propose a minimal fix at each candidate step, earliest first.
    3. Screen candidates at K=3, then run K=5 on the earliest promising step.
    4. Attributed only if edited lower bound > control upper bound.
    5. If no step flips it, the run is marked 'unattributable'.
    """

    def __init__(
        self,
        recorder: Recorder,
        agent_fn_factory: Any,
        oracle_fn: Any | None = None,
        ledger: NaturalLedger | None = None,
    ) -> None:
        self.recorder = recorder
        self.engine = ReplayEngine(recorder)
        self.agent_fn_factory = agent_fn_factory
        self.oracle_fn = oracle_fn
        self.ledger = ledger

    def _failed_runs(self, agent: str | None = None) -> list[dict[str, Any]]:
        """Query naturally failed base runs (not fork children)."""
        query = (
            "SELECT * FROM runs WHERE outcome = 'failed' "
            "AND fork_id IS NULL AND parent_run_id IS NULL"
        )
        params: list[Any] = []
        if agent:
            query += " AND agent = ?"
            params.append(agent)
        return self.recorder.database.query(query, params)

    def _steps_for_run(self, run_id: str) -> list[dict[str, Any]]:
        return self.recorder.database.query(
            "SELECT * FROM steps WHERE run_id = ? ORDER BY seq",
            (run_id,),
        )

    def _load_output(self, step: dict[str, Any]) -> Any:
        output_hash = step.get("output_hash")
        if not output_hash:
            return None
        try:
            return self.recorder.store.load_json(output_hash)
        except (OSError, ValueError):
            return None

    async def label_run(
        self,
        run_id: str,
        oracle_fixes: dict[str, Any] | None = None,
    ) -> NaturalLabel:
        """Try to attribute a naturally failed run to a specific step.

        oracle_fixes: mapping from step addr to the correct output value,
        obtained from the oracle (solver or gold sub-answers).
        """
        steps = self._steps_for_run(run_id)
        if not steps:
            return NaturalLabel(
                run_id=run_id,
                candidate_addr=None,
                verdict="unattributable",
                fix_pass_rate=None,
                control_pass_rate=None,
                fix_ci=None,
                control_ci=None,
                attempts=0,
                created_at=datetime.now(UTC).isoformat(),
            )

        oracle_fixes = oracle_fixes or {}
        agent_fn = self.agent_fn_factory(run_id)
        attempts = 0
        inconclusive_step: tuple[str, ReplayBatch] | None = None

        # Try each step, earliest first
        for step in steps:
            addr = step["addr"]
            fix_value = oracle_fixes.get(addr)
            if fix_value is None:
                continue

            attempts += 1

            # Screen at K=3
            try:
                screen = await self.engine.replay(
                    run_id,
                    agent_fn,
                    edits=[override_output(addr, fix_value, known_good=True)],
                    mode="cone",
                    samples=3,
                    control=True,
                    branch_name=f"natural-screen-{addr}",
                )
            except Exception:
                logger.debug("Screen replay failed at %s", addr, exc_info=True)
                continue

            # If fix doesn't help at all, skip
            if screen.fix_pass_rate == 0:
                continue

            # Promising — run K=5 for statistical power
            try:
                full = await self.engine.replay(
                    run_id,
                    agent_fn,
                    edits=[override_output(addr, fix_value, known_good=True)],
                    mode="cone",
                    samples=5,
                    control=True,
                    branch_name=f"natural-full-{addr}",
                )
            except Exception:
                logger.debug("Full replay failed at %s", addr, exc_info=True)
                continue

            # Check if fix lower bound > control upper bound
            if full.fix_interval[0] > (full.control_interval[1] if full.control_interval else 0):
                # Attributed!
                self._persist_label(run_id, addr)
                return NaturalLabel(
                    run_id=run_id,
                    candidate_addr=addr,
                    verdict="attributed",
                    fix_pass_rate=full.fix_pass_rate,
                    control_pass_rate=full.control_pass_rate,
                    fix_ci=full.fix_interval,
                    control_ci=full.control_interval,
                    attempts=attempts,
                    created_at=datetime.now(UTC).isoformat(),
                    samples=len(full.edited),
                )

            # Overlapping bounds — record as inconclusive
            logger.debug(
                "Natural label at %s: inconclusive (fix_ci=%s, ctrl_ci=%s)",
                addr,
                full.fix_interval,
                full.control_interval,
            )
            if inconclusive_step is None:
                inconclusive_step = (addr, full)

        if inconclusive_step is not None:
            inc_addr, inc_batch = inconclusive_step
            return NaturalLabel(
                run_id=run_id,
                candidate_addr=inc_addr,
                verdict="natural_inconclusive",
                fix_pass_rate=inc_batch.fix_pass_rate,
                control_pass_rate=inc_batch.control_pass_rate,
                fix_ci=inc_batch.fix_interval,
                control_ci=inc_batch.control_interval,
                attempts=attempts,
                created_at=datetime.now(UTC).isoformat(),
                samples=len(inc_batch.edited),
            )

        # No step flipped the run
        return NaturalLabel(
            run_id=run_id,
            candidate_addr=None,
            verdict="unattributable",
            fix_pass_rate=None,
            control_pass_rate=None,
            fix_ci=None,
            control_ci=None,
            attempts=attempts,
            created_at=datetime.now(UTC).isoformat(),
        )

    def _persist_label(self, run_id: str, root_addr: str) -> None:
        """Write a natural_auto label for the attributed run."""
        self.recorder.database.execute(
            """
            INSERT OR IGNORE INTO labels(
                run_id, root_addr, fault_type, source, recovered, manifest_addr, verified
            ) VALUES (?, ?, 'natural', 'natural_auto', 0, NULL, 1)
            """,
            (run_id, root_addr),
        )

    async def label_all(
        self,
        agent: str | None = None,
        oracle_fixes_fn: Any | None = None,
        *,
        limit: int | None = None,
        select_seed: str = "natural",
    ) -> list[NaturalLabel]:
        """Label naturally failed runs of an agent; returns the labels made by this call.

        ``limit`` caps how many failed runs are examined in total, counting those already in
        the ledger. Runs are taken in a seeded pseudo-random order, so a capped sample does not
        depend on recording order and a restarted session continues the same sample.
        """
        results = []
        runs = sorted(
            self._failed_runs(agent),
            key=lambda row: hashlib.sha256(f"{select_seed}:{row['run_id']}".encode()).hexdigest(),
        )
        examined = len(self.ledger.labels) if self.ledger is not None else 0
        logger.info("Found %d naturally failed runs to label", len(runs))

        for run in runs:
            if limit is not None and examined >= limit:
                break
            if self.ledger is not None and run["run_id"] in self.ledger.labels:
                continue
            # Check if already labeled
            existing = self.recorder.database.one(
                "SELECT * FROM labels WHERE run_id = ?", (run["run_id"],)
            )
            if existing:
                continue

            oracle_fixes = {}
            if oracle_fixes_fn:
                oracle_fixes = oracle_fixes_fn(run["run_id"])

            result = await self.label_run(run["run_id"], oracle_fixes)
            results.append(result)
            examined += 1
            if self.ledger is not None:
                self.ledger.append(result)
            logger.info(
                "Natural label for %s: %s (addr=%s, attempts=%d)",
                run["run_id"],
                result.verdict,
                result.candidate_addr,
                result.attempts,
            )

        return results
