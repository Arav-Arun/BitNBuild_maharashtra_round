"""Fork labeling: classify replay outcomes as POSITIVE, RECOVERED, or FLAKY."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from blackbox.replay import ReplayBatch


class ForkLabel(str, Enum):
    """Outcome classification of a fault-injection fork."""

    POSITIVE = "positive"
    """The edited fork reproducibly fails and the paired control passes."""

    RECOVERED = "recovered"
    """The fork still passes despite the fault — a hard negative."""

    FLAKY = "flaky"
    """The no-edit control also fails, so the example is unreliable."""


@dataclass(frozen=True, slots=True)
class ForkResult:
    """Complete record of a single fault-injection experiment."""

    base_run_id: str
    fork_id: str
    target_addr: str
    fault_code: str
    fault_type: str
    label: ForkLabel
    fix_pass_rate: float
    control_pass_rate: float | None
    manifest_addr: str | None
    distractor_addr: str | None
    distractor_fault_code: str | None
    samples: int
    held_out: bool
    created_at: str


class Labeler:
    """Classify a ReplayBatch from a fault injection fork."""

    def __init__(self, database: Any) -> None:
        self.database = database

    def classify(
        self,
        batch: ReplayBatch,
        *,
        target_addr: str,
        fault_code: str,
        fault_type: str,
        held_out: bool = False,
        distractor_addr: str | None = None,
        distractor_fault_code: str | None = None,
    ) -> ForkResult:
        """Classify the batch and persist the label to the database."""
        # Determine label
        fix_failed = batch.fix_pass_rate < 1.0
        control_passed = True
        if batch.control_pass_rate is not None:
            control_passed = batch.control_pass_rate > 0.0

        if not control_passed:
            label = ForkLabel.FLAKY
        elif fix_failed:
            label = ForkLabel.POSITIVE
        else:
            label = ForkLabel.RECOVERED

        # Find manifestation step: first step that diverged from original
        manifest_addr = self._find_manifestation(batch, target_addr)

        result = ForkResult(
            base_run_id=batch.base_run_id,
            fork_id=batch.fork_id,
            target_addr=target_addr,
            fault_code=fault_code,
            fault_type=fault_type,
            label=label,
            fix_pass_rate=batch.fix_pass_rate,
            control_pass_rate=batch.control_pass_rate,
            manifest_addr=manifest_addr,
            distractor_addr=distractor_addr,
            distractor_fault_code=distractor_fault_code,
            samples=len(batch.edited),
            held_out=held_out,
            created_at=datetime.now(UTC).isoformat(),
        )

        # Persist label to the labels table for each edited run
        if label == ForkLabel.POSITIVE:
            for run in batch.edited:
                if run.outcome == "failed":
                    self._persist_label(
                        run.run_id,
                        target_addr,
                        fault_type,
                        manifest_addr,
                        recovered=0,
                    )
        elif label == ForkLabel.RECOVERED:
            for run in batch.edited:
                if run.outcome == "passed":
                    self._persist_label(
                        run.run_id,
                        target_addr,
                        fault_type,
                        manifest_addr,
                        recovered=1,
                    )

        return result

    def _find_manifestation(
        self, batch: ReplayBatch, target_addr: str
    ) -> str | None:
        """Find the first step after the target whose cache status is 'live'."""
        if not batch.edited:
            return None
        statuses = batch.edited[0].statuses
        base_steps = self.database.query(
            "SELECT addr, seq FROM steps WHERE run_id = ? ORDER BY seq",
            (batch.base_run_id,),
        )
        addr_to_seq = {s["addr"]: s["seq"] for s in base_steps}
        target_seq = addr_to_seq.get(target_addr, -1)

        # Steps that ran live after the target step
        live_addrs = [
            addr for addr, status in statuses.items()
            if status in {"live", "edited"}
            and addr != target_addr
            and addr_to_seq.get(addr, 999999) > target_seq
        ]
        if not live_addrs:
            return None
        live_with_seq = [(addr, addr_to_seq.get(addr, 999999)) for addr in live_addrs]
        live_with_seq.sort(key=lambda x: x[1])
        return live_with_seq[0][0] if live_with_seq else None

    def _persist_label(
        self,
        run_id: str,
        root_addr: str,
        fault_type: str,
        manifest_addr: str | None,
        recovered: int = 0,
    ) -> None:
        """Write a label row for the fork run."""
        self.database.execute(
            """
            INSERT OR IGNORE INTO labels(
                run_id, root_addr, fault_type, source, recovered, manifest_addr, verified
            ) VALUES (?, ?, ?, 'injected', ?, ?, 0)
            """,
            (run_id, root_addr, fault_type, recovered, manifest_addr),
        )

