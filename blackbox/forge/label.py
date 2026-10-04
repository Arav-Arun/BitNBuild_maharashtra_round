"""Fork labeling: classify replay outcomes as POSITIVE, RECOVERED, FLAKY, or UNSTABLE.

A label is a property of the whole fork, not of one sample: POSITIVE needs every
edited sample to fail and every paired control to pass, so a single unlucky sample
can never become a training example.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from blackbox.replay import ReplayBatch


class ForkLabel(str, Enum):
    """Outcome classification of a fault-injection fork."""

    POSITIVE = "positive"
    """Every edited sample fails and every paired control passes."""

    RECOVERED = "recovered"
    """Every edited sample still passes despite the fault — a hard negative."""

    FLAKY = "flaky"
    """At least one no-edit control fails, so the base pass is not reproducible."""

    UNSTABLE = "unstable"
    """Controls pass but edited samples disagree; discarded like FLAKY."""


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
    confidence: str = "high"
    # Which of several independent edits at the same step this fork is (0 for the first).
    variant: int = 0


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
        controls_reproduce = batch.control_pass_rate is None or batch.control_pass_rate == 1.0
        if not controls_reproduce:
            label = ForkLabel.FLAKY
        elif batch.fix_pass_rate == 0.0:
            label = ForkLabel.POSITIVE
        elif batch.fix_pass_rate == 1.0:
            label = ForkLabel.RECOVERED
        else:
            label = ForkLabel.UNSTABLE

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
            confidence="high" if len(batch.edited) >= 3 else "low",
        )

        # Every edited sample agrees for these labels, so each sample run carries
        # the fork's label; FLAKY and UNSTABLE forks never reach the labels table.
        if label in {ForkLabel.POSITIVE, ForkLabel.RECOVERED}:
            for run in batch.edited:
                self._persist_label(
                    run.run_id,
                    target_addr,
                    fault_type,
                    manifest_addr,
                    recovered=int(label == ForkLabel.RECOVERED),
                )

        return result

    def _find_manifestation(self, batch: ReplayBatch, target_addr: str) -> str | None:
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
            addr
            for addr, status in statuses.items()
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
