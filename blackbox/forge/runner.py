"""Bulk async runner with concurrency cap, progress logging, and resumability."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random as random_module
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from blackbox.forge.inject import FaultInjector
from blackbox.forge.label import ForkLabel, ForkResult
from blackbox.forge.operators import FaultOperator, all_operators, seen_operators

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ForgeProgress:
    """Running statistics for a forge session."""

    total_attempts: int = 0
    positive: int = 0
    recovered: int = 0
    flaky: int = 0
    errors: int = 0
    per_operator: dict[str, Counter] = field(default_factory=lambda: {})
    results: list[ForkResult] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)

    def record(self, result: ForkResult) -> None:
        self.total_attempts += 1
        self.results.append(result)
        if result.label == ForkLabel.POSITIVE:
            self.positive += 1
        elif result.label == ForkLabel.RECOVERED:
            self.recovered += 1
        elif result.label == ForkLabel.FLAKY:
            self.flaky += 1
        counter = self.per_operator.setdefault(result.fault_code, Counter())
        counter[result.label.value] += 1

    def record_error(self) -> None:
        self.total_attempts += 1
        self.errors += 1

    @property
    def flaky_rate(self) -> float:
        total = self.positive + self.recovered + self.flaky
        return self.flaky / total if total > 0 else 0.0

    @property
    def elapsed(self) -> float:
        return time.time() - self.started_at

    def summary(self) -> dict[str, Any]:
        return {
            "total_attempts": self.total_attempts,
            "positive": self.positive,
            "recovered": self.recovered,
            "flaky": self.flaky,
            "errors": self.errors,
            "flaky_rate": round(self.flaky_rate, 4),
            "elapsed_seconds": round(self.elapsed, 1),
            "per_operator": {
                code: dict(counts)
                for code, counts in self.per_operator.items()
            },
        }


class ForgeRunner:
    """Async bulk runner for fault injection experiments.

    Features:
    - Concurrency cap to respect API rate limits
    - Resumable: saves progress to a checkpoint file
    - Progress logging every N attempts
    - Distractor injection for 30% of positive forks
    """

    def __init__(
        self,
        injector: FaultInjector,
        *,
        concurrency: int = 4,
        checkpoint_dir: Path | None = None,
        distractor_rate: float = 0.3,
        seed: int = 42,
    ) -> None:
        self.injector = injector
        self.concurrency = concurrency
        self.checkpoint_dir = checkpoint_dir
        self.distractor_rate = distractor_rate
        self.rng = random_module.Random(seed)
        self._semaphore = asyncio.Semaphore(concurrency)
        self.progress = ForgeProgress()

    def _checkpoint_path(self) -> Path | None:
        if self.checkpoint_dir is None:
            return None
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return self.checkpoint_dir / "forge_checkpoint.json"

    def _save_checkpoint(self) -> None:
        path = self._checkpoint_path()
        if path is None:
            return
        data = {
            "summary": self.progress.summary(),
            "completed_fork_ids": [r.fork_id for r in self.progress.results],
            "saved_at": datetime.now(UTC).isoformat(),
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load_completed(self) -> set[str]:
        path = self._checkpoint_path()
        if path is None or not path.exists():
            return set()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return set(data.get("completed_fork_ids", []))
        except (json.JSONDecodeError, KeyError):
            return set()

    async def _run_one(
        self,
        operator: FaultOperator,
        *,
        agent: str | None = None,
        samples: int = 1,
        control: bool = True,
        with_distractor: bool = False,
    ) -> ForkResult | None:
        """Run a single fault injection experiment under the semaphore."""
        async with self._semaphore:
            try:
                if with_distractor:
                    from blackbox.forge.operators import T4Timeout500, C4RepeatedLoop
                    distractor_ops = [T4Timeout500(), C4RepeatedLoop()]
                    dist_op = self.rng.choice(distractor_ops)
                    result = await self.injector.inject_with_distractor(
                        operator,
                        dist_op,
                        self.rng,
                        agent=agent,
                        samples=samples,
                        control=control,
                    )
                else:
                    result = await self.injector.inject_random(
                        operator,
                        self.rng,
                        agent=agent,
                        samples=samples,
                        control=control,
                    )

                if result is not None:
                    self.progress.record(result)
                else:
                    self.progress.record_error()

                # Log progress every 10 attempts
                if self.progress.total_attempts % 10 == 0:
                    logger.info(
                        "Forge progress: %d attempts, %d positive, %d recovered, "
                        "%d flaky, %d errors (%.1fs)",
                        self.progress.total_attempts,
                        self.progress.positive,
                        self.progress.recovered,
                        self.progress.flaky,
                        self.progress.errors,
                        self.progress.elapsed,
                    )
                    self._save_checkpoint()

                return result
            except Exception:
                self.progress.record_error()
                logger.exception("Forge error with operator %s", operator.spec.code)
                return None

    async def run(
        self,
        *,
        target_positive: int = 360,
        max_attempts: int = 1000,
        operators: list[FaultOperator] | None = None,
        agent: str | None = None,
        samples: int = 1,
        control: bool = True,
    ) -> ForgeProgress:
        """Run fault injection experiments until the target is met.

        Distributes attempts across operators in round-robin fashion.
        Adds distractors to 30% of positive results.
        """
        operators = operators or all_operators()
        if not operators:
            raise ValueError("no operators provided")

        logger.info(
            "Starting Forge: target=%d positive, max=%d attempts, "
            "%d operators, concurrency=%d",
            target_positive,
            max_attempts,
            len(operators),
            self.concurrency,
        )

        attempt = 0
        op_index = 0

        while self.progress.positive < target_positive and attempt < max_attempts:
            operator = operators[op_index % len(operators)]
            op_index += 1
            attempt += 1

            # 30% chance of adding a distractor
            with_distractor = self.rng.random() < self.distractor_rate

            await self._run_one(
                operator,
                agent=agent,
                samples=samples,
                control=control,
                with_distractor=with_distractor,
            )

        self._save_checkpoint()
        logger.info("Forge complete: %s", json.dumps(self.progress.summary(), indent=2))
        return self.progress

    async def run_per_operator(
        self,
        *,
        count_per_operator: int = 20,
        operators: list[FaultOperator] | None = None,
        agent: str | None = None,
        samples: int = 1,
        control: bool = True,
    ) -> ForgeProgress:
        """Run a fixed number of forks per operator for testing."""
        operators = operators or all_operators()
        logger.info(
            "Forge per-operator: %d per op, %d operators",
            count_per_operator,
            len(operators),
        )

        for operator in operators:
            for i in range(count_per_operator):
                await self._run_one(
                    operator,
                    agent=agent,
                    samples=samples,
                    control=control,
                )

        self._save_checkpoint()
        logger.info("Forge per-operator complete: %s", json.dumps(self.progress.summary()))
        return self.progress

    def freeze_dataset(self, output_dir: Path) -> str:
        """Export the dataset to Parquet and JSON, ensuring paired seeds and counts.

        Returns the dataset version hash.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export enriched labels
        labels = self.injector.recorder.database.query(
            "SELECT * FROM labels ORDER BY run_id"
        )
        enriched_labels = []
        for row in labels:
            item = dict(row)
            run_id = row["run_id"]
            run_row = self.injector.recorder.database.one(
                "SELECT parent_run_id, fork_id, seed FROM runs WHERE run_id = ?",
                (run_id,),
            )
            base_run_id = None
            seed_id = None
            repro_count = 1
            ctrl_count = 1
            fix_rate = None
            ctrl_rate = None

            if run_row:
                seed_id = run_row.get("seed")
                fork_id = run_row.get("fork_id")
                if fork_id:
                    fork_row = self.injector.recorder.database.one(
                        "SELECT base_run_id, samples, fix_pass_rate, control_pass_rate FROM forks WHERE fork_id = ?",
                        (fork_id,),
                    )
                    if fork_row:
                        base_run_id = fork_row.get("base_run_id")
                        repro_count = fork_row.get("samples") or 1
                        ctrl_count = repro_count
                        fix_rate = fork_row.get("fix_pass_rate")
                        ctrl_rate = fork_row.get("control_pass_rate")
                if not base_run_id:
                    base_run_id = run_row.get("parent_run_id")

            # Fallback to in-memory progress results
            for res in self.progress.results:
                if run_row and res.fork_id == run_row.get("fork_id"):
                    base_run_id = base_run_id or res.base_run_id
                    repro_count = res.samples
                    ctrl_count = res.samples
                    fix_rate = res.fix_pass_rate if fix_rate is None else fix_rate
                    ctrl_rate = res.control_pass_rate if ctrl_rate is None else ctrl_rate
                    break

            item["base_run_id"] = base_run_id
            item["seed_id"] = seed_id
            item["reproduction_count"] = repro_count
            item["control_count"] = ctrl_count
            item["fix_pass_rate"] = fix_rate
            item["control_pass_rate"] = ctrl_rate
            enriched_labels.append(item)

        labels_path = output_dir / "labels.json"
        labels_path.write_text(
            json.dumps(enriched_labels, indent=2, default=str),
            encoding="utf-8",
        )

        # Export fork results
        results_data = [asdict(r) for r in self.progress.results]
        results_path = output_dir / "forge_results.json"
        results_path.write_text(
            json.dumps(results_data, indent=2, default=str),
            encoding="utf-8",
        )

        # Export summary
        summary_path = output_dir / "forge_summary.json"
        summary_path.write_text(
            json.dumps(self.progress.summary(), indent=2),
            encoding="utf-8",
        )

        # Export Parquet if polars or pyarrow is available
        try:
            import polars as pl

            if enriched_labels:
                pl.DataFrame(enriched_labels).write_parquet(output_dir / "labels.parquet")
            if results_data:
                pl.DataFrame(results_data).write_parquet(output_dir / "forge_results.parquet")
        except Exception as exc:
            logger.debug("Parquet export skipped: %s", exc)

        # Compute dataset version hash
        combined = json.dumps(
            {"labels": enriched_labels, "results": results_data},
            sort_keys=True,
            default=str,
        ).encode("utf-8")
        version_hash = hashlib.sha256(combined).hexdigest()

        version_path = output_dir / "DATASET_VERSION"
        version_path.write_text(version_hash, encoding="utf-8")

        logger.info("Dataset frozen: %d labels, hash=%s", len(enriched_labels), version_hash[:16])
        return version_hash
