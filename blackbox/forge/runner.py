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
from blackbox.forge.operators import FaultOperator, all_operators

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ForgeProgress:
    """Running statistics for a forge session."""

    total_attempts: int = 0
    positive: int = 0
    recovered: int = 0
    flaky: int = 0
    unstable: int = 0
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
        elif result.label == ForkLabel.UNSTABLE:
            self.unstable += 1
        counter = self.per_operator.setdefault(result.fault_code, Counter())
        counter[result.label.value] += 1

    def record_error(self) -> None:
        self.total_attempts += 1
        self.errors += 1

    @property
    def flaky_rate(self) -> float:
        total = self.positive + self.recovered + self.flaky + self.unstable
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
            "unstable": self.unstable,
            "errors": self.errors,
            "flaky_rate": round(self.flaky_rate, 4),
            "elapsed_seconds": round(self.elapsed, 1),
            "per_operator": {code: dict(counts) for code, counts in self.per_operator.items()},
        }


def _result_from_json(data: dict[str, Any]) -> ForkResult:
    return ForkResult(**{**data, "label": ForkLabel(data["label"])})


class ForgeRunner:
    """Async bulk runner for fault injection experiments.

    Features:
    - Up to ``concurrency`` replays in flight at once
    - Resumable: every result is appended to ``forge_results.jsonl`` as soon as it
      finishes, and a new runner on the same checkpoint directory continues from it
    - Progress logging every 10 attempts
    - Distractor injection for 30% of attempts
    """

    def __init__(
        self,
        injector: FaultInjector,
        *,
        concurrency: int = 4,
        checkpoint_dir: Path | None = None,
        distractor_rate: float = 0.3,
        seed: int = 42,
        resume: bool = True,
    ) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be positive")
        self.injector = injector
        self.concurrency = concurrency
        self.checkpoint_dir = checkpoint_dir
        self.distractor_rate = distractor_rate
        self.seed = seed
        self._semaphore = asyncio.Semaphore(concurrency)
        self.progress = ForgeProgress()
        if resume:
            self._resume()
        else:
            self._discard_checkpoint()
        # Reseed from the resume point so a restarted session explores new forks
        # instead of replaying the random choices of the first session.
        self.rng = random_module.Random(f"{seed}:{self.progress.total_attempts}")

    def _checkpoint_path(self) -> Path | None:
        if self.checkpoint_dir is None:
            return None
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return self.checkpoint_dir / "forge_checkpoint.json"

    def _results_path(self) -> Path | None:
        if self.checkpoint_dir is None:
            return None
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return self.checkpoint_dir / "forge_results.jsonl"

    def _discard_checkpoint(self) -> None:
        for path in (self._checkpoint_path(), self._results_path()):
            if path is not None and path.exists():
                path.unlink()

    def _resume(self) -> None:
        results_path = self._results_path()
        if results_path is None or not results_path.exists():
            return
        for line_number, line in enumerate(results_path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            try:
                self.progress.record(_result_from_json(json.loads(line)))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                # A crash mid-write can truncate only the final line.
                logger.warning(
                    "Skipping unreadable result line %d in %s", line_number + 1, results_path
                )
        checkpoint_path = self._checkpoint_path()
        if checkpoint_path is not None and checkpoint_path.exists():
            try:
                saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                errors = int(saved.get("summary", {}).get("errors", 0))
            except (json.JSONDecodeError, TypeError, ValueError):
                errors = 0
            self.progress.errors += errors
            self.progress.total_attempts += errors
        logger.info(
            "Resuming Forge from %s: %d results (%d positive), %d errors",
            results_path,
            len(self.progress.results),
            self.progress.positive,
            self.progress.errors,
        )

    def _append_result(self, result: ForkResult) -> None:
        path = self._results_path()
        if path is None:
            return
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(asdict(result), default=str) + "\n")

    def _save_checkpoint(self) -> None:
        path = self._checkpoint_path()
        if path is None:
            return
        data = {
            "summary": self.progress.summary(),
            "completed_fork_ids": [r.fork_id for r in self.progress.results],
            "saved_at": datetime.now(UTC).isoformat(),
        }
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
        temporary.replace(path)

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
                    from blackbox.forge.operators import C4RepeatedLoop, T4Timeout500

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
            except Exception:
                result = None
                logger.exception("Forge error with operator %s", operator.spec.code)

            if result is not None:
                self.progress.record(result)
                self._append_result(result)
            else:
                self.progress.record_error()
            self._save_checkpoint()

            # Log progress every 10 attempts
            if self.progress.total_attempts % 10 == 0:
                logger.info(
                    "Forge progress: %d attempts, %d positive, %d recovered, "
                    "%d flaky, %d unstable, %d errors (%.1fs)",
                    self.progress.total_attempts,
                    self.progress.positive,
                    self.progress.recovered,
                    self.progress.flaky,
                    self.progress.unstable,
                    self.progress.errors,
                    self.progress.elapsed,
                )
            return result

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

        Both limits count every attempt recorded so far, including those of a
        resumed session. Operators rotate round-robin; up to ``concurrency``
        attempts run at once, and no new attempt starts once the target is met.
        """
        operators = operators or all_operators()
        if not operators:
            raise ValueError("no operators provided")

        logger.info(
            "Starting Forge: target=%d positive, max=%d attempts, %d operators, concurrency=%d",
            target_positive,
            max_attempts,
            len(operators),
            self.concurrency,
        )

        op_index = self.progress.total_attempts
        pending: set[asyncio.Task[ForkResult | None]] = set()
        try:
            while True:
                while (
                    len(pending) < self.concurrency
                    and self.progress.positive < target_positive
                    and self.progress.total_attempts + len(pending) < max_attempts
                ):
                    operator = operators[op_index % len(operators)]
                    op_index += 1
                    with_distractor = self.rng.random() < self.distractor_rate
                    pending.add(
                        asyncio.create_task(
                            self._run_one(
                                operator,
                                agent=agent,
                                samples=samples,
                                control=control,
                                with_distractor=with_distractor,
                            )
                        )
                    )
                if not pending:
                    break
                _, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

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

        await asyncio.gather(
            *(
                self._run_one(operator, agent=agent, samples=samples, control=control)
                for operator in operators
                for _ in range(count_per_operator)
            )
        )

        self._save_checkpoint()
        logger.info("Forge per-operator complete: %s", json.dumps(self.progress.summary()))
        return self.progress

    def freeze_dataset(self, output_dir: Path) -> str:
        """Export the dataset to Parquet and JSON, ensuring paired seeds and counts.

        Injected labels are exported only for forks this runner recorded, so labels
        left in the database by an abandoned session cannot leak into the freeze.
        Returns the dataset version hash.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        database = self.injector.recorder.database
        results_by_fork = {result.fork_id: result for result in self.progress.results}

        # Export enriched labels
        labels = database.query("SELECT * FROM labels ORDER BY run_id")
        enriched_labels = []
        for row in labels:
            item = dict(row)
            run_id = row["run_id"]
            run_row = database.one(
                "SELECT parent_run_id, fork_id, seed FROM runs WHERE run_id = ?",
                (run_id,),
            )
            fork_id = run_row.get("fork_id") if run_row else None
            if row["source"] == "injected" and fork_id not in results_by_fork:
                continue
            base_run_id = None
            seed_id = None
            repro_count = 1
            ctrl_count = 1
            fix_rate = None
            ctrl_rate = None

            if run_row:
                seed_id = run_row.get("seed")
                if fork_id:
                    fork_row = database.one(
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

            result = results_by_fork.get(fork_id) if fork_id else None
            if result is not None:
                base_run_id = base_run_id or result.base_run_id
                repro_count = result.samples
                ctrl_count = result.samples
                fix_rate = result.fix_pass_rate if fix_rate is None else fix_rate
                ctrl_rate = result.control_pass_rate if ctrl_rate is None else ctrl_rate

            item["fork_id"] = fork_id
            item["base_run_id"] = base_run_id
            item["seed_id"] = seed_id
            item["reproduction_count"] = repro_count
            item["control_count"] = ctrl_count
            item["fix_pass_rate"] = fix_rate
            item["control_pass_rate"] = ctrl_rate
            item["confidence"] = "high" if repro_count >= 3 else "low"
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
                pl.DataFrame(
                    [{**row, "label": str(row["label"].value)} for row in results_data]
                ).write_parquet(output_dir / "forge_results.parquet")
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
