"""The blame ranker: LightGBM LambdaRank + temperature calibration + conformal sets (Task 8).

Three layers turn trees into an answer a user can trust:

1. **Ranker.** ``LGBMRanker`` with the lambdarank objective learns to put the root step
   first within each failed run (a run is a "query", its steps are the "documents").
   Graded relevance (root 3, ±1 hop 2, ±2 hops 1, distractors 0) rewards near misses.
2. **Temperature.** Ranking scores are not probabilities. A per-run softmax over
   ``score / T``, with ``T`` fitted on validation runs by minimising the negative log
   likelihood of the true root, turns them into calibrated percentages.
3. **Conformal set.** Split conformal prediction on validation runs picks a probability
   threshold so that the set {steps with p >= threshold} contains the root in at least
   ``1 - alpha`` of exchangeable runs. A set larger than ``max_set`` means the evidence
   cannot isolate a culprit, and the diagnosis abstains instead of naming one.
"""

from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from blackbox.ml.dataset import RootLabel, Trace, graded_relevance
from blackbox.ml.features import (
    FEATURE_NAMES,
    GROUPS,
    Matrix,
    Reference,
    build_matrix,
)

MODEL_FORMAT = "blackbox-diagnoser-v2"


@dataclass(slots=True)
class DiagnoserConfig:
    groups: tuple[str, ...] = GROUPS
    view: str = "full"
    edges: str = "full"
    alpha: float = 0.1
    max_set: int = 3
    seed: int = 7
    n_estimators: int = 600
    learning_rate: float = 0.03
    num_leaves: int = 31
    min_child_samples: int = 10
    early_stopping_rounds: int = 50
    # Calibrate on out-of-fold training scores; 0 disables cross-fitting (val only).
    calibration_folds: int = 5
    # "fault": leave-one-fault-family-out scores (honest on unseen faults);
    # "task": task-grouped folds plus validation (in-distribution only).
    calibrate_on: str = "fault"


@dataclass(slots=True)
class Diagnosis:
    """What the API and UI show for one failed run."""

    run_id: str
    ranking: list[tuple[str, float]]  # (addr, calibrated probability), most blamed first
    conformal_set: list[str]
    abstain: bool
    model_version: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "ranking": [{"addr": a, "probability": round(p, 4)} for a, p in self.ranking],
            "conformal_set": self.conformal_set,
            "abstain": self.abstain,
            "model_version": self.model_version,
        }


def softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    z = scores / temperature
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def relevance_vector(matrix: Matrix, index: int, trace: Trace, label: RootLabel) -> np.ndarray:
    grades = graded_relevance(trace, label)
    return np.asarray([grades.get(addr, 0) for addr in matrix.addrs[index]], dtype=float)


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


@dataclass(slots=True)
class Diagnoser:
    reference: Reference
    config: DiagnoserConfig = field(default_factory=DiagnoserConfig)
    booster: Any = None
    temperature: float = 1.0
    threshold: float = 0.0
    best_iteration: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- features ---------------------------------------------------------------

    def matrix(self, traces: Sequence[Trace]) -> Matrix:
        return build_matrix(
            traces,
            self.reference,
            view=self.config.view,
            edges=self.config.edges,
            groups=self.config.groups,
        )

    # -- training ---------------------------------------------------------------

    def fit(
        self,
        train: Sequence[tuple[Trace, RootLabel]],
        val: Sequence[tuple[Trace, RootLabel]],
        *,
        relevance_override: Mapping[str, np.ndarray] | None = None,
    ) -> Diagnoser:
        """Train on ``train``; early-stop on ``val``; calibrate on held-out scores.

        ``relevance_override`` maps run IDs to replacement relevance vectors, for train
        and val alike (the shuffled-label check must not early-stop on real labels).
        """
        from lightgbm import early_stopping, log_evaluation

        override = relevance_override or {}
        train_matrix = self.matrix([t for t, _ in train])
        val_matrix = self.matrix([t for t, _ in val])

        def relevance(matrix: Matrix, pairs: Sequence[tuple[Trace, RootLabel]]) -> np.ndarray:
            return np.concatenate(
                [
                    override[t.run_id]
                    if t.run_id in override
                    else relevance_vector(matrix, i, t, label)
                    for i, (t, label) in enumerate(pairs)
                ]
            )

        y_train = relevance(train_matrix, train)
        y_val = relevance(val_matrix, val)
        ranker = self._ranker(self.config.n_estimators)
        ranker.fit(
            train_matrix.X,
            y_train,
            group=train_matrix.group_sizes,
            eval_X=(val_matrix.X,),
            eval_y=(y_val,),
            eval_group=[val_matrix.group_sizes],
            eval_at=[1, 3],
            callbacks=[
                early_stopping(
                    self.config.early_stopping_rounds, first_metric_only=True, verbose=False
                ),
                log_evaluation(0),
            ],
        )
        self.booster = ranker.booster_
        self.best_iteration = ranker.best_iteration_ or None

        scores, addrs, labels = [], [], []
        if self.config.calibration_folds >= 2:
            scores, addrs, labels = self._out_of_fold(train, train_matrix, y_train)
        if self.config.calibrate_on != "fault" or not scores:
            # In-distribution calibration: validation scores plus task-grouped folds.
            scores += self._raw_scores(val_matrix)
            addrs += list(val_matrix.addrs)
            labels += [label for _, label in val]
        self._calibrate(scores, addrs, labels)
        self.metadata.update(
            n_train_runs=len(train),
            n_val_runs=len(val),
            n_calibration_runs=len(labels),
            trained_at=datetime.now(UTC).isoformat(),
            git_commit=git_commit(),
        )
        return self

    def _ranker(self, n_estimators: int) -> Any:
        from lightgbm import LGBMRanker

        return LGBMRanker(
            objective="lambdarank",
            n_estimators=n_estimators,
            learning_rate=self.config.learning_rate,
            num_leaves=self.config.num_leaves,
            min_child_samples=self.config.min_child_samples,
            colsample_bytree=0.8,
            subsample=0.8,
            subsample_freq=1,
            lambdarank_truncation_level=5,
            random_state=self.config.seed,
            verbose=-1,
        )

    def _out_of_fold(
        self, train: Sequence[tuple[Trace, RootLabel]], matrix: Matrix, y: np.ndarray
    ) -> tuple[list[np.ndarray], list[list[str]], list[RootLabel]]:
        """Score each training run with a model that never saw its group (cross-fitting).

        With ``calibrate_on="fault"`` (the default) each fold holds out whole fault types,
        so every calibration score comes from a model meeting that fault family for the
        first time: the situation S1 and real failures put it in. Calibrating only on
        familiar faults makes the model confidently wrong on new ones. With
        ``calibrate_on="task"`` folds hold out tasks instead (in-distribution calibration).
        Each fold model uses the early-stopped iteration count.
        """
        if self.config.calibrate_on == "fault":
            key = [label.fault_type for _, label in train]
        else:
            key = [t.task_id for t, _ in train]
        groups = sorted(set(key))
        folds = min(self.config.calibration_folds, len(groups))
        if self.config.calibrate_on == "fault":
            folds = len(groups)  # leave one fault family out
        if folds < 2:
            return [], [], []
        fold_of = {group: i % folds for i, group in enumerate(groups)}
        rows = [np.arange(matrix.offsets[i], matrix.offsets[i + 1]) for i in range(len(train))]
        scores: list[np.ndarray] = []
        addrs: list[list[str]] = []
        labels: list[RootLabel] = []
        for fold in range(folds):
            inside = [i for i in range(len(train)) if fold_of[key[i]] != fold]
            outside = [i for i in range(len(train)) if fold_of[key[i]] == fold]
            if not inside or not outside:
                continue
            index = np.concatenate([rows[i] for i in inside])
            model = self._ranker(self.best_iteration or self.config.n_estimators)
            model.fit(matrix.X[index], y[index], group=[len(rows[i]) for i in inside])
            for i in outside:
                scores.append(model.predict(matrix.X[rows[i]]))
                addrs.append(matrix.addrs[i])
                labels.append(train[i][1])
        return scores, addrs, labels

    def _raw_scores(self, matrix: Matrix) -> list[np.ndarray]:
        if matrix.X.shape[0] == 0:
            return []
        flat = self.booster.predict(matrix.X, num_iteration=self.best_iteration)
        return [flat[matrix.run_slice(i)] for i in range(len(matrix.run_ids))]

    def _calibrate(
        self,
        scores: Sequence[np.ndarray],
        addrs: Sequence[list[str]],
        labels: Sequence[RootLabel],
    ) -> None:
        roots = [
            a.index(label.root_addr) for a, label in zip(addrs, labels) if label.root_addr in a
        ]
        kept = [s for s, a, label in zip(scores, addrs, labels) if label.root_addr in a]
        best = (math.inf, 1.0)
        for temperature in np.geomspace(0.02, 50, 240):
            nll = -np.mean(
                [math.log(softmax(s, temperature)[r] + 1e-12) for s, r in zip(kept, roots)]
            )
            best = min(best, (nll, float(temperature)))
        self.temperature = best[1]
        # Split conformal (LAC): nonconformity = 1 - p(root). The finite-sample quantile
        # gives >= 1 - alpha coverage on runs exchangeable with the calibration runs.
        nonconformity = sorted(1 - softmax(s, self.temperature)[r] for s, r in zip(kept, roots))
        n = len(nonconformity)
        k = min(n, math.ceil((n + 1) * (1 - self.config.alpha)))
        quantile = nonconformity[k - 1] if n else 1.0
        self.threshold = 1 - quantile

    # -- inference --------------------------------------------------------------

    def diagnose_many(self, traces: Sequence[Trace]) -> list[Diagnosis]:
        matrix = self.matrix(traces)
        diagnoses = []
        version = self.metadata.get("model_version", MODEL_FORMAT)
        for i, scores in enumerate(self._raw_scores(matrix)):
            probabilities = softmax(scores, self.temperature)
            order = np.argsort(-probabilities, kind="stable")
            ranking = [(matrix.addrs[i][j], float(probabilities[j])) for j in order]
            members = [addr for addr, p in ranking if p >= self.threshold] or [ranking[0][0]]
            diagnoses.append(
                Diagnosis(
                    run_id=matrix.run_ids[i],
                    ranking=ranking,
                    conformal_set=members,
                    abstain=len(members) > self.config.max_set,
                    model_version=version,
                )
            )
        return diagnoses

    def diagnose(self, trace: Trace) -> Diagnosis:
        return self.diagnose_many([trace])[0]

    def feature_importance(self) -> dict[str, float]:
        gains = self.booster.feature_importance(importance_type="gain")
        total = gains.sum() or 1.0
        return {name: float(g / total) for name, g in zip(FEATURE_NAMES, gains)}

    # -- persistence ------------------------------------------------------------

    def save(self, directory: Path, *, dataset_hash: str | None = None) -> None:
        """Model, feature list, T, conformal threshold, reference, dataset hash and commit."""
        directory.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(directory / "model.txt"), num_iteration=self.best_iteration)
        (directory / "reference.json").write_text(
            json.dumps(self.reference.to_dict(), sort_keys=True), encoding="utf-8"
        )
        meta = {
            "format": MODEL_FORMAT,
            "feature_names": list(FEATURE_NAMES),
            "config": {**asdict(self.config), "groups": list(self.config.groups)},
            "temperature": self.temperature,
            "conformal_threshold": self.threshold,
            "dataset_hash": dataset_hash,
            **self.metadata,
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> Diagnoser:
        from lightgbm import Booster

        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        if meta.get("format") != MODEL_FORMAT:
            raise ValueError(f"unsupported model format in {directory}")
        if meta["feature_names"] != list(FEATURE_NAMES):
            raise ValueError("model was trained with a different feature list; retrain it")
        config = DiagnoserConfig(**{**meta["config"], "groups": tuple(meta["config"]["groups"])})
        reference = Reference.from_dict(
            json.loads((directory / "reference.json").read_text(encoding="utf-8"))
        )
        diagnoser = cls(reference=reference, config=config)
        diagnoser.booster = Booster(model_file=str(directory / "model.txt"))
        diagnoser.temperature = meta["temperature"]
        diagnoser.threshold = meta["conformal_threshold"]
        diagnoser.metadata = {
            k: v
            for k, v in meta.items()
            if k not in {"format", "feature_names", "config", "temperature", "conformal_threshold"}
        }
        diagnoser.metadata.setdefault("model_version", directory.name)
        return diagnoser


# ---------------------------------------------------------------------------
# Run-level detector: "is this run likely to fail at all?" (the Runs page risk column)
# ---------------------------------------------------------------------------

RUN_AGGREGATES = (
    ("anomaly", "max"),
    ("anomaly", "sum"),
    ("status_error", "sum"),
    ("parse_failed", "sum"),
    ("num_max_absz", "max"),
    ("date_older_z", "max"),
    ("keys_missing_frac", "max"),
    ("io_conflicts", "max"),
    ("earlier_conflicts", "max"),
    ("travel_constraint_mismatch_count", "max"),
    ("travel_budget_math_error", "max"),
    ("travel_quote_age_days", "max"),
    ("travel_currency_conflict", "max"),
    ("n_steps", "max"),
)


def run_aggregates(matrix: Matrix) -> np.ndarray:
    columns = [(FEATURE_NAMES.index(name), op) for name, op in RUN_AGGREGATES]
    rows = []
    for i in range(len(matrix.run_ids)):
        block = matrix.X[matrix.run_slice(i)]
        rows.append(
            [
                float(np.nanmax(block[:, c]) if op == "max" else np.nansum(block[:, c]))
                if block.size and not np.all(np.isnan(block[:, c]))
                else math.nan
                for c, op in columns
            ]
        )
    return np.asarray(rows, dtype=float)


def train_run_detector(matrix: Matrix, failed: Iterable[int], seed: int = 7) -> Any:
    from lightgbm import LGBMClassifier

    failed = set(failed)
    y = np.asarray([int(i in failed) for i in range(len(matrix.run_ids))])
    classifier = LGBMClassifier(
        n_estimators=200, learning_rate=0.05, num_leaves=15, random_state=seed, verbose=-1
    )
    classifier.fit(run_aggregates(matrix), y)
    return classifier
