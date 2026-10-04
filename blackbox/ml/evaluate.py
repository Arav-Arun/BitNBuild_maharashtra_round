"""`make eval`: train, compare against baselines, ablate, and run integrity checks (Task 8).

Writes ``data/eval/*.json`` (read by ``GET /eval`` and the Results page) and saves the
model trained on the train split to ``data/models/diagnoser-v1``. Every number carries
its sample size and a 95% bootstrap interval; ablations report paired differences.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from blackbox.eval import baselines
from blackbox.eval.metrics import auroc, paired_difference, per_run, summarize
from blackbox.eval.splits import Splits, check_splits, leave_one_agent_out, make_splits
from blackbox.explain.precedents import Precedents
from blackbox.ml.dataset import Corpus, Trace, load_corpus
from blackbox.ml.features import (
    EDGE_VIEWS,
    GROUPS,
    OBSERVABILITY_VIEWS,
    Reference,
    build_matrix,
)
from blackbox.ml.model import (
    Diagnoser,
    DiagnoserConfig,
    relevance_vector,
    run_aggregates,
    train_run_detector,
)
from blackbox.store import SQLiteDatabase

logger = logging.getLogger(__name__)

GROUP_NAMES = {
    "A": "structure",
    "B": "telemetry",
    "C": "validity",
    "D": "grounding",
    "F": "novelty",
    "G": "context",
    "H": "lineage",
}


class Experiment:
    """One corpus, one split, one reference: everything trained here is paired."""

    def __init__(self, corpus: Corpus, splits: Splits, base: DiagnoserConfig | None = None) -> None:
        self.corpus = corpus
        self.splits = splits
        self.base = base or DiagnoserConfig()
        self.reference = Reference.fit(corpus.traces[r] for r in splits.reference)

    def pairs(self, run_ids: Sequence[str]):
        return [(self.corpus.traces[r], self.corpus.labels[r]) for r in run_ids]

    def train(self, relevance_override=None, **changes: Any) -> Diagnoser:
        """Train with the base config, overriding only ``changes`` (e.g. ``groups=("A",)``)."""
        diagnoser = Diagnoser(self.reference, replace(self.base, **changes))
        return diagnoser.fit(
            self.pairs(self.splits.train),
            self.pairs(self.splits.val),
            relevance_override=relevance_override,
        )

    def score(self, diagnoser: Diagnoser, run_ids: Sequence[str]) -> dict[str, dict[str, float]]:
        if not run_ids:
            return {}
        diagnoses = diagnoser.diagnose_many([self.corpus.traces[r] for r in run_ids])
        rankings = {d.run_id: [a for a, _ in d.ranking] for d in diagnoses}
        probabilities = {d.run_id: dict(d.ranking) for d in diagnoses}
        return per_run(rankings, self.corpus, probabilities)

    def matrix(self, run_ids: Sequence[str]):
        return build_matrix([self.corpus.traces[r] for r in run_ids], self.reference)


def _summaries(per_split: dict[str, dict[str, dict[str, float]]]) -> dict[str, Any]:
    return {split: summarize(values) for split, values in per_split.items() if values}


def _top1(summary: dict[str, Any], split: str) -> float | None:
    value = summary.get(split, {}).get("top1", {}).get("value")
    return None if value is None or math.isnan(value) else round(value, 4)


# ---------------------------------------------------------------------------
# Calibration and abstention
# ---------------------------------------------------------------------------


def calibration_report(diagnoser: Diagnoser, experiment: Experiment, run_ids: Sequence[str]):
    if not run_ids:
        return {"n": 0}
    diagnoses = diagnoser.diagnose_many([experiment.corpus.traces[r] for r in run_ids])
    confidences, correct, covered, sizes, abstained, selective = [], [], [], [], [], []
    for diagnosis in diagnoses:
        root = experiment.corpus.labels[diagnosis.run_id].root_addr
        top, probability = diagnosis.ranking[0]
        confidences.append(probability)
        correct.append(float(top == root))
        covered.append(float(root in diagnosis.conformal_set))
        sizes.append(len(diagnosis.conformal_set))
        abstained.append(float(diagnosis.abstain))
        if not diagnosis.abstain:
            selective.append(float(top == root))
    bins = np.linspace(0, 1, 11)
    reliability, ece = [], 0.0
    for low, high in zip(bins, bins[1:]):
        members = [
            i for i, c in enumerate(confidences) if low <= c < high or (high == 1 and c == 1)
        ]
        if not members:
            continue
        confidence = float(np.mean([confidences[i] for i in members]))
        accuracy = float(np.mean([correct[i] for i in members]))
        ece += len(members) / len(confidences) * abs(confidence - accuracy)
        reliability.append(
            {
                "bin": [round(low, 1), round(high, 1)],
                "n": len(members),
                "confidence": confidence,
                "accuracy": accuracy,
            }
        )
    return {
        "n": len(diagnoses),
        "target_coverage": 1 - diagnoser.config.alpha,
        "coverage": float(np.mean(covered)),
        "mean_set_size": float(np.mean(sizes)),
        "abstain_rate": float(np.mean(abstained)),
        "selective_top1": float(np.mean(selective)) if selective else None,
        "selective_n": len(selective),
        "ece": float(ece),
        "reliability": reliability,
        "temperature": diagnoser.temperature,
        "threshold": diagnoser.threshold,
    }


# ---------------------------------------------------------------------------
# Integrity checks
# ---------------------------------------------------------------------------


def shuffled_labels(experiment: Experiment, eval_runs: Sequence[str], seed: int = 7):
    """Train on relevance permuted within each run; it should collapse to random.

    Validation labels are shuffled too: early stopping on real labels would pick the
    luckiest iteration and smuggle real signal into a model meant to have none.
    """
    rng = np.random.default_rng(seed)
    pairs = experiment.pairs(experiment.splits.train + experiment.splits.val)
    matrix = build_matrix([t for t, _ in pairs], experiment.reference)
    override = {
        trace.run_id: rng.permutation(relevance_vector(matrix, i, trace, label))
        for i, (trace, label) in enumerate(pairs)
    }
    diagnoser = experiment.train(relevance_override=override)
    summary = summarize(experiment.score(diagnoser, eval_runs))
    sizes = [len(experiment.corpus.traces[r].steps) for r in eval_runs]
    return {
        "top1": summary.get("top1"),
        "random_expectation": float(np.mean([1 / n for n in sizes])) if sizes else None,
        "n": summary["n"],
    }


def _surface(step) -> dict[str, float]:
    """Formatting-only properties of a stored output: nothing about whether it is right.

    Length and values are deliberately excluded: a 404 payload *is* shorter than a
    catalog, and that difference is the fault, not a forgery tell.
    """
    output = step.output
    features = {
        "keys_sorted": math.nan,
        "compact_separators": math.nan,
        "has_usage": math.nan,
        "has_role": math.nan,
        "float_ints": 0.0,
    }
    content = None
    if isinstance(output, dict) and "choices" in output:
        features["has_usage"] = float("usage" in output)
        try:
            message = output["choices"][0]["message"]
            features["has_role"] = float("role" in message)
            content = message.get("content")
        except (KeyError, IndexError, TypeError):
            pass
    text = content if isinstance(content, str) else json.dumps(output, default=str)
    features["float_ints"] = float(len(re.findall(r"\d\.0\b", text)))
    if isinstance(content, str):
        features["compact_separators"] = float('", "' not in content and '": ' not in content)
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict) and len(parsed) >= 2:
                features["keys_sorted"] = float(list(parsed) == sorted(parsed))
        except ValueError:
            pass
    return features


def artifact_audit(corpus: Corpus, seed: int = 7) -> dict[str, Any]:
    """Can surface formatting alone tell an injected step from a healthy one?

    Positives are the injected root steps; negatives are the same addresses in passing
    runs. Grouped 5-fold cross-validation by task. AUROC above 0.65 means the forge
    leaves a forgery tell the ranker could learn instead of the fault itself.
    """
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import GroupKFold

    rows, y, groups = [], [], []
    roots_by_addr: dict[str, int] = {}
    for run_id, label in corpus.labels.items():
        if label.source != "injected":
            continue
        trace = corpus.traces[run_id]
        step = next((s for s in trace.steps if s.addr == label.root_addr), None)
        if step is None or step.kind not in {"llm", "tool", "retrieval"}:
            continue
        rows.append(_surface(step))
        y.append(1)
        groups.append(trace.task_id)
        roots_by_addr[step.addr] = roots_by_addr.get(step.addr, 0) + 1
    for run_id in corpus.passing:
        trace = corpus.traces[run_id]
        for step in trace.steps:
            if step.addr in roots_by_addr:
                rows.append(_surface(step))
                y.append(0)
                groups.append(trace.task_id)
    if len(set(y)) < 2 or len(set(groups)) < 5:
        return {"auroc": None, "n": len(y), "note": "not enough data"}
    names = list(rows[0])
    X = np.asarray([[row[n] for n in names] for row in rows], dtype=float)
    labels = np.asarray(y)
    scores = np.zeros(len(labels))
    importances = np.zeros(len(names))
    for train_index, test_index in GroupKFold(n_splits=5).split(X, labels, groups):
        classifier = LGBMClassifier(n_estimators=100, num_leaves=7, random_state=seed, verbose=-1)
        classifier.fit(X[train_index], labels[train_index])
        scores[test_index] = classifier.predict_proba(X[test_index])[:, 1]
        importances += classifier.booster_.feature_importance(importance_type="gain")
    total = importances.sum() or 1.0
    value = auroc(labels, scores)
    return {
        "auroc": value,
        "threshold": 0.65,
        "passed": bool(value <= 0.65),
        "n_injected": int(labels.sum()),
        "n_healthy": int((labels == 0).sum()),
        "drivers": {n: round(float(g / total), 4) for n, g in zip(names, importances)},
    }


def single_feature_oracle(experiment: Experiment, run_ids: Sequence[str]) -> dict[str, Any]:
    """Best top-1 any single feature reaches by itself; ~1.0 would mean a leaked column."""
    if not run_ids:
        return {}
    matrix = experiment.matrix(run_ids)
    best: dict[str, float] = {}
    for column, name in enumerate(matrix.feature_names):
        hits = 0
        for i, run_id in enumerate(matrix.run_ids):
            values = np.nan_to_num(matrix.X[matrix.run_slice(i), column], nan=-np.inf)
            hits += (
                matrix.addrs[i][int(np.argmax(values))]
                == experiment.corpus.labels[run_id].root_addr
            )
        best[name] = hits / len(matrix.run_ids)
    top = sorted(best.items(), key=lambda item: -item[1])[:5]
    return {"top_features": dict(top), "leak_suspected": bool(top and top[0][1] >= 0.99)}


# ---------------------------------------------------------------------------
# Run-level detector, replay savings and latency
# ---------------------------------------------------------------------------


def run_level(experiment: Experiment) -> dict[str, Any]:
    """Failed vs healthy (passing + recovered) runs, trained on train tasks, tested on test tasks."""
    from blackbox.eval.splits import task_bucket

    corpus = experiment.corpus

    def runs(failed: list[str], bucket: str) -> tuple[list[str], list[int]]:
        healthy = [
            r
            for r in corpus.recovered + corpus.passing
            if task_bucket(corpus.traces[r].task_id) == bucket
        ]
        return failed + healthy, [1] * len(failed) + [0] * len(healthy)

    train_ids, train_y = runs(list(experiment.splits.train), "train")
    test_ids, test_y = runs(list(experiment.splits.s0), "test")
    if not test_ids or len(set(test_y)) < 2 or len(set(train_y)) < 2:
        return {"auroc": None}
    train_matrix = experiment.matrix(train_ids)
    detector = train_run_detector(train_matrix, [i for i, v in enumerate(train_y) if v])
    scores = detector.predict_proba(run_aggregates(experiment.matrix(test_ids)))[:, 1]
    return {
        "auroc": auroc(test_y, scores),
        "n_failed": sum(test_y),
        "n_healthy": len(test_y) - sum(test_y),
    }


def replay_savings(data_dirs: Sequence[Path]) -> dict[str, Any]:
    cached = reexecuted = forks = 0
    for directory in data_dirs:
        path = Path(directory) / "blackbox.db"
        if not path.exists():
            continue
        database = SQLiteDatabase(path)
        try:
            row = database.one(
                "SELECT COUNT(*) AS n, SUM(cached_steps) AS c, SUM(reexec_steps) AS r "
                "FROM forks WHERE branch_name LIKE 'forge-%'"
            )
        finally:
            database.close()
        forks += row["n"] or 0
        cached += row["c"] or 0
        reexecuted += row["r"] or 0
    total = cached + reexecuted
    return {"forks": forks, "cached_fraction": cached / total if total else None}


# ---------------------------------------------------------------------------
# The whole evaluation
# ---------------------------------------------------------------------------


def run_evaluation(
    data_dirs: Sequence[Path],
    out_dir: Path = Path("data/eval"),
    model_dir: Path = Path("data/models/diagnoser-v1"),
    *,
    ablations: bool = True,
    base: DiagnoserConfig | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    corpus = load_corpus(data_dirs)
    if not corpus.labels:
        raise ValueError(
            f"no labelled failures in {list(map(str, data_dirs))}; run the forge first"
        )
    splits = make_splits(corpus)
    check_splits(corpus, splits)
    if not splits.train or not splits.val:
        raise ValueError("train and val splits need labelled runs; generate more forks")
    experiment = Experiment(corpus, splits, base)
    evaluation = splits.evaluation()
    logger.info(
        "Corpus: %d failed, %d passing, %d recovered; train=%d val=%d S0=%d S1=%d S4=%d",
        len(corpus.labels),
        len(corpus.passing),
        len(corpus.recovered),
        len(splits.train),
        len(splits.val),
        len(splits.s0),
        len(splits.s1),
        len(splits.s4),
    )

    # The diagnoser, saved for the API.
    diagnoser = experiment.train()
    diagnoser.save(model_dir, dataset_hash=corpus.dataset_hash)
    # Similar-past-case index for Task 9: training roots only, never val or test runs.
    Precedents.fit(experiment.matrix(splits.train), [corpus.labels[r] for r in splits.train]).save(
        model_dir
    )
    main = {split: experiment.score(diagnoser, runs) for split, runs in evaluation.items()}

    # Latency: featurize + score, per trace, on the test runs.
    test_traces: list[Trace] = [corpus.traces[r] for r in splits.s0 + splits.s1]
    tick = time.perf_counter()
    if test_traces:
        diagnoser.diagnose_many(test_traces)
    ms_per_trace = (time.perf_counter() - tick) * 1000 / max(len(test_traces), 1)

    # Leaderboard: the ranker and every baseline on the same runs.
    methods: dict[str, dict[str, dict[str, dict[str, float]]]] = {"blackbox": main}
    for name, baseline in baselines.UNTRAINED.items():
        methods[name] = {}
        for split, runs in evaluation.items():
            if runs:
                matrix = experiment.matrix(runs)
                methods[name][split] = per_run(baseline(matrix), corpus)
    position = experiment.train(groups=("A",))
    methods["position_only"] = {s: experiment.score(position, r) for s, r in evaluation.items()}
    leaderboard = {
        name: {
            "splits": _summaries(per_split),
            "vs_blackbox": {
                split: paired_difference(per_split[split], main[split])
                for split in per_split
                if per_split[split] and name != "blackbox"
            },
        }
        for name, per_split in methods.items()
    }
    for agent, (agent_splits, test_runs) in leave_one_agent_out(corpus).items():
        transfer = Experiment(corpus, agent_splits, base)
        if transfer.splits.train and transfer.splits.val:
            leaderboard["blackbox"]["splits"][f"S3:{agent}"] = summarize(
                transfer.score(transfer.train(), test_runs)
            )

    calibration = {
        split: calibration_report(diagnoser, experiment, runs)
        for split, runs in evaluation.items()
        if runs
    }

    # Ablations: feature groups, then the two recorder-value comparisons.
    ablation_results: dict[str, Any] = {}
    if ablations:
        groups = {}
        for group in GROUPS:
            model = experiment.train(groups=tuple(g for g in GROUPS if g != group))
            scored = {s: experiment.score(model, r) for s, r in evaluation.items() if r}
            groups[f"-{group} {GROUP_NAMES[group]}"] = {
                "splits": _summaries(scored),
                "vs_full": {s: paired_difference(main[s], scored[s]) for s in scored},
            }
        ablation_results["feature_groups"] = groups
        for kind, views, key in (
            ("edges", EDGE_VIEWS, "edges"),
            ("observability", OBSERVABILITY_VIEWS, "view"),
        ):
            results = {}
            for view in views:
                if view == "full":
                    scored = main
                else:
                    model = experiment.train(**{key: view})
                    scored = {s: experiment.score(model, r) for s, r in evaluation.items() if r}
                results[view] = {
                    "splits": _summaries(scored),
                    "vs_full": {
                        s: paired_difference(main[s], scored[s]) for s in scored if scored[s]
                    },
                }
            ablation_results[kind] = results

    held = splits.s0 + splits.s1
    integrity = {
        "shuffled_labels": shuffled_labels(experiment, held),
        "artifact_audit": artifact_audit(corpus),
        "single_feature_oracle": single_feature_oracle(experiment, held),
        "s1_leak_flag": bool((_top1(leaderboard["blackbox"]["splits"], "S1") or 0) > 0.85),
    }
    savings = replay_savings(data_dirs)
    run_detector = run_level(experiment)
    importance = diagnoser.feature_importance()

    headline_split = "S1" if splits.s1 else "S0"
    best_baseline = max(
        (n for n in methods if n != "blackbox"),
        key=lambda n: _top1(leaderboard[n]["splits"], headline_split) or -1,
    )
    summary = {
        # The GET /eval contract: sample_size, metrics, ablations.
        "sample_size": len(corpus.labels),
        "metrics": {
            key: value
            for key, value in {
                "s0_top1": _top1(leaderboard["blackbox"]["splits"], "S0"),
                "s1_top1": _top1(leaderboard["blackbox"]["splits"], "S1"),
                "s4_top1": _top1(leaderboard["blackbox"]["splits"], "S4"),
                "best_baseline_top1": _top1(leaderboard[best_baseline]["splits"], headline_split),
                "run_auroc": run_detector.get("auroc"),
                "replay_savings": savings["cached_fraction"],
                "ms_per_trace": round(ms_per_trace, 3),
            }.items()
            if value is not None
        },
        "ablations": {
            kind: {
                view: _top1(result["splits"], headline_split) or 0.0
                for view, result in ablation_results.get(kind, {}).items()
            }
            for kind in ("edges", "observability")
            if kind in ablation_results
        },
        "details": {
            "headline_split": headline_split,
            "best_baseline": best_baseline,
            "splits": {
                "train": len(splits.train),
                "val": len(splits.val),
                "S0": len(splits.s0),
                "S1": len(splits.s1),
                "S4": len(splits.s4),
            },
            "dataset_hash": corpus.dataset_hash,
            "model_dir": str(model_dir),
            "seconds": round(time.perf_counter() - started, 1),
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "summary.json": summary,
        "leaderboard.json": leaderboard,
        "ablations.json": ablation_results,
        "calibration.json": calibration,
        "integrity.json": integrity,
        "run_detector.json": run_detector,
        "feature_importance.json": dict(sorted(importance.items(), key=lambda kv: -kv[1])),
    }
    for name, payload in files.items():
        (out_dir / name).write_text(
            json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8"
        )
    return summary
