"""Localization metrics with bootstrap confidence intervals (Task 8).

Every metric is first computed per run (one number per failed run), then aggregated.
That lets one bootstrap over runs give a 95% interval for any metric, and lets two
methods evaluated on the same runs be compared *paired*: resampling the per-run
differences answers "is B really better than A here?" instead of "do the bars differ?".
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from blackbox.ml.dataset import Corpus

BOOTSTRAP_SAMPLES = 1000
PER_RUN_METRICS = ("top1", "top3", "mrr", "within1")


def per_run(
    rankings: Mapping[str, Sequence[str]],
    corpus: Corpus,
    probabilities: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, dict[str, float]]:
    """Per-run metric values. ``rankings[run]`` lists step addresses, most blamed first.

    - top1 / top3: the root is ranked first / in the first three
    - mrr: 1 / rank of the root
    - within1: the top suspect is the root or a step adjacent to it in run order
    - distractor_top1: a planted recoverable fault was blamed first (lower is better)
    - distractor_prob: probability mass on that distractor, when probabilities exist
    """
    results = {}
    for run_id, ranking in rankings.items():
        label = corpus.labels[run_id]
        order = corpus.traces[run_id].addrs
        rank = ranking.index(label.root_addr) + 1 if label.root_addr in ranking else None
        top = ranking[0] if ranking else None
        within = (
            top is not None
            and label.root_addr in order
            and abs(order.index(top) - order.index(label.root_addr)) <= 1
        )
        row = {
            "top1": float(rank == 1),
            "top3": float(rank is not None and rank <= 3),
            "mrr": 1.0 / rank if rank else 0.0,
            "within1": float(within),
        }
        if label.distractor_addr and label.distractor_addr in order:
            row["distractor_top1"] = float(top == label.distractor_addr)
            if probabilities is not None:
                row["distractor_prob"] = float(
                    probabilities[run_id].get(label.distractor_addr, 0.0)
                )
        results[run_id] = row
    return results


def bootstrap_ci(
    values: Sequence[float], *, seed: int = 0, samples: int = BOOTSTRAP_SAMPLES
) -> tuple[float, float]:
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = data[rng.integers(0, data.size, size=(samples, data.size))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def summarize(per_run_values: Mapping[str, Mapping[str, float]], *, seed: int = 0) -> dict:
    """{"n": runs, metric: {"value", "ci"}} for every metric present in any run."""
    summary: dict = {"n": len(per_run_values)}
    names = sorted({name for row in per_run_values.values() for name in row})
    for name in names:
        values = [row[name] for row in per_run_values.values() if name in row]
        summary[name] = {
            "value": float(np.mean(values)) if values else float("nan"),
            "ci": list(bootstrap_ci(values, seed=seed)),
            "n": len(values),
        }
    return summary


def paired_difference(
    a: Mapping[str, Mapping[str, float]],
    b: Mapping[str, Mapping[str, float]],
    metric: str = "top1",
    *,
    seed: int = 0,
) -> dict:
    """Mean of (b - a) over the runs both methods scored, with a paired bootstrap CI."""
    shared = sorted(set(a) & set(b))
    differences = [b[run][metric] - a[run][metric] for run in shared]
    return {
        "metric": metric,
        "n": len(shared),
        "difference": float(np.mean(differences)) if differences else float("nan"),
        "ci": list(bootstrap_ci(differences, seed=seed)),
    }


def auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Rank-based AUROC (probability a positive outscores a negative; ties count half)."""
    y = np.asarray(labels)
    s = np.asarray(scores, dtype=float)
    positives, negatives = s[y == 1], s[y == 0]
    if positives.size == 0 or negatives.size == 0:
        return float("nan")
    greater = (positives[:, None] > negatives[None, :]).sum()
    ties = (positives[:, None] == negatives[None, :]).sum()
    return float((greater + 0.5 * ties) / (positives.size * negatives.size))
