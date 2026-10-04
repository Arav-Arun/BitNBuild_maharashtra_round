"""Untrained baselines the diagnoser must beat (Task 7).

Each returns ``{run_id: [addr, ...]}``, most blamed first. The trained position-only
baseline lives with the ranker (it is the ranker restricted to feature group A).
"""

from __future__ import annotations

import random

import numpy as np

from blackbox.ml.features import FEATURE_NAMES, Matrix

ANOMALY = FEATURE_NAMES.index("anomaly")
ERROR_SIGNALS = [
    FEATURE_NAMES.index(name)
    for name in ("status_error", "parse_failed", "has_error_type", "finish_length")
]


def random_step(matrix: Matrix, *, seed: int = 0) -> dict[str, list[str]]:
    rng = random.Random(seed)
    rankings = {}
    for run_id, addrs in zip(matrix.run_ids, matrix.addrs):
        shuffled = list(addrs)
        rng.shuffle(shuffled)
        rankings[run_id] = shuffled
    return rankings


def last_step(matrix: Matrix) -> dict[str, list[str]]:
    return {run_id: list(reversed(addrs)) for run_id, addrs in zip(matrix.run_ids, matrix.addrs)}


def first_error(matrix: Matrix) -> dict[str, list[str]]:
    """Blame the first step with an explicit error signal; then later steps, last first.

    The rule an engineer applies by hand: scroll down to the first red line.
    """
    rankings = {}
    for index, (run_id, addrs) in enumerate(zip(matrix.run_ids, matrix.addrs)):
        rows = matrix.X[matrix.run_slice(index)]
        errored = [i for i in range(len(addrs)) if np.nansum(rows[i, ERROR_SIGNALS]) > 0]
        rest = [i for i in reversed(range(len(addrs))) if i not in errored]
        rankings[run_id] = [addrs[i] for i in errored + rest]
    return rankings


def anomaly_max(matrix: Matrix) -> dict[str, list[str]]:
    """Blame the most unusual step versus healthy runs; ties go to the earlier step."""
    rankings = {}
    for index, (run_id, addrs) in enumerate(zip(matrix.run_ids, matrix.addrs)):
        scores = np.nan_to_num(matrix.X[matrix.run_slice(index), ANOMALY])
        order = sorted(range(len(addrs)), key=lambda i: (-scores[i], i))
        rankings[run_id] = [addrs[i] for i in order]
    return rankings


UNTRAINED = {
    "random": random_step,
    "last_step": last_step,
    "first_error": first_error,
    "anomaly_max": anomaly_max,
}
