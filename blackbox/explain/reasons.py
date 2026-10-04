"""Why the model blamed a step: TreeSHAP contributions turned into plain sentences.

Contributions come from the LightGBM booster itself (``pred_contrib=True``), so a reason
is exactly what moved this step's score, not a post-hoc story. Where a feature has a
field-level counterpart in :mod:`blackbox.explain.evidence`, the reason cites that field.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from blackbox.explain.evidence import Citation, Finding
from blackbox.ml.features import FEATURE_GROUPS, FEATURE_NAMES, Matrix

GROUP_NAMES = {
    "A": "structure",
    "B": "telemetry",
    "C": "validity",
    "D": "grounding",
    "F": "novelty",
    "G": "context",
    "H": "lineage",
}

# feature -> finding codes whose field citations support it
FEATURE_FINDINGS: dict[str, tuple[str, ...]] = {
    "date_days_older": ("stale_date",),
    "date_older_z": ("stale_date",),
    "num_max_absz": ("value_outlier",),
    "num_mean_absz": ("value_outlier",),
    "num_absz_pct": ("value_outlier",),
    "keys_unseen_frac": ("unseen_field",),
    "keys_missing_frac": ("missing_field",),
    "type_mismatch_frac": ("type_mismatch",),
    "parse_failed": ("parse_failed",),
    "status_error": ("error_payload",),
    "has_error_type": ("exception",),
    "empty_output": ("empty_output",),
    "empty_arg": ("empty_argument",),
    "io_conflicts": ("conflict_input",),
    "earlier_conflicts": ("conflict_earlier",),
    "conflict_pct": ("conflict_earlier", "conflict_input"),
    "anomaly": (
        "stale_date",
        "value_outlier",
        "conflict_input",
        "conflict_earlier",
        "unseen_field",
        "type_mismatch",
    ),
    "anomaly_pct": (
        "stale_date",
        "value_outlier",
        "conflict_input",
        "conflict_earlier",
        "unseen_field",
    ),
}


@dataclass(slots=True)
class Reason:
    feature: str
    group: str
    value: float
    contribution: float
    text: str
    citations: list[Citation] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "group": GROUP_NAMES.get(self.group, self.group),
            "value": None if math.isnan(self.value) else round(self.value, 4),
            "contribution": round(self.contribution, 4),
            "text": self.text,
            "citations": [c.as_dict() for c in self.citations],
        }


def _plural(n: float, word: str) -> str:
    k = int(round(n))
    return f"{k} {word}{'' if k == 1 else 's'}"


def describe(feature: str, value: float, row: dict[str, float]) -> str | None:
    """A sentence for a feature value that pushed the step's score up, or None if the
    feature carries no explanation a reader could act on."""
    v = value
    if math.isnan(v):
        return None
    templates = {
        "date_days_older": lambda: (
            f"Its data is dated {_plural(v, 'day')} older than in healthy runs."
        ),
        "date_older_z": lambda: "Its dates are unusually old compared with healthy runs.",
        "num_max_absz": lambda: (
            f"A number in its output is {v:.1f} spreads away from healthy runs."
        ),
        "num_mean_absz": lambda: "Its numbers are unusual compared with healthy runs.",
        "num_absz_pct": lambda: "It has the most unusual numbers in this run.",
        "keys_unseen_frac": lambda: "Its output has fields healthy runs never produce.",
        "keys_missing_frac": lambda: "Fields healthy runs always produce are missing.",
        "type_mismatch_frac": lambda: "Some fields have a different type than in healthy runs.",
        "parse_failed": lambda: "The model's reply could not be parsed.",
        "status_error": lambda: "The call returned an error payload.",
        "has_error_type": lambda: "The step raised an exception.",
        "empty_output": lambda: "The step returned nothing.",
        "empty_arg": lambda: "It was called with an empty argument.",
        "bad_numbers": lambda: (
            "Its output contains invalid numbers (NaN, negative or zero amounts)."
        ),
        "dup_list_items": lambda: "Its output repeats the same item.",
        "ungrounded_num_frac": lambda: "It introduced numbers that appear nowhere in its input.",
        "io_conflicts": lambda: "Its output contradicts its own input.",
        "earlier_conflicts": lambda: "Its output contradicts values earlier steps agreed on.",
        "conflict_pct": lambda: "It has the most contradictions with earlier steps in this run.",
        "unseen_string_frac": lambda: "It produced values never seen in healthy runs.",
        "leaf_count_z": lambda: "Its output has an unusual number of fields.",
        "text_len_z": lambda: "Its output is an unusual length.",
        "anomaly": lambda: "It looks unlike healthy runs of the same step.",
        "anomaly_pct": lambda: "It is the most unusual step in this run.",
        "is_first_anomaly": lambda: (
            "It is the first unusual step in the run; the problems after it are downstream."
        ),
        "later_anomalies": lambda: (
            f"{_plural(v, 'later step')} also look unusual, consistent with damage spreading from here."
        ),
        "later_status_errors": lambda: f"{_plural(v, 'later step')} reported errors after it.",
        "n_descendants": lambda: f"Its output reaches {_plural(v, 'later step')}.",
        "frac_descendants": lambda: (
            f"Its output flows into {round(v * 100)}% of the steps after it."
        ),
        "reaches_last": lambda: "Its output reaches the final answer.",
        "n_children": lambda: f"{_plural(v, 'step')} read its output directly.",
        "later_overwritten": lambda: "A later step overwrote its value.",
        "retries": lambda: f"It needed {_plural(v, 'retry')}.",
        "finish_length": lambda: "The model stopped because it hit the length limit.",
        "prev_anomaly": lambda: "The step just before it is also unusual.",
        "next_anomaly": lambda: "The step right after it is unusual, a likely symptom.",
    }
    template = templates.get(feature)
    if template is None:
        return None
    # Binary flags only explain when they are on.
    if (
        feature
        in {
            "parse_failed",
            "status_error",
            "has_error_type",
            "empty_output",
            "empty_arg",
            "reaches_last",
            "is_first_anomaly",
            "finish_length",
            "later_overwritten",
        }
        and v < 0.5
    ):
        return None
    if (
        feature in {"n_descendants", "n_children", "later_anomalies", "later_status_errors"}
        and v < 1
    ):
        return None
    return template()


def contributions(
    booster: Any, matrix: Matrix, run_index: int, best_iteration: int | None
) -> np.ndarray:
    """TreeSHAP contributions for one run: shape (steps, features + 1 bias)."""
    rows = matrix.X[matrix.run_slice(run_index)]
    if rows.shape[0] == 0:
        return np.zeros((0, len(FEATURE_NAMES) + 1))
    return np.asarray(booster.predict(rows, pred_contrib=True, num_iteration=best_iteration))


def top_reasons(
    contrib_row: np.ndarray,
    feature_row: np.ndarray,
    findings: list[Finding],
    *,
    k: int = 3,
) -> list[Reason]:
    """The ``k`` strongest positive contributions that have a readable explanation."""
    values = dict(zip(FEATURE_NAMES, feature_row.tolist()))
    order = np.argsort(-contrib_row[: len(FEATURE_NAMES)])
    reasons: list[Reason] = []
    used_texts: set[str] = set()
    for index in order:
        contribution = float(contrib_row[index])
        if contribution <= 0 or len(reasons) >= k:
            break
        name = FEATURE_NAMES[index]
        text = describe(name, values[name], values)
        if text is None or text in used_texts:
            continue
        codes = FEATURE_FINDINGS.get(name, ())
        supporting = [f for f in findings if f.code in codes]
        if supporting:
            text = supporting[0].message
            if text in used_texts:
                continue
        used_texts.add(text)
        reasons.append(
            Reason(
                feature=name,
                group=FEATURE_GROUPS[name],
                value=float(values[name]),
                contribution=contribution,
                text=text,
                citations=[f.citation for f in supporting[:3]],
            )
        )
    return reasons
