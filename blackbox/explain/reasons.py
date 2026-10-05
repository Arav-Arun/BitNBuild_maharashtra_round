"""Why the ranker blamed a step: SHAP reasons in plain language, backed by cited evidence.

Two layers, kept separate so the UI can show "the model thinks" apart from "the data says":

- **Evidence lines** are deterministic checks of the step's output against the healthy
  profile of the same address (and against a same-task passing twin when one exists):
  error payloads, unparsable replies, missing/unseen/retyped fields, numeric outliers,
  stale dates, values that differ from the twin. Each cites ``{addr, json_pointer}``,
  where the pointer addresses the step's semantic output (an LLM step's parsed reply).
- **Reasons** are the top TreeSHAP contributions (``pred_contrib=True``) for the suspect,
  relative to the other steps of the same run (the ranker only orders steps within a
  run, so "why first" means "why above the rest"). Each feature maps to a sentence and
  cites the evidence lines that measure the same thing.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np

from blackbox.diff import pointer_changes
from blackbox.explain.evidence import Citation, Finding
from blackbox.ml.dataset import TraceStep
from blackbox.ml.features import (
    DATE,
    FEATURE_GROUPS,
    FEATURE_NAMES,
    Matrix,
    _AddrProfile,
    _is_number,
    is_error_payload,
    output_payload,
)
from blackbox.recorder import canonical_json

MAX_EVIDENCE = 8

GROUP_NAMES = {
    "A": "structure",
    "B": "telemetry",
    "C": "validity",
    "D": "grounding",
    "F": "novelty",
    "G": "context",
    "H": "lineage",
}


def _escape(key: Any) -> str:
    return str(key).replace("~", "~0").replace("/", "~1")


def walk(value: Any, pointer: str = "", pattern: str = "") -> Iterator[tuple[str, str, Any]]:
    """Scalar leaves as (RFC 6901 pointer, profile path with ``*`` for list items, value)."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield from walk(item, f"{pointer}/{_escape(key)}", f"{pattern}/{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from walk(item, f"{pointer}/{index}", f"{pattern}/*")
    else:
        yield pointer or "", pattern or "/", value


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, str) and len(value) > 60:
        return repr(value[:57] + "...")
    return repr(value) if isinstance(value, str) else str(value)


def _age(days: float) -> str:
    if days >= 60:
        return f"{days / 30.44:.0f} months"
    return f"{days:.0f} day{'s' if round(days) != 1 else ''}"


def _line(step: TraceStep, pointer: str, kind: str, text: str, **extra: Any) -> dict[str, Any]:
    return {"addr": step.addr, "json_pointer": pointer, "kind": kind, "text": text, **extra}


def same_input(step: TraceStep, twin: TraceStep) -> bool:
    """Did both steps receive the same request (ignoring the sampling seed)?"""

    def strip(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: v for k, v in value.items() if k != "seed"}
        return value

    if step.input is None and twin.input is None:
        return True
    try:
        return canonical_json(strip(step.input)) == canonical_json(strip(twin.input))
    except (TypeError, ValueError):
        return False


def evidence_lines(
    step: TraceStep,
    profile: _AddrProfile | None,
    twin: TraceStep | None = None,
    twin_run_id: str | None = None,
) -> list[dict[str, Any]]:
    """Deterministic, cited findings about one step's output."""
    payload, parsed = output_payload(step)
    lines: list[dict[str, Any]] = []
    if not parsed:
        lines.append(_line(step, "", "parse_failure", "The model's reply is not valid JSON."))
    if is_error_payload(payload):
        pointer = "/status" if "status" in payload else "/error"
        detail = payload.get("status", payload.get("error"))
        lines.append(
            _line(
                step,
                pointer,
                "error_payload",
                f"The step returned an error payload ({pointer[1:]} = {_fmt(detail)}).",
            )
        )
    if step.error_type:
        lines.append(
            _line(step, "", "exception", f"The runtime recorded a {step.error_type} here.")
        )
    if step.kind != "state" and payload in (None, {}, [], ""):
        lines.append(_line(step, "", "empty_output", "The step's output is empty."))

    if profile is not None and profile.runs and payload is not None:
        seen_patterns = set()
        for pointer, pattern, value in walk(payload):
            seen_patterns.add(pattern)
            types = profile.types.get(pattern)
            if pattern not in profile.paths and pattern != "/":
                lines.append(
                    _line(
                        step,
                        pointer,
                        "unseen_field",
                        f"Field {pointer} never appears in passing runs of this step.",
                    )
                )
            elif types and type(value).__name__ != types.most_common(1)[0][0]:
                expected = types.most_common(1)[0][0]
                lines.append(
                    _line(
                        step,
                        pointer,
                        "type_mismatch",
                        f"{pointer} is a {type(value).__name__}; passing runs have a {expected}.",
                    )
                )
            if _is_number(value) and pattern in profile.stats:
                median, scale = profile.stats[pattern]
                z = abs(float(value) - median) / scale
                if z >= 4:
                    lines.append(
                        _line(
                            step,
                            pointer,
                            "numeric_outlier",
                            f"{pointer} is {_fmt(value)}; passing runs have about "
                            f"{_fmt(median)} ({z:.1f} robust SDs away).",
                            observed=value,
                            expected=median,
                            z=round(z, 2),
                        )
                    )
            elif (
                isinstance(value, str)
                and DATE.fullmatch(value[:10])
                and pattern in (profile.date_stats)
            ):
                try:
                    day = date.fromisoformat(value[:10]).toordinal()
                except ValueError:
                    continue
                median, scale = profile.date_stats[pattern]
                older = median - day
                if older >= 1 and older / scale >= 3:
                    usual = date.fromordinal(int(round(median))).isoformat()
                    lines.append(
                        _line(
                            step,
                            pointer,
                            "stale_date",
                            f"{pointer} is {value[:10]}, {_age(older)} older than in passing "
                            f"runs ({usual}).",
                            observed=value[:10],
                            expected=usual,
                            days_older=round(older),
                        )
                    )
        common = {p for p, c in profile.paths.items() if c >= 0.9 * profile.runs}
        for pattern in sorted(common - seen_patterns):
            if "*" in pattern or pattern == "/":
                continue
            pointer = "/".join(_escape(part) for part in pattern.split("/"))
            lines.append(
                _line(
                    step,
                    pointer,
                    "missing_field",
                    f"Field {pointer} is missing; passing runs of this step always have it.",
                )
            )

    if twin is not None and same_input(step, twin):
        for change in pointer_changes(twin.output, step.output, limit=5):
            lines.append(
                _line(
                    step,
                    change["pointer"],
                    "differs_from_twin",
                    f"{change['pointer'] or 'The output'} is {_fmt(change['right'])} here but "
                    f"{_fmt(change['left'])} in passing run {twin_run_id} given the same input.",
                    observed=change["right"],
                    expected=change["left"],
                )
            )
    return lines[:MAX_EVIDENCE]


# ---------------------------------------------------------------------------
# Feature -> sentence
# ---------------------------------------------------------------------------


def _count(noun: str) -> Callable[[float], str]:
    return lambda v: f"{v:.0f} {noun}{'' if round(v) == 1 else 's'}"


def _clause(noun: str, singular: str, plural: str) -> Callable[[float], str]:
    """'1 step looks' / '3 steps look' / 'No step looks'."""

    def text(v: float) -> str:
        if round(v) == 0:
            return f"No {noun} {singular}"
        return f"{_count(noun)(v)} {singular if round(v) == 1 else plural}"

    return text


TEMPLATES: dict[str, Callable[[float], str]] = {
    "rel_pos": lambda v: f"It runs {v:.0%} of the way through the run.",
    "seq": lambda v: f"It is step {v:.0f} of the run.",
    "n_steps": lambda v: f"The run has {v:.0f} steps.",
    "steps_after": lambda v: f"{_clause('step', 'runs', 'run')(v)} after it.",
    "is_last": lambda v: "It is the last step." if v else "It is not the last step.",
    "kind_llm": lambda v: "It is a model call." if v else "It is not a model call.",
    "kind_tool": lambda v: "It is a tool call." if v else "It is not a tool call.",
    "kind_retrieval": lambda v: "It is a retrieval." if v else "It is not a retrieval.",
    "kind_state": lambda v: "It is a state update." if v else "It is not a state update.",
    "n_reads": lambda v: f"It reads {_count('state key')(v)}.",
    "n_writes": lambda v: f"It writes {_count('state key')(v)}.",
    "has_error_type": lambda v: (
        "The runtime recorded an exception here." if v else "No exception was recorded here."
    ),
    "retries": lambda v: f"It was retried {v:.0f} times.",
    "finish_length": lambda v: (
        "The model stopped at its token limit." if v else "The model finished normally."
    ),
    "status_error": lambda v: (
        "It returned an error payload." if v else "It did not return an error payload."
    ),
    "parse_failed": lambda v: "The model's reply is not valid JSON." if v else "Its reply parses.",
    "empty_output": lambda v: "Its output is empty." if v else "Its output is not empty.",
    "bad_numbers": lambda v: f"{_clause('number', 'is', 'are')(v)} negative or not finite.",
    "dup_list_items": lambda v: f"A list in its output repeats {_count('item')(v)}.",
    "keys_unseen_frac": lambda v: f"{v:.0%} of its output fields never appear in passing runs.",
    "keys_missing_frac": lambda v: f"{v:.0%} of the fields passing runs always have are missing.",
    "type_mismatch_frac": lambda v: f"{v:.0%} of its fields have a different type than usual.",
    "empty_arg": lambda v: "One of its arguments is empty." if v else "Its arguments are filled.",
    "ungrounded_num_frac": lambda v: f"{v:.0%} of the numbers it output are not in its input.",
    "io_conflicts": lambda v: (
        f"{_clause('output field', 'contradicts', 'contradict')(v)} its own input "
        "(beyond what is normal here)."
    ),
    "earlier_conflicts": lambda v: (
        f"{_clause('output field', 'contradicts', 'contradict')(v)} values all earlier "
        "steps agreed on."
    ),
    "num_max_absz": lambda v: f"A number in its output is {v:.1f} robust SDs from passing runs.",
    "num_mean_absz": lambda v: f"Its numbers are on average {v:.1f} robust SDs from passing runs.",
    "date_days_older": lambda v: (
        f"Its data is {_age(v)} older than in passing runs."
        if v > 0
        else "Its dates match passing runs."
    ),
    "date_older_z": lambda v: f"Its date is {v:.1f} robust SDs older than in passing runs.",
    "unseen_string_frac": lambda v: f"{v:.0%} of its text values never appear in passing runs.",
    "leaf_count_z": lambda v: f"Its number of output fields is unusual (z = {v:+.1f}).",
    "text_len_z": lambda v: f"Its output length is unusual for this step (z = {v:+.1f}).",
    "anomaly": lambda v: f"Its combined anomaly score is {v:.1f} (0 for a healthy step).",
    "prev_anomaly": lambda v: (
        f"The step before it looks healthy (anomaly {v:.1f}), so the problem starts here."
        if v < 1
        else f"The step before it is anomalous too (anomaly {v:.1f})."
    ),
    "next_anomaly": lambda v: (
        f"The step after it is anomalous (anomaly {v:.1f}): damage spreading downstream."
        if v >= 1
        else f"The step after it looks healthy (anomaly {v:.1f})."
    ),
    "earlier_anomalies": lambda v: (
        "No earlier step looks anomalous."
        if round(v) == 0
        else f"{_clause('earlier step', 'also looks', 'also look')(v)} anomalous."
    ),
    "later_anomalies": lambda v: (
        "No later step looks anomalous."
        if round(v) == 0
        else f"{_clause('later step', 'looks', 'look')(v)} anomalous, consistent with damage "
        "spreading from here."
    ),
    "is_first_anomaly": lambda v: (
        "It is the first anomalous step in the run." if v else "It is not the first anomalous step."
    ),
    "later_status_errors": lambda v: f"{_clause('later step', 'returns', 'return')(v)} errors.",
    "anomaly_pct": lambda v: f"It is more anomalous than {v:.0%} of the run's steps.",
    "num_absz_pct": lambda v: f"Its numbers are more unusual than {v:.0%} of the run's steps.",
    "conflict_pct": lambda v: f"It has more contradictions than {v:.0%} of the run's steps.",
    "n_parents": lambda v: f"It consumes values from {_count('earlier step')(v)}.",
    "n_children": lambda v: f"{_clause('step', 'consumes', 'consume')(v)} its output directly.",
    "n_descendants": lambda v: f"Its output reaches {_count('later step')(v)}.",
    "frac_descendants": lambda v: f"Its output reaches {v:.0%} of the rest of the run.",
    "reaches_last": lambda v: (
        "Its output reaches the final answer."
        if v
        else "Its output does not reach the final answer."
    ),
    "depth": lambda v: f"It sits {v:.0f} hops deep in the data flow.",
    "later_overwritten": lambda v: (
        "A later step overwrites what it wrote." if v else "Nothing later overwrites what it wrote."
    ),
    "travel_constraint_mismatch_count": lambda v: (
        f"It conflicts with {_count('requested trip constraint')(v)}."
    ),
    "travel_constraint_mismatch_frac": lambda v: (
        f"It conflicts with {v:.0%} of the trip constraints checked."
    ),
    "travel_budget_math_error": lambda v: (
        f"Its travel budget differs from the visible fare, hotel, and visa inputs by up to {v:.0%}."
    ),
    "travel_quote_age_days": lambda v: (
        f"Its travel quote is {v:.0f} days older than the freshest quote in this run."
    ),
    "travel_currency_conflict": lambda v: (
        "Its FX currency does not match the travel quote currency."
        if v
        else "Its travel quote currencies agree."
    ),
}

# Which evidence lines measure the same thing as a feature (for citations).
FEATURE_EVIDENCE: dict[str, set[str]] = {
    "status_error": {"error_payload"},
    "later_status_errors": {"error_payload"},
    "has_error_type": {"exception"},
    "parse_failed": {"parse_failure"},
    "empty_output": {"empty_output"},
    "keys_unseen_frac": {"unseen_field"},
    "keys_missing_frac": {"missing_field"},
    "type_mismatch_frac": {"type_mismatch"},
    "num_max_absz": {"numeric_outlier", "differs_from_twin"},
    "num_mean_absz": {"numeric_outlier", "differs_from_twin"},
    "num_absz_pct": {"numeric_outlier"},
    "date_days_older": {"stale_date"},
    "date_older_z": {"stale_date"},
    "unseen_string_frac": {"differs_from_twin"},
    "anomaly": {
        "error_payload",
        "parse_failure",
        "unseen_field",
        "missing_field",
        "type_mismatch",
        "numeric_outlier",
        "stale_date",
    },
    "anomaly_pct": {"numeric_outlier", "stale_date", "error_payload", "missing_field"},
    "is_first_anomaly": {"numeric_outlier", "stale_date", "error_payload", "missing_field"},
}


def describe(feature: str, value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return f"{feature} is unavailable for this step."
    template = TEMPLATES.get(feature)
    return template(float(value)) if template else f"{feature} = {value:.3g}."


def shap_reasons(
    booster: Any,
    X: np.ndarray,
    index: int,
    evidence: list[dict[str, Any]],
    feature_names: tuple[str, ...],
    *,
    top: int = 3,
) -> list[dict[str, Any]]:
    """Top features pushing step ``index`` above the other steps of its run."""
    contributions = np.asarray(booster.predict(X, pred_contrib=True))[:, :-1]
    if len(X) > 1:
        others = np.delete(contributions, index, axis=0).mean(axis=0)
        delta = contributions[index] - others
    else:
        delta = contributions[index]
    order = [int(i) for i in np.argsort(-delta) if delta[i] > 0][:top]
    reasons = []
    for column in order:
        name = feature_names[column]
        value = float(X[index, column])
        kinds = FEATURE_EVIDENCE.get(name, set())
        reasons.append(
            {
                "feature": name,
                "group": FEATURE_GROUPS[name],
                "value": None if math.isnan(value) else round(value, 4),
                "contribution": round(float(delta[column]), 4),
                "text": describe(name, value),
                "cites": [
                    {"addr": line["addr"], "json_pointer": line["json_pointer"]}
                    for line in evidence
                    if line["kind"] in kinds
                ][:3],
            }
        )
    return reasons


@dataclass(slots=True)
class Reason:
    """A TreeSHAP contribution rendered for the existing report contract."""

    feature: str
    group: str
    value: float
    contribution: float
    text: str
    citations: list[Citation]

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "group": GROUP_NAMES.get(self.group, self.group),
            "value": None if math.isnan(self.value) else round(self.value, 4),
            "contribution": round(self.contribution, 4),
            "text": self.text,
            "citations": [citation.as_dict() for citation in self.citations],
        }


def _describe_report(feature: str, value: float, row: dict[str, float]) -> str | None:
    """Legacy report wording, kept alongside the evidence-first SHAP wording."""
    v = value
    if math.isnan(v):
        return None
    templates = {
        "date_days_older": lambda: f"Its data is dated {round(v)} days older than in healthy runs.",
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
        "io_conflicts": lambda: "Its output contradicts its own input.",
        "earlier_conflicts": lambda: "Its output contradicts values earlier steps agreed on.",
        "anomaly": lambda: "It looks unlike healthy runs of the same step.",
        "reaches_last": lambda: "Its output reaches the final answer.",
        "n_descendants": lambda: f"Its output reaches {round(v)} later steps.",
    }
    template = templates.get(feature)
    if template is None:
        return None
    if (
        feature
        in {"parse_failed", "status_error", "has_error_type", "empty_output", "reaches_last"}
        and v < 0.5
    ):
        return None
    return template()


def contributions(
    booster: Any, matrix: Matrix, run_index: int, best_iteration: int | None
) -> np.ndarray:
    """TreeSHAP contributions for one run, including the bias column."""
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
    """Build the concise report reasons used by the existing diagnosis endpoint."""
    values = dict(zip(FEATURE_NAMES, feature_row.tolist()))
    reasons: list[Reason] = []
    used: set[str] = set()
    for index in np.argsort(-contrib_row[: len(FEATURE_NAMES)]):
        contribution = float(contrib_row[index])
        if contribution <= 0 or len(reasons) >= k:
            break
        feature = FEATURE_NAMES[index]
        text = _describe_report(feature, values[feature], values)
        if text is None or text in used:
            continue
        related = [
            finding for finding in findings if finding.code in FEATURE_EVIDENCE.get(feature, set())
        ]
        citations = [finding.citation for finding in related[:3]]
        used.add(text)
        reasons.append(
            Reason(
                feature,
                FEATURE_GROUPS[feature],
                float(values[feature]),
                contribution,
                text,
                citations,
            )
        )
    return reasons


__all__ = [
    "TEMPLATES",
    "describe",
    "evidence_lines",
    "same_input",
    "shap_reasons",
    "walk",
]
