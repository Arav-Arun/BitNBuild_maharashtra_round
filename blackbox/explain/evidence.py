"""Deterministic, field-level evidence for one step, measured against healthy runs.

Every finding cites an exact JSON pointer into the step's semantic payload, so the UI can
jump to and highlight the field. Pointers are relative to ``semantic_input`` /
``semantic_output`` below (parsed LLM content, or a tool's arguments and result), which is
what the API shows as a step's input and output.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from blackbox.ml.dataset import Trace, TraceStep
from blackbox.ml.features import (
    Reference,
    _date,
    _is_number,
    input_payload,
    is_error_payload,
    output_payload,
)

STALE_DAYS_Z = 3.0
OUTLIER_Z = 4.0
EXPECTED_FIELD_SHARE = 0.9


@dataclass(frozen=True, slots=True)
class Citation:
    addr: str
    pointer: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"addr": self.addr, "json_pointer": self.pointer}


@dataclass(slots=True)
class Finding:
    code: str
    message: str
    citation: Citation
    severity: float
    related: list[Citation] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "citation": self.citation.as_dict(),
            "severity": round(self.severity, 3),
            "related": [c.as_dict() for c in self.related],
        }


def semantic_output(step: TraceStep) -> Any:
    payload, _ = output_payload(step)
    return payload


def semantic_input(step: TraceStep) -> Any:
    return input_payload(step)


def _escape(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def leaves(value: Any, pointer: str = "") -> Iterator[tuple[str, str, Any]]:
    """Yield (json_pointer, profile_path, scalar) for every scalar leaf.

    ``profile_path`` uses ``*`` for list indices, matching the keys of :class:`Reference`.
    """

    def walk(node: Any, ptr: str, path: str) -> Iterator[tuple[str, str, Any]]:
        if isinstance(node, dict):
            for key, item in node.items():
                yield from walk(item, f"{ptr}/{_escape(str(key))}", f"{path}/{key}")
        elif isinstance(node, list):
            for index, item in enumerate(node):
                yield from walk(item, f"{ptr}/{index}", f"{path}/*")
        else:
            yield ptr or "", path or "/", node

    yield from walk(value, pointer, "")


def _leaf_name(path: str) -> str | None:
    if "/*" in path:
        return None
    return path.rsplit("/", 1)[-1] or None


def _show(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= 60 else f"{text[:57]}…"


def _fmt_number(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:g}"


def step_findings(trace: Trace, step: TraceStep, reference: Reference) -> list[Finding]:
    """All deterministic findings for ``step``, most severe first."""
    findings: list[Finding] = []
    profile = reference.profiles.get(step.addr)
    payload, parsed = output_payload(step)
    out_ptr = "/output"

    if not parsed:
        findings.append(
            Finding(
                "parse_failed",
                "The model's reply is not valid JSON, so later steps cannot read it.",
                Citation(step.addr, out_ptr),
                5.0,
            )
        )
    if is_error_payload(payload):
        key = "error" if isinstance(payload, dict) and "error" in payload else "status"
        detail = payload.get(key) if isinstance(payload, dict) else None
        findings.append(
            Finding(
                "error_payload",
                f"The call returned an error ({_show(detail)}).",
                Citation(step.addr, f"{out_ptr}/{key}"),
                4.5,
            )
        )
    if step.error_type:
        findings.append(
            Finding(
                "exception",
                f"The step raised {step.error_type}.",
                Citation(step.addr, None),
                5.0,
            )
        )
    if payload in ({}, [], "") or (payload is None and step.kind in {"tool", "retrieval"}):
        findings.append(
            Finding("empty_output", "The step returned nothing.", Citation(step.addr, out_ptr), 3.5)
        )

    observed = list(leaves(payload, out_ptr)) if payload is not None else []
    if profile is not None and profile.runs >= 3:
        for pointer, path, value in observed:
            if isinstance(value, list) and not value:
                continue
            if (day := _date(value)) is not None and path in profile.date_stats:
                median, scale = profile.date_stats[path]
                older = median - day
                if older > 0 and older / scale >= STALE_DAYS_Z:
                    usual = date.fromordinal(int(round(median))).isoformat()
                    findings.append(
                        Finding(
                            "stale_date",
                            f"{_leaf_name(path) or path} is {int(older)} days older than in "
                            f"healthy runs ({value} vs usually {usual}).",
                            Citation(step.addr, pointer),
                            min(5.0, 2.0 + older / scale / 10),
                        )
                    )
            elif _is_number(value) and path in profile.stats:
                median, scale = profile.stats[path]
                z = (float(value) - median) / scale if scale else 0.0
                if abs(z) >= OUTLIER_Z:
                    findings.append(
                        Finding(
                            "value_outlier",
                            f"{_leaf_name(path) or path} = {_fmt_number(float(value))} is far from "
                            f"healthy runs (typically about {_fmt_number(median)}).",
                            Citation(step.addr, pointer),
                            min(4.5, 1.5 + abs(z) / 10),
                        )
                    )
            dominant = profile.types.get(path)
            if dominant:
                expected = dominant.most_common(1)[0][0]
                actual = type(value).__name__
                numeric = {"int", "float"}
                if actual != expected and not {actual, expected} <= numeric:
                    findings.append(
                        Finding(
                            "type_mismatch",
                            f"{_leaf_name(path) or path} is a {actual}, but healthy runs "
                            f"always produce a {expected}.",
                            Citation(step.addr, pointer),
                            3.0,
                        )
                    )
        seen_paths = {path for _, path, _ in observed}
        for path in sorted(seen_paths - set(profile.paths)):
            pointer = next(p for p, pa, _ in observed if pa == path)
            findings.append(
                Finding(
                    "unseen_field",
                    f"Field {path} never appears in healthy runs of this step.",
                    Citation(step.addr, pointer),
                    2.5,
                )
            )
        for path, count in sorted(profile.paths.items()):
            if count / profile.runs >= EXPECTED_FIELD_SHARE and path not in seen_paths:
                if any(path.startswith(f"{p}/") for p in seen_paths if "/*" in path):
                    continue
                findings.append(
                    Finding(
                        "missing_field",
                        f"Expected field {path} is missing (present in "
                        f"{round(100 * count / profile.runs)}% of healthy runs).",
                        Citation(step.addr, out_ptr),
                        3.0,
                    )
                )

    findings.extend(_conflict_findings(trace, step, observed))
    findings.extend(_empty_argument_findings(step))
    findings.sort(key=lambda f: -f.severity)
    return findings


def _conflict_findings(
    trace: Trace, step: TraceStep, observed: list[tuple[str, str, Any]]
) -> list[Finding]:
    """Values that contradict the step's own input or a value earlier steps agreed on."""
    findings: list[Finding] = []
    inputs = semantic_input(step)
    input_named: dict[str, tuple[str, Any]] = {}
    if inputs is not None:
        for pointer, path, value in leaves(inputs, "/input"):
            if (name := _leaf_name(path)) is not None:
                input_named[name] = (pointer, value)
    for pointer, path, value in observed:
        name = _leaf_name(path)
        if name is None or name not in input_named:
            continue
        in_pointer, in_value = input_named[name]
        if in_value != value and not (
            _is_number(in_value)
            and _is_number(value)
            and abs(float(in_value) - float(value)) < 1e-9
        ):
            findings.append(
                Finding(
                    "conflict_input",
                    f"{name} changed from {_show(in_value)} in the step's input to "
                    f"{_show(value)} in its output.",
                    Citation(step.addr, pointer),
                    3.5,
                    [Citation(step.addr, in_pointer)],
                )
            )

    agreed: dict[str, tuple[str, str, Any]] = {}
    disagreed: set[str] = set()
    for earlier in sorted(trace.steps, key=lambda s: s.seq):
        if earlier.seq >= step.seq:
            break
        payload = semantic_output(earlier)
        if payload is None:
            continue
        for pointer, path, value in leaves(payload, "/output"):
            name = _leaf_name(path)
            if name is None or name in disagreed:
                continue
            if name in agreed and agreed[name][2] != value:
                disagreed.add(name)
                agreed.pop(name, None)
            elif name not in agreed:
                agreed[name] = (earlier.addr, pointer, value)
    for pointer, path, value in observed:
        name = _leaf_name(path)
        if name is None or name not in agreed:
            continue
        src_addr, src_pointer, src_value = agreed[name]
        same = src_value == value or (
            _is_number(src_value)
            and _is_number(value)
            and abs(float(src_value) - float(value)) < 1e-6
        )
        if not same:
            findings.append(
                Finding(
                    "conflict_earlier",
                    f"{name} is {_show(value)} here but {_show(src_value)} in {src_addr}.",
                    Citation(step.addr, pointer),
                    3.0,
                    [Citation(src_addr, src_pointer)],
                )
            )
    return findings


def _empty_argument_findings(step: TraceStep) -> list[Finding]:
    if step.kind not in {"tool", "retrieval"}:
        return []
    args = semantic_input(step)
    if not isinstance(args, dict):
        return []
    findings = []
    for pointer, path, value in leaves(args, "/input"):
        if isinstance(value, str) and not value.strip():
            findings.append(
                Finding(
                    "empty_argument",
                    f"Called {step.name} with an empty {_leaf_name(path) or path}.",
                    Citation(step.addr, pointer),
                    3.5,
                )
            )
    return findings
