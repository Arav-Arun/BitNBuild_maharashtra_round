"""Shared trace contract. Indices are zero-based; timestamps use ISO 8601."""
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Literal

StepType = Literal["model_call", "tool_call", "retrieval", "state_update"]
Outcome = Literal["success", "fail"]


@dataclass
class Labels:
    """Supervision only: never pass these fields to feature extraction."""
    run_outcome: Outcome
    is_culprit: bool | None = None


@dataclass
class Step:
    run_id: str
    step_index: int
    parent_step_index: int | None
    step_type: StepType
    input: dict[str, Any]
    output: dict[str, Any]
    tool_name: str | None
    retrieved_context: list[str]
    state_before_ref: str
    state_after_ref: str
    status: Literal["ok", "error"]
    started_at: str
    ended_at: str
    label: Labels | None = None

    def __post_init__(self):
        if self.step_index < 0:
            raise ValueError("step_index must be nonnegative")
        if self.parent_step_index is not None and not 0 <= self.parent_step_index < self.step_index:
            raise ValueError("parent must precede the step")
        if self.step_type not in {"model_call", "tool_call", "retrieval", "state_update"}:
            raise ValueError("unknown step type")
        if self.status not in {"ok", "error"}:
            raise ValueError("unknown step status")
        for ref in (self.state_before_ref, self.state_after_ref):
            if len(ref) != 64 or any(c not in "0123456789abcdef" for c in ref):
                raise ValueError("checkpoint refs must be SHA-256 hex digests")
        start, end = datetime.fromisoformat(self.started_at), datetime.fromisoformat(self.ended_at)
        if start.tzinfo is None or end.tzinfo is None or end < start:
            raise ValueError("timestamps must be timezone-aware and ordered")
        if self.label and self.label.run_outcome not in {"success", "fail"}:
            raise ValueError("unknown label outcome")


@dataclass
class Run:
    run_id: str
    task_id: str
    outcome: Outcome
    checker_reason: str
    model_name: str
    seed: int
    initial_state_ref: str
    steps: list[Step] = field(default_factory=list)
    fault_type: str | None = None
    culprit_step_index: int | None = None
    schema_version: int = 1

    def validate(self):
        if self.schema_version != 1 or self.outcome not in {"success", "fail"}:
            raise ValueError("unsupported schema version or outcome")
        previous = self.initial_state_ref
        if len(previous) != 64 or any(c not in "0123456789abcdef" for c in previous):
            raise ValueError("invalid initial checkpoint reference")
        for index, step in enumerate(self.steps):
            step.__post_init__()
            if step.run_id != self.run_id or step.step_index != index:
                raise ValueError("steps must belong to the run and have contiguous indices")
            if step.state_before_ref != previous:
                raise ValueError("broken checkpoint chain")
            if step.label and step.label.run_outcome != self.outcome:
                raise ValueError("step label disagrees with run outcome")
            if step.label and step.label.is_culprit is not None:
                if step.label.is_culprit != (index == self.culprit_step_index):
                    raise ValueError("culprit labels disagree with metadata")
            previous = step.state_after_ref
        if self.culprit_step_index is not None:
            if not 0 <= self.culprit_step_index < len(self.steps):
                raise ValueError("culprit index outside trace")
        if self.fault_type is not None and self.culprit_step_index is None:
            raise ValueError("injected faults require a culprit index")

    def to_dict(self):
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["steps"] = [Step(**{**s, "label": Labels(**s["label"]) if s.get("label") else None})
                         for s in data.get("steps", [])]
        run = cls(**data)
        run.validate()
        return run
