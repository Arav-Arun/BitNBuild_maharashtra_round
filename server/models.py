"""Shared API contracts consumed by the server and web fixtures."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunSummary(ContractModel):
    run_id: str
    status: Literal["passed", "failed", "running"]
    agent: str
    task: str
    steps: int = Field(ge=0)
    duration_ms: float = Field(ge=0)
    risk: float = Field(ge=0, le=1)
    top_suspect: str | None = None


class Edge(ContractModel):
    src_addr: str
    src_pointer: str | None = None
    dst_addr: str
    dst_pointer: str | None = None
    kind: Literal["state", "message", "inferred"]


class StepDetail(ContractModel):
    addr: str
    seq: int = Field(ge=0)
    kind: Literal["llm", "tool", "retrieval", "state", "value"]
    name: str
    input: Any = None
    output: Any = None
    state_before: dict[str, Any]
    state_after: dict[str, Any]
    cache_status: Literal["cached", "invalidated", "live", "edited"]


class Diagnosis(ContractModel):
    run_id: str
    suspects: list[dict[str, Any]]
    conformal_set: list[str] = Field(default_factory=list)
    abstain: bool = False
    verdict: Literal["VERIFIED", "REFUTED", "INCONCLUSIVE"] | None = None
    # Evidence (Task 9): every item cites step addresses (and JSON Pointers) in the run.
    reasons: list[dict[str, Any]] = Field(default_factory=list)
    damage_path: dict[str, Any] | None = None
    precedents: dict[str, Any] | None = None
    verification: list[dict[str, Any]] = Field(default_factory=list)
    narrative: dict[str, Any] | None = None
    twin_run_id: str | None = None
    model_version: str | None = None


class ForkEvent(ContractModel):
    event: Literal["step", "outcome", "summary"]
    data: dict[str, Any]


class RunDetail(ContractModel):
    run: RunSummary
    steps: list[StepDetail]
    edges: list[Edge]


class DiffRow(ContractModel):
    addr: str
    status: Literal["same", "cached", "changed", "new", "removed"]
    left: Any = None
    right: Any = None
    changes: list[dict[str, Any]] = Field(default_factory=list)  # {pointer, left, right}


class DiffResponse(ContractModel):
    left_run_id: str
    right_run_id: str
    first_divergence: str | None = None
    rows: list[DiffRow]
    state_diff: list[dict[str, Any]] = Field(default_factory=list)  # {key, status, left?, right?}
    outcome: dict[str, Any] | None = None  # {left, right, flipped}


class EvalResponse(ContractModel):
    sample_size: int = Field(ge=0)
    metrics: dict[str, float]
    ablations: dict[str, dict[str, float]]
