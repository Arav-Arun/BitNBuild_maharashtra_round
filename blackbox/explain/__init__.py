"""Turn a ranking into evidence and proof (Task 9).

``Explainer.explain`` builds the evidence bundle for one failed run without executing
anything: ranking, conformal set, abstention, and for each top suspect the SHAP reasons,
cited evidence lines, path of damage and similar past cases. ``investigate`` adds the
K-sample verifier and a narrative, and ``save`` stores the result in ``diagnoses``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from blackbox.diff import nearest_passing
from blackbox.explain.damage import damage_path
from blackbox.explain.narrative import llm_narrative, template_narrative, validate
from blackbox.explain.precedents import Precedents
from blackbox.explain.reasons import evidence_lines, shap_reasons
from blackbox.explain.verifier import (
    Verifier,
    first_self_divergence,
    record_verified,
    summarize,
)
from blackbox.ml.dataset import Trace, load_trace
from blackbox.ml.features import FEATURE_NAMES
from blackbox.ml.model import Diagnoser
from blackbox.sdk import Recorder

TOP = 3


class Explainer:
    def __init__(
        self,
        diagnoser: Diagnoser,
        recorder: Recorder,
        *,
        precedents: Precedents | None = None,
        verifier: Verifier | None = None,
        top: int = TOP,
    ) -> None:
        self.diagnoser = diagnoser
        self.recorder = recorder
        self.precedents = precedents
        self.verifier = verifier
        self.top = top

    @classmethod
    def from_model_dir(cls, model_dir: Path, recorder: Recorder, **kwargs: Any) -> Explainer:
        return cls(
            Diagnoser.load(model_dir),
            recorder,
            precedents=Precedents.load(model_dir),
            **kwargs,
        )

    @property
    def model_version(self) -> str:
        return str(self.diagnoser.metadata.get("model_version", "diagnoser"))

    def _trace(self, run_id: str) -> Trace:
        return load_trace(self.recorder.database, self.recorder.store, run_id)

    def twin(self, trace: Trace) -> Trace | None:
        twin_id = nearest_passing(self.recorder.database, trace.run_id)
        return self._trace(twin_id) if twin_id else None

    def explain(self, run_id: str, *, trace: Trace | None = None) -> dict[str, Any]:
        """Everything except intervention: cheap enough to compute on page load."""
        trace = trace or self._trace(run_id)
        diagnosis = self.diagnoser.diagnose(trace)
        matrix = self.diagnoser.matrix([trace])
        addrs = matrix.addrs[0]
        anomaly = dict(zip(addrs, matrix.X[:, FEATURE_NAMES.index("anomaly")].tolist()))
        twin = self.twin(trace)
        comparable = twin if twin is not None and twin.task_id == trace.task_id else None
        steps = {s.addr: s for s in trace.steps}
        twin_steps = {s.addr: s for s in comparable.steps} if comparable else {}
        # Deterministic diff evidence, separate from the model: the first step that got
        # the same-task twin's exact input but produced something else.
        divergence = first_self_divergence(trace, comparable) if comparable else None

        suspects = []
        for addr, probability in diagnosis.ranking[: self.top]:
            index = addrs.index(addr)
            evidence = evidence_lines(
                steps[addr],
                self.diagnoser.reference.profiles.get(addr),
                twin_steps.get(addr),
                comparable.run_id if comparable else None,
            )
            suspects.append(
                {
                    "addr": addr,
                    "probability": round(probability, 4),
                    "reasons": shap_reasons(
                        self.diagnoser.booster, matrix.X, index, evidence, FEATURE_NAMES
                    ),
                    "evidence": evidence,
                    "damage_path": damage_path(trace, addr, twin=comparable, anomaly=anomaly),
                    "precedents": self.precedents.query(matrix.X[index], addr=addr)
                    if self.precedents
                    else None,
                }
            )
        return {
            "run_id": trace.run_id,
            "agent": trace.agent,
            "task_id": trace.task_id,
            "outcome": trace.outcome,
            "model_version": self.model_version,
            "ranking": [{"addr": a, "probability": round(p, 4)} for a, p in diagnosis.ranking],
            "conformal_set": diagnosis.conformal_set,
            "conformal_size": len(diagnosis.conformal_set),
            "coverage_target": round(1 - self.diagnoser.config.alpha, 4),
            "abstain": diagnosis.abstain,
            "twin_run_id": twin.run_id if twin else None,
            "twin_same_task": comparable is not None,
            "twin_divergence": divergence,
            "suspects": suspects,
            "verification": [],
            "verdict": None,
        }

    async def investigate(
        self,
        run_id: str,
        *,
        verify: bool = True,
        narrate: str = "template",
        llm_client: Any = None,
        llm_model: str = "openai/gpt-oss-120b",
        record_labels: bool = True,
    ) -> dict[str, Any]:
        """``explain`` + the verifier on the top suspects + a narrative."""
        trace = self._trace(run_id)
        bundle = self.explain(run_id, trace=trace)
        if verify and self.verifier is not None:
            twin = self._trace(bundle["twin_run_id"]) if bundle["twin_same_task"] else None
            bundle["verification"] = await self.verifier.verify(
                trace,
                [(s["addr"], s["probability"]) for s in bundle["suspects"]],
                twin=twin,
                templates={
                    s["addr"]: s["precedents"]["fix"]
                    for s in bundle["suspects"]
                    if s["precedents"] and s["precedents"].get("fix")
                },
                evidence={s["addr"]: s["evidence"] for s in bundle["suspects"]},
                extra=[(bundle["twin_divergence"], 0.0)] if bundle["twin_divergence"] else [],
            )
            bundle.update(summarize(bundle["verification"]))
            if record_labels and bundle["verified_addr"]:
                suspect = next(
                    (s for s in bundle["suspects"] if s["addr"] == bundle["verified_addr"]), {}
                )
                fault = (suspect.get("precedents") or {}).get("fault_type")
                record_verified(self.recorder.database, run_id, bundle["verified_addr"], fault)
        if narrate == "llm" and llm_client is not None:
            bundle["narrative"] = await llm_narrative(
                _for_narrative(bundle), set(trace.addrs), llm_client, llm_model
            )
        else:
            bundle["narrative"] = template_narrative(bundle)
        bundle["narrative"]["problems"] = validate(
            bundle["narrative"], _for_narrative(bundle), set(trace.addrs)
        )
        return bundle

    def save(self, bundle: dict[str, Any]) -> None:
        evidence = {
            key: bundle.get(key)
            for key in (
                "suspects",
                "verification",
                "verdict",
                "verified_addr",
                "narrative",
                "twin_run_id",
                "twin_same_task",
                "twin_divergence",
            )
        }
        self.recorder.database.execute(
            """
            INSERT OR REPLACE INTO diagnoses(
                run_id, model_version, ranking_json, conformal_set_json, abstain,
                evidence_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                bundle["run_id"],
                bundle["model_version"],
                json.dumps(bundle["ranking"]),
                json.dumps(bundle["conformal_set"]),
                int(bundle["abstain"]),
                json.dumps(evidence, default=str),
                datetime.now(UTC).isoformat(),
            ),
        )


def _for_narrative(bundle: dict[str, Any]) -> dict[str, Any]:
    """The part of the bundle a narrative may draw on (no raw neighbour lists)."""
    return {
        "abstain": bundle["abstain"],
        "conformal_set": bundle["conformal_set"],
        "conformal_size": bundle["conformal_size"],
        "coverage_target": bundle["coverage_target"],
        "twin_divergence": bundle.get("twin_divergence"),
        "suspects": [
            {
                **{k: v for k, v in s.items() if k not in {"precedents", "damage_path"}},
                "damage_path": {
                    k: (s["damage_path"] or {}).get(k)
                    for k in (
                        "nodes",
                        "reaches_final",
                        "reached",
                        "symptoms",
                        "path_to_final",
                        "suspect_matches_twin",
                    )
                },
                "precedents": {
                    k: (s["precedents"] or {}).get(k)
                    for k in ("summary", "fault_type", "votes", "k", "fix")
                }
                if s.get("precedents")
                else None,
            }
            for s in bundle["suspects"]
        ],
        "verification": bundle.get("verification") or [],
    }


def contract(bundle: dict[str, Any]) -> dict[str, Any]:
    """The ``GET /runs/{id}/diagnosis`` body (``server.models.Diagnosis``)."""
    top = bundle["suspects"][0] if bundle["suspects"] else None
    probabilities = {r["addr"]: r["probability"] for r in bundle["ranking"]}
    reasons_by_addr = {s["addr"]: s["reasons"] for s in bundle["suspects"]}
    return {
        "run_id": bundle["run_id"],
        "suspects": [
            {
                "addr": addr,
                "probability": probabilities[addr],
                "reason": (reasons_by_addr.get(addr) or [{}])[0].get("text"),
            }
            for addr in [
                r["addr"] for r in bundle["ranking"][: max(TOP, len(bundle["conformal_set"]))]
            ]
        ],
        "conformal_set": bundle["conformal_set"],
        "abstain": bundle["abstain"],
        "verdict": bundle.get("verdict"),
        "reasons": top["reasons"] if top else [],
        "damage_path": top["damage_path"] if top else None,
        "precedents": top["precedents"] if top else None,
        "verification": bundle.get("verification") or [],
        "narrative": bundle.get("narrative"),
        "twin_run_id": bundle.get("twin_run_id"),
        "model_version": bundle.get("model_version"),
    }


__all__ = ["Explainer", "contract"]
