"""Similar past cases: k-nearest root steps from the training data, and the fix they suggest.

The index holds the feature rows of the *root* step of every training run (never a
validation or test run), standardised with robust training statistics. A suspect's
neighbours vote on its fault type; the winning type picks a fix template, which the
verifier can use when no oracle or twin value exists.
"""

from __future__ import annotations

import json
import warnings
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from blackbox.ml.dataset import RootLabel
from blackbox.ml.features import FEATURE_NAMES, Matrix

FILE_NAME = "precedents.json"

# fault type -> how it is usually fixed. ``kind`` is what the verifier can apply:
# ``twin`` (a passing run's value), ``rerun`` (retry the call), ``fresh`` (re-fetch
# with fresh=true).
FIX_TEMPLATES: dict[str, dict[str, str]] = {
    "wrong_value": {"kind": "twin", "text": "restore the value from a passing run of this step"},
    "stale_data": {"kind": "fresh", "text": "re-fetch the data with fresh=true"},
    "empty_404": {"kind": "rerun", "text": "retry the call"},
    "timeout_500": {"kind": "rerun", "text": "retry the call"},
    "schema_drift": {"kind": "twin", "text": "map the response back to the expected schema"},
    "irrelevant_documents": {"kind": "rerun", "text": "re-run the retrieval"},
    "poisoned_fact": {"kind": "twin", "text": "replace the document with a trusted copy"},
    "wrong_arguments": {"kind": "rerun", "text": "re-generate the model's tool arguments"},
    "wrong_tool": {"kind": "rerun", "text": "re-generate the model's reply"},
    "hallucinated_value": {"kind": "rerun", "text": "re-generate the reply grounded in its input"},
    "stops_too_early": {"kind": "rerun", "text": "re-generate the reply with room to finish"},
    "instruction_misread": {"kind": "rerun", "text": "re-generate the reply against the brief"},
    "constraint_dropped": {"kind": "twin", "text": "restore the dropped constraint"},
    "state_corruption": {"kind": "twin", "text": "restore the state value from a passing run"},
    "repeated_loop": {"kind": "rerun", "text": "re-run the step once"},
}


@dataclass(slots=True)
class Precedents:
    X: np.ndarray  # standardised root rows
    center: np.ndarray
    scale: np.ndarray
    fault_types: list[str]
    run_ids: list[str]
    addrs: list[str]
    sources: list[str]

    @classmethod
    def fit(cls, matrix: Matrix, labels: Sequence[RootLabel]) -> Precedents:
        """``matrix`` rows are the training runs, in the same order as ``labels``."""
        rows, fault_types, run_ids, addrs, sources = [], [], [], [], []
        for i, label in enumerate(labels):
            run_addrs = matrix.addrs[i]
            if label.root_addr not in run_addrs:
                continue
            rows.append(matrix.X[matrix.offsets[i] + run_addrs.index(label.root_addr)])
            fault_types.append(label.fault_type)
            run_ids.append(label.run_id)
            addrs.append(label.root_addr)
            sources.append(label.source)
        raw = np.asarray(rows, dtype=float).reshape(-1, len(FEATURE_NAMES))
        center = np.zeros(raw.shape[1])
        scale = np.ones(raw.shape[1])
        if len(raw):
            # Robust per-feature scale (IQR, else std, else 1); all-NaN columns stay 0/1.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                center = np.nan_to_num(np.nanmedian(raw, axis=0))
                iqr = np.nan_to_num(
                    np.nanpercentile(raw, 75, axis=0) - np.nanpercentile(raw, 25, axis=0)
                )
                std = np.nan_to_num(np.nanstd(raw, axis=0))
            scale = np.where(iqr > 0, iqr / 1.349, np.where(std > 0, std, 1.0))
        index = cls(raw, center, scale, fault_types, run_ids, addrs, sources)
        index.X = index._standardise(raw)
        return index

    def _standardise(self, rows: np.ndarray) -> np.ndarray:
        rows = np.where(np.isnan(rows), self.center, rows)
        return np.clip((rows - self.center) / self.scale, -10, 10)

    def query(self, row: np.ndarray, *, k: int = 12, addr: str | None = None) -> dict[str, Any]:
        if not len(self.X):
            return {"k": 0, "neighbors": [], "fault_type": None}
        z = self._standardise(np.asarray(row, dtype=float)[None, :])[0]
        distances = np.sqrt(((self.X - z) ** 2).sum(axis=1))
        nearest = np.argsort(distances, kind="stable")[:k]
        votes = Counter(self.fault_types[i] for i in nearest)
        fault_type, count = votes.most_common(1)[0]
        template = FIX_TEMPLATES.get(fault_type, {"kind": "twin", "text": "restore a known value"})
        same_addr = sum(self.addrs[i] == addr for i in nearest) if addr else 0
        verified = sum(self.sources[i] == "verified" for i in nearest)
        summary = (
            f"Looks like {count} of {len(nearest)} past {fault_type.replace('_', ' ')} "
            f"failures; the usual fix is to {template['text']}."
        )
        return {
            "k": len(nearest),
            "fault_type": fault_type,
            "votes": dict(votes.most_common()),
            "fix": {"fault_type": fault_type, **template},
            "same_step": same_addr,
            "verified": verified,
            "summary": summary,
            "neighbors": [
                {
                    "run_id": self.run_ids[i],
                    "addr": self.addrs[i],
                    "fault_type": self.fault_types[i],
                    "distance": round(float(distances[i]), 3),
                }
                for i in nearest
            ],
        }

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "feature_names": list(FEATURE_NAMES),
            "X": np.round(self.X, 5).tolist(),
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "fault_types": self.fault_types,
            "run_ids": self.run_ids,
            "addrs": self.addrs,
            "sources": self.sources,
        }
        (directory / FILE_NAME).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> Precedents | None:
        path = directory / FILE_NAME
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["feature_names"] != list(FEATURE_NAMES):
            raise ValueError("precedent index was built with a different feature list")
        width = len(FEATURE_NAMES)
        return cls(
            X=np.asarray(data["X"], dtype=float).reshape(-1, width),
            center=np.asarray(data["center"], dtype=float),
            scale=np.asarray(data["scale"], dtype=float),
            fault_types=data["fault_types"],
            run_ids=data["run_ids"],
            addrs=data["addrs"],
            sources=data["sources"],
        )
