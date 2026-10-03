"""Phase 3 extension point: features, run-level splits, fitting, ranking.

Features must use an explicit allowlist of observable step fields. Never feed
labels, fault_type, culprit_step_index, run_id, or checkpoint hashes to a model.
Fit contrast references and preprocessing on training runs only.
"""

from typing import Protocol

from blackbox.schema import Run


class Ranker(Protocol):
    def rank(self, run: Run) -> list[tuple[int, float]]:
        """Return (step_index, score) pairs in descending suspicion order."""
        ...
