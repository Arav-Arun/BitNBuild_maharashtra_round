"""Grouped evaluation splits (Task 7).

Every split is grouped by ``task_id``: all forks of one scenario land in the same bucket,
so the model can never score by recognising a scenario it trained on. Buckets come from
a stable hash, so the same corpus always produces the same split.

=====  ====================================================================
train  injected, seen fault types, train tasks (+ verified/human labels)
val    injected, seen fault types, val tasks: early stopping and calibration
S0     injected, seen fault types, test tasks: known faults on new tasks
S1     injected, held-out fault types, non-train tasks: unseen fault families
S3     leave one agent out (only when the corpus has two or more agents)
S4     natural failures: test-only, never trained on
=====  ====================================================================
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from blackbox.ml.dataset import Corpus

TRAIN_FRACTION = 0.6
VAL_FRACTION = 0.2


def task_bucket(task_id: str, salt: str = "blackbox-v1") -> str:
    digest = hashlib.sha256(f"{salt}:{task_id}".encode()).digest()
    position = int.from_bytes(digest[:8], "big") / 2**64
    if position < TRAIN_FRACTION:
        return "train"
    if position < TRAIN_FRACTION + VAL_FRACTION:
        return "val"
    return "test"


@dataclass(slots=True)
class Splits:
    train: list[str] = field(default_factory=list)
    val: list[str] = field(default_factory=list)
    s0: list[str] = field(default_factory=list)
    s1: list[str] = field(default_factory=list)
    s4: list[str] = field(default_factory=list)
    # Healthy runs from train tasks only: the novelty reference must not see test tasks.
    reference: list[str] = field(default_factory=list)

    def evaluation(self) -> dict[str, list[str]]:
        return {"S0": self.s0, "S1": self.s1, "S4": self.s4}


def make_splits(corpus: Corpus, *, agents: set[str] | None = None) -> Splits:
    """Split ``corpus``; with ``agents``, only runs of those agents are used."""
    splits = Splits()
    for run_id, label in sorted(corpus.labels.items()):
        trace = corpus.traces[run_id]
        if agents is not None and trace.agent not in agents:
            continue
        bucket = task_bucket(trace.task_id)
        if label.source == "natural_auto":
            splits.s4.append(run_id)
        elif label.source in {"verified", "human"}:
            # The retraining loop: confirmed diagnoses only ever add training data.
            if bucket == "train":
                splits.train.append(run_id)
        elif label.held_out:
            if bucket != "train":
                splits.s1.append(run_id)
        elif bucket == "train":
            splits.train.append(run_id)
        elif bucket == "val":
            splits.val.append(run_id)
        else:
            splits.s0.append(run_id)
    splits.reference = [
        run_id
        for run_id in corpus.passing
        if task_bucket(corpus.traces[run_id].task_id) == "train"
        and (agents is None or corpus.traces[run_id].agent in agents)
    ]
    return splits


def leave_one_agent_out(corpus: Corpus) -> dict[str, tuple[Splits, list[str]]]:
    """S3: for each agent, train on the others and test on all of its labelled failures.

    The held-out agent's *passing* runs may join the novelty reference: they carry no
    labels, and a deployment always has healthy traces of the agent it debugs.
    """
    agents = {corpus.traces[r].agent for r in corpus.labels}
    if len(agents) < 2:
        return {}
    result = {}
    for agent in sorted(agents):
        others = agents - {agent}
        splits = make_splits(corpus, agents=others)
        splits.reference += [r for r in corpus.passing if corpus.traces[r].agent == agent]
        test = [
            r
            for r, label in corpus.labels.items()
            if corpus.traces[r].agent == agent and label.source == "injected"
        ]
        result[agent] = (splits, test)
    return result


def check_splits(corpus: Corpus, splits: Splits) -> None:
    """The Task 7 split assertions: no task crosses buckets, no unseen type in training."""
    tasks = {
        name: {corpus.traces[r].task_id for r in runs}
        for name, runs in (("train", splits.train), ("val", splits.val), ("S0", splits.s0))
    }
    for a, b in (("train", "val"), ("train", "S0"), ("val", "S0")):
        overlap = tasks[a] & tasks[b]
        if overlap:
            raise AssertionError(f"tasks in both {a} and {b}: {sorted(overlap)[:5]}")
    if tasks["train"] & {corpus.traces[r].task_id for r in splits.s1}:
        raise AssertionError("an S1 task also appears in training")
    if any(corpus.labels[r].held_out for r in splits.train + splits.val):
        raise AssertionError("a held-out fault type leaked into training")
    if set(splits.s4) & set(splits.train + splits.val):
        raise AssertionError("natural failures must stay test-only")
