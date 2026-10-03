"""MuSiQue-style normalized exact match and multiset token F1 over answer aliases."""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass


def normalize_answer(text: str) -> str:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def token_f1(prediction: str, expected: str) -> float:
    predicted = normalize_answer(prediction).split()
    gold = normalize_answer(expected).split()
    if not predicted or not gold:
        return float(predicted == gold)
    overlap = sum((Counter(predicted) & Counter(gold)).values())
    return 2 * overlap / (len(predicted) + len(gold))


@dataclass(frozen=True)
class CheckResult:
    passed: bool
    exact_match: bool
    f1: float
    reason: str


def check_answer(text: str, answer: str, aliases: tuple[str, ...] = ()) -> CheckResult:
    candidates = [candidate for candidate in (answer, *aliases) if candidate.strip()]
    if not candidates:
        raise ValueError("at least one nonempty gold answer is required")
    if not isinstance(text, str) or not text.strip():
        return CheckResult(False, False, 0, "No answer produced")
    exact = any(normalize_answer(text) == normalize_answer(gold) for gold in candidates)
    f1 = max(token_f1(text, gold) for gold in candidates)
    passed = exact or f1 >= 0.8
    reason = "Exact match" if exact else f"Token F1 {f1:.3f} {'meets' if passed else 'below'} 0.8"
    return CheckResult(passed, exact, f1, reason)
