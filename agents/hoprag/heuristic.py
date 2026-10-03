"""Network-free lexical baseline. This is not an LLM or an oracle-backed fixture."""

from __future__ import annotations

import json
import re
from collections import Counter

from agents.hoprag.tools import tokenize

STOPWORDS = set(
    "a an the of in at on is was were are to for and or by who what which where when".split()
)


def extract(question: str, documents: list[dict]) -> dict:
    """Choose an overlapping sentence, then a likely date/entity span from it.

    Deliberately small and inspectable. It never sees aliases, gold decomposition,
    supporting flags, or the un-read corpus. Weak benchmark scores are expected.
    """
    query = set(tokenize(question)) - STOPWORDS
    candidates = []
    for document in documents:
        for sentence in re.split(r"(?<=[.!?])\s+", document["text"]):
            overlap = len(query & set(tokenize(sentence)))
            candidates.append((overlap, sentence, document["doc_id"]))
    if not candidates:
        return {"text": "", "doc_ids": []}
    # Stable ties: the first (highest BM25-ranked) document wins.
    _, sentence, doc_id = max(candidates, key=lambda item: item[0])
    dates = re.findall(r"\b(?:1[0-9]{3}|20[0-9]{2})\b", sentence)
    if re.search(r"\b(when|year)\b", question, re.I) and dates:
        return {"text": dates[0], "doc_ids": [doc_id]}
    names = re.findall(r"\b[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*){0,4}", sentence)
    names = [name for name in names if set(tokenize(name)) - query - STOPWORDS]
    if names:
        text = max(names, key=lambda name: (len(set(tokenize(name)) - query), len(name)))
    else:
        text = sentence.strip()
    return {"text": text, "doc_ids": [doc_id]}


class HeuristicClient:
    def __init__(self, hops=3):
        if not 2 <= hops <= 4:
            raise ValueError("heuristic hops must be between 2 and 4")
        self.hops = hops
        self.calls = Counter()

    async def chat(self, **request):
        envelope = json.loads(request["messages"][-1]["content"])
        role, task = envelope["role"], envelope["task"]
        self.calls[role] += 1
        if role == "decompose":
            question = task["question"]
            result = {"questions": [question] + [f"#{i} {question}" for i in range(1, self.hops)]}
        elif role in {"hop", "final"}:
            result = extract(task["question"], task["documents"])
        else:
            raise ValueError(f"unknown heuristic role: {role}")
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": json.dumps(result)},
                    "finish_reason": "stop",
                }
            ]
        }
