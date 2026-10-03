"""Question-local BM25 tools, with corpus fingerprints in every request key."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict

from rank_bm25 import BM25Okapi

from agents.hoprag.data import Question, public_payload
from blackbox.recorder import content_hash


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


class RetrievalTools:
    def __init__(self, question: Question):
        if not 1 <= len(question.paragraphs) <= 20:
            raise ValueError("each question requires 1–20 paragraphs")
        self.paragraphs = {p.doc_id: p for p in question.paragraphs}
        if len(self.paragraphs) != len(question.paragraphs):
            raise ValueError("paragraph IDs must be unique")
        self.ordered = sorted(self.paragraphs.values(), key=lambda p: p.doc_id)
        tokenized = [tokenize(f"{p.title} {p.text}") for p in self.ordered]
        if not any(tokenized):
            raise ValueError("corpus has no searchable text")
        self.index = BM25Okapi(tokenized)
        self.corpus_id = content_hash(public_payload(question))
        # Bound tool names alone do not identify their backing corpus to the SDK.
        self.calls = Counter()

    def _check(self, corpus_id: str):
        if corpus_id != self.corpus_id:
            raise ValueError("request does not belong to this paragraph corpus")

    def search(self, query: str, *, corpus_id: str, k: int = 3) -> list[dict]:
        self._check(corpus_id)
        self.calls["search"] += 1
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        if type(k) is not int or not 1 <= k <= 20:
            raise ValueError("k must be between 1 and 20")
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.index.get_scores(tokens)
        order = sorted(
            range(len(scores)), key=lambda i: (-float(scores[i]), self.ordered[i].doc_id)
        )
        return [
            {
                "doc_id": self.ordered[i].doc_id,
                "title": self.ordered[i].title,
                "score": float(scores[i]),
            }
            for i in order[:k]
        ]

    def read(self, doc_id: int, *, corpus_id: str) -> dict:
        self._check(corpus_id)
        self.calls["read"] += 1
        if type(doc_id) is not int or doc_id not in self.paragraphs:
            raise ValueError(f"unknown paragraph ID: {doc_id}")
        return asdict(self.paragraphs[doc_id])

    def answer(self, text: str, *, corpus_id: str) -> dict:
        self._check(corpus_id)
        self.calls["answer"] += 1
        if not isinstance(text, str):
            raise ValueError("answer must be a string")
        return {"text": text.strip()}
