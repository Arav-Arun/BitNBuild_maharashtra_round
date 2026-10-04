"""Explicit deterministic test double for offline HopRAG data. Not an LLM benchmark.

It plays a competent reader, the role an LLM fills when ``--client groq`` is used. Its
answers come from MuSiQue's gold decomposition, but only through the agent's own
observations: a hop is answered correctly only when the paragraph the agent actually
read is that hop's gold supporting paragraph. A wrong query, a wrong document or an
irrelevant hit makes it fall back to the lexical extractor, so retrieval and decision
faults still propagate to a wrong final answer exactly as they would with a model.
"""

from __future__ import annotations

import json
from collections.abc import Iterable

from agents.hoprag.data import Example
from agents.hoprag.heuristic import extract
from agents.hoprag.tools import tokenize

MODEL = "hoprag-reader-fixture-v1"


def _paragraph_key(title: str, text: str) -> str:
    return f"{title}\n{text}"


def _overlap(a: str, b: str) -> int:
    return len(set(tokenize(a.replace("#", " "))) & set(tokenize(b.replace("#", " "))))


class GoldReaderClient:
    def __init__(self, examples: Iterable[Example]) -> None:
        self.by_question: dict[str, Example] = {}
        # gold supporting paragraph -> the (example, hop) pairs it answers
        self.support: dict[str, list[tuple[Example, int]]] = {}
        for example in examples:
            self.by_question[example.question.text] = example
            paragraphs = {p.doc_id: p for p in example.question.paragraphs}
            for index, hop in enumerate(example.gold_hops):
                paragraph = paragraphs[hop.paragraph_support_idx]
                key = _paragraph_key(paragraph.title, paragraph.text)
                self.support.setdefault(key, []).append((example, index))

    def _hop(self, question: str, documents: list[dict]) -> dict:
        matches = []
        for document in documents:
            key = _paragraph_key(document.get("title", ""), document.get("text", ""))
            for example, index in self.support.get(key, []):
                hop = example.gold_hops[index]
                matches.append((_overlap(question, hop.question), hop.answer, document["doc_id"]))
        if not matches:
            return extract(question, documents)
        _, answer, doc_id = max(matches, key=lambda match: match[0])
        return {"text": answer, "doc_ids": [doc_id]}

    def _final(self, task: dict) -> dict:
        observations = [o for o in task.get("observations") or [] if o.get("text", "").strip()]
        if observations and len(observations) == len(task.get("observations") or []):
            last = observations[-1]
            allowed = {document["doc_id"] for document in task["documents"]}
            doc_ids = [doc_id for doc_id in last.get("doc_ids", []) if doc_id in allowed]
            if doc_ids:
                return {"text": last["text"], "doc_ids": doc_ids}
        return extract(task["question"], task["documents"])

    async def chat(self, **request) -> dict:
        envelope = json.loads(request["messages"][-1]["content"])
        role, task = envelope["role"], envelope["task"]
        if role == "decompose":
            example = self.by_question.get(task["question"])
            if example is None:
                raise ValueError("the fixture reader only knows MuSiQue questions it was given")
            result = {"questions": [hop.question for hop in example.gold_hops]}
        elif role == "hop":
            result = self._hop(task["question"], task["documents"])
        elif role == "final":
            result = self._final(task)
        else:
            raise ValueError(f"unknown HopRAG role: {role}")
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": json.dumps(result)},
                    "finish_reason": "stop",
                }
            ]
        }
