"""Bounded, attributed HotpotQA downloads; immutable per-task replay snapshots."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote

import httpx

DATASET = "hotpotqa/hotpot_qa"
SOURCE = f"https://huggingface.co/datasets/{DATASET}"
LICENSE = "CC-BY-SA-4.0"


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def download(directory: Path, *, count: int = 24, split: str = "validation") -> Path:
    if not 1 <= count <= 100 or split not in {"train", "validation"}:
        raise ValueError("Choose train/validation and a count from 1 to 100")
    with httpx.Client(timeout=90) as client:
        response = client.get(
            "https://datasets-server.huggingface.co/rows",
            params={
                "dataset": DATASET,
                "config": "distractor",
                "split": split,
                "offset": 0,
                "length": count,
            },
        )
        response.raise_for_status()
    rows = [item["row"] for item in response.json()["rows"]]
    if len(rows) != count:
        raise ValueError("Dataset download returned an incomplete slice")
    path = directory / f"hotpotqa-{split}.json"
    save_json(
        path,
        {
            "source": SOURCE,
            "license": LICENSE,
            "config": "distractor",
            "split": split,
            "downloaded_at": datetime.now(UTC).isoformat(),
            "rows": rows,
            "sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
        },
    )
    return path


def load(directory: Path) -> list[dict]:
    rows = []
    for path in sorted(directory.glob("hotpotqa-*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(
            {
                **row,
                "split": payload["split"],
                "source": payload["source"],
                "downloaded_at": payload["downloaded_at"],
            }
            for row in payload["rows"]
        )
    if not rows:
        raise ValueError("Download documents first: python -m agents.research download")
    return rows


def normalize(text: str) -> str:
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", re.sub(r"[^\w\s]", "", text.lower())).split())


def documents(row: dict) -> list[dict]:
    return [
        {
            "title": title,
            "sentences": sentences,
            "url": "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_")),
            "source": SOURCE,
        }
        for title, sentences in zip(row["context"]["title"], row["context"]["sentences"])
    ]


def task(rows: list[dict], prompt: str, *, empty_retrieval: bool = False) -> dict:
    match = next((row for row in rows if normalize(row["question"]) == normalize(prompt)), None)
    if match:
        docs = documents(match)
    else:
        # Custom questions search the downloaded corpus, not an invented web response.
        terms = set(normalize(prompt).split()) - {"what", "which", "who", "where", "was", "is"}
        corpus = {doc["title"]: doc for row in rows for doc in documents(row)}
        ranked = sorted(
            corpus.values(),
            key=lambda doc: len(
                terms & set(normalize(doc["title"] + " " + " ".join(doc["sentences"])).split())
            ),
            reverse=True,
        )
        docs = [
            doc
            for doc in ranked[:8]
            if terms & set(normalize(doc["title"] + " " + " ".join(doc["sentences"])).split())
        ]
        if not docs:
            raise ValueError(
                "No matching documents in the downloaded corpus. Try a listed question."
            )
    return {
        "question": prompt.strip(),
        "documents": docs,
        "empty_retrieval": empty_retrieval,
        "benchmark_id": match["id"] if match else None,
        "split": match.get("split") if match else None,
        "expected_answer": match["answer"] if match else None,
        "support_titles": sorted(set(match["supporting_facts"]["title"])) if match else [],
        "source": SOURCE,
        "license": LICENSE,
        "document_snapshot_sha256": hashlib.sha256(
            json.dumps(docs, sort_keys=True).encode()
        ).hexdigest(),
        "downloaded_at": match.get("downloaded_at") if match else rows[0].get("downloaded_at"),
        "evaluation": "HotpotQA answer and supporting documents" if match else "Grounding only",
    }


def snapshot(directory: Path, task_id: str) -> dict:
    return json.loads(
        (directory / "research-tasks" / f"{task_id}.json").read_text(encoding="utf-8")
    )
