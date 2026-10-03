"""MuSiQue-Ans adapter with a strict boundary between observations and gold labels."""

from __future__ import annotations

import hashlib
import json
import random
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

DATASET_REVISION = "22873a405dd809893b22ada0b499299fb612d2df"
DATASET_URL = (
    f"https://huggingface.co/datasets/bdsaglam/musique/resolve/{DATASET_REVISION}/"
    "musique_ans_v1.0_dev.jsonl"
)
DATASET_SHA256 = "15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b"
SOURCE_URL = "https://github.com/StonyBrookNLP/musique"


class RawParagraph(BaseModel):
    model_config = ConfigDict(strict=True)
    idx: int = Field(ge=0)
    title: str
    paragraph_text: str = Field(min_length=1)
    is_supporting: bool


class GoldHop(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True)
    id: int
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    paragraph_support_idx: int


class RawExample(BaseModel):
    model_config = ConfigDict(strict=True)
    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    paragraphs: list[RawParagraph] = Field(min_length=1, max_length=20)
    question_decomposition: list[GoldHop] = Field(min_length=2, max_length=4)
    answer: str = Field(min_length=1)
    answer_aliases: list[str]
    answerable: bool

    @model_validator(mode="after")
    def consistent(self):
        ids = {p.idx for p in self.paragraphs}
        if len(ids) != len(self.paragraphs):
            raise ValueError("paragraph IDs must be unique")
        if not self.answerable:
            raise ValueError("HopRAG requires MuSiQue-Ans answerable examples")
        if any(h.paragraph_support_idx not in ids for h in self.question_decomposition):
            raise ValueError("gold support refers to an absent paragraph")
        return self


@dataclass(frozen=True)
class Paragraph:
    doc_id: int
    title: str
    text: str


@dataclass(frozen=True)
class Question:
    question_id: str
    text: str
    paragraphs: tuple[Paragraph, ...]


@dataclass(frozen=True)
class Example:
    question: Question
    answer: str
    aliases: tuple[str, ...]
    gold_hops: tuple[GoldHop, ...]

    def oracle(self):
        """For future oracle interventions only; never passed to the agent."""
        return {
            "question_id": self.question.question_id,
            "answer": self.answer,
            "answer_aliases": list(self.aliases),
            "question_decomposition": [h.model_dump() for h in self.gold_hops],
        }


def parse_example(data: dict) -> Example:
    raw = RawExample.model_validate(data)
    return Example(
        Question(
            raw.id,
            raw.question,
            tuple(Paragraph(p.idx, p.title, p.paragraph_text) for p in raw.paragraphs),
        ),
        raw.answer,
        tuple(raw.answer_aliases),
        tuple(raw.question_decomposition),
    )


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_examples(path: Path) -> list[Example]:
    examples = []
    seen = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                example = parse_example(json.loads(line))
                if example.question.question_id in seen:
                    raise ValueError("duplicate question ID")
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            seen.add(example.question.question_id)
            examples.append(example)
    if not examples:
        raise ValueError("dataset is empty")
    return examples


def select_examples(examples: list[Example], count=50, seed=7) -> list[Example]:
    """Seeded round-robin sampling across gold hop counts, outside agent inference."""
    if not 1 <= count <= len(examples):
        raise ValueError(f"count must be between 1 and {len(examples)}")
    groups = defaultdict(list)
    for example in examples:
        groups[len(example.gold_hops)].append(example)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    selected = []
    while len(selected) < count:
        for hops in sorted(groups):
            if groups[hops] and len(selected) < count:
                selected.append(groups[hops].pop())
    return selected


def download_dataset(destination: Path) -> Path:
    """Fetch the pinned dev mirror and verify its published LFS SHA-256.

    Uses the official JSONL format. The HF mirror is explicitly attributed;
    callers can instead supply the authors' original JSONL via --dataset.
    """
    if destination.exists():
        if file_hash(destination) != DATASET_SHA256:
            raise ValueError(f"Existing dataset has a different checksum: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, suffix=".part", delete=False
        ) as out:
            temporary = Path(out.name)
            size = 0
            with httpx.stream("GET", DATASET_URL, follow_redirects=True, timeout=60) as response:
                response.raise_for_status()
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > 40_000_000:
                        raise ValueError("dataset download exceeds the expected size limit")
                    out.write(chunk)
        if file_hash(temporary) != DATASET_SHA256:
            raise ValueError("downloaded dataset failed SHA-256 verification")
        temporary.replace(destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return destination


def public_payload(question: Question) -> dict:
    return asdict(question)
