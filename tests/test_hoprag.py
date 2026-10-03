import json
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agents.hoprag.__main__ import run_suite
from agents.hoprag.agent import evaluated_agent, resolve_question
from agents.hoprag.checker import check_answer, normalize_answer, token_f1
from agents.hoprag.data import (
    download_dataset,
    load_examples,
    parse_example,
    public_payload,
    select_examples,
)
from agents.hoprag.heuristic import HeuristicClient
from agents.hoprag.tools import RetrievalTools
from blackbox.replay import ReplayEngine
from blackbox.sdk import Recorder


def raw_example(hops=2, index=0):
    """Original synthetic test data, not copied MuSiQue questions."""
    entities = ["Mira Vale", "Lumen", "Aster", "Vega", "ORION"]
    paragraphs = [
        {
            "idx": i,
            "title": entities[i],
            "paragraph_text": f"A fact about {entities[i]}. Answer: {entities[i + 1] if i < hops - 1 else 'ORION'}.",
            "is_supporting": True,
        }
        for i in range(hops)
    ]
    paragraphs.extend(
        {
            "idx": i,
            "title": f"Unrelated document {i}",
            "paragraph_text": f"Irrelevant filler concerning sample {i}.",
            "is_supporting": False,
        }
        for i in range(hops, 20)
    )
    return {
        "id": f"synthetic-{hops}-{index}",
        "question": "What is the launch code linked to Mira Vale?",
        "paragraphs": paragraphs,
        "answer": "ORION",
        "answer_aliases": ["GOLD_ALIAS_ONLY"],
        "answerable": True,
        "question_decomposition": [
            {
                "id": i,
                "question": f"GOLD_SUBQUESTION_ONLY_{i}",
                "answer": entities[i + 1] if i < hops - 1 else "ORION",
                "paragraph_support_idx": i,
            }
            for i in range(hops)
        ],
    }


class ScriptedClient:
    """Reads the explicit Answer marker in our synthetic public documents."""

    def __init__(self, hops=2, invalid_role=None):
        self.hops = hops
        self.invalid_role = invalid_role
        self.requests = []

    async def chat(self, **request):
        self.requests.append(request)
        envelope = json.loads(request["messages"][-1]["content"])
        role, task = envelope["role"], envelope["task"]
        if role == self.invalid_role:
            result = {"text": "invented", "doc_ids": [999]}
        elif role == "decompose":
            result = {
                "questions": ["Find fact about Mira Vale"]
                + [f"Find fact about #{i}" for i in range(1, self.hops)]
            }
        else:
            document = task["documents"][-1]
            result = {
                "text": document["text"].split("Answer: ")[-1].rstrip("."),
                "doc_ids": [document["doc_id"]],
            }
        return {"choices": [{"message": {"content": json.dumps(result)}, "finish_reason": "stop"}]}


class DataAndScoringTests(unittest.TestCase):
    def test_adapter_separates_oracles_and_keeps_actual_paragraph_count(self):
        raw = raw_example()
        raw["paragraphs"].pop()
        example = parse_example(raw)
        visible = json.dumps(public_payload(example.question))
        self.assertEqual(len(example.question.paragraphs), 19)
        for private in (
            "is_supporting",
            "question_decomposition",
            "GOLD_ALIAS_ONLY",
            "GOLD_SUBQUESTION_ONLY",
        ):
            self.assertNotIn(private, visible)
        oracle = example.oracle()
        self.assertEqual(oracle["question_decomposition"][0]["paragraph_support_idx"], 0)
        self.assertEqual(oracle["answer_aliases"], ["GOLD_ALIAS_ONLY"])

    def test_invalid_and_duplicate_dataset_rows_fail_with_line_numbers(self):
        for mutation in (
            {"answerable": False},
            {"question_decomposition": []},
            {"paragraphs": [raw_example()["paragraphs"][0]] * 20},
        ):
            with self.subTest(mutation=mutation):
                with self.assertRaises(ValueError):
                    parse_example({**raw_example(), **mutation})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.jsonl"
            row = json.dumps(raw_example())
            path.write_text(row + "\n" + row + "\n")
            with self.assertRaisesRegex(ValueError, r":2: duplicate question ID"):
                load_examples(path)

    def test_download_does_not_overwrite_an_existing_unknown_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.jsonl"
            path.write_text("existing user data")
            with self.assertRaisesRegex(ValueError, "different checksum"):
                download_dataset(path)
            self.assertEqual(path.read_text(), "existing user data")

    def test_sample_is_seeded_balanced_and_without_duplicates(self):
        examples = [parse_example(raw_example(hops, i)) for hops in (2, 3, 4) for i in range(10)]
        selected = select_examples(examples, 12, 7)
        self.assertEqual(Counter(len(e.gold_hops) for e in selected), {2: 4, 3: 4, 4: 4})
        self.assertEqual(selected, select_examples(examples, 12, 7))
        self.assertNotEqual(selected, select_examples(examples, 12, 8))
        self.assertEqual(len({e.question.question_id for e in selected}), 12)
        with self.assertRaises(ValueError):
            select_examples(examples, 31)

    def test_aliases_normalization_multiset_f1_and_threshold(self):
        self.assertEqual(normalize_answer(" The U.S.,  "), "us")
        self.assertTrue(check_answer("NYC", "New York City", ("NYC",)).exact_match)
        check = check_answer("red planet mars", "the red planet")
        self.assertEqual(check.f1, 0.8)
        self.assertTrue(check.passed)
        self.assertFalse(check.exact_match)
        self.assertFalse(check_answer("red planet mars venus", "red planet").passed)
        self.assertEqual(token_f1("red red", "red blue"), 0.5)
        self.assertFalse(check_answer("", "ORION").passed)
        with self.assertRaises(ValueError):
            check_answer("", "", ())

    def test_bm25_is_local_stable_and_returns_only_public_fields(self):
        example = parse_example(raw_example())
        tools = RetrievalTools(example.question)
        hits = tools.search("Mira Vale", corpus_id=tools.corpus_id)
        self.assertEqual(hits[0]["doc_id"], 0)
        doc = tools.read(0, corpus_id=tools.corpus_id)
        self.assertEqual(set(doc), {"doc_id", "title", "text"})
        ties = tools.search("xyzunmatched", corpus_id=tools.corpus_id)
        self.assertEqual([h["doc_id"] for h in ties], [0, 1, 2])
        self.assertEqual(tools.search(" ", corpus_id=tools.corpus_id), [])
        with self.assertRaises(ValueError):
            tools.read(99, corpus_id=tools.corpus_id)
        with self.assertRaises(ValueError):
            tools.search("Mira", corpus_id="wrong-corpus")
        changed_doc = replace(example.question.paragraphs[0], text="Different version")
        changed_question = replace(
            example.question, paragraphs=(changed_doc, *example.question.paragraphs[1:])
        )
        self.assertNotEqual(tools.corpus_id, RetrievalTools(changed_question).corpus_id)

    def test_forward_and_missing_references_are_rejected(self):
        self.assertEqual(
            resolve_question("Where did #1 meet #2?", ["A", "B"]), "Where did A meet B?"
        )
        for template in ("#0", "#3", " "):
            with self.assertRaises(ValueError):
                resolve_question(template, ["A", "B"])


class HopRAGTests(unittest.IsolatedAsyncioTestCase):
    async def test_later_hop_can_read_the_same_top_ranked_document(self):
        raw = raw_example()
        raw["paragraphs"][0]["paragraph_text"] = (
            "Mira Vale founded Lumen observatory. Lumen observatory launch code is ORION."
        )
        raw["paragraphs"][1].update(title="Unrelated", paragraph_text="Irrelevant filler.")
        raw["question_decomposition"][1]["paragraph_support_idx"] = 0
        example = parse_example(raw)

        class SharedDocumentClient:
            async def chat(self, **request):
                envelope = json.loads(request["messages"][-1]["content"])
                role, task = envelope["role"], envelope["task"]
                if role == "decompose":
                    result = {"questions": ["Mira Vale observatory", "#1 launch code"]}
                elif any("ORION" in doc["text"] for doc in task["documents"]):
                    result = {
                        "text": "Lumen"
                        if role == "hop" and task["question"] == "Mira Vale observatory"
                        else "ORION",
                        "doc_ids": [0],
                    }
                else:
                    result = {"text": "", "doc_ids": []}
                return {"choices": [{"message": {"content": json.dumps(result)}}]}

        with tempfile.TemporaryDirectory() as directory:
            recorder = Recorder(directory, mode="offline", llm_client=SharedDocumentClient())
            try:
                agent = evaluated_agent(example, RetrievalTools(example.question))
                with recorder.run("hoprag", example.question.question_id, 7) as run:
                    await agent(run)
                self.assertEqual(run.outcome, "passed")
                state = run.state.as_dict()
                self.assertEqual(state["hop1_doc1"]["doc_id"], 0)
                self.assertEqual(state["hop2_hits"][0]["doc_id"], 0)
                self.assertEqual(state["hop2_doc1"]["doc_id"], 0)
                batch = await ReplayEngine(recorder).replay(run.run_id, agent)
                self.assertEqual(batch.reexecuted_steps, 0)
                self.assertEqual(batch.edited[0].outcome, "passed")
            finally:
                recorder.close()

    async def test_two_three_four_hops_and_fully_cached_unchanged_replay(self):
        for hops in (2, 3, 4):
            with self.subTest(hops=hops), tempfile.TemporaryDirectory() as directory:
                example = parse_example(raw_example(hops))
                client = ScriptedClient(hops)
                recorder = Recorder(directory, mode="offline", llm_client=client)
                try:
                    tools = RetrievalTools(example.question)
                    agent = evaluated_agent(example, tools, model="test-script")
                    with recorder.run("hoprag", example.question.question_id, 7) as run:
                        await agent(run)
                    self.assertEqual(run.outcome, "passed")
                    steps = recorder.database.query(
                        "SELECT * FROM steps WHERE run_id=? ORDER BY seq", (run.run_id,)
                    )
                    self.assertEqual(len(steps), 3 + hops * 3)
                    self.assertTrue(all(s["input_hash"] and s["output_hash"] for s in steps))
                    final_ref = steps[-1]["state_after"]
                    calls, chats = dict(tools.calls), len(client.requests)
                    batch = await ReplayEngine(recorder).replay(run.run_id, agent)
                    self.assertEqual(batch.reexecuted_steps, 0)
                    self.assertEqual(batch.cached_steps, len(steps))
                    self.assertEqual(batch.edited[0].state_after, final_ref)
                    self.assertEqual(batch.edited[0].outcome, run.outcome)
                    self.assertEqual(dict(tools.calls), calls)
                    self.assertEqual(len(client.requests), chats)
                    self.assertEqual(
                        steps,
                        recorder.database.query(
                            "SELECT * FROM steps WHERE run_id=? ORDER BY seq", (run.run_id,)
                        ),
                    )
                    recorded_text = json.dumps(client.requests) + json.dumps(run.state.as_dict())
                    self.assertNotIn("GOLD_ALIAS_ONLY", recorded_text)
                    self.assertNotIn("GOLD_SUBQUESTION_ONLY", recorded_text)
                    edges = recorder.database.query(
                        "SELECT * FROM edges WHERE run_id=?", (run.run_id,)
                    )
                    self.assertTrue(
                        any(
                            e["src_addr"] == "hop1/answer#1" and e["dst_addr"] == "hop2/search#1"
                            for e in edges
                        )
                    )
                finally:
                    recorder.close()

    async def test_unread_citations_fail_and_are_recorded(self):
        example = parse_example(raw_example())
        with tempfile.TemporaryDirectory() as directory:
            recorder = Recorder(
                directory, mode="offline", llm_client=ScriptedClient(invalid_role="hop")
            )
            try:
                agent = evaluated_agent(example, RetrievalTools(example.question))
                with self.assertRaisesRegex(ValueError, "cited a document"):
                    with recorder.run("hoprag", example.question.question_id, 7) as run:
                        await agent(run)
                self.assertEqual(run.outcome, "failed")
                row = recorder.database.one(
                    "SELECT error_type FROM steps WHERE addr='hop1/answer#1'"
                )
                self.assertEqual(row["error_type"], "ValueError")
            finally:
                recorder.close()

    async def test_heuristic_and_scored_runner_require_no_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "fixture.jsonl"
            dataset.write_text("\n".join(json.dumps(raw_example(hops)) for hops in (2, 3, 4)))
            args = SimpleNamespace(
                client="heuristic",
                dataset=dataset,
                count=3,
                seed=7,
                data_dir=root / "runs",
                hops=3,
                read_k=1,
                verify_replay=True,
                report=root / "report.json",
            )
            with patch("socket.socket.connect", side_effect=AssertionError("Network is disabled")):
                report = await run_suite(args)
            self.assertEqual(report["count"], 3)
            self.assertEqual(report["pipeline_errors"], 0)
            self.assertEqual(report["replay_verified"], 3)
            self.assertEqual(report["status"], "complete")
            oracle = json.loads(Path(report["oracle_file"]).read_text())
            self.assertEqual(len(oracle["questions"]), 3)
            self.assertEqual(json.loads(args.report.read_text())["count"], 3)
            with self.assertRaises(ValueError):
                HeuristicClient(hops=5)


if __name__ == "__main__":
    unittest.main()
