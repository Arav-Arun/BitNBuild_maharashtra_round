"""Research grounding, answer isolation, exact snapshot replay and regression export."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agents.research.agent import ResearchAgent, check_answer
from agents.research.dataset import SOURCE, save_json, task
from blackbox.forge.adapters import make_adapter
from blackbox.replay import ReplayEngine, patch_tool_args
from blackbox.sdk import Recorder

ROW = {
    "id": "public-question-1",
    "question": "In which city is the publisher of Example Magazine based?",
    "answer": "Pune",
    "split": "train",
    "supporting_facts": {"title": ["Magazine", "Publisher"]},
    "context": {
        "title": ["Magazine", "Publisher"],
        "sentences": [
            ["Example Magazine is published by Example Press."],
            ["Example Press is based in Pune."],
        ],
    },
}


class TestModel:
    def __init__(self):
        self.requests = []

    async def chat(self, **request):
        self.requests.append(request)
        payload = json.loads(request["messages"][-1]["content"])
        docs = payload["sources"]["documents"]
        content = {
            "answer": "Pune" if docs else "",
            "abstained": not docs,
            "citations": [
                {"title": d["title"], "sentence_id": 0, "quote": d["sentences"][0]} for d in docs
            ],
        }
        return {
            "choices": [{"message": {"content": json.dumps(content)}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 200, "completion_tokens": 40},
        }


class ResearchTests(unittest.TestCase):
    def test_reference_answers_and_fault_labels_never_enter_model_requests(self):
        async def execute(directory):
            model = TestModel()
            recorder = Recorder(directory, mode="live", llm_client=model)
            try:
                payload = task([ROW], ROW["question"])
                with recorder.run(
                    "research", "RESEARCH-test", 7, model="openai/gpt-oss-20b"
                ) as run:
                    await ResearchAgent(payload, model="openai/gpt-oss-20b")(run)
                self.assertEqual(run.outcome, "passed")
                from blackbox.ml.dataset import load_trace

                trace = load_trace(recorder.database, recorder.store, run.run_id)
                self.assertNotIn("checker/state#1", trace.addrs)
                self.assertFalse(any("checker/state#1" in edge for edge in trace.edges))
                self.assertEqual(len(model.requests), 2)
                self.assertNotIn("expected_answer", json.dumps(model.requests))
                self.assertNotIn("support_titles", json.dumps(model.requests))
                self.assertNotIn("empty_retrieval", json.dumps(model.requests))
                self.assertEqual(model.requests[0]["response_format"]["type"], "json_schema")
            finally:
                recorder.close()

        with tempfile.TemporaryDirectory() as directory:
            asyncio.run(execute(directory))

    def test_invalid_citations_fail_even_when_the_answer_matches(self):
        payload = task([ROW], ROW["question"])
        sources = ResearchAgent(payload, model="test").retrieve_documents(ROW["question"])
        result = check_answer(
            payload,
            {
                "answer": "Pune",
                "abstained": False,
                "citations": [
                    {"title": "Publisher", "sentence_id": 0, "quote": "Invented quotation"}
                ],
            },
            sources,
        )
        self.assertFalse(result["passed"])
        self.assertIn("does not match", result["reason"])

    def test_custom_questions_have_no_benchmark_accuracy_claim(self):
        payload = task([ROW], "Where is Example Press based?")
        self.assertIsNone(payload["expected_answer"])
        self.assertEqual(payload["evaluation"], "Grounding only")
        with self.assertRaisesRegex(ValueError, "No matching documents"):
            task([ROW], "zyxwv nonsense unknown")

    def test_research_replay_and_export_use_saved_documents_without_network(self):
        async def execute(directory):
            recorder = Recorder(directory, mode="live", llm_client=TestModel())
            task_id = "RESEARCH-snapshot"
            payload = task([ROW], ROW["question"], empty_retrieval=True)
            save_json(directory / "research-tasks" / f"{task_id}.json", payload)
            try:
                with recorder.run("research", task_id, 7, model="research-test") as run:
                    await ResearchAgent(payload, model="research-test")(run)
                self.assertEqual(run.outcome, "failed")
                adapter = make_adapter(recorder, argparse.Namespace(agent="research"))
                batch = await ReplayEngine(recorder).replay(
                    run.run_id,
                    adapter.factory(run.run_id),
                    edits=[patch_tool_args("retriever/tool#1", {"restore": True}, known_good=True)],
                    samples=5,
                    control=True,
                )
                self.assertEqual(batch.verdict, "VERIFIED")
                from blackbox.export.regression import export

                result = export(recorder, batch.fork_id, directory / "exports")
                self.assertTrue(
                    (result.fixture_dir / "research-tasks" / f"{task_id}.json").is_file()
                )
                replay = subprocess.run(
                    [sys.executable, str(result.test_path)],
                    capture_output=True,
                    text=True,
                    env={**os.environ, "MODE": "recorded"},
                )
                self.assertEqual(replay.returncode, 0, replay.stderr + replay.stdout)
            finally:
                recorder.close()

        with tempfile.TemporaryDirectory() as directory:
            asyncio.run(execute(Path(directory)))

    def test_live_api_records_research_and_hides_answers_in_question_list(self):
        from fastapi.testclient import TestClient

        from server.app import app

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_json(
                root / "research" / "hotpotqa-train.json",
                {"rows": [ROW], "split": "train", "source": SOURCE, "downloaded_at": "2026-10-04"},
            )
            with patch.dict(os.environ, {"DATA_DIR": str(root), "MODE": "live"}):
                with TestClient(app) as client:
                    svc = app.state.service
                    from dataclasses import replace

                    svc.settings = replace(svc.settings, groq_api_key="test-provider-key")
                    svc.agents["research"].recorder.llm_client = TestModel()
                    questions = client.get("/research/questions").json()
                    self.assertNotIn("answer", questions["items"][0])
                    self.assertNotIn("supporting_facts", questions["items"][0])
                    healthy = client.post(
                        "/tasks/run", json={"workflow": "research", "prompt": ROW["question"]}
                    )
                    self.assertEqual(healthy.status_code, 201, healthy.text)
                    response = client.post(
                        "/tasks/run",
                        json={
                            "workflow": "research",
                            "prompt": ROW["question"],
                            "inject_empty_retrieval": True,
                        },
                    )
                    self.assertEqual(response.status_code, 201, response.text)
                    result = response.json()
                    self.assertEqual(result["status"], "failed")
                    detail = client.get(f"/runs/{result['run_id']}").json()
                    self.assertEqual(detail["task_text"], ROW["question"])
                    self.assertEqual(detail["run"]["cost"]["llm_calls"], 2)
                    self.assertEqual(detail["run"]["origin"], "injected")
                    twin = svc.nearest_twin(result["run_id"])
                    self.assertIsNotNone(twin)
                    self.assertTrue(twin.same_task)
                    from server.models import NearestTwin

                    NearestTwin(
                        run=twin.summary,
                        similarity=twin.similarity,
                        same_task=twin.same_task,
                        basis=twin.basis,
                        known_good_values=[],
                    )
                    self.assertEqual(twin.summary.run_id, healthy.json()["run_id"])


if __name__ == "__main__":
    unittest.main()
