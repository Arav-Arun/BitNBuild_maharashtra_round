import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import httpx

from blackbox.api.reader import build_diff
from blackbox.config import Settings
from blackbox.llm import AsyncLLMClient, parse_duration
from blackbox.store import SQLiteDatabase
from server.models import (
    Diagnosis,
    DiffResponse,
    EvalResponse,
    RunDetail,
    RunList,
)


class TaskOneTests(unittest.TestCase):
    def test_settings_file_and_environment_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text("MODE=offline\nAGENT_MODEL=file-model\nDATA_DIR=tmp-data\n")
            settings = Settings.load(env_file, {"AGENT_MODEL": "environment-model"})
            self.assertEqual(settings.mode, "offline")
            self.assertEqual(settings.agent_model, "environment-model")
            self.assertEqual(settings.data_dir, Path("tmp-data"))

    def test_schema_initializes_with_wal_and_expected_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteDatabase(Path(directory) / "blackbox.db") as database:
                journal = database.one("PRAGMA journal_mode")
                tables = {
                    row["name"]
                    for row in database.query("SELECT name FROM sqlite_master WHERE type = 'table'")
                }
            self.assertEqual(journal["journal_mode"], "wal")
            self.assertTrue(
                {
                    "runs",
                    "steps",
                    "edges",
                    "cassette",
                    "forks",
                    "labels",
                    "diagnoses",
                    "regression_exports",
                }.issubset(tables)
            )

    def test_mock_fixtures_match_contracts(self):
        root = Path(__file__).parents[1] / "web/mocks"

        def load(name: str):
            return json.loads((root / f"{name}.json").read_text())

        self.assertGreater(RunList.model_validate(load("runs")).total, 0)
        self.assertTrue(Diagnosis.model_validate(load("diagnosis")).run_id)
        self.assertTrue(RunDetail.model_validate(load("run-detail")).steps)
        self.assertTrue(DiffResponse.model_validate(load("diff")).first_divergence)
        self.assertTrue(EvalResponse.model_validate(load("eval")).model_dump())
        events = load("fork-events-verified")["events"]
        self.assertEqual(events[-1]["event"], "summary")

    def test_api_diff_marks_dependency_path_from_first_changed_step(self):
        root = Path(__file__).parents[1]
        detail = RunDetail.model_validate(
            json.loads((root / "web/mocks/run-detail.json").read_text())
        )
        steps = list(detail.steps)
        fx_index = next(index for index, step in enumerate(steps) if step.addr == "fx/tool#1")
        fx = steps[fx_index]
        steps[fx_index] = fx.model_copy(
            update={
                "hashes": fx.hashes.model_copy(update={"output": "f" * 64}),
                "cache_status": "edited",
            }
        )
        fork = detail.model_copy(
            update={
                "run": detail.run.model_copy(
                    update={"run_id": "tc-fork", "parent_run_id": detail.run.run_id}
                ),
                "steps": steps,
            }
        )

        diff = build_diff(detail, fork, invalidated=None, nearest=None)
        self.assertEqual(diff.first_divergence, "fx/tool#1")
        inside = {row.addr for row in diff.rows if row.in_cone}
        self.assertIn("fx/tool#1", inside)
        self.assertIn("budget/tool#1", inside)
        self.assertIn("writer/chat#1", inside)

    def test_llm_client_retries_429_and_honours_openai_shape(self):
        attempts = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return httpx.Response(429, headers={"retry-after": "0"})
            body = json.loads(request.content)
            self.assertEqual(body["model"], "test-model")
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "pong"}}]},
                headers={"x-ratelimit-remaining-requests": "12"},
            )

        async def exercise() -> dict:
            async with AsyncLLMClient(
                api_key="test",
                base_url="https://example.invalid/v1",
                max_retries=1,
                transport=httpx.MockTransport(handler),
            ) as client:
                return await client.chat(
                    messages=[{"role": "user", "content": "ping"}],
                    model="test-model",
                )

        result = asyncio.run(exercise())
        self.assertEqual(result["choices"][0]["message"]["content"], "pong")
        self.assertEqual(attempts, 2)
        self.assertAlmostEqual(parse_duration("2m59.5s"), 179.5)


if __name__ == "__main__":
    unittest.main()
