import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import httpx

from blackbox.config import Settings
from blackbox.llm import AsyncLLMClient, parse_duration
from blackbox.store import SQLiteDatabase
from server.models import (
    Diagnosis,
    DiffResponse,
    EvalResponse,
    ForkEvent,
    RunDetail,
    RunSummary,
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
        root = Path(__file__).parents[1]
        runs = json.loads((root / "web/mocks/runs.json").read_text())
        diagnosis = json.loads((root / "web/mocks/diagnosis.json").read_text())
        run_detail = json.loads((root / "web/mocks/run-detail.json").read_text())
        fork_events = json.loads((root / "web/mocks/fork-events.json").read_text())
        diff = json.loads((root / "web/mocks/diff.json").read_text())
        evaluation = json.loads((root / "web/mocks/eval.json").read_text())
        self.assertEqual(len([RunSummary.model_validate(run) for run in runs]), 2)
        self.assertEqual(Diagnosis.model_validate(diagnosis).run_id, "TC-0412")
        self.assertEqual(RunDetail.model_validate(run_detail).steps[0].addr, "fx/tool#1")
        self.assertEqual(len([ForkEvent.model_validate(event) for event in fork_events]), 3)
        self.assertEqual(DiffResponse.model_validate(diff).first_divergence, "fx/tool#1")
        self.assertEqual(EvalResponse.model_validate(evaluation).sample_size, 405)

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
