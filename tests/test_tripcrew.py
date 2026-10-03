import asyncio
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from agents.tripcrew import TravelAPI, TripCrew, check_plan, generate_scenarios
from agents.tripcrew.__main__ import run_suite
from agents.tripcrew.fixture_client import FixtureClient
from agents.tripcrew.scenarios import catalog, solve
from blackbox.replay import ReplayDivergence, ReplayEngine, patch_prompt, patch_tool_args
from blackbox.sdk import Recorder


def oracle_plan(scenario):
    oracle = solve(scenario)
    return {
        **{
            key: scenario.constraints()[key]
            for key in (
                "scenario_id",
                "origin",
                "destination",
                "departure",
                "return_date",
                "adults",
            )
        },
        **oracle,
        "itinerary": "Round trip with the selected hotel.",
    }


class ScenarioTests(unittest.TestCase):
    def test_300_seeded_scenarios_are_feasible_and_reproducible(self):
        scenarios = generate_scenarios()
        self.assertEqual(scenarios, generate_scenarios())
        self.assertNotEqual(scenarios, generate_scenarios(seed=8))
        self.assertEqual(len({s.scenario_id for s in scenarios}), 300)
        self.assertEqual(len({s.origin for s in scenarios}), 5)
        self.assertEqual(len({s.destination for s in scenarios}), 6)
        for scenario in scenarios:
            self.assertLessEqual(scenario.expected_total_inr, scenario.budget_inr)
            self.assertEqual(solve(scenario)["total_inr"], scenario.expected_total_inr)
            self.assertTrue(check_plan(scenario, oracle_plan(scenario)).passed)

    def test_five_reference_totals(self):
        # Independently inspected arithmetic is documented in docs/tripcrew.md.
        totals = [134999.04, 52249.98, 87875.84, 36186.40, 44750.01]
        for scenario, total in zip(generate_scenarios(5), totals):
            with self.subTest(scenario=scenario.scenario_id):
                self.assertEqual(scenario.expected_total_inr, total)
                self.assertTrue(
                    check_plan(scenario, {**oracle_plan(scenario), "total_inr": total}).passed
                )

    def test_checker_rejects_constraint_violations_and_bad_shapes(self):
        scenario = generate_scenarios(8)[7]  # All three hard constraints enabled.
        good = oracle_plan(scenario)
        mutations = [
            {"flight_id": "F-basic"},
            {"hotel_id": "H-basic"},
            {"flight_id": "invented"},
            {"hotel_id": "invented"},
            {"origin": "wrong"},
            {"destination": "wrong"},
            {"scenario_id": "wrong"},
            {"departure": "2020-01-01"},
            {"return_date": "2020-01-02"},
            {"adults": 99},
            {"total_inr": good["total_inr"] + 1.01},
            {"total_inr": float("nan")},
            {"total_inr": True},
            {"total_inr": "1"},
            {"itinerary": ""},
            {"extra": "field"},
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.assertFalse(check_plan(scenario, {**good, **mutation}).passed)
        self.assertTrue(check_plan(scenario, {**good, "total_inr": good["total_inr"] + 1}).passed)
        self.assertFalse(check_plan(replace(scenario, budget_inr=1), good).passed)
        self.assertFalse(check_plan(scenario, {}).passed)

    def test_tools_have_dates_and_explicit_fresh_fx(self):
        scenario = generate_scenarios(1)[0]
        api = TravelAPI([scenario], stale_fx=True)
        data = catalog(scenario)
        for function in (api.get_weather, api.fx_rate, api.visa_rules):
            self.assertIn("as_of", function(scenario.scenario_id, scenario.destination))
        fresh = api.fx_rate(scenario.scenario_id, scenario.destination, fresh=True)
        stale = api.fx_rate(scenario.scenario_id, scenario.destination)
        self.assertEqual(fresh["rate"], data["fresh_rate"])
        self.assertNotEqual(fresh["rate"], stale["rate"])
        for function in (api.search_flights, api.search_hotels):
            result = function(
                scenario.scenario_id,
                **{k: v for k, v in scenario.constraints().items() if k != "scenario_id"},
            )
            self.assertIn("as_of", result)
            self.assertEqual(len(result["options"]), 3)


class TripCrewTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.scenarios = generate_scenarios(20)
        self.client = FixtureClient()
        self.recorder = Recorder(self.directory.name, mode="offline", llm_client=self.client)

    def tearDown(self):
        self.recorder.close()
        self.directory.cleanup()

    async def record(self, scenario, api=None):
        agent = TripCrew(scenario, api or TravelAPI([scenario]), model="tripcrew-fixture-v1")
        with self.recorder.run(
            "tripcrew", scenario.scenario_id, 7, model="tripcrew-fixture-v1"
        ) as run:
            await agent(run)
        return agent, run

    async def test_20_end_to_end_runs_record_16_steps_with_scoped_parallel_workers(self):
        passed = 0
        for scenario in self.scenarios:
            _, run = await self.record(scenario)
            passed += run.outcome == "passed"
            rows = self.recorder.database.query("SELECT * FROM steps WHERE run_id=?", (run.run_id,))
            self.assertEqual(len(rows), 16)
            self.assertTrue(all(row["state_before"] and row["state_after"] for row in rows))
            self.assertTrue(check_plan(scenario, run.state.as_dict()["final_plan"]).passed)
        self.assertEqual(passed, 20)
        self.assertGreaterEqual(self.client.max_active, 2)
        for request in self.client.requests:
            envelope = json.loads(request["messages"][-1]["content"])
            payload = envelope["task"]
            self.assertNotIn("expected_total_inr", json.dumps(payload))
            if envelope["role"].startswith(("flight_", "hotel_")):
                self.assertNotIn("fx", payload)
                self.assertNotIn("budget", payload)
                self.assertNotIn("weather", payload)

    async def test_unchanged_replay_uses_no_new_calls_and_preserves_final_state(self):
        api = TravelAPI([self.scenarios[0]])
        agent, run = await self.record(self.scenarios[0], api)
        before_calls = dict(api.calls)
        before_chat = len(self.client.requests)
        final_ref = self.recorder.database.one(
            "SELECT state_after FROM steps WHERE run_id=? ORDER BY seq DESC LIMIT 1", (run.run_id,)
        )["state_after"]
        batch = await ReplayEngine(self.recorder).replay(run.run_id, agent)
        self.assertEqual(dict(api.calls), before_calls)
        self.assertEqual(len(self.client.requests), before_chat)
        self.assertEqual(batch.edited[0].state_after, final_ref)
        self.assertEqual(batch.edited[0].outcome, "passed")
        self.assertTrue(
            all(
                status == "cached"
                for addr, status in batch.edited[0].statuses.items()
                if "/state#" not in addr
            )
        )

    async def test_fx_fix_only_reexecutes_fx_budget_writer_verifier(self):
        scenario = self.scenarios[0]
        api = TravelAPI([scenario], stale_fx=True)
        agent, run = await self.record(scenario, api)
        self.assertEqual(run.outcome, "failed")
        original = self.recorder.database.query("SELECT * FROM steps WHERE run_id=?", (run.run_id,))
        before = dict(api.calls)
        batch = await ReplayEngine(self.recorder).replay(
            run.run_id, agent, edits=[patch_tool_args("fx/tool#1", {"fresh": True})]
        )
        self.assertEqual(batch.edited[0].outcome, "passed")
        executed = {
            addr
            for addr, status in batch.edited[0].statuses.items()
            if status == "live" and "/state#" not in addr
        }
        self.assertEqual(
            executed, {"fx/tool#1", "budget/tool#1", "writer/chat#1", "verifier/chat#1"}
        )
        for name in ("search_flights", "search_hotels", "get_weather", "visa_rules"):
            self.assertEqual(api.calls[name], before[name])
        self.assertEqual(api.calls["fx_rate"], before["fx_rate"] + 1)
        self.assertEqual(
            original,
            self.recorder.database.query("SELECT * FROM steps WHERE run_id=?", (run.run_id,)),
        )

    async def test_repeated_unchanged_replay_preserves_all_request_keys(self):
        api = TravelAPI([self.scenarios[0]])
        agent, run = await self.record(self.scenarios[0], api)
        calls, chats = dict(api.calls), len(self.client.requests)
        batch = await ReplayEngine(self.recorder).replay(run.run_id, agent, samples=5)
        for replay in batch.edited:
            self.assertEqual(replay.outcome, "passed")
            self.assertTrue(
                all(
                    status == "cached"
                    for addr, status in replay.statuses.items()
                    if "/state#" not in addr
                )
            )
        self.assertEqual(dict(api.calls), calls)
        self.assertEqual(len(self.client.requests), chats)

    async def test_five_fx_repairs_keep_prefix_cached_and_pair_live_model_seeds(self):
        scenario = self.scenarios[0]
        agent, run = await self.record(scenario, TravelAPI([scenario], stale_fx=True))
        before = len(self.client.requests)
        batch = await ReplayEngine(self.recorder).replay(
            run.run_id,
            agent,
            edits=[patch_tool_args("fx/tool#1", {"fresh": True})],
            samples=5,
            control=True,
        )
        self.assertEqual(batch.fix_pass_rate, 1)
        self.assertEqual(batch.control_pass_rate, 0)
        for replay in [*batch.edited, *batch.controls]:
            for addr in (
                "planner/chat#1",
                "flight/chat#1",
                "hotel/chat#1",
                "flight/chat#2",
                "hotel/chat#2",
            ):
                self.assertEqual(replay.statuses[addr], "cached")
        writer_seeds = [
            request["seed"]
            for request in self.client.requests[before:]
            if json.loads(request["messages"][-1]["content"])["role"] == "writer"
        ]
        self.assertEqual(writer_seeds, [seed for seed in range(7, 12) for _ in range(2)])

    async def test_seed_normalization_does_not_hide_other_request_drift(self):
        agent, run = await self.record(self.scenarios[0])

        async def drifting(replay_run):
            if replay_run.seed != 7:
                agent.model = "unexpected-model-change"
            await agent(replay_run)

        with self.assertRaisesRegex(ReplayDivergence, "request hash changed before the edit"):
            await ReplayEngine(self.recorder).replay(run.run_id, drifting, samples=2)

    async def test_explicit_seed_edit_is_not_normalized_back_to_the_base_seed(self):
        agent, run = await self.record(self.scenarios[0])
        batch = await ReplayEngine(self.recorder).replay(
            run.run_id,
            agent,
            edits=[patch_prompt("planner/chat#1", {"seed": 8})],
            samples=2,
        )
        for replay in batch.edited:
            step = self.recorder.database.one(
                "SELECT input_hash FROM steps WHERE run_id=? AND addr='planner/chat#1'",
                (replay.run_id,),
            )
            self.assertEqual(self.recorder.store.load_json(step["input_hash"])["seed"], 8)

    async def test_report_survives_timeout_after_a_completed_run(self):
        original_chat = self.client.chat

        async def failing(**request):
            task = json.loads(request["messages"][-1]["content"])
            if task["role"] == "planner" and "TC-7-0002" in task["task"]["request"]:
                raise httpx.ReadTimeout("simulated timeout")
            return await original_chat(**request)

        self.client.chat = failing
        root = Path(self.directory.name)
        args = SimpleNamespace(
            client="fixture",
            count=2,
            seed=7,
            command="run",
            stale_fx=False,
            data_dir=root / "suite",
            report=root / "report.json",
        )
        with patch("agents.tripcrew.__main__.FixtureClient", return_value=self.client):
            with self.assertRaises(httpx.ReadTimeout):
                await run_suite(args)
        report = json.loads(args.report.read_text())
        self.assertEqual(report["status"], "interrupted")
        self.assertEqual(report["count"], 2)
        self.assertEqual(report["completed_count"], 1)
        self.assertEqual(report["passed"], 1)
        self.assertEqual(report["pass_rate"], 1)
        self.assertTrue(report["runs"][0]["passed"])
        self.assertEqual(report["runs"][1]["error_type"], "ReadTimeout")
        self.assertFalse(args.report.with_suffix(".json.tmp").exists())

    async def test_first_run_cancellation_saves_a_valid_interrupted_report(self):
        async def cancelled(**request):
            raise asyncio.CancelledError()

        self.client.chat = cancelled
        root = Path(self.directory.name)
        args = SimpleNamespace(
            client="fixture",
            count=1,
            seed=7,
            command="run",
            stale_fx=False,
            data_dir=root / "cancelled",
            report=root / "cancelled.json",
        )
        with patch("agents.tripcrew.__main__.FixtureClient", return_value=self.client):
            with self.assertRaises(asyncio.CancelledError):
                await run_suite(args)
        report = json.loads(args.report.read_text())
        self.assertEqual(report["status"], "interrupted")
        self.assertEqual(report["completed_count"], 0)
        self.assertEqual(report["pass_rate"], 0)
        self.assertEqual(report["runs"][0]["status"], "interrupted")

    async def test_successful_suite_report_is_marked_complete(self):
        root = Path(self.directory.name)
        args = SimpleNamespace(
            client="fixture",
            count=1,
            seed=7,
            command="run",
            stale_fx=False,
            data_dir=root / "complete",
            report=root / "complete.json",
        )
        report = await run_suite(args)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["completed_count"], 1)
        self.assertEqual(report, json.loads(args.report.read_text()))

    async def test_malformed_parallel_worker_fails_without_deadlock(self):
        original_chat = self.client.chat

        async def malformed(**request):
            envelope = json.loads(request["messages"][-1]["content"])
            if envelope["role"] == "hotel_query":
                return {"choices": [{"message": {"content": "not JSON"}}]}
            return await original_chat(**request)

        self.client.chat = malformed
        with self.assertRaisesRegex(ValueError, "hotel_query returned invalid"):
            await asyncio.wait_for(self.record(self.scenarios[0]), timeout=5)
        row = self.recorder.database.one("SELECT outcome FROM runs")
        self.assertEqual(row["outcome"], "failed")

    async def test_parallel_prefix_divergence_fails_without_deadlock(self):
        agent, run = await self.record(self.scenarios[0])
        self.recorder.database.execute(
            "UPDATE steps SET state_before=? WHERE run_id=? AND addr='flight/chat#1'",
            ("0" * 64, run.run_id),
        )
        with self.assertRaises(ReplayDivergence):
            await asyncio.wait_for(ReplayEngine(self.recorder).replay(run.run_id, agent), timeout=5)


if __name__ == "__main__":
    unittest.main()
