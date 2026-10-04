"""User-entered task parsing and the offline TripCrew run path."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from agents.tripcrew import TravelAPI, TripCrew
from agents.tripcrew.fixture_client import FixtureClient
from agents.tripcrew.prompt import catalog_seed, parse_trip_prompt
from agents.tripcrew.scenarios import catalog, money
from blackbox.api.agents import TripCrewProfile
from blackbox.forge.adapters import TripCrewAdapter
from blackbox.replay import Edit, ReplayEngine, patch_tool_args
from blackbox.sdk import Recorder

PROMPT = (
    "Plan a trip from Mumbai to Singapore departing 2026-12-12, returning 2026-12-15, "
    "for 4 adults. Budget INR 160000. Vegetarian: no; refundable: no; no red-eye: no."
)


class PromptParserTests(unittest.TestCase):
    def test_parses_supported_trip_and_computes_oracle_budget(self):
        scenario = parse_trip_prompt(PROMPT, scenario_id="PROMPT-TEST123456", seed=13)
        self.assertEqual((scenario.origin, scenario.destination), ("Mumbai", "Singapore"))
        self.assertEqual((scenario.departure, scenario.return_date), ("2026-12-12", "2026-12-15"))
        self.assertEqual(scenario.adults, 4)
        self.assertEqual(scenario.budget_inr, 160000)
        self.assertGreater(scenario.expected_total_inr, 0)
        self.assertFalse(scenario.vegetarian)
        self.assertFalse(scenario.refundable)
        self.assertFalse(scenario.no_red_eye)

    def test_run_summary_handles_decimal_budget_and_sentence_punctuation(self):
        request = (
            "Trip PROMPT-TEST123456: travel from Mumbai to Singapore departing 2026-12-12, "
            "returning 2026-12-15, for 4 adults. Budget INR 160000.00. Vegetarian: false."
        )
        first_input = {
            "messages": [
                {
                    "role": "user",
                    "content": json.dumps(
                        {"task": {"request": request, "original_prompt": PROMPT}}
                    ),
                }
            ]
        }
        summary, full_request = TripCrewProfile("tripcrew", "TripCrew", "", "final_plan").task(
            "run-1", first_input
        )
        self.assertEqual(summary, "Mumbai → Singapore, 4 adults, under ₹1,60,000")
        self.assertEqual(full_request, PROMPT)

    def test_rejects_missing_details_and_unsupported_cities(self):
        with self.assertRaisesRegex(ValueError, "departure and return dates"):
            parse_trip_prompt(
                "From Mumbai to Singapore for 2 adults with budget INR 50000",
                scenario_id="PROMPT-TEST123457",
                seed=7,
            )
        with self.assertRaisesRegex(ValueError, "Destination must be one of"):
            parse_trip_prompt(
                PROMPT.replace("Singapore", "Osaka"), scenario_id="PROMPT-TEST123458", seed=7
            )
        # Words after a valid city are not part of its name, so the real problem is reported.
        with self.assertRaisesRegex(ValueError, "departure and return dates"):
            parse_trip_prompt(
                "from Mumbai to Singapore next Friday for 2 adults budget INR 90000",
                scenario_id="PROMPT-TEST123459",
                seed=7,
            )

    def test_accepts_rupee_formats_and_common_city_names(self):
        cases = {
            "from Mumbai to Singapore, 2026-12-12 to 2026-12-15, 2 adults, budget ₹1,00,000": (
                "Mumbai",
                "Singapore",
                100_000,
            ),
            "from Bombay to Dubai departing 2026-12-12 returning 2026-12-14 for 2 adults, "
            "budget 1.5 lakh": ("Mumbai", "Dubai", 150_000),
            "from Bangalore to London departing 2026-12-12 returning 2026-12-18 for 1 adult, "
            "Rs. 90,000": ("Bengaluru", "London", 90_000),
            "from new delhi to Paris 2026-12-12 to 2026-12-20 for 2 travellers, 250000 rupees": (
                "Delhi",
                "Paris",
                250_000,
            ),
        }
        for prompt, (origin, destination, budget) in cases.items():
            with self.subTest(prompt=prompt):
                scenario = parse_trip_prompt(prompt, scenario_id="PROMPT-TEST123460", seed=7)
                self.assertEqual((scenario.origin, scenario.destination), (origin, destination))
                self.assertEqual(scenario.budget_inr, budget)

    def test_no_red_eye_preference_accepts_hyphenated_text_and_explicit_no(self):
        for preference, expected in (
            ("No red-eye", True),
            ("No red eye", True),
            ("No red-eye: no", False),
        ):
            with self.subTest(preference=preference):
                scenario = parse_trip_prompt(
                    PROMPT.split(" Vegetarian:")[0] + f" {preference}.",
                    scenario_id="PROMPT-PREFERENCE",
                    seed=7,
                )
                self.assertEqual(scenario.no_red_eye, expected)


# The one-click examples on the web "New task" page (web/components/pages/NewRunPage.tsx).
DEMO_EXAMPLES = [
    "Plan a trip from Delhi to Tokyo departing 2026-12-12, returning 2026-12-17, for 2 adults. "
    "Budget ₹1,00,000.",
    "Family holiday from Chennai to Bangkok, 2026-12-20 to 2026-12-24, 3 adults, budget "
    "₹1,40,000. Vegetarian: yes.",
    "Weekend from Hyderabad to Dubai departing 2026-12-05, returning 2026-12-08, for 2 adults. "
    "Budget Rs. 1,00,000. No red-eye: yes.",
    "Solo trip from Bangalore to London departing 2026-12-01, returning 2026-12-07, for 1 adult. "
    "Budget ₹80,000. Refundable: yes.",
    "Plan a trip from Hyderabad to London departing 2026-12-12, returning 2026-12-17, "
    "for 2 adults. Budget ₹80,000.",
]


class DemoExampleTests(unittest.TestCase):
    def test_examples_parse_and_fit_their_budget_with_headroom(self):
        # With headroom, the only failure an old exchange rate causes is the INR total, so the
        # suggested fix can be VERIFIED. The budget is a constraint, not an inventory seed.
        for prompt in DEMO_EXAMPLES:
            with self.subTest(prompt=prompt):
                scenario_id, seed = catalog_seed(prompt)
                scenario = parse_trip_prompt(prompt, scenario_id=scenario_id, seed=seed)
                self.assertLessEqual(scenario.expected_total_inr, 0.85 * scenario.budget_inr)
                self.assertEqual(
                    catalog_seed(" ".join(prompt.upper().split())), (scenario_id, seed)
                )

    def test_changing_only_budget_keeps_the_same_catalog_and_repairable_fixture(self):
        prompt = DEMO_EXAMPLES[-1]
        larger_budget = prompt.replace("₹80,000", "₹90,000")
        scenario_id, seed = catalog_seed(prompt)
        larger_id, larger_seed = catalog_seed(larger_budget)
        self.assertEqual((scenario_id, seed), (larger_id, larger_seed))
        scenario = parse_trip_prompt(prompt, scenario_id=scenario_id, seed=seed)
        self.assertLess(scenario.expected_total_inr, scenario.budget_inr)

        data = catalog(scenario)
        basic_flight = next(flight for flight in data["flights"] if flight["id"] == "F-basic")
        basic_hotel = next(hotel for hotel in data["hotels"] if hotel["id"] == "H-basic")
        stale_total = money(
            basic_flight["fare_inr_per_adult"] * scenario.adults
            + basic_hotel["nightly_local_per_room"]
            * scenario.nights
            * ((scenario.adults + 1) // 2)
            * data["fresh_rate"]
            * 0.85
            + data["visa_fee_inr_per_adult"] * scenario.adults
        )
        self.assertLess(stale_total, scenario.budget_inr)


class PromptRunTests(unittest.IsolatedAsyncioTestCase):
    async def test_hyderabad_london_old_fx_fix_verifies_and_preserves_prompt_id(self):
        prompt = DEMO_EXAMPLES[-1]
        scenario_id, seed = catalog_seed(prompt)
        scenario = parse_trip_prompt(prompt, scenario_id=scenario_id, seed=seed)
        self.assertLess(scenario.expected_total_inr, scenario.budget_inr)

        with tempfile.TemporaryDirectory() as directory:
            recorder = Recorder(Path(directory), mode="offline", llm_client=FixtureClient())
            try:
                task_id = "PROMPT-DEMO-HYD-LON"
                metadata_dir = Path(directory) / "prompt-scenarios"
                metadata_dir.mkdir()
                (metadata_dir / f"{task_id}.json").write_text(
                    json.dumps({"scenario": asdict(scenario), "stale_fx": True, "prompt": prompt}),
                    encoding="utf-8",
                )
                with recorder.run("tripcrew", task_id, seed, model="tripcrew-fixture-v1") as run:
                    await TripCrew(
                        scenario,
                        TravelAPI([scenario], stale_fx=True),
                        model="tripcrew-fixture-v1",
                        task_prompt=prompt,
                    )(run)
                self.assertEqual(run.outcome, "failed")
                self.assertIn("Incorrect INR total", run.checker_reason)

                adapter = TripCrewAdapter(
                    recorder,
                    argparse.Namespace(agent="tripcrew", stale_fx=False, stale_fx_seeds=()),
                )
                known_good_fx = adapter.oracle_fixes(run.run_id)["fx/tool#1"]
                result = await ReplayEngine(recorder).replay(
                    run.run_id,
                    adapter.factory(run.run_id),
                    edits=[Edit("fx/tool#1", "patch_tool_result", known_good_fx, known_good=True)],
                    samples=5,
                    control=True,
                )
                self.assertEqual(result.fix_pass_rate, 1.0)
                self.assertEqual(result.control_pass_rate, 0.0)
                self.assertEqual(result.verdict, "VERIFIED")
                self.assertTrue(
                    all(
                        "scenario_id does not match" not in (run.reason or "")
                        for run in result.edited
                    )
                )
            finally:
                recorder.close()

    async def test_offline_task_is_recorded_and_replay_metadata_survives_adapter_rebuild(self):
        scenario_id = "PROMPT-TEST123459"
        scenario = parse_trip_prompt(PROMPT, scenario_id=scenario_id, seed=77)
        with tempfile.TemporaryDirectory() as directory:
            recorder = Recorder(Path(directory), mode="offline", llm_client=FixtureClient())
            try:
                run = recorder.run(
                    "tripcrew", scenario_id, scenario.seed, model="tripcrew-fixture-v1"
                )
                with run:
                    await TripCrew(
                        scenario,
                        TravelAPI([scenario], stale_fx=True),
                        model="tripcrew-fixture-v1",
                        task_prompt=PROMPT,
                    )(run)

                self.assertEqual(run.outcome, "failed")
                self.assertEqual(
                    recorder.database.one(
                        "SELECT COUNT(*) AS n FROM steps WHERE run_id=?", (run.run_id,)
                    )["n"],
                    16,
                )
                metadata_dir = Path(directory) / "prompt-scenarios"
                metadata_dir.mkdir()
                (metadata_dir / f"{scenario_id}.json").write_text(
                    json.dumps({"scenario": asdict(scenario), "stale_fx": True, "prompt": PROMPT}),
                    encoding="utf-8",
                )
                adapter = TripCrewAdapter(
                    recorder,
                    argparse.Namespace(stale_fx=False, stale_fx_seeds=()),
                )
                restored = adapter._scenario(run.run_id)
                self.assertEqual(restored, scenario)
                self.assertTrue(adapter._stale_fx(run.run_id))
                batch = await ReplayEngine(recorder).replay(
                    run.run_id,
                    adapter.factory(run.run_id),
                    edits=[patch_tool_args("fx/tool#1", {"fresh": True}, known_good=True)],
                    samples=1,
                    control=True,
                )
                self.assertEqual(batch.edited[0].outcome, "passed")
                self.assertEqual(batch.controls[0].outcome, "failed")
            finally:
                recorder.close()


if __name__ == "__main__":
    unittest.main()
