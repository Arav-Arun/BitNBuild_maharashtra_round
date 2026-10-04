"""Task 9: reasons, damage path, precedents, verifier, narrative, diff and regression export."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import socket
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from blackbox.diff import align, compare, leaves
from blackbox.explain.damage import damage_path
from blackbox.explain.narrative import template_narrative, validate
from blackbox.explain.reasons import TEMPLATES, describe, evidence_lines, walk
from blackbox.explain.verifier import first_self_divergence, record_verified, samples_needed
from blackbox.ml.features import FEATURE_NAMES, Reference

sys.path.insert(0, str(Path(__file__).parent))
import test_ml  # noqa: E402  (a module import, so unittest does not re-run its classes)

_toy_trace = test_ml._toy_trace


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.reference = Reference.fit([_toy_trace(f"h{i}", rate=64.0 + i * 0.1) for i in range(6)])

    def test_walk_yields_rfc6901_pointers_and_profile_paths(self):
        value = {"a/b": [{"x": 1}, {"x": 2}], "t~": "s"}
        self.assertEqual(
            list(walk(value)),
            [("/a~1b/0/x", "/a/b/*/x", 1), ("/a~1b/1/x", "/a/b/*/x", 2), ("/t~0", "/t~", "s")],
        )
        self.assertEqual(leaves(value), {"/a~1b/0/x": 1, "/a~1b/1/x": 2, "/t~0": "s"})

    def test_wrong_value_and_stale_date_are_cited_by_pointer(self):
        trace = _toy_trace(rate=640.0, as_of="2026-03-03")
        step = trace.steps[1]
        lines = evidence_lines(step, self.reference.profiles[step.addr])
        by_kind = {line["kind"]: line for line in lines}
        self.assertEqual(by_kind["numeric_outlier"]["json_pointer"], "/rate")
        self.assertEqual(by_kind["stale_date"]["json_pointer"], "/as_of")
        self.assertIn("7 months older", by_kind["stale_date"]["text"])
        self.assertTrue(all(line["addr"] == "fx/tool#1" for line in lines))

    def test_error_payload_and_twin_difference(self):
        broken = _toy_trace(error=True)
        lines = evidence_lines(broken.steps[1], self.reference.profiles["fx/tool#1"])
        self.assertIn("error_payload", {line["kind"] for line in lines})
        twin = _toy_trace("twin")
        wrong = _toy_trace(rate=99.0)
        lines = evidence_lines(wrong.steps[1], None, twin.steps[1], "twin")
        self.assertEqual(
            [(line["kind"], line["json_pointer"]) for line in lines],
            [("differs_from_twin", "/rate")],
        )

    def test_every_feature_has_a_sentence(self):
        self.assertEqual(set(TEMPLATES), set(FEATURE_NAMES))
        self.assertEqual(describe("later_anomalies", 0.0), "No later step looks anomalous.")
        self.assertIn("1 earlier step also looks", describe("earlier_anomalies", 1.0))
        self.assertIn("unavailable", describe("anomaly", float("nan")))


class DamagePathTests(unittest.TestCase):
    def test_tags_follow_the_twin_and_reach_the_final_step(self):
        broken = _toy_trace(rate=640.0)
        broken.steps[3].output = {"text": "ok"}  # the writer recovers: unaffected
        twin = _toy_trace("twin")
        path = damage_path(broken, "fx/tool#1", twin=twin)
        tags = {node["addr"]: node["tag"] for node in path["nodes"]}
        self.assertEqual(
            tags, {"fx/tool#1": "root", "budget/tool#1": "symptom", "writer/tool#1": "unaffected"}
        )
        self.assertEqual(path["path_to_final"], ["fx/tool#1", "budget/tool#1", "writer/tool#1"])
        self.assertFalse(path["suspect_matches_twin"])
        upstream = damage_path(broken, "a/tool#1", twin=twin)
        self.assertTrue(upstream["suspect_matches_twin"])

    def test_without_a_twin_tags_come_from_anomaly(self):
        trace = _toy_trace()
        path = damage_path(trace, "fx/tool#1", anomaly={"budget/tool#1": 2.0})
        self.assertEqual(path["basis"], "anomaly")
        self.assertEqual(path["symptoms"], 1)

    def test_first_self_divergence_skips_damaged_inputs(self):
        twin = _toy_trace("twin")
        broken = _toy_trace(rate=640.0)
        # budget's input differs too, but only fx diverged on identical input.
        broken.steps[2].input = {"args": {"rate": 640.0}}
        twin.steps[2].input = {"args": {"rate": 64.0}}
        self.assertEqual(first_self_divergence(broken, twin), "fx/tool#1")


class NarrativeTests(unittest.TestCase):
    def _bundle(self):
        return {
            "abstain": False,
            "conformal_set": ["fx/tool#1"],
            "conformal_size": 1,
            "coverage_target": 0.9,
            "suspects": [
                {
                    "addr": "fx/tool#1",
                    "probability": 0.82,
                    "reasons": [{"text": "Its data is 7 months older than in passing runs."}],
                    "evidence": [
                        {"addr": "fx/tool#1", "json_pointer": "/as_of", "text": "/as_of is old"}
                    ],
                    "damage_path": {
                        "nodes": [{"addr": "fx/tool#1", "tag": "root"}],
                        "reached": 0,
                        "symptoms": 0,
                    },
                    "precedents": None,
                }
            ],
            "verification": [
                {
                    "addr": "fx/tool#1",
                    "samples": 5,
                    "fix_rate": 1.0,
                    "control_rate": 0.0,
                    "verdict": "VERIFIED",
                }
            ],
        }

    def test_template_is_grounded_and_cites_real_steps(self):
        bundle = self._bundle()
        narrative = template_narrative(bundle)
        self.assertEqual(validate(narrative, bundle, {"fx/tool#1"}), [])
        self.assertTrue(all(claim["cites"] for claim in narrative["claims"]))

    def test_validator_rejects_unknown_steps_and_invented_numbers(self):
        bundle = self._bundle()
        bad = {"claims": [{"text": "Rate was off by 37%.", "cites": ["ghost/tool#9"]}]}
        problems = validate(bad, bundle, {"fx/tool#1"})
        self.assertTrue(any("unknown step" in p for p in problems))
        self.assertTrue(any("37" in p for p in problems))
        self.assertEqual(validate({"claims": []}, bundle, set()), ["no claims"])


class VerifierMathTests(unittest.TestCase):
    def test_samples_needed(self):
        self.assertIsNone(samples_needed(0.2, 0.4))
        needed = samples_needed(0.6, 0.2)
        self.assertIsNotNone(needed)
        self.assertGreater(needed, 5)
        self.assertEqual(samples_needed(1.0, 0.0), 4)


class AlignTests(unittest.TestCase):
    def test_lcs_alignment_when_control_flow_changes(self):
        from blackbox.diff import _Step

        def step(addr, seq, kind="tool", name=None, args="h"):
            return _Step(addr, seq, kind, name or addr, None, None, None, None, "live", args)

        left = [step("a", 0), step("b", 1), step("c", 2)]
        right = [step("a", 0), step("x", 1), step("b", 2, args="other"), step("c", 3)]
        pairs = [(lt and lt.addr, rt and rt.addr) for lt, rt in align(left, right)]
        self.assertEqual(pairs, [("a", "a"), ("b", "b"), (None, "x"), ("c", "c")])


class DemoForkDiffTests(unittest.TestCase):
    """The pitch demo: stale FX fails; re-fetching fresh FX at fx/tool#1 flips it."""

    def test_demo_fork_diverges_first_at_fx_and_changes_only_the_cone(self):
        from agents.tripcrew import TravelAPI, TripCrew, generate_scenarios
        from agents.tripcrew.fixture_client import FixtureClient
        from blackbox.replay import ReplayEngine, patch_tool_args
        from blackbox.sdk import Recorder

        async def record_and_fork(recorder):
            for scenario in generate_scenarios(10):
                agent = TripCrew(
                    scenario, TravelAPI([scenario], stale_fx=True), model="tripcrew-fixture-v1"
                )
                with recorder.run(
                    "tripcrew", scenario.scenario_id, 7, model="tripcrew-fixture-v1"
                ) as run:
                    await agent(run)
                if run.outcome == "failed":
                    batch = await ReplayEngine(recorder).replay(
                        run.run_id,
                        agent,
                        edits=[patch_tool_args("fx/tool#1", {"fresh": True}, known_good=True)],
                    )
                    return run.run_id, batch
            self.fail("no stale-FX scenario failed")

        with tempfile.TemporaryDirectory() as directory:
            recorder = Recorder(Path(directory), mode="offline", llm_client=FixtureClient())
            try:
                run_id, batch = asyncio.run(record_and_fork(recorder))
                diff = compare(recorder.database, recorder.store, run_id, batch.edited[0].run_id)
            finally:
                recorder.close()
        self.assertEqual(diff["first_divergence"], "fx/tool#1")
        self.assertEqual(diff["outcome"]["left"]["outcome"], "failed")
        self.assertEqual(diff["outcome"]["right"]["outcome"], "passed")
        self.assertTrue(diff["outcome"]["flipped"])
        changed = {row["addr"] for row in diff["rows"] if row["status"] == "changed"}
        self.assertIn("fx/tool#1", changed)
        self.assertTrue(changed <= batch.invalidated, changed - batch.invalidated)
        statuses = {row["addr"]: row["status"] for row in diff["rows"]}
        self.assertEqual(statuses["planner/chat#1"], "cached")
        fx = next(row for row in diff["rows"] if row["addr"] == "fx/tool#1")
        self.assertIn("/as_of", {change["pointer"] for change in fx["changes"]})
        self.assertTrue(any(entry["status"] == "changed" for entry in diff["state_diff"]))


class ExplainEndToEndTests(unittest.TestCase):
    """Fixture TripCrew corpus → trained diagnoser → explain, verify, diff and export."""

    @classmethod
    def setUpClass(cls):
        from blackbox.config import Settings
        from blackbox.eval.splits import make_splits
        from blackbox.explain import Explainer
        from blackbox.explain.precedents import Precedents
        from blackbox.explain.verifier import Verifier
        from blackbox.forge.__main__ import RoutingClient, make_adapter
        from blackbox.ml.dataset import load_corpus
        from blackbox.ml.evaluate import Experiment
        from blackbox.ml.model import DiagnoserConfig
        from blackbox.sdk import Recorder

        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.data = cls.root / "tripcrew"
        asyncio.run(test_ml.EndToEndTests._generate(cls.data))
        cls.corpus = load_corpus([cls.data])
        splits = make_splits(cls.corpus)
        experiment = Experiment(
            cls.corpus,
            splits,
            DiagnoserConfig(n_estimators=80, calibration_folds=2, early_stopping_rounds=10),
        )
        cls.diagnoser = experiment.train()
        cls.precedents = Precedents.fit(
            experiment.matrix(splits.train), [cls.corpus.labels[r] for r in splits.train]
        )
        settings = Settings.load()
        cls.client = RoutingClient(settings)
        cls.recorder = Recorder(cls.data, mode="offline", settings=settings, llm_client=cls.client)
        adapter = make_adapter(cls.recorder, argparse.Namespace(agent="tripcrew", stale_fx=False))
        cls.verifier = Verifier(
            cls.recorder, adapter.factory, oracle=adapter.oracle_fixes, samples=5
        )
        cls.explainer = Explainer(
            cls.diagnoser, cls.recorder, precedents=cls.precedents, verifier=cls.verifier
        )
        cls.runs = sorted(splits.s0 + splits.val)[:6]
        cls.bundles = {
            run_id: asyncio.run(cls.explainer.investigate(run_id, record_labels=False))
            for run_id in cls.runs
        }

    @classmethod
    def tearDownClass(cls):
        asyncio.run(cls.client.aclose())
        cls.recorder.close()
        cls.directory.cleanup()

    def test_bundle_has_every_part_and_matches_the_contract(self):
        from blackbox.explain import contract

        for run_id, bundle in self.bundles.items():
            with self.subTest(run_id=run_id):
                trace_addrs = set(self.corpus.traces[run_id].addrs)
                self.assertEqual(len(bundle["suspects"]), 3)
                for suspect in bundle["suspects"]:
                    self.assertLessEqual(len(suspect["reasons"]), 3)
                    for reason in suspect["reasons"]:
                        self.assertIn(reason["feature"], FEATURE_NAMES)
                        for cite in reason["cites"]:
                            self.assertIn(cite["addr"], trace_addrs)
                    self.assertEqual(suspect["damage_path"]["nodes"][0]["tag"], "root")
                    self.assertEqual(suspect["precedents"]["k"], 12)
                self.assertTrue(bundle["twin_same_task"])
                body = contract(bundle)
                self.assertEqual(body["run_id"], run_id)
                self.assertTrue(body["suspects"])
                self.assertTrue(all(suspect["addr"] in trace_addrs for suspect in body["suspects"]))

    def test_narrative_claims_cite_existing_steps(self):
        for run_id, bundle in self.bundles.items():
            with self.subTest(run_id=run_id):
                addrs = set(self.corpus.traces[run_id].addrs)
                claims = bundle["narrative"]["claims"]
                self.assertTrue(claims)
                for claim in claims:
                    self.assertTrue(set(claim["cites"]) <= addrs, claim)
                self.assertEqual(bundle["narrative"]["problems"], [])

    def test_verifier_confirms_roots_and_never_a_wrong_step(self):
        verified_roots = 0
        roots_tested = 0
        for run_id, bundle in self.bundles.items():
            root = self.corpus.labels[run_id].root_addr
            for check in bundle["verification"]:
                with self.subTest(run_id=run_id, addr=check["addr"]):
                    if check["addr"] != root:
                        self.assertNotEqual(check["verdict"], "VERIFIED")
                    else:
                        roots_tested += 1
                        verified_roots += check["verdict"] == "VERIFIED"
                    if check["masked_by"]:
                        self.assertEqual(check["samples"], 0)
        self.assertGreater(roots_tested, 0)
        self.assertGreaterEqual(verified_roots / roots_tested, 0.6)

    def test_verified_label_feeds_the_retraining_loop(self):
        run_id = next(r for r in self.corpus.passing)
        self.assertTrue(record_verified(self.recorder.database, run_id, "fx/tool#1", "stale_data"))
        row = self.recorder.database.one("SELECT * FROM labels WHERE run_id = ?", (run_id,))
        self.assertEqual((row["source"], row["verified"]), ("verified", 1))
        injected = self.runs[0]
        wrong = next(a for a in self.corpus.traces[injected].addrs if a != _root(self, injected))
        self.assertFalse(record_verified(self.recorder.database, injected, wrong))

    def test_diff_against_itself_and_against_the_verified_fork(self):
        run_id = self.runs[0]
        same = compare(self.recorder.database, self.recorder.store, run_id, run_id)
        self.assertEqual({row["status"] for row in same["rows"]}, {"same"})
        self.assertIsNone(same["first_divergence"])

        check = _verified(self)
        fixed = compare(
            self.recorder.database, self.recorder.store, check["run_id"], check["fix_run_id"]
        )
        self.assertEqual(fixed["first_divergence"], check["addr"])
        self.assertTrue(fixed["outcome"]["flipped"])
        changed = {row["addr"] for row in fixed["rows"] if row["status"] == "changed"}
        self.assertTrue(changed <= set(check["cone"]), changed - set(check["cone"]))
        self.assertIn(check["addr"], changed)

    def test_regression_export_runs_offline_and_fails_without_its_fix(self):
        from blackbox.export.regression import ExportError, export, no_network, replay_fixture

        check = _verified(self)
        with self.assertRaises(ExportError):
            refuted = next(
                c
                for b in self.bundles.values()
                for c in b["verification"]
                if c.get("fork_id") and c["verdict"] != "VERIFIED"
            )
            export(self.recorder, refuted["fork_id"], self.root / "tests")
        result = export(self.recorder, check["fork_id"], self.root / "tests")
        self.assertTrue(result.test_path.exists())
        with no_network(), self.assertRaises(OSError):
            socket.create_connection(("192.0.2.1", 80), timeout=1)

        spec = importlib.util.spec_from_file_location("exported", result.test_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        outcome = unittest.TextTestRunner(verbosity=0).run(
            unittest.defaultTestLoader.loadTestsFromModule(module)
        )
        self.assertTrue(outcome.wasSuccessful(), outcome.failures + outcome.errors)

        without_fix = replay_fixture(result.fixture_dir, [])
        self.assertEqual(without_fix["original_outcome"], "failed")
        self.assertEqual(without_fix["patched_outcome"], "failed")

    def test_precedents_round_trip(self):
        from blackbox.explain.precedents import Precedents

        with tempfile.TemporaryDirectory() as directory:
            self.precedents.save(Path(directory))
            restored = Precedents.load(Path(directory))
        row = self.diagnoser.matrix([self.corpus.traces[self.runs[0]]]).X[0]
        self.assertEqual(self.precedents.query(row)["neighbors"], restored.query(row)["neighbors"])
        self.assertTrue(np.allclose(self.precedents.X, restored.X, atol=1e-4))


def _root(case: ExplainEndToEndTests, run_id: str) -> str:
    return case.corpus.labels[run_id].root_addr


def _verified(case: ExplainEndToEndTests) -> dict:
    for run_id, bundle in case.bundles.items():
        for check in bundle["verification"]:
            if check["verdict"] == "VERIFIED":
                return {**check, "run_id": run_id}
    case.skipTest("no VERIFIED fork in this corpus sample")


if __name__ == "__main__":
    unittest.main()
