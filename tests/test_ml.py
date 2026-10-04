"""Tasks 7–8: features, splits, baselines, ranker, calibration and evaluation."""

from __future__ import annotations

import asyncio
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from blackbox.eval.baselines import UNTRAINED, first_error
from blackbox.eval.metrics import auroc, bootstrap_ci, paired_difference, per_run, summarize
from blackbox.eval.splits import check_splits, make_splits, task_bucket
from blackbox.ml.dataset import RootLabel, Trace, TraceStep, graded_relevance, load_corpus
from blackbox.ml.features import (
    EDGE_VIEWS,
    FEATURE_NAMES,
    OBSERVABILITY_VIEWS,
    FeatureBuilder,
    Reference,
    assert_no_leakage,
    build_matrix,
    materialize,
)


def _step(addr, seq, kind="tool", output=None, input=None, writes=(), reads=()):
    return TraceStep(
        addr=addr,
        seq=seq,
        kind=kind,
        name=addr.split("/")[1].split("#")[0],
        role=addr.split("/")[0],
        input=input,
        output=output,
        reads=tuple(reads),
        writes=tuple(writes),
        finish_reason=None,
        error_type=None,
        retries=0,
    )


def _toy_trace(run_id="run", rate=64.0, as_of="2026-10-03", error=False):
    fx = {"status": 500, "error": "boom"} if error else {"as_of": as_of, "rate": rate}
    return Trace(
        run_id=run_id,
        agent="toy",
        task_id=f"task-{run_id}",
        outcome="failed",
        steps=[
            _step("a/tool#1", 0, output={"n": 2}, input={"args": {"q": "x"}}, writes=["n@v1"]),
            _step("fx/tool#1", 1, output=fx, input={"args": {"n": 2}}, writes=["fx@v1"]),
            _step("budget/tool#1", 2, output={"total": rate * 2}, writes=["total@v1"]),
            _step("writer/tool#1", 3, output={"text": "ok"}, writes=["text@v1"]),
        ],
        edges=[
            ("a/tool#1", "fx/tool#1", "state"),
            ("fx/tool#1", "budget/tool#1", "state"),
            ("budget/tool#1", "writer/tool#1", "state"),
        ],
    )


def _label(trace, root="fx/tool#1", distractor=None, held_out=False):
    return RootLabel(
        run_id=trace.run_id,
        root_addr=root,
        fault_type="wrong_value",
        source="injected",
        held_out=held_out,
        distractor_addr=distractor,
        base_run_id=None,
    )


class LeakageTests(unittest.TestCase):
    def test_feature_matrix_has_no_label_or_replay_columns(self):
        assert_no_leakage(FEATURE_NAMES)
        for forbidden in ("run_id", "fault_type", "root_addr", "cache_status", "latency_ms"):
            with self.subTest(forbidden=forbidden):
                with self.assertRaises(AssertionError):
                    assert_no_leakage([*FEATURE_NAMES, forbidden])

    def test_builder_only_accepts_traces(self):
        builder = FeatureBuilder(Reference())
        with self.assertRaises(TypeError):
            builder.trace_rows(_label(_toy_trace()))

    def test_replay_bookkeeping_is_not_loaded(self):
        self.assertFalse(hasattr(TraceStep, "cache_status"))
        self.assertFalse(hasattr(TraceStep, "latency_ms"))


class FeatureTests(unittest.TestCase):
    def setUp(self):
        healthy = [_toy_trace(f"h{i}", rate=64.0 + i * 0.1) for i in range(6)]
        self.reference = Reference.fit(healthy)

    def _row(self, trace, addr):
        rows = FeatureBuilder(self.reference).trace_rows(trace)
        return rows[[s.addr for s in trace.steps].index(addr)]

    def test_wrong_value_is_a_numeric_outlier(self):
        row = self._row(_toy_trace(rate=640.0), "fx/tool#1")
        self.assertGreater(row["num_max_absz"], 6)

    def test_stale_date_is_older_than_healthy_runs(self):
        row = self._row(_toy_trace(as_of="2026-03-03"), "fx/tool#1")
        self.assertGreater(row["date_days_older"], 200)
        self.assertGreater(row["date_older_z"], 6)

    def test_error_payload_and_first_anomaly(self):
        trace = _toy_trace(error=True)
        rows = FeatureBuilder(self.reference).trace_rows(trace)
        self.assertEqual(rows[1]["status_error"], 1.0)
        self.assertEqual(rows[1]["is_first_anomaly"], 1.0)
        self.assertEqual(rows[0]["next_anomaly"], rows[1]["anomaly"])

    def test_lineage_counts_descendants(self):
        rows = FeatureBuilder(self.reference).trace_rows(_toy_trace())
        self.assertEqual(rows[0]["n_descendants"], 3)
        self.assertEqual(rows[1]["reaches_last"], 1.0)
        self.assertEqual(rows[3]["n_descendants"], 0)

    def test_reference_round_trips_through_json(self):
        restored = Reference.from_dict(json.loads(json.dumps(self.reference.to_dict())))
        trace = _toy_trace(rate=640.0)
        a = FeatureBuilder(self.reference).trace_rows(trace)
        b = FeatureBuilder(restored).trace_rows(trace)
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))

    def test_views_share_runs_and_rows_and_strip_what_they_claim(self):
        traces = [_toy_trace("x", rate=640.0), _toy_trace("y", error=True)]
        shapes = set()
        for view in OBSERVABILITY_VIEWS:
            for edges in EDGE_VIEWS:
                matrix = build_matrix(traces, self.reference, view=view, edges=edges)
                shapes.add((tuple(matrix.run_ids), matrix.X.shape))
        self.assertEqual(len(shapes), 1)
        otel = materialize(traces[0], "otel")
        self.assertTrue(all(s.input is None and not s.writes for s in otel.steps))
        self.assertTrue(all(kind == "protocol" for _, _, kind in otel.edges))
        outputs_only = materialize(traces[0], "outputs_only")
        self.assertEqual(outputs_only.edges, [])
        self.assertEqual(outputs_only.steps[1].output, traces[0].steps[1].output)

    def test_disabled_groups_are_blank(self):
        matrix = build_matrix([_toy_trace()], self.reference, groups=("A",))
        for name in ("anomaly", "n_descendants", "status_error"):
            self.assertTrue(np.isnan(matrix.X[:, FEATURE_NAMES.index(name)]).all())
        self.assertFalse(np.isnan(matrix.X[:, FEATURE_NAMES.index("rel_pos")]).any())


class RelevanceTests(unittest.TestCase):
    def test_graded_by_causal_hops_with_distractor_zeroed(self):
        trace = _toy_trace()
        grades = graded_relevance(trace, _label(trace))
        self.assertEqual(
            grades, {"a/tool#1": 2, "fx/tool#1": 3, "budget/tool#1": 2, "writer/tool#1": 1}
        )
        grades = graded_relevance(trace, _label(trace, distractor="a/tool#1"))
        self.assertEqual(grades["a/tool#1"], 0)


class MetricTests(unittest.TestCase):
    def _corpus(self):
        from blackbox.ml.dataset import Corpus

        traces = {f"r{i}": _toy_trace(f"r{i}") for i in range(4)}
        labels = {r: _label(t, distractor="writer/tool#1") for r, t in traces.items()}
        return Corpus(traces=traces, labels=labels, passing=[], recovered=[])

    def test_per_run_metrics(self):
        corpus = self._corpus()
        rankings = {
            "r0": ["fx/tool#1", "a/tool#1", "budget/tool#1", "writer/tool#1"],
            "r1": ["budget/tool#1", "fx/tool#1", "a/tool#1", "writer/tool#1"],
            "r2": ["writer/tool#1", "a/tool#1", "budget/tool#1", "fx/tool#1"],
        }
        rows = per_run(rankings, corpus)
        self.assertEqual(rows["r0"]["top1"], 1.0)
        self.assertEqual(rows["r1"]["mrr"], 0.5)
        self.assertEqual(rows["r1"]["within1"], 1.0)
        self.assertEqual(rows["r2"]["top3"], 0.0)
        self.assertEqual(rows["r2"]["distractor_top1"], 1.0)
        summary = summarize(rows)
        self.assertEqual(summary["n"], 3)
        self.assertAlmostEqual(summary["top1"]["value"], 1 / 3)

    def test_bootstrap_and_paired_difference(self):
        low, high = bootstrap_ci([1.0] * 10)
        self.assertEqual((low, high), (1.0, 1.0))
        low, high = bootstrap_ci([0, 1] * 50)
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)
        a = {f"r{i}": {"top1": 0.0} for i in range(10)}
        b = {f"r{i}": {"top1": 1.0} for i in range(10)}
        diff = paired_difference(a, b)
        self.assertEqual(diff["difference"], 1.0)
        self.assertEqual(diff["n"], 10)

    def test_auroc(self):
        self.assertEqual(auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertEqual(auroc([0, 1], [0.5, 0.5]), 0.5)
        self.assertTrue(math.isnan(auroc([1, 1], [0.1, 0.2])))

    def test_first_error_baseline_blames_the_error_step(self):
        healthy = Reference.fit([_toy_trace(f"h{i}") for i in range(4)])
        matrix = build_matrix([_toy_trace("e", error=True)], healthy)
        self.assertEqual(first_error(matrix)["e"][0], "fx/tool#1")
        for name, baseline in UNTRAINED.items():
            with self.subTest(baseline=name):
                ranking = baseline(matrix)["e"]
                self.assertEqual(sorted(ranking), sorted(matrix.addrs[0]))


class SplitTests(unittest.TestCase):
    def test_buckets_are_stable(self):
        self.assertEqual(task_bucket("TC-7-0001"), task_bucket("TC-7-0001"))
        buckets = {task_bucket(f"T{i}") for i in range(200)}
        self.assertEqual(buckets, {"train", "val", "test"})


class EndToEndTests(unittest.TestCase):
    """Fixture TripCrew → forge → corpus → splits → ranker → evaluation JSON."""

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.data = Path(cls.directory.name) / "tripcrew"
        asyncio.run(cls._generate(cls.data))
        cls.corpus = load_corpus([cls.data])

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    @staticmethod
    async def _generate(directory):
        import argparse

        from agents.tripcrew import TravelAPI, TripCrew, generate_scenarios
        from agents.tripcrew.fixture_client import FixtureClient
        from blackbox.config import Settings
        from blackbox.forge.__main__ import RoutingClient, make_adapter
        from blackbox.forge.inject import FaultInjector
        from blackbox.forge.operators import (
            T1WrongValue,
            T2StaleData,
            T3Empty404,
            T4Timeout500,
            T5SchemaDrift,
        )
        from blackbox.forge.runner import ForgeRunner
        from blackbox.sdk import Recorder

        scenarios = generate_scenarios(30)
        recorder = Recorder(directory, mode="offline", llm_client=FixtureClient())
        api = TravelAPI(scenarios)
        try:
            for scenario in scenarios:
                with recorder.run(
                    "tripcrew", scenario.scenario_id, 7, model="tripcrew-fixture-v1"
                ) as run:
                    await TripCrew(scenario, api, model="tripcrew-fixture-v1")(run)
        finally:
            recorder.close()
        settings = Settings.load()
        recorder = Recorder(
            directory, mode="offline", settings=settings, llm_client=RoutingClient(settings)
        )
        try:
            adapter = make_adapter(recorder, argparse.Namespace(agent="tripcrew", stale_fx=False))
            runner = ForgeRunner(
                FaultInjector(recorder, adapter.factory), concurrency=8, distractor_rate=0.3
            )
            await runner.run(
                target_positive=80,
                max_attempts=150,
                operators=[
                    T1WrongValue(),
                    T3Empty404(),
                    T4Timeout500(),
                    T2StaleData(),
                    T5SchemaDrift(),
                ],
                samples=1,
                control=False,
            )
        finally:
            recorder.close()

    def test_one_example_per_fork_and_labels_separate(self):
        self.assertGreater(len(self.corpus.labels), 50)
        self.assertEqual(len(self.corpus.passing), 30)
        for run_id in self.corpus.labels:
            self.assertTrue(run_id.endswith("-fix-0"))
            self.assertIn(self.corpus.labels[run_id].root_addr, self.corpus.traces[run_id].addrs)

    def test_splits_are_grouped_and_hold_out_unseen_faults(self):
        splits = make_splits(self.corpus)
        check_splits(self.corpus, splits)
        expected_s1 = sorted(
            r
            for r, label in self.corpus.labels.items()
            if label.held_out and task_bucket(self.corpus.traces[r].task_id) != "train"
        )
        self.assertEqual(sorted(splits.s1), expected_s1)
        self.assertTrue(any(label.held_out for label in self.corpus.labels.values()))
        reference_tasks = {self.corpus.traces[r].task_id for r in splits.reference}
        self.assertTrue(all(task_bucket(t) == "train" for t in reference_tasks))

    def test_ranker_trains_calibrates_and_round_trips(self):
        from blackbox.eval.splits import make_splits
        from blackbox.ml.evaluate import Experiment
        from blackbox.ml.model import Diagnoser

        experiment = Experiment(self.corpus, make_splits(self.corpus))
        diagnoser = experiment.train()
        traces = [self.corpus.traces[r] for r in experiment.splits.s0[:5]]
        first = diagnoser.diagnose_many(traces)
        for diagnosis in first:
            self.assertAlmostEqual(sum(p for _, p in diagnosis.ranking), 1.0, places=6)
            self.assertTrue(diagnosis.conformal_set)
            self.assertEqual(diagnosis.abstain, len(diagnosis.conformal_set) > 3)
        with tempfile.TemporaryDirectory() as directory:
            diagnoser.save(Path(directory) / "m", dataset_hash="abc")
            restored = Diagnoser.load(Path(directory) / "m")
        second = restored.diagnose_many(traces)
        for a, b in zip(first, second):
            self.assertEqual([x for x, _ in a.ranking], [x for x, _ in b.ranking])
            np.testing.assert_allclose([p for _, p in a.ranking], [p for _, p in b.ranking])
        self.assertGreater(diagnoser.metadata["n_calibration_runs"], len(experiment.splits.val))

    def test_evaluation_writes_contract_json(self):
        from blackbox.ml.evaluate import run_evaluation
        from blackbox.ml.model import DiagnoserConfig
        from server.models import EvalResponse

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "eval"
            fast = DiagnoserConfig(n_estimators=60, calibration_folds=2, early_stopping_rounds=10)
            summary = run_evaluation([self.data], out, Path(directory) / "model", base=fast)
            EvalResponse.model_validate(
                {k: summary[k] for k in ("sample_size", "metrics", "ablations")}
            )
            for name in ("leaderboard", "ablations", "calibration", "integrity"):
                self.assertTrue((out / f"{name}.json").exists())
            leaderboard = json.loads((out / "leaderboard.json").read_text())
            self.assertIn("first_error", leaderboard)
            self.assertIn("S0", leaderboard["blackbox"]["splits"])
            self.assertEqual(
                set(json.loads((out / "ablations.json").read_text())["edges"]),
                {"full", "protocol", "none"},
            )
            integrity = json.loads((out / "integrity.json").read_text())
            shuffled = integrity["shuffled_labels"]
            self.assertLess(shuffled["top1"]["value"], summary["metrics"]["s0_top1"])
            self.assertFalse(integrity["single_feature_oracle"]["leak_suspected"])


if __name__ == "__main__":
    unittest.main()
