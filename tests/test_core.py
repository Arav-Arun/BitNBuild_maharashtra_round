import json
import tempfile
import unittest
from pathlib import Path

from blackbox.agent.sample import record_sample
from blackbox.agent.tasks import TASKS
from blackbox.eval import localization_metrics
from blackbox.recorder import Store
from blackbox.replay import load_before
from blackbox.schema import Run


class CoreTests(unittest.TestCase):
    def test_record_restore_and_immutable_run(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            run = record_sample(store)
            self.assertEqual(len(store.load_run(run.run_id).steps), 8)
            initial, _ = load_before(store, run.run_id, 0)
            self.assertEqual(initial["messages"], [])
            state, step = load_before(store, run.run_id, 4)
            self.assertEqual(len(state["messages"]), 4)
            self.assertEqual(store.save_checkpoint(state), step.state_before_ref)
            with self.assertRaises(FileExistsError):
                store.save_run(run)
            with self.assertRaises(ValueError):
                load_before(store, run.run_id, -1)
            payload = run.to_dict()
            payload["steps"][2]["state_before_ref"] = "0" * 64
            with self.assertRaises(ValueError):
                Run.from_dict(payload)

    def test_integrity_and_path_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            ref = store.save_checkpoint({"value": 1})
            (Path(directory) / "checkpoints" / f"{ref}.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "integrity"):
                store.load_checkpoint(ref)
            with self.assertRaises(ValueError):
                store.load_run("../escape")

    def test_tasks_and_example(self):
        for task in TASKS:
            self.assertTrue(task.check(task.expected)[0])
            self.assertFalse(task.check({"total": -1})[0])
        example = Path(__file__).resolve().parents[1] / "examples" / "sample_trace.json"
        self.assertEqual(len(Run.from_dict(json.loads(example.read_text())).steps), 8)

    def test_metrics(self):
        result = localization_metrics([[0, 1, 2], [2, 0, 1]], [0, 1])
        self.assertEqual(result["recall@1"], 0.5)
        self.assertEqual(result["recall@3"], 1.0)
        self.assertAlmostEqual(result["mrr"], 2 / 3)
        with self.assertRaises(ValueError):
            localization_metrics([[1, 1]], [1])


if __name__ == "__main__":
    unittest.main()
