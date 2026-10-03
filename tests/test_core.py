import tempfile
import unittest
from pathlib import Path

from blackbox.eval import localization_metrics
from blackbox.recorder import Store


class CoreTests(unittest.TestCase):
    def test_checkpoint_round_trip_is_content_addressed(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            state = {"messages": [{"role": "user", "content": "hi"}], "memory": {"k": 1}}
            ref = store.save_checkpoint(state)
            self.assertEqual(store.load_checkpoint(ref), state)
            self.assertEqual(store.save_checkpoint({"memory": {"k": 1}, **state}), ref)

    def test_integrity_and_ref_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            ref = store.save_checkpoint({"value": 1})
            (Path(directory) / "checkpoints" / f"{ref}.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "integrity"):
                store.load_checkpoint(ref)
            with self.assertRaises(ValueError):
                store.load_checkpoint("../escape")

    def test_metrics(self):
        result = localization_metrics([[0, 1, 2], [2, 0, 1]], [0, 1])
        self.assertEqual(result["recall@1"], 0.5)
        self.assertEqual(result["recall@3"], 1.0)
        self.assertAlmostEqual(result["mrr"], 2 / 3)
        with self.assertRaises(ValueError):
            localization_metrics([[1, 1]], [1])


if __name__ == "__main__":
    unittest.main()
