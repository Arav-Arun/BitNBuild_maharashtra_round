import unittest

from blackbox.explain.evidence import leaves, step_findings
from blackbox.explain.report import damage_path, is_evaluation_step, visible_failure
from blackbox.ml.dataset import Trace, TraceStep
from blackbox.ml.features import Reference


def tool(addr: str, seq: int, args: dict, result: object, error: str | None = None) -> TraceStep:
    return TraceStep(
        addr=addr,
        seq=seq,
        kind="tool",
        name=addr.split("/", 1)[0],
        role=addr.split("/", 1)[0],
        input={"args": args},
        output=result,
        reads=(),
        writes=(),
        finish_reason=None,
        error_type=error,
        retries=0,
    )


def trace() -> Trace:
    steps = [
        tool("fx/tool#1", 1, {"pair": "SGD/INR"}, {"rate": 64.1, "as_of": "2026-12-01"}),
        tool("budget/tool#1", 2, {"rate": 64.1}, {"total_inr": 120000}),
        tool("writer/tool#1", 3, {"total_inr": 120000}, {"plan": "ok"}),
        tool("weather/tool#1", 4, {"city": "Singapore"}, {"rain": True}),
        tool("checker/tool#1", 5, {"plan": "ok"}, {"passed": False}),
    ]
    edges = [
        ("fx/tool#1", "budget/tool#1", "state"),
        ("budget/tool#1", "writer/tool#1", "state"),
        ("writer/tool#1", "checker/tool#1", "state"),
    ]
    return Trace("run-1", "tripcrew", "TC-1", "fail", steps, edges)


class ExplainTests(unittest.TestCase):
    def test_leaves_use_json_pointers_and_profile_paths(self):
        found = list(leaves({"a/b": [1, {"c": 2}]}, "/output"))
        self.assertEqual(
            found,
            [("/output/a~1b/0", "/a/b/*", 1), ("/output/a~1b/1/c", "/a/b/*/c", 2)],
        )

    def test_evaluation_steps_are_never_suspects(self):
        self.assertTrue(is_evaluation_step("checker/tool#1"))
        self.assertTrue(is_evaluation_step("x/tool#1", "judge"))
        self.assertFalse(is_evaluation_step("fx/tool#1"))

    def test_damage_path_tags_root_symptoms_and_unaffected(self):
        run = trace()
        failure = visible_failure(run)
        self.assertEqual(failure, "weather/tool#1")
        tags = {row["addr"]: row["tag"] for row in damage_path(run, "fx/tool#1", "writer/tool#1")}
        self.assertEqual(tags["fx/tool#1"], "root")
        self.assertEqual(tags["budget/tool#1"], "symptom")
        self.assertEqual(tags["writer/tool#1"], "symptom")
        self.assertEqual(tags["weather/tool#1"], "unaffected")
        self.assertNotIn("checker/tool#1", tags)

    def test_findings_cite_the_conflicting_field(self):
        run = trace()
        run.steps[1] = tool("budget/tool#1", 2, {"rate": 64.1}, {"rate": 6.41, "total_inr": 1})
        findings = step_findings(run, run.steps[1], Reference.fit([]))
        conflict = next(f for f in findings if f.code == "conflict_input")
        self.assertEqual(conflict.citation.pointer, "/output/rate")
        self.assertEqual(conflict.related[0].pointer, "/input/rate")


if __name__ == "__main__":
    unittest.main()
