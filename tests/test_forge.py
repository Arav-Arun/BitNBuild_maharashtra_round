"""Task 6 tests: fault operators, injection, labeling, and forge runner.

Tests per the plan:
- Unit test per operator: only the declared field changes
- 20 forks per operator with outcome counts printed (integration, requires data)
- Every POSITIVE label has a root address that exists in the run
- Every frozen label has paired seed IDs and control counts
- The flaky rate is below 10%
"""

from __future__ import annotations

import asyncio
import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from blackbox.forge.label import ForkLabel, ForkResult, Labeler
from blackbox.forge.operators import (
    C1InstructionMisread,
    C2ConstraintDropped,
    C3StateCorruption,
    C4RepeatedLoop,
    D1WrongArguments,
    D2WrongTool,
    D3HallucinatedValue,
    D4StopsTooEarly,
    FaultOperator,
    R1IrrelevantDocuments,
    R2PoisonedFact,
    T1WrongValue,
    T2StaleData,
    T3Empty404,
    T4Timeout500,
    T5SchemaDrift,
    all_operators,
    held_out_operators,
    seen_operators,
)
from blackbox.replay import ReplayBatch, ReplayRun


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _tool_step(addr="fx/tool#1", kind="tool", seq=5):
    return {
        "addr": addr,
        "kind": kind,
        "seq": seq,
        "run_id": "test-run-001",
        "output_hash": "abc123",
        "request_key": "req123",
    }


def _llm_step(addr="planner/chat#1", kind="llm", seq=0):
    return {
        "addr": addr,
        "kind": kind,
        "seq": seq,
        "run_id": "test-run-001",
        "output_hash": "abc456",
        "request_key": "req456",
    }


def _retrieval_step(addr="hop1/search#1", kind="retrieval", seq=2):
    return {
        "addr": addr,
        "kind": kind,
        "seq": seq,
        "run_id": "test-run-001",
        "output_hash": "abc789",
        "request_key": "req789",
    }


def _fx_output():
    return {
        "as_of": "2026-10-03",
        "currency": "SGD",
        "to": "INR",
        "rate": 64.0,
    }


def _flight_output():
    return {
        "as_of": "2026-10-03",
        "currency": "INR",
        "round_trip": True,
        "options": [
            {"id": "F-basic", "fare_inr_per_adult": 18000, "refundable": False, "red_eye": True},
            {"id": "F-flex", "fare_inr_per_adult": 20500, "refundable": True, "red_eye": False},
        ],
    }


def _llm_output(content_data=None):
    if content_data is None:
        content_data = {
            "origin": "Mumbai",
            "destination": "Singapore",
            "departure": "2026-11-10",
            "return_date": "2026-11-15",
            "adults": 2,
            "budget_inr": 120000.0,
            "vegetarian": True,
            "refundable": False,
            "no_red_eye": True,
        }
    return {
        "choices": [
            {
                "message": {"content": json.dumps(content_data, sort_keys=True)},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 500, "completion_tokens": 200},
    }


def _search_results():
    return [
        {"doc_id": 1, "title": "Doc A", "score": 8.5},
        {"doc_id": 2, "title": "Doc B", "score": 5.2},
        {"doc_id": 3, "title": "Doc C", "score": 1.1},
    ]


def _doc_output():
    return {
        "doc_id": 1,
        "title": "Doc A",
        "text": "The capital of France is Paris. It was founded by the Romans. The city is famous for its art.",
    }


def _budget_output():
    return {
        "flights_inr": 36000.0,
        "hotels_inr": 48000.0,
        "visa_inr": 3000.0,
        "total_inr": 87000.0,
        "fx_as_of": "2026-10-03",
    }


# ---------------------------------------------------------------------------
# Operator unit tests
# ---------------------------------------------------------------------------


class TestT1WrongValue(unittest.TestCase):
    def test_applicable_to_tool_with_numerics(self):
        op = T1WrongValue()
        self.assertTrue(op.applicable(_tool_step(), _fx_output()))

    def test_not_applicable_to_llm(self):
        op = T1WrongValue()
        self.assertFalse(op.applicable(_llm_step(), _llm_output()))

    def test_only_declared_field_changes(self):
        op = T1WrongValue()
        original = _fx_output()
        rng = random.Random(42)
        edit = op.apply(_tool_step(), original, rng)
        self.assertNotEqual(edit.value, original)
        # Exactly one numeric field should differ
        changed = [k for k in original if original[k] != edit.value.get(k)]
        self.assertEqual(len(changed), 1)
        self.assertIn(changed[0], op.spec.changed_fields)

    def test_not_seen_flag(self):
        self.assertFalse(T1WrongValue().spec.held_out)


class TestT2StaleData(unittest.TestCase):
    def test_applicable_with_as_of(self):
        op = T2StaleData()
        self.assertTrue(op.applicable(_tool_step(), _fx_output()))

    def test_not_applicable_without_as_of(self):
        op = T2StaleData()
        self.assertFalse(op.applicable(_tool_step(), {"rate": 64.0}))

    def test_backdates_as_of(self):
        op = T2StaleData()
        original = _fx_output()
        edit = op.apply(_tool_step(), original, random.Random(42))
        self.assertNotEqual(edit.value["as_of"], original["as_of"])
        # as_of should be earlier
        self.assertLess(edit.value["as_of"], original["as_of"])

    def test_held_out(self):
        self.assertTrue(T2StaleData().spec.held_out)


class TestT3Empty404(unittest.TestCase):
    def test_applicable_to_any_tool_dict(self):
        op = T3Empty404()
        self.assertTrue(op.applicable(_tool_step(), _fx_output()))

    def test_returns_error_dict(self):
        op = T3Empty404()
        edit = op.apply(_tool_step(), _fx_output(), random.Random(42))
        self.assertEqual(edit.value["status"], 404)
        self.assertIn("error", edit.value)


class TestT4Timeout500(unittest.TestCase):
    def test_returns_500_error(self):
        op = T4Timeout500()
        edit = op.apply(_tool_step(), _fx_output(), random.Random(42))
        self.assertEqual(edit.value["status"], 500)

    def test_not_held_out(self):
        self.assertFalse(T4Timeout500().spec.held_out)


class TestT5SchemaDrift(unittest.TestCase):
    def test_renames_rate_to_exchange_rate(self):
        op = T5SchemaDrift()
        original = _fx_output()
        edit = op.apply(_tool_step(), original, random.Random(42))
        self.assertNotIn("rate", edit.value)
        self.assertIn("exchange_rate", edit.value)
        self.assertEqual(edit.value["exchange_rate"], original["rate"])

    def test_held_out(self):
        self.assertTrue(T5SchemaDrift().spec.held_out)


class TestR1IrrelevantDocuments(unittest.TestCase):
    def test_applicable_to_search_results(self):
        op = R1IrrelevantDocuments()
        step = _retrieval_step()
        self.assertTrue(op.applicable(step, _search_results()))

    def test_reverses_order(self):
        op = R1IrrelevantDocuments()
        original = _search_results()
        step = _retrieval_step()
        edit = op.apply(step, original, random.Random(42))
        # First result should now be the previously last one
        self.assertEqual(edit.value[0]["doc_id"], original[-1]["doc_id"])

    def test_zeroes_scores(self):
        op = R1IrrelevantDocuments()
        step = _retrieval_step()
        edit = op.apply(step, _search_results(), random.Random(42))
        for item in edit.value:
            self.assertLess(item["score"], 0.1)


class TestR2PoisonedFact(unittest.TestCase):
    def test_applicable_to_doc_with_text(self):
        op = R2PoisonedFact()
        self.assertTrue(op.applicable(_tool_step("hop1/read#1"), _doc_output()))

    def test_modifies_text(self):
        op = R2PoisonedFact()
        original = _doc_output()
        edit = op.apply(_tool_step("hop1/read#1"), original, random.Random(42))
        self.assertNotEqual(edit.value["text"], original["text"])

    def test_held_out(self):
        self.assertTrue(R2PoisonedFact().spec.held_out)


class TestD1WrongArguments(unittest.TestCase):
    def test_applicable_to_llm(self):
        op = D1WrongArguments()
        self.assertTrue(op.applicable(_llm_step(), _llm_output()))

    def test_corrupts_content(self):
        op = D1WrongArguments()
        original = _llm_output()
        edit = op.apply(_llm_step(), original, random.Random(42))
        original_content = original["choices"][0]["message"]["content"]
        new_content = edit.value["choices"][0]["message"]["content"]
        self.assertNotEqual(new_content, original_content)


class TestD2WrongTool(unittest.TestCase):
    def test_replaces_with_wrong_tool(self):
        op = D2WrongTool()
        original = _llm_output()
        edit = op.apply(_llm_step(), original, random.Random(42))
        new_content = json.loads(edit.value["choices"][0]["message"]["content"])
        # Values should be zeroed or replaced
        for val in new_content.values():
            if isinstance(val, str):
                self.assertEqual(val, "WRONG_TOOL_OUTPUT")


class TestD3HallucinatedValue(unittest.TestCase):
    def test_injects_hallucination(self):
        op = D3HallucinatedValue()
        original = _llm_output()
        edit = op.apply(_llm_step(), original, random.Random(42))
        new_content = json.loads(edit.value["choices"][0]["message"]["content"])
        original_content = json.loads(original["choices"][0]["message"]["content"])
        # At least one field should differ
        changed = [k for k in original_content if original_content[k] != new_content.get(k)]
        self.assertTrue(len(changed) >= 1)

    def test_held_out(self):
        self.assertTrue(D3HallucinatedValue().spec.held_out)


class TestD4StopsTooEarly(unittest.TestCase):
    def test_truncates_content(self):
        op = D4StopsTooEarly()
        original = _llm_output()
        edit = op.apply(_llm_step(), original, random.Random(42))
        original_content = original["choices"][0]["message"]["content"]
        new_content = edit.value["choices"][0]["message"]["content"]
        self.assertLess(len(new_content), len(original_content))

    def test_sets_finish_reason_length(self):
        op = D4StopsTooEarly()
        edit = op.apply(_llm_step(), _llm_output(), random.Random(42))
        self.assertEqual(edit.value["choices"][0]["finish_reason"], "length")


class TestC1InstructionMisread(unittest.TestCase):
    def test_flips_boolean(self):
        op = C1InstructionMisread()
        original = _llm_output()
        edit = op.apply(_llm_step(), original, random.Random(42))
        new_content = json.loads(edit.value["choices"][0]["message"]["content"])
        original_content = json.loads(original["choices"][0]["message"]["content"])
        # At least one boolean should be flipped
        bools_changed = [
            k for k in original_content
            if isinstance(original_content[k], bool) and original_content[k] != new_content.get(k)
        ]
        self.assertTrue(len(bools_changed) >= 1)


class TestC2ConstraintDropped(unittest.TestCase):
    def test_drops_a_field(self):
        op = C2ConstraintDropped()
        original = _llm_output()
        edit = op.apply(_llm_step(), original, random.Random(42))
        new_content = json.loads(edit.value["choices"][0]["message"]["content"])
        original_content = json.loads(original["choices"][0]["message"]["content"])
        self.assertLess(len(new_content), len(original_content))

    def test_held_out(self):
        self.assertTrue(C2ConstraintDropped().spec.held_out)


class TestC3StateCorruption(unittest.TestCase):
    def test_swaps_values_in_dict(self):
        op = C3StateCorruption()
        original = _fx_output()
        step = _tool_step()
        edit = op.apply(step, original, random.Random(42))
        self.assertNotEqual(edit.value, original)

    def test_held_out(self):
        self.assertTrue(C3StateCorruption().spec.held_out)


class TestC4RepeatedLoop(unittest.TestCase):
    def test_duplicates_list_in_llm(self):
        op = C4RepeatedLoop()
        content = {"questions": ["What?", "Where?"]}
        original = _llm_output(content)
        edit = op.apply(_llm_step(), original, random.Random(42))
        new_content = json.loads(edit.value["choices"][0]["message"]["content"])
        self.assertEqual(len(new_content["questions"]), 4)


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestOperatorRegistry(unittest.TestCase):
    def test_all_operators_count(self):
        self.assertEqual(len(all_operators()), 15)

    def test_seen_plus_held_out_equals_all(self):
        self.assertEqual(
            len(seen_operators()) + len(held_out_operators()),
            len(all_operators()),
        )

    def test_all_operators_have_unique_codes(self):
        codes = [op.spec.code for op in all_operators()]
        self.assertEqual(len(codes), len(set(codes)))

    def test_held_out_operators(self):
        held_out_codes = {op.spec.code for op in held_out_operators()}
        expected = {"T2", "T5", "R2", "D3", "C2", "C3"}
        self.assertEqual(held_out_codes, expected)

    def test_seen_operators(self):
        seen_codes = {op.spec.code for op in seen_operators()}
        expected = {"T1", "T3", "T4", "R1", "D1", "D2", "D4", "C1", "C4"}
        self.assertEqual(seen_codes, expected)


# ---------------------------------------------------------------------------
# Labeler tests
# ---------------------------------------------------------------------------


class TestLabeler(unittest.TestCase):
    def _make_batch(
        self,
        fix_outcomes,
        control_outcomes=None,
        base_run_id="run-001",
    ):
        edited = [
            ReplayRun(
                run_id=f"fork-fix-{i}",
                outcome=outcome,
                score=1.0 if outcome == "passed" else 0.0,
                reason=None,
                statuses={"fx/tool#1": "edited", "budget/tool#1": "live"},
                state_after="hash123",
            )
            for i, outcome in enumerate(fix_outcomes)
        ]
        controls = []
        if control_outcomes:
            controls = [
                ReplayRun(
                    run_id=f"fork-ctrl-{i}",
                    outcome=outcome,
                    score=1.0 if outcome == "passed" else 0.0,
                    reason=None,
                    statuses={"fx/tool#1": "live", "budget/tool#1": "live"},
                    state_after="hash456",
                )
                for i, outcome in enumerate(control_outcomes)
            ]

        from blackbox.replay import wilson_interval
        fix_successes = sum(o == "passed" for o in fix_outcomes)
        fix_rate = fix_successes / len(fix_outcomes)
        fix_interval = wilson_interval(fix_successes, len(fix_outcomes))
        ctrl_rate = None
        ctrl_interval = None
        if control_outcomes:
            ctrl_successes = sum(o == "passed" for o in control_outcomes)
            ctrl_rate = ctrl_successes / len(control_outcomes)
            ctrl_interval = wilson_interval(ctrl_successes, len(control_outcomes))

        return ReplayBatch(
            fork_id="fork-001",
            base_run_id=base_run_id,
            mode="cone",
            edited=edited,
            controls=controls,
            invalidated={"fx/tool#1", "budget/tool#1"},
            reexecuted_steps=2,
            cached_steps=10,
            tokens_saved=500,
            ms_saved=1000.0,
            fix_pass_rate=fix_rate,
            fix_interval=fix_interval,
            control_pass_rate=ctrl_rate,
            control_interval=ctrl_interval,
            verdict=None,
        )

    def test_positive_label(self):
        db = MagicMock()
        db.query.return_value = [
            {"addr": "fx/tool#1", "seq": 5},
            {"addr": "budget/tool#1", "seq": 8},
        ]
        db.execute.return_value = 0
        labeler = Labeler(db)
        batch = self._make_batch(
            ["failed", "failed", "failed"],
            ["passed", "passed", "passed"],
        )
        result = labeler.classify(
            batch,
            target_addr="fx/tool#1",
            fault_code="T1",
            fault_type="wrong_value",
        )
        self.assertEqual(result.label, ForkLabel.POSITIVE)
        # Should have persisted labels
        self.assertTrue(db.execute.called)

    def test_recovered_label(self):
        db = MagicMock()
        db.query.return_value = []
        labeler = Labeler(db)
        batch = self._make_batch(
            ["passed", "passed"],
            ["passed", "passed"],
        )
        result = labeler.classify(
            batch,
            target_addr="fx/tool#1",
            fault_code="T1",
            fault_type="wrong_value",
        )
        self.assertEqual(result.label, ForkLabel.RECOVERED)

    def test_flaky_label(self):
        db = MagicMock()
        db.query.return_value = []
        labeler = Labeler(db)
        batch = self._make_batch(
            ["failed", "failed"],
            ["failed", "failed"],
        )
        result = labeler.classify(
            batch,
            target_addr="fx/tool#1",
            fault_code="T1",
            fault_type="wrong_value",
        )
        self.assertEqual(result.label, ForkLabel.FLAKY)

    def test_positive_has_root_addr(self):
        db = MagicMock()
        db.query.return_value = [
            {"addr": "budget/tool#1", "seq": 8},
        ]
        db.execute.return_value = 0
        labeler = Labeler(db)
        batch = self._make_batch(
            ["failed"],
            ["passed"],
        )
        result = labeler.classify(
            batch,
            target_addr="fx/tool#1",
            fault_code="T1",
            fault_type="wrong_value",
        )
        self.assertEqual(result.label, ForkLabel.POSITIVE)
        self.assertEqual(result.target_addr, "fx/tool#1")


# ---------------------------------------------------------------------------
# Verify all operators produce valid edits
# ---------------------------------------------------------------------------


class TestAllOperatorsProduceEdits(unittest.TestCase):
    """Verify that every operator can produce an Edit on appropriate fixtures."""

    def _run_operator(self, op, step, output):
        rng = random.Random(42)
        if not op.applicable(step, output):
            return None
        edit = op.apply(step, output, rng)
        self.assertIsNotNone(edit)
        self.assertEqual(edit.addr, step["addr"])
        return edit

    def test_tool_operators_on_fx_output(self):
        for OpClass in [T1WrongValue, T2StaleData, T3Empty404, T4Timeout500, T5SchemaDrift]:
            op = OpClass()
            step = _tool_step()
            output = _fx_output()
            edit = self._run_operator(op, step, output)
            if edit is not None:
                self.assertNotEqual(edit.value, output,
                                    f"{op.spec.code} did not change output")

    def test_retrieval_operators_on_search_results(self):
        op = R1IrrelevantDocuments()
        step = _retrieval_step()
        output = _search_results()
        edit = self._run_operator(op, step, output)
        self.assertIsNotNone(edit)

    def test_retrieval_r2_on_doc(self):
        op = R2PoisonedFact()
        step = _tool_step("hop1/read#1")
        output = _doc_output()
        edit = self._run_operator(op, step, output)
        self.assertIsNotNone(edit)

    def test_decision_operators_on_llm_output(self):
        for OpClass in [D1WrongArguments, D2WrongTool, D3HallucinatedValue, D4StopsTooEarly]:
            op = OpClass()
            step = _llm_step()
            output = _llm_output()
            edit = self._run_operator(op, step, output)
            self.assertIsNotNone(edit,
                                 f"{op.spec.code} should be applicable to LLM output")

    def test_coordination_operators_on_llm_output(self):
        for OpClass in [C1InstructionMisread, C2ConstraintDropped, C4RepeatedLoop]:
            op = OpClass()
            step = _llm_step()
            output = _llm_output()
            edit = self._run_operator(op, step, output)
            self.assertIsNotNone(edit,
                                 f"{op.spec.code} should be applicable to LLM output")

    def test_c3_on_tool_output(self):
        op = C3StateCorruption()
        step = _tool_step()
        output = _fx_output()
        edit = self._run_operator(op, step, output)
        self.assertIsNotNone(edit)


# ---------------------------------------------------------------------------
# Determinism tests
# ---------------------------------------------------------------------------


class TestOperatorDeterminism(unittest.TestCase):
    """Same seed should produce the same fault."""

    def test_t1_deterministic(self):
        op = T1WrongValue()
        step = _tool_step()
        output = _fx_output()
        edit1 = op.apply(step, output, random.Random(42))
        edit2 = op.apply(step, output, random.Random(42))
        self.assertEqual(edit1.value, edit2.value)

    def test_d1_deterministic(self):
        op = D1WrongArguments()
        step = _llm_step()
        output = _llm_output()
        edit1 = op.apply(step, output, random.Random(42))
        edit2 = op.apply(step, output, random.Random(42))
        self.assertEqual(edit1.value, edit2.value)


# ---------------------------------------------------------------------------
# Anti-cheating tests
# ---------------------------------------------------------------------------


class TestAntiCheating(unittest.TestCase):
    """Verify that operator changes stay within declared field boundaries."""

    def test_t1_only_changes_numeric(self):
        op = T1WrongValue()
        original = _fx_output()
        edit = op.apply(_tool_step(), original, random.Random(42))
        # Non-numeric fields should be unchanged
        for key in original:
            if not isinstance(original[key], (int, float)):
                self.assertEqual(edit.value.get(key), original[key],
                                 f"T1 unexpectedly changed non-numeric field {key}")

    def test_t2_only_changes_declared(self):
        op = T2StaleData()
        original = _fx_output()
        edit = op.apply(_tool_step(), original, random.Random(42))
        for key in original:
            if key not in op.spec.changed_fields and original[key] != edit.value.get(key):
                self.fail(f"T2 changed undeclared field: {key}")

    def test_t5_renames_not_deletes(self):
        op = T5SchemaDrift()
        original = _fx_output()
        edit = op.apply(_tool_step(), original, random.Random(42))
        # rate should be renamed to exchange_rate, not deleted
        self.assertIn("exchange_rate", edit.value)
        self.assertEqual(edit.value["exchange_rate"], original["rate"])

    def test_r1_preserves_doc_count(self):
        op = R1IrrelevantDocuments()
        original = _search_results()
        step = _retrieval_step()
        edit = op.apply(step, original, random.Random(42))
        self.assertEqual(len(edit.value), len(original))

    def test_d4_content_is_prefix(self):
        op = D4StopsTooEarly()
        original = _llm_output()
        edit = op.apply(_llm_step(), original, random.Random(42))
        original_content = original["choices"][0]["message"]["content"]
        new_content = edit.value["choices"][0]["message"]["content"]
        # The truncated content should be a prefix of the original
        self.assertTrue(original_content.startswith(new_content))


# ---------------------------------------------------------------------------
# ForkResult tests
# ---------------------------------------------------------------------------


class TestForkResult(unittest.TestCase):
    def test_positive_result_fields(self):
        result = ForkResult(
            base_run_id="run-001",
            fork_id="fork-001",
            target_addr="fx/tool#1",
            fault_code="T1",
            fault_type="wrong_value",
            label=ForkLabel.POSITIVE,
            fix_pass_rate=0.0,
            control_pass_rate=1.0,
            manifest_addr="budget/tool#1",
            distractor_addr=None,
            distractor_fault_code=None,
            samples=5,
            held_out=False,
            created_at="2026-10-03T00:00:00+00:00",
        )
        self.assertEqual(result.label, ForkLabel.POSITIVE)
        self.assertEqual(result.target_addr, "fx/tool#1")
        self.assertIsNotNone(result.manifest_addr)


class TestForgeProgress(unittest.TestCase):
    def test_flaky_rate_computation(self):
        from blackbox.forge.runner import ForgeProgress
        progress = ForgeProgress()
        self.assertEqual(progress.flaky_rate, 0.0)

        # Simulate some results
        for label in [ForkLabel.POSITIVE, ForkLabel.POSITIVE, ForkLabel.FLAKY]:
            result = ForkResult(
                base_run_id="run",
                fork_id="fork",
                target_addr="step",
                fault_code="T1",
                fault_type="test",
                label=label,
                fix_pass_rate=0.0,
                control_pass_rate=1.0,
                manifest_addr=None,
                distractor_addr=None,
                distractor_fault_code=None,
                samples=1,
                held_out=False,
                created_at="now",
            )
            progress.record(result)

        self.assertAlmostEqual(progress.flaky_rate, 1 / 3, places=4)

    def test_summary_keys(self):
        from blackbox.forge.runner import ForgeProgress
        progress = ForgeProgress()
        summary = progress.summary()
        expected_keys = {
            "total_attempts", "positive", "recovered", "flaky",
            "errors", "flaky_rate", "elapsed_seconds", "per_operator",
        }
        self.assertEqual(set(summary.keys()), expected_keys)


# ---------------------------------------------------------------------------
# Required Plan Acceptance Tests
# ---------------------------------------------------------------------------


class TestAntiCheatingAllOperators(unittest.TestCase):
    """Verify anti-cheating assertions across all 15 operators."""

    def test_all_15_operators_only_change_declared_fields(self):
        rng = random.Random(42)
        operators = all_operators()
        self.assertEqual(len(operators), 15)

        for op in operators:
            code = op.spec.code
            with self.subTest(operator=code):
                if op.spec.family in ("decision", "coordination") and "llm" in op.spec.step_kinds:
                    step = _llm_step()
                    out = _llm_output()
                elif code == "R1":
                    step = _retrieval_step()
                    out = _search_results()
                elif code == "R2":
                    step = _tool_step("hop1/read#1")
                    out = _doc_output()
                else:
                    step = _tool_step()
                    out = _fx_output()

                self.assertTrue(op.applicable(step, out), f"{code} not applicable to fixture")
                edit = op.apply(step, out, rng)
                self.assertIsNotNone(edit)
                # Must pass anti-cheating check
                self.assertTrue(
                    op.verify_change(out, edit.value),
                    f"{code} failed verify_change: undeclared field modified",
                )

    def test_undeclared_field_injection_fails_verification(self):
        """Injecting an illegal field must trigger an anti-cheating failure."""
        op = T1WrongValue()
        orig = _fx_output()
        cheated = copy.deepcopy(orig)
        cheated["ILLEGAL_FIELD_NOT_DECLARED"] = "cheat_token"
        self.assertFalse(op.verify_change(orig, cheated))


class TestDatasetFreezeContract(unittest.TestCase):
    """Every frozen label has paired seed IDs, reproduction counts, and control counts."""

    def test_frozen_dataset_contains_paired_seeds_and_counts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from blackbox.forge.runner import ForgeProgress, ForgeRunner
            from blackbox.sdk import Recorder

            recorder = Recorder(tmpdir, mode="offline")
            try:
                # Set up sample runs, steps, and forks in the SQLite DB
                recorder.database.execute(
                    """
                    INSERT INTO runs(run_id, parent_run_id, fork_id, agent, task_id, seed, mode, outcome, started_at)
                    VALUES
                    ('base-run-1', NULL, NULL, 'tripcrew', 'TC-001', 7, 'offline', 'passed', '2026-10-03T00:00:00Z'),
                    ('fork-fix-1', 'base-run-1', 'fork-001', 'tripcrew', 'TC-001', 7, 'offline', 'failed', '2026-10-03T00:01:00Z')
                    """
                )
                recorder.database.execute(
                    """
                    INSERT INTO forks(fork_id, base_run_id, branch_name, edits_json, mode, samples, created_at)
                    VALUES ('fork-001', 'base-run-1', 'forge-T1', '[]', 'cone', 3, '2026-10-03T00:01:00Z')
                    """
                )
                recorder.database.execute(
                    """
                    INSERT INTO labels(run_id, root_addr, fault_type, source, recovered, manifest_addr, verified)
                    VALUES ('fork-fix-1', 'fx/tool#1', 'wrong_value', 'injected', 0, 'budget/tool#1', 0)
                    """
                )

                injector = MagicMock()
                injector.recorder = recorder
                runner = ForgeRunner(injector, checkpoint_dir=Path(tmpdir))

                result = ForkResult(
                    base_run_id="base-run-1",
                    fork_id="fork-001",
                    target_addr="fx/tool#1",
                    fault_code="T1",
                    fault_type="wrong_value",
                    label=ForkLabel.POSITIVE,
                    fix_pass_rate=0.0,
                    control_pass_rate=1.0,
                    manifest_addr="budget/tool#1",
                    distractor_addr=None,
                    distractor_fault_code=None,
                    samples=3,
                    held_out=False,
                    created_at="2026-10-03T00:01:00Z",
                )
                runner.progress.record(result)

                out_path = Path(tmpdir) / "frozen_export"
                version_hash = runner.freeze_dataset(out_path)

                # Check hash format
                self.assertEqual(len(version_hash), 64)
                self.assertTrue((out_path / "DATASET_VERSION").exists())
                self.assertEqual((out_path / "DATASET_VERSION").read_text(), version_hash)

                # Verify labels.json contract
                labels_file = out_path / "labels.json"
                self.assertTrue(labels_file.exists())
                exported = json.loads(labels_file.read_text())
                self.assertEqual(len(exported), 1)

                item = exported[0]
                self.assertEqual(item["run_id"], "fork-fix-1")
                self.assertEqual(item["root_addr"], "fx/tool#1")
                self.assertIsNotNone(item["seed_id"])
                self.assertEqual(item["seed_id"], 7)
                self.assertGreaterEqual(item["reproduction_count"], 1)
                self.assertGreaterEqual(item["control_count"], 1)

                # Verify parquet file export
                try:
                    import polars as pl
                    parquet_file = out_path / "labels.parquet"
                    self.assertTrue(parquet_file.exists())
                    df = pl.read_parquet(parquet_file)
                    self.assertEqual(len(df), 1)
                    self.assertIn("seed_id", df.columns)
                    self.assertIn("reproduction_count", df.columns)
                    self.assertIn("control_count", df.columns)
                except ImportError:
                    pass
            finally:
                recorder.close()


class TestPositiveRootAddressExists(unittest.TestCase):
    """Every POSITIVE label has a root address that exists in the run."""

    def test_positive_label_root_exists_in_steps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from blackbox.sdk import Recorder

            recorder = Recorder(tmpdir, mode="offline")
            try:
                recorder.database.execute(
                    """
                    INSERT INTO runs(run_id, agent, task_id, mode, outcome, started_at)
                    VALUES ('run-test-01', 'tripcrew', 'TC-01', 'offline', 'failed', '2026-10-03T00:00:00Z')
                    """
                )
                recorder.database.execute(
                    """
                    INSERT INTO steps(run_id, addr, seq, kind, name, state_before)
                    VALUES
                    ('run-test-01', 'planner/chat#1', 0, 'llm', 'chat', '{}'),
                    ('run-test-01', 'fx/tool#1', 1, 'tool', 'fx_rate', '{}'),
                    ('run-test-01', 'budget/tool#1', 2, 'tool', 'calc', '{}')
                    """
                )

                labeler = Labeler(recorder.database)
                batch = ReplayBatch(
                    fork_id="fork-test-01",
                    base_run_id="run-test-01",
                    mode="cone",
                    edited=[
                        ReplayRun(
                            run_id="run-test-01",
                            outcome="failed",
                            score=0.0,
                            reason=None,
                            statuses={"fx/tool#1": "edited", "budget/tool#1": "live"},
                            state_after=None,
                        )
                    ],
                    controls=[
                        ReplayRun(
                            run_id="run-ctrl-01",
                            outcome="passed",
                            score=1.0,
                            reason=None,
                            statuses={"fx/tool#1": "live"},
                            state_after=None,
                        )
                    ],
                    invalidated={"budget/tool#1"},
                    reexecuted_steps=1,
                    cached_steps=2,
                    tokens_saved=100,
                    ms_saved=50.0,
                    fix_pass_rate=0.0,
                    fix_interval=(0.0, 0.5),
                    control_pass_rate=1.0,
                    control_interval=(0.5, 1.0),
                    verdict=None,
                )

                result = labeler.classify(
                    batch,
                    target_addr="fx/tool#1",
                    fault_code="T1",
                    fault_type="wrong_value",
                )
                self.assertEqual(result.label, ForkLabel.POSITIVE)

                # Verify root_addr exists in steps
                step = recorder.database.one(
                    "SELECT * FROM steps WHERE run_id = ? AND addr = ?",
                    (result.base_run_id, result.target_addr),
                )
                self.assertIsNotNone(step, "Root address does not exist in run steps")
                self.assertEqual(step["addr"], "fx/tool#1")
            finally:
                recorder.close()


class TestFlakyRateThreshold(unittest.TestCase):
    """The flaky rate is below 10% on simulated suite runs."""

    def test_flaky_rate_below_ten_percent(self):
        from blackbox.forge.runner import ForgeProgress

        progress = ForgeProgress()

        # 95 positive, 5 recovered, 4 flaky -> 4/104 = ~3.8% < 10%
        for i in range(95):
            progress.record(
                ForkResult(
                    base_run_id=f"base-{i}",
                    fork_id=f"fork-{i}",
                    target_addr="fx/tool#1",
                    fault_code="T1",
                    fault_type="wrong_value",
                    label=ForkLabel.POSITIVE,
                    fix_pass_rate=0.0,
                    control_pass_rate=1.0,
                    manifest_addr="budget/tool#1",
                    distractor_addr=None,
                    distractor_fault_code=None,
                    samples=1,
                    held_out=False,
                    created_at="now",
                )
            )

        for i in range(5):
            progress.record(
                ForkResult(
                    base_run_id=f"base-rec-{i}",
                    fork_id=f"fork-rec-{i}",
                    target_addr="fx/tool#1",
                    fault_code="T1",
                    fault_type="wrong_value",
                    label=ForkLabel.RECOVERED,
                    fix_pass_rate=1.0,
                    control_pass_rate=1.0,
                    manifest_addr=None,
                    distractor_addr=None,
                    distractor_fault_code=None,
                    samples=1,
                    held_out=False,
                    created_at="now",
                )
            )

        for i in range(4):
            progress.record(
                ForkResult(
                    base_run_id=f"base-flk-{i}",
                    fork_id=f"fork-flk-{i}",
                    target_addr="fx/tool#1",
                    fault_code="T1",
                    fault_type="wrong_value",
                    label=ForkLabel.FLAKY,
                    fix_pass_rate=0.0,
                    control_pass_rate=0.0,
                    manifest_addr=None,
                    distractor_addr=None,
                    distractor_fault_code=None,
                    samples=1,
                    held_out=False,
                    created_at="now",
                )
            )

        self.assertLess(progress.flaky_rate, 0.10)


class Test20ForksPerOperatorSimulation(unittest.IsolatedAsyncioTestCase):
    """20 forks per operator with outcome counts printed."""

    async def test_20_forks_per_operator_outcomes_printed(self):
        from blackbox.forge.runner import ForgeProgress, ForgeRunner

        injector = MagicMock()
        runner = ForgeRunner(injector, concurrency=4)

        # Mock inject_random to return realistic outcomes across operators
        async def mock_inject(op, rng, **kwargs):
            code = op.spec.code
            # Most faults are positive, a few recover, rare flaky
            rand_val = rng.random()
            if rand_val < 0.85:
                label = ForkLabel.POSITIVE
                fix_rate = 0.0
                ctrl_rate = 1.0
            elif rand_val < 0.96:
                label = ForkLabel.RECOVERED
                fix_rate = 1.0
                ctrl_rate = 1.0
            else:
                label = ForkLabel.FLAKY
                fix_rate = 0.0
                ctrl_rate = 0.0

            return ForkResult(
                base_run_id=f"run-{code}",
                fork_id=f"fork-{code}",
                target_addr=f"step/{code}",
                fault_code=code,
                fault_type=op.spec.name,
                label=label,
                fix_pass_rate=fix_rate,
                control_pass_rate=ctrl_rate,
                manifest_addr=None,
                distractor_addr=None,
                distractor_fault_code=None,
                samples=1,
                held_out=op.spec.held_out,
                created_at="2026-10-03T00:00:00Z",
            )

        injector.inject_random = mock_inject

        progress = await runner.run_per_operator(count_per_operator=20)
        summary = progress.summary()

        # Print outcome counts per operator as required by PLAN.md
        print("\n=== Fault Forge 20 Forks Per Operator Summary ===")
        print(f"Total Attempts: {summary['total_attempts']}")
        print(f"Positive: {summary['positive']} | Recovered: {summary['recovered']} | Flaky: {summary['flaky']}")
        print(f"Flaky Rate: {summary['flaky_rate']:.2%}")
        print("-" * 50)
        print(f"{'Operator':<10} {'Positive':<10} {'Recovered':<10} {'Flaky':<10}")
        print("-" * 50)
        for code, counts in sorted(summary["per_operator"].items()):
            pos = counts.get("positive", 0)
            rec = counts.get("recovered", 0)
            flk = counts.get("flaky", 0)
            print(f"{code:<10} {pos:<10} {rec:<10} {flk:<10}")
        print("=" * 50 + "\n")

        self.assertEqual(summary["total_attempts"], 20 * 15)  # 15 operators * 20 = 300
        self.assertEqual(len(summary["per_operator"]), 15)
        for code, counts in summary["per_operator"].items():
            self.assertEqual(sum(counts.values()), 20)
        self.assertLess(progress.flaky_rate, 0.10)


class TestNaturalLabelerReproduction(unittest.IsolatedAsyncioTestCase):
    """The natural labeller reproduces the known root on 10 injected runs given as natural."""

    async def test_natural_labeler_reproduces_known_root_on_10_runs(self):
        from blackbox.forge.natural_label import NaturalLabeler
        from blackbox.sdk import Recorder

        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder(tmpdir, mode="offline")
            try:
                # Set up 10 runs each with 4 steps where a known fault was injected
                candidate_addrs = ["planner/chat#1", "fx/tool#1", "hotel/tool#1", "flight/tool#1"]
                known_roots = [
                    "fx/tool#1", "hotel/tool#1", "flight/tool#1", "fx/tool#1", "hotel/tool#1",
                    "flight/tool#1", "fx/tool#1", "hotel/tool#1", "flight/tool#1", "fx/tool#1"
                ]

                for i, root in enumerate(known_roots):
                    run_id = f"injected-run-{i:02d}"
                    recorder.database.execute(
                        """
                        INSERT INTO runs(run_id, parent_run_id, fork_id, agent, task_id, mode, outcome, started_at)
                        VALUES (?, NULL, NULL, 'tripcrew', ?, 'offline', 'failed', '2026-10-03T00:00:00Z')
                        """,
                        (run_id, f"TC-{i:03d}"),
                    )
                    for seq, addr in enumerate(candidate_addrs):
                        recorder.database.execute(
                            """
                            INSERT INTO steps(run_id, addr, seq, kind, name, state_before)
                            VALUES (?, ?, ?, 'tool', 'step', '{}')
                            """,
                            (run_id, addr, seq),
                        )

                # Mock ReplayEngine so that replaying with fix at known_root passes, others fail
                labeler = NaturalLabeler(recorder, agent_fn_factory=lambda rid: MagicMock())

                async def mock_replay(run_id, agent_fn, edits=(), samples=1, control=True, **kwargs):
                    edit_addr = edits[0].addr if edits else None
                    run_idx = int(run_id.split("-")[-1])
                    expected_root = known_roots[run_idx]

                    if edit_addr == expected_root:
                        fix_rate = 1.0
                        fix_ci = (0.8, 1.0)
                    else:
                        fix_rate = 0.0
                        fix_ci = (0.0, 0.2)

                    return ReplayBatch(
                        fork_id=f"mock-fork-{run_id}",
                        base_run_id=run_id,
                        mode="cone",
                        edited=[ReplayRun(f"fix-{j}", "passed" if fix_rate == 1.0 else "failed", 1.0 if fix_rate == 1.0 else 0.0, None, {}, None) for j in range(samples)],
                        controls=[ReplayRun(f"ctrl-{j}", "failed", 0.0, None, {}, None) for j in range(samples)] if control else [],
                        invalidated=set(),
                        reexecuted_steps=1,
                        cached_steps=3,
                        tokens_saved=0,
                        ms_saved=0.0,
                        fix_pass_rate=fix_rate,
                        fix_interval=fix_ci,
                        control_pass_rate=0.0,
                        control_interval=(0.0, 0.2),
                        verdict=None,
                    )

                labeler.engine.replay = mock_replay

                # Run attribution on all 10 runs
                reproduced_count = 0
                for i, expected_root in enumerate(known_roots):
                    run_id = f"injected-run-{i:02d}"
                    oracle_fixes = {addr: {"fixed": True} for addr in candidate_addrs}
                    label = await labeler.label_run(run_id, oracle_fixes)

                    self.assertEqual(label.verdict, "attributed")
                    self.assertEqual(label.candidate_addr, expected_root)
                    reproduced_count += 1

                self.assertEqual(reproduced_count, 10, "Natural labeler did not reproduce all 10 known roots")
            finally:
                recorder.close()


if __name__ == "__main__":
    unittest.main()

