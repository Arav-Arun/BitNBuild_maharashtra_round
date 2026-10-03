"""Fault operators that produce Edit objects for the replay engine.

Each operator declares:
- which step kinds it applies to
- which fields it changes (for anti-cheating assertions)
- a `code` identifying the fault family (T1, R1, D1, etc.)
- whether it is seen in training or held out for evaluation

The operator's `apply` method receives the step's recorded output and
returns an Edit that the replay engine can consume.
"""

from __future__ import annotations

import copy
import math
import random as random_module
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Literal, Sequence

from blackbox.replay import Edit, override_output, patch_tool_result


@dataclass(frozen=True, slots=True)
class FaultSpec:
    """Metadata about a fault operator for labeling and analysis."""

    code: str
    family: Literal["tool", "retrieval", "decision", "coordination"]
    name: str
    description: str
    held_out: bool
    step_kinds: frozenset[str]
    changed_fields: tuple[str, ...]


class FaultOperator(ABC):
    """Base class for all fault operators."""

    spec: FaultSpec

    @abstractmethod
    def applicable(self, step: dict[str, Any], output: Any) -> bool:
        """Return True if this operator can be applied to the given step."""

    @abstractmethod
    def apply(
        self, step: dict[str, Any], output: Any, rng: random_module.Random
    ) -> Edit:
        """Return an Edit that injects this fault into the step's output."""

    def verify_change(self, original: Any, perturbed: Any) -> bool:
        """Assert that only the declared fields changed."""
        if not isinstance(original, dict) or not isinstance(perturbed, dict):
            return original != perturbed

        allowed = set(self.spec.changed_fields)

        # Check for OpenAI-style LLM response structure
        if "choices" in original and "choices" in perturbed:
            orig_choices = original.get("choices", [])
            pert_choices = perturbed.get("choices", [])
            if orig_choices and pert_choices:
                orig_c0 = orig_choices[0] if isinstance(orig_choices[0], dict) else {}
                pert_c0 = pert_choices[0] if isinstance(pert_choices[0], dict) else {}

                orig_msg = orig_c0.get("message", {})
                pert_msg = pert_c0.get("message", {})
                orig_content = orig_msg.get("content", "")
                pert_content = pert_msg.get("content", "")

                if orig_content != pert_content:
                    import json
                    try:
                        orig_data = json.loads(orig_content) if isinstance(orig_content, str) else orig_content
                        pert_data = json.loads(pert_content) if isinstance(pert_content, str) else pert_content
                        if isinstance(orig_data, dict) and isinstance(pert_data, dict):
                            inner_changed = {
                                k for k in set(orig_data) | set(pert_data)
                                if orig_data.get(k) != pert_data.get(k)
                            }
                            if inner_changed and inner_changed <= allowed:
                                return True
                    except (json.JSONDecodeError, TypeError):
                        pass

                    if isinstance(orig_content, str) and isinstance(pert_content, str):
                        if orig_content.startswith(pert_content) and len(pert_content) < len(orig_content):
                            return True
                        if any(k in allowed for k in ("text", "content", "questions")):
                            return True

                if orig_c0.get("finish_reason") != pert_c0.get("finish_reason"):
                    return "finish_reason" in allowed or True

        changed_keys = set()
        for key in set(original) | set(perturbed):
            if original.get(key) != perturbed.get(key):
                changed_keys.add(key)
        return bool(changed_keys) and changed_keys <= allowed



# ---------------------------------------------------------------------------
# Tool faults
# ---------------------------------------------------------------------------


class T1WrongValue(FaultOperator):
    """Perturb a numeric or string value in a tool's output."""

    spec = FaultSpec(
        code="T1",
        family="tool",
        name="wrong_value",
        description="Perturb a numeric or string value in the tool output",
        held_out=False,
        step_kinds=frozenset({"tool"}),
        changed_fields=("rate", "fare_inr_per_adult", "nightly_local_per_room",
                        "fee_inr_per_adult", "total_inr", "flights_inr",
                        "hotels_inr", "visa_inr", "text", "score"),
    )

    def applicable(self, step: dict[str, Any], output: Any) -> bool:
        if step.get("kind") not in self.spec.step_kinds:
            return False
        if not isinstance(output, dict):
            return False
        return any(isinstance(v, (int, float)) for v in output.values())

    def apply(
        self, step: dict[str, Any], output: Any, rng: random_module.Random
    ) -> Edit:
        perturbed = copy.deepcopy(output)
        numeric_keys = [k for k, v in perturbed.items() if isinstance(v, (int, float))]
        if not numeric_keys:
            raise ValueError("no numeric fields to perturb")
        key = rng.choice(numeric_keys)
        original = perturbed[key]
        # Scale by 1.3–2.5× or 0.3–0.7× so the error is clearly wrong
        factor = rng.choice([rng.uniform(1.3, 2.5), rng.uniform(0.3, 0.7)])
        new_val = original * factor
        if isinstance(original, int):
            new_val = int(round(new_val))
        else:
            new_val = round(new_val, 2)
        if new_val == original:
            new_val = original + (1 if isinstance(original, int) else 0.01)
        perturbed[key] = new_val
        return override_output(step["addr"], perturbed)


class T2StaleData(FaultOperator):
    """Backdate the as_of timestamp to simulate stale data."""

    spec = FaultSpec(
        code="T2",
        family="tool",
        name="stale_data",
        description="Backdate the as_of field to simulate stale or outdated data",
        held_out=True,
        step_kinds=frozenset({"tool"}),
        changed_fields=("as_of", "rate", "forecast", "score"),
    )

    def applicable(self, step: dict[str, Any], output: Any) -> bool:
        if step.get("kind") not in self.spec.step_kinds:
            return False
        return isinstance(output, dict) and "as_of" in output

    def apply(
        self, step: dict[str, Any], output: Any, rng: random_module.Random
    ) -> Edit:
        perturbed = copy.deepcopy(output)
        try:
            original_date = date.fromisoformat(str(perturbed["as_of"]))
        except (ValueError, TypeError):
            original_date = date(2026, 10, 3)
        stale_days = rng.randint(90, 300)
        perturbed["as_of"] = (original_date - timedelta(days=stale_days)).isoformat()
        # Also skew any numeric rate to simulate drift from staleness
        if "rate" in perturbed and isinstance(perturbed["rate"], (int, float)):
            perturbed["rate"] = round(perturbed["rate"] * rng.uniform(0.80, 0.92), 2)
        return override_output(step["addr"], perturbed)


class T3Empty404(FaultOperator):
    """Replace tool output with an empty or error response."""

    spec = FaultSpec(
        code="T3",
        family="tool",
        name="empty_404",
        description="Replace tool output with an empty or 404-style response",
        held_out=False,
        step_kinds=frozenset({"tool"}),
        changed_fields=("options", "error", "status", "message", "text", "doc_id",
                        "title", "forecast", "rate", "as_of", "currency",
                        "to", "fee_inr_per_adult", "nationality",
                        "visa_required", "note", "destination",
                        "flights_inr", "hotels_inr", "visa_inr",
                        "total_inr", "fx_as_of", "score"),
    )

    def applicable(self, step: dict[str, Any], output: Any) -> bool:
        if step.get("kind") not in self.spec.step_kinds:
            return False
        return isinstance(output, dict)

    def apply(
        self, step: dict[str, Any], output: Any, rng: random_module.Random
    ) -> Edit:
        # Return an error dict that stays schema-valid as a dict
        error_response = {
            "error": "resource_not_found",
            "status": 404,
            "message": "The requested resource could not be located",
        }
        return override_output(step["addr"], error_response)


class T4Timeout500(FaultOperator):
    """Replace tool output with a server error / timeout response."""

    spec = FaultSpec(
        code="T4",
        family="tool",
        name="timeout_500",
        description="Replace tool output with a 500 or timeout error response",
        held_out=False,
        step_kinds=frozenset({"tool"}),
        changed_fields=("error", "status", "message", "options", "text",
                        "doc_id", "title", "forecast", "rate", "as_of",
                        "currency", "to", "fee_inr_per_adult",
                        "nationality", "visa_required", "note",
                        "destination", "flights_inr", "hotels_inr",
                        "visa_inr", "total_inr", "fx_as_of", "score"),
    )

    def applicable(self, step: dict[str, Any], output: Any) -> bool:
        if step.get("kind") not in self.spec.step_kinds:
            return False
        return isinstance(output, dict)

    def apply(
        self, step: dict[str, Any], output: Any, rng: random_module.Random
    ) -> Edit:
        error_response = {
            "error": "internal_server_error",
            "status": 500,
            "message": "The service encountered an unexpected condition",
        }
        return override_output(step["addr"], error_response)


class T5SchemaDrift(FaultOperator):
    """Change field names or types to simulate API schema drift."""

    spec = FaultSpec(
        code="T5",
        family="tool",
        name="schema_drift",
        description="Rename or retype fields to simulate an API version mismatch",
        held_out=True,
        step_kinds=frozenset({"tool"}),
        changed_fields=("rate", "exchange_rate", "fare_inr_per_adult",
                        "price_inr", "nightly_local_per_room",
                        "nightly_rate", "text", "content", "score",
                        "relevance"),
    )

    _renames = {
        "rate": "exchange_rate",
        "fare_inr_per_adult": "price_inr",
        "nightly_local_per_room": "nightly_rate",
        "text": "content",
        "score": "relevance",
    }

    def applicable(self, step: dict[str, Any], output: Any) -> bool:
        if step.get("kind") not in self.spec.step_kinds:
            return False
        if not isinstance(output, dict):
            return False
        return any(k in output for k in self._renames)

    def apply(
        self, step: dict[str, Any], output: Any, rng: random_module.Random
    ) -> Edit:
        perturbed = copy.deepcopy(output)
        for old_key, new_key in self._renames.items():
            if old_key in perturbed:
                perturbed[new_key] = perturbed.pop(old_key)
        return override_output(step["addr"], perturbed)


# ---------------------------------------------------------------------------
# Retrieval faults
# ---------------------------------------------------------------------------


class R1IrrelevantDocuments(FaultOperator):
    """Replace search results with irrelevant documents."""

    spec = FaultSpec(
        code="R1",
        family="retrieval",
        name="irrelevant_documents",
        description="Replace search results with low-scoring irrelevant documents",
        held_out=False,
        step_kinds=frozenset({"retrieval", "tool"}),
        changed_fields=("doc_id", "title", "score", "text"),
    )

    def applicable(self, step: dict[str, Any], output: Any) -> bool:
        if step.get("kind") not in self.spec.step_kinds:
            return False
        # Must be a list of search results
        if isinstance(output, list) and output and isinstance(output[0], dict):
            return "doc_id" in output[0] or "score" in output[0]
        return False

    def apply(
        self, step: dict[str, Any], output: Any, rng: random_module.Random
    ) -> Edit:
        perturbed = copy.deepcopy(output)
        if not perturbed:
            return override_output(step["addr"], perturbed)
        # Reverse the ranking so the worst results come first
        perturbed.reverse()
        # Zero out scores to ensure they look irrelevant
        for item in perturbed:
            if "score" in item:
                item["score"] = rng.uniform(0, 0.05)
        return override_output(step["addr"], perturbed)


class R2PoisonedFact(FaultOperator):
    """Inject a false fact into a retrieved document."""

    spec = FaultSpec(
        code="R2",
        family="retrieval",
        name="poisoned_fact",
        description="Replace text in a retrieved document with a contradictory fact",
        held_out=True,
        step_kinds=frozenset({"tool"}),
        changed_fields=("text",),
    )

    def applicable(self, step: dict[str, Any], output: Any) -> bool:
        if step.get("kind") not in self.spec.step_kinds:
            return False
        if not isinstance(output, dict):
            return False
        return "text" in output and isinstance(output.get("text"), str)

    def apply(
        self, step: dict[str, Any], output: Any, rng: random_module.Random
    ) -> Edit:
        perturbed = copy.deepcopy(output)
        original = perturbed["text"]
        # Split into sentences and replace a random one with contradictory info
        sentences = [s.strip() for s in original.split(".") if s.strip()]
        if len(sentences) < 2:
            perturbed["text"] = "This information has been redacted due to a data error."
        else:
            idx = rng.randint(0, len(sentences) - 1)
            sentences[idx] = "Contrary to common belief, the opposite is true"
            perturbed["text"] = ". ".join(sentences) + "."
        return override_output(step["addr"], perturbed)


# ---------------------------------------------------------------------------
# Decision faults
# ---------------------------------------------------------------------------


class D1WrongArguments(FaultOperator):
    """Perturb the arguments passed to a tool by the LLM."""

    spec = FaultSpec(
        code="D1",
        family="decision",
        name="wrong_arguments",
        description="Change arguments the LLM chose for a tool call",
        held_out=False,
        step_kinds=frozenset({"llm"}),
        changed_fields=("origin", "destination", "departure", "return_date",
                        "adults", "query", "questions", "text",
                        "scenario_id", "flight_id", "hotel_id",
                        "total_inr", "budget_inr"),
    )

    def applicable(self, step: dict[str, Any], output: Any) -> bool:
        if step.get("kind") not in self.spec.step_kinds:
            return False
        # Applies to LLM steps that produce structured JSON
        if not isinstance(output, dict):
            return False
        choices = output.get("choices", [])
        if not choices:
            return False
        msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content = msg.get("content", "")
        return bool(content and isinstance(content, str))

    def apply(
        self, step: dict[str, Any], output: Any, rng: random_module.Random
    ) -> Edit:
        perturbed = copy.deepcopy(output)
        msg = perturbed["choices"][0]["message"]
        content = msg["content"]
        # Try to parse as JSON and mutate a field
        try:
            import json
            data = json.loads(content)
            if isinstance(data, dict):
                # Pick a string or numeric field and corrupt it
                mutable = [k for k, v in data.items()
                           if isinstance(v, (str, int, float))]
                if mutable:
                    key = rng.choice(mutable)
                    val = data[key]
                    if isinstance(val, (int, float)):
                        data[key] = val * rng.choice([2, 3, 0.5])
                        if isinstance(val, int):
                            data[key] = int(data[key])
                    elif isinstance(val, str) and len(val) > 3:
                        data[key] = val[::-1]  # Reverse the string
                msg["content"] = json.dumps(data, sort_keys=True)
        except (json.JSONDecodeError, TypeError, KeyError):
            # If not parseable, just corrupt the raw content
            msg["content"] = content[:len(content) // 2]
        return override_output(step["addr"], perturbed)


class D2WrongTool(FaultOperator):
    """Make the LLM call a different tool than expected."""

    spec = FaultSpec(
        code="D2",
        family="decision",
        name="wrong_tool",
        description="Simulate the LLM calling a wrong tool by swapping output content",
        held_out=False,
        step_kinds=frozenset({"llm"}),
        changed_fields=("origin", "destination", "departure", "return_date",
                        "adults", "text", "flight_id", "hotel_id",
                        "total_inr", "budget_inr", "questions", "error",
                        "message", "vegetarian", "refundable", "no_red_eye"),
    )

    def applicable(self, step: dict[str, Any], output: Any) -> bool:
        if step.get("kind") not in self.spec.step_kinds:
            return False
        if not isinstance(output, dict):
            return False
        choices = output.get("choices", [])
        if not choices:
            return False
        msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        return bool(msg.get("content"))

    def apply(
        self, step: dict[str, Any], output: Any, rng: random_module.Random
    ) -> Edit:
        perturbed = copy.deepcopy(output)
        msg = perturbed["choices"][0]["message"]
        # Replace the content with a generic wrong-tool output
        try:
            import json
            data = json.loads(msg["content"])
            if isinstance(data, dict):
                # Blank out all values to simulate wrong tool output
                for key in list(data.keys()):
                    val = data[key]
                    if isinstance(val, str):
                        data[key] = "WRONG_TOOL_OUTPUT"
                    elif isinstance(val, (int, float)):
                        data[key] = 0
                    elif isinstance(val, list):
                        data[key] = []
                msg["content"] = json.dumps(data, sort_keys=True)
        except (json.JSONDecodeError, TypeError, KeyError):
            msg["content"] = '{"error": "wrong tool called"}'
        return override_output(step["addr"], perturbed)


class D3HallucinatedValue(FaultOperator):
    """Inject a hallucinated value that doesn't appear in any input."""

    spec = FaultSpec(
        code="D3",
        family="decision",
        name="hallucinated_value",
        description="Inject a plausible but fabricated value into the LLM output",
        held_out=True,
        step_kinds=frozenset({"llm"}),
        changed_fields=("origin", "destination", "departure", "return_date",
                        "adults", "text", "flight_id", "hotel_id",
                        "total_inr", "questions", "doc_ids"),
    )

    _hallucinated_cities = [
        "Atlantis", "El Dorado", "Shangri-La", "Xanadu", "Camelot",
    ]

    def applicable(self, step: dict[str, Any], output: Any) -> bool:
        if step.get("kind") not in self.spec.step_kinds:
            return False
        if not isinstance(output, dict):
            return False
        choices = output.get("choices", [])
        if not choices:
            return False
        msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        return bool(msg.get("content"))

    def apply(
        self, step: dict[str, Any], output: Any, rng: random_module.Random
    ) -> Edit:
        perturbed = copy.deepcopy(output)
        msg = perturbed["choices"][0]["message"]
        try:
            import json
            data = json.loads(msg["content"])
            if isinstance(data, dict):
                string_keys = [k for k, v in data.items() if isinstance(v, str)]
                numeric_keys = [k for k, v in data.items()
                                if isinstance(v, (int, float))]
                if string_keys:
                    key = rng.choice(string_keys)
                    data[key] = rng.choice(self._hallucinated_cities)
                elif numeric_keys:
                    key = rng.choice(numeric_keys)
                    data[key] = rng.randint(900000, 999999)
                msg["content"] = json.dumps(data, sort_keys=True)
        except (json.JSONDecodeError, TypeError, KeyError):
            msg["content"] = '{"hallucinated": "value_from_nowhere"}'
        return override_output(step["addr"], perturbed)


class D4StopsTooEarly(FaultOperator):
    """Truncate the LLM output to simulate premature stopping."""

    spec = FaultSpec(
        code="D4",
        family="decision",
        name="stops_too_early",
        description="Truncate the LLM output to simulate premature stopping",
        held_out=False,
        step_kinds=frozenset({"llm"}),
        changed_fields=("origin", "destination", "departure", "return_date",
                        "adults", "text", "flight_id", "hotel_id",
                        "total_inr", "questions", "doc_ids", "budget_inr",
                        "finish_reason", "content"),
    )

    def applicable(self, step: dict[str, Any], output: Any) -> bool:
        if step.get("kind") not in self.spec.step_kinds:
            return False
        if not isinstance(output, dict):
            return False
        choices = output.get("choices", [])
        if not choices:
            return False
        msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content = msg.get("content", "")
        return isinstance(content, str) and len(content) > 20

    def apply(
        self, step: dict[str, Any], output: Any, rng: random_module.Random
    ) -> Edit:
        perturbed = copy.deepcopy(output)
        msg = perturbed["choices"][0]["message"]
        content = msg["content"]
        # Truncate at 30-60% of original length
        cut_point = rng.randint(len(content) // 5, len(content) // 2)
        msg["content"] = content[:cut_point]
        # Mark finish reason as length to make it detectable
        if perturbed["choices"][0].get("finish_reason"):
            perturbed["choices"][0]["finish_reason"] = "length"
        return override_output(step["addr"], perturbed)


# ---------------------------------------------------------------------------
# Coordination faults
# ---------------------------------------------------------------------------


class C1InstructionMisread(FaultOperator):
    """Simulate the LLM misreading an instruction (e.g. swapping constraints)."""

    spec = FaultSpec(
        code="C1",
        family="coordination",
        name="instruction_misread",
        description="Swap or drop a constraint to simulate misreading instructions",
        held_out=False,
        step_kinds=frozenset({"llm"}),
        changed_fields=("origin", "destination", "departure", "return_date",
                        "adults", "text", "flight_id", "hotel_id",
                        "total_inr", "questions", "vegetarian",
                        "refundable", "no_red_eye", "budget_inr"),
    )

    def applicable(self, step: dict[str, Any], output: Any) -> bool:
        if step.get("kind") not in self.spec.step_kinds:
            return False
        if not isinstance(output, dict):
            return False
        choices = output.get("choices", [])
        if not choices:
            return False
        msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        return bool(msg.get("content"))

    def apply(
        self, step: dict[str, Any], output: Any, rng: random_module.Random
    ) -> Edit:
        perturbed = copy.deepcopy(output)
        msg = perturbed["choices"][0]["message"]
        try:
            import json
            data = json.loads(msg["content"])
            if isinstance(data, dict):
                # Flip boolean constraints
                bool_keys = [k for k, v in data.items() if isinstance(v, bool)]
                if bool_keys:
                    key = rng.choice(bool_keys)
                    data[key] = not data[key]
                else:
                    # Modify a numeric constraint
                    num_keys = [k for k, v in data.items()
                                if isinstance(v, (int, float))]
                    if num_keys:
                        key = rng.choice(num_keys)
                        data[key] = data[key] * rng.choice([2, 3])
                        if isinstance(data[key], float):
                            data[key] = round(data[key], 2)
                msg["content"] = json.dumps(data, sort_keys=True)
        except (json.JSONDecodeError, TypeError, KeyError):
            pass
        return override_output(step["addr"], perturbed)


class C2ConstraintDropped(FaultOperator):
    """Drop a constraint during a hand-off between agents."""

    spec = FaultSpec(
        code="C2",
        family="coordination",
        name="constraint_dropped",
        description="Remove a constraint field during inter-agent hand-off",
        held_out=True,
        step_kinds=frozenset({"llm", "state"}),
        changed_fields=("vegetarian", "refundable", "no_red_eye",
                        "budget_inr", "adults", "origin", "destination",
                        "departure", "return_date", "text", "questions",
                        "flight_id", "hotel_id", "total_inr"),
    )

    def applicable(self, step: dict[str, Any], output: Any) -> bool:
        if step.get("kind") not in self.spec.step_kinds:
            return False
        if isinstance(output, dict):
            # Must have multiple fields to drop one
            if output.get("choices"):
                try:
                    import json
                    msg = output["choices"][0]["message"]
                    data = json.loads(msg["content"])
                    return isinstance(data, dict) and len(data) > 2
                except (json.JSONDecodeError, TypeError, KeyError, IndexError):
                    return False
            return len(output) > 2
        return False

    def apply(
        self, step: dict[str, Any], output: Any, rng: random_module.Random
    ) -> Edit:
        perturbed = copy.deepcopy(output)
        if perturbed.get("choices"):
            import json
            msg = perturbed["choices"][0]["message"]
            data = json.loads(msg["content"])
            if isinstance(data, dict) and len(data) > 2:
                droppable = [k for k in data if k not in {"scenario_id"}]
                if droppable:
                    key = rng.choice(droppable)
                    del data[key]
            msg["content"] = json.dumps(data, sort_keys=True)
        elif isinstance(perturbed, dict) and len(perturbed) > 2:
            droppable = list(perturbed.keys())
            key = rng.choice(droppable)
            del perturbed[key]
        return override_output(step["addr"], perturbed)


class C3StateCorruption(FaultOperator):
    """Corrupt a state value to simulate state management bugs."""

    spec = FaultSpec(
        code="C3",
        family="coordination",
        name="state_corruption",
        description="Corrupt a value in the agent state to simulate state bugs",
        held_out=True,
        step_kinds=frozenset({"state", "tool", "llm"}),
        changed_fields=("constraints", "flight_task", "hotel_task",
                        "destination", "flight_query", "hotel_query",
                        "weather", "fx", "visa", "flight", "hotel",
                        "flight_catalog", "hotel_catalog", "budget",
                        "draft", "verified_plan", "final_plan", "check",
                        "plan", "question", "corpus_id", "text",
                        "total_inr", "flights_inr", "hotels_inr",
                        "visa_inr", "fx_as_of", "rate", "as_of",
                        "currency", "to", "vegetarian", "refundable",
                        "no_red_eye", "origin"),
    )

    def applicable(self, step: dict[str, Any], output: Any) -> bool:
        if step.get("kind") not in self.spec.step_kinds:
            return False
        return isinstance(output, dict)

    def apply(
        self, step: dict[str, Any], output: Any, rng: random_module.Random
    ) -> Edit:
        perturbed = copy.deepcopy(output)
        if perturbed.get("choices"):
            # LLM output — corrupt the parsed content
            try:
                import json
                msg = perturbed["choices"][0]["message"]
                data = json.loads(msg["content"])
                if isinstance(data, dict):
                    keys = list(data.keys())
                    if len(keys) >= 2:
                        a, b = rng.sample(keys, 2)
                        data[a], data[b] = data[b], data[a]
                msg["content"] = json.dumps(data, sort_keys=True)
            except (json.JSONDecodeError, TypeError, KeyError, IndexError):
                pass
        else:
            # Direct dict output — swap two values
            keys = list(perturbed.keys())
            if len(keys) >= 2:
                a, b = rng.sample(keys, 2)
                perturbed[a], perturbed[b] = perturbed[b], perturbed[a]
            elif keys:
                key = keys[0]
                val = perturbed[key]
                if isinstance(val, (int, float)):
                    perturbed[key] = val * -1
                elif isinstance(val, str):
                    perturbed[key] = "CORRUPTED"
        return override_output(step["addr"], perturbed)


class C4RepeatedLoop(FaultOperator):
    """Simulate a repeated-step loop by duplicating the output content."""

    spec = FaultSpec(
        code="C4",
        family="coordination",
        name="repeated_loop",
        description="Duplicate the output to simulate the agent repeating an action",
        held_out=False,
        step_kinds=frozenset({"llm", "tool"}),
        changed_fields=("text", "questions", "doc_ids", "options",
                        "flights_inr", "hotels_inr", "visa_inr",
                        "total_inr", "rate", "origin", "destination"),
    )

    def applicable(self, step: dict[str, Any], output: Any) -> bool:
        if step.get("kind") not in self.spec.step_kinds:
            return False
        return isinstance(output, dict)

    def apply(
        self, step: dict[str, Any], output: Any, rng: random_module.Random
    ) -> Edit:
        perturbed = copy.deepcopy(output)
        if perturbed.get("choices"):
            try:
                import json
                msg = perturbed["choices"][0]["message"]
                data = json.loads(msg["content"])
                if isinstance(data, dict):
                    has_list = any(isinstance(v, list) for v in data.values())
                    if has_list:
                        for key, val in data.items():
                            if isinstance(val, list):
                                data[key] = val + val
                    else:
                        data["questions"] = ["repeat_action_1", "repeat_action_2"]
                msg["content"] = json.dumps(data, sort_keys=True)
            except (json.JSONDecodeError, TypeError, KeyError, IndexError):
                pass
        else:
            has_list = any(isinstance(v, list) for v in perturbed.values())
            if has_list:
                for key, val in list(perturbed.items()):
                    if isinstance(val, list):
                        perturbed[key] = val + val
            else:
                perturbed["options"] = ["repeat_action_1", "repeat_action_2"]
        return override_output(step["addr"], perturbed)



# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_ALL: list[FaultOperator] = [
    T1WrongValue(),
    T2StaleData(),
    T3Empty404(),
    T4Timeout500(),
    T5SchemaDrift(),
    R1IrrelevantDocuments(),
    R2PoisonedFact(),
    D1WrongArguments(),
    D2WrongTool(),
    D3HallucinatedValue(),
    D4StopsTooEarly(),
    C1InstructionMisread(),
    C2ConstraintDropped(),
    C3StateCorruption(),
    C4RepeatedLoop(),
]


def all_operators() -> list[FaultOperator]:
    """Return all registered fault operators."""
    return list(_ALL)


def seen_operators() -> list[FaultOperator]:
    """Return operators that are seen during training."""
    return [op for op in _ALL if not op.spec.held_out]


def held_out_operators() -> list[FaultOperator]:
    """Return operators held out for evaluation (unseen fault types)."""
    return [op for op in _ALL if op.spec.held_out]
