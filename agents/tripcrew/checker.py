"""Score the final answer against the original task and fresh synthetic catalog."""

from dataclasses import dataclass

from pydantic import ValidationError

from agents.tripcrew.models import Plan
from agents.tripcrew.scenarios import Scenario, catalog, solve


@dataclass(frozen=True)
class CheckResult:
    passed: bool
    reason: str
    expected_total_inr: float


def check_plan(scenario: Scenario, answer: dict) -> CheckResult:
    oracle = solve(scenario)
    reasons = []
    try:
        plan = Plan.model_validate(answer)
    except (ValidationError, TypeError, ValueError) as error:
        return CheckResult(False, f"Invalid final plan: {error}", oracle["total_inr"])
    for key in ("scenario_id", "origin", "destination", "departure", "return_date", "adults"):
        if getattr(plan, key) != getattr(scenario, key):
            reasons.append(f"{key} does not match requested trip")
    data = catalog(scenario)
    flight = next((f for f in data["flights"] if f["id"] == plan.flight_id), None)
    hotel = next((h for h in data["hotels"] if h["id"] == plan.hotel_id), None)
    if flight is None or hotel is None:
        reasons.append("Unknown flight or hotel")
    else:
        if scenario.refundable and not (flight["refundable"] and hotel["refundable"]):
            reasons.append("Refundable constraint violated")
        if scenario.no_red_eye and flight["red_eye"]:
            reasons.append("No red-eye constraint violated")
        if scenario.vegetarian and not hotel["vegetarian"]:
            reasons.append("Vegetarian constraint violated")
        if plan.flight_id != oracle["flight_id"] or plan.hotel_id != oracle["hotel_id"]:
            reasons.append("Selections are not the cheapest feasible combination")
    if plan.total_inr > scenario.budget_inr:
        reasons.append("Budget exceeded")
    if oracle["total_inr"] > scenario.budget_inr:
        reasons.append("Actual catalog total exceeds budget")
    if abs(plan.total_inr - oracle["total_inr"]) > 1:
        reasons.append(
            f"Incorrect INR total: got {plan.total_inr:.2f}, expected {oracle['total_inr']:.2f}"
        )
    return CheckResult(
        not reasons,
        "; ".join(reasons) or "All constraints and fresh-catalog total match",
        oracle["total_inr"],
    )
