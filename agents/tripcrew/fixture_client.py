"""Explicit deterministic test double. Its pass rate is not an LLM benchmark."""

import asyncio
import json
import re

# Like a model shown an error page or an empty catalog, the double still answers; the
# agent then fails on the reply. Raising inside the call instead would leave the step
# with no recorded request, which no replay could reproduce.
NO_MATCH = "none-available"


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _cheapest(catalog, price, feasible):
    """ID of the cheapest feasible option, or NO_MATCH for an error page or drifted schema."""
    options = catalog.get("options") if isinstance(catalog, dict) else None
    candidates = [
        o
        for o in (options if isinstance(options, list) else [])
        if isinstance(o, dict)
        and isinstance(o.get(price), (int, float))
        and "id" in o
        and feasible(o)
    ]
    return min(candidates, key=lambda o: o[price])["id"] if candidates else NO_MATCH


class FixtureClient:
    def __init__(self):
        self.requests = []
        self.active = 0
        self.max_active = 0

    async def chat(self, **request):
        self.requests.append(request)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0)
            envelope = json.loads(request["messages"][-1]["content"])
            role, task = envelope["role"], envelope["task"]
            if role == "planner":
                match = re.fullmatch(
                    r"Trip (.+): travel from (.+) to (.+) departing ([\d-]+), returning ([\d-]+), "
                    r"for (\d+) adults\. Budget INR ([\d.]+)\. Vegetarian: (true|false); "
                    r"refundable: (true|false); no red-eye: (true|false)\. "
                    r"Choose the cheapest feasible flight and hotel\. Include required visa fees\.",
                    task["request"],
                )
                if match is None:
                    raise ValueError("Fixture client only supports generated TripCrew requests")
                values = match.groups()
                keys = ("scenario_id", "origin", "destination", "departure", "return_date")
                result = dict(zip(keys, values[:5]))
                result.update(
                    adults=int(values[5]),
                    budget_inr=float(values[6]),
                    vegetarian=values[7] == "true",
                    refundable=values[8] == "true",
                    no_red_eye=values[9] == "true",
                )
            elif role in {"flight_query", "hotel_query"}:
                result = task
            elif role == "flight_select":
                query = _mapping(task.get("query"))
                result = {
                    "flight_id": _cheapest(
                        task.get("catalog"),
                        "fare_inr_per_adult",
                        lambda f: (
                            (not query.get("refundable") or f.get("refundable"))
                            and (not query.get("no_red_eye") or not f.get("red_eye"))
                        ),
                    )
                }
            elif role == "hotel_select":
                query = _mapping(task.get("query"))
                result = {
                    "hotel_id": _cheapest(
                        task.get("catalog"),
                        "nightly_local_per_room",
                        lambda h: (
                            (not query.get("refundable") or h.get("refundable"))
                            and (not query.get("vegetarian") or h.get("vegetarian"))
                        ),
                    )
                }
            elif role == "writer":
                constraints = _mapping(task.get("constraints"))
                result = {
                    key: constraints.get(key)
                    for key in (
                        "scenario_id",
                        "origin",
                        "destination",
                        "departure",
                        "return_date",
                        "adults",
                    )
                }
                result.update(
                    flight_id=_mapping(task.get("flight")).get("id"),
                    hotel_id=_mapping(task.get("hotel")).get("id"),
                    total_inr=_mapping(task.get("budget")).get("total_inr"),
                    itinerary="Fly to the destination, check in, explore, and return on the requested date.",
                )
            elif role == "verifier":
                result = task.get("plan")
            else:
                raise ValueError(f"Unknown fixture role {role}")
            return {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": json.dumps(result)},
                        "finish_reason": "stop",
                    }
                ]
            }
        finally:
            self.active -= 1
