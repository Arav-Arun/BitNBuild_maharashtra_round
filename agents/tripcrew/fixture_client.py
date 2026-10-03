"""Explicit deterministic test double. Its pass rate is not an LLM benchmark."""

import asyncio
import json
import re


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
                options = [
                    f
                    for f in task["catalog"]["options"]
                    if (not task["query"]["refundable"] or f["refundable"])
                    and (not task["query"]["no_red_eye"] or not f["red_eye"])
                ]
                result = {"flight_id": min(options, key=lambda f: f["fare_inr_per_adult"])["id"]}
            elif role == "hotel_select":
                options = [
                    h
                    for h in task["catalog"]["options"]
                    if (not task["query"]["refundable"] or h["refundable"])
                    and (not task["query"]["vegetarian"] or h["vegetarian"])
                ]
                result = {"hotel_id": min(options, key=lambda h: h["nightly_local_per_room"])["id"]}
            elif role == "writer":
                result = {
                    key: task["constraints"][key]
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
                    flight_id=task["flight"]["id"],
                    hotel_id=task["hotel"]["id"],
                    total_inr=task["budget"]["total_inr"],
                    itinerary="Fly to the destination, check in, explore, and return on the requested date.",
                )
            elif role == "verifier":
                result = task["plan"]
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
