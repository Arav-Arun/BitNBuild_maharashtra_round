"""Scoped JSON prompts; no worker gets another branch's observations."""

import json


def messages(role, payload, schema):
    return [
        {
            "role": "system",
            "content": (
                f"You are TripCrew's {role} worker. Return only a JSON object matching this schema: "
                f"{json.dumps(schema.model_json_schema())}. "
                "Use only supplied facts. Do not invent prices or IDs. "
                "For flight/hotel queries copy the relevant constraints. For selections choose the "
                "cheapest feasible option respecting all supplied hard constraints. Flight fares "
                "are round-trip INR per adult. Hotel rates are local currency per room per night. "
                "The writer must preserve the calculated total and IDs and provide an itinerary. "
                "The verifier must return the complete final plan, correcting discrepancies using "
                "only the supplied evidence. The planner must parse every request constraint."
            ),
        },
        {"role": "user", "content": json.dumps({"role": role, "task": payload}, sort_keys=True)},
    ]
