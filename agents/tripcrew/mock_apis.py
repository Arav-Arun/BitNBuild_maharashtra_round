"""Deterministic local travel tools. Never book, pay, or call external APIs."""

from __future__ import annotations

import math
from collections import Counter
from decimal import Decimal

from agents.tripcrew.scenarios import AS_OF, Scenario, catalog, money


class TravelAPI:
    def __init__(self, scenarios: list[Scenario], *, stale_fx=False):
        self.scenarios = {s.scenario_id: s for s in scenarios}
        self.stale_fx = stale_fx
        self.calls = Counter()

    def _get(self, name, scenario_id):
        self.calls[name] += 1
        scenario = self.scenarios[scenario_id]
        return scenario, catalog(scenario)

    def search_flights(self, scenario_id, **query):
        scenario, data = self._get("search_flights", scenario_id)
        matches = all(
            query.get(k) == getattr(scenario, k)
            for k in ("origin", "destination", "departure", "return_date", "adults")
        )
        return {
            "as_of": AS_OF,
            "currency": "INR",
            "round_trip": True,
            "options": data["flights"] if matches else [],
        }

    def search_hotels(self, scenario_id, **query):
        scenario, data = self._get("search_hotels", scenario_id)
        matches = all(
            query.get(k) == getattr(scenario, k)
            for k in ("destination", "departure", "return_date", "adults")
        )
        return {
            "as_of": AS_OF,
            "currency": data["currency"],
            "nights": scenario.nights,
            "rooms": math.ceil(scenario.adults / 2),
            "options": data["hotels"] if matches else [],
        }

    def get_weather(self, scenario_id, destination):
        scenario, _ = self._get("get_weather", scenario_id)
        if destination != scenario.destination:
            raise ValueError("unknown destination")
        return {"as_of": AS_OF, "destination": destination, "forecast": "Mild, possible rain"}

    def fx_rate(self, scenario_id, destination, fresh=False):
        scenario, data = self._get("fx_rate", scenario_id)
        if destination != scenario.destination:
            raise ValueError("unknown destination")
        stale = self.stale_fx and not fresh
        return {
            "as_of": "2026-03-03" if stale else AS_OF,
            "currency": data["currency"],
            "to": "INR",
            "rate": money(data["fresh_rate"] * (0.85 if stale else 1)),
        }

    def visa_rules(self, scenario_id, destination):
        scenario, data = self._get("visa_rules", scenario_id)
        if destination != scenario.destination:
            raise ValueError("unknown destination")
        return {
            "as_of": AS_OF,
            "nationality": "Indian",
            "visa_required": True,
            "fee_inr_per_adult": data["visa_fee_inr_per_adult"],
            "note": "Synthetic visa fixture; not real travel guidance",
        }


def calculate_budget(flight, hotel, fx, visa, adults, nights, rooms):
    """Agent calculator uses observed FX; it has no access to the oracle."""
    airfare = Decimal(str(flight["fare_inr_per_adult"])) * adults
    lodging = (
        Decimal(str(hotel["nightly_local_per_room"])) * nights * rooms * Decimal(str(fx["rate"]))
    )
    visa_cost = Decimal(str(visa["fee_inr_per_adult"])) * adults
    return {
        "flights_inr": money(airfare),
        "hotels_inr": money(lodging),
        "visa_inr": money(visa_cost),
        "total_inr": money(airfare + lodging + visa_cost),
        "fx_as_of": fx["as_of"],
    }
