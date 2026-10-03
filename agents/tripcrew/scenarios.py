"""Seeded synthetic travel catalog. No prices or visa rules are real-world advice."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal

AS_OF = "2026-10-03"
DESTINATIONS = {
    "Singapore": ("SGD", 64.0),
    "Bangkok": ("THB", 2.6),
    "Dubai": ("AED", 23.0),
    "London": ("GBP", 112.0),
    "Tokyo": ("JPY", 0.58),
    "Paris": ("EUR", 96.0),
}
ORIGINS = ("Mumbai", "Delhi", "Bengaluru", "Chennai", "Hyderabad")


def money(value):
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    seed: int
    origin: str
    destination: str
    departure: str
    return_date: str
    adults: int
    budget_inr: float
    vegetarian: bool
    refundable: bool
    no_red_eye: bool
    expected_total_inr: float

    @property
    def nights(self):
        return (date.fromisoformat(self.return_date) - date.fromisoformat(self.departure)).days

    def constraints(self):
        return {k: v for k, v in asdict(self).items() if k not in {"seed", "expected_total_inr"}}

    @property
    def request(self):
        return (
            f"Trip {self.scenario_id}: travel from {self.origin} to {self.destination} "
            f"departing {self.departure}, returning {self.return_date}, for {self.adults} adults. "
            f"Budget INR {self.budget_inr:.2f}. Vegetarian: {str(self.vegetarian).lower()}; "
            f"refundable: {str(self.refundable).lower()}; "
            f"no red-eye: {str(self.no_red_eye).lower()}. "
            "Choose the cheapest feasible flight and hotel. Include required visa fees."
        )


def catalog(scenario: Scenario):
    digest = hashlib.sha256(f"{scenario.seed}:{scenario.scenario_id}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    currency, rate = DESTINATIONS[scenario.destination]
    flight_price = rng.randrange(12000, 32000, 500)
    hotel_price = money(rng.randrange(2500, 6500, 250) / rate)
    flights = [
        {"id": "F-basic", "fare_inr_per_adult": flight_price, "refundable": False, "red_eye": True},
        {
            "id": "F-flex",
            "fare_inr_per_adult": flight_price + 2500,
            "refundable": True,
            "red_eye": False,
        },
        {
            "id": "F-premium",
            "fare_inr_per_adult": flight_price + 6500,
            "refundable": True,
            "red_eye": False,
        },
    ]
    hotels = [
        {
            "id": "H-basic",
            "nightly_local_per_room": hotel_price,
            "vegetarian": False,
            "refundable": False,
        },
        {
            "id": "H-flex",
            "nightly_local_per_room": money(hotel_price * 1.25),
            "vegetarian": True,
            "refundable": True,
        },
        {
            "id": "H-premium",
            "nightly_local_per_room": money(hotel_price * 1.75),
            "vegetarian": True,
            "refundable": True,
        },
    ]
    return {
        "flights": flights,
        "hotels": hotels,
        "currency": currency,
        "fresh_rate": rate,
        "visa_fee_inr_per_adult": 1500 + rng.randrange(4) * 500,
    }


def solve(scenario: Scenario):
    """Independent oracle: enumerate feasible combinations using fresh catalog rates."""
    data = catalog(scenario)
    choices = []
    for flight in data["flights"]:
        if scenario.refundable and not flight["refundable"]:
            continue
        if scenario.no_red_eye and flight["red_eye"]:
            continue
        for hotel in data["hotels"]:
            if scenario.vegetarian and not hotel["vegetarian"]:
                continue
            if scenario.refundable and not hotel["refundable"]:
                continue
            total = (
                Decimal(str(flight["fare_inr_per_adult"])) * scenario.adults
                + Decimal(str(hotel["nightly_local_per_room"]))
                * scenario.nights
                * math.ceil(scenario.adults / 2)
                * Decimal(str(data["fresh_rate"]))
                + Decimal(str(data["visa_fee_inr_per_adult"])) * scenario.adults
            )
            choices.append((money(total), flight["id"], hotel["id"]))
    total, flight_id, hotel_id = min(choices)
    return {"total_inr": total, "flight_id": flight_id, "hotel_id": hotel_id}


def generate_scenarios(count=300, seed=7):
    if count < 1:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    scenarios = []
    from dataclasses import replace

    for index in range(count):
        departure = date(2026, 11, 1) + timedelta(days=rng.randrange(120))
        scenario = Scenario(
            f"TC-{seed}-{index + 1:04}",
            seed,
            ORIGINS[index % len(ORIGINS)],
            tuple(DESTINATIONS)[index % len(DESTINATIONS)],
            departure.isoformat(),
            (departure + timedelta(days=rng.randrange(2, 8))).isoformat(),
            rng.randrange(1, 5),
            1.0,
            bool(index & 1),
            bool(index & 2),
            bool(index & 4),
            0.0,
        )
        oracle = solve(scenario)
        scenario = replace(
            scenario,
            expected_total_inr=oracle["total_inr"],
            budget_inr=money(oracle["total_inr"] * rng.uniform(1.02, 1.20)),
        )
        scenarios.append(scenario)
    return scenarios
