"""Parse the bounded natural-language task supported by the offline TripCrew demo."""

from __future__ import annotations

import re
from datetime import date

from agents.tripcrew.scenarios import DESTINATIONS, ORIGINS, Scenario, solve


def _preference(prompt: str, name: str, *, implied: bool = False) -> bool:
    match = re.search(
        rf"\b{re.escape(name)}\b\s*(?:is\s*)?[:=]?\s*(yes|no|true|false)\b",
        prompt,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).casefold() in {"yes", "true"}
    return implied and bool(re.search(rf"\b{re.escape(name)}\b", prompt, re.IGNORECASE))


def parse_trip_prompt(prompt: str, *, scenario_id: str, seed: int) -> Scenario:
    """Turn an explicit trip request into validated constraints for the local demo agent.

    The offline runner deliberately supports the small synthetic catalog only. It does not
    pretend to execute arbitrary tasks or provide real travel inventory.
    """
    text = " ".join(prompt.split())
    route = re.search(
        r"\bfrom\s+([A-Za-z][A-Za-z .'-]*?)\s+to\s+([A-Za-z][A-Za-z .'-]*?)"
        r"(?=\s+(?:departing|depart|leaving|on|for|with|budget)\b|[,;.]|$)",
        text,
        re.IGNORECASE,
    )
    if not route:
        raise ValueError("Include a route like ‘from Mumbai to Singapore’.")
    origins = {city.casefold(): city for city in ORIGINS}
    destinations = {city.casefold(): city for city in DESTINATIONS}
    origin = origins.get(route.group(1).strip().casefold())
    destination = destinations.get(route.group(2).strip().casefold())
    if not origin:
        raise ValueError(f"Origin must be one of: {', '.join(ORIGINS)}.")
    if not destination:
        raise ValueError(f"Destination must be one of: {', '.join(DESTINATIONS)}.")

    dates = re.findall(r"\b20\d{2}-\d{2}-\d{2}\b", text)
    if len(dates) < 2:
        raise ValueError("Include departure and return dates in YYYY-MM-DD format.")
    try:
        departure, return_date = (date.fromisoformat(value) for value in dates[:2])
    except ValueError as error:
        raise ValueError("One of the dates is not a real calendar date.") from error
    if return_date <= departure:
        raise ValueError("The return date must be after the departure date.")

    travelers = re.search(r"\b(\d+)\s*(?:adults?|people|travelers?|travellers?)\b", text, re.I)
    if not travelers:
        raise ValueError("Include the number of adults or travelers, from 1 to 6.")
    adults = int(travelers.group(1))
    if not 1 <= adults <= 6:
        raise ValueError("The offline demo supports 1 to 6 travelers.")

    budget = re.search(
        r"\bbudget\s*(?:is\s*)?(?:of\s*)?(?:INR\s*)?([\d,]+(?:\.\d{1,2})?)"
        r"|\bINR\s*([\d,]+(?:\.\d{1,2})?)",
        text,
        re.I,
    )
    if not budget:
        raise ValueError("Include a budget, for example ‘budget INR 160000’.")
    budget_inr = float((budget.group(1) or budget.group(2)).replace(",", ""))
    if not 1000 <= budget_inr <= 100_000_000:
        raise ValueError("Budget must be between INR 1,000 and INR 100,000,000.")

    draft = Scenario(
        scenario_id=scenario_id,
        seed=seed,
        origin=origin,
        destination=destination,
        departure=departure.isoformat(),
        return_date=return_date.isoformat(),
        adults=adults,
        budget_inr=budget_inr,
        vegetarian=_preference(text, "vegetarian", implied=True),
        refundable=_preference(text, "refundable", implied=True),
        no_red_eye=_preference(text, "no red-eye") or _preference(text, "no red eye", implied=True),
        expected_total_inr=0,
    )
    return Scenario(**{**draft.__dict__, "expected_total_inr": solve(draft)["total_inr"]})
