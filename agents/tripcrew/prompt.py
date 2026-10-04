"""Parse the bounded natural-language task supported by the offline TripCrew demo."""

from __future__ import annotations

import hashlib
import re
from datetime import date

from agents.tripcrew.scenarios import DESTINATIONS, ORIGINS, Scenario, solve

# Common alternative names people type for the supported cities.
CITY_ALIASES = {
    "bombay": "Mumbai",
    "new delhi": "Delhi",
    "bangalore": "Bengaluru",
    "madras": "Chennai",
}
CURRENCY = r"(?:INR|Rs\.?|rupees|₹)"
AMOUNT = r"(\d[\d,]*(?:\.\d{1,2})?)\s*(lakhs?|lacs?)?"


def catalog_seed(prompt: str) -> tuple[str, int]:
    """Stable scenario identity; changing only the budget must not change catalog prices."""
    price_request = prompt
    # Currency and budget amounts describe a constraint, not the trip inventory. Excluding
    # them means raising a budget cannot silently reshuffle prices and change the task itself.
    for pattern in (
        rf"\bbudget\s*(?:is\s*)?(?:of\s*)?(?:about\s*)?{CURRENCY}?\s*{AMOUNT}",
        rf"{CURRENCY}\s*{AMOUNT}",
        rf"\b{AMOUNT}\s*(?:INR|rupees)\b",
    ):
        price_request = re.sub(pattern, "budget", price_request, flags=re.IGNORECASE)
    digest = hashlib.sha256(" ".join(price_request.split()).casefold().encode("utf-8")).hexdigest()
    return f"PROMPT-{digest[:12].upper()}", int(digest[:8], 16)


def _city(text: str, cities: tuple[str, ...] | dict) -> str | None:
    """The supported city a phrase starts with, so trailing words ("next Friday") are ignored."""
    phrase = text.strip().casefold()
    names = {city.casefold(): city for city in cities}
    names.update({alias: city for alias, city in CITY_ALIASES.items() if city in cities})
    for name in sorted(names, key=len, reverse=True):
        if phrase == name or phrase.startswith(name + " "):
            return names[name]
    return None


def _budget(text: str) -> float | None:
    """INR budget from 'budget INR 1,60,000', '₹1.6 lakh', 'Rs 90000' or '90000 rupees'."""
    patterns = (
        rf"\bbudget\s*(?:is\s*)?(?:of\s*)?(?:about\s*)?{CURRENCY}?\s*{AMOUNT}",
        rf"{CURRENCY}\s*{AMOUNT}",
        rf"\b{AMOUNT}\s*(?:INR|rupees)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = float(match.group(1).replace(",", ""))
            return value * 100_000 if match.group(2) else value
    return None


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
        r"(?=\s+(?:departing|depart|leaving|returning|on|for|with|budget|next|this|in|between)\b"
        r"|\s+\d|[,;.]|$)",
        text,
        re.IGNORECASE,
    )
    if not route:
        raise ValueError("Include a route like ‘from Mumbai to Singapore’.")
    origin = _city(route.group(1), ORIGINS)
    destination = _city(route.group(2), tuple(DESTINATIONS))
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

    budget_inr = _budget(text)
    if budget_inr is None:
        raise ValueError("Include a budget, for example ‘budget ₹1,60,000’ or ‘budget 1.6 lakh’.")
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
