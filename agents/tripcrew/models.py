"""TripCrew's structured model outputs and scenario contract."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Constraints(StrictModel):
    scenario_id: str
    origin: str
    destination: str
    departure: str
    return_date: str
    adults: int = Field(ge=1, le=6)
    budget_inr: float = Field(gt=0)
    vegetarian: bool
    refundable: bool
    no_red_eye: bool

    @model_validator(mode="after")
    def valid_dates(self):
        if date.fromisoformat(self.return_date) <= date.fromisoformat(self.departure):
            raise ValueError("return date must follow departure")
        return self


class FlightQuery(StrictModel):
    origin: str
    destination: str
    departure: str
    return_date: str
    adults: int
    refundable: bool
    no_red_eye: bool


class HotelQuery(StrictModel):
    destination: str
    departure: str
    return_date: str
    adults: int
    vegetarian: bool
    refundable: bool


class FlightChoice(StrictModel):
    flight_id: str


class HotelChoice(StrictModel):
    hotel_id: str


class Plan(StrictModel):
    scenario_id: str
    origin: str
    destination: str
    departure: str
    return_date: str
    adults: int = Field(ge=1)
    flight_id: str
    hotel_id: str
    total_inr: float = Field(ge=0)
    itinerary: str = Field(min_length=1)
