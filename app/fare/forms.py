"""The one way input gets in: a trip, validated field by field.

The page's form and the JSON API both validate through ``TripForm``, so they accept exactly the same
trips. A zone must be one of the city's zones, offered as a fixed list; the pickup time must be a
date and time in the one documented format, within 2009 to 2099; the distance must be a decimal
number of miles with at most two places. Anything else is refused with a message that names only
what was expected, and nothing a client sends reaches the model except through these fields.

``trip_from_json`` checks the API's JSON before the form sees it: an object with exactly the four
keys, integers for the zones (a bool is not one), a string for the time and a plain number for the
distance. ``NaN`` and ``Infinity``, which Python's ``json`` would otherwise accept, are refused.
"""

import json
from datetime import datetime
from decimal import Decimal
from typing import Any

from django import forms
from django.core.exceptions import ValidationError

from fare.serving import Zone

DATETIME_FORMATS: list[str] = ["%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S"]
EARLIEST: datetime = datetime(2009, 1, 1)  # noqa: DTZ001 - the first month the TLC published; New York time
LATEST: datetime = datetime(2099, 12, 31, 23, 59, 59)  # noqa: DTZ001
MIN_MILES: Decimal = Decimal("0.1")
MAX_MILES: Decimal = Decimal("100")
FIELDS: frozenset[str] = frozenset({"pickup_zone", "dropoff_zone", "pickup_time", "trip_distance"})


class WallClockField(forms.DateTimeField):
    """A New York date and time in exactly ``DATETIME_FORMATS``.

    Django's own field also tries ``datetime.fromisoformat``, which takes UTC offsets, dates alone
    and other ISO spellings. A trip's time is a wall-clock time like the trip records', so only the
    documented formats parse, and an offset is never accepted.
    """

    def to_python(self, value: Any) -> datetime | None:
        """Parse the field.

        Args:
            value: The submitted value.

        Returns:
            A naive datetime, or None when empty.

        Raises:
            ValidationError: If it is not in one of the formats.

        """
        if value in self.empty_values:
            return None
        if isinstance(value, datetime):
            return value
        text: str = str(value).strip()
        for pattern in DATETIME_FORMATS:
            try:
                return datetime.strptime(text, pattern)  # noqa: DTZ007 - New York wall-clock time
            except ValueError:
                continue
        raise ValidationError(self.error_messages["invalid"], code="invalid")


class TripForm(forms.Form):
    """A trip to price."""

    # Django's own choice message repeats the value it got; these name only what is expected.
    pickup_zone = forms.TypedChoiceField(
        coerce=int, choices=(), label="Pickup zone", error_messages={"invalid_choice": "Choose one of the zones."}
    )
    dropoff_zone = forms.TypedChoiceField(
        coerce=int, choices=(), label="Drop-off zone", error_messages={"invalid_choice": "Choose one of the zones."}
    )
    pickup_time = WallClockField(
        input_formats=DATETIME_FORMATS,
        label="Pickup time (New York)",
        widget=forms.DateTimeInput(attrs={"type": "datetime-local"}, format=DATETIME_FORMATS[0]),
        error_messages={"invalid": "Enter a date and time such as 2026-08-14T08:30."},
    )
    trip_distance = forms.DecimalField(
        min_value=MIN_MILES,
        max_value=MAX_MILES,
        max_digits=5,
        decimal_places=2,
        label="Distance (miles)",
        widget=forms.NumberInput(attrs={"step": "0.01"}),
    )

    def __init__(self, *args: Any, zones: dict[int, Zone], **kwargs: Any) -> None:
        """Offer the model's zones as the only choices.

        Args:
            *args: Passed to ``forms.Form``.
            zones: The zones the model knows, by location ID.
            **kwargs: Passed to ``forms.Form``.

        """
        super().__init__(*args, **kwargs)
        choices: list[tuple[int, str]] = [
            (zone.location_id, f"{zone.name} ({zone.borough})")
            for zone in sorted(zones.values(), key=lambda zone: (zone.borough, zone.name, zone.location_id))
        ]
        for name in ("pickup_zone", "dropoff_zone"):
            field = self.fields[name]
            assert isinstance(field, forms.TypedChoiceField)
            field.choices = choices

    def clean_pickup_time(self) -> datetime:
        """Keep the pickup time inside the published years.

        Returns:
            The pickup time.

        Raises:
            forms.ValidationError: If it is out of range.

        """
        value: datetime = self.cleaned_data["pickup_time"]
        if not EARLIEST <= value <= LATEST:
            raise forms.ValidationError(f"Choose a time from {EARLIEST:%Y} to {LATEST:%Y}.")
        return value


class JSONTripError(ValueError):
    """The API's JSON is not a trip; the message is safe to show the client."""


def _refuse_constant(name: str) -> Any:
    """Refuse ``NaN``, ``Infinity`` and ``-Infinity``, which are not JSON.

    Args:
        name: The constant.

    Raises:
        JSONTripError: Always.

    """
    raise JSONTripError(f"{name} is not a number")


def trip_from_json(body: bytes) -> dict[str, str]:
    """Check the API's JSON and turn it into form data.

    Args:
        body: The request body.

    Returns:
        The four fields as strings, for ``TripForm`` to validate.

    Raises:
        JSONTripError: If the body is not exactly such an object.

    """
    try:
        payload: Any = json.loads(body, parse_constant=_refuse_constant, parse_float=Decimal)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise JSONTripError("the body is not valid JSON") from error
    if not isinstance(payload, dict):
        raise JSONTripError("expected a JSON object")
    if set(payload) != FIELDS:
        # Name only our own keys: nothing the client sent is echoed back.
        raise JSONTripError(f"expected exactly the keys {', '.join(sorted(FIELDS))}")
    for name in ("pickup_zone", "dropoff_zone"):
        if type(payload[name]) is not int:
            raise JSONTripError(f"{name} must be an integer")
    if not isinstance(payload["pickup_time"], str):
        raise JSONTripError("pickup_time must be a string such as 2026-08-14T08:30")
    if type(payload["trip_distance"]) not in (int, Decimal):
        raise JSONTripError("trip_distance must be a number")
    return {name: str(payload[name]) for name in FIELDS}
