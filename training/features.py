"""The features, computed the same way in training and in the web app.

A pickup time is carried as an epoch: int64 microseconds since 1970-01-01, counting its wall-clock
time as if it were UTC. That is what fast-inference reads a naive ``datetime`` and a naive Parquet
timestamp as, and the TLC files store naive New York times, so the hour and weekday below are New
York's. Both come from integer arithmetic on the epoch, so no calendar or time zone is involved.

Everything else is the fast-inference pipeline, which is saved with the model: the location IDs and
time features become embedding indices, with rare and unseen values sharing index 0, and the trip
distance is made roughly normal.
"""

from datetime import datetime, timedelta
from typing import Any

import numpy as np
from fast_inference import Cast, LabelEncoder, Pipeline, PowerTransformer, Select

EPOCH: datetime = datetime(1970, 1, 1)  # noqa: DTZ001 - the epoch of naive wall-clock time
ONE_MICROSECOND: timedelta = timedelta(microseconds=1)
MICROSECONDS_PER_HOUR: int = 3_600_000_000
MICROSECONDS_PER_DAY: int = 86_400_000_000
# 1970-01-01 was a Thursday; with Monday as 0 that is day 3.
EPOCH_WEEKDAY: int = 3

CATEGORICAL: list[str] = ["PULocationID", "DOLocationID", "pickup_hour", "pickup_weekday"]
# Values seen fewer times than this share index 0 with values never seen, so that index's embedding
# learns from the rare ones instead of keeping its random start.
MIN_COUNT: int = 10
NUMERIC: list[str] = ["trip_distance"]
FEATURES: list[str] = CATEGORICAL + NUMERIC


def epoch_microseconds(value: datetime) -> int:
    """Count a naive datetime's wall-clock time in microseconds since 1970-01-01.

    This is the rule fast-inference applies to a naive datetime. ``datetime.timestamp()`` is not:
    it reads a naive value in the machine's own time zone.

    Args:
        value: A naive datetime.

    Returns:
        The epoch in microseconds.

    Raises:
        ValueError: If the datetime carries a time zone.

    """
    if value.utcoffset() is not None:
        raise ValueError("expected a naive datetime (New York wall-clock time)")
    return (value - EPOCH) // ONE_MICROSECOND


def time_features(pickup_us: int) -> tuple[int, int]:
    """Return the hour of day and the weekday (Monday is 0) of an epoch.

    Args:
        pickup_us: Pickup time in epoch microseconds.

    Returns:
        The hour, 0 to 23, and the weekday, 0 to 6.

    """
    return pickup_us // MICROSECONDS_PER_HOUR % 24, (pickup_us // MICROSECONDS_PER_DAY + EPOCH_WEEKDAY) % 7


def time_feature_arrays(pickup_us: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized ``time_features``: numpy's ``//`` floors as Python's does.

    Args:
        pickup_us: Pickup times in epoch microseconds, as int64.

    Returns:
        The hours and the weekdays, as int64 arrays.

    """
    return pickup_us // MICROSECONDS_PER_HOUR % 24, (pickup_us // MICROSECONDS_PER_DAY + EPOCH_WEEKDAY) % 7


def record(pickup_zone: int, dropoff_zone: int, pickup: datetime, trip_distance: float) -> dict[str, Any]:
    """Build the record the feature pipeline transforms for one trip.

    Args:
        pickup_zone: TLC location ID of the pickup.
        dropoff_zone: TLC location ID of the drop-off.
        pickup: Naive New York pickup time.
        trip_distance: Trip distance in miles.

    Returns:
        The record, keyed like the prepared training data.

    """
    hour, weekday = time_features(epoch_microseconds(pickup))
    return {
        "PULocationID": pickup_zone,
        "DOLocationID": dropoff_zone,
        "pickup_hour": hour,
        "pickup_weekday": weekday,
        "trip_distance": trip_distance,
    }


def build_pipeline() -> Pipeline:
    """Declare the feature pipeline; fitting it learns the vocabularies and the distance transform.

    Returns:
        An unfitted pipeline whose output is exactly ``FEATURES``, in that order.

    """
    return Pipeline(
        [
            ("categories", Cast(types={name: "utf8" for name in CATEGORICAL})),
            ("indices", LabelEncoder(columns=CATEGORICAL, min_count=MIN_COUNT)),
            ("distance", PowerTransformer(columns=NUMERIC)),
            ("features", Select(columns=FEATURES)),
        ]
    )
