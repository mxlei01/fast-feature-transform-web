"""Download one month of NYC TLC yellow taxi trips and prepare it for training.

The Taxi and Limousine Commission publishes each month's trips as a Parquet file, typically two
months late. The cleaning below keeps ordinary metered trips paid by card or cash: the fare such a
trip shows is what the model learns to predict. Flex Fare trips (``payment_type`` 0) are left out,
because their fare is an upfront quote, and so are the flat and negotiated rates.

The month splits by pickup time: the trips of its last seven days validate the model trained on the
days before them, so validation never sees a day the model was trained on.
"""

import logging
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute
import pyarrow.parquet as pq

from training.features import MICROSECONDS_PER_DAY, epoch_microseconds, time_feature_arrays

logger = logging.getLogger(__name__)
# pyarrow generates its compute functions when the module loads, so they have no static types.
pc: Any = pyarrow.compute

TRIPS_URL: str = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{month}.parquet"
ZONES_URL: str = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
MAX_DOWNLOAD_BYTES: int = 256 * 1024 * 1024  # a month is about 60 MB
DOWNLOAD_TIMEOUT_SECONDS: float = 60.0
VALIDATION_DAYS: int = 7
MICROSECONDS_PER_MINUTE: int = 60_000_000
ROW_GROUP_ROWS: int = 262_144

RAW_COLUMNS: list[str] = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "RatecodeID",
    "PULocationID",
    "DOLocationID",
    "payment_type",
    "fare_amount",
]
PREPARED_COLUMNS: list[str] = [
    "PULocationID",
    "DOLocationID",
    "pickup_hour",
    "pickup_weekday",
    "trip_distance",
    "fare_amount",
    "pickup_us",
]


@dataclass(frozen=True)
class Prepared:
    """The prepared splits of one month.

    Attributes:
        train: Parquet file of the training trips.
        valid: Parquet file of the validation trips.
        rows_read: Trips in the month's file.
        rows_kept: Trips that passed the cleaning.

    """

    train: Path
    valid: Path
    rows_read: int
    rows_kept: int


def download(url: str, path: Path, *, max_bytes: int = MAX_DOWNLOAD_BYTES) -> Path:
    """Fetch a URL into a file once; a file already there is reused.

    The body goes to a temporary name first, so an interrupted download never looks complete.

    Args:
        url: An https URL.
        path: Where to keep the file.
        max_bytes: Refuse a body larger than this.

    Returns:
        The path.

    Raises:
        ValueError: If the URL is not https or the body is too large.

    """
    if path.exists():
        return path
    if not url.startswith("https://"):
        raise ValueError(f"refusing to download {url!r}: only https URLs are fetched")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial: Path = path.with_name(path.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "fast-feature-transform-web"})
    logger.info("downloading %s", url)
    received: int = 0
    # Bandit's B310 warns about urlopen's other schemes: the scheme is checked above, and the URLs are this
    # module's constants.
    with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response, partial.open("wb") as out:  # nosec B310
        while chunk := response.read(1 << 20):
            received += len(chunk)
            if received > max_bytes:
                out.close()
                partial.unlink()
                raise ValueError(f"{url} is larger than {max_bytes} bytes")
            out.write(chunk)
    partial.replace(path)
    logger.info("downloaded %s (%d bytes)", path.name, received)
    return path


def month_bounds(month: str) -> tuple[int, int]:
    """Return a month's first and one-past-last microsecond, as epochs.

    Args:
        month: ``YYYY-MM``.

    Returns:
        The start and the end of the month in epoch microseconds.

    """
    start: datetime = datetime.strptime(month, "%Y-%m")  # noqa: DTZ007 - New York wall-clock time
    end: datetime = (start + timedelta(days=32)).replace(day=1)
    return epoch_microseconds(start), epoch_microseconds(end)


def clean(trips: pa.Table, month: str) -> pa.Table:
    """Keep the trips the model is for, and add the time features.

    Kept: pickups inside the month, card or cash payment, the standard metered rate, a fare from
    the $3.00 meter drop to $200, a distance above 0 and up to 100 miles, a ride of 1 to 180
    minutes, 1 to 6 passengers, and both locations inside the city's zones 1 to 263 (264 and 265
    are unknown and outside the city).

    Args:
        trips: The month's file, with at least ``RAW_COLUMNS``.
        month: The month the file covers, ``YYYY-MM``.

    Returns:
        ``PREPARED_COLUMNS``, in pickup order.

    """
    start, end = month_bounds(month)
    pickup: pa.ChunkedArray = pc.cast(trips["tpep_pickup_datetime"], pa.int64())
    dropoff: pa.ChunkedArray = pc.cast(trips["tpep_dropoff_datetime"], pa.int64())
    ride: pa.ChunkedArray = pc.subtract(dropoff, pickup)
    conditions: list[pa.ChunkedArray] = [
        pc.and_(pc.greater_equal(pickup, start), pc.less(pickup, end)),
        pc.is_in(trips["payment_type"], pa.array([1, 2])),
        pc.equal(trips["RatecodeID"], 1),
        pc.and_(pc.greater_equal(trips["fare_amount"], 3.0), pc.less_equal(trips["fare_amount"], 200.0)),
        pc.and_(pc.greater(trips["trip_distance"], 0.0), pc.less_equal(trips["trip_distance"], 100.0)),
        pc.and_(pc.greater_equal(ride, MICROSECONDS_PER_MINUTE), pc.less_equal(ride, 180 * MICROSECONDS_PER_MINUTE)),
        pc.and_(pc.greater_equal(trips["passenger_count"], 1), pc.less_equal(trips["passenger_count"], 6)),
        pc.and_(pc.greater_equal(trips["PULocationID"], 1), pc.less_equal(trips["PULocationID"], 263)),
        pc.and_(pc.greater_equal(trips["DOLocationID"], 1), pc.less_equal(trips["DOLocationID"], 263)),
    ]
    keep: pa.ChunkedArray = conditions[0]
    for condition in conditions[1:]:
        keep = pc.and_(keep, condition)
    # A null anywhere in a condition drops the trip.
    kept: pa.Table = trips.filter(pc.fill_null(keep, False))
    pickup_us: np.ndarray = pc.cast(kept["tpep_pickup_datetime"], pa.int64()).to_numpy()
    hours, weekdays = time_feature_arrays(pickup_us)
    prepared: pa.Table = pa.table(
        {
            "PULocationID": pc.cast(kept["PULocationID"], pa.int64()),
            "DOLocationID": pc.cast(kept["DOLocationID"], pa.int64()),
            "pickup_hour": pa.array(hours, pa.int64()),
            "pickup_weekday": pa.array(weekdays, pa.int64()),
            "trip_distance": pc.cast(kept["trip_distance"], pa.float64()),
            "fare_amount": pc.cast(kept["fare_amount"], pa.float64()),
            "pickup_us": pa.array(pickup_us, pa.int64()),
        }
    )
    return prepared.sort_by("pickup_us")


def prepare(month: str, data_dir: Path) -> Prepared:
    """Download a month, clean it, and write its training and validation splits.

    Args:
        month: ``YYYY-MM``.
        data_dir: Directory for the raw file and the prepared splits.

    Returns:
        Where the splits are, and how many trips were kept.

    """
    raw: Path = download(TRIPS_URL.format(month=month), data_dir / "raw" / f"yellow_tripdata_{month}.parquet")
    trips: pa.Table = pq.read_table(raw, columns=RAW_COLUMNS)
    prepared: pa.Table = clean(trips, month)
    _, end = month_bounds(month)
    cutoff: int = end - VALIDATION_DAYS * MICROSECONDS_PER_DAY
    out: Path = data_dir / "prepared" / month
    out.mkdir(parents=True, exist_ok=True)
    train: Path = out / "train.parquet"
    valid: Path = out / "valid.parquet"
    pq.write_table(prepared.filter(pc.less(prepared["pickup_us"], cutoff)), train, row_group_size=ROW_GROUP_ROWS)
    pq.write_table(
        prepared.filter(pc.greater_equal(prepared["pickup_us"], cutoff)), valid, row_group_size=ROW_GROUP_ROWS
    )
    logger.info("kept %d of %d trips from %s", prepared.num_rows, trips.num_rows, month)
    return Prepared(train=train, valid=valid, rows_read=trips.num_rows, rows_kept=prepared.num_rows)
