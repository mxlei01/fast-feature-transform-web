"""Synthetic trips shaped like the TLC files, so the tests never download anything."""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from training.data import clean

MONTH: str = "2026-08"


def raw_trips(rows: int, seed: int = 0) -> pa.Table:
    """Make a month of plausible trips with the TLC schema's types.

    Args:
        rows: Number of trips.
        seed: Seed for the values.

    Returns:
        A table with the columns ``training.data.clean`` reads.

    """
    rng: np.random.Generator = np.random.default_rng(seed)
    start: datetime = datetime(2026, 8, 1)  # noqa: DTZ001 - New York wall-clock time, as in the files
    pickup: list[datetime] = [start + timedelta(seconds=int(s)) for s in rng.integers(0, 31 * 86_400, rows)]
    miles: np.ndarray = np.round(rng.gamma(2.0, 1.6, rows), 2) + 0.1
    minutes: np.ndarray = np.clip(miles * 4 + rng.normal(5, 2, rows), 1.5, 170)
    dropoff: list[datetime] = [p + timedelta(minutes=float(m)) for p, m in zip(pickup, minutes, strict=True)]
    zones: np.ndarray = rng.integers(1, 264, (rows, 2))
    fare: np.ndarray = np.round(3.0 + 2.8 * miles + 0.35 * minutes + (zones[:, 0] % 7) * 0.4, 2)
    return pa.table(
        {
            "tpep_pickup_datetime": pa.array(pickup, pa.timestamp("us")),
            "tpep_dropoff_datetime": pa.array(dropoff, pa.timestamp("us")),
            "passenger_count": pa.array(rng.integers(1, 5, rows), pa.int64()),
            "trip_distance": pa.array(miles, pa.float64()),
            "RatecodeID": pa.array(np.ones(rows, np.int64), pa.int64()),
            "PULocationID": pa.array(zones[:, 0], pa.int32()),
            "DOLocationID": pa.array(zones[:, 1], pa.int32()),
            "payment_type": pa.array(rng.integers(1, 3, rows), pa.int64()),
            "fare_amount": pa.array(np.minimum(fare, 199.0), pa.float64()),
        }
    )


@pytest.fixture
def prepared(tmp_path: Path) -> tuple[Path, Path]:
    """Write cleaned training and validation splits of synthetic trips.

    Args:
        tmp_path: pytest-provided temporary directory.

    Returns:
        The training and validation Parquet files.

    """
    table: pa.Table = clean(raw_trips(4_000), MONTH)
    train: Path = tmp_path / "train.parquet"
    valid: Path = tmp_path / "valid.parquet"
    pq.write_table(table.slice(0, 3_000), train, row_group_size=1_000)
    pq.write_table(table.slice(3_000), valid)
    return train, valid
