import io
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from training import data
from training.data import PREPARED_COLUMNS, clean, download, month_bounds
from training.features import epoch_microseconds, time_features
from training.tests.conftest import MONTH, raw_trips

EPOCH: datetime = datetime(1970, 1, 1)  # noqa: DTZ001 - naive wall-clock time is the subject


def test_month_bounds_cover_the_whole_month() -> None:
    start, end = month_bounds("2026-08")
    assert start == epoch_microseconds(datetime(2026, 8, 1))  # noqa: DTZ001
    assert end == epoch_microseconds(datetime(2026, 9, 1))  # noqa: DTZ001
    assert month_bounds("2026-12")[1] == epoch_microseconds(datetime(2027, 1, 1))  # noqa: DTZ001


def test_clean_keeps_only_ordinary_metered_trips() -> None:
    good: pa.Table = raw_trips(50, seed=3)
    rows: list[dict[str, Any]] = good.to_pylist()
    base: dict[str, Any] = rows[0]
    pickup: datetime = base["tpep_pickup_datetime"]
    rejected: list[dict[str, Any]] = [
        {**base, "tpep_pickup_datetime": datetime(2026, 7, 31, 23, 0)},  # noqa: DTZ001 - before the month
        {**base, "payment_type": 0},  # a Flex Fare quote
        {**base, "payment_type": 4},  # a dispute
        {**base, "RatecodeID": 2},  # the JFK flat rate
        {**base, "RatecodeID": None},
        {**base, "fare_amount": 2.99},
        {**base, "fare_amount": 200.01},
        {**base, "trip_distance": 0.0},
        {**base, "trip_distance": 100.01},
        {**base, "tpep_dropoff_datetime": pickup + timedelta(seconds=59)},
        {**base, "tpep_dropoff_datetime": pickup + timedelta(minutes=181)},
        {**base, "tpep_dropoff_datetime": pickup - timedelta(minutes=5)},
        {**base, "passenger_count": 0},
        {**base, "passenger_count": 7},
        {**base, "passenger_count": None},
        {**base, "PULocationID": 264},
        {**base, "DOLocationID": 265},
        {**base, "DOLocationID": 0},
    ]
    table: pa.Table = pa.Table.from_pylist(rows + rejected, schema=good.schema)
    cleaned: pa.Table = clean(table, MONTH)
    assert cleaned.column_names == PREPARED_COLUMNS
    assert cleaned.num_rows == len(rows)
    pickups: list[int] = cleaned["pickup_us"].to_pylist()
    assert pickups == sorted(pickups)
    for trip in cleaned.to_pylist():
        assert (trip["pickup_hour"], trip["pickup_weekday"]) == time_features(trip["pickup_us"])


def test_clean_keeps_a_trip_on_each_boundary() -> None:
    base: dict[str, Any] = raw_trips(1, seed=4).to_pylist()[0]
    pickup: datetime = datetime(2026, 8, 31, 23, 59, 59)  # noqa: DTZ001 - the month's last second
    edge: dict[str, Any] = {
        **base,
        "tpep_pickup_datetime": pickup,
        "tpep_dropoff_datetime": pickup + timedelta(minutes=180),
        "fare_amount": 3.0,
        "trip_distance": 100.0,
        "passenger_count": 6,
        "PULocationID": 1,
        "DOLocationID": 263,
    }
    cleaned: pa.Table = clean(pa.Table.from_pylist([edge], schema=raw_trips(1).schema), MONTH)
    assert cleaned.num_rows == 1
    assert cleaned["pickup_us"][0].as_py() == (pickup - EPOCH) // timedelta(microseconds=1)
    assert cleaned["fare_amount"].to_pylist() == [3.0]


def test_download_fetches_once_and_refuses_other_schemes_and_oversized_bodies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def fake_urlopen(request: Any, timeout: float) -> io.BytesIO:
        calls.append(request.full_url)
        return io.BytesIO(b"x" * 10)

    monkeypatch.setattr(data.urllib.request, "urlopen", fake_urlopen)
    target: Path = tmp_path / "raw" / "file.bin"
    assert download("https://example.test/file.bin", target) == target
    assert target.read_bytes() == b"x" * 10
    download("https://example.test/file.bin", target)
    assert calls == ["https://example.test/file.bin"]  # the second call reuses the file

    with pytest.raises(ValueError, match="only https"):
        download("http://example.test/plain.bin", tmp_path / "plain.bin")
    with pytest.raises(ValueError, match="only https"):
        download("file:///etc/hosts", tmp_path / "hosts")
    with pytest.raises(ValueError, match="larger than 5 bytes"):
        download("https://example.test/big.bin", tmp_path / "big.bin", max_bytes=5)
    assert not (tmp_path / "big.bin").exists()
    assert not (tmp_path / "big.bin.part").exists()


def test_prepare_splits_by_pickup_time_without_overlap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw: pa.Table = raw_trips(3_000, seed=5)

    def fake_download(url: str, path: Path, **kwargs: Any) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(raw, path)
        return path

    monkeypatch.setattr(data, "download", fake_download)
    prepared = data.prepare(MONTH, tmp_path)
    train: pa.Table = pq.read_table(prepared.train)
    valid: pa.Table = pq.read_table(prepared.valid)
    _, end = month_bounds(MONTH)
    cutoff: int = end - data.VALIDATION_DAYS * 86_400_000_000
    assert max(train["pickup_us"].to_pylist()) < cutoff <= min(valid["pickup_us"].to_pylist())
    assert train.num_rows + valid.num_rows == prepared.rows_kept == clean(raw, MONTH).num_rows
    assert prepared.rows_read == 3_000
