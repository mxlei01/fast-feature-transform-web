from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fast_inference import Pipeline, StandardScaler

from training.features import FEATURES, build_pipeline, epoch_microseconds, record, time_feature_arrays, time_features
from training.train import open_memmap

EPOCH: datetime = datetime(1970, 1, 1)  # noqa: DTZ001 - naive wall-clock time is the subject


def test_epoch_microseconds_is_the_rule_fast_inference_and_arrow_apply(tmp_path: Path) -> None:
    source: Path = tmp_path / "x.parquet"
    pq.write_table(pa.table({"x": [1.0, 2.0]}), source)
    scaler: Pipeline = Pipeline([("scale", StandardScaler(columns=["x"]))]).fit([source])
    moments: list[datetime] = [EPOCH, EPOCH + timedelta(seconds=1_788_220_800), EPOCH - timedelta(microseconds=1)]
    for moment in moments:
        expected: int = pa.array([moment], pa.timestamp("us")).cast(pa.int64())[0].as_py()
        assert epoch_microseconds(moment) == expected
        # fast-inference reads a datetime under any key as the same epoch.
        assert scaler.transform_dict({"x": 1.0, "pickup": moment})["pickup"] == expected
    with pytest.raises(ValueError, match="naive"):
        epoch_microseconds(datetime(2026, 8, 1, tzinfo=UTC))


def test_time_features_match_the_calendar_in_both_forms() -> None:
    rng: np.random.Generator = np.random.default_rng(1)
    seconds: np.ndarray = rng.integers(-5 * 365 * 86_400, 70 * 365 * 86_400, 2_000)
    epochs: np.ndarray = seconds * 1_000_000 + rng.integers(0, 1_000_000, seconds.size)
    hours, weekdays = time_feature_arrays(epochs)
    for epoch, hour, weekday in zip(epochs.tolist(), hours.tolist(), weekdays.tolist(), strict=True):
        moment: datetime = EPOCH + timedelta(microseconds=epoch)
        assert time_features(epoch) == (moment.hour, moment.weekday()) == (hour, weekday)


def test_a_served_record_gives_the_training_row_for_the_same_trip(prepared: tuple[Path, Path], tmp_path: Path) -> None:
    train, _ = prepared
    pipeline: Pipeline = build_pipeline().fit([train], threads=1)
    rows: np.ndarray = open_memmap(pipeline.transform_memmap([train], output_dir=tmp_path / "mmap"))
    assert list(rows.dtype.names or ()) == FEATURES
    trips: list[dict] = pq.read_table(train).to_pylist()
    assert len(rows) == len(trips)
    for row, trip in zip(rows, trips, strict=True):
        pickup: datetime = EPOCH + timedelta(microseconds=trip["pickup_us"])
        served: dict = pipeline.transform_dict(
            record(trip["PULocationID"], trip["DOLocationID"], pickup, trip["trip_distance"])
        )
        assert list(served) == FEATURES
        assert tuple(served[name] for name in FEATURES) == row.tolist()
