"""A small model trained once per session on synthetic trips, served to every test."""

import csv
from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from django.core.cache import cache
from pytest_django.fixtures import Settings

from fare import serving
from training.data import clean
from training.tests.conftest import MONTH, raw_trips
from training.train import train_from_prepared

# A zone name carrying markup: every page must show it as text.
HOSTILE_ZONE: str = '<script>alert("zone")</script>'


@pytest.fixture(scope="session")
def artifacts(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train a one-epoch model and write the artifacts the app serves.

    Args:
        tmp_path_factory: pytest's session temporary directories.

    Returns:
        The artifacts directory.

    """
    root: Path = tmp_path_factory.mktemp("fare")
    table = clean(raw_trips(3_000), MONTH)
    pq.write_table(table.slice(0, 2_400), root / "train.parquet")
    pq.write_table(table.slice(2_400), root / "valid.parquet")
    out: Path = root / "artifacts"
    train_from_prepared(root / "train.parquet", root / "valid.parquet", root / "work", out, epochs=1, threads=1)
    with (out / "zones.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["LocationID", "Borough", "Zone", "service_zone"])
        for location in range(1, 266):
            name: str = HOSTILE_ZONE if location == 7 else f"Zone {location}"
            writer.writerow([location, "Queens" if location % 2 else "Brooklyn", name, "Boro Zone"])
    return out


@pytest.fixture(autouse=True)
def served_artifacts(artifacts: Path, settings: Settings) -> Iterator[Path]:
    """Point the app at the trained artifacts, with a fresh cache and rate limit for each test.

    Args:
        artifacts: The trained artifacts.
        settings: pytest-django's settings override.

    Yields:
        The artifacts directory.

    """
    settings.FARE_ARTIFACTS_DIR = artifacts
    settings.FARE_RATE_LIMIT = 1_000
    serving.forget()
    cache.clear()
    yield artifacts
    serving.forget()
