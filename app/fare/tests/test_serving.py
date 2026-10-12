import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import pytest
from django.conf import settings

from fare import serving
from training.features import CATEGORICAL, FEATURES, NUMERIC

REPO: Path = Path(__file__).resolve().parents[3]

PICKUP: datetime = datetime(2026, 8, 14, 17, 30)  # noqa: DTZ001 - New York wall-clock time


def test_a_prediction_traces_every_step_to_the_features_the_network_saw(served_artifacts: Path) -> None:
    model = serving.served()
    assert model is not None
    prediction = serving.predict(model, 132, 230, PICKUP, 17.5)
    assert math.isfinite(prediction.fare) and prediction.fare > 0
    assert list(prediction.features) == FEATURES
    assert [step.stage for step in prediction.trace] == ["categories", "indices", "distance", "features"]
    assert prediction.trace[-1].output == prediction.features
    assert prediction.record == {
        "PULocationID": 132,
        "DOLocationID": 230,
        "pickup_hour": 17,
        "pickup_weekday": 4,
        "trip_distance": 17.5,
    }
    # The cast turns every categorical value into text, and the encoder turns it into an index.
    assert prediction.trace[0].changed == {"PULocationID", "DOLocationID", "pickup_hour", "pickup_weekday"}
    assert prediction.trace[1].changed == {"PULocationID", "DOLocationID", "pickup_hour", "pickup_weekday"}
    assert prediction.trace[2].changed == {"trip_distance"}
    assert all(value >= 0 for value in prediction.timings_us.values())


def test_the_model_loads_once_and_unknown_zones_use_the_shared_index(served_artifacts: Path) -> None:
    first = serving.served()
    assert first is not None and serving.served() is first
    assert set(first.zones) == set(range(1, 264))
    prediction = serving.predict(first, 1, 263, PICKUP, 1.0)
    assert math.isfinite(prediction.fare)


def test_a_rare_zone_is_flagged_as_using_the_shared_estimate(served_artifacts: Path) -> None:
    model = serving.served()
    assert model is not None
    seen: set[str] = set(model.states["indices"]["label_maps"]["PULocationID"])
    rare: int = next(zone for zone in range(1, 264) if str(zone) not in seen)
    prediction = serving.predict(model, rare, 236, PICKUP, 2.4)
    assert prediction.features["PULocationID"] == 0
    assert "PULocationID" in prediction.unknown
    assert math.isfinite(prediction.fare)


def test_the_committed_model_loads_and_prices_an_ordinary_trip() -> None:
    model = serving.load(REPO / "artifacts")
    assert model.info["config"]["categorical"] == CATEGORICAL
    assert model.info["config"]["numeric"] == NUMERIC
    assert model.stages[-1]["params"]["columns"] == FEATURES
    prediction = serving.predict(model, 161, 236, datetime(2026, 8, 12, 9, 15), 2.4)  # noqa: DTZ001
    assert 8.0 < prediction.fare < 30.0  # Midtown to the Upper East Side, 2.4 miles
    assert prediction.unknown == []


def test_a_network_from_another_training_run_is_refused(served_artifacts: Path, tmp_path: Path) -> None:
    # A pipeline whose vocabulary differs from the one the network was trained with, as an
    # interrupted run that replaced only the pipeline would leave behind.
    mixed: Path = tmp_path / "mixed"
    shutil.copytree(served_artifacts, mixed)
    manifest_path: Path = mixed / "pipeline" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    state = next(op["state"] for op in manifest["ops"] if op["kernel"] == "label_encode")
    zones: dict[str, int] = state["label_maps"]["PULocationID"]
    del zones[max(zones, key=zones.__getitem__)]
    state["vocab_sizes"]["PULocationID"] -= 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="does not fit the pipeline's vocabulary"):
        serving.load(mixed)
    settings.FARE_ARTIFACTS_DIR = mixed
    assert serving.served() is None  # the site answers "no model" rather than mispricing
