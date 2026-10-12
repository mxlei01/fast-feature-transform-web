"""Load the trained model once per process, and price trips with it.

``FARE_ARTIFACTS_DIR`` holds what ``training.train`` wrote: the fitted fast-inference pipeline
(``pipeline/manifest.json``), the network (``model.pt`` and ``model.json``) and the zone names
(``zones.csv``). They are read on first use and shared by every request after that. A fitted
pipeline is immutable and safe to share across threads, and the network only ever runs in
inference mode.

The trace shows what each step of the pipeline made of a trip. It runs the trip through one
single-step pipeline per stage, each loaded from a manifest that holds just that stage (the format
fast-inference documents for its artifacts). Since the pipeline itself is that sequence of steps,
the last step's output is exactly what the model was given. The trace runs after the timed
prediction, so it never counts toward the timings shown.
"""

import csv
import json
import logging
import math
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from django.conf import settings
from fast_inference import Pipeline

from training import model as fare_model
from training.features import record

logger = logging.getLogger(__name__)

LAST_CITY_ZONE: int = 263  # 264 and 265 are "Unknown" and "Outside of NYC"


@dataclass(frozen=True)
class Zone:
    """A TLC taxi zone.

    Attributes:
        location_id: The zone's ID in the trip records.
        borough: Its borough.
        name: Its name.

    """

    location_id: int
    borough: str
    name: str


@dataclass(frozen=True)
class TraceStep:
    """One step of the pipeline applied to one trip.

    Attributes:
        stage: The step's dotted stage name.
        kernel: The fast-inference kernel that ran.
        output: The record after the step.
        changed: Keys the step added or whose value it changed.

    """

    stage: str
    kernel: str
    output: dict[str, Any]
    changed: frozenset[str]


@dataclass(frozen=True)
class Prediction:
    """A priced trip.

    Attributes:
        fare: Predicted metered fare in dollars, before tips, tolls and surcharges.
        record: What the pipeline was given.
        features: What the pipeline gave the network.
        trace: The pipeline step by step.
        timings_us: Microseconds spent building the record, in the pipeline and in the network.
        unknown: Categorical features whose value was rare or unseen in training, so the network used
            the estimate it keeps for all such values.

    """

    fare: float
    record: dict[str, Any]
    features: dict[str, Any]
    trace: list[TraceStep]
    timings_us: dict[str, float]
    unknown: list[str]


@dataclass(frozen=True)
class Served:
    """Everything loaded from one artifacts directory.

    Attributes:
        pipeline: The fitted feature pipeline.
        model: The network, in evaluation mode.
        info: ``model.json``: the network's configuration, metrics and training facts.
        zones: City zones by location ID.
        stages: The pipeline's ``stages_json``, parsed.
        states: The pipeline's ``stages_state``, parsed.
        steps: Per stage, its name, kernel and single-step pipeline, for the trace.

    """

    pipeline: Pipeline
    model: fare_model.FareNet
    info: dict[str, Any]
    zones: dict[int, Zone]
    stages: list[dict[str, Any]]
    states: dict[str, Any]
    steps: list[tuple[str, str, Pipeline]]


def read_zones(path: Path) -> dict[int, Zone]:
    """Read the TLC zone lookup, keeping the city's zones 1 to 263.

    Args:
        path: ``zones.csv``, with the columns LocationID, Borough, Zone and service_zone.

    Returns:
        Zones by location ID.

    """
    zones: dict[int, Zone] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            location: int = int(row["LocationID"])
            if 1 <= location <= LAST_CITY_ZONE:
                zones[location] = Zone(location, row["Borough"], row["Zone"])
    return zones


def single_step_pipelines(artifact: Path) -> list[tuple[str, str, Pipeline]]:
    """Load one pipeline per stage of an artifact, each holding only that stage.

    Args:
        artifact: A fast-inference artifact directory.

    Returns:
        Per stage, in execution order: its name, its kernel and its single-step pipeline.

    """
    manifest: dict[str, Any] = json.loads((artifact / "manifest.json").read_text())
    header: dict[str, Any] = {key: value for key, value in manifest.items() if key != "ops"}
    steps: list[tuple[str, str, Pipeline]] = []
    with tempfile.TemporaryDirectory(prefix="fare-trace-") as scratch:
        for index, op in enumerate(manifest["ops"]):
            directory: Path = Path(scratch) / str(index)
            directory.mkdir()
            (directory / "manifest.json").write_text(json.dumps({**header, "ops": [op]}))
            steps.append((op["stage"], op["kernel"], Pipeline.load(directory)))
    return steps


def check_pairing(info: dict[str, Any], stages: list[dict[str, Any]], states: dict[str, Any]) -> None:
    """Refuse a pipeline and a network that were not trained together.

    The pipeline's last step must select exactly the network's inputs, in order, and every
    embedding must have one row per encoded value plus the shared row 0.

    Args:
        info: The network's ``model.json``.
        stages: The pipeline's ``stages_json``, parsed.
        states: The pipeline's ``stages_state``, parsed.

    Raises:
        ValueError: If they do not match.

    """
    config: dict[str, Any] = info["config"]
    inputs: list[str] = config["categorical"] + config["numeric"]
    last: dict[str, Any] = stages[-1]
    if last["kernel"] != "select" or last["params"]["columns"] != inputs:
        raise ValueError("the pipeline does not end by selecting the network's inputs")
    maps: dict[str, Any] = next(
        (states[stage["stage"]]["label_maps"] for stage in stages if stage["kernel"] == "label_encode"), {}
    )
    for name in config["categorical"]:
        if name not in maps or config["vocab_sizes"][name] != len(maps[name]) + 1:
            raise ValueError(f"the network's embedding for {name} does not fit the pipeline's vocabulary")


def load(directory: Path) -> Served:
    """Read an artifacts directory, checking that its pipeline and network belong together.

    Args:
        directory: Where ``training.train`` wrote its artifacts.

    Returns:
        The loaded model and everything shown beside it.

    """
    pipeline: Pipeline = Pipeline.load(directory / "pipeline")
    network, info = fare_model.load(directory)
    stages: list[dict[str, Any]] = json.loads(pipeline.stages_json())
    states: dict[str, Any] = json.loads(pipeline.stages_state())
    check_pairing(info, stages, states)
    return Served(
        pipeline=pipeline,
        model=network,
        info=info,
        zones=read_zones(directory / "zones.csv"),
        stages=stages,
        states=states,
        steps=single_step_pipelines(directory / "pipeline"),
    )


_lock: threading.Lock = threading.Lock()
_served: dict[Path, Served] = {}


def served() -> Served | None:
    """Return the model for ``FARE_ARTIFACTS_DIR``, loading it on first use.

    Returns:
        The loaded model, or None when no trained model is there.

    """
    directory: Path = Path(settings.FARE_ARTIFACTS_DIR)
    loaded: Served | None = _served.get(directory)
    if loaded is not None:
        return loaded
    with _lock:
        loaded = _served.get(directory)
        if loaded is None:
            if not (directory / "model.json").exists():
                logger.warning("no trained model in %s; run `make train`", directory)
                return None
            # One record at a time: more threads would only contend with other requests.
            torch.set_num_threads(1)
            try:
                loaded = load(directory)
            except ValueError, RuntimeError, OSError:  # a mismatched pair, a corrupt file, a missing one
                logger.exception("refusing the artifacts in %s", directory)
                return None
            _served[directory] = loaded
            logger.info("loaded the fare model from %s", directory)
    return loaded


def forget() -> None:
    """Drop every loaded model, so the next request loads again (for tests)."""
    with _lock:
        _served.clear()


def predict(model: Served, pickup_zone: int, dropoff_zone: int, pickup_time: datetime, miles: float) -> Prediction:
    """Price one trip.

    Args:
        model: The loaded model.
        pickup_zone: Pickup location ID.
        dropoff_zone: Drop-off location ID.
        pickup_time: Naive New York pickup time.
        miles: Trip distance in miles.

    Returns:
        The fare, what went into it and how long each part took.

    Raises:
        ValueError: If the network's output is not a finite number.

    """
    started: int = time.perf_counter_ns()
    trip: dict[str, Any] = record(pickup_zone, dropoff_zone, pickup_time, miles)
    built: int = time.perf_counter_ns()
    features: dict[str, Any] = model.pipeline.transform_dict(trip)
    transformed: int = time.perf_counter_ns()
    # The network's own configuration says which feature feeds which input.
    categorical: list[str] = model.info["config"]["categorical"]
    numeric: list[str] = model.info["config"]["numeric"]
    categories: torch.Tensor = torch.tensor([[features[name] for name in categorical]], dtype=torch.int64)
    numbers: torch.Tensor = torch.tensor([[features[name] for name in numeric]], dtype=torch.float32)
    with torch.inference_mode():
        fare: float = float(torch.expm1(model.model(categories, numbers))[0])
    finished: int = time.perf_counter_ns()
    if not math.isfinite(fare):
        raise ValueError("the model's prediction is not a finite number")

    trace: list[TraceStep] = []
    current: dict[str, Any] = trip
    for stage, kernel, step in model.steps:
        output: dict[str, Any] = step.transform_dict(current)
        changed: frozenset[str] = frozenset(
            key
            for key, value in output.items()
            if key not in current or type(current[key]) is not type(value) or current[key] != value
        )
        trace.append(TraceStep(stage, kernel, output, changed))
        current = output
    return Prediction(
        fare=fare,
        record=trip,
        features=features,
        trace=trace,
        timings_us={
            "record": (built - started) / 1000,
            "pipeline": (transformed - built) / 1000,
            "network": (finished - transformed) / 1000,
        },
        unknown=[name for name in categorical if features[name] == 0],
    )
