import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from fast_inference import Pipeline

from training import data
from training import train as train_module
from training.features import CATEGORICAL, FEATURES, NUMERIC
from training.model import FareNet, load
from training.tests.conftest import raw_trips
from training.train import main, train_from_prepared


def test_training_writes_artifacts_that_reload_and_predict(prepared: tuple[Path, Path], tmp_path: Path) -> None:
    train, valid = prepared
    out: Path = tmp_path / "artifacts"
    metadata: dict[str, Any] = train_from_prepared(train, valid, tmp_path / "work", out, epochs=2, threads=1)
    assert [entry["epoch"] for entry in metadata["history"]] == [1, 2]
    assert metadata["train_trips"] == 3_000
    assert math.isfinite(metadata["valid_mae"])
    assert math.isfinite(metadata["baseline_valid_mae"])

    document: dict[str, Any] = json.loads((out / "model.json").read_text())
    assert document["config"]["categorical"] == CATEGORICAL
    assert document["config"]["numeric"] == NUMERIC
    pipeline: Pipeline = Pipeline.load(out / "pipeline")
    stages: list[dict[str, Any]] = json.loads(pipeline.stages_json())
    assert [stage["stage"] for stage in stages] == ["categories", "indices", "distance", "features"]
    assert stages[-1]["params"]["columns"] == FEATURES
    states: dict[str, Any] = json.loads(pipeline.stages_state())
    for name in CATEGORICAL:
        assert document["config"]["vocab_sizes"][name] == len(states["indices"]["label_maps"][name]) + 1

    model, _ = load(out)
    assert isinstance(model, FareNet)
    assert not model.training
    categories: torch.Tensor = torch.tensor([[1, 2, 3, 4], [0, 0, 0, 0]])
    numbers: torch.Tensor = torch.tensor([[0.5], [-1.0]])
    with torch.inference_mode():
        predicted: np.ndarray = torch.expm1(model(categories, numbers)).numpy()
    assert predicted.shape == (2,)
    assert np.isfinite(predicted).all()


def test_training_is_reproducible_for_a_seed(prepared: tuple[Path, Path], tmp_path: Path) -> None:
    train, valid = prepared
    first: dict[str, Any] = train_from_prepared(train, valid, tmp_path / "w1", tmp_path / "a1", epochs=1, threads=1)
    second: dict[str, Any] = train_from_prepared(train, valid, tmp_path / "w2", tmp_path / "a2", epochs=1, threads=1)
    assert first["history"] == second["history"]


def _stub_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    raw: pa.Table = raw_trips(3_000, seed=6)

    def fake_download(url: str, path: Path, **kwargs: Any) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        if url == data.ZONES_URL:
            lines: list[str] = ['"LocationID","Borough","Zone","service_zone"']
            lines += [f'{zone},"Queens","Zone {zone}","Boro Zone"' for zone in range(1, 266)]
            path.write_text("\r\n".join(lines) + "\r\n")
        else:
            pq.write_table(raw, path)
        return path

    monkeypatch.setattr(data, "download", fake_download)


def test_the_command_line_writes_complete_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_downloads(monkeypatch)
    out: Path = tmp_path / "artifacts"
    assert main(["--data-dir", str(tmp_path / "data"), "--out", str(out), "--epochs", "1", "--threads", "1"]) == 0
    assert sorted(path.name for path in out.iterdir()) == ["model.json", "model.pt", "pipeline", "zones.csv"]
    document: dict[str, Any] = json.loads((out / "model.json").read_text())
    assert (document["month"], document["rows_read"]) == ("2026-08", 3_000)
    assert document["train_trips"] + document["valid_trips"] == document["rows_kept"]
    assert not (tmp_path / "artifacts.staging").exists()


def test_an_interrupted_run_leaves_the_served_artifacts_alone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_downloads(monkeypatch)
    out: Path = tmp_path / "artifacts"
    arguments: list[str] = ["--data-dir", str(tmp_path / "data"), "--out", str(out), "--epochs", "1", "--threads", "1"]
    assert main(arguments) == 0
    before: dict[str, bytes] = {path.name: path.read_bytes() for path in out.rglob("*") if path.is_file()}

    def interrupted(*args: Any, **kwargs: Any) -> list[dict[str, float]]:
        raise KeyboardInterrupt

    monkeypatch.setattr(train_module, "fit_network", interrupted)
    with pytest.raises(KeyboardInterrupt):
        main([*arguments, "--seed", "7"])
    assert {path.name: path.read_bytes() for path in out.rglob("*") if path.is_file()} == before
