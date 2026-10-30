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
from torch import nn

from training import data
from training import train as train_module
from training.features import CATEGORICAL, FEATURES, NUMERIC
from training.model import FareNet, ModelConfig, load
from training.tests.conftest import raw_trips
from training.train import (
    fares,
    fit_network,
    main,
    open_memmap,
    predict,
    regression_metrics,
    replace_directory,
    tensors,
    train_from_prepared,
)


def test_training_writes_artifacts_that_reload_and_predict(prepared: tuple[Path, Path], tmp_path: Path) -> None:
    train, valid = prepared
    out: Path = tmp_path / "artifacts"
    metadata: dict[str, Any] = train_from_prepared(
        train, valid, tmp_path / "work", out, max_epochs=2, patience=2, batch_size=512, threads=1
    )
    assert [entry["epoch"] for entry in metadata["history"]] == [1, 2]
    assert metadata["best_epoch"] in (1, 2)
    assert metadata["train_trips"] == 3_000
    for metrics in (metadata["train_metrics"], metadata["valid_metrics"], metadata["baseline_valid_metrics"]):
        assert set(metrics) == {"r2", "mae", "rmse"}
        assert all(math.isfinite(value) for value in metrics.values())
    assert metadata["valid_mae"] == metadata["valid_metrics"]["mae"]
    assert metadata["baseline_valid_mae"] == metadata["baseline_valid_metrics"]["mae"]

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
        predicted: np.ndarray = model.fares(model(categories, numbers)).numpy()
    assert predicted.shape == (2,)
    assert np.isfinite(predicted).all()


def test_training_is_reproducible_for_a_seed(prepared: tuple[Path, Path], tmp_path: Path) -> None:
    train, valid = prepared
    first: dict[str, Any] = train_from_prepared(train, valid, tmp_path / "w1", tmp_path / "a1", max_epochs=1, threads=1)
    second: dict[str, Any] = train_from_prepared(
        train, valid, tmp_path / "w2", tmp_path / "a2", max_epochs=1, threads=1
    )
    assert first["history"] == second["history"]


@pytest.mark.parametrize("target", ["dollars", "log1p"])
def test_the_output_starts_at_the_average_target(prepared: tuple[Path, Path], tmp_path: Path, target: str) -> None:
    train, valid = prepared
    out: Path = tmp_path / "artifacts"
    # With no step size, training leaves the starting weights as they were.
    train_from_prepared(train, valid, tmp_path / "work", out, target=target, max_epochs=1, learning_rate=0.0, threads=1)
    model, _ = load(out)
    dollars: np.ndarray = fares(train)
    average: float = float(np.mean(np.log1p(dollars) if target == "log1p" else dollars))
    assert model.output.bias.item() == pytest.approx(average, rel=1e-6)


@pytest.mark.parametrize("target", ["dollars", "log1p"])
def test_the_saved_network_is_the_kept_epoch_and_its_metrics_are_in_dollars(
    prepared: tuple[Path, Path], tmp_path: Path, target: str
) -> None:
    train, valid = prepared
    out: Path = tmp_path / "artifacts"
    # A step size this large makes the run stop early, so the epoch kept is not the last one.
    metadata: dict[str, Any] = train_from_prepared(
        train,
        valid,
        tmp_path / "work",
        out,
        target=target,
        max_epochs=30,
        patience=1,
        batch_size=256,
        learning_rate=0.01,
        threads=1,
    )
    assert metadata["best_epoch"] == len(metadata["history"]) - 1 < 30
    model, document = load(out)
    assert document["config"]["target"] == target
    # Batch normalization counts the steps it trained: 11 full batches of 256 trips per epoch.
    batch_norm = model.hidden[1]
    assert isinstance(batch_norm, nn.BatchNorm1d)
    assert batch_norm.num_batches_tracked is not None
    assert batch_norm.num_batches_tracked.item() == metadata["best_epoch"] * (3_000 // 256)

    pipeline: Pipeline = Pipeline.load(out / "pipeline")
    categories, numbers = tensors(open_memmap(pipeline.transform_memmap([valid], output_dir=tmp_path / "check")))
    output: torch.Tensor = predict(model, categories, numbers)
    actual: np.ndarray = fares(valid)
    expected: dict[str, float] = regression_metrics(actual, model.fares(output).numpy())
    assert metadata["valid_metrics"] == pytest.approx(expected)
    kept: dict[str, float] = metadata["history"][metadata["best_epoch"] - 1]
    in_target_space: torch.Tensor = torch.from_numpy(
        (np.log1p(actual) if target == "log1p" else actual).astype(np.float32)
    )
    assert kept["valid_loss"] == pytest.approx(nn.functional.mse_loss(output, in_target_space).item(), rel=1e-5)
    assert kept["valid_mae"] == pytest.approx(expected["mae"], rel=1e-5)


def test_training_stops_once_validation_stops_improving_and_keeps_the_best_weights() -> None:
    torch.manual_seed(0)
    model: FareNet = FareNet(ModelConfig(["zone"], ["miles"], {"zone": 4}))
    rows: torch.Generator = torch.Generator().manual_seed(1)
    categories: torch.Tensor = torch.randint(0, 4, (640, 1), generator=rows)
    numbers: torch.Tensor = torch.randn(640, 1, generator=rows)
    target: torch.Tensor = 3 * numbers[:, 0] + categories[:, 0]
    valid: tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray] = (
        categories[512:],
        numbers[512:],
        target[512:],
        target[512:].numpy(),
    )
    # A step size this large makes the validation loss wander, so the patience runs out early.
    history, best_epoch = fit_network(
        model,
        (categories[:512], numbers[:512], target[:512]),
        valid,
        max_epochs=100,
        patience=2,
        batch_size=64,
        learning_rate=0.5,
        seed=0,
    )
    best: dict[str, float] = min(history, key=lambda entry: entry["valid_loss"])
    assert best_epoch is not None and best_epoch == best["epoch"]
    assert len(history) == best_epoch + 2 < 100
    assert history[-1]["valid_loss"] > best["valid_loss"]
    assert not model.training
    restored: float = nn.functional.mse_loss(predict(model, valid[0], valid[1]), valid[2]).item()
    assert restored == pytest.approx(best["valid_loss"], rel=1e-6)


def test_a_last_batch_of_one_trip_is_dropped() -> None:
    # Batch normalization cannot train on a single row, so the remainder of 65 trips in 64s must go.
    model: FareNet = FareNet(ModelConfig(["zone"], ["miles"], {"zone": 4}))
    categories: torch.Tensor = torch.zeros(65, 1, dtype=torch.int64)
    numbers: torch.Tensor = torch.linspace(-1, 1, 65).unsqueeze(1)
    target: torch.Tensor = numbers[:, 0]
    trips: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = (categories, numbers, target)
    history, _ = fit_network(
        model, trips, (*trips, target.numpy()), max_epochs=1, patience=1, batch_size=64, learning_rate=1e-3, seed=0
    )
    assert len(history) == 1


def test_fewer_trips_than_a_batch_train_as_one_batch() -> None:
    model: FareNet = FareNet(ModelConfig(["zone"], ["miles"], {"zone": 4}))
    numbers: torch.Tensor = torch.linspace(-1, 1, 40).unsqueeze(1)
    trips: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = (
        torch.zeros(40, 1, dtype=torch.int64),
        numbers,
        numbers[:, 0],
    )
    before: torch.Tensor = model.output.weight.detach().clone()
    history, _ = fit_network(
        model, trips, (*trips, trips[2].numpy()), max_epochs=1, patience=1, batch_size=64, learning_rate=1e-3, seed=0
    )
    assert history[0]["train_loss"] > 0
    assert not torch.equal(model.output.weight, before)


def test_a_batch_of_one_trip_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model: FareNet = FareNet(ModelConfig(["zone"], ["miles"], {"zone": 4}))
    for count, batch_size in ((1, 64), (2, 1)):
        trips: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = (
            torch.zeros(count, 1, dtype=torch.int64),
            torch.zeros(count, 1),
            torch.zeros(count),
        )
        with pytest.raises(ValueError, match="at least two trips"):
            fit_network(
                model,
                trips,
                (*trips, np.zeros(count)),
                max_epochs=1,
                patience=1,
                batch_size=batch_size,
                learning_rate=1e-3,
                seed=0,
            )

    def no_download(*args: Any, **kwargs: Any) -> Path:
        raise AssertionError("the options are checked before anything is downloaded")

    monkeypatch.setattr(data, "download", no_download)
    for option in (["--batch-size", "1"], ["--max-epochs", "0"], ["--patience", "0"]):
        with pytest.raises(SystemExit):
            main(["--data-dir", str(tmp_path / "data"), *option])


def test_a_run_that_never_reaches_a_finite_validation_loss_fails() -> None:
    model: FareNet = FareNet(ModelConfig(["zone"], ["miles"], {"zone": 4}))
    numbers: torch.Tensor = torch.linspace(-1, 1, 8).unsqueeze(1)
    trips: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = (
        torch.zeros(8, 1, dtype=torch.int64),
        numbers,
        numbers[:, 0],
    )
    unknowable: torch.Tensor = torch.full((8,), math.nan)
    with pytest.raises(ValueError, match="no epoch reached a finite validation loss"):
        fit_network(
            model,
            trips,
            (trips[0], trips[1], unknowable, unknowable.numpy()),
            max_epochs=3,
            patience=1,
            batch_size=4,
            learning_rate=1e-3,
            seed=0,
        )


def test_regression_metrics_are_in_dollars() -> None:
    metrics: dict[str, float] = regression_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 4.0]))
    assert metrics == pytest.approx({"r2": 0.5, "mae": 1 / 3, "rmse": math.sqrt(1 / 3)})
    assert regression_metrics(np.array([5.0, 5.0]), np.array([5.0, 6.0]))["r2"] == 0.0


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
    assert main(["--data-dir", str(tmp_path / "data"), "--out", str(out), "--max-epochs", "1", "--threads", "1"]) == 0
    assert sorted(path.name for path in out.iterdir()) == ["model.json", "model.pt", "pipeline", "zones.csv"]
    document: dict[str, Any] = json.loads((out / "model.json").read_text())
    assert (document["month"], document["rows_read"]) == ("2026-08", 3_000)
    assert document["train_trips"] + document["valid_trips"] == document["rows_kept"]
    assert not (tmp_path / "artifacts.staging").exists()


def test_the_command_line_passes_its_training_options(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_downloads(monkeypatch)
    options: dict[str, Any] = {}

    def fit(model: FareNet, train: Any, valid: Any, **kwargs: Any) -> tuple[list[dict[str, float]], int]:
        options.update(kwargs)
        return [{"epoch": 1, "train_loss": 1.0, "valid_loss": 1.0, "valid_mae": 1.0}], 1

    monkeypatch.setattr(train_module, "fit_network", fit)
    arguments: list[str] = ["--data-dir", str(tmp_path / "data"), "--out", str(tmp_path / "artifacts")]
    arguments += ["--max-epochs", "9", "--patience", "7", "--batch-size", "128", "--seed", "3", "--threads", "1"]
    assert main(arguments) == 0
    assert options == {"max_epochs": 9, "patience": 7, "batch_size": 128, "learning_rate": 1e-3, "seed": 3}


def test_an_interrupted_run_leaves_the_served_artifacts_alone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_downloads(monkeypatch)
    out: Path = tmp_path / "artifacts"
    arguments: list[str] = [
        "--data-dir",
        str(tmp_path / "data"),
        "--out",
        str(out),
        "--max-epochs",
        "1",
        "--threads",
        "1",
    ]
    assert main(arguments) == 0
    before: dict[str, bytes] = {path.name: path.read_bytes() for path in out.rglob("*") if path.is_file()}

    def interrupted(*args: Any, **kwargs: Any) -> tuple[list[dict[str, float]], int]:
        raise KeyboardInterrupt

    monkeypatch.setattr(train_module, "fit_network", interrupted)
    with pytest.raises(KeyboardInterrupt):
        main([*arguments, "--seed", "7"])
    assert {path.name: path.read_bytes() for path in out.rglob("*") if path.is_file()} == before


def _two_versions(served: Path, staging: Path) -> None:
    for directory, version in ((served, "served"), (staging, "new")):
        directory.mkdir()
        (directory / "model.json").write_text(version)


def _refuse_to_move(monkeypatch: pytest.MonkeyPatch, staging: Path, error: BaseException) -> None:
    rename = Path.rename

    def failing(self: Path, target: Path) -> Path:
        if self == staging:
            raise error
        return rename(self, target)

    monkeypatch.setattr(Path, "rename", failing)


# REGRESSION: a swap stopped between its two renames puts the served directory back. It used to leave no
# artifacts directory at all, and the next run then deleted artifacts.old, the only copy left.
@pytest.mark.parametrize("error", [OSError("the swap failed"), KeyboardInterrupt()], ids=["error", "interrupt"])
def test_a_swap_stopped_between_its_renames_puts_the_served_artifacts_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: BaseException
) -> None:
    out: Path = tmp_path / "artifacts"
    staging: Path = tmp_path / "artifacts.staging"
    _two_versions(out, staging)
    _refuse_to_move(monkeypatch, staging, error)
    with pytest.raises(type(error)):
        replace_directory(staging, out)
    assert (out / "model.json").read_text() == "served"
    assert (staging / "model.json").read_text() == "new"


def test_a_ctrl_c_during_the_first_rename_puts_the_served_artifacts_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out: Path = tmp_path / "artifacts"
    staging: Path = tmp_path / "artifacts.staging"
    _two_versions(out, staging)
    rename = Path.rename

    def interrupted(self: Path, target: Path) -> Path:
        moved: Path = rename(self, target)
        if self == out:
            raise KeyboardInterrupt  # the signal arrived during the rename; Python raises once it returns
        return moved

    monkeypatch.setattr(Path, "rename", interrupted)
    with pytest.raises(KeyboardInterrupt):
        replace_directory(staging, out)
    assert (out / "model.json").read_text() == "served"
    assert (staging / "model.json").read_text() == "new"


# REGRESSION: a swap that finds only artifacts.old, left by one killed between its renames, keeps it until the new
# directory is in place. It used to delete it first, so a second failure lost the last served copy.
def test_a_swap_keeps_a_killed_swaps_old_directory_until_the_new_one_is_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out: Path = tmp_path / "artifacts"
    retired: Path = tmp_path / "artifacts.old"
    staging: Path = tmp_path / "artifacts.staging"
    _two_versions(retired, staging)
    with monkeypatch.context() as patch:
        _refuse_to_move(patch, staging, OSError("the swap failed"))
        with pytest.raises(OSError, match="the swap failed"):
            replace_directory(staging, out)
    assert (out / "model.json").read_text() == "served"
    replace_directory(staging, out)
    assert (out / "model.json").read_text() == "new"
    assert not retired.exists()
    assert not staging.exists()


def test_a_failed_first_swap_raises_its_own_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out: Path = tmp_path / "artifacts"
    staging: Path = tmp_path / "artifacts.staging"
    staging.mkdir()
    (staging / "model.json").write_text("new")
    _refuse_to_move(monkeypatch, staging, OSError("the swap failed"))
    with pytest.raises(OSError, match="the swap failed"):
        replace_directory(staging, out)
    assert not out.exists()
    assert (staging / "model.json").read_text() == "new"


def test_a_ctrl_c_after_the_second_rename_keeps_the_new_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out: Path = tmp_path / "artifacts"
    staging: Path = tmp_path / "artifacts.staging"
    _two_versions(out, staging)
    rename = Path.rename

    def interrupted(self: Path, target: Path) -> Path:
        moved: Path = rename(self, target)
        if self == staging:
            raise KeyboardInterrupt  # the signal arrived during the rename; Python raises once it returns
        return moved

    monkeypatch.setattr(Path, "rename", interrupted)
    with pytest.raises(KeyboardInterrupt):
        replace_directory(staging, out)
    assert (out / "model.json").read_text() == "new"
