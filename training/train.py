"""Train the fare model and write the artifacts the web app serves.

    uv run python -m training.train --month 2026-08

The month's trips are downloaded and cleaned (``training.data``), the fast-inference pipeline is
fitted on the training days and writes both splits straight into memmaps, and the network trains on
those. ``artifacts/`` then holds everything serving needs: the fitted pipeline, the weights, the
configuration with its metrics, and the zone names.
"""

import argparse
import copy
import json
import logging
import math
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import fast_inference
import numpy as np
import pyarrow.parquet as pq
import torch
from fast_inference import LabelEncoder, Pipeline
from torch import nn

from training import data, log
from training.features import CATEGORICAL, NUMERIC, build_pipeline
from training.model import FareNet, ModelConfig, save

logger = logging.getLogger(__name__)

DEFAULT_MONTH: str = "2026-08"
EVALUATION_BATCH: int = 65_536
# What the network predicts; see training.model.
TARGET: str = "dollars"


def open_memmap(index_path: Path) -> np.ndarray:
    """Open the records a fast-inference memmap holds.

    Args:
        index_path: The ``index.json`` that ``transform_memmap`` returned.

    Returns:
        A read-only structured array, one field per pipeline output column.

    Raises:
        ValueError: If the index is not a little-endian fast-inference memmap.

    """
    index: dict[str, Any] = json.loads(index_path.read_text())
    if index.get("format") != "fast-inference-memmap" or index.get("byte_order") != "little":
        raise ValueError(f"{index_path} is not a little-endian fast-inference memmap index")
    dtype: np.dtype = np.dtype([(column["name"], column["dtype"]) for column in index["columns"]])
    if index["rows"] == 0:
        return np.empty(0, dtype)
    return np.memmap(index_path.parent / index["file"], dtype=dtype, mode="r", shape=(index["rows"],))


def tensors(records: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Split memmap records into the network's two inputs.

    Args:
        records: Structured records with the ``CATEGORICAL`` and ``NUMERIC`` fields.

    Returns:
        Embedding indices (int64) and numeric features (float32), one column per feature.

    """
    categories: np.ndarray = np.stack([records[name] for name in CATEGORICAL], axis=1).astype(np.int64)
    numbers: np.ndarray = np.stack([records[name] for name in NUMERIC], axis=1).astype(np.float32)
    return torch.from_numpy(categories), torch.from_numpy(numbers)


def fares(path: Path) -> np.ndarray:
    """Read the fares of a prepared split, in its row order.

    Args:
        path: Prepared Parquet file.

    Returns:
        The fares in dollars.

    """
    return pq.read_table(path, columns=["fare_amount"])["fare_amount"].to_numpy()


def predict(model: FareNet, categories: torch.Tensor, numbers: torch.Tensor) -> torch.Tensor:
    """Run the network in evaluation mode, in batches.

    Args:
        model: The network.
        categories: Embedding indices.
        numbers: Numeric features.

    Returns:
        The outputs, in the network's target space.

    """
    model.eval()
    out: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(categories), EVALUATION_BATCH):
            stop: int = start + EVALUATION_BATCH
            out.append(model(categories[start:stop], numbers[start:stop]))
    return torch.cat(out) if out else torch.empty(0)


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """R squared, mean absolute error and root mean squared error, in dollars.

    Args:
        actual: True fares.
        predicted: Predicted fares.

    Returns:
        ``{"r2", "mae", "rmse"}``.

    """
    error: np.ndarray = predicted.astype(np.float64) - actual
    spread: float = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "r2": 1.0 - float(np.sum(error**2)) / spread if spread > 0 else 0.0,
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
    }


def baseline_metrics(
    train_miles: np.ndarray, train_fares: np.ndarray, miles: np.ndarray, actual: np.ndarray
) -> dict[str, float]:
    """Score a straight line from distance to fare, for comparison.

    Args:
        train_miles: Training distances.
        train_fares: Training fares.
        miles: Validation distances.
        actual: Validation fares.

    Returns:
        The line's metrics on the validation trips.

    """
    slope, intercept = np.polyfit(train_miles, train_fares, 1)
    return regression_metrics(actual, slope * miles + intercept)


def fit_network(
    model: FareNet,
    train: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    valid: tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray],
    *,
    max_epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[list[dict[str, float]], int]:
    """Train with Adam on squared error, stopping once validation loss stops improving.

    Each epoch shuffles the training trips into full batches and drops a trailing partial batch,
    since batch normalization cannot train on a single row; fewer trips than a batch make one
    batch of them all. After ``patience`` epochs without a lower validation loss the run stops,
    and the weights of the best epoch are kept.

    Args:
        model: The network to train in place.
        train: Embedding indices, numeric features and targets of the training trips.
        valid: Embedding indices, numeric features, targets, and fares in dollars of the
            validation trips.
        max_epochs: The most passes over the training trips.
        patience: Epochs to wait for a lower validation loss.
        batch_size: Trips per step.
        learning_rate: Adam's step size.
        seed: Seed for the shuffling.

    Returns:
        One entry per epoch run (training loss, validation loss and validation mean absolute
        error in dollars), and the epoch whose weights were kept.

    Raises:
        ValueError: If a batch could hold a single trip, or no epoch reached a finite validation
            loss, so there is nothing worth keeping.

    """
    categories, numbers, target = train
    if batch_size < 2 or len(target) < 2:
        raise ValueError("batch normalization needs at least two trips in every batch")
    valid_categories, valid_numbers, valid_target, valid_fares = valid
    optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    shuffle: torch.Generator = torch.Generator().manual_seed(seed)
    usable: int = len(target) - len(target) % batch_size if len(target) >= batch_size else len(target)
    history: list[dict[str, float]] = []
    best_loss: float = math.inf
    best_state: dict[str, torch.Tensor] = {}
    best_epoch: int | None = None
    stale: int = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        order: torch.Tensor = torch.randperm(len(target), generator=shuffle)[:usable]
        total: float = 0.0
        for start in range(0, usable, batch_size):
            batch: torch.Tensor = order[start : start + batch_size]
            optimizer.zero_grad()
            loss: torch.Tensor = nn.functional.mse_loss(model(categories[batch], numbers[batch]), target[batch])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        output: torch.Tensor = predict(model, valid_categories, valid_numbers)
        valid_loss: float = nn.functional.mse_loss(output, valid_target).item()
        valid_mae: float = float(np.mean(np.abs(model.fares(output).numpy() - valid_fares)))
        history.append(
            {"epoch": epoch, "train_loss": total / max(usable, 1), "valid_loss": valid_loss, "valid_mae": valid_mae}
        )
        logger.info(
            "epoch %d: train loss %.5f, validation loss %.5f, validation MAE $%.3f",
            epoch,
            history[-1]["train_loss"],
            valid_loss,
            valid_mae,
        )
        if valid_loss < best_loss:
            best_loss, best_epoch, stale = valid_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= patience:
                break
    if best_epoch is None:
        raise ValueError("no epoch reached a finite validation loss")
    model.load_state_dict(best_state)
    model.eval()
    return history, best_epoch


def train_from_prepared(
    train_path: Path,
    valid_path: Path,
    work_dir: Path,
    out_dir: Path,
    *,
    target: str = TARGET,
    max_epochs: int = 300,
    patience: int = 15,
    batch_size: int = 4096,
    learning_rate: float = 1e-3,
    threads: int = 4,
    seed: int = 0,
) -> dict[str, Any]:
    """Fit the pipeline and the network on prepared splits, and write the artifacts.

    Args:
        train_path: Prepared training trips.
        valid_path: Prepared validation trips.
        work_dir: Scratch directory for the memmaps.
        out_dir: Artifact directory: ``pipeline/``, ``model.pt`` and ``model.json`` go here.
        target: What the network predicts: ``"dollars"`` or ``"log1p"`` of dollars.
        max_epochs: The most passes over the training trips.
        patience: Epochs to wait for a lower validation loss before stopping.
        batch_size: Trips per step.
        learning_rate: Adam's step size.
        threads: Threads for fast-inference and torch.
        seed: Seed for the weights and the shuffling.

    Returns:
        The metadata written to ``model.json`` beside the configuration.

    """
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    pipeline: Pipeline = build_pipeline().fit([train_path], threads=threads)
    pipeline.save(out_dir / "pipeline")
    encoder = pipeline["indices"]
    assert isinstance(encoder, LabelEncoder)
    vocab_sizes: dict[str, int] = {name: encoder.vocab_size(name) for name in CATEGORICAL}

    train_records: np.ndarray = open_memmap(pipeline.transform_memmap([train_path], output_dir=work_dir / "train"))
    valid_records: np.ndarray = open_memmap(pipeline.transform_memmap([valid_path], output_dir=work_dir / "valid"))
    train_fares: np.ndarray = fares(train_path)
    valid_fares: np.ndarray = fares(valid_path)
    if len(train_records) != len(train_fares) or len(valid_records) != len(valid_fares):
        raise ValueError("the memmaps and the prepared splits disagree on the number of trips")
    train_categories, train_numbers = tensors(train_records)
    valid_categories, valid_numbers = tensors(valid_records)

    model: FareNet = FareNet(ModelConfig(CATEGORICAL, NUMERIC, vocab_sizes, target=target))

    def targets(dollars: np.ndarray) -> torch.Tensor:
        return torch.from_numpy((np.log1p(dollars) if target == "log1p" else dollars).astype(np.float32))

    train_target: torch.Tensor = targets(train_fares)
    # The output starts at the average target, so the first epochs refine fares instead of climbing to them.
    with torch.no_grad():
        model.output.bias.fill_(train_target.mean().item())
    history, best_epoch = fit_network(
        model,
        (train_categories, train_numbers, train_target),
        (valid_categories, valid_numbers, targets(valid_fares), valid_fares),
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
    )
    train_metrics: dict[str, float] = regression_metrics(
        train_fares, model.fares(predict(model, train_categories, train_numbers)).numpy()
    )
    valid_metrics: dict[str, float] = regression_metrics(
        valid_fares, model.fares(predict(model, valid_categories, valid_numbers)).numpy()
    )
    distances = pq.read_table(train_path, columns=["trip_distance"])["trip_distance"].to_numpy()
    valid_distances = pq.read_table(valid_path, columns=["trip_distance"])["trip_distance"].to_numpy()
    baseline: dict[str, float] = baseline_metrics(distances, train_fares, valid_distances, valid_fares)
    metadata: dict[str, Any] = {
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "fast_inference_version": fast_inference.__version__,
        "torch_version": torch.__version__,
        "train_trips": len(train_fares),
        "valid_trips": len(valid_fares),
        "history": history,
        "best_epoch": best_epoch,
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "baseline_valid_metrics": baseline,
        "valid_mae": valid_metrics["mae"],
        "baseline_valid_mae": baseline["mae"],
    }
    save(model, out_dir, metadata)
    logger.info(
        "validation MAE $%.3f, RMSE $%.3f, R2 %.4f; a line from distance alone: MAE $%.3f",
        valid_metrics["mae"],
        valid_metrics["rmse"],
        valid_metrics["r2"],
        baseline["mae"],
    )
    return metadata


def replace_directory(finished: Path, target: Path) -> None:
    """Put a finished artifacts directory where the old one was.

    A server reads the directory when it starts, so it must never meet a new pipeline beside an
    old network: training writes everything elsewhere first and only then swaps it in. A swap
    stopped between its two renames, by an error or Ctrl-C, puts the old directory back; one
    killed there leaves only the old directory, under its .old name, which the next swap keeps
    until the new directory is in place.

    Args:
        finished: The complete new directory.
        target: Where it belongs.

    """
    retired: Path = target.with_name(target.name + ".old")
    try:
        if target.exists():
            shutil.rmtree(retired, ignore_errors=True)
            target.rename(retired)
        finished.rename(target)
    except BaseException:
        if retired.exists() and not target.exists():
            retired.rename(target)
        raise
    shutil.rmtree(retired, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    """Run the command line.

    Args:
        argv: Arguments, without the program name; the process's own when None.

    Returns:
        The exit status.

    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--month", default=DEFAULT_MONTH, help="YYYY-MM of the TLC file to train on")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="downloads, splits and memmaps")
    parser.add_argument("--out", type=Path, default=Path("artifacts"), help="where the served artifacts go")
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.batch_size < 2:
        parser.error("--batch-size must be at least 2: batch normalization cannot train on one trip")
    if args.max_epochs < 1 or args.patience < 1:
        parser.error("--max-epochs and --patience must be at least 1")
    log.configure()
    prepared: data.Prepared = data.prepare(args.month, args.data_dir)
    zones: Path = data.download(data.ZONES_URL, args.data_dir / "raw" / "taxi_zone_lookup.csv", max_bytes=1 << 20)
    staging: Path = args.out.with_name(args.out.name + ".staging")
    shutil.rmtree(staging, ignore_errors=True)
    metadata: dict[str, Any] = train_from_prepared(
        prepared.train,
        prepared.valid,
        args.data_dir / "memmap" / args.month,
        staging,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        threads=args.threads,
        seed=args.seed,
    )
    shutil.copyfile(zones, staging / "zones.csv")
    document: dict[str, Any] = json.loads((staging / "model.json").read_text())
    document.update({"month": args.month, "rows_read": prepared.rows_read, "rows_kept": prepared.rows_kept})
    (staging / "model.json").write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    replace_directory(staging, args.out)
    logger.info("artifacts written to %s (validation MAE $%.3f)", args.out, metadata["valid_mae"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
