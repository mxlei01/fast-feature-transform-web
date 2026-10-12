"""Train the fare model and write the artifacts the web app serves.

    uv run python -m training.train --month 2026-08

The month's trips are downloaded and cleaned (``training.data``), the fast-inference pipeline is
fitted on the training days and writes both splits straight into memmaps, and the network trains on
those. ``artifacts/`` then holds everything serving needs: the fitted pipeline, the weights, the
configuration with its metrics, and the zone names.
"""

import argparse
import json
import logging
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


def predict(model: FareNet, categories: torch.Tensor, numbers: torch.Tensor) -> np.ndarray:
    """Predict fares in dollars, in batches.

    Args:
        model: The network.
        categories: Embedding indices.
        numbers: Numeric features.

    Returns:
        The predicted fares.

    """
    model.eval()
    out: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(categories), EVALUATION_BATCH):
            stop: int = start + EVALUATION_BATCH
            out.append(torch.expm1(model(categories[start:stop], numbers[start:stop])).numpy())
    return np.concatenate(out) if out else np.empty(0, np.float32)


def baseline_mae(train_miles: np.ndarray, train_fares: np.ndarray, miles: np.ndarray, actual: np.ndarray) -> float:
    """Mean absolute error of a straight line from distance to fare, for comparison.

    Args:
        train_miles: Training distances.
        train_fares: Training fares.
        miles: Validation distances.
        actual: Validation fares.

    Returns:
        The line's mean absolute error on the validation trips, in dollars.

    """
    slope, intercept = np.polyfit(train_miles, train_fares, 1)
    return float(np.mean(np.abs(slope * miles + intercept - actual)))


def fit_network(
    model: FareNet,
    train: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    valid: tuple[torch.Tensor, torch.Tensor, np.ndarray],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> list[dict[str, float]]:
    """Train with Adam on squared error of ``log1p(fare)``, reporting validation error each epoch.

    Args:
        model: The network to train in place.
        train: Embedding indices, numeric features and ``log1p`` fares of the training trips.
        valid: Embedding indices, numeric features and fares in dollars of the validation trips.
        epochs: Passes over the training trips.
        batch_size: Trips per step.
        learning_rate: Adam's step size.
        seed: Seed for the shuffling.

    Returns:
        One entry per epoch: training loss and validation mean absolute error in dollars.

    """
    categories, numbers, target = train
    optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    shuffle: torch.Generator = torch.Generator().manual_seed(seed)
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        order: torch.Tensor = torch.randperm(len(target), generator=shuffle)
        total: float = 0.0
        for start in range(0, len(order), batch_size):
            batch: torch.Tensor = order[start : start + batch_size]
            optimizer.zero_grad()
            loss: torch.Tensor = nn.functional.mse_loss(model(categories[batch], numbers[batch]), target[batch])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        valid_mae: float = float(np.mean(np.abs(predict(model, valid[0], valid[1]) - valid[2])))
        history.append({"epoch": epoch, "train_loss": total / max(len(order), 1), "valid_mae": valid_mae})
        logger.info("epoch %d: train loss %.5f, validation MAE $%.3f", epoch, history[-1]["train_loss"], valid_mae)
    return history


def train_from_prepared(
    train_path: Path,
    valid_path: Path,
    work_dir: Path,
    out_dir: Path,
    *,
    epochs: int = 10,
    batch_size: int = 4096,
    learning_rate: float = 2e-3,
    threads: int = 4,
    seed: int = 0,
) -> dict[str, Any]:
    """Fit the pipeline and the network on prepared splits, and write the artifacts.

    Args:
        train_path: Prepared training trips.
        valid_path: Prepared validation trips.
        work_dir: Scratch directory for the memmaps.
        out_dir: Artifact directory: ``pipeline/``, ``model.pt`` and ``model.json`` go here.
        epochs: Passes over the training trips.
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

    model: FareNet = FareNet(ModelConfig.for_vocabularies(CATEGORICAL, NUMERIC, vocab_sizes))
    target: torch.Tensor = torch.from_numpy(np.log1p(train_fares).astype(np.float32))
    history: list[dict[str, float]] = fit_network(
        model,
        (train_categories, train_numbers, target),
        (valid_categories, valid_numbers, valid_fares),
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
    )
    distances = pq.read_table(train_path, columns=["trip_distance"])["trip_distance"].to_numpy()
    valid_distances = pq.read_table(valid_path, columns=["trip_distance"])["trip_distance"].to_numpy()
    metadata: dict[str, Any] = {
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "fast_inference_version": fast_inference.__version__,
        "torch_version": torch.__version__,
        "train_trips": len(train_fares),
        "valid_trips": len(valid_fares),
        "history": history,
        "valid_mae": history[-1]["valid_mae"] if history else None,
        "baseline_valid_mae": baseline_mae(distances, train_fares, valid_distances, valid_fares),
    }
    save(model, out_dir, metadata)
    logger.info(
        "validation MAE $%.3f against $%.3f for a line from distance alone",
        metadata["valid_mae"],
        metadata["baseline_valid_mae"],
    )
    return metadata


def replace_directory(finished: Path, target: Path) -> None:
    """Put a finished artifacts directory where the old one was.

    A server reads the directory when it starts, so it must never meet a new pipeline beside an
    old network: training writes everything elsewhere first and only then swaps it in.

    Args:
        finished: The complete new directory.
        target: Where it belongs.

    """
    retired: Path = target.with_name(target.name + ".old")
    shutil.rmtree(retired, ignore_errors=True)
    if target.exists():
        target.rename(retired)
    finished.rename(target)
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
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
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
        epochs=args.epochs,
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
