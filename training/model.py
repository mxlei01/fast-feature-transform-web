"""The fare network: categorical embeddings and numeric features into a batch-normalized MLP.

Every categorical feature arrives as the fast-inference LabelEncoder's index, where 0 is the slot
for a rare or unseen value. All of them share one embedding table: each feature's indices are
shifted by an offset so they land in their own rows, and one lookup embeds every feature of a batch
at once. The embeddings, flattened, join the numeric features and go through hidden layers of
Linear, BatchNorm, ReLU and Dropout to a single output.

The output is the fare in the space the configuration's ``target`` names: ``"dollars"``, or
``"log1p"`` for the logarithm of one plus the fare. ``fares`` turns it back into dollars.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

TARGETS: frozenset[str] = frozenset({"dollars", "log1p"})


@dataclass(frozen=True)
class ModelConfig:
    """Everything needed to rebuild the network around its saved weights.

    Attributes:
        categorical: Categorical feature names, in input order.
        numeric: Numeric feature names, in input order.
        vocab_sizes: Embedding rows per categorical feature, index 0 included.
        target: What the output is: ``"dollars"`` or ``"log1p"`` of dollars.
        embedding_dim: Width of every feature's embedding.
        hidden: Width of each hidden layer.
        dropout: Dropout after each hidden layer while training.

    """

    categorical: list[str]
    numeric: list[str]
    vocab_sizes: dict[str, int]
    target: str = "dollars"
    embedding_dim: int = 8
    hidden: list[int] = field(default_factory=lambda: [128, 64, 32])
    dropout: float = 0.3

    def __post_init__(self) -> None:
        """Check the target and that every categorical feature has a vocabulary.

        Raises:
            ValueError: If either is wrong.

        """
        if self.target not in TARGETS:
            raise ValueError(f"target must be one of {sorted(TARGETS)}")
        missing: list[str] = [name for name in self.categorical if name not in self.vocab_sizes]
        if missing:
            raise ValueError(f"no vocabulary size for {missing}")


class FareNet(nn.Module):
    """Predict a trip's fare from embedding indices and numeric features."""

    offsets: torch.Tensor

    def __init__(self, config: ModelConfig) -> None:
        """Build the layers a configuration describes.

        Args:
            config: The network's shape.

        """
        super().__init__()
        self.config: ModelConfig = config
        starts: list[int] = []
        rows: int = 0
        for name in config.categorical:
            starts.append(rows)
            rows += config.vocab_sizes[name]
        self.register_buffer("offsets", torch.tensor(starts, dtype=torch.int64))
        self.embedding: nn.Embedding = nn.Embedding(rows, config.embedding_dim)
        width: int = len(config.categorical) * config.embedding_dim + len(config.numeric)
        layers: list[nn.Module] = []
        for size in config.hidden:
            layers += [nn.Linear(width, size), nn.BatchNorm1d(size), nn.ReLU(), nn.Dropout(config.dropout)]
            width = size
        self.hidden: nn.Sequential = nn.Sequential(*layers)
        self.output: nn.Linear = nn.Linear(width, 1)

    def forward(self, categories: torch.Tensor, numbers: torch.Tensor) -> torch.Tensor:
        """Predict for a batch.

        Args:
            categories: Embedding indices, int64, one column per categorical feature.
            numbers: Numeric features, float32, one column per numeric feature.

        Returns:
            One output per row, in the configuration's target space.

        """
        embedded: torch.Tensor = self.embedding(categories + self.offsets)
        return self.output(self.hidden(torch.cat([embedded.flatten(1), numbers], dim=1))).squeeze(1)

    def fares(self, output: torch.Tensor) -> torch.Tensor:
        """Turn outputs into fares in dollars.

        Args:
            output: What ``forward`` returned.

        Returns:
            The fares.

        """
        return torch.expm1(output) if self.config.target == "log1p" else output


def save(model: FareNet, directory: Path, metadata: dict[str, Any]) -> None:
    """Write ``model.pt`` (the weights) and ``model.json`` (the configuration and metadata).

    Args:
        model: The trained network.
        directory: Artifact directory.
        metadata: Training facts to keep beside the configuration, such as metrics.

    """
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), directory / "model.pt")
    document: dict[str, Any] = {"config": asdict(model.config), **metadata}
    (directory / "model.json").write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def load(directory: Path) -> tuple[FareNet, dict[str, Any]]:
    """Rebuild a saved network for inference.

    The weights load with ``weights_only=True``, which reads tensors only and never runs pickled code.
    They must have been trained with the configuration beside them: the offsets saved with the
    shared embedding table must be the ones its vocabulary sizes give.

    Args:
        directory: Artifact directory written by ``save``.

    Returns:
        The network in evaluation mode, and the parsed ``model.json``.

    Raises:
        ValueError: If ``model.json`` holds no configuration this code can build, or ``model.pt``
            was trained with other vocabularies.

    """
    document: dict[str, Any] = json.loads((directory / "model.json").read_text())
    try:
        model: FareNet = FareNet(ModelConfig(**document["config"]))
    except (KeyError, TypeError) as error:  # written by other code, or no configuration at all
        raise ValueError(f"{directory / 'model.json'} holds no configuration this code can build") from error
    offsets: torch.Tensor = model.offsets.clone()
    model.load_state_dict(torch.load(directory / "model.pt", weights_only=True))
    if not torch.equal(model.offsets, offsets):
        raise ValueError(f"{directory / 'model.pt'} was not trained with the vocabularies in model.json")
    model.eval()
    return model, document
