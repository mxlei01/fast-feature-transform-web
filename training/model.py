"""The fare network: an embedding per categorical feature, joined with the numeric ones, into an MLP.

Each categorical feature arrives as the fast-inference LabelEncoder's index, where 0 is the shared
slot for an unknown or missing value, so ``vocab_size`` rows make its embedding table. The network
predicts ``log1p`` of the fare, which treats a dollar off a $10 trip like ten dollars off a $100 one.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

MAX_EMBEDDING_DIM: int = 16


@dataclass(frozen=True)
class ModelConfig:
    """Everything needed to rebuild the network around its saved weights.

    Attributes:
        categorical: Categorical feature names, in input order.
        numeric: Numeric feature names, in input order.
        vocab_sizes: Embedding rows per categorical feature, index 0 included.
        embedding_dims: Embedding width per categorical feature.
        hidden: Width of each hidden layer.
        dropout: Dropout after each hidden layer while training.

    """

    categorical: list[str]
    numeric: list[str]
    vocab_sizes: dict[str, int]
    embedding_dims: dict[str, int]
    hidden: list[int]
    dropout: float

    @classmethod
    def for_vocabularies(
        cls,
        categorical: list[str],
        numeric: list[str],
        vocab_sizes: dict[str, int],
        *,
        hidden: tuple[int, ...] = (128, 64),
        dropout: float = 0.1,
    ) -> ModelConfig:
        """Size each embedding from its vocabulary: about half of it, at most 16 wide.

        Args:
            categorical: Categorical feature names, in input order.
            numeric: Numeric feature names, in input order.
            vocab_sizes: Embedding rows per categorical feature.
            hidden: Width of each hidden layer.
            dropout: Dropout after each hidden layer.

        Returns:
            The configuration.

        """
        dims: dict[str, int] = {name: min(MAX_EMBEDDING_DIM, (vocab_sizes[name] + 1) // 2) for name in categorical}
        return cls(categorical, numeric, dict(vocab_sizes), dims, list(hidden), dropout)


class FareNet(nn.Module):
    """Predict ``log1p(fare)`` from embedding indices and numeric features."""

    def __init__(self, config: ModelConfig) -> None:
        """Build the layers a configuration describes.

        Args:
            config: The network's shape.

        """
        super().__init__()
        self.config: ModelConfig = config
        self.embeddings: nn.ModuleList = nn.ModuleList(
            nn.Embedding(config.vocab_sizes[name], config.embedding_dims[name]) for name in config.categorical
        )
        width: int = sum(config.embedding_dims[name] for name in config.categorical) + len(config.numeric)
        layers: list[nn.Module] = []
        for size in config.hidden:
            layers += [nn.Linear(width, size), nn.ReLU(), nn.Dropout(config.dropout)]
            width = size
        layers.append(nn.Linear(width, 1))
        self.mlp: nn.Sequential = nn.Sequential(*layers)

    def forward(self, categories: torch.Tensor, numbers: torch.Tensor) -> torch.Tensor:
        """Predict for a batch.

        Args:
            categories: Embedding indices, int64, one column per categorical feature.
            numbers: Numeric features, float32, one column per numeric feature.

        Returns:
            ``log1p(fare)`` per row.

        """
        parts: list[torch.Tensor] = [embed(categories[:, i]) for i, embed in enumerate(self.embeddings)]
        parts.append(numbers)
        return self.mlp(torch.cat(parts, dim=1)).squeeze(1)


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

    Args:
        directory: Artifact directory written by ``save``.

    Returns:
        The network in evaluation mode, and the parsed ``model.json``.

    """
    document: dict[str, Any] = json.loads((directory / "model.json").read_text())
    model: FareNet = FareNet(ModelConfig(**document["config"]))
    model.load_state_dict(torch.load(directory / "model.pt", weights_only=True))
    model.eval()
    return model, document
