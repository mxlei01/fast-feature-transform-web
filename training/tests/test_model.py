import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch

from training.model import FareNet, ModelConfig, load, save


def test_each_feature_has_its_own_rows_in_the_shared_table() -> None:
    model: FareNet = FareNet(ModelConfig(["a", "b", "c"], ["x"], {"a": 3, "b": 5, "c": 2}))
    assert model.offsets.tolist() == [0, 3, 8]
    assert model.embedding.num_embeddings == 10
    assert model.embedding.embedding_dim == 8
    assert [layer.out_features for layer in model.hidden if isinstance(layer, torch.nn.Linear)] == [128, 64, 32]
    assert model.output.in_features == 32


def test_the_configuration_is_checked() -> None:
    with pytest.raises(ValueError, match="target"):
        ModelConfig(["a"], ["x"], {"a": 2}, target="cents")
    with pytest.raises(ValueError, match="vocabulary"):
        ModelConfig(["a", "b"], ["x"], {"a": 2})


def test_fares_undo_the_target_transform() -> None:
    output: torch.Tensor = torch.tensor([0.0, 2.5])
    assert FareNet(ModelConfig(["a"], ["x"], {"a": 2})).fares(output).tolist() == [0.0, 2.5]
    logged: FareNet = FareNet(ModelConfig(["a"], ["x"], {"a": 2}, target="log1p"))
    assert logged.fares(output).tolist() == pytest.approx([0.0, math.expm1(2.5)])


def test_a_saved_network_loads_with_the_same_outputs(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model: FareNet = FareNet(ModelConfig(["a", "b"], ["x"], {"a": 3, "b": 4}, target="log1p"))
    model.eval()
    save(model, tmp_path, {"note": 1})
    loaded, document = load(tmp_path)
    assert document["note"] == 1
    assert document["config"]["vocab_sizes"] == {"a": 3, "b": 4}
    assert loaded.config == model.config
    assert not loaded.training
    categories: torch.Tensor = torch.tensor([[0, 3], [2, 1]])
    numbers: torch.Tensor = torch.tensor([[0.5], [-1.0]])
    with torch.inference_mode():
        assert torch.equal(loaded(categories, numbers), model(categories, numbers))


def test_each_feature_looks_up_its_own_rows() -> None:
    model: FareNet = FareNet(ModelConfig(["a", "b", "c"], ["x"], {"a": 3, "b": 5, "c": 2}))
    model.eval()
    looked_up: list[torch.Tensor] = []
    model.embedding.register_forward_hook(lambda module, inputs, output: looked_up.append(inputs[0]))
    with torch.inference_mode():
        model(torch.tensor([[2, 4, 1], [0, 0, 0]]), torch.tensor([[0.5], [-1.0]]))
    assert looked_up[0].tolist() == [[2, 7, 9], [0, 3, 8]]


def test_a_configuration_from_other_code_is_refused(tmp_path: Path) -> None:
    save(FareNet(ModelConfig(["a"], ["x"], {"a": 3})), tmp_path, {})
    document: dict[str, Any] = json.loads((tmp_path / "model.json").read_text())
    document["config"]["embedding_dims"] = {"a": 2}  # what the first version of the network saved
    (tmp_path / "model.json").write_text(json.dumps(document))
    with pytest.raises(ValueError, match="no configuration this code can build"):
        load(tmp_path)
    (tmp_path / "model.json").write_text("{}")
    with pytest.raises(ValueError, match="no configuration this code can build"):
        load(tmp_path)


def test_weights_trained_with_other_vocabularies_are_refused(tmp_path: Path) -> None:
    # The same number of rows in all, split differently between the features.
    save(FareNet(ModelConfig(["a", "b"], ["x"], {"a": 3, "b": 5})), tmp_path, {})
    document: dict[str, Any] = json.loads((tmp_path / "model.json").read_text())
    document["config"]["vocab_sizes"] = {"a": 5, "b": 3}
    (tmp_path / "model.json").write_text(json.dumps(document))
    with pytest.raises(ValueError, match="not trained with the vocabularies"):
        load(tmp_path)
