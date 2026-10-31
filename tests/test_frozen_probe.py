from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import torch

from src.training.frozen_probe import (
    clears_constant_baseline,
    constant_positive_metrics,
    masked_mean_pool,
    select_probe_splits,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"sample_id": "tr", "text": "good", "sentiment": "positive", "split": "train"},
            {"sample_id": "va", "text": "ok", "sentiment": "neutral", "split": "val"},
            {"sample_id": "te", "text": "secret test text", "sentiment": "negative", "split": "test"},
            {"sample_id": "blank", "text": "  ", "sentiment": "negative", "split": "train"},
        ]
    )


def test_probe_split_keeps_train_and_validation_text_only() -> None:
    train, validation = select_probe_splits(_frame())
    assert train["sample_id"].tolist() == ["tr"]
    assert validation["sample_id"].tolist() == ["va"]
    assert "te" not in set(train["sample_id"]).union(validation["sample_id"])
    assert "blank" not in set(train["sample_id"]).union(validation["sample_id"])


def test_masked_mean_pool_ignores_padding() -> None:
    hidden = torch.tensor([[[1.0, 1.0], [3.0, 5.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    pooled = masked_mean_pool(hidden, mask)
    assert torch.allclose(pooled, torch.tensor([[2.0, 3.0]]))


def test_constant_positive_gate_rejects_a_single_class_model() -> None:
    metrics = constant_positive_metrics([0, 0, 1, 2])
    decision = clears_constant_baseline(metrics, baseline_macro_f1=0.0)
    assert decision["predicts_all_classes"] is False
    assert decision["clears_gate"] is False
    assert metrics["predicted_class_distribution"] == {"positive": 4, "neutral": 0, "negative": 0}


def test_gate_requires_both_all_classes_and_a_higher_macro_f1() -> None:
    metrics = {
        "f1_macro": 0.30,
        "predicted_class_distribution": {"positive": 3, "neutral": 2, "negative": 1},
    }
    assert clears_constant_baseline(metrics, baseline_macro_f1=0.247)["clears_gate"] is True
    assert clears_constant_baseline(metrics, baseline_macro_f1=0.30)["clears_gate"] is False
    metrics["predicted_class_distribution"]["negative"] = 0
    assert clears_constant_baseline(metrics, baseline_macro_f1=0.247)["clears_gate"] is False


def test_linear_head_separates_three_clusters_without_an_encoder() -> None:
    script_path = Path("scripts/probe_frozen_text_head.py")
    spec = importlib.util.spec_from_file_location("probe_frozen_text_head", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    torch.manual_seed(0)
    labels = ["positive", "neutral", "negative"]
    train_features = []
    train_targets = []
    train_names = []
    validation_features = []
    validation_targets = []
    for index, name in enumerate(labels):
        cluster = torch.randn(30, 4) + torch.tensor([index * 4.0, 0.0, 0.0, 0.0])
        train_features.append(cluster[:24])
        validation_features.append(cluster[24:])
        train_targets.append(torch.full((24,), index, dtype=torch.long))
        validation_targets.append(torch.full((6,), index, dtype=torch.long))
        train_names.extend([name] * 24)
    _head, _history, metrics = module.train_linear_head(
        torch.cat(train_features),
        torch.cat(train_targets),
        train_names,
        torch.cat(validation_features),
        torch.cat(validation_targets),
        learning_rate=1e-1,
        weight_decay=0.0,
        batch_size=12,
        epochs=40,
        patience=8,
        seed=0,
        device=torch.device("cpu"),
    )
    assert metrics["f1_macro"] > 0.9
    assert all(metrics["predicted_class_distribution"][label] > 0 for label in labels)


def test_masked_mean_pooling_ignores_the_pooler_output() -> None:
    from types import SimpleNamespace

    from src.text.encoder import TransformerTextEncoder

    encoder = TransformerTextEncoder.__new__(TransformerTextEncoder)
    encoder.hidden_size = 2

    class FakeEncoder:
        def __call__(self, input_ids, attention_mask):
            hidden = torch.tensor([[[1.0, 1.0], [3.0, 5.0], [100.0, 100.0]]])
            return SimpleNamespace(last_hidden_state=hidden, pooler_output=torch.full((1, 2), 99.0))

    encoder.encoder = FakeEncoder()
    tokens = torch.ones((1, 3), dtype=torch.long)
    mask = torch.tensor([[1, 1, 0]])
    encoder.pooling = "masked_mean"
    assert torch.allclose(encoder.forward(tokens, mask), torch.tensor([[2.0, 3.0]]))
    encoder.pooling = "pooler"
    assert torch.allclose(encoder.forward(tokens, mask), torch.full((1, 2), 99.0))


def test_probe_head_loads_into_dropout_classifier(tmp_path) -> None:
    from torch import nn

    from src.training.frozen_probe import load_probe_classifier

    classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(4, 3))
    weight = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    bias = torch.tensor([0.1, 0.2, 0.3])
    path = tmp_path / "head.pt"
    torch.save(
        {
            "representation": "masked_mean",
            "labels": ["positive", "neutral", "negative"],
            "state_dict": {"weight": weight, "bias": bias},
            "best_epoch": 4,
        },
        path,
    )
    info = load_probe_classifier(classifier, path, ["positive", "neutral", "negative"])
    assert info["weight_shape"] == [3, 4]
    assert torch.equal(classifier[1].weight, weight)
    assert torch.equal(classifier[1].bias, bias)
