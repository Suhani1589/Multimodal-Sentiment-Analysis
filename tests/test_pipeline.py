from __future__ import annotations

from src.evaluation.evaluate import build_comparison_table
from src.training.metrics import compute_classification_metrics


def test_metrics_keys() -> None:
    metrics = compute_classification_metrics([0, 1, 2], [0, 2, 2], ["positive", "neutral", "negative"])
    assert metrics["accuracy"] == 2 / 3
    assert "f1_macro" in metrics
    assert len(metrics["confusion_matrix"]) == 3


def test_comparison_table() -> None:
    table = build_comparison_table({"Text-only": {"accuracy": 1.0, "precision_macro": 1.0, "recall_macro": 1.0, "f1_macro": 1.0, "f1_weighted": 1.0}})
    assert list(table.columns) == ["model", "accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]
    assert table.iloc[0]["model"] == "Text-only"
