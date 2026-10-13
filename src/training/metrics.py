from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support


def compute_classification_metrics(y_true: list[int], y_pred: list[int], labels: list[str]) -> dict[str, Any]:
    """Compute sentiment classification metrics in serializable form."""
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    _, _, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    report = classification_report(y_true, y_pred, target_names=labels, zero_division=0, output_dict=True)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(labels)))).astype(int).tolist(),
        "per_class": report,
    }


def class_distribution(labels: list[str]) -> dict[str, int]:
    values, counts = np.unique(labels, return_counts=True)
    return {str(value): int(count) for value, count in zip(values, counts)}
