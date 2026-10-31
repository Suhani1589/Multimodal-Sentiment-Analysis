from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from src.training.metrics import compute_classification_metrics


LABELS = ["positive", "neutral", "negative"]
# Validation macro-F1 of a classifier that predicts positive for every row
# of the text-usable validation split used by the earlier full-split runs.
CONSTANT_POSITIVE_MACRO_F1 = 0.24713682941531043


def select_probe_splits(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return text-usable train and validation rows.

    Held-out test rows are dropped before any representation is computed.
    """
    if "split" not in frame.columns:
        raise ValueError("Prepared data has no split column")
    required = {"sample_id", "text", "sentiment", "split"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Prepared data is missing columns: {sorted(missing)}")
    labeled = frame.loc[frame["split"].astype(str).str.lower().isin(["train", "val"])].copy()
    has_text = labeled["text"].notna() & labeled["text"].astype(str).str.strip().ne("")
    labeled = labeled.loc[has_text].copy()
    train = labeled.loc[labeled["split"].astype(str).str.lower().eq("train")].reset_index(drop=True)
    validation = labeled.loc[labeled["split"].astype(str).str.lower().eq("val")].reset_index(drop=True)
    if train.empty or validation.empty:
        raise ValueError("Training and validation splits must both contain usable text")
    used = set(train["split"].astype(str).str.lower()).union(validation["split"].astype(str).str.lower())
    if used != {"train", "val"}:
        raise ValueError(f"Probe rows must be train and validation only, found {sorted(used)}")
    return train, validation


def masked_mean_pool(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Average token states at positions the attention mask keeps, including the CLS token."""
    mask = attention_mask.unsqueeze(-1).to(dtype=hidden_state.dtype)
    summed = (hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


def constant_positive_metrics(y_true: list[int], labels: list[str] | None = None) -> dict[str, Any]:
    """Score a classifier that predicts the positive class for every row."""
    label_names = list(labels or LABELS)
    positive_index = label_names.index("positive")
    metrics = compute_classification_metrics(y_true, [positive_index] * len(y_true), label_names)
    metrics["model"] = "constant_positive_baseline"
    return metrics


def clears_constant_baseline(
    metrics: dict[str, Any],
    baseline_macro_f1: float = CONSTANT_POSITIVE_MACRO_F1,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """A probe is useful only when it beats the constant-positive score and uses every class."""
    label_names = list(labels or LABELS)
    predicted = metrics.get("predicted_class_distribution") or {}
    classes_predicted = [label for label in label_names if int(predicted.get(label, 0)) > 0]
    exceeds = float(metrics["f1_macro"]) > float(baseline_macro_f1)
    predicts_all = len(classes_predicted) == len(label_names)
    return {
        "predicts_all_classes": predicts_all,
        "classes_predicted": classes_predicted,
        "macro_f1_exceeds_constant_positive_baseline": exceeds,
        "constant_positive_macro_f1": float(baseline_macro_f1),
        "clears_gate": predicts_all and exceeds,
    }


def representation_separation(features: torch.Tensor, labels: torch.Tensor, n_classes: int = 3) -> dict[str, float]:
    """Describe whether class centroids are farther apart than examples of the same class."""
    values = features.detach().float().cpu()
    targets = labels.detach().cpu().long()
    normalized = F.normalize(values, dim=-1)
    gram = normalized @ normalized.T
    same = targets.unsqueeze(0) == targets.unsqueeze(1)
    diagonal = torch.eye(values.shape[0], dtype=torch.bool)
    within = gram[same & ~diagonal]
    between = gram[~same]
    centroids = []
    within_distances = []
    for class_index in range(n_classes):
        class_rows = values[targets == class_index]
        centroid = class_rows.mean(dim=0)
        centroids.append(centroid)
        within_distances.append(float((class_rows - centroid).norm(dim=-1).mean()))
    centroid_matrix = torch.stack(centroids)
    between_distances = [
        float((centroid_matrix[i] - centroid_matrix[j]).norm())
        for i in range(n_classes)
        for j in range(i + 1, n_classes)
    ]
    return {
        "feature_dimension_std_mean": float(values.std(dim=0).mean()),
        "feature_norm_mean": float(values.norm(dim=-1).mean()),
        "cosine_within_class": float(within.mean()) if within.numel() else float("nan"),
        "cosine_between_class": float(between.mean()) if between.numel() else float("nan"),
        "mean_within_class_l2": float(sum(within_distances) / len(within_distances)),
        "mean_centroid_l2": float(sum(between_distances) / len(between_distances)),
    }


def load_probe_classifier(
    classifier: nn.Module,
    checkpoint_path: Union[str, Path],
    expected_labels: list[str],
) -> dict[str, Any]:
    """Load a frozen-probe linear head into a text classifier and verify the copy."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Probe head not found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    labels = [str(label) for label in payload.get("labels", [])]
    if labels != [str(label) for label in expected_labels]:
        raise ValueError(f"Probe labels {labels} do not match {expected_labels}")
    if payload.get("representation") != "masked_mean":
        raise ValueError("Probe head was not trained on masked-mean representations")
    linear = classifier[-1] if isinstance(classifier, nn.Sequential) else classifier
    if not isinstance(linear, nn.Linear):
        raise TypeError("Text classifier does not end in a linear layer")
    state = payload["state_dict"]
    expected_weight = tuple(linear.weight.shape)
    expected_bias = tuple(linear.bias.shape)
    if tuple(state["weight"].shape) != expected_weight or tuple(state["bias"].shape) != expected_bias:
        raise ValueError(
            f"Probe head shape {tuple(state['weight'].shape)} does not match classifier {expected_weight}"
        )
    linear.load_state_dict({key: state[key].detach().cpu() for key in ("weight", "bias")}, strict=True)
    if not torch.equal(linear.weight.detach().cpu(), state["weight"].detach().cpu()):
        raise RuntimeError("Probe weight did not load into the classifier")
    if not torch.equal(linear.bias.detach().cpu(), state["bias"].detach().cpu()):
        raise RuntimeError("Probe bias did not load into the classifier")
    return {
        "checkpoint": str(path),
        "representation": payload["representation"],
        "labels": labels,
        "best_epoch": payload.get("best_epoch"),
        "loaded_into": "classifier.linear",
        "weight_shape": list(expected_weight),
    }


def nearest_centroid_accuracy(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    validation_features: torch.Tensor,
    validation_labels: torch.Tensor,
    n_classes: int = 3,
) -> float:
    """Classify validation rows by the nearest training-class centroid."""
    centroids = []
    for class_index in range(n_classes):
        class_rows = train_features[train_labels == class_index]
        if class_rows.numel() == 0:
            raise ValueError(f"Nearest-centroid diagnostic has no training rows for class {class_index}")
        centroids.append(class_rows.float().mean(dim=0))
    distances = torch.cdist(validation_features.float(), torch.stack(centroids))
    predicted = distances.argmin(dim=-1)
    return float((predicted.cpu() == validation_labels.cpu().long()).float().mean())
