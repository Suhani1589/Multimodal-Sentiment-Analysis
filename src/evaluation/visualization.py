from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay


def save_class_distribution(df: pd.DataFrame, output_path: Union[str, Path]) -> None:
    counts = df["sentiment"].value_counts().sort_index()
    counts.plot(kind="bar", title="Sentiment Class Distribution")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def save_training_curves(history: list[dict[str, Any]], output_path: Union[str, Path]) -> None:
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train"]["loss"] for row in history]
    val_loss = [row["val"]["loss"] for row in history]
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def save_confusion_matrix(matrix: list[list[int]], labels: list[str], output_path: Union[str, Path]) -> None:
    ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels).plot(values_format="d")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def save_final_figures(
    histories: dict[str, list[dict[str, Any]]],
    metrics_by_model: dict[str, dict[str, Any]],
    labels: list[str],
    output_dir: Union[str, Path],
) -> None:
    """Write comparison curves and test confusion matrices without filling missing runs."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    names = ["text", "image", "multimodal"]
    titles = {"text": "Text only", "image": "Image only", "multimodal": "Multimodal"}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for name in names:
        history = histories.get(name, [])
        if not history:
            continue
        epochs = [item["epoch"] for item in history]
        axes[0].plot(epochs, [item["train"]["loss"] for item in history], marker="o", label=f"{titles[name]} train")
        axes[0].plot(epochs, [item["val"]["loss"] for item in history], marker=".", linestyle="--", label=f"{titles[name]} val")
        axes[1].plot(epochs, [item["val"]["f1_macro"] for item in history], marker="o", label=titles[name])
    axes[0].set(title="Training and validation loss", xlabel="Epoch", ylabel="Cross-entropy loss")
    axes[1].set(title="Validation macro-F1", xlabel="Epoch", ylabel="Macro-F1", ylim=(0, 1))
    for axis in axes:
        axis.grid(alpha=0.25)
        if axis.lines:
            axis.legend(fontsize=8)
    fig.savefig(output / "training_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), constrained_layout=True)
    for axis, name in zip(axes, names):
        result = metrics_by_model.get(name, {})
        matrix = result.get("confusion_matrix") if result.get("status") == "evaluated" else None
        if matrix:
            display = ConfusionMatrixDisplay(confusion_matrix=np.asarray(matrix), display_labels=labels)
            display.plot(ax=axis, values_format="d", colorbar=False, cmap="Blues")
            axis.set_title(titles[name])
            axis.set_xlabel("Predicted")
            axis.set_ylabel("Actual")
        else:
            axis.set_axis_off()
            axis.text(0.5, 0.5, f"{titles[name]}\nnot evaluated", ha="center", va="center")
    fig.savefig(output / "test_confusion_matrices.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    metric_names = ["accuracy", "f1_macro", "f1_weighted"]
    comparison_names = [*names, "majority_class_baseline"]
    comparison_titles = ["Text only", "Image only", "Multimodal", "Majority baseline"]
    positions = list(range(len(comparison_names)))
    width = 0.24
    for index, metric_name in enumerate(metric_names):
        values = [
            metrics_by_model.get(name, {}).get(metric_name)
            if metrics_by_model.get(name, {}).get("status") == "evaluated"
            else float("nan")
            for name in comparison_names
        ]
        axis.bar([p + (index - 1) * width for p in positions], values, width, label=metric_name.replace("_", " "))
    axis.set_xticks(positions, comparison_titles)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("Held-out test comparison")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.savefig(output / "test_metric_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
