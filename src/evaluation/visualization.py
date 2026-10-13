from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import matplotlib.pyplot as plt
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
