from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

import pandas as pd


def save_metrics(metrics: dict[str, Any], path: Union[str, Path]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


def load_metrics(path: Union[str, Path]) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def build_comparison_table(metrics_by_model: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for model_name, metrics in metrics_by_model.items():
        rows.append(
            {
                "model": model_name,
                "accuracy": metrics.get("accuracy"),
                "precision_macro": metrics.get("precision_macro"),
                "recall_macro": metrics.get("recall_macro"),
                "f1_macro": metrics.get("f1_macro"),
                "f1_weighted": metrics.get("f1_weighted"),
            }
        )
    return pd.DataFrame(rows)
