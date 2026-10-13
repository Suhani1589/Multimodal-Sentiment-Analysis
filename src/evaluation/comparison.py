from __future__ import annotations

from pathlib import Path
from typing import Union

from src.evaluation.evaluate import build_comparison_table, load_metrics


def compare_metric_files(metric_files: dict[str, Union[str, Path]], output_csv: Union[str, Path]) -> None:
    metrics = {model_name: load_metrics(path) for model_name, path in metric_files.items() if Path(path).exists()}
    table = build_comparison_table(metrics)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_csv, index=False)
