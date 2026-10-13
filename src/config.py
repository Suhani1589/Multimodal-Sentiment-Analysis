from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import yaml


def load_config(path: Union[str, Path] = "configs/config.yaml") -> dict[str, Any]:
    """Load a YAML configuration file."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def ensure_dirs(config: dict[str, Any]) -> None:
    """Create configured output directories if they are missing."""
    for key in ["checkpoints_dir", "metrics_dir", "plots_dir", "predictions_dir", "logs_dir"]:
        Path(config["paths"][key]).mkdir(parents=True, exist_ok=True)
