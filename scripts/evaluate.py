from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.evaluation.comparison import compare_metric_files


def main() -> None:
    config = load_config()
    metrics_dir = Path(config["paths"]["metrics_dir"])
    compare_metric_files(
        {
            "Text-only": metrics_dir / "text_results.json",
            "Image-only": metrics_dir / "image_results.json",
            "Text + Image": metrics_dir / "multimodal_results.json",
        },
        metrics_dir / "model_comparison.csv",
    )
    print(f"Wrote comparison table to {metrics_dir / 'model_comparison.csv'}")


if __name__ == "__main__":
    main()
