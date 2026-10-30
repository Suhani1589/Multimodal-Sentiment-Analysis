from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.evaluation.checkpoint_evaluation import (
    MODEL_TITLES,
    evaluate_checkpoint,
    held_out_test_rows,
    save_evaluation_outputs,
)
from src.evaluation.evaluate import build_comparison_table
from src.utils import get_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate checkpoints on the held-out test split.")
    parser.add_argument("--config", default="configs/config.yaml", help="YAML experiment configuration")
    parser.add_argument("--model", choices=["all", "text", "image", "multimodal"], default="all")
    parser.add_argument("--checkpoint-dir", help="Checkpoint directory (defaults to config)")
    parser.add_argument("--output-dir", help="Output root (contains metrics/ and predictions/)")
    args = parser.parse_args()

    config = load_config(args.config)
    frame = pd.read_csv(config["dataset"]["prepared_csv"])
    if "split" not in frame:
        raise ValueError("Prepared dataset has no split column; run scripts/prepare_data.py first")
    test_frame = held_out_test_rows(frame)
    if test_frame.empty:
        raise ValueError("Prepared dataset has no held-out test samples")

    checkpoint_dir = Path(args.checkpoint_dir or config["paths"]["checkpoints_dir"])
    root = Path(args.output_dir) if args.output_dir else None
    metrics_dir = root / "metrics" if root else Path(config["paths"]["metrics_dir"])
    predictions_dir = root / "predictions" if root else Path(config["paths"]["predictions_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    requested = [args.model] if args.model != "all" else ["text", "image", "multimodal"]
    comparison: dict[str, dict] = {}
    device = get_device(config["project"]["device"])
    for model_type in ["text", "image", "multimodal"]:
        if model_type not in requested:
            result = {
                "status": "not_evaluated",
                "model": model_type,
                "reason": "not_selected",
                "test_samples_evaluated": 0,
                "test_samples_skipped": 0,
                "class_order": config["dataset"]["label_order"],
            }
            comparison[MODEL_TITLES[model_type]] = result
            continue
        checkpoint = checkpoint_dir / f"{model_type}_best.pt"
        if not checkpoint.exists():
            result = {
                "status": "not_evaluated",
                "model": model_type,
                "reason": "checkpoint_missing",
                "checkpoint": str(checkpoint),
                "test_samples_evaluated": 0,
                "test_samples_skipped": 0,
                "class_order": config["dataset"]["label_order"],
            }
            with (metrics_dir / f"{model_type}_results.json").open("w", encoding="utf-8") as file:
                json.dump(result, file, indent=2)
            pd.DataFrame(
                columns=["sample_id", "true_label", "predicted_label"]
                + [f"probability_{label}" for label in config["dataset"]["label_order"]]
            ).to_csv(predictions_dir / f"{model_type}_predictions.csv", index=False)
            pd.DataFrame(
                columns=["true_label"] + [f"predicted_{label}" for label in config["dataset"]["label_order"]]
            ).to_csv(metrics_dir / f"{model_type}_confusion_matrix.csv", index=False)
        else:
            result, predictions = evaluate_checkpoint(
                model_type, checkpoint, test_frame, runtime_config=config, device=device
            )
            save_evaluation_outputs(
                model_type,
                result,
                predictions,
                metrics_dir=metrics_dir,
                predictions_dir=predictions_dir,
            )
        comparison[MODEL_TITLES[model_type]] = result
        if result["status"] == "evaluated":
            print(
                f"{model_type}: n={result['test_samples_evaluated']} "
                f"accuracy={result['accuracy']:.4f} macro_f1={result['f1_macro']:.4f}"
            )
        else:
            print(f"{model_type}: not evaluated ({result['reason']})")

    table = build_comparison_table(comparison)
    status_by_name = {name: result["status"] for name, result in comparison.items()}
    table["status"] = table["model"].map(status_by_name)
    table["test_samples_evaluated"] = table["model"].map(
        {name: result["test_samples_evaluated"] for name, result in comparison.items()}
    )
    table.to_csv(metrics_dir / "model_comparison.csv", index=False)
    print(f"Saved held-out evaluation outputs under {metrics_dir} and {predictions_dir}")


if __name__ == "__main__":
    main()
