from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.evaluation.visualization import save_final_figures
from src.training.metrics import compute_classification_metrics


MODELS = ("text", "image", "multimodal")
LABELS = ["positive", "neutral", "negative"]
DEFAULT_RUNS = {
    "text": Path("checkpoints/text-baseline"),
    "image": Path("checkpoints/image-subset-baseline"),
    "multimodal": Path("checkpoints/multimodal-subset-baseline"),
}
DEFAULT_EVALUATIONS = {
    "text": Path("results/text-baseline-evaluation"),
    "image": Path("results/image-subset-evaluation"),
    "multimodal": Path("results/multimodal-subset-evaluation"),
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _model_report(name: str, run_dir: Path, evaluation_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run = _read_json(run_dir / f"{name}_run.json")
    history = _read_json(run_dir / f"{name}_history.json")
    if not isinstance(history, list):
        history = []
    metrics = _read_json(evaluation_dir / "metrics" / f"{name}_results.json")
    checkpoint_name = run.get("best_checkpoint", f"{name}_best.pt")
    checkpoint = run_dir / checkpoint_name
    reason = None
    if not run:
        reason = "run_metadata_missing"
    elif not checkpoint.is_file():
        reason = "best_checkpoint_missing"
    elif metrics.get("status") != "evaluated":
        reason = metrics.get("reason", "held_out_evaluation_missing")
    elif metrics.get("class_order") != LABELS:
        reason = "class_order_mismatch"

    result = dict(metrics)
    saved_config = run.get("config", {})
    image_config = saved_config.get("image_model", {})
    text_config = saved_config.get("text_model", {})
    initialization = run.get("pretrained_initialization") or {
        "text_encoder": text_config.get("model_name") if name in {"text", "multimodal"} else None,
        "image_encoder": (
            "ResNet50_Weights.IMAGENET1K_V2" if image_config.get("pretrained") else "random_initialization"
        ) if name in {"image", "multimodal"} else None,
        "load_pretrained_requested": True,
        "inferred_from_saved_config": True,
    }
    source_status = result.get("status")
    result.update(
        {
            "model": name,
            "status": "evaluated" if reason is None else "not_evaluated",
            "reason": reason,
            "source_evaluation_status": source_status,
            "checkpoint": str(checkpoint) if checkpoint.is_file() else None,
            "training_sample_count": run.get("training_sample_count"),
            "validation_sample_count": run.get("validation_sample_count"),
            "epochs_completed": len(history),
            "best_epoch": run.get("best_epoch"),
            "best_validation_macro_f1": run.get("best_validation_macro_f1"),
            "training_duration_seconds": run.get("training_duration_seconds"),
            "pretrained_initialization": initialization,
            "parameter_count": None,
            "inference_latency_ms": None,
            "profiling_status": "not_measured",
        }
    )
    predicted_labels = []
    prediction_path = evaluation_dir / "predictions" / f"{name}_predictions.csv"
    if prediction_path.is_file():
        prediction_frame = pd.read_csv(prediction_path)
        if "predicted_label" in prediction_frame:
            predicted_labels = prediction_frame["predicted_label"].dropna().astype(str).unique().tolist()
    result["predicted_class_count"] = len(predicted_labels) if predicted_labels else None
    result["single_class_prediction_collapse"] = len(predicted_labels) == 1 if predicted_labels else None
    result["prediction_source"] = str(prediction_path) if prediction_path.is_file() else None
    if reason is not None:
        for key in (
            "accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted",
            "per_class", "confusion_matrix", "test_samples_evaluated",
        ):
            result[key] = None
    return result, history


def build_report(
    *,
    run_dirs: dict[str, Path],
    evaluation_dirs: dict[str, Path],
    split_path: Path,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    if not split_path.is_file():
        raise FileNotFoundError(f"Prepared split data is missing: {split_path}")
    frame = pd.read_csv(split_path)
    if not {"split", "sentiment"}.issubset(frame.columns):
        raise ValueError("Prepared split data must contain split and sentiment columns")
    test_labels = frame.loc[frame["split"].astype(str).str.lower().eq("test"), "sentiment"].astype(str).str.lower()
    if test_labels.empty:
        raise ValueError("Prepared split data has no held-out test labels")
    unexpected = sorted(set(test_labels) - set(LABELS))
    if unexpected:
        raise ValueError(f"Unexpected test sentiment labels: {unexpected}")

    reports: dict[str, Any] = {}
    histories: dict[str, list[dict[str, Any]]] = {}
    for name in MODELS:
        reports[name], histories[name] = _model_report(name, run_dirs[name], evaluation_dirs[name])

    distribution = {label: int((test_labels == label).sum()) for label in LABELS}
    majority_label = max(LABELS, key=lambda label: distribution[label])
    baseline_predictions = [LABELS.index(majority_label)] * len(test_labels)
    baseline_truth = [LABELS.index(label) for label in test_labels]
    baseline_metrics = compute_classification_metrics(baseline_truth, baseline_predictions, LABELS)
    baseline = {
        **baseline_metrics,
        "model": "majority_class_baseline",
        "status": "evaluated",
        "target_class": majority_label,
        "test_samples_evaluated": len(test_labels),
        "class_order": LABELS,
    }
    dataset = {
        "prepared_split_path": str(split_path),
        "sample_count": int(len(frame)),
        "test_sample_count": int(len(test_labels)),
        "test_class_distribution": distribution,
        "label_order": LABELS,
        "test_protocol": "Existing saved split; no split regeneration or test-driven tuning.",
    }
    return {"dataset": dataset, "majority_class_baseline": baseline, "models": reports}, histories


def write_report(report: dict[str, Any], histories: dict[str, list[dict[str, Any]]], output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing report files in {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}-", dir=output_dir.parent) as temporary:
        staging = Path(temporary) / output_dir.name
        staging.mkdir()
        metrics_dir = staging / "metrics"
        figures_dir = staging / "figures"
        predictions_dir = staging / "predictions"
        for directory in (metrics_dir, figures_dir, predictions_dir):
            directory.mkdir(parents=True, exist_ok=True)

        with (metrics_dir / "experiment_summary.json").open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)
        rows = []
        for name in (*MODELS, "majority_class_baseline"):
            result = report["models"].get(name) if name != "majority_class_baseline" else report["majority_class_baseline"]
            rows.append(
                {
                    "model": name,
                    "status": result.get("status"),
                    "reason": result.get("reason"),
                    "training_samples": result.get("training_sample_count"),
                    "validation_samples": result.get("validation_sample_count"),
                    "best_epoch": result.get("best_epoch"),
                    "training_duration_seconds": result.get("training_duration_seconds"),
                    "test_samples": result.get("test_samples_evaluated"),
                    "accuracy": result.get("accuracy"),
                    "precision_macro": result.get("precision_macro"),
                    "recall_macro": result.get("recall_macro"),
                    "f1_macro": result.get("f1_macro"),
                    "f1_weighted": result.get("f1_weighted"),
                    "single_class_prediction_collapse": result.get("single_class_prediction_collapse"),
                }
            )
            if name in MODELS:
                with (metrics_dir / f"{name}_results.json").open("w", encoding="utf-8") as file:
                    json.dump(result, file, indent=2)
                source_predictions = result.get("prediction_source")
                if source_predictions and Path(source_predictions).is_file():
                    shutil.copy2(source_predictions, predictions_dir / f"{name}_predictions.csv")
        pd.DataFrame(rows).to_csv(metrics_dir / "model_comparison.csv", index=False)
        save_final_figures(
            histories,
            {**report["models"], "majority_class_baseline": report["majority_class_baseline"]},
            LABELS,
            figures_dir,
        )
        if output_dir.exists():
            output_dir.rmdir()
        staging.rename(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize saved experiments without retraining or reevaluating models.")
    parser.add_argument("--text-run-dir", type=Path, default=DEFAULT_RUNS["text"])
    parser.add_argument("--image-run-dir", type=Path, default=DEFAULT_RUNS["image"])
    parser.add_argument("--multimodal-run-dir", type=Path, default=DEFAULT_RUNS["multimodal"])
    parser.add_argument("--text-evaluation-dir", type=Path, default=DEFAULT_EVALUATIONS["text"])
    parser.add_argument("--image-evaluation-dir", type=Path, default=DEFAULT_EVALUATIONS["image"])
    parser.add_argument("--multimodal-evaluation-dir", type=Path, default=DEFAULT_EVALUATIONS["multimodal"])
    parser.add_argument("--splits", type=Path, default=Path("data/processed/splits.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/final"))
    args = parser.parse_args()
    run_dirs = {name: getattr(args, f"{name}_run_dir") for name in MODELS}
    evaluation_dirs = {name: getattr(args, f"{name}_evaluation_dir") for name in MODELS}
    report, histories = build_report(run_dirs=run_dirs, evaluation_dirs=evaluation_dirs, split_path=args.splits)
    write_report(report, histories, args.output_dir)
    print(f"Wrote final report to {args.output_dir}")


if __name__ == "__main__":
    main()
