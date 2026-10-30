from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader

from src.data.dataset import MultimodalSentimentDataset, SentimentLabels
from src.model_factory import build_model
from src.text.tokenizer import build_tokenizer
from src.training.metrics import compute_classification_metrics
from src.utils import get_device, seed_everything
from src.vision.transforms import build_image_transform


MODEL_TITLES = {"text": "Text-only", "image": "Image-only", "multimodal": "Multimodal"}


def held_out_test_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    if "split" not in dataframe.columns:
        raise ValueError("Prepared dataset has no split column")
    return dataframe.loc[dataframe["split"].astype(str).str.lower().eq("test")].copy()


def select_evaluation_rows(
    dataframe: pd.DataFrame, *, model_type: str, raw_dir: str | Path
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Keep usable rows and record every modality-specific skip reason."""
    rows: list[int] = []
    skipped: list[dict[str, str]] = []
    root = Path(raw_dir)
    for index, row in dataframe.iterrows():
        reasons: list[str] = []
        if model_type in {"text", "multimodal"} and (
            pd.isna(row.get("text")) or not str(row.get("text")).strip()
        ):
            reasons.append("missing_text")
        if model_type in {"image", "multimodal"}:
            image_value = row.get("image_path")
            if pd.isna(image_value) or not str(image_value).strip():
                reasons.append("missing_image_path")
            else:
                image_path = Path(str(image_value))
                if not image_path.is_absolute():
                    image_path = root / image_path
                try:
                    with Image.open(image_path) as image:
                        image.verify()
                except FileNotFoundError:
                    reasons.append("missing_image")
                except (UnidentifiedImageError, OSError):
                    reasons.append("unreadable_image")
        if reasons:
            skipped.append(
                {"sample_id": str(row.get("sample_id", index)), "reason": ";".join(reasons)}
            )
        else:
            rows.append(index)
    return dataframe.loc[rows].copy(), skipped


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # Older supported PyTorch releases do not expose weights_only.
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported checkpoint format: {path}")
    if "model_state_dict" not in payload:
        payload = {"model_state_dict": payload}
    return payload


@torch.inference_mode()
def evaluate_checkpoint(
    model_type: str,
    checkpoint_path: str | Path,
    test_frame: pd.DataFrame,
    *,
    runtime_config: dict[str, Any],
    device: torch.device | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    checkpoint_path = Path(checkpoint_path)
    device = device or get_device(runtime_config["project"]["device"])
    payload = _load_checkpoint(checkpoint_path, device)
    saved_config = payload.get("config", runtime_config)
    seed_everything(int(payload.get("seed", saved_config["project"]["seed"])))
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    labels = payload.get("labels", saved_config["dataset"]["label_order"])
    labels = [str(label).lower() for label in labels]
    expected_labels = [str(label).lower() for label in saved_config["dataset"]["label_order"]]
    if expected_labels != ["positive", "neutral", "negative"]:
        raise ValueError("Evaluation requires fixed label order: positive, neutral, negative")
    if labels != expected_labels:
        raise ValueError(f"Checkpoint label order {labels} does not match saved config {expected_labels}")

    usable, skipped = select_evaluation_rows(
        test_frame, model_type=model_type, raw_dir=runtime_config["dataset"]["raw_dir"]
    )
    if usable.empty:
        raise ValueError(f"No usable test samples for {model_type}; skipped {len(skipped)} rows")
    tokenizer = None
    if model_type != "image":
        tokenizer_path = payload.get("tokenizer_path")
        local_tokenizer = checkpoint_path.parent / tokenizer_path if tokenizer_path else None
        tokenizer = build_tokenizer(str(local_tokenizer)) if local_tokenizer and local_tokenizer.exists() else build_tokenizer(saved_config["text_model"]["model_name"])
    transform = (
        build_image_transform(saved_config["preprocessing"]["image_size"], train=False)
        if model_type in {"image", "multimodal"}
        else None
    )
    dataset = MultimodalSentimentDataset(
        usable,
        raw_dir=runtime_config["dataset"]["raw_dir"],
        labels=SentimentLabels(labels),
        tokenizer=tokenizer,
        image_transform=transform,
        max_length=int(saved_config["preprocessing"]["max_text_length"]),
        mode=model_type,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(saved_config["training"]["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    model = build_model(
        model_type,
        saved_config,
        load_pretrained=False,
        text_encoder_config=payload.get("model_config", {}).get("text_encoder_config"),
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    true_ids: list[int] = []
    pred_ids: list[int] = []
    probability_rows: list[list[float]] = []
    sample_ids: list[str] = []
    for batch in loader:
        tensor_batch = {
            key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()
        }
        logits = model(**tensor_batch)
        probabilities = torch.softmax(logits, dim=-1).cpu()
        predictions = probabilities.argmax(dim=-1)
        true_ids.extend(batch["label"].tolist())
        pred_ids.extend(predictions.tolist())
        probability_rows.extend(probabilities.tolist())
        sample_ids.extend(str(value) for value in batch["sample_id"])

    metrics = compute_classification_metrics(true_ids, pred_ids, labels)
    metrics.update(
        {
            "status": "evaluated",
            "model": model_type,
            "checkpoint": str(checkpoint_path),
            "test_samples_evaluated": len(true_ids),
            "test_samples_skipped": len(skipped),
            "skipped_samples": skipped,
            "class_order": labels,
        }
    )
    prediction_frame = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "true_label": [labels[index] for index in true_ids],
            "predicted_label": [labels[index] for index in pred_ids],
        }
    )
    for class_index, label in enumerate(labels):
        prediction_frame[f"probability_{label}"] = [row[class_index] for row in probability_rows]
    return metrics, prediction_frame


def save_evaluation_outputs(
    model_type: str,
    metrics: dict[str, Any],
    predictions: pd.DataFrame,
    *,
    metrics_dir: str | Path,
    predictions_dir: str | Path,
) -> None:
    metrics_output = Path(metrics_dir)
    prediction_output = Path(predictions_dir)
    metrics_output.mkdir(parents=True, exist_ok=True)
    prediction_output.mkdir(parents=True, exist_ok=True)
    with (metrics_output / f"{model_type}_results.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)
    predictions.to_csv(prediction_output / f"{model_type}_predictions.csv", index=False)
    confusion = pd.DataFrame(
        metrics["confusion_matrix"],
        index=[f"true_{label}" for label in metrics["class_order"]],
        columns=[f"predicted_{label}" for label in metrics["class_order"]],
    )
    confusion.to_csv(metrics_output / f"{model_type}_confusion_matrix.csv")
