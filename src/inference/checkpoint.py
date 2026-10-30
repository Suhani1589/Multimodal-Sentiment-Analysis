from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import torch
from PIL import Image

from src.config import load_config
from src.data.preprocessing import normalize_text
from src.model_factory import build_model
from src.text.tokenizer import build_tokenizer
from src.utils import get_device
from src.vision.transforms import build_image_transform


@dataclass
class LoadedPredictor:
    model_type: str
    model: torch.nn.Module
    labels: list[str]
    config: dict[str, Any]
    tokenizer: Optional[Any]
    device: torch.device
    checkpoint_path: Path


def load_checkpoint_predictor(
    model_type: str,
    checkpoint_path: Union[str, Path],
    *,
    fallback_config_path: Union[str, Path] = "configs/config.yaml",
) -> LoadedPredictor:
    """Reconstruct an inference model using only its saved checkpoint weights."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch versions before the weights_only argument
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"Unsupported checkpoint format: {path}")
    if payload.get("model_type") != model_type:
        raise ValueError(
            f"Checkpoint is for {payload.get('model_type')!r}, requested {model_type!r}"
        )

    config = payload.get("config") or load_config(fallback_config_path)
    labels = [str(label).lower() for label in payload.get("labels", config["dataset"]["label_order"])]
    if labels != [str(label).lower() for label in config["dataset"]["label_order"]]:
        raise ValueError("Checkpoint label order does not match its saved configuration")
    device = get_device(config["project"].get("device", "auto"))

    tokenizer = None
    if model_type in {"text", "multimodal"}:
        tokenizer_relative = payload.get("tokenizer_path")
        local_tokenizer = path.parent / tokenizer_relative if tokenizer_relative else None
        tokenizer_name = str(local_tokenizer) if local_tokenizer and local_tokenizer.is_dir() else (
            payload.get("tokenizer_name") or config["text_model"]["model_name"]
        )
        try:
            tokenizer = build_tokenizer(tokenizer_name)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load tokenizer {tokenizer_name!r}. Check that the checkpoint's tokenizer "
                "files are present or that the encoder is available in the local model cache."
            ) from exc

    model = build_model(
        model_type,
        config,
        load_pretrained=False,
        text_encoder_config=payload.get("model_config", {}).get("text_encoder_config"),
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return LoadedPredictor(model_type, model, labels, config, tokenizer, device, path)


def prepare_prediction_batch(
    predictor: LoadedPredictor,
    *,
    text: str = "",
    image: Optional[Image.Image] = None,
) -> dict[str, Any]:
    """Apply the checkpoint's training preprocessing to user supplied modalities."""
    if predictor.model_type in {"text", "multimodal"} and not text.strip():
        raise ValueError("Non-empty text is required for this model")
    if predictor.model_type in {"image", "multimodal"} and image is None:
        raise ValueError("An image is required for this model")

    config = predictor.config
    batch: dict[str, Any] = {"sample_id": "inference", "text": text, "language": "unknown"}
    if predictor.tokenizer is not None:
        preprocessing = config["preprocessing"]
        normalized = normalize_text(
            text,
            remove_urls=preprocessing["remove_urls"],
            replace_usernames=preprocessing["replace_usernames"],
            lowercase=preprocessing["lowercase"],
        )
        batch.update(
            predictor.tokenizer(
                normalized,
                truncation=True,
                padding="max_length",
                max_length=preprocessing["max_text_length"],
                return_tensors="pt",
            )
        )
    if image is not None:
        transform = build_image_transform(config["preprocessing"]["image_size"], train=False)
        batch["image"] = transform(image.convert("RGB")).unsqueeze(0)
    return batch


def predict_batch(predictor: LoadedPredictor, batch: dict[str, Any]) -> dict[str, Any]:
    tensor_batch = {
        key: value.to(predictor.device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    with torch.inference_mode():
        probabilities = torch.softmax(predictor.model(**tensor_batch), dim=-1)[0].cpu().tolist()
    best_index = max(range(len(probabilities)), key=probabilities.__getitem__)
    return {
        "predicted_sentiment": predictor.labels[best_index],
        "confidence": float(probabilities[best_index]),
        "probabilities": dict(zip(predictor.labels, map(float, probabilities))),
        "checkpoint": str(predictor.checkpoint_path),
    }
