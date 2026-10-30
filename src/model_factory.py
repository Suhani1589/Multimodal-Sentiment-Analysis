from __future__ import annotations

from typing import Any, Optional

from torch import nn

from src.multimodal.model import MultimodalSentimentModel
from src.text.model import TextSentimentModel
from src.vision.model import ImageSentimentModel


def build_model(
    model_type: str,
    config: dict,
    *,
    load_pretrained: bool = True,
    text_encoder_config: Optional[dict[str, Any]] = None,
) -> nn.Module:
    """Build a project model from its saved configuration."""
    labels = config["dataset"]["label_order"]
    if model_type == "text":
        return TextSentimentModel(
            config["text_model"]["model_name"],
            num_classes=len(labels),
            dropout=config["text_model"]["dropout"],
            freeze_encoder=config["text_model"]["freeze_encoder"],
            pretrained=load_pretrained,
            encoder_config=text_encoder_config,
        )
    if model_type == "image":
        return ImageSentimentModel(
            backbone=config["image_model"]["backbone"],
            pretrained=config["image_model"]["pretrained"] if load_pretrained else False,
            num_classes=len(labels),
            dropout=config["image_model"]["dropout"],
            freeze_encoder=config["image_model"]["freeze_encoder"],
        )
    if model_type == "multimodal":
        return MultimodalSentimentModel(
            text_model_name=config["text_model"]["model_name"],
            image_backbone=config["image_model"]["backbone"],
            pretrained_image=config["image_model"]["pretrained"] if load_pretrained else False,
            pretrained_text=load_pretrained,
            text_encoder_config=text_encoder_config,
            num_classes=len(labels),
            projection_dim=config["multimodal_model"]["projection_dim"],
            hidden_dim=config["multimodal_model"]["hidden_dim"],
            dropout=config["multimodal_model"]["dropout"],
            freeze_text_encoder=config["text_model"]["freeze_encoder"],
            freeze_image_encoder=config["image_model"]["freeze_encoder"],
        )
    raise ValueError(f"Unknown model type: {model_type}")
