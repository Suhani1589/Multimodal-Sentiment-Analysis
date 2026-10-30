from __future__ import annotations

import torch
from torch import nn
from typing import Optional
from typing import Any

from src.multimodal.fusion import ConcatenationFusion
from src.text.encoder import TransformerTextEncoder
from src.vision.encoder import CNNImageEncoder


class MultimodalSentimentModel(nn.Module):
    """Text-image sentiment classifier using simple concatenation fusion."""

    def __init__(
        self,
        text_model_name: str,
        image_backbone: str = "resnet50",
        pretrained_image: bool = True,
        num_classes: int = 3,
        projection_dim: int = 256,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        freeze_text_encoder: bool = False,
        freeze_image_encoder: bool = False,
        text_encoder: Optional[nn.Module] = None,
        image_encoder: Optional[nn.Module] = None,
        text_dim: Optional[int] = None,
        image_dim: Optional[int] = None,
        pretrained_text: bool = True,
        text_encoder_config: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.text_encoder = text_encoder or TransformerTextEncoder(
            text_model_name,
            freeze_encoder=freeze_text_encoder,
            pretrained=pretrained_text,
            model_config=text_encoder_config,
        )
        self.image_encoder = image_encoder or CNNImageEncoder(
            image_backbone, pretrained=pretrained_image, freeze_encoder=freeze_image_encoder
        )
        resolved_text_dim = text_dim or getattr(self.text_encoder, "hidden_size")
        resolved_image_dim = image_dim or getattr(self.image_encoder, "output_dim")
        self.fusion = ConcatenationFusion(resolved_text_dim, resolved_image_dim, projection_dim, hidden_dim, dropout)
        self.classifier = nn.Linear(self.fusion.output_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, image: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        text_features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        image_features = self.image_encoder(image)
        fused = self.fusion(text_features, image_features)
        return self.classifier(fused)
