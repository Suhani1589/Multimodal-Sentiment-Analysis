from __future__ import annotations

import torch
from torch import nn
from typing import Optional

from src.vision.encoder import CNNImageEncoder


class ImageSentimentModel(nn.Module):
    """Image-only sentiment classifier."""

    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        num_classes: int = 3,
        dropout: float = 0.2,
        freeze_encoder: bool = False,
        encoder: Optional[nn.Module] = None,
        encoder_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder or CNNImageEncoder(backbone, pretrained=pretrained, freeze_encoder=freeze_encoder)
        output_dim = encoder_dim or getattr(self.encoder, "output_dim")
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(output_dim, num_classes))

    def forward(self, image: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        features = self.encoder(image)
        return self.classifier(features)
