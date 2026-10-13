from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class CNNImageEncoder(nn.Module):
    """Pretrained torchvision visual encoder with classifier removed."""

    def __init__(self, backbone: str = "resnet50", pretrained: bool = True, freeze_encoder: bool = False) -> None:
        super().__init__()
        if backbone != "resnet50":
            raise ValueError("Currently supported image backbone: resnet50")
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        self.output_dim = int(model.fc.in_features)
        model.fc = nn.Identity()
        self.encoder = model
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.encoder(image)
