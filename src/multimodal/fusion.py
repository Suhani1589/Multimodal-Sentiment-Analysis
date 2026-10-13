from __future__ import annotations

import torch
from torch import nn


class ConcatenationFusion(nn.Module):
    """Project text and image features, concatenate them, and fuse with an MLP."""

    def __init__(self, text_dim: int, image_dim: int, projection_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.text_projection = nn.Sequential(nn.Linear(text_dim, projection_dim), nn.ReLU(), nn.LayerNorm(projection_dim))
        self.image_projection = nn.Sequential(
            nn.Linear(image_dim, projection_dim), nn.ReLU(), nn.LayerNorm(projection_dim)
        )
        self.fusion = nn.Sequential(
            nn.Linear(projection_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.output_dim = hidden_dim

    def forward(self, text_features: torch.Tensor, image_features: torch.Tensor) -> torch.Tensor:
        text_projected = self.text_projection(text_features)
        image_projected = self.image_projection(image_features)
        return self.fusion(torch.cat([text_projected, image_projected], dim=-1))
