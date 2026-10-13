from __future__ import annotations

import torch
from torch import nn
from typing import Optional

from src.text.encoder import TransformerTextEncoder


class TextSentimentModel(nn.Module):
    """Text-only sentiment classifier."""

    def __init__(
        self,
        model_name: str,
        num_classes: int = 3,
        dropout: float = 0.2,
        freeze_encoder: bool = False,
        encoder: Optional[nn.Module] = None,
        encoder_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder or TransformerTextEncoder(model_name, freeze_encoder=freeze_encoder)
        hidden_size = encoder_dim or getattr(self.encoder, "hidden_size")
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, num_classes))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        features = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(features)
