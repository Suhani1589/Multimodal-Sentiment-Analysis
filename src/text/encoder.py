from __future__ import annotations

import torch
from torch import nn


class TransformerTextEncoder(nn.Module):
    """Pooled multilingual Transformer encoder."""

    def __init__(self, model_name: str, freeze_encoder: bool = False) -> None:
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = int(self.encoder.config.hidden_size)
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if getattr(outputs, "pooler_output", None) is not None:
            return outputs.pooler_output
        mask = attention_mask.unsqueeze(-1).float()
        summed = (outputs.last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return summed / counts
