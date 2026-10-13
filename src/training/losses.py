from __future__ import annotations

import torch
from torch import nn
from typing import Optional


def build_cross_entropy_loss(class_weights: Optional[list[float]] = None, device: Optional[torch.device] = None) -> nn.Module:
    weights = torch.tensor(class_weights, dtype=torch.float32, device=device) if class_weights else None
    return nn.CrossEntropyLoss(weight=weights)
