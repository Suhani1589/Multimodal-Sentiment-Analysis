from __future__ import annotations

from typing import Any

import torch


@torch.no_grad()
def predict_proba(model: torch.nn.Module, batch: dict[str, Any], labels: list[str], device: torch.device) -> dict[str, Any]:
    """Run model inference and return label probabilities."""
    model.eval()
    model.to(device)
    tensor_batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
    logits = model(**tensor_batch)
    probabilities = torch.softmax(logits, dim=-1).detach().cpu()[0]
    best_index = int(probabilities.argmax().item())
    return {
        "predicted_sentiment": labels[best_index],
        "confidence": float(probabilities[best_index].item()),
        "probabilities": {label: float(probabilities[index].item()) for index, label in enumerate(labels)},
    }
