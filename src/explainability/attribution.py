from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F


def _embedding_layer(model: torch.nn.Module) -> torch.nn.Module:
    text_encoder = getattr(model, "text_encoder", None) or getattr(model, "encoder", None)
    transformer = getattr(text_encoder, "encoder", None)
    embeddings = getattr(transformer, "embeddings", None)
    if embeddings is None:
        raise ValueError("This model does not expose a Transformer embedding layer for token attribution")
    return embeddings


def token_gradient_attribution(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    image: Optional[torch.Tensor] = None,
    target_class: Optional[int] = None,
) -> dict[str, Any]:
    """Return normalized gradient-times-embedding saliency per input token.

    For multimodal models, the supplied image is held fixed. These scores indicate
    local model sensitivity and are not human-readable reasons or causal evidence.
    """
    was_training = model.training
    model.eval()
    captured: list[torch.Tensor] = []
    layer = _embedding_layer(model)

    def capture(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        captured.append(output)

    handle = layer.register_forward_hook(capture)
    try:
        kwargs: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if image is not None:
            kwargs["image"] = image
        logits = model(**kwargs)
        if not captured:
            raise RuntimeError("Transformer embedding hook did not capture token embeddings")
        predicted = int(logits[0].argmax().item()) if target_class is None else int(target_class)
        gradients = torch.autograd.grad(logits[0, predicted], captured[0], retain_graph=False)[0]
        scores = (captured[0] * gradients).sum(dim=-1).abs()
        scores = scores * attention_mask.to(scores.dtype)
        denominator = scores.max(dim=-1, keepdim=True).values.clamp_min(torch.finfo(scores.dtype).eps)
        scores = (scores / denominator).detach().cpu()
        return {"target_class": predicted, "scores": scores[0].tolist(), "method": "gradient_times_embedding"}
    finally:
        handle.remove()
        if was_training:
            model.train()


def _last_resnet_block(model: torch.nn.Module) -> torch.nn.Module:
    image_encoder = getattr(model, "image_encoder", None) or getattr(model, "encoder", None)
    resnet = getattr(image_encoder, "encoder", None)
    layer4 = getattr(resnet, "layer4", None)
    if layer4 is None or len(layer4) == 0:
        raise ValueError("This model does not expose a ResNet layer4 block for Grad-CAM")
    return layer4[-1]


def grad_cam(
    model: torch.nn.Module,
    image: torch.Tensor,
    *,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    target_class: Optional[int] = None,
) -> dict[str, Any]:
    """Compute a normalized Grad-CAM map for ResNet-based image predictions."""
    was_training = model.training
    model.eval()
    activations: list[torch.Tensor] = []

    def capture(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        activations.append(output)

    handle = _last_resnet_block(model).register_forward_hook(capture)
    try:
        kwargs: dict[str, torch.Tensor] = {"image": image}
        if input_ids is not None:
            if attention_mask is None:
                raise ValueError("attention_mask is required with input_ids for multimodal Grad-CAM")
            kwargs.update(input_ids=input_ids, attention_mask=attention_mask)
        logits = model(**kwargs)
        if not activations:
            raise RuntimeError("ResNet hook did not capture an activation map")
        predicted = int(logits[0].argmax().item()) if target_class is None else int(target_class)
        gradients = torch.autograd.grad(logits[0, predicted], activations[0], retain_graph=False)[0]
        weights = gradients.mean(dim=(-2, -1), keepdim=True)
        heatmap = torch.relu((weights * activations[0]).sum(dim=1, keepdim=True))
        heatmap = F.interpolate(heatmap, size=image.shape[-2:], mode="bilinear", align_corners=False)
        heatmap = heatmap[0, 0]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()).clamp_min(1e-12)
        return {"target_class": predicted, "heatmap": heatmap.detach().cpu(), "method": "grad_cam"}
    finally:
        handle.remove()
        if was_training:
            model.train()
