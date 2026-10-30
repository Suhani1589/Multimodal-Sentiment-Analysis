from __future__ import annotations

from typing import Any, Optional

import torch

from src.explainability.attribution import token_gradient_attribution


def explain_text_prediction(
    model: torch.nn.Module,
    tokenizer: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    image: Optional[torch.Tensor] = None,
    target_class: Optional[int] = None,
) -> dict[str, Any]:
    """Return token-level gradient-times-embedding saliency, not natural-language rationale."""
    result = token_gradient_attribution(
        model,
        input_ids,
        attention_mask,
        image=image,
        target_class=target_class,
    )
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].detach().cpu().tolist())
    keep = attention_mask[0].detach().cpu().tolist()
    special_tokens = set(getattr(tokenizer, "all_special_tokens", []))
    result["tokens"] = [
        {"token": token, "score": score}
        for token, score, active in zip(tokens, result["scores"], keep)
        if active and token not in special_tokens
    ]
    return result
