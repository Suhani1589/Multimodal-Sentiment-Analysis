from __future__ import annotations

from typing import Optional, Union


def simple_token_importance(text: str, probabilities: Optional[list[float]] = None) -> list[dict[str, Union[float, str]]]:
    """A lightweight presentation fallback until gradient attribution is enabled."""
    tokens = text.split()
    if not tokens:
        return []
    base_score = max(probabilities) if probabilities else 1.0
    return [{"token": token, "score": float(base_score / len(tokens))} for token in tokens]
