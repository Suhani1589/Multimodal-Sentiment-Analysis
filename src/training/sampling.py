from __future__ import annotations

from collections import Counter
from typing import Optional

import torch
from torch.utils.data import WeightedRandomSampler


def build_training_sampler(
    labels: list[str],
    class_order: list[str],
    *,
    strategy: str,
    seed: int,
) -> Optional[WeightedRandomSampler]:
    """Build a reproducible balanced sampler for training rows only."""
    if strategy == "none" or strategy == "class_weights":
        return None
    if strategy != "weighted_sampler":
        raise ValueError("training.imbalance_strategy must be none, class_weights, or weighted_sampler")
    counts = Counter(str(label).strip().lower() for label in labels)
    missing = [label for label in class_order if counts.get(label, 0) == 0]
    if missing:
        raise ValueError(f"Balanced sampling requires training examples for every class; missing: {missing}")
    sample_weights = torch.tensor(
        [1.0 / counts[str(label).strip().lower()] for label in labels],
        dtype=torch.double,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(
        sample_weights,
        num_samples=len(labels),
        replacement=True,
        generator=generator,
    )
