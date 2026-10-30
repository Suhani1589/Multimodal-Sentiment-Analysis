from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset


def build_eval_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
) -> DataLoader:
    """Build an ordered evaluation loader without consuming the model's global RNG stream."""
    generator = torch.Generator()
    generator.manual_seed(seed + 1)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        generator=generator,
        num_workers=num_workers,
    )
