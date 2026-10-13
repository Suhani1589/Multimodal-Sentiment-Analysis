from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data.dataset import MultimodalSentimentDataset, SentimentLabels


class SimpleTokenizer:
    """Small tokenizer for dataset smoke tests without downloading model weights."""

    def __call__(
        self,
        text: str,
        *,
        truncation: bool,
        padding: str,
        max_length: int,
        return_tensors: str,
    ) -> dict[str, Any]:
        import torch

        token_ids = [min(ord(char), 30_000) for char in str(text)]
        if truncation:
            token_ids = token_ids[:max_length]
        attention = [1] * len(token_ids)
        if padding == "max_length":
            pad = max_length - len(token_ids)
            token_ids = token_ids + [0] * max(pad, 0)
            attention = attention + [0] * max(pad, 0)
        return {
            "input_ids": torch.tensor([token_ids], dtype=torch.long),
            "attention_mask": torch.tensor([attention], dtype=torch.long),
        }


def basic_image_transform(image: Any) -> Any:
    import numpy as np
    import torch

    resized = image.resize((224, 224))
    array = np.asarray(resized, dtype="float32") / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def main() -> None:
    try:
        import torch  # noqa: F401
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise SystemExit(f"Dataset smoke test requires torch: {exc}") from exc

    config = load_config()
    prepared_csv = Path(config["dataset"]["prepared_csv"])
    if not prepared_csv.exists():
        raise SystemExit(f"Prepared split file is missing: {prepared_csv}. Run python3 scripts/prepare_data.py first.")

    df = pd.read_csv(prepared_csv)
    if "split" not in df.columns:
        raise SystemExit(f"Prepared split file has no split column: {prepared_csv}")
    train_df = df[df["split"] == "train"].head(max(config["training"]["batch_size"], 1))
    if train_df.empty:
        raise SystemExit("No train rows available for smoke test.")

    labels = SentimentLabels(config["dataset"]["label_order"])
    tokenizer = SimpleTokenizer()
    for mode in ("text", "image", "multimodal"):
        ds = MultimodalSentimentDataset(
            train_df,
            raw_dir=config["dataset"]["raw_dir"],
            labels=labels,
            tokenizer=tokenizer,
            image_transform=basic_image_transform,
            max_length=config["preprocessing"]["max_text_length"],
            mode=mode,
        )
        batch = next(iter(DataLoader(ds, batch_size=min(2, len(ds)))))
        keys = ", ".join(sorted(batch.keys()))
        print(f"{mode} batch ok: {keys}")


if __name__ == "__main__":
    main()
