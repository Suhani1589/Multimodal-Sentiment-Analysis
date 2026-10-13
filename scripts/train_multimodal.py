from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import ensure_dirs, load_config
from src.data.dataset import MultimodalSentimentDataset, SentimentLabels
from src.multimodal.model import MultimodalSentimentModel
from src.text.tokenizer import build_tokenizer
from src.training.losses import build_cross_entropy_loss
from src.training.trainer import Trainer
from src.utils import get_device, seed_everything
from src.vision.transforms import build_image_transform


def main() -> None:
    config = load_config()
    ensure_dirs(config)
    seed_everything(config["project"]["seed"])
    df = pd.read_csv(config["dataset"]["prepared_csv"])
    labels = SentimentLabels(config["dataset"]["label_order"])
    tokenizer = build_tokenizer(config["text_model"]["model_name"])
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    train_ds = MultimodalSentimentDataset(
        train_df,
        raw_dir=config["dataset"]["raw_dir"],
        labels=labels,
        tokenizer=tokenizer,
        image_transform=build_image_transform(config["preprocessing"]["image_size"], train=True),
        max_length=config["preprocessing"]["max_text_length"],
        mode="multimodal",
    )
    val_ds = MultimodalSentimentDataset(
        val_df,
        raw_dir=config["dataset"]["raw_dir"],
        labels=labels,
        tokenizer=tokenizer,
        image_transform=build_image_transform(config["preprocessing"]["image_size"], train=False),
        max_length=config["preprocessing"]["max_text_length"],
        mode="multimodal",
    )
    train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["training"]["batch_size"])
    model = MultimodalSentimentModel(
        text_model_name=config["text_model"]["model_name"],
        image_backbone=config["image_model"]["backbone"],
        pretrained_image=config["image_model"]["pretrained"],
        num_classes=len(labels.labels),
        projection_dim=config["multimodal_model"]["projection_dim"],
        hidden_dim=config["multimodal_model"]["hidden_dim"],
        dropout=config["multimodal_model"]["dropout"],
        freeze_text_encoder=config["text_model"]["freeze_encoder"],
        freeze_image_encoder=config["image_model"]["freeze_encoder"],
    )
    device = get_device(config["project"]["device"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])
    trainer = Trainer(
        model,
        optimizer=optimizer,
        criterion=build_cross_entropy_loss(device=device),
        device=device,
        labels=labels.labels,
        checkpoint_dir=config["paths"]["checkpoints_dir"],
        model_name="multimodal",
        patience=config["training"]["early_stopping_patience"],
    )
    trainer.fit(train_loader, val_loader, config["training"]["epochs"])


if __name__ == "__main__":
    main()
