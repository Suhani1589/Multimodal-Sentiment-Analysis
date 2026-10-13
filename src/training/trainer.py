from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.metrics import compute_classification_metrics


class Trainer:
    """Reusable supervised trainer with early stopping and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        *,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        labels: list[str],
        checkpoint_dir: Union[str, Path],
        model_name: str,
        patience: int = 2,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.labels = labels
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_name = model_name
        self.patience = patience
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _move_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        return {key: value.to(self.device) if torch.is_tensor(value) else value for key, value in batch.items()}

    def _forward_loss(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        labels = batch["label"]
        logits = self.model(**batch)
        loss = self.criterion(logits, labels)
        return loss, logits

    def train_epoch(self, loader: DataLoader) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for batch in tqdm(loader, desc="train", leave=False):
            batch = self._move_batch(batch)
            self.optimizer.zero_grad(set_to_none=True)
            loss, logits = self._forward_loss(batch)
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss.item()) * len(batch["label"])
            predictions = logits.argmax(dim=-1)
            correct += int((predictions == batch["label"]).sum().item())
            total += int(len(batch["label"]))
        return {"loss": total_loss / max(total, 1), "accuracy": correct / max(total, 1)}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict[str, Any]:
        self.model.eval()
        total_loss = 0.0
        y_true: list[int] = []
        y_pred: list[int] = []
        for batch in tqdm(loader, desc="eval", leave=False):
            batch = self._move_batch(batch)
            loss, logits = self._forward_loss(batch)
            total_loss += float(loss.item()) * len(batch["label"])
            y_true.extend(batch["label"].detach().cpu().tolist())
            y_pred.extend(logits.argmax(dim=-1).detach().cpu().tolist())
        metrics = compute_classification_metrics(y_true, y_pred, self.labels)
        metrics["loss"] = total_loss / max(len(y_true), 1)
        return metrics

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int) -> list[dict[str, Any]]:
        history: list[dict[str, Any]] = []
        best_f1 = -1.0
        stale_epochs = 0
        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
            history.append(row)
            if val_metrics["f1_macro"] > best_f1:
                best_f1 = float(val_metrics["f1_macro"])
                stale_epochs = 0
                torch.save(self.model.state_dict(), self.checkpoint_dir / f"{self.model_name}_best.pt")
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break
        torch.save(self.model.state_dict(), self.checkpoint_dir / f"{self.model_name}_final.pt")
        with (self.checkpoint_dir / f"{self.model_name}_history.json").open("w", encoding="utf-8") as file:
            json.dump(history, file, indent=2)
        return history
