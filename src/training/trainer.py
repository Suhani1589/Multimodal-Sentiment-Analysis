from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional, Union

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
        checkpoint_metadata: dict[str, Any] | None = None,
        scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau] = None,
        overwrite: bool = False,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.labels = labels
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_name = model_name
        self.patience = patience
        self.checkpoint_metadata = checkpoint_metadata or {}
        self.scheduler = scheduler
        self.overwrite = overwrite
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _move_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        return {key: value.to(self.device) if torch.is_tensor(value) else value for key, value in batch.items()}

    def _forward_loss(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        labels = batch["label"]
        logits = self.model(**batch)
        loss = self.criterion(logits, labels)
        return loss, logits

    def _diagnostic_groups(self) -> dict[str, nn.Module]:
        groups: dict[str, nn.Module] = {"classifier": self.model.classifier} if hasattr(self.model, "classifier") else {}
        if hasattr(self.model, "text_encoder"):
            groups["text_encoder"] = self.model.text_encoder
        elif self.model_name == "text" and hasattr(self.model, "encoder"):
            groups["text_encoder"] = self.model.encoder
        if hasattr(self.model, "image_encoder"):
            groups["image_encoder"] = self.model.image_encoder
        elif self.model_name == "image" and hasattr(self.model, "encoder"):
            groups["image_encoder"] = self.model.encoder
        return groups

    def _snapshot_diagnostic_parameters(self, batch: dict[str, Any]) -> dict[str, dict[str, Any]]:
        snapshots: dict[str, dict[str, Any]] = {}
        token_ids = batch.get("input_ids")
        for group_name, module in self._diagnostic_groups().items():
            for parameter_name, parameter in module.named_parameters():
                if not parameter.requires_grad:
                    continue
                indices = None
                if parameter_name.endswith("word_embeddings.weight") and token_ids is not None:
                    indices = torch.unique(token_ids.detach().reshape(-1))
                    before = parameter.detach().index_select(0, indices).clone()
                else:
                    before = parameter.detach().reshape(-1)[:256].clone()
                snapshots[group_name] = {
                    "parameter_name": parameter_name,
                    "parameter": parameter,
                    "indices": indices,
                    "before": before,
                    "trainable_parameter_tensors": sum(
                        1 for item in module.parameters() if item.requires_grad
                    ),
                }
                break
        return snapshots

    def _gradient_diagnostics(
        self, snapshots: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        diagnostics = {}
        for group_name, snapshot in snapshots.items():
            module = self._diagnostic_groups()[group_name]
            parameter = snapshot["parameter"]
            gradient = parameter.grad
            if gradient is not None and snapshot["indices"] is not None:
                probe = gradient.index_select(0, snapshot["indices"]).detach()
            elif gradient is not None:
                probe = gradient.detach().reshape(-1)[:256]
            else:
                probe = None
            diagnostics[group_name] = {
                "trainable_parameter_tensors": snapshot["trainable_parameter_tensors"],
                "gradient_parameter_tensors": sum(
                    1 for item in module.parameters() if item.requires_grad and item.grad is not None
                ),
                "representative_parameter": snapshot["parameter_name"],
                "representative_gradient_norm": float(probe.norm().item()) if probe is not None else None,
                "_snapshot": snapshot,
            }
        return diagnostics

    @staticmethod
    def _finish_update_diagnostics(diagnostics: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        result = {}
        for group_name, values in diagnostics.items():
            snapshot = values.pop("_snapshot")
            parameter = snapshot["parameter"].detach()
            if snapshot["indices"] is not None:
                after = parameter.index_select(0, snapshot["indices"])
            else:
                after = parameter.reshape(-1)[:256]
            values["representative_update_abs_mean"] = float((after - snapshot["before"]).abs().mean().item())
            result[group_name] = values
        return result

    def train_epoch(self, loader: DataLoader) -> dict[str, Any]:
        self.model.train()
        total_loss = 0.0
        total = 0
        y_true: list[int] = []
        y_pred: list[int] = []
        gradient_diagnostics = None
        for batch in tqdm(loader, desc="train", leave=False):
            batch = self._move_batch(batch)
            snapshots = self._snapshot_diagnostic_parameters(batch) if total == 0 else None
            self.optimizer.zero_grad(set_to_none=True)
            loss, logits = self._forward_loss(batch)
            loss.backward()
            if snapshots is not None:
                gradient_diagnostics = self._gradient_diagnostics(snapshots)
            self.optimizer.step()
            if snapshots is not None and gradient_diagnostics is not None:
                gradient_diagnostics = self._finish_update_diagnostics(gradient_diagnostics)
            total_loss += float(loss.item()) * len(batch["label"])
            predictions = logits.argmax(dim=-1)
            total += int(len(batch["label"]))
            y_true.extend(batch["label"].detach().cpu().tolist())
            y_pred.extend(predictions.detach().cpu().tolist())
        metrics = compute_classification_metrics(y_true, y_pred, self.labels)
        metrics["loss"] = total_loss / max(total, 1)
        metrics["gradient_diagnostics"] = gradient_diagnostics or {}
        return metrics

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
        best_path = self.checkpoint_dir / f"{self.model_name}_best.pt"
        final_path = self.checkpoint_dir / f"{self.model_name}_final.pt"
        history_path = self.checkpoint_dir / f"{self.model_name}_history.json"
        metadata_path = self.checkpoint_dir / f"{self.model_name}_run.json"
        existing = [path for path in (best_path, final_path, history_path, metadata_path) if path.exists()]
        if existing and not self.overwrite:
            raise FileExistsError(
                "Run outputs already exist: " + ", ".join(str(path) for path in existing)
                + ". Choose a new --output-dir or pass --overwrite."
            )
        history: list[dict[str, Any]] = []
        best_f1 = -1.0
        best_epoch = 0
        stale_epochs = 0
        run_start = time.perf_counter()
        for epoch in range(1, epochs + 1):
            epoch_start = time.perf_counter()
            learning_rate = float(self.optimizer.param_groups[0]["lr"])
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            row = {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "train_metric_scope": "online_minibatch_outputs_before_optimizer_update",
                "validation_metric_scope": "post_epoch_eval_mode",
                "learning_rate": learning_rate,
                "epoch_duration_seconds": time.perf_counter() - epoch_start,
            }
            history.append(row)
            if self.scheduler is not None:
                self.scheduler.step(float(val_metrics["f1_macro"]))
            # Strict greater-than keeps the earliest epoch when validation scores tie.
            if val_metrics["f1_macro"] > best_f1:
                best_f1 = float(val_metrics["f1_macro"])
                best_epoch = epoch
                stale_epochs = 0
                self._save_checkpoint(best_path, epoch, best_f1, best_epoch)
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break
        self._save_checkpoint(final_path, len(history), best_f1, best_epoch)
        with history_path.open("w", encoding="utf-8") as file:
            json.dump(history, file, indent=2)
        metadata = dict(self.checkpoint_metadata)
        metadata.update(
            {
                "best_validation_macro_f1": best_f1,
                "best_epoch": best_epoch,
                "training_duration_seconds": time.perf_counter() - run_start,
                "best_checkpoint": best_path.name,
                "checkpoint_tie_break_rule": "earliest_epoch_wins",
            }
        )
        with metadata_path.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)
        return history

    def _save_checkpoint(self, path: Path, epoch: int, best_f1: float, best_epoch: int) -> None:
        tokenizer_path = self.checkpoint_metadata.get("tokenizer_path")
        if tokenizer_path:
            tokenizer_dir = self.checkpoint_dir / tokenizer_path
            if tokenizer_dir.exists():
                self.checkpoint_metadata["tokenizer_files"] = sorted(
                    file.name for file in tokenizer_dir.iterdir() if file.is_file()
                )
        payload = {
            "format_version": 1,
            "model_type": self.model_name,
            "model_state_dict": self.model.state_dict(),
            "labels": self.labels,
            "epoch": epoch,
            "selection_metric": "validation_macro_f1",
            "best_validation_macro_f1": best_f1,
            "best_epoch": best_epoch,
            **self.checkpoint_metadata,
        }
        torch.save(payload, path)
