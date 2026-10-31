from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data.dataset import MultimodalSentimentDataset, SentimentLabels
from src.evaluation.visualization import save_confusion_matrix
from src.text.tokenizer import build_tokenizer
from src.training.frozen_probe import (
    LABELS,
    clears_constant_baseline,
    constant_positive_metrics,
    masked_mean_pool,
    nearest_centroid_accuracy,
    representation_separation,
    select_probe_splits,
)
from src.training.metrics import compute_classification_metrics
from src.training.sampling import build_training_sampler
from src.utils import get_device, seed_everything


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float):
        return value
    if hasattr(value, "item"):
        return value.item()
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")


def _extract_representations(
    frame: pd.DataFrame,
    *,
    raw_dir: str,
    tokenizer: Any,
    encoder: nn.Module,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    dataset = MultimodalSentimentDataset(
        frame,
        raw_dir=raw_dir,
        labels=SentimentLabels(LABELS),
        tokenizer=tokenizer,
        max_length=max_length,
        mode="text",
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    pooler_parts: list[torch.Tensor] = []
    mean_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    sample_ids: list[str] = []
    encoder.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
            if outputs.pooler_output is None:
                raise RuntimeError("xlm-roberta-base did not return a pooler representation")
            pooler_parts.append(outputs.pooler_output.detach().cpu())
            mean_parts.append(masked_mean_pool(outputs.last_hidden_state, attention_mask).detach().cpu())
            label_parts.append(batch["label"].detach().cpu().view(-1))
            sample_ids.extend(str(value) for value in batch["sample_id"])
            if batch_index == 1 or batch_index % 50 == 0:
                print(f"  extracted {min(batch_index * batch_size, len(dataset))}/{len(dataset)}", flush=True)
    return {
        "pooler": torch.cat(pooler_parts),
        "masked_mean": torch.cat(mean_parts),
        "labels": torch.cat(label_parts).long(),
        "sample_ids": sample_ids,
    }


def _predict(head: nn.Module, features: torch.Tensor, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    head.eval()
    probabilities: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            logits = head(features[start : start + batch_size].to(device))
            probabilities.append(torch.softmax(logits, dim=-1).cpu())
    stacked = torch.cat(probabilities)
    return stacked, stacked.argmax(dim=-1)


def _confidence_summary(probabilities: torch.Tensor, labels: list[str]) -> dict[str, Any]:
    confidence = probabilities.max(dim=-1).values
    return {
        "mean_confidence": float(confidence.mean()),
        "median_confidence": float(confidence.median()),
        "min_confidence": float(confidence.min()),
        "max_confidence": float(confidence.max()),
        "mean_class_probability": {
            label: float(probabilities[:, index].mean()) for index, label in enumerate(labels)
        },
    }


def _save_plots(
    history: list[dict[str, Any]],
    metrics: dict[str, Any],
    output_dir: Path,
    baseline_macro_f1: float,
) -> None:
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(epochs, [row["train_loss"] for row in history], marker="o", label="Train, sampled")
    axes[0].plot(epochs, [row["val_loss"] for row in history], marker="o", label="Validation")
    axes[0].set(title="Cross-entropy loss", xlabel="Epoch", ylabel="Loss")
    axes[1].plot(epochs, [row["val_f1_macro"] for row in history], marker="o", label="Validation macro-F1")
    axes[1].axhline(baseline_macro_f1, color="0.4", linestyle="--", label="Constant positive")
    axes[1].set(title="Validation macro-F1", xlabel="Epoch", ylabel="Macro-F1", ylim=(0, 1))
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.savefig(output_dir / "training_curves.png", dpi=200)
    plt.close(figure)

    save_confusion_matrix(
        np.asarray(metrics["confusion_matrix"], dtype=int),
        LABELS,
        output_dir / "validation_confusion_matrix.png",
    )
    true_counts = [metrics["true_class_distribution"][label] for label in LABELS]
    predicted_counts = [metrics["predicted_class_distribution"][label] for label in LABELS]
    positions = range(len(LABELS))
    figure, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
    axis.bar([position - 0.18 for position in positions], true_counts, width=0.36, label="True")
    axis.bar([position + 0.18 for position in positions], predicted_counts, width=0.36, label="Predicted")
    axis.set_xticks(list(positions), LABELS)
    axis.set(title="Validation class counts", ylabel="Rows")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(output_dir / "validation_prediction_distribution.png", dpi=200)
    plt.close(figure)


def train_linear_head(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    train_names: list[str],
    validation_features: torch.Tensor,
    validation_labels: torch.Tensor,
    *,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    patience: int,
    seed: int,
    device: torch.device,
) -> tuple[nn.Linear, list[dict[str, Any]], dict[str, Any]]:
    seed_everything(seed)
    head = nn.Linear(train_features.shape[1], len(LABELS)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    if getattr(criterion, "weight", None) is not None:
        raise RuntimeError("The probe must use unweighted cross-entropy")
    sampler = build_training_sampler(train_names, LABELS, strategy="weighted_sampler", seed=seed)
    train_loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
    )
    best_f1 = -1.0
    best_epoch = 0
    best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        head.train()
        loss_sum = 0.0
        seen = 0
        train_true: list[int] = []
        train_pred: list[int] = []
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(features)
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite probe loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * int(targets.shape[0])
            seen += int(targets.shape[0])
            train_true.extend(targets.detach().cpu().tolist())
            train_pred.extend(logits.argmax(dim=-1).detach().cpu().tolist())
        train_metrics = compute_classification_metrics(train_true, train_pred, LABELS)
        val_probabilities, val_predictions = _predict(head, validation_features, batch_size, device)
        val_metrics = compute_classification_metrics(validation_labels.tolist(), val_predictions.tolist(), LABELS)
        val_loss = float(
            nn.functional.nll_loss(torch.log(val_probabilities.clamp_min(1e-12)), validation_labels).item()
        )
        row = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": loss_sum / max(seen, 1),
            "train_metric_scope": "weighted_sampler_epoch",
            "train_accuracy": train_metrics["accuracy"],
            "train_f1_macro": train_metrics["f1_macro"],
            "train_predicted_class_distribution": train_metrics["predicted_class_distribution"],
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_f1_macro": val_metrics["f1_macro"],
            "val_predicted_class_distribution": val_metrics["predicted_class_distribution"],
            "val_confidence": _confidence_summary(val_probabilities, LABELS),
        }
        history.append(row)
        print(
            f"  epoch {epoch}: val_acc={row['val_accuracy']:.4f} val_macro_f1={row['val_f1_macro']:.4f} "
            f"val_pred={row['val_predicted_class_distribution']}",
            flush=True,
        )
        if row["val_f1_macro"] > best_f1:
            best_f1 = float(row["val_f1_macro"])
            best_epoch = epoch
            stale_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    head.load_state_dict(best_state)
    head.to(device)
    probabilities, predictions = _predict(head, validation_features, batch_size, device)
    metrics = compute_classification_metrics(validation_labels.tolist(), predictions.tolist(), LABELS)
    metrics["loss"] = float(
        nn.functional.nll_loss(torch.log(probabilities.clamp_min(1e-12)), validation_labels).item()
    )
    metrics["confidence"] = _confidence_summary(probabilities, LABELS)
    metrics["best_epoch"] = best_epoch
    metrics["epochs_completed"] = len(history)
    metrics["probabilities"] = probabilities
    metrics["predictions"] = predictions
    return head, history, metrics


def _save_representation_run(
    name: str,
    output_dir: Path,
    head: nn.Module,
    history: list[dict[str, Any]],
    metrics: dict[str, Any],
    validation_ids: list[str],
    validation_labels: torch.Tensor,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    validation_features: torch.Tensor,
    baseline_macro_f1: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    probabilities = metrics.pop("probabilities")
    predictions = metrics.pop("predictions")
    decision = clears_constant_baseline(metrics, baseline_macro_f1)
    diagnostics = representation_separation(validation_features, validation_labels)
    diagnostics["nearest_centroid_validation_accuracy"] = nearest_centroid_accuracy(
        train_features, train_labels, validation_features, validation_labels
    )
    payload = {
        "representation": name,
        "encoder": "xlm-roberta-base",
        "encoder_source": "pretrained",
        "encoder_frozen": True,
        "checkpoint_loaded": False,
        "class_weighted_loss": False,
        "sampling_strategy": "weighted_sampler_on_training_rows_only",
        "learning_rate": config["training"]["learning_rate"],
        "selection_metric": "validation_macro_f1",
        "best_epoch": metrics["best_epoch"],
        "gate": decision,
        **{key: value for key, value in metrics.items()},
    }
    _write_json(output_dir / "metrics.json", payload)
    _write_json(output_dir / "history.json", {"epochs": history})
    _write_json(output_dir / "representation_diagnostics.json", diagnostics)
    confusion = pd.DataFrame(
        metrics["confusion_matrix"],
        index=[f"true_{label}" for label in LABELS],
        columns=[f"predicted_{label}" for label in LABELS],
    )
    confusion.to_csv(output_dir / "validation_confusion_matrix.csv")
    prediction_frame = pd.DataFrame(
        {
            "sample_id": validation_ids,
            "split": "val",
            "true_label": [LABELS[int(index)] for index in validation_labels.tolist()],
            "predicted_label": [LABELS[int(index)] for index in predictions.tolist()],
        }
    )
    for index, label in enumerate(LABELS):
        prediction_frame[f"probability_{label}"] = probabilities[:, index].tolist()
    prediction_frame.to_csv(output_dir / "validation_predictions.csv", index=False)
    torch.save(
        {
            "representation": name,
            "labels": LABELS,
            "state_dict": {key: value.detach().cpu() for key, value in head.state_dict().items()},
            "best_epoch": metrics["best_epoch"],
        },
        output_dir / "head.pt",
    )
    _save_plots(history, metrics, output_dir, baseline_macro_f1)
    payload["representation_diagnostics"] = diagnostics
    return payload


def main() -> None:
    config = load_config("configs/probe-text-head.yaml")
    output_dir = Path(config["output_dir"])
    summary_path = output_dir / "summary.json"
    features_path = output_dir / "features.pt"
    if summary_path.exists():
        raise FileExistsError(f"{summary_path} already exists. The completed probe will not be overwritten.")
    resume = features_path.is_file()
    if output_dir.exists() and not resume:
        raise FileExistsError(
            f"{output_dir} already exists without a feature cache. The probe will not overwrite it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if not (output_dir / "config.yaml").exists():
        shutil.copyfile("configs/probe-text-head.yaml", output_dir / "config.yaml")
    seed = int(config["seed"])
    seed_everything(seed)
    device = get_device("auto")
    print(f"Device: {device}", flush=True)

    frame = pd.read_csv(config["data"]["prepared_csv"])
    frame = frame.loc[frame["split"].astype(str).str.lower().ne("test")].copy()
    train_frame, validation_frame = select_probe_splits(frame)
    if set(train_frame["split"].astype(str).str.lower()).union(validation_frame["split"].astype(str).str.lower()) != {
        "train",
        "val",
    }:
        raise RuntimeError("Probe frames include a split other than train and validation")
    print(
        f"Train rows: {len(train_frame)} Validation rows: {len(validation_frame)} Test rows used: 0",
        flush=True,
    )

    if resume:
        print(f"Reusing feature cache {features_path}", flush=True)
        cached = torch.load(features_path, map_location="cpu", weights_only=False)
        if cached.get("test_split_included"):
            raise RuntimeError("Feature cache records test rows and cannot be used")
        train_cache = {
            "pooler": cached["train_pooler"],
            "masked_mean": cached["train_masked_mean"],
            "labels": cached["train_labels"],
            "sample_ids": cached["train_sample_ids"],
        }
        validation_cache = {
            "pooler": cached["validation_pooler"],
            "masked_mean": cached["validation_masked_mean"],
            "labels": cached["validation_labels"],
            "sample_ids": cached["validation_sample_ids"],
        }
    else:
        tokenizer = build_tokenizer(config["encoder"]["model_name"])
        from transformers import AutoModel

        encoder = AutoModel.from_pretrained(config["encoder"]["model_name"])
        for parameter in encoder.parameters():
            parameter.requires_grad = False
        encoder.to(device)
        trainable_encoder = sum(parameter.requires_grad for parameter in encoder.parameters())
        if trainable_encoder != 0:
            raise RuntimeError("Encoder parameters remained trainable")
        print("Frozen pretrained encoder loaded. Trainable encoder tensors: 0", flush=True)

        started = time.perf_counter()
        print("Extracting training representations", flush=True)
        train_cache = _extract_representations(
            train_frame,
            raw_dir=config["data"]["raw_dir"],
            tokenizer=tokenizer,
            encoder=encoder,
            max_length=int(config["encoder"]["max_length"]),
            batch_size=int(config["encoder"]["extraction_batch_size"]),
            device=device,
        )
        print("Extracting validation representations", flush=True)
        validation_cache = _extract_representations(
            validation_frame,
            raw_dir=config["data"]["raw_dir"],
            tokenizer=tokenizer,
            encoder=encoder,
            max_length=int(config["encoder"]["max_length"]),
            batch_size=int(config["encoder"]["extraction_batch_size"]),
            device=device,
        )
        torch.save(
            {
                "train_pooler": train_cache["pooler"],
                "train_masked_mean": train_cache["masked_mean"],
                "train_labels": train_cache["labels"],
                "train_sample_ids": train_cache["sample_ids"],
                "validation_pooler": validation_cache["pooler"],
                "validation_masked_mean": validation_cache["masked_mean"],
                "validation_labels": validation_cache["labels"],
                "validation_sample_ids": validation_cache["sample_ids"],
                "test_split_included": False,
            },
            features_path,
        )
        print(f"Representation extraction took {time.perf_counter() - started:.1f}s", flush=True)
        del encoder

    train_encoded = [LABELS.index(label) for label in train_frame["sentiment"].astype(str).str.lower()]
    validation_encoded = [LABELS.index(label) for label in validation_frame["sentiment"].astype(str).str.lower()]
    if train_encoded != train_cache["labels"].tolist() or validation_encoded != validation_cache["labels"].tolist():
        raise RuntimeError("Extracted labels do not match the source rows")
    if train_cache["sample_ids"] != train_frame["sample_id"].astype(str).tolist():
        raise RuntimeError("Extracted training ids do not match the source rows")
    if validation_cache["sample_ids"] != validation_frame["sample_id"].astype(str).tolist():
        raise RuntimeError("Extracted validation ids do not match the source rows")

    baseline = constant_positive_metrics(validation_cache["labels"].tolist())
    _write_json(output_dir / "constant_positive_baseline.json", baseline)
    baseline_f1 = float(baseline["f1_macro"])

    def run_named(name: str, train_key: str, validation_key: str) -> dict[str, Any]:
        print(f"Training linear head on {name}", flush=True)
        head, history, metrics = train_linear_head(
            train_cache[train_key],
            train_cache["labels"],
            train_frame["sentiment"].astype(str).str.lower().tolist(),
            validation_cache[validation_key],
            validation_cache["labels"],
            learning_rate=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
            batch_size=int(config["training"]["batch_size"]),
            epochs=int(config["training"]["epochs"]),
            patience=int(config["training"]["early_stopping_patience"]),
            seed=seed,
            device=device,
        )
        return _save_representation_run(
            name,
            output_dir / name,
            head,
            history,
            metrics,
            validation_cache["sample_ids"],
            validation_cache["labels"],
            train_cache[train_key],
            train_cache["labels"],
            validation_cache[validation_key],
            baseline_f1,
            config,
        )

    pooler_dir = output_dir / "pooler"
    if (pooler_dir / "metrics.json").is_file():
        print("Reusing saved pooler head. It will not be retrained.", flush=True)
        pooler_result = json.loads((pooler_dir / "metrics.json").read_text(encoding="utf-8"))
        pooler_result["representation_diagnostics"] = json.loads(
            (pooler_dir / "representation_diagnostics.json").read_text(encoding="utf-8")
        )
        history = json.loads((pooler_dir / "history.json").read_text(encoding="utf-8"))["epochs"]
        if not (pooler_dir / "validation_confusion_matrix.png").is_file():
            _save_plots(history, pooler_result, pooler_dir, baseline_f1)
    else:
        pooler_result = run_named("pooler", "pooler", "pooler")
    results = {"pooler": pooler_result}
    if not pooler_result["gate"]["clears_gate"]:
        print("Pooler probe did not clear the constant-positive gate. Training masked-mean head.", flush=True)
        results["masked_mean"] = run_named("masked_mean", "masked_mean", "masked_mean")
    else:
        print("Pooler probe cleared the gate. Masked-mean head was not trained.", flush=True)

    selected_name = "pooler" if pooler_result["gate"]["clears_gate"] else (
        "masked_mean" if results.get("masked_mean", {}).get("gate", {}).get("clears_gate") else None
    )
    summary = {
        "test_split_used": False,
        "existing_checkpoints_modified": False,
        "constant_positive_validation_macro_f1": baseline_f1,
        "historical_constant_positive_validation_macro_f1": 0.24713682941531043,
        "pooler_clears_gate": pooler_result["gate"]["clears_gate"],
        "masked_mean_trained": "masked_mean" in results,
        "masked_mean_clears_gate": results.get("masked_mean", {}).get("gate", {}).get("clears_gate"),
        "selected_representation": selected_name,
        "supports_full_finetuning": selected_name is not None,
        "results": {
            name: {
                "accuracy": result["accuracy"],
                "f1_macro": result["f1_macro"],
                "predicted_class_distribution": result["predicted_class_distribution"],
                "per_class_f1": {
                    label: result["per_class"][label]["f1-score"] for label in LABELS
                },
                "gate": result["gate"],
                "confidence": result["confidence"],
                "representation_diagnostics": result["representation_diagnostics"],
            }
            for name, result in results.items()
        },
    }
    _write_json(output_dir / "summary.json", summary)
    print("PROBE_COMPLETE", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
