from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.data.dataset import MultimodalSentimentDataset, SentimentLabels
from src.model_factory import build_model
from src.text.tokenizer import build_tokenizer
from src.training.frozen_probe import load_probe_classifier
from src.training.losses import build_cross_entropy_loss
from src.training.loaders import build_eval_dataloader
from src.training.sampling import build_training_sampler
from src.training.trainer import Trainer
from src.utils import get_device, seed_everything
from src.vision.transforms import build_image_transform


MODEL_TYPES = ("text", "image", "multimodal")
EXPECTED_LABELS = ["positive", "neutral", "negative"]


def _fixed_validation_subset(val_frame: pd.DataFrame, labels: list[str], per_class: int, seed: int) -> pd.DataFrame:
    pieces = []
    for offset, label in enumerate(labels):
        class_rows = val_frame.loc[val_frame["sentiment"].astype(str).str.lower().eq(label)]
        if class_rows.empty:
            raise ValueError(f"Validation split has no {label} rows for the diversity subset")
        pieces.append(class_rows.sample(n=min(per_class, len(class_rows)), random_state=seed + offset))
    return pd.concat(pieces)


def _masked_mean_feature_delta(model, dataset, feature_cache_path: str, device: torch.device) -> float:
    """Compare live masked-mean vectors with the frozen probe's cached validation features."""
    try:
        cache = torch.load(feature_cache_path, map_location="cpu", weights_only=False)
    except TypeError:
        cache = torch.load(feature_cache_path, map_location="cpu")
    id_to_index = {str(sample_id): index for index, sample_id in enumerate(cache["validation_sample_ids"])}
    cached_rows = []
    live_rows = []
    model.eval()
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=8, shuffle=False):
            sample_ids = batch["sample_id"]
            if torch.is_tensor(sample_ids):
                sample_ids = [str(value) for value in sample_ids.tolist()]
            else:
                sample_ids = [str(value) for value in sample_ids]
            missing = [sample_id for sample_id in sample_ids if sample_id not in id_to_index]
            if missing:
                raise RuntimeError(f"Probe feature cache has no rows for {missing[:3]}")
            indices = [id_to_index[sample_id] for sample_id in sample_ids]
            cached_rows.append(cache["validation_masked_mean"][indices].float().cpu())
            moved = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            live_rows.append(
                model.encoder(input_ids=moved["input_ids"], attention_mask=moved["attention_mask"]).float().cpu()
            )
    cached = torch.cat(cached_rows)
    live = torch.cat(live_rows)
    if cached.shape != live.shape:
        raise RuntimeError(f"Feature shape {tuple(live.shape)} does not match cache {tuple(cached.shape)}")
    return float((live - cached).abs().max())


def run_training(
    model_type: str,
    *,
    config_path: str = "configs/config.yaml",
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    text_model_name: Optional[str] = None,
    output_dir: Optional[str] = None,
    overwrite: bool = False,
) -> list[dict]:
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unsupported model type: {model_type}")
    if epochs is not None and epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if batch_size is not None and batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if max_train_samples is not None and max_train_samples < 1:
        raise ValueError("--max-train-samples must be at least 1")
    if max_val_samples is not None and max_val_samples < 1:
        raise ValueError("--max-val-samples must be at least 1")
    config = load_config(config_path)
    if text_model_name:
        config["text_model"]["model_name"] = text_model_name
    seed = int(config["project"]["seed"])
    seed_everything(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    dataset_config = config["dataset"]
    if [str(label).lower() for label in dataset_config["label_order"]] != EXPECTED_LABELS:
        raise ValueError(f"dataset.label_order must be {EXPECTED_LABELS} for comparable experiments")
    prepared_path = Path(dataset_config["prepared_csv"])
    if not prepared_path.exists():
        raise FileNotFoundError(f"Prepared data is missing: {prepared_path}. Run scripts/prepare_data.py first.")
    frame = pd.read_csv(prepared_path)
    if "split" not in frame.columns:
        raise ValueError(f"Prepared data has no split column: {prepared_path}")
    frame = frame.loc[frame["split"].astype(str).str.lower().ne("test")].copy()
    require_text = model_type in {"text", "multimodal"} or bool(dataset_config.get("require_nonempty_text", False))
    if require_text:
        has_text = frame["text"].notna() & frame["text"].astype(str).str.strip().ne("")
        print(f"{model_type} usable text-image rows: {int(has_text.sum())}/{len(frame)}")
        frame = frame.loc[has_text].copy()
    train_frame = frame.loc[frame["split"] == "train"].copy()
    val_frame = frame.loc[frame["split"] == "val"].copy()
    if max_train_samples is not None and len(train_frame) > max_train_samples:
        train_frame = train_frame.sample(n=max_train_samples, random_state=seed).sort_index()
    if max_val_samples is not None and len(val_frame) > max_val_samples:
        val_frame = val_frame.sample(n=max_val_samples, random_state=seed + 1).sort_index()
    if train_frame.empty or val_frame.empty:
        raise ValueError("Training and validation splits must both contain usable samples.")
    diversity_per_class = int(config["training"].get("diversity_samples_per_class") or 0)
    diversity_frame = None
    if diversity_per_class and model_type in {"text", "multimodal"}:
        diversity_frame = _fixed_validation_subset(val_frame, EXPECTED_LABELS, diversity_per_class, seed)

    output_path = Path(output_dir) if output_dir else Path(config["paths"]["checkpoints_dir"])
    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer_relative_path = f"{model_type}_tokenizer" if model_type != "image" else None
    run_outputs = [
        output_path / f"{model_type}_best.pt",
        output_path / f"{model_type}_final.pt",
        output_path / f"{model_type}_history.json",
        output_path / f"{model_type}_run.json",
    ]
    if tokenizer_relative_path:
        run_outputs.append(output_path / tokenizer_relative_path)
    existing = [path for path in run_outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Run outputs already exist: " + ", ".join(str(path) for path in existing)
            + ". Choose a new --output-dir or pass --overwrite."
        )

    labels = SentimentLabels(dataset_config["label_order"])
    tokenizer = build_tokenizer(config["text_model"]["model_name"]) if model_type != "image" else None
    image_transform = None
    if model_type in {"image", "multimodal"}:
        image_transform = build_image_transform(config["preprocessing"]["image_size"], train=True)
    train_dataset = MultimodalSentimentDataset(
        train_frame,
        raw_dir=dataset_config["raw_dir"],
        labels=labels,
        tokenizer=tokenizer,
        image_transform=image_transform,
        max_length=config["preprocessing"]["max_text_length"],
        mode=model_type,
    )
    eval_transform = build_image_transform(config["preprocessing"]["image_size"], train=False) if model_type in {"image", "multimodal"} else None
    val_dataset = MultimodalSentimentDataset(
        val_frame,
        raw_dir=dataset_config["raw_dir"],
        labels=labels,
        tokenizer=tokenizer,
        image_transform=eval_transform,
        max_length=config["preprocessing"]["max_text_length"],
        mode=model_type,
    )
    resolved_batch = int(batch_size or config["training"]["batch_size"])
    generator = torch.Generator()
    generator.manual_seed(seed)
    training_config = config["training"]
    imbalance_strategy = training_config.get(
        "imbalance_strategy", "class_weights" if training_config.get("use_class_weights", False) else "none"
    )
    train_sampler = build_training_sampler(
        train_frame["sentiment"].astype(str).str.lower().tolist(),
        labels.labels,
        strategy=imbalance_strategy,
        seed=seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=resolved_batch,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=config["training"]["num_workers"],
        generator=generator,
    )
    val_loader = build_eval_dataloader(
        val_dataset,
        batch_size=resolved_batch,
        seed=seed,
        num_workers=config["training"]["num_workers"],
    )
    diversity_dataset = None
    diversity_loader = None
    if diversity_frame is not None and tokenizer is not None:
        diversity_dataset = MultimodalSentimentDataset(
            diversity_frame,
            raw_dir=dataset_config["raw_dir"],
            labels=labels,
            tokenizer=tokenizer,
            image_transform=None,
            max_length=config["preprocessing"]["max_text_length"],
            mode="text",
        )
        diversity_loader = DataLoader(diversity_dataset, batch_size=resolved_batch, shuffle=False)

    model = build_model(model_type, config, load_pretrained=True)
    probe_path = training_config.get("initialize_classifier_from")
    probe_load = None
    if probe_path:
        if model_type != "text":
            raise ValueError("A probe classification head can only initialize the text model")
        if config["text_model"].get("pooling") != "masked_mean":
            raise ValueError("The saved probe head requires text_model.pooling: masked_mean")
        probe_load = load_probe_classifier(model.classifier, probe_path, labels.labels)
        print(f"Loaded probe head from {probe_path}: {probe_load}", flush=True)
    device = get_device(config["project"]["device"])
    print(f"Training device: {device.type}", flush=True)
    model.to(device)
    feature_delta = None
    feature_cache = training_config.get("probe_feature_cache")
    if probe_load is not None and feature_cache and diversity_dataset is not None:
        feature_delta = _masked_mean_feature_delta(model, diversity_dataset, feature_cache, device)
        print(f"Masked-mean feature max abs difference vs probe cache: {feature_delta:.6g}", flush=True)
    train_counts = train_frame["sentiment"].astype(str).str.lower().value_counts()
    class_weights = None
    if imbalance_strategy == "class_weights":
        class_weights = [
            len(train_frame) / (len(labels.labels) * max(int(train_counts.get(label, 0)), 1))
            for label in labels.labels
        ]
    criterion = build_cross_entropy_loss(class_weights=class_weights, device=device)
    weight_decay = float(config["training"]["weight_decay"])
    encoder_learning_rate = training_config.get("encoder_learning_rate")
    head_learning_rate = training_config.get("head_learning_rate")
    if model_type == "text" and encoder_learning_rate is not None and head_learning_rate is not None:
        encoder_parameters = [parameter for parameter in model.encoder.parameters() if parameter.requires_grad]
        classifier_parameters = [parameter for parameter in model.classifier.parameters() if parameter.requires_grad]
        if not encoder_parameters or not classifier_parameters:
            raise RuntimeError("Encoder and classifier both need trainable parameters")
        learning_rate = float(encoder_learning_rate)
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": encoder_parameters,
                    "lr": learning_rate,
                    "weight_decay": weight_decay,
                    "name": "encoder",
                },
                {
                    "params": classifier_parameters,
                    "lr": float(head_learning_rate),
                    "weight_decay": weight_decay,
                    "name": "classifier",
                },
            ]
        )
    elif (
        model_type == "multimodal"
        and encoder_learning_rate is not None
        and head_learning_rate is not None
    ):
        text_parameters = [parameter for parameter in model.text_encoder.parameters() if parameter.requires_grad]
        image_parameters = [parameter for parameter in model.image_encoder.parameters() if parameter.requires_grad]
        head_parameters = [
            parameter
            for parameter in list(model.fusion.parameters()) + list(model.classifier.parameters())
            if parameter.requires_grad
        ]
        if not text_parameters or not image_parameters or not head_parameters:
            raise RuntimeError("Text encoder, image encoder, and fusion head all need trainable parameters")
        learning_rate = float(encoder_learning_rate)
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": text_parameters,
                    "lr": learning_rate,
                    "weight_decay": weight_decay,
                    "name": "text_encoder",
                },
                {
                    "params": image_parameters,
                    "lr": float(config["training"]["image_learning_rate"]),
                    "weight_decay": weight_decay,
                    "name": "image_encoder",
                },
                {
                    "params": head_parameters,
                    "lr": float(head_learning_rate),
                    "weight_decay": weight_decay,
                    "name": "classifier",
                },
            ]
        )
    else:
        learning_rate = (
            config["training"]["image_learning_rate"] if model_type == "image" else config["training"]["learning_rate"]
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    gradient_clip_norm = training_config.get("gradient_clip_norm")
    if gradient_clip_norm is not None:
        gradient_clip_norm = float(gradient_clip_norm)
    scheduler_config = config["training"].get("scheduler", {})
    scheduler = None
    if scheduler_config.get("enabled", False):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(scheduler_config.get("factor", 0.5)),
            patience=int(scheduler_config.get("patience", 1)),
            min_lr=float(scheduler_config.get("min_lr", 1e-7)),
        )
    config_snapshot = copy.deepcopy(config)
    config_snapshot["training"]["epochs"] = int(epochs or config["training"]["epochs"])
    config_snapshot["training"]["batch_size"] = resolved_batch
    config_snapshot["paths"]["checkpoints_dir"] = str(output_path)
    checkpoint_metadata = {
        "config": config_snapshot,
        "config_file": str(Path(config_path)),
        "seed": seed,
        "label_to_id": dict(labels.label_to_id),
        "id_to_label": dict(labels.id_to_label),
        "model_config": {
            "text_model_name": config["text_model"]["model_name"],
            "image_backbone": config["image_model"]["backbone"],
            "text_pooling": config["text_model"].get("pooling", "pooler"),
            "text_max_length": config["preprocessing"]["max_text_length"],
            "image_size": config["preprocessing"]["image_size"],
            "fusion": config["multimodal_model"]["fusion"] if model_type == "multimodal" else None,
            "text_encoder_config": (
                model.encoder.encoder.config.to_dict()
                if model_type == "text"
                else model.text_encoder.encoder.config.to_dict()
                if model_type == "multimodal"
                else None
            ),
        },
        "image_preprocessing": (
            {
                "train": "RandomResizedCrop(scale=0.8..1.0) + RandomHorizontalFlip + ImageNet normalization",
                "evaluation": "Resize(image_size) + ImageNet normalization",
                "weights": "ResNet50_Weights.IMAGENET1K_V2" if config["image_model"]["pretrained"] else None,
            }
            if model_type in {"image", "multimodal"}
            else None
        ),
        "preprocessing": copy.deepcopy(config["preprocessing"]),
        "tokenizer_name": config["text_model"]["model_name"] if model_type != "image" else None,
        "tokenizer_path": tokenizer_relative_path,
        "class_weighting": "inverse_frequency_from_training_split" if class_weights else "none",
        "sampling_strategy": imbalance_strategy,
        "class_weights": class_weights,
        "training_class_distribution": {
            label: int(train_counts.get(label, 0)) for label in labels.labels
        },
        "training_sample_count": int(len(train_frame)),
        "validation_sample_count": int(len(val_frame)),
        "pretrained_initialization": {
            "text_encoder": config["text_model"]["model_name"] if model_type in {"text", "multimodal"} else None,
            "image_encoder": (
                "ResNet50_Weights.IMAGENET1K_V2"
                if model_type in {"image", "multimodal"} and config["image_model"].get("pretrained", False)
                else "random_initialization"
                if model_type in {"image", "multimodal"}
                else None
            ),
            "load_pretrained_requested": True,
            "model_initialization_completed": True,
        },
        "scheduler": scheduler_config or {"enabled": False},
        "encoder_learning_rate": None if encoder_learning_rate is None else float(encoder_learning_rate),
        "head_learning_rate": None if head_learning_rate is None else float(head_learning_rate),
        "gradient_clip_norm": gradient_clip_norm,
        "probe_head_initialization": probe_load,
        "diversity_validation_sample_ids": (
            None if diversity_frame is None else [str(sample_id) for sample_id in diversity_frame["sample_id"].tolist()]
        ),
    }
    shutil.copyfile(config_path, output_path / "experiment_config.yaml")
    trainer = Trainer(
        model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        labels=labels.labels,
        checkpoint_dir=output_path,
        model_name=model_type,
        patience=config["training"]["early_stopping_patience"],
        checkpoint_metadata=checkpoint_metadata,
        scheduler=scheduler,
        overwrite=overwrite,
        gradient_clip_norm=gradient_clip_norm,
        diversity_loader=diversity_loader,
    )
    if probe_load is not None:
        initialization_metrics = trainer.evaluate(val_loader)
        expected_macro_f1 = float(training_config.get("expected_initialization_macro_f1", 0.3561))
        tolerance = float(training_config.get("initialization_macro_f1_tolerance", 0.03))
        feature_tolerance = float(training_config.get("feature_max_abs_tolerance", 0.01))
        reasons = []
        if feature_delta is None:
            reasons.append("masked-mean features were not compared with the probe cache")
        elif feature_delta > feature_tolerance:
            reasons.append(f"masked-mean features differ from the probe cache by {feature_delta:.6g}")
        observed_macro_f1 = float(initialization_metrics["f1_macro"])
        if abs(observed_macro_f1 - expected_macro_f1) > tolerance:
            reasons.append(
                f"initialization validation macro-F1 {observed_macro_f1:.4f} is outside "
                f"{expected_macro_f1:.4f} ± {tolerance:.4f}"
            )
        if initialization_metrics.get("single_class_prediction_collapse"):
            reasons.append("initialization does not predict every class")
        initialization_record = {
            "accepted": not reasons,
            "reasons": reasons,
            "feature_max_abs_difference": feature_delta,
            "expected_validation_macro_f1": expected_macro_f1,
            "validation": initialization_metrics,
            "probe_head": probe_load,
        }
        record_path = output_path / ("initialization_validation.json" if not reasons else "initialization_failure.json")
        with record_path.open("w", encoding="utf-8") as file:
            json.dump(initialization_record, file, indent=2, default=lambda value: value.item())
        print(
            f"Initialization validation: accuracy={initialization_metrics['accuracy']:.4f} "
            f"macro_f1={observed_macro_f1:.4f} predictions={initialization_metrics['predicted_class_distribution']}",
            flush=True,
        )
        if reasons:
            raise RuntimeError("Probe initialization check failed: " + "; ".join(reasons))
        checkpoint_metadata["initialization_validation"] = {
            "validation_accuracy": float(initialization_metrics["accuracy"]),
            "validation_macro_f1": observed_macro_f1,
            "validation_loss": float(initialization_metrics["loss"]),
            "feature_max_abs_difference": feature_delta,
            "predicted_class_distribution": initialization_metrics["predicted_class_distribution"],
            "per_class": initialization_metrics["per_class"],
        }
    if tokenizer is not None:
        tokenizer.save_pretrained(output_path / tokenizer_relative_path)
    history = trainer.fit(train_loader, val_loader, int(epochs or config["training"]["epochs"]))
    print(f"Saved {model_type} checkpoint and history under {output_path}")
    return history


def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/config.yaml", help="YAML experiment configuration")
    parser.add_argument("--epochs", type=int, help="Override configured epoch count")
    parser.add_argument("--batch-size", type=int, help="Override configured batch size")
    parser.add_argument("--max-train-samples", type=int, help="Limit train rows for a bounded smoke run")
    parser.add_argument("--max-val-samples", type=int, help="Limit validation rows for a bounded smoke run")
    parser.add_argument("--text-model-name", help="Override configured text encoder (for example, xlm-roberta-base)")
    parser.add_argument("--output-dir", help="Checkpoint output directory")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace existing run artifacts")


def training_cli(model_type: Optional[str] = None) -> None:
    parser = argparse.ArgumentParser(description="Train an IndicSent-M sentiment model.")
    if model_type is None:
        parser.add_argument("--model", choices=MODEL_TYPES, required=True)
    add_training_arguments(parser)
    args = parser.parse_args()
    run_training(
        model_type or args.model,
        config_path=args.config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        text_model_name=args.text_model_name,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
