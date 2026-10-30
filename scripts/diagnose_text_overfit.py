from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data.dataset import MultimodalSentimentDataset, SentimentLabels
from src.model_factory import build_model
from src.text.tokenizer import build_tokenizer
from src.training.losses import build_cross_entropy_loss
from src.training.loaders import build_eval_dataloader
from src.training.metrics import compute_classification_metrics
from src.training.overfit_diagnostic import select_overfit_subset
from src.utils import get_device, seed_everything


def _gradient_norm(parameters: list[torch.nn.Parameter]) -> tuple[float, bool]:
    norms = [parameter.grad.detach().float().norm() for parameter in parameters if parameter.grad is not None]
    if not norms:
        return 0.0, False
    value = torch.stack(norms).norm()
    return float(value.item()), bool(torch.isfinite(value).item())


def _evaluate_subset(model, loader, criterion, device, labels) -> dict[str, Any]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    loss_sum = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            logits = model(**batch)
            loss_sum += float(criterion(logits, batch["label"]).item()) * len(batch["label"])
            y_true.extend(batch["label"].cpu().tolist())
            y_pred.extend(logits.argmax(dim=-1).cpu().tolist())
    metrics = compute_classification_metrics(y_true, y_pred, labels)
    metrics["loss"] = loss_sum / max(len(y_true), 1)
    return metrics


def run_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_steps < 1 or args.batch_size < 1 or args.log_every < 1:
        raise ValueError("max-steps, batch-size, and log-every must be positive")
    output_path = Path(args.output_json)
    if output_path.exists():
        raise FileExistsError(f"Diagnostic output already exists: {output_path}")

    config = load_config(args.config)
    config["text_model"]["model_name"] = args.text_model_name
    encoder_learning_rate = (
        args.encoder_learning_rate
        if args.encoder_learning_rate is not None
        else float(config["training"]["learning_rate"])
    )
    classifier_learning_rate = (
        args.classifier_learning_rate
        if args.classifier_learning_rate is not None
        else float(config["training"]["learning_rate"])
    )
    weight_decay = args.weight_decay if args.weight_decay is not None else float(config["training"]["weight_decay"])
    seed = int(config["project"]["seed"] if args.seed is None else args.seed)
    seed_everything(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)

    labels = [str(label).lower() for label in config["dataset"]["label_order"]]
    frame = pd.read_csv(config["dataset"]["prepared_csv"])
    subset = select_overfit_subset(
        frame,
        samples_per_class=args.samples_per_class,
        labels=labels,
        seed=seed,
    )
    tokenizer = build_tokenizer(args.text_model_name)
    label_mapping = SentimentLabels(labels)
    dataset = MultimodalSentimentDataset(
        subset,
        raw_dir=config["dataset"]["raw_dir"],
        labels=label_mapping,
        tokenizer=tokenizer,
        max_length=config["preprocessing"]["max_text_length"],
        mode="text",
    )
    train_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        dataset,
        batch_size=min(args.batch_size, len(dataset)),
        shuffle=True,
        generator=train_generator,
        num_workers=0,
    )
    eval_loader = build_eval_dataloader(
        dataset,
        batch_size=min(args.batch_size, len(dataset)),
        seed=seed,
        num_workers=0,
    )

    model = build_model("text", config, load_pretrained=True)
    encoder_parameters = [parameter for parameter in model.encoder.parameters() if parameter.requires_grad]
    classifier_parameters = [parameter for parameter in model.classifier.parameters() if parameter.requires_grad]
    if not encoder_parameters or not classifier_parameters:
        raise RuntimeError("Expected both the text encoder and classifier to have trainable parameters")
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": encoder_learning_rate},
            {"params": classifier_parameters, "lr": classifier_learning_rate},
        ],
        weight_decay=weight_decay,
    )
    criterion = build_cross_entropy_loss(device=get_device(config["project"]["device"]))
    device = get_device(config["project"]["device"])
    model.to(device)

    probe_parameters = {
        "encoder": encoder_parameters[0],
        "classifier": classifier_parameters[0],
    }
    before_values = {
        name: parameter.detach().reshape(-1)[:256].clone()
        for name, parameter in probe_parameters.items()
    }
    diagnostics = []
    optimizer_updates = 0
    iterator = iter(train_loader)
    model.train()
    for step in range(1, args.max_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(**batch)
        loss = criterion(logits, batch["label"])
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at diagnostic step {step}: {float(loss.detach())}")
        loss.backward()
        measure_gradients = step == 1 or step == args.max_steps
        if measure_gradients:
            encoder_norm, encoder_finite = _gradient_norm(encoder_parameters)
            classifier_norm, classifier_finite = _gradient_norm(classifier_parameters)
        else:
            encoder_norm = classifier_norm = None
            encoder_finite = classifier_finite = None
        optimizer.step()
        optimizer_updates += 1

        if step == 1 or step % args.log_every == 0 or step == args.max_steps:
            subset_metrics = _evaluate_subset(model, eval_loader, criterion, device, labels)
            report = {
                "step": step,
                "last_batch_loss": float(loss.detach().item()),
                "subset_train_loss": subset_metrics["loss"],
                "subset_train_accuracy": subset_metrics["accuracy"],
                "subset_train_macro_f1": subset_metrics["f1_macro"],
                "per_class": subset_metrics["per_class"],
                "true_class_distribution": subset_metrics["true_class_distribution"],
                "predicted_class_distribution": subset_metrics["predicted_class_distribution"],
                "confusion_matrix": subset_metrics["confusion_matrix"],
                "gradient_norms": {"encoder": encoder_norm, "classifier": classifier_norm},
                "gradient_finite": {"encoder": encoder_finite, "classifier": classifier_finite},
            }
            diagnostics.append(report)
            print(
                f"step={step} subset_loss={report['subset_train_loss']:.4f} "
                f"accuracy={report['subset_train_accuracy']:.3f} "
                f"macro_f1={report['subset_train_macro_f1']:.3f} "
                f"predictions={report['predicted_class_distribution']} "
                f"grad_norms={report['gradient_norms']}"
            )
            model.train()

    parameter_updates = {}
    for name, parameter in probe_parameters.items():
        delta = parameter.detach().reshape(-1)[:256] - before_values[name]
        parameter_updates[name] = {
            "parameter_name": next(
                parameter_name
                for parameter_name, candidate in (
                    list(model.encoder.named_parameters()) if name == "encoder" else list(model.classifier.named_parameters())
                )
                if candidate is parameter
            ),
            "first_256_abs_mean_delta": float(delta.abs().mean().item()),
            "first_256_update_norm": float(delta.norm().item()),
        }

    sample_examples = []
    for _, row in subset.groupby("sentiment", sort=False).head(2).iterrows():
        encoded = tokenizer(
            str(row["text"]),
            truncation=True,
            padding="max_length",
            max_length=config["preprocessing"]["max_text_length"],
            return_tensors="pt",
        )
        active_ids = encoded["input_ids"][0][encoded["attention_mask"][0].bool()].tolist()
        sample_examples.append(
            {
                "sample_id": str(row["sample_id"]),
                "text_preview": str(row["text"])[:120],
                "encoded_label": label_mapping.encode(row["sentiment"]),
                "decoded_label": label_mapping.decode(label_mapping.encode(row["sentiment"])),
                "token_count_with_special_tokens": len(active_ids),
                "token_preview": tokenizer.convert_ids_to_tokens(active_ids)[:12],
            }
        )

    result = {
        "model_name": args.text_model_name,
        "seed": seed,
        "training_split_only": True,
        "weighted_sampling": False,
        "class_weighted_loss": False,
        "sample_count": len(subset),
        "samples_per_class": args.samples_per_class,
        "class_distribution": subset["sentiment"].value_counts().reindex(labels, fill_value=0).to_dict(),
        "sample_examples": sample_examples,
        "max_steps": args.max_steps,
        "optimizer": "AdamW",
        "encoder_learning_rate": encoder_learning_rate,
        "classifier_learning_rate": classifier_learning_rate,
        "weight_decay": weight_decay,
        "gradient_clipping": None,
        "trainable_encoder_parameter_tensors": len(encoder_parameters),
        "trainable_classifier_parameter_tensors": len(classifier_parameters),
        "optimizer_updates": optimizer_updates,
        "parameter_updates": parameter_updates,
        "diagnostics": diagnostics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    print(f"Saved training-only overfit diagnostic to {output_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Test whether XLM-R can memorize a tiny training-only subset.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--text-model-name", default="xlm-roberta-base")
    parser.add_argument("--samples-per-class", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--encoder-learning-rate", type=float)
    parser.add_argument("--classifier-learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-json", default="results/diagnostics/text-overfit-xlmroberta.json")
    run_diagnostic(parser.parse_args())


if __name__ == "__main__":
    main()
