from __future__ import annotations

import json

import pandas as pd
import pytest
import torch
from torch import nn

from src.config import load_config
from src.evaluation import checkpoint_evaluation as checkpoint_eval
from src.evaluation.evaluate import build_comparison_table
from src.training.metrics import compute_classification_metrics
from src.training.trainer import Trainer
from src.inference.checkpoint import LoadedPredictor, load_checkpoint_predictor, predict_batch
from src.explainability.attribution import grad_cam, token_gradient_attribution
from scripts.build_final_report import build_report
from src.training.sampling import build_training_sampler
from src.training.overfit_diagnostic import select_overfit_subset
from src.training.loaders import build_eval_dataloader


def test_metrics_keys() -> None:
    metrics = compute_classification_metrics([0, 1, 2], [0, 2, 2], ["positive", "neutral", "negative"])
    assert metrics["accuracy"] == 2 / 3
    assert "f1_macro" in metrics
    assert len(metrics["confusion_matrix"]) == 3


def test_metrics_keep_fixed_class_order_when_class_is_absent() -> None:
    metrics = compute_classification_metrics([0, 0], [0, 1], ["positive", "neutral", "negative"])
    assert metrics["confusion_matrix"] == [[1, 1, 0], [0, 0, 0], [0, 0, 0]]
    assert "negative" in metrics["per_class"]
    assert metrics["predicted_class_distribution"] == {"positive": 1, "neutral": 1, "negative": 0}


def test_comparison_table() -> None:
    table = build_comparison_table({"Text-only": {"accuracy": 1.0, "precision_macro": 1.0, "recall_macro": 1.0, "f1_macro": 1.0, "f1_weighted": 1.0}})
    assert list(table.columns) == ["model", "accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]
    assert table.iloc[0]["model"] == "Text-only"


def test_evaluation_selects_test_split_only() -> None:
    frame = pd.DataFrame({"sample_id": ["tr", "va", "te"], "split": ["train", "val", "test"]})
    selected = checkpoint_eval.held_out_test_rows(frame)
    assert selected["sample_id"].tolist() == ["te"]


def test_evaluation_skips_missing_modalities(tmp_path) -> None:
    from PIL import Image

    Image.new("RGB", (4, 4)).save(tmp_path / "ok.png")
    frame = pd.DataFrame(
        [
            {"sample_id": "ok", "text": "hello", "image_path": "ok.png"},
            {"sample_id": "no-text", "text": "  ", "image_path": "ok.png"},
            {"sample_id": "no-image", "text": "hello", "image_path": "missing.png"},
        ]
    )
    usable, skipped = checkpoint_eval.select_evaluation_rows(frame, model_type="multimodal", raw_dir=tmp_path)
    assert usable["sample_id"].tolist() == ["ok"]
    assert skipped == [
        {"sample_id": "no-text", "reason": "missing_text"},
        {"sample_id": "no-image", "reason": "missing_image"},
    ]


def test_checkpoint_reconstruction_evaluates_predictions(tmp_path, monkeypatch) -> None:
    from transformers import BertConfig

    class TinyTokenizer:
        def __call__(self, text, **kwargs):
            return {
                "input_ids": torch.tensor([[len(text), 1]], dtype=torch.long),
                "attention_mask": torch.ones((1, 2), dtype=torch.long),
            }

    monkeypatch.setattr(checkpoint_eval, "build_tokenizer", lambda *args, **kwargs: TinyTokenizer())
    config = load_config()
    config["dataset"]["raw_dir"] = str(tmp_path)
    config["text_model"]["model_name"] = "tiny-bert-test"
    encoder_config = BertConfig(
        vocab_size=32, hidden_size=8, num_hidden_layers=1, num_attention_heads=2, intermediate_size=16
    ).to_dict()
    from src.model_factory import build_model

    checkpoint = tmp_path / "text_best.pt"
    source_model = build_model(
        "text", config, load_pretrained=False, text_encoder_config=encoder_config
    )
    torch.save(
        {
            "model_state_dict": source_model.state_dict(),
            "model_type": "text",
            "labels": ["positive", "neutral", "negative"],
            "config": config,
            "model_config": {"text_encoder_config": encoder_config},
        },
        checkpoint,
    )
    frame = pd.DataFrame(
        [
            {"sample_id": "t1", "text": "good", "image_path": "unused", "sentiment": "positive"},
            {"sample_id": "t2", "text": "bad", "image_path": "unused", "sentiment": "neutral"},
        ]
    )
    metrics, predictions = checkpoint_eval.evaluate_checkpoint(
        "text", checkpoint, frame, runtime_config=config, device=torch.device("cpu")
    )
    assert metrics["test_samples_evaluated"] == 2
    assert metrics["class_order"] == ["positive", "neutral", "negative"]
    assert predictions.columns.tolist() == [
        "sample_id", "true_label", "predicted_label", "probability_positive", "probability_neutral", "probability_negative"
    ]
    metric_dir = tmp_path / "metrics"
    prediction_dir = tmp_path / "predictions"
    checkpoint_eval.save_evaluation_outputs(
        "text", metrics, predictions, metrics_dir=metric_dir, predictions_dir=prediction_dir
    )
    assert json.loads((metric_dir / "text_results.json").read_text())["test_samples_evaluated"] == 2
    assert pd.read_csv(prediction_dir / "text_predictions.csv").shape[0] == 2
    assert pd.read_csv(metric_dir / "text_confusion_matrix.csv").shape == (3, 4)


def test_trainer_saves_checkpoint_metadata_and_protects_existing_run(tmp_path) -> None:
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(2, 4)
            self.classifier = nn.Linear(4, 3)

        def forward(self, input_ids, **kwargs):
            return self.classifier(self.encoder(input_ids.float()))

    rows = [
        {"input_ids": torch.tensor([1.0, 0.0]), "label": torch.tensor(0)},
        {"input_ids": torch.tensor([0.0, 1.0]), "label": torch.tensor(1)},
        {"input_ids": torch.tensor([1.0, 1.0]), "label": torch.tensor(2)},
    ]
    loader = torch.utils.data.DataLoader(rows, batch_size=3)
    model = TinyModel()
    trainer = Trainer(
        model,
        optimizer=torch.optim.Adam(model.parameters()),
        criterion=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        labels=["positive", "neutral", "negative"],
        checkpoint_dir=tmp_path,
        model_name="text",
        patience=1,
        checkpoint_metadata={"seed": 7, "tokenizer_name": "tiny"},
    )
    history = trainer.fit(loader, loader, 1)
    payload = torch.load(tmp_path / "text_best.pt", map_location="cpu", weights_only=False)
    assert len(history) == 1
    assert history[0]["learning_rate"] > 0
    assert history[0]["epoch_duration_seconds"] >= 0
    assert history[0]["train_metric_scope"] == "online_minibatch_outputs_before_optimizer_update"
    assert history[0]["validation_metric_scope"] == "post_epoch_eval_mode"
    assert history[0]["train"]["predicted_class_distribution"]
    for module_name in ("text_encoder", "classifier"):
        diagnostic = history[0]["train"]["gradient_diagnostics"][module_name]
        assert diagnostic["gradient_parameter_tensors"] > 0
        assert diagnostic["representative_gradient_norm"] > 0
        assert diagnostic["representative_update_abs_mean"] > 0
    assert payload["selection_metric"] == "validation_macro_f1"
    assert payload["seed"] == 7
    assert (tmp_path / "text_history.json").exists()
    with pytest.raises(FileExistsError):
        trainer.fit(loader, loader, 1)


def test_trainer_keeps_earliest_checkpoint_when_validation_macro_f1_ties(tmp_path, monkeypatch) -> None:
    model = nn.Linear(2, 3)
    trainer = Trainer(
        model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        criterion=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        labels=["positive", "neutral", "negative"],
        checkpoint_dir=tmp_path,
        model_name="text",
    )
    metrics = {
        "accuracy": 0.6,
        "precision_macro": 0.2,
        "recall_macro": 1 / 3,
        "f1_macro": 0.25,
        "confusion_matrix": [[6, 0, 0], [3, 0, 0], [1, 0, 0]],
    }
    monkeypatch.setattr(trainer, "train_epoch", lambda _loader: {"loss": 1.0})
    monkeypatch.setattr(trainer, "evaluate", lambda _loader: dict(metrics))
    history = trainer.fit([], [], 3)
    checkpoint = torch.load(tmp_path / "text_best.pt", map_location="cpu", weights_only=False)
    run = json.loads((tmp_path / "text_run.json").read_text())
    assert len(history) == 3
    assert checkpoint["selection_metric"] == "validation_macro_f1"
    assert checkpoint["best_epoch"] == run["best_epoch"] == 1


def test_token_gradient_attribution_masks_padding_and_returns_token_scores() -> None:
    class TinyTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = nn.Embedding(16, 4)

    class TinyEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = TinyTransformer()

        def forward(self, input_ids, attention_mask):
            values = self.encoder.embeddings(input_ids)
            mask = attention_mask.unsqueeze(-1).to(values.dtype)
            return (values * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)

    class TinyTextModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = TinyEncoder()
            self.classifier = nn.Linear(4, 3)

        def forward(self, input_ids, attention_mask):
            return self.classifier(self.encoder(input_ids, attention_mask))

    model = TinyTextModel()
    ids = torch.tensor([[1, 2, 0]])
    mask = torch.tensor([[1, 1, 0]])
    result = token_gradient_attribution(model, ids, mask)
    assert result["method"] == "gradient_times_embedding"
    assert len(result["scores"]) == 3
    assert result["scores"][2] == 0


def test_grad_cam_returns_normalized_spatial_map() -> None:
    class TinyResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer4 = nn.Sequential(nn.Conv2d(3, 4, kernel_size=1), nn.ReLU())

        def forward(self, image):
            return self.layer4(image).mean(dim=(-2, -1))

    class TinyImageEncoder(nn.Module):
        output_dim = 4

        def __init__(self):
            super().__init__()
            self.encoder = TinyResNet()

        def forward(self, image):
            return self.encoder(image)

    class TinyImageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = TinyImageEncoder()
            self.classifier = nn.Linear(4, 3)

        def forward(self, image):
            return self.classifier(self.encoder(image))

    result = grad_cam(TinyImageModel(), torch.rand(1, 3, 12, 10))
    assert result["method"] == "grad_cam"
    assert result["heatmap"].shape == (12, 10)
    assert float(result["heatmap"].min()) >= 0
    assert float(result["heatmap"].max()) <= 1


def test_prediction_helper_returns_label_probabilities(tmp_path) -> None:
    class TinyModel(nn.Module):
        def forward(self, input_ids, **kwargs):
            return input_ids.float()

    predictor = LoadedPredictor(
        model_type="text",
        model=TinyModel(),
        labels=["positive", "neutral", "negative"],
        config={},
        tokenizer=None,
        device=torch.device("cpu"),
        checkpoint_path=tmp_path / "model.pt",
    )
    result = predict_batch(predictor, {"input_ids": torch.tensor([[3.0, 1.0, 0.0]])})
    assert result["predicted_sentiment"] == "positive"
    assert set(result["probabilities"]) == {"positive", "neutral", "negative"}
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)


def test_checkpoint_loader_reports_missing_model_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        load_checkpoint_predictor("image", tmp_path / "missing.pt")


def test_weighted_sampler_balances_only_training_examples_reproducibly() -> None:
    labels = ["positive"] * 900 + ["neutral"] * 90 + ["negative"] * 10
    sampler_a = build_training_sampler(
        labels, ["positive", "neutral", "negative"], strategy="weighted_sampler", seed=17
    )
    sampler_b = build_training_sampler(
        labels, ["positive", "neutral", "negative"], strategy="weighted_sampler", seed=17
    )
    draw_a = list(sampler_a)
    draw_b = list(sampler_b)
    assert draw_a == draw_b
    predicted_distribution = {label: 0 for label in ("positive", "neutral", "negative")}
    for index in draw_a:
        predicted_distribution[labels[index]] += 1
    assert all(280 <= count <= 390 for count in predicted_distribution.values())


def test_weighted_sampler_requires_each_class_in_training_split() -> None:
    with pytest.raises(ValueError, match="missing"):
        build_training_sampler(
            ["positive", "neutral"],
            ["positive", "neutral", "negative"],
            strategy="weighted_sampler",
            seed=1,
        )


def test_overfit_diagnostic_subset_is_deterministic_and_training_only() -> None:
    frame = pd.DataFrame(
        [
            {
                "sample_id": f"{split}-{label}-{index}",
                "split": split,
                "sentiment": label,
                "text": f"{label} sample {index}",
            }
            for split in ("train", "val", "test")
            for label in ("positive", "neutral", "negative")
            for index in range(4)
        ]
    )
    first = select_overfit_subset(frame, samples_per_class=2, labels=["positive", "neutral", "negative"], seed=6)
    second = select_overfit_subset(frame, samples_per_class=2, labels=["positive", "neutral", "negative"], seed=6)
    assert first.sample_id.tolist() == second.sample_id.tolist()
    assert first.split.unique().tolist() == ["train"]
    assert first.sentiment.value_counts().to_dict() == {"positive": 2, "neutral": 2, "negative": 2}


def test_eval_loader_does_not_advance_global_dropout_rng() -> None:
    dataset = torch.utils.data.TensorDataset(torch.arange(8))
    torch.manual_seed(101)
    expected_next_random = torch.rand(4)
    torch.manual_seed(101)
    loader = build_eval_dataloader(dataset, batch_size=2, seed=24)
    next(iter(loader))
    observed_next_random = torch.rand(4)
    assert torch.equal(observed_next_random, expected_next_random)


def test_final_report_marks_missing_runs_and_computes_majority_reference(tmp_path) -> None:
    splits = tmp_path / "splits.csv"
    pd.DataFrame(
        {
            "split": ["train", "val", "test", "test", "test"],
            "sentiment": ["positive", "neutral", "positive", "positive", "negative"],
        }
    ).to_csv(splits, index=False)
    run_dirs = {name: tmp_path / "missing" / name for name in ("text", "image", "multimodal")}
    evaluation_dirs = {name: tmp_path / "missing_eval" / name for name in ("text", "image", "multimodal")}
    report, histories = build_report(run_dirs=run_dirs, evaluation_dirs=evaluation_dirs, split_path=splits)
    assert report["majority_class_baseline"]["target_class"] == "positive"
    assert report["majority_class_baseline"]["accuracy"] == pytest.approx(2 / 3)
    assert all(report["models"][name]["status"] == "not_evaluated" for name in run_dirs)
    assert all(report["models"][name]["accuracy"] is None for name in run_dirs)
    assert histories == {"text": [], "image": [], "multimodal": []}
