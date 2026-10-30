from __future__ import annotations

import pandas as pd
import pytest
import torch
from PIL import Image

from src.data.dataset import MultimodalSentimentDataset, SentimentLabels
from src.data.memotion_adapter import standardize_memotion_metadata
from src.data.preprocessing import normalize_text
from src.data.validation import build_validation_report, filter_valid_samples, validate_dataframe
from src.data.splits import SplitConfig, create_splits


def test_normalize_text_preserves_emoji() -> None:
    assert normalize_text("Hi   @abc https://x.test 😂") == "Hi @user 😂"


def test_validate_dataframe_rejects_bad_label() -> None:
    df = pd.DataFrame(
        [{"sample_id": "1", "image_path": "images/1.jpg", "text": "ok", "language": "hi", "sentiment": "mixed"}]
    )
    issues = validate_dataframe(
        df,
        required_columns=["sample_id", "image_path", "text", "language", "sentiment"],
        allowed_labels=["positive", "neutral", "negative"],
    )
    assert issues


def test_dataset_loads_image(tmp_path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (8, 8), color="red").save(image_dir / "1.jpg")
    df = pd.DataFrame(
        [{"sample_id": "1", "image_path": "images/1.jpg", "text": "great", "language": "en", "sentiment": "positive"}]
    )
    ds = MultimodalSentimentDataset(df, raw_dir=tmp_path, labels=SentimentLabels(["positive", "neutral", "negative"]), mode="image")
    item = ds[0]
    assert item["image"].mode == "RGB"
    assert item["label"].item() == 0


def test_dataset_keeps_text_and_sentiment_from_the_same_source_row() -> None:
    class MarkerTokenizer:
        def __call__(self, text, **kwargs):
            marker = 11 if text.startswith("bright") else 22
            return {
                "input_ids": torch.tensor([[marker, 0]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 0]], dtype=torch.long),
            }

    frame = pd.DataFrame(
        [
            {"sample_id": "p", "text": "bright and happy", "sentiment": "positive", "language": "hinglish"},
            {"sample_id": "n", "text": "awful and sad", "sentiment": "negative", "language": "hinglish"},
        ]
    )
    dataset = MultimodalSentimentDataset(
        frame,
        raw_dir=".",
        labels=SentimentLabels(["positive", "neutral", "negative"]),
        tokenizer=MarkerTokenizer(),
        mode="text",
    )
    positive, negative = dataset[0], dataset[1]
    assert positive["sample_id"] == "p" and positive["text"].startswith("bright")
    assert positive["input_ids"][0].item() == 11 and positive["label"].item() == 0
    assert negative["sample_id"] == "n" and negative["text"].startswith("awful")
    assert negative["input_ids"][0].item() == 22 and negative["label"].item() == 2


def test_dataset_missing_image_raises(tmp_path) -> None:
    df = pd.DataFrame(
        [{"sample_id": "1", "image_path": "images/missing.jpg", "text": "great", "language": "en", "sentiment": "positive"}]
    )
    ds = MultimodalSentimentDataset(df, raw_dir=tmp_path, labels=SentimentLabels(["positive", "neutral", "negative"]), mode="image")
    with pytest.raises(FileNotFoundError):
        _ = ds[0]


def test_memotion_adapter_maps_schema_and_labels(tmp_path) -> None:
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame(
        [
            {
                "image_name": "1.jpg",
                "text_corrected": "Great meme",
                "overall_sentiment": "very_positive",
            }
        ]
    ).to_csv(csv_path, index=False)
    df = standardize_memotion_metadata(csv_path, image_prefix="images", default_language="hinglish")
    assert list(df.columns) == ["sample_id", "image_path", "text", "language", "sentiment"]
    assert df.iloc[0]["image_path"] == "images/1.jpg"
    assert df.iloc[0]["sentiment"] == "positive"


def test_memotion_adapter_recovers_missing_corrected_text_from_ocr(tmp_path) -> None:
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame(
        [{"image_name": "1.jpg", "text_corrected": None, "text_ocr": "OCR only", "overall_sentiment": "neutral"}]
    ).to_csv(csv_path, index=False)
    df = standardize_memotion_metadata(csv_path, image_prefix="images")
    assert df.iloc[0]["text"] == "OCR only"


def test_sentiment_label_order_is_fixed() -> None:
    labels = SentimentLabels(["positive", "neutral", "negative"])
    assert labels.encode("positive") == 0
    assert labels.encode("neutral") == 1
    assert labels.encode("negative") == 2
    assert [labels.decode(index) for index in range(3)] == ["positive", "neutral", "negative"]


def test_grouped_splits_are_reproducible_and_keep_groups_together() -> None:
    frame = pd.DataFrame(
        [
            {"sample_id": f"{group}-{copy}", "sentiment": label, "image_content_group": f"image-{group}"}
            for group, label in enumerate(["positive", "neutral", "negative"] * 10)
            for copy in range(2)
        ]
    )
    config = SplitConfig(0.7, 0.15, 0.15, seed=29, group_column="image_content_group")
    first = create_splits(frame, config)
    second = create_splits(frame, config)
    assert first.set_index("sample_id")["split"].sort_index().equals(
        second.set_index("sample_id")["split"].sort_index()
    )
    assert first.groupby("image_content_group")["split"].nunique().max() == 1


def test_validation_separates_missing_and_empty_text(tmp_path) -> None:
    df = pd.DataFrame(
        [
            {"sample_id": "1", "image_path": "images/1.jpg", "text": None, "language": "hi", "sentiment": "positive"},
            {"sample_id": "2", "image_path": "images/2.jpg", "text": "   ", "language": "hi", "sentiment": "neutral"},
        ]
    )
    report = build_validation_report(
        df,
        raw_dir=tmp_path,
        required_columns=["sample_id", "image_path", "text", "language", "sentiment"],
        allowed_labels=["positive", "neutral", "negative"],
    )
    assert report["missing_text_count"] == 1
    assert report["empty_text_count"] == 1


def test_filter_keeps_image_only_rows_when_requested(tmp_path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (2, 2)).save(image_dir / "1.jpg")
    df = pd.DataFrame(
        [{"sample_id": "1", "image_path": "images/1.jpg", "text": None, "language": "hi", "sentiment": "positive"}]
    )
    retained, excluded = filter_valid_samples(
        df,
        raw_dir=tmp_path,
        required_columns=["sample_id", "image_path", "text", "language", "sentiment"],
        allowed_labels=["positive", "neutral", "negative"],
        allow_missing_text=True,
    )
    assert len(retained) == 1
    assert excluded.empty


def test_validation_report_counts_problems(tmp_path) -> None:
    df = pd.DataFrame(
        [
            {"sample_id": "1", "image_path": "images/1.jpg", "text": "", "language": "hi", "sentiment": "positive"},
            {"sample_id": "1", "image_path": "images/1.jpg", "text": "ok", "language": "hi", "sentiment": "mixed"},
        ]
    )
    report = build_validation_report(
        df,
        raw_dir=tmp_path,
        required_columns=["sample_id", "image_path", "text", "language", "sentiment"],
        allowed_labels=["positive", "neutral", "negative"],
    )
    assert report["total_samples"] == 2
    assert report["missing_image_count"] == 2
    assert report["duplicate_count"] == 2
    assert report["invalid_sentiment_count"] == 1
