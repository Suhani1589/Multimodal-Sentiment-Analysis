from __future__ import annotations

import pandas as pd
import pytest
from PIL import Image

from src.data.dataset import MultimodalSentimentDataset, SentimentLabels
from src.data.memotion_adapter import standardize_memotion_metadata
from src.data.preprocessing import normalize_text
from src.data.validation import build_validation_report, validate_dataframe


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
