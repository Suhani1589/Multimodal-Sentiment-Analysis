from __future__ import annotations

import sys
import hashlib
import argparse
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import ensure_dirs, load_config
from src.data.preprocessing import normalize_text
from src.data.splits import SplitConfig, create_splits
from src.data.validation import (
    build_validation_report,
    filter_valid_samples,
    save_validation_report,
    validate_dataframe,
)
from src.training.metrics import class_distribution


OFFICIAL_SPLIT_MAP = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "dev": "val",
    "test": "test",
    "testing": "test",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate metadata and create a reproducible dataset split file.")
    parser.add_argument("--output-csv", help="Write a new split file, for example data/processed/splits_v2.csv")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the selected split file (existing experiments may depend on its assignments)",
    )
    args = parser.parse_args()
    config = load_config()
    output_path = Path(args.output_csv) if args.output_csv else Path(config["dataset"]["prepared_csv"])
    if output_path.exists() and not args.overwrite:
        raise SystemExit(
            f"Split file already exists: {output_path}. Choose --output-csv for a versioned split "
            "or pass --overwrite to replace it explicitly."
        )
    ensure_dirs(config)
    dataset_config = config["dataset"]
    csv_path = Path(dataset_config["metadata_csv"])

    if not csv_path.exists():
        print(f"Dataset CSV is missing: {csv_path}")
        print("Create data/raw/metadata.csv with columns: sample_id,image_path,text,language,sentiment")
        raise SystemExit(1)

    df = pd.read_csv(csv_path)
    schema_issues = validate_dataframe(
        df,
        required_columns=dataset_config["required_columns"],
        allowed_labels=dataset_config["label_order"],
    )
    if schema_issues:
        raise SystemExit("Dataset metadata validation failed:\n- " + "\n- ".join(schema_issues))

    valid_df, rejected = filter_valid_samples(
        df,
        raw_dir=dataset_config["raw_dir"],
        required_columns=dataset_config["required_columns"],
        allowed_labels=dataset_config["label_order"],
        allow_missing_text=True,
    )
    report = build_validation_report(
        df,
        raw_dir=dataset_config["raw_dir"],
        required_columns=dataset_config["required_columns"],
        allowed_labels=dataset_config["label_order"],
    )
    report["retained_samples"] = int(len(valid_df))
    report["excluded_samples"] = int(len(rejected))
    report["excluded_records"] = rejected.to_dict(orient="records")
    report["text_missing_records"] = df.loc[
        df["text"].isna() | df["text"].astype(str).str.strip().eq(""), ["sample_id", "image_path"]
    ].assign(
        reason=lambda rows: df.loc[rows.index, "text"].map(
            lambda value: "missing_text" if pd.isna(value) else "empty_text"
        )
    ).to_dict(orient="records")
    report_path = (
        Path("results") / "dataset_validation_report.json"
        if output_path == Path(dataset_config["prepared_csv"])
        else Path("results/validation") / f"{output_path.stem}_report.json"
    )
    save_validation_report(report, report_path)
    if rejected.empty is False:
        print(f"Excluded {len(rejected)} rows with invalid IDs, labels, or images; see {report_path}")

    preprocessing = config["preprocessing"]
    valid_df["text"] = valid_df["text"].map(
        lambda value: normalize_text(
            value,
            remove_urls=preprocessing["remove_urls"],
            replace_usernames=preprocessing["replace_usernames"],
            lowercase=preprocessing["lowercase"],
        )
    )

    raw_root = Path(dataset_config["raw_dir"])
    valid_df["image_content_group"] = valid_df["image_path"].map(
        lambda value: hashlib.sha256(
            (raw_root / str(value)).read_bytes()
            if not Path(str(value)).is_absolute()
            else Path(str(value)).read_bytes()
        ).hexdigest()
    )
    if "official_split" in valid_df.columns:
        split_df = valid_df.copy()
        split_df["split"] = split_df["official_split"].astype(str).str.strip().str.lower().map(OFFICIAL_SPLIT_MAP)
        if split_df["split"].isna().any():
            unknown = sorted(split_df.loc[split_df["split"].isna(), "official_split"].dropna().astype(str).unique())
            raise ValueError(f"Unknown official split values: {unknown}")
    else:
        split_df = create_splits(
            valid_df,
            SplitConfig(
                train_size=dataset_config["train_size"],
                val_size=dataset_config["val_size"],
                test_size=dataset_config["test_size"],
                seed=config["project"]["seed"],
                group_column="image_content_group",
            ),
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(output_path, index=False)
    split_report = build_validation_report(
        split_df,
        raw_dir=dataset_config["raw_dir"],
        required_columns=dataset_config["required_columns"],
        allowed_labels=dataset_config["label_order"],
    )
    split_report["split_distribution"] = split_df["split"].value_counts().to_dict()
    split_report["retained_samples"] = int(len(split_df))
    split_report["excluded_samples"] = int(len(rejected))
    split_report["excluded_records"] = rejected.to_dict(orient="records")
    split_report["text_missing_records"] = report["text_missing_records"]
    split_report["source_missing_text_count"] = report["missing_text_count"]
    split_report["source_empty_text_count"] = report["empty_text_count"]
    split_report["modality_usable_samples"] = {
        "image_only": int(len(split_df)),
        "text_only": int(split_df["text"].notna().mul(split_df["text"].astype(str).str.strip().ne("")).sum()),
        "multimodal": int(split_df["text"].notna().mul(split_df["text"].astype(str).str.strip().ne("")).sum()),
    }
    split_report["split_class_distribution"] = {
        str(split): {str(label): int(count) for label, count in group["sentiment"].value_counts().items()}
        for split, group in split_df.groupby("split")
    }
    split_report["duplicate_image_content_groups"] = int(
        split_df["image_content_group"].duplicated(keep=False).groupby(split_df["image_content_group"]).any().sum()
    )
    split_report["image_content_group_cross_split_count"] = int(
        split_df.groupby("image_content_group")["split"].nunique().gt(1).sum()
    )
    save_validation_report(split_report, report_path)
    from src.evaluation.visualization import save_class_distribution

    class_plot_path = (
        Path(config["paths"]["plots_dir"]) / "class_distribution.png"
        if output_path == Path(dataset_config["prepared_csv"])
        else Path(config["paths"]["plots_dir"]) / f"{output_path.stem}_class_distribution.png"
    )
    save_class_distribution(split_df, class_plot_path)
    print(f"Wrote prepared split file: {output_path}")
    print(f"Wrote validation report: {report_path}")
    print(pd.Series(class_distribution(split_df["sentiment"].tolist())).to_string())


if __name__ == "__main__":
    main()
