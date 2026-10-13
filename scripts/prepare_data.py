from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import ensure_dirs, load_config
from src.data.preprocessing import normalize_text
from src.data.splits import SplitConfig, create_splits
from src.data.validation import build_validation_report, load_and_validate_metadata, save_validation_report
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
    config = load_config()
    ensure_dirs(config)
    dataset_config = config["dataset"]
    csv_path = Path(dataset_config["metadata_csv"])

    if not csv_path.exists():
        print(f"Dataset CSV is missing: {csv_path}")
        print("Create data/raw/metadata.csv with columns: sample_id,image_path,text,language,sentiment")
        raise SystemExit(1)

    df, issues = load_and_validate_metadata(
        csv_path,
        raw_dir=dataset_config["raw_dir"],
        required_columns=dataset_config["required_columns"],
        allowed_labels=dataset_config["label_order"],
    )
    report = build_validation_report(
        df,
        raw_dir=dataset_config["raw_dir"],
        required_columns=dataset_config["required_columns"],
        allowed_labels=dataset_config["label_order"],
    )
    report_path = Path("results") / "dataset_validation_report.json"
    save_validation_report(report, report_path)
    if issues:
        print("Dataset validation failed:")
        for issue in issues:
            print(f"- {issue}")
        print(f"Wrote validation report: {report_path}")
        raise SystemExit(1)

    preprocessing = config["preprocessing"]
    df["text"] = df["text"].map(
        lambda value: normalize_text(
            value,
            remove_urls=preprocessing["remove_urls"],
            replace_usernames=preprocessing["replace_usernames"],
            lowercase=preprocessing["lowercase"],
        )
    )

    if "official_split" in df.columns:
        split_df = df.copy()
        split_df["split"] = split_df["official_split"].astype(str).str.strip().str.lower().map(OFFICIAL_SPLIT_MAP)
        if split_df["split"].isna().any():
            unknown = sorted(split_df.loc[split_df["split"].isna(), "official_split"].dropna().astype(str).unique())
            raise ValueError(f"Unknown official split values: {unknown}")
    else:
        split_df = create_splits(
            df,
            SplitConfig(
                train_size=dataset_config["train_size"],
                val_size=dataset_config["val_size"],
                test_size=dataset_config["test_size"],
                seed=config["project"]["seed"],
                group_column=dataset_config.get("group_column"),
            ),
        )
    output_path = Path(dataset_config["prepared_csv"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(output_path, index=False)
    from src.evaluation.visualization import save_class_distribution

    save_class_distribution(split_df, Path(config["paths"]["plots_dir"]) / "class_distribution.png")
    split_report = build_validation_report(
        split_df,
        raw_dir=dataset_config["raw_dir"],
        required_columns=dataset_config["required_columns"],
        allowed_labels=dataset_config["label_order"],
    )
    split_report["split_distribution"] = split_df["split"].value_counts().to_dict()
    save_validation_report(split_report, report_path)
    print(f"Wrote prepared split file: {output_path}")
    print(f"Wrote validation report: {report_path}")
    print(pd.Series(class_distribution(split_df["sentiment"].tolist())).to_string())


if __name__ == "__main__":
    main()
