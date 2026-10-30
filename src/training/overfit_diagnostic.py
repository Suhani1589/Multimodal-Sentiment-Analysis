from __future__ import annotations

import pandas as pd


def select_overfit_subset(
    frame: pd.DataFrame,
    *,
    samples_per_class: int,
    labels: list[str],
    seed: int,
) -> pd.DataFrame:
    """Select a deterministic, class-represented subset from training rows only."""
    if samples_per_class < 1:
        raise ValueError("samples_per_class must be at least one")
    required = {"split", "sentiment", "text", "sample_id"}
    missing_columns = required.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"Training data is missing columns: {sorted(missing_columns)}")
    train = frame.loc[frame["split"].astype(str).str.lower().eq("train")].copy()
    has_text = train["text"].notna() & train["text"].astype(str).str.strip().ne("")
    train = train.loc[has_text]
    selected = []
    for class_index, label in enumerate(labels):
        class_rows = train.loc[train["sentiment"].astype(str).str.lower().eq(label)]
        if len(class_rows) < samples_per_class:
            raise ValueError(
                f"Need {samples_per_class} usable training rows for {label}, found {len(class_rows)}"
            )
        selected.append(class_rows.sample(n=samples_per_class, random_state=seed + class_index))
    subset = pd.concat(selected).sort_index().copy()
    if subset["sample_id"].duplicated().any():
        raise ValueError("Selected training subset contains duplicate sample IDs")
    if not subset["split"].astype(str).str.lower().eq("train").all():
        raise ValueError("Overfit subset selection included non-training data")
    return subset
