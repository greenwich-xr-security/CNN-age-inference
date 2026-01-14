"""Shared dataset utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def filter_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Keep dorsal images with known ages and cast ages to float."""
    df = df[df["aspect"].str.contains("dorsal", case=False, na=False)]
    df = df[df["age"].notna()]
    df = df.copy()
    df["age"] = df["age"].astype(float)
    return df.reset_index(drop=True)


def filter_metadata_ssl(df: pd.DataFrame) -> pd.DataFrame:
    """Keep dorsal images; SSL does not require age labels."""
    df = df[df["aspect"].str.contains("dorsal", case=False, na=False)]
    df = df.copy()
    return df.reset_index(drop=True)


def stratified_user_split(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    stratification: str = "minorAdults",
    adult_threshold: float = 18.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-user split with optional stratification by adult/minor or age bins."""
    if "user_id" not in df.columns or "age" not in df.columns:
        raise ValueError("DataFrame must include 'user_id' and 'age' columns for stratification.")
    if stratification == "no":
        user_ids = df["user_id"].unique()
        return train_test_split(user_ids, test_size=test_size, random_state=random_state)

    per_user = (
        df.groupby("user_id")["age"]
        .mean()
        .reset_index(name="mean_age")
    )
    if per_user.empty:
        raise ValueError("No user records available after filtering; cannot stratify.")

    def _map_age_to_bin(age: float) -> int:
        if age < 13:
            return 0
        if age < 16:
            return 1
        if age < 18:
            return 2
        if age < 25:
            return 3
        if age < 35:
            return 4
        if age < 45:
            return 5
        if age < 60:
            return 6
        return 7

    if stratification == "minorAdults":
        labels = (per_user["mean_age"].to_numpy() >= adult_threshold).astype(int)
    elif stratification == "bins":
        labels = per_user["mean_age"].apply(_map_age_to_bin).to_numpy()
    else:
        raise ValueError(f"Unsupported stratification mode: {stratification}")
    user_ids = per_user["user_id"].to_numpy()

    stratify = None
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    if unique_labels.size > 1:
        n_test = np.ceil(label_counts * test_size).astype(int)
        n_train = label_counts - n_test
        if np.all(n_test >= 1) and np.all(n_train >= 1):
            stratify = labels

    train_ids, test_ids = train_test_split(
        user_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    return train_ids, test_ids
