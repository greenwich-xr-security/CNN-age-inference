"""Shared dataset utilities."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


def filter_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Keep dorsal images with known ages and cast ages to float."""
    df = df[df["aspect"].str.contains("dorsal", case=False, na=False)]
    df = df[df["age"].notna()]
    df = df.copy()
    df["age"] = df["age"].astype(float)
    return df.reset_index(drop=True)


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


def build_kfold_user_splits(
    df: pd.DataFrame,
    *,
    k: int,
    random_state: int = 42,
    stratification: str = "minorAdults",
    adult_threshold: float = 18.0,
) -> list[list[str]]:
    """Return k disjoint validation folds of user_ids."""
    if k < 2:
        raise ValueError("k must be at least 2 for k-fold splits.")
    if "user_id" not in df.columns or "age" not in df.columns:
        raise ValueError("DataFrame must include 'user_id' and 'age' columns for k-fold splits.")

    per_user = (
        df.groupby("user_id")["age"]
        .mean()
        .reset_index(name="mean_age")
    )
    if per_user.empty:
        raise ValueError("No user records available after filtering; cannot build folds.")

    user_ids = per_user["user_id"].astype(str).to_numpy()
    if user_ids.size < k:
        raise ValueError(f"Not enough users ({user_ids.size}) to build {k} folds.")

    labels = None
    if stratification == "minorAdults":
        labels = (per_user["mean_age"].to_numpy() >= adult_threshold).astype(int)
    elif stratification == "bins":
        labels = per_user["mean_age"].apply(_map_age_to_bin).to_numpy()
    elif stratification != "no":
        raise ValueError(f"Unsupported stratification mode: {stratification}")

    if labels is None:
        splitter = KFold(n_splits=k, shuffle=True, random_state=random_state)
        splits = splitter.split(user_ids)
    else:
        try:
            splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
            splits = splitter.split(user_ids, labels)
        except ValueError:
            splitter = KFold(n_splits=k, shuffle=True, random_state=random_state)
            splits = splitter.split(user_ids)

    folds: list[list[str]] = []
    for _, val_idx in splits:
        folds.append(user_ids[val_idx].tolist())
    return folds


def save_kfold_splits(
    folds: list[list[str]],
    path: str | Path,
    *,
    stratification: str,
    seed: int,
) -> Path:
    path = Path(path)
    payload = {
        "k": len(folds),
        "stratification": stratification,
        "seed": int(seed),
        "folds": folds,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
    return path


def load_kfold_splits(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    if "folds" not in payload:
        raise ValueError("Fold file is missing 'folds'.")
    return payload
