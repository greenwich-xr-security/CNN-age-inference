"""Shared dataset utilities."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import KFold


def filter_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Keep dorsal images with known ages and cast ages to float."""
    df = df[df["aspect"].str.contains("dorsal", case=False, na=False)]
    df = df[df["age"].notna()]
    df = df.copy()
    df["age"] = df["age"].astype(float)
    return df.reset_index(drop=True)


def build_kfold_user_splits(
    df: pd.DataFrame,
    *,
    k: int,
    random_state: int = 42,
) -> list[list[str]]:
    """Return k disjoint validation folds of user_ids."""
    if k < 2:
        raise ValueError("k must be at least 2 for k-fold splits.")
    if "user_id" not in df.columns:
        raise ValueError("DataFrame must include 'user_id' for k-fold splits.")

    user_ids = df["user_id"].astype(str).unique()
    if user_ids.size < k:
        raise ValueError(f"Not enough users ({user_ids.size}) to build {k} folds.")

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
    seed: int,
) -> Path:
    path = Path(path)
    payload = {
        "k": len(folds),
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
