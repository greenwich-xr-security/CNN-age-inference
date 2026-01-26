"""Shared dataset utilities."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import KFold


def dataset_composition_stats(df: pd.DataFrame) -> dict[str, int]:
    """Return dataset composition stats (samples/users per source)."""
    stats: dict[str, int] = {
        "total_samples": int(len(df)),
    }
    if "user_id" in df.columns:
        stats["total_users"] = int(df["user_id"].nunique())
    else:
        stats["total_users"] = 0

    if "source" in df.columns and not df.empty:
        sample_counts = df["source"].value_counts()
        if "user_id" in df.columns:
            user_counts = df.groupby("source")["user_id"].nunique()
        else:
            user_counts = pd.Series(dtype=int)
        sources = sorted(set(sample_counts.index) | set(user_counts.index), key=lambda x: str(x))
        for source in sources:
            key = str(source).strip().lower().replace(" ", "_")
            stats[f"source_{key}_samples"] = int(sample_counts.get(source, 0))
            stats[f"source_{key}_users"] = int(user_counts.get(source, 0))
    return stats


def _limit_samples_per_user(
    df: pd.DataFrame,
    max_samples_per_user: int | None,
) -> pd.DataFrame:
    if max_samples_per_user is None:
        return df
    try:
        max_samples = int(max_samples_per_user)
    except (TypeError, ValueError):
        return df
    if max_samples <= 0:
        return df
    if "user_id" not in df.columns:
        return df

    grouped = df.groupby("user_id", sort=False)
    limited = grouped.head(max_samples)
    dropped = len(df) - len(limited)
    if dropped > 0:
        over_count = int((grouped.size() > max_samples).sum())
        print(
            f"Per-user sample cap applied ({max_samples} samples/user): "
            f"dropped {dropped} samples from {over_count} users."
        )
    return limited


def filter_metadata(
    df: pd.DataFrame,
    *,
    max_samples_per_user: int | None = 16,
) -> pd.DataFrame:
    """Keep dorsal images with known ages, cast ages to float, and cap samples per user."""
    df = df[df["aspect"].str.contains("dorsal", case=False, na=False)]
    df = df[df["age"].notna()]
    df = df.copy()
    df["age"] = df["age"].astype(float)
    df = _limit_samples_per_user(df, max_samples_per_user)
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
