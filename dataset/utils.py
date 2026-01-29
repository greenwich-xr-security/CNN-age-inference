"""Shared dataset utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold


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


def compute_age_weight_map(
    ages: Iterable[float] | pd.Series,
    *,
    eps: float = 1.0,
    power: float = 1.0,
    min_w: float = 0.25,
    max_w: float = 5.0,
    normalise: bool = True,
) -> dict[int, float]:
    """
    Build an inverse-frequency weight map keyed by rounded integer age.

    weight(age) = ((count(age) + eps) ** -power), optionally normalised to mean 1 and clipped.
    """
    if eps < 0:
        raise ValueError("eps must be non-negative.")
    if power < 0:
        raise ValueError("power must be non-negative.")
    weights: dict[int, float] = {}
    age_series = pd.Series(list(ages), dtype=float).dropna()
    if age_series.empty:
        return weights

    age_int = age_series.round().astype(int)
    counts = age_int.value_counts()
    raw_weights = (counts + eps) ** (-power)
    if normalise and not raw_weights.empty:
        raw_weights = raw_weights / raw_weights.mean()

    clipped = raw_weights.clip(lower=min_w, upper=max_w)
    for age_value, weight in clipped.items():
        weights[int(age_value)] = float(weight)
    return weights


def oversample_by_age(
    df: pd.DataFrame,
    *,
    target_per_age: int | None = None,
    max_multiplier: float = 3.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Oversample under-represented integer ages by duplicating rows with replacement.

    - target_per_age: desired samples per age (defaults to the max age count).
    - max_multiplier: cap growth per age (e.g., 3.0 means an age can grow to 3x its original size).
    """
    if df.empty or "age" not in df.columns:
        return df

    max_multiplier = max(1.0, float(max_multiplier))
    work = df.reset_index(drop=True)
    age_series = work["age"].astype(float)
    age_int = age_series.round().astype(int)
    counts = age_int.value_counts()
    if counts.empty:
        return work

    target = target_per_age if target_per_age is not None else int(counts.max())
    target = max(int(target), 1)

    parts = [work]
    for age_value, count in counts.items():
        desired = min(target, int(round(count * max_multiplier)))
        if desired <= count:
            continue
        need = desired - count
        mask = age_int == age_value
        subset = work[mask]
        if subset.empty:
            continue
        sampled = subset.sample(n=need, replace=True, random_state=seed)
        parts.append(sampled)

    if len(parts) == 1:
        return work
    return (
        pd.concat(parts, ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )


def build_kfold_user_splits(
    df: pd.DataFrame,
    *,
    k: int,
    random_state: int = 42,
    stratify_on: str | None = None,
) -> list[list[str]]:
    """Return k disjoint validation folds of user_ids."""
    if k < 2:
        raise ValueError("k must be at least 2 for k-fold splits.")
    if "user_id" not in df.columns:
        raise ValueError("DataFrame must include 'user_id' for k-fold splits.")

    user_ids = df["user_id"].astype(str).unique()
    if user_ids.size < k:
        raise ValueError(f"Not enough users ({user_ids.size}) to build {k} folds.")

    if stratify_on:
        if stratify_on not in df.columns:
            raise ValueError(f"stratify_on column '{stratify_on}' not found in DataFrame.")
        labels = (
            df.drop_duplicates(subset="user_id")
            .set_index("user_id")[stratify_on]
            .reindex(user_ids)
        )
        splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
        splits = splitter.split(user_ids, labels)
    else:
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
