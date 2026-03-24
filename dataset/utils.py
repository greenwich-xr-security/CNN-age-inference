"""Shared dataset utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split as _sklearn_train_test_split


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


_SOURCE_PRIORITY = {"handrgbd": 0, "hagrid": 1, "primary": 2, "archive": 3}


def _prepare_sample_selection_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Attach stable quality-priority columns used by sample-capping logic."""
    work = df.copy()
    if "source" in work.columns:
        work["_source_priority"] = work["source"].map(_SOURCE_PRIORITY).fillna(99).astype(int)
    else:
        work["_source_priority"] = 99

    if "wall_label" in work.columns:
        wall_numeric = pd.to_numeric(work["wall_label"], errors="coerce")
        work["_wall_penalty"] = wall_numeric.eq(3).fillna(False).astype(int)
    else:
        work["_wall_penalty"] = 0

    if "lights_label" in work.columns:
        lights_text = work["lights_label"].fillna("").astype(str).str.strip().str.lower()
        work["_lights_penalty"] = lights_text.eq("off").astype(int)
    else:
        work["_lights_penalty"] = 0

    work["_selection_order"] = range(len(work))
    return work


def _limit_samples_per_user(
    df: pd.DataFrame,
    max_samples_per_user: int | None,
) -> pd.DataFrame:
    """Cap samples per user, keeping higher-quality samples first."""
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

    work = _prepare_sample_selection_frame(df)
    work = work.sort_values(
        ["user_id", "_source_priority", "_wall_penalty", "_lights_penalty", "_selection_order"],
        kind="stable",
    )
    limited = work.groupby("user_id", sort=False).head(max_samples)
    dropped = len(df) - len(limited)
    if dropped > 0:
        over_count = int((work.groupby("user_id").size() > max_samples).sum())
        print(
            f"Per-user sample cap applied ({max_samples} samples/user): "
            f"dropped {dropped} samples from {over_count} users while "
            f"penalising wall 3 / lights off when available."
        )
    drop_cols = ["_source_priority", "_wall_penalty", "_lights_penalty", "_selection_order"]
    limited = limited.drop(columns=[col for col in drop_cols if col in limited.columns])
    return limited.reset_index(drop=True)


def _limit_samples_per_age_bin(
    df: pd.DataFrame,
    max_samples_per_age: int | None,
) -> pd.DataFrame:
    """Cap total samples per integer age year, maximising user coverage first.

    Selection strategy inside each age bin:
    - keep one best-quality sample per user before taking a second from any user
    - prefer higher-priority sources
    - penalise wall 3 and lights off when those labels are present
    - preserve original row order as the final tiebreaker
    """
    if max_samples_per_age is None:
        return df
    try:
        max_s = int(max_samples_per_age)
    except (TypeError, ValueError):
        return df
    if max_s <= 0:
        return df
    if "age" not in df.columns:
        return df

    work = _prepare_sample_selection_frame(df)
    work["_age_bin"] = work["age"].round().astype(int)

    if "user_id" in work.columns:
        work = work.sort_values(
            ["_age_bin", "user_id", "_source_priority", "_wall_penalty", "_lights_penalty", "_selection_order"],
            kind="stable",
        )
        work["_user_round"] = work.groupby(["_age_bin", "user_id"], sort=False).cumcount()
        work = work.sort_values(
            ["_age_bin", "_user_round", "_source_priority", "_wall_penalty", "_lights_penalty", "_selection_order"],
            kind="stable",
        )
    else:
        work = work.sort_values(
            ["_age_bin", "_source_priority", "_wall_penalty", "_lights_penalty", "_selection_order"],
            kind="stable",
        )

    limited = work.groupby("_age_bin", sort=False).head(max_s)
    dropped = len(work) - len(limited)
    if dropped > 0:
        print(
            f"Per-age-bin sample cap applied ({max_s} samples/year): "
            f"dropped {dropped} samples while preserving user diversity and "
            f"penalising wall 3 / lights off when available."
        )
    drop_cols = [
        "_age_bin",
        "_source_priority",
        "_wall_penalty",
        "_lights_penalty",
        "_selection_order",
        "_user_round",
    ]
    limited = limited.drop(columns=[col for col in drop_cols if col in limited.columns])
    return limited.reset_index(drop=True)


def filter_metadata(
    df: pd.DataFrame,
    *,
    max_samples_per_user: int | None = None,
    max_samples_per_age_bin: int | None = 200,
) -> pd.DataFrame:
    """Keep dorsal images with known ages, cast ages to float, and apply diversity-aware sample caps."""
    df = df[df["aspect"].str.contains("dorsal", case=False, na=False)]
    df = df[df["age"].notna()]
    df = df.copy()
    df["age"] = df["age"].astype(float)
    df = _limit_samples_per_user(df, max_samples_per_user)
    df = _limit_samples_per_age_bin(df, max_samples_per_age_bin)
    return df.reset_index(drop=True)


def _format_age_bin_label(lower: int, upper: int) -> str:
    if lower == upper:
        return str(int(lower))
    return f"{int(lower)}-{int(upper)}"


def build_user_age_strata(
    user_ages: pd.Series,
    *,
    fine_bin_width: int = 2,
    coarse_start_age: int = 51,
    coarse_bin_width: int = 5,
    min_count: int = 1,
) -> pd.Series:
    """
    Build ordered user-level age-bin labels for split stratification.

    The default strategy uses fine 2-year bins through age 50 and wider bins
    above that, then merges adjacent sparse bins until every bin reaches
    ``min_count`` users when possible.
    """
    if fine_bin_width < 1:
        raise ValueError("fine_bin_width must be >= 1.")
    if coarse_bin_width < 1:
        raise ValueError("coarse_bin_width must be >= 1.")
    if min_count < 1:
        raise ValueError("min_count must be >= 1.")

    ages = pd.Series(user_ages).dropna().astype(float).round().astype(int)
    ages.index = ages.index.astype(str)
    if ages.empty:
        return pd.Series(dtype="category", name="age_stratum")

    min_age = int(ages.min())
    max_age = int(ages.max())
    ranges: list[list[int]] = []

    start = min_age
    fine_end_age = min(max_age, int(coarse_start_age) - 1)
    while start <= fine_end_age:
        end = min(start + fine_bin_width - 1, fine_end_age)
        ranges.append([start, end])
        start = end + 1

    while start <= max_age:
        end = min(start + coarse_bin_width - 1, max_age)
        ranges.append([start, end])
        start = end + 1

    def range_count(bounds: list[int]) -> int:
        lower, upper = bounds
        return int(((ages >= lower) & (ages <= upper)).sum())

    while len(ranges) > 1:
        counts = [range_count(bounds) for bounds in ranges]
        small_positions = [idx for idx, count in enumerate(counts) if count < min_count]
        if not small_positions:
            break

        # Prefer widening the older tail first; it is the sparsest region.
        idx = small_positions[-1]
        if idx == 0:
            merge_into = 1
        elif idx == len(ranges) - 1:
            merge_into = idx - 1
        else:
            left_count = counts[idx - 1]
            right_count = counts[idx + 1]
            merge_into = idx - 1 if left_count <= right_count else idx + 1

        if merge_into < idx:
            ranges[merge_into][1] = ranges[idx][1]
            del ranges[idx]
        else:
            ranges[merge_into][0] = ranges[idx][0]
            del ranges[idx]

    labels = pd.Series(index=ages.index, dtype="object", name="age_stratum")
    ordered_labels: list[str] = []
    for lower, upper in ranges:
        label = _format_age_bin_label(lower, upper)
        ordered_labels.append(label)
        labels.loc[(ages >= lower) & (ages <= upper)] = label

    return labels.astype(pd.CategoricalDtype(categories=ordered_labels, ordered=True))


def summarise_user_stratification(
    user_labels: pd.Series,
    split_to_user_ids: dict[str, Iterable[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarise per-split user label distributions and simple drift metrics."""
    labels = pd.Series(user_labels).dropna().copy()
    labels.index = labels.index.astype(str)
    if labels.empty:
        empty_stats = pd.DataFrame(
            columns=[
                "split",
                "stratum",
                "users",
                "fraction",
                "overall_users",
                "overall_fraction",
                "abs_frac_diff",
            ]
        )
        empty_quality = pd.DataFrame(
            columns=["split", "total_users", "max_abs_frac_diff", "mean_abs_frac_diff"]
        )
        return empty_stats, empty_quality

    if isinstance(labels.dtype, pd.CategoricalDtype):
        strata = list(labels.cat.categories)
    else:
        strata = sorted(str(label) for label in labels.unique())

    overall_counts = labels.value_counts().reindex(strata, fill_value=0)
    overall_total = int(labels.size)
    stats_rows: list[dict] = []
    quality_rows: list[dict] = []

    all_splits = {"overall": labels.index.tolist()}
    all_splits.update(split_to_user_ids)

    for split_name, user_ids in all_splits.items():
        split_ids = [str(uid) for uid in user_ids]
        split_labels = labels.reindex(split_ids).dropna()
        split_counts = split_labels.value_counts().reindex(strata, fill_value=0)
        split_total = int(split_labels.size)
        abs_diffs: list[float] = []

        for stratum in strata:
            users = int(split_counts[stratum])
            fraction = users / split_total if split_total else 0.0
            overall_users = int(overall_counts[stratum])
            overall_fraction = overall_users / overall_total if overall_total else 0.0
            abs_frac_diff = abs(fraction - overall_fraction)
            abs_diffs.append(abs_frac_diff)
            stats_rows.append(
                {
                    "split": split_name,
                    "stratum": stratum,
                    "users": users,
                    "fraction": fraction,
                    "overall_users": overall_users,
                    "overall_fraction": overall_fraction,
                    "abs_frac_diff": abs_frac_diff,
                }
            )

        quality_rows.append(
            {
                "split": split_name,
                "total_users": split_total,
                "max_abs_frac_diff": max(abs_diffs) if abs_diffs else 0.0,
                "mean_abs_frac_diff": sum(abs_diffs) / len(abs_diffs) if abs_diffs else 0.0,
            }
        )

    stats_df = pd.DataFrame(stats_rows)
    quality_df = pd.DataFrame(quality_rows)
    return stats_df, quality_df


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


def build_held_out_test_split(
    df: pd.DataFrame,
    *,
    test_size: float = 0.15,
    random_state: int = 42,
    stratify_adult: bool = True,
    stratify_labels: pd.Series | dict[str, str] | Iterable[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Split users into (train_dev_ids, test_ids) with optional user-level stratification.

    Returns (train_dev_ids, test_ids) — both as lists of user_id strings.
    The test set is held out and must never be used during training or tuning.
    """
    user_ids = df["user_id"].astype(str).unique()

    stratify = None
    if stratify_labels is not None:
        if isinstance(stratify_labels, pd.Series):
            aligned = stratify_labels.copy()
            aligned.index = aligned.index.astype(str)
            stratify = aligned.reindex(user_ids)
            if stratify.isna().any():
                missing = user_ids[stratify.isna().to_numpy()]
                raise ValueError(f"Missing stratify labels for user_ids: {missing[:5].tolist()}")
            stratify = stratify.astype(str).tolist()
        elif isinstance(stratify_labels, dict):
            stratify = [str(stratify_labels[str(uid)]) for uid in user_ids]
        else:
            stratify = [str(label) for label in stratify_labels]
            if len(stratify) != len(user_ids):
                raise ValueError("stratify_labels length must match the number of unique users.")
    elif stratify_adult:
        user_age = df.groupby("user_id")["age"].mean()
        user_class = {str(uid): ("adult" if age >= 18.0 else "minor") for uid, age in user_age.items()}
        stratify = [user_class.get(uid, "adult") for uid in user_ids]

    train_dev_ids, test_ids = _sklearn_train_test_split(
        user_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    return list(train_dev_ids), list(test_ids)


def save_test_split(
    test_ids: list[str],
    path: str | Path,
    *,
    seed: int,
    test_size: float,
    stratified: bool,
) -> Path:
    """Persist the held-out test user IDs to a JSON file."""
    path = Path(path)
    payload = {
        "test_size": float(test_size),
        "seed": int(seed),
        "stratified": bool(stratified),
        "test_user_ids": [str(uid) for uid in test_ids],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
    return path


def load_test_split(path: str | Path) -> dict:
    """Load a held-out test split JSON produced by save_test_split()."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    if "test_user_ids" not in payload:
        raise ValueError("Test split file is missing 'test_user_ids'.")
    return payload
