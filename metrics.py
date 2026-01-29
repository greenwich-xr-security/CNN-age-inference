from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

LOG_VAR_MIN = -10.0
LOG_VAR_MAX = 10.0
CHALLENGE_PROB_TAU = 0.5  # Probability threshold to auto-allow without document
CHALLENGE_BINS: Tuple[Tuple[str, float, float], ...] = (
    ("10-12", 10.0, 12.0),
    ("13-15", 13.0, 15.0),
    ("16-17", 16.0, 17.0),
)
CHALLENGE_FNR_BINS: Tuple[Tuple[str, float, float | None], ...] = (
    ("18-19", 18.0, 19.0),
    ("20-24", 20.0, 24.0),
    ("25-29", 25.0, 29.0),
    ("30-39", 30.0, 39.0),
    ("40-49", 40.0, 49.0),
    ("50+", 50.0, None),
)
# Aggregation behaviour: if True, discard per-user remainders smaller than group_size.
AGGREGATION_DISCARD_REMAINDER = True


@dataclass(frozen=True)
class LossWeights:
    """Container for regression loss weights."""

    nll: float = 1.0
    mse: float = 0.0
    mae: float = 0.0

    def validate(self) -> None:
        for name, value in (("nll", self.nll), ("mse", self.mse), ("mae", self.mae)):
            if value < 0:
                raise ValueError(f"{name} weight must be non-negative (got {value}).")
        if self.nll + self.mse + self.mae <= 0:
            raise ValueError("At least one loss weight must be greater than zero.")


def _reduce_mean(values: torch.Tensor, sample_weights: torch.Tensor | None) -> torch.Tensor:
    if sample_weights is None:
        return torch.mean(values)
    weights = sample_weights.to(values.device, dtype=values.dtype)
    values = values.reshape(-1)
    weights = weights.reshape(-1)
    weight_sum = torch.sum(weights)
    if weight_sum <= 0:
        return torch.mean(values)
    return torch.sum(values * weights) / weight_sum


def gaussian_nll_loss(
    pred_mean: torch.Tensor,
    pred_log_var: torch.Tensor,
    target: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Negative log-likelihood under a Gaussian with optional per-sample weights."""
    log_var = torch.clamp(pred_log_var, min=LOG_VAR_MIN, max=LOG_VAR_MAX)
    inv_var = torch.exp(-log_var)
    loss = 0.5 * (log_var + (target - pred_mean) ** 2 * inv_var)
    return _reduce_mean(loss, sample_weights)


def mse_loss(
    pred_mean: torch.Tensor,
    target: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean squared error with optional per-sample weights."""
    return _reduce_mean((pred_mean - target) ** 2, sample_weights)


def mae_loss(
    pred_mean: torch.Tensor,
    target: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean absolute error with optional per-sample weights."""
    return _reduce_mean(torch.abs(pred_mean - target), sample_weights)


def intra_user_spread_loss(
    pred_mean: torch.Tensor,
    user_ids: Sequence[object],
) -> torch.Tensor:
    """Penalize per-user prediction variance within a batch."""
    if pred_mean.ndim != 1:
        pred_mean = pred_mean.view(-1)
    if pred_mean.numel() != len(user_ids):
        raise ValueError("pred_mean and user_ids must have the same length.")

    groups: dict[str, list[int]] = {}
    for idx, uid in enumerate(user_ids):
        key = str(uid)
        groups.setdefault(key, []).append(idx)

    losses = []
    device = pred_mean.device
    for indices in groups.values():
        if len(indices) < 2:
            continue
        idx_tensor = torch.tensor(indices, device=device, dtype=torch.long)
        vals = pred_mean.index_select(0, idx_tensor)
        var = torch.var(vals, unbiased=False)
        losses.append(var)

    if not losses:
        return pred_mean.new_tensor(0.0)
    return torch.mean(torch.stack(losses))


def weighted_regression_loss(
    pred_mean: torch.Tensor,
    pred_log_var: torch.Tensor,
    target: torch.Tensor,
    weights: LossWeights,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return weighted combination of Gaussian NLL, MSE, and MAE components."""
    total_loss: torch.Tensor | None = None
    if weights.nll > 0:
        nll = gaussian_nll_loss(pred_mean, pred_log_var, target, sample_weights)
        total_loss = weights.nll * nll
    if weights.mse > 0:
        mse = mse_loss(pred_mean, target, sample_weights)
        total_loss = mse * weights.mse if total_loss is None else total_loss + weights.mse * mse
    if weights.mae > 0:
        mae = mae_loss(pred_mean, target, sample_weights)
        total_loss = mae * weights.mae if total_loss is None else total_loss + weights.mae * mae

    if total_loss is None:
        raise ValueError("weighted_regression_loss requires at least one positive weight.")
    return total_loss


def embedding_variance_loss(
    z: torch.Tensor,
    user_ids: Sequence[object],
    ages: torch.Tensor,
    *,
    age_slack: float = 1.0,
) -> torch.Tensor:
    """
    Encourage embedding consistency within a user, down-weighted by age gaps.
    weight = 1 / (1 + |age_i - age_j| / age_slack)
    """
    if z.ndim == 1:
        z = z.unsqueeze(1)
    age_slack = max(float(age_slack), 1e-6)
    device = z.device
    ages = ages.to(device, dtype=z.dtype).view(-1)
    grouped: dict[str, list[int]] = {}
    for idx, uid in enumerate(user_ids):
        grouped.setdefault(str(uid), []).append(idx)

    total = z.new_tensor(0.0)
    weight_sum = z.new_tensor(0.0)
    for indices in grouped.values():
        if len(indices) < 2:
            continue
        idx_tensor = torch.tensor(indices, device=device, dtype=torch.long)
        z_grp = z.index_select(0, idx_tensor)
        age_grp = ages.index_select(0, idx_tensor)
        # pairwise weights and distances
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                delta_age = torch.abs(age_grp[i] - age_grp[j])
                weight = 1.0 / (1.0 + delta_age / age_slack)
                dist_sq = torch.sum((z_grp[i] - z_grp[j]) ** 2)
                total = total + weight * dist_sq
                weight_sum = weight_sum + weight
    if weight_sum <= 0:
        return z.new_tensor(0.0)
    return total / weight_sum


def embedding_contrastive_loss(
    z: torch.Tensor,
    user_ids: Sequence[object],
    ages: torch.Tensor,
    *,
    margin: float = 1.0,
    age_thresh: float = 5.0,
) -> torch.Tensor:
    """
    Push embeddings from different users apart when their ages differ by more than age_thresh.
    Uses hinge: max(0, margin - ||z_i - z_j||).
    """
    if z.ndim == 1:
        z = z.unsqueeze(1)
    device = z.device
    ages = ages.to(device, dtype=z.dtype).view(-1)
    user_ids = [str(u) for u in user_ids]
    n = z.size(0)
    if n < 2:
        return z.new_tensor(0.0)
    margin = float(margin)
    age_thresh = float(age_thresh)
    total = z.new_tensor(0.0)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if user_ids[i] == user_ids[j]:
                continue
            if torch.abs(ages[i] - ages[j]) < age_thresh:
                continue
            dist = torch.norm(z[i] - z[j], p=2)
            loss_ij = F.relu(margin - dist)
            total = total + loss_ij
            count += 1
    if count == 0:
        return z.new_tensor(0.0)
    return total / count


def aggregate_predictions_by_user(
    user_ids: Sequence[object],
    targets: Sequence[float],
    pred_means: Sequence[float],
    pred_log_vars: Sequence[float],
    *,
    group_size: int,
    rng: random.Random | None = None,
) -> dict:
    """
    Aggregate per-sample predictions into random per-user groups of size *group_size*.

    - Samples are grouped per user_id, shuffled, then chunked into groups of size *group_size*.
    - Remainders smaller than group_size are dropped when AGGREGATION_DISCARD_REMAINDER is True.
    - Means and targets are averaged; variances are averaged then converted back to log-variance.
    """
    if group_size <= 0:
        raise ValueError(f"group_size must be positive (got {group_size}).")
    if rng is None:
        rng = random.Random()

    user_buckets: dict[str, list[tuple[float, float, float]]] = {}
    for uid, tgt, mean, log_var in zip(user_ids, targets, pred_means, pred_log_vars):
        key = str(uid)
        user_buckets.setdefault(key, []).append((float(tgt), float(mean), float(log_var)))

    agg_targets: list[float] = []
    agg_means: list[float] = []
    agg_log_vars: list[float] = []

    for samples in user_buckets.values():
        if group_size == 1:
            for tgt, mean, log_var in samples:
                clamped_lv = float(np.clip(log_var, LOG_VAR_MIN, LOG_VAR_MAX))
                agg_targets.append(tgt)
                agg_means.append(mean)
                agg_log_vars.append(clamped_lv)
            continue

        rng.shuffle(samples)
        full_groups, remainder = divmod(len(samples), group_size)
        cursor = 0

        def _append_group(chunk: Sequence[tuple[float, float, float]]):
            tgt_vals, mean_vals, lv_vals = zip(*chunk)
            var_vals = [float(np.exp(np.clip(lv, LOG_VAR_MIN, LOG_VAR_MAX))) for lv in lv_vals]
            mean_var = float(np.mean(var_vals))
            agg_targets.append(float(np.mean(tgt_vals)))
            agg_means.append(float(np.mean(mean_vals)))
            agg_log_vars.append(float(np.log(max(mean_var, 1e-12))))

        for _ in range(full_groups):
            chunk = samples[cursor : cursor + group_size]
            cursor += group_size
            _append_group(chunk)
        if remainder and not AGGREGATION_DISCARD_REMAINDER:
            chunk = samples[cursor:]
            _append_group(chunk)

    return {
        "targets": np.asarray(agg_targets, dtype=float),
        "pred_mean": np.asarray(agg_means, dtype=float),
        "pred_log_var": np.asarray(agg_log_vars, dtype=float),
    }


# ---------------------------------------------------------------------------
# Evaluation helpers


def _safe_rate(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def compute_adult_probabilities(
    pred_means,
    pred_log_vars,
    *,
    age_threshold: float = 18.0,
) -> np.ndarray:
    """Return P(age >= threshold) from predicted Gaussian parameters."""
    means = torch.as_tensor(pred_means, dtype=torch.float32, device="cpu")
    log_vars = torch.as_tensor(pred_log_vars, dtype=torch.float32, device="cpu")
    log_vars = torch.clamp(log_vars, min=LOG_VAR_MIN, max=LOG_VAR_MAX)
    std = torch.exp(0.5 * log_vars)
    std = torch.clamp(std, min=1e-3)
    z = (age_threshold - means) / std
    cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    adult_prob = torch.clamp(1.0 - cdf, min=0.0, max=1.0)
    return adult_prob.numpy()


def compute_age_gate_curves(
    targets,
    pred_means,
    pred_log_vars,
    *,
    age_threshold: float = 18.0,
    num_thresholds: int = 101,
) -> dict:
    """Compute ROC-style metrics for both policy cases using adult probabilities."""
    targets_arr = np.asarray(targets, dtype=float)
    adult_prob = compute_adult_probabilities(pred_means, pred_log_vars, age_threshold=age_threshold)
    tau_values = np.linspace(0.0, 1.0, num=num_thresholds)

    is_adult = targets_arr >= age_threshold
    is_minor = ~is_adult
    adult_total = int(is_adult.sum())
    minor_total = int(is_minor.sum())

    def build_case(admit_mask, positive_mask, negative_mask):
        tp = np.logical_and(admit_mask, positive_mask).sum()
        fp = np.logical_and(admit_mask, negative_mask).sum()
        fn = np.logical_and(~admit_mask, positive_mask).sum()
        tn = np.logical_and(~admit_mask, negative_mask).sum()
        pos_total = positive_mask.sum()
        neg_total = negative_mask.sum()
        tpr = _safe_rate(tp, pos_total)
        fpr = _safe_rate(fp, neg_total)
        fnr = _safe_rate(fn, pos_total)
        tnr = _safe_rate(tn, neg_total)
        return fpr, tpr, fnr, tnr

    case1_fprs = []
    case1_tprs = []
    case1_fnrs = []
    case1_tnrs = []
    case2_fprs = []
    case2_tprs = []
    case2_fnrs = []
    case2_tnrs = []

    for tau in tau_values:
        admit_adult = adult_prob >= tau  # Case 1
        fpr1, tpr1, fnr1, tnr1 = build_case(admit_adult, is_adult, is_minor)
        case1_fprs.append(fpr1)
        case1_tprs.append(tpr1)
        case1_fnrs.append(fnr1)
        case1_tnrs.append(tnr1)

        admit_minor = adult_prob < tau  # Case 2
        fpr2, tpr2, fnr2, tnr2 = build_case(admit_minor, is_minor, is_adult)
        case2_fprs.append(fpr2)
        case2_tprs.append(tpr2)
        case2_fnrs.append(fnr2)
        case2_tnrs.append(tnr2)

    def compute_auc(fprs, tprs):
        fprs_arr = np.asarray(fprs, dtype=float)
        tprs_arr = np.asarray(tprs, dtype=float)
        order = np.argsort(fprs_arr)
        if fprs_arr.size == 0:
            return 0.0
        return float(np.trapz(tprs_arr[order], fprs_arr[order]))

    results = {
        "adult_prob": adult_prob,
        "case1": {
            "fpr": np.asarray(case1_fprs, dtype=float),
            "tpr": np.asarray(case1_tprs, dtype=float),
            "fnr": np.asarray(case1_fnrs, dtype=float),
            "tnr": np.asarray(case1_tnrs, dtype=float),
            "thresholds": tau_values,
            "auc": compute_auc(case1_fprs, case1_tprs),
            "adult_total": adult_total,
            "minor_total": minor_total,
        },
        "case2": {
            "fpr": np.asarray(case2_fprs, dtype=float),
            "tpr": np.asarray(case2_tprs, dtype=float),
            "fnr": np.asarray(case2_fnrs, dtype=float),
            "tnr": np.asarray(case2_tnrs, dtype=float),
            "thresholds": tau_values,
            "auc": compute_auc(case2_fprs, case2_tprs),
            "adult_total": adult_total,
            "minor_total": minor_total,
        },
    }
    return results


def compute_challenge_fpr_table(
    targets,
    pred_means,
    pred_log_vars,
    *,
    thresholds: Iterable[float],
    prob_threshold: float = CHALLENGE_PROB_TAU,
    bins: Iterable[tuple[str, float, float]] = CHALLENGE_BINS,
) -> list[dict]:
    """
    Compute FPR per bin for minors using adult probabilities versus an age challenge threshold.

    For each threshold:
    - minors = targets < threshold
    - FPR per bin = (# of bin minors allowed) / (total minors)
    - total = (# of all minors allowed) / (total minors)
    If bins cover all minors, total matches the global FPR for that threshold.
    """
    targets_arr = np.asarray(targets, dtype=float)
    preds_arr = np.asarray(pred_means, dtype=float)
    log_vars_arr = np.asarray(pred_log_vars, dtype=float)
    rows: list[dict] = []
    for thr in thresholds:
        adult_prob = compute_adult_probabilities(preds_arr, log_vars_arr, age_threshold=thr)
        allow_mask = adult_prob >= prob_threshold
        minors_mask_global = targets_arr < thr
        minors_total = int(minors_mask_global.sum())

        row: dict[str, float] = {"threshold": float(thr)}
        bin_fprs = []
        total_fp = 0
        for label, lower, upper in bins:
            bin_mask = (targets_arr >= lower) & (targets_arr <= upper) & minors_mask_global
            fp = int(np.logical_and(allow_mask, bin_mask).sum())
            fpr = _safe_rate(fp, minors_total)
            row[label] = fpr
            bin_fprs.append(fpr)
            total_fp += fp

        row["total"] = _safe_rate(total_fp, minors_total) if bin_fprs else 0.0
        rows.append(row)
    return rows


def compute_challenge_fpr_table_weighted(
    targets,
    pred_means,
    pred_log_vars,
    *,
    thresholds: Iterable[float],
    prob_threshold: float = CHALLENGE_PROB_TAU,
    bins: Iterable[tuple[str, float, float]] = CHALLENGE_BINS,
) -> list[dict]:
    """
    Compute FPR per bin (relative to each bin's count) and an overall count-weighted FPR.

    total = sum(fp_bin) / sum(count_bin), where bin counts are restricted to minors (< threshold).
    """
    targets_arr = np.asarray(targets, dtype=float)
    preds_arr = np.asarray(pred_means, dtype=float)
    log_vars_arr = np.asarray(pred_log_vars, dtype=float)
    rows: list[dict] = []
    for thr in thresholds:
        adult_prob = compute_adult_probabilities(preds_arr, log_vars_arr, age_threshold=thr)
        allow_mask = adult_prob >= prob_threshold
        minors_mask_global = targets_arr < thr

        row: dict[str, float] = {"threshold": float(thr)}
        total_fp = 0
        total_count = 0
        for label, lower, upper in bins:
            bin_mask = (targets_arr >= lower) & (targets_arr <= upper) & minors_mask_global
            bin_total = int(bin_mask.sum())
            fp = int(np.logical_and(allow_mask, bin_mask).sum())
            fpr = _safe_rate(fp, bin_total)
            row[label] = fpr
            total_fp += fp
            total_count += bin_total

        row["total"] = _safe_rate(total_fp, total_count)
        rows.append(row)
    return rows


def compute_challenge_fnr_table_case1(
    targets,
    pred_means,
    pred_log_vars,
    *,
    thresholds: Iterable[float],
    prob_threshold: float = CHALLENGE_PROB_TAU,
    bins: Iterable[tuple[str, float, float | None]] = CHALLENGE_FNR_BINS,
) -> list[dict]:
    """
    Compute FNR per adult bin for challenge Case 1 (admit adults).

    FNR per bin = (# of adult samples in bin that were blocked) / (total samples overall).
    Returns one row per threshold with keys: threshold, <bin labels...>, total.
    """
    targets_arr = np.asarray(targets, dtype=float)
    preds_arr = np.asarray(pred_means, dtype=float)
    log_vars_arr = np.asarray(pred_log_vars, dtype=float)
    overall_total = int(targets_arr.size)
    rows: list[dict] = []
    for thr in thresholds:
        adult_prob = compute_adult_probabilities(preds_arr, log_vars_arr, age_threshold=thr)
        allow_mask = adult_prob >= prob_threshold
        fn_mask = ~allow_mask  # blocked

        row: dict[str, float] = {"threshold": float(thr)}
        total_fn = 0
        for label, lower, upper in bins:
            lower_val = lower
            upper_val = float("inf") if upper is None else upper
            bin_mask = (targets_arr >= lower_val) & (targets_arr <= upper_val)
            fn = int(np.logical_and(fn_mask, bin_mask).sum())
            fnr = _safe_rate(fn, overall_total)
            row[label] = fnr
            total_fn += fn

        row["total"] = _safe_rate(total_fn, overall_total) if overall_total > 0 else 0.0
        rows.append(row)
    return rows
