import argparse
import random
from pathlib import Path
from typing import Callable, Iterable, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.age import AgeDataset
from dataset.hand_metadata import (
    get_dataset_root,
    load_combined_metadata,
    set_dataset_root,
)
from dataset.transforms import build_transforms
from dataset.utils import filter_metadata, stratified_user_split
from displayUtils import DisplayUtils
from metrics import (
    CHALLENGE_BINS,
    CHALLENGE_PROB_TAU,
    LossWeights,
    compute_age_gate_curves,
    compute_challenge_fpr_table,
    weighted_regression_loss,
)
from models import resolve_model_builder

# Example (Windows): python train_age.py --data-root "C:\Users\Staff\OneDrive - University of Greenwich\HandsDatasets" --output-dir runs\b4_efficientnet --model b4 --img-size 380 --batch-size 32 --epochs 40 --seed 42 --lr 0.0003
# --- Config -----------------------------------------------------------------
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 40
DEFAULT_LR = 3e-4
DEFAULT_MODEL_VARIANT = "b7"
DEFAULT_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_PATIENCE = 20


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train EfficientNet hand age regressor."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Path to the dataset root directory. Overrides the default or env var.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory where checkpoints and plots will be saved.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_VARIANT,
        help=(
            "Backbone to use. EfficientNet: b0-b7. ConvNeXt: convnext_{tiny,small,base,large,xlarge} "
            "or aliases cnt,cns,cnb,cnl,cnx."
        ),
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=None,
        help="Override the input resolution (discouraged). By default the canonical size for the chosen model is used.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Mini-batch size for training (default: 32).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs (default: 40).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed for RNGs, ensuring reproducibility (default: 42).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help="Learning rate for AdamW optimizer (default: 3e-4).",
    )
    parser.add_argument(
        "--stratification",
        type=str,
        default="minorAdults",
        choices=["no", "minorAdults", "bins"],
        help=(
            "Dataset split mode: "
            "no=unstratified per-user random split; "
            "minorAdults=preserve adult/minor ratio (default); "
            "bins=preserve multi-bin age ratios."
        ),
    )
    parser.add_argument(
        "--no-stratified-user-split",
        action="store_true",
        help="Disable per-user stratification when splitting the dataset.",
    )
    parser.add_argument(
        "--loss-weight-nll",
        type=float,
        default=0.5,
        help="Weight for the Gaussian NLL component (default: 0.5; set to 1.0 for legacy behaviour).",
    )
    parser.add_argument(
        "--loss-weight-mse",
        type=float,
        default=0.25,
        help="Weight for the MSE component (default: 0.25).",
    )
    parser.add_argument(
        "--loss-weight-mae",
        type=float,
        default=0.25,
        help="Weight for the MAE component (default: 0.25).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
        help=f"Early stopping patience in epochs (default: {DEFAULT_PATIENCE}).",
    )
    args = parser.parse_args()
    loss_weights = LossWeights(
        nll=args.loss_weight_nll,
        mse=args.loss_weight_mse,
        mae=args.loss_weight_mae,
    )
    loss_weights.validate()
    set_random_seed(args.seed)
    model_builder, default_size, model_desc, model_key = resolve_model_builder(
        args.model
    )
    if args.img_size is not None and args.img_size != default_size:
        print(
            f"[train] Ignoring requested --img-size {args.img_size}; "
            f"{model_desc} uses {default_size}."
        )
    img_size = default_size
    if args.data_root:
        set_dataset_root(args.data_root)
    active_root = get_dataset_root()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_transform, test_transform = build_transforms(img_size)
    metadata = filter_metadata(load_combined_metadata(root=active_root))
    stratification_mode = "no" if args.no_stratified_user_split else args.stratification
    if stratification_mode == "no":
        user_ids = metadata["user_id"].unique()
        train_ids, test_ids = train_test_split(
            user_ids, test_size=0.2, random_state=args.seed
        )
    else:
        train_ids, test_ids = stratified_user_split(
            metadata,
            test_size=0.2,
            random_state=args.seed,
            stratification=stratification_mode,
        )
    train_meta = metadata[metadata["user_id"].isin(train_ids)]
    test_meta = metadata[metadata["user_id"].isin(test_ids)]
    print(
        f"Using dataset root: {active_root}\n"
        f"Saving artifacts to: {output_dir}\n"
        f"Train users: {train_meta['user_id'].nunique()} | Train images: {len(train_meta)}\n"
        f"Test users:  {test_meta['user_id'].nunique()} | Test images:  {len(test_meta)}\n"
        f"Model: {model_desc} | Image size: {img_size} | Batch size: {args.batch_size}\n"
        f"Epochs: {args.epochs} | Learning rate: {args.lr:.2e} | Seed: {args.seed}"
    )
    print(
        f"Loss weights -> NLL: {loss_weights.nll:.3f}, "
        f"MSE: {loss_weights.mse:.3f}, MAE: {loss_weights.mae:.3f}"
    )
    split_desc = {
        "no": "Unstratified per-user split (random).",
        "minorAdults": "Stratified per-user split (adult/minor aware).",
        "bins": "Stratified per-user split (multi-bin age labels).",
    }.get(stratification_mode, f"Split mode: {stratification_mode}")
    print(f"Split mode: {split_desc}")
    train_ds = AgeDataset(train_meta, transform=train_transform)
    test_ds = AgeDataset(test_meta, transform=test_transform)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    model = model_builder()
    if DEVICE.type == "cuda":
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            print(f"Using {gpu_count} GPUs via DataParallel.")
            model = nn.DataParallel(model)
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_val_loss = float("inf")
    best_model_path = output_dir / f"{model_key}_age_regressor.pth"
    history_log_path = output_dir / "history.log"
    min_delta = 0.001
    patience = max(1, int(args.patience))
    epochs_without_improvement = 0
    history_entries: list[dict] = []
    best_scatter: Optional[dict] = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_mae = 0.0
        running_mse = 0.0
        running_std = 0.0
        for images, ages in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            images, ages = images.to(DEVICE), ages.to(DEVICE)
            optimizer.zero_grad()
            pred_mean, pred_log_var = model(images)
            loss = weighted_regression_loss(pred_mean, pred_log_var, ages, loss_weights)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            mae = torch.mean(torch.abs(pred_mean - ages)).item()
            mse = torch.mean((pred_mean - ages) ** 2).item()
            avg_std = torch.mean(
                torch.exp(0.5 * torch.clamp(pred_log_var.detach(), min=-10.0, max=10.0))
            ).item()
            running_mae += mae
            running_mse += mse
            running_std += avg_std
        denom = max(1, len(train_loader))
        train_loss = running_loss / denom
        train_mae = running_mae / denom
        train_mse = running_mse / denom
        train_std = running_std / denom
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_mse = 0.0
        val_std = 0.0
        val_targets = []
        val_predictions = []
        val_log_vars = []
        with torch.no_grad():
            for images, ages in test_loader:
                images, ages = images.to(DEVICE), ages.to(DEVICE)
                pred_mean, pred_log_var = model(images)
                batch_loss = weighted_regression_loss(
                    pred_mean, pred_log_var, ages, loss_weights
                ).item()
                val_loss += batch_loss
                val_mae += torch.mean(torch.abs(pred_mean - ages)).item()
                val_mse += torch.mean((pred_mean - ages) ** 2).item()
                val_std += torch.mean(
                    torch.exp(
                        0.5 * torch.clamp(pred_log_var.detach(), min=-10.0, max=10.0)
                    )
                ).item()
                val_targets.extend(ages.detach().cpu().tolist())
                val_predictions.extend(pred_mean.detach().cpu().tolist())
                val_log_vars.extend(pred_log_var.detach().cpu().tolist())
        denom = max(1, len(test_loader))
        val_loss /= denom
        val_mae /= denom
        val_mse /= denom
        val_std /= denom
        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, train_mae={train_mae:.4f}, train_mse={train_mse:.4f}, train_std={train_std:.4f} | "
            f"val_loss={val_loss:.4f}, val_mae={val_mae:.4f}, val_mse={val_mse:.4f}, val_std={val_std:.4f}"
        )
        with history_log_path.open("a", encoding="utf-8") as log_fp:
            log_fp.write(
                f"Epoch {epoch},train_loss={train_loss:.6f},train_mae={train_mae:.6f},train_mse={train_mse:.6f},"
                f"train_std={train_std:.6f},val_loss={val_loss:.6f},val_mae={val_mae:.6f},val_mse={val_mse:.6f},val_std={val_std:.6f}\n"
            )
        history_entries.append(
            {
                "epoch": epoch,
                "train_mae": train_mae,
                "val_mae": val_mae,
                "train_rmse": float(np.sqrt(train_mse)),
                "val_rmse": float(np.sqrt(val_mse)),
            }
        )
        # Save the model and evaluation artifacts only if validation loss improves
        if val_loss < best_val_loss:
            improvement = (
                float("inf")
                if best_val_loss == float("inf")
                else best_val_loss - val_loss
            )
            best_val_loss = val_loss
            model_to_save = (
                model.module if isinstance(model, nn.DataParallel) else model
            )
            torch.save(model_to_save.state_dict(), best_model_path)
            best_scatter = {
                "targets": list(val_targets),
                "predictions": list(val_predictions),
                "epoch": epoch,
            }
            print(f"Saved best model to {best_model_path} (val_loss={val_loss:.4f})")
            val_targets_arr = np.asarray(val_targets, dtype=float)
            val_means_arr = np.asarray(val_predictions, dtype=float)
            val_log_vars_arr = np.asarray(val_log_vars, dtype=float)
            gate_results = compute_age_gate_curves(
                val_targets_arr,
                val_means_arr,
                val_log_vars_arr,
                age_threshold=18.0,
                num_thresholds=201,
            )
            preds_dump_path = output_dir / "best_val_predictions.npz"
            np.savez(
                preds_dump_path,
                targets=val_targets_arr,
                pred_mean=val_means_arr,
                pred_log_var=val_log_vars_arr,
                adult_prob=gate_results["adult_prob"],
                epoch=epoch,
            )
            roc_case1_path = output_dir / f"roc_case1_adult_gate_epoch{epoch}.png"
            roc_case2_path = output_dir / f"roc_case2_child_gate_epoch{epoch}.png"
            DisplayUtils.plot_roc_curve(
                gate_results["case1"]["fpr"],
                gate_results["case1"]["tpr"],
                thresholds=gate_results["case1"]["thresholds"],
                save_path=roc_case1_path,
                title="ROC - Adult Content Gate (admit adults)",
                auc_value=gate_results["case1"]["auc"],
                show=False,
            )
            DisplayUtils.plot_roc_curve(
                gate_results["case2"]["fpr"],
                gate_results["case2"]["tpr"],
                thresholds=gate_results["case2"]["thresholds"],
                save_path=roc_case2_path,
                title="ROC - Child Platform Gate (admit minors)",
                auc_value=gate_results["case2"]["auc"],
                show=False,
            )
            metrics_csv_path = output_dir / "age_gate_metrics.csv"
            with metrics_csv_path.open("w", encoding="utf-8") as metrics_fp:
                metrics_fp.write("case,tau,fpr,fnr,tpr,tnr\n")
                for case_name, case_data in (
                    ("adult_content_gate", gate_results["case1"]),
                    ("child_platform_gate", gate_results["case2"]),
                ):
                    for tau, fpr, fnr, tpr_val, tnr in zip(
                        case_data["thresholds"],
                        case_data["fpr"],
                        case_data["fnr"],
                        case_data["tpr"],
                        case_data["tnr"],
                    ):
                        metrics_fp.write(
                            f"{case_name},{tau:.4f},{fpr:.6f},{fnr:.6f},{tpr_val:.6f},{tnr:.6f}\n"
                        )
            print(
                f"Updated ROC plots ({roc_case1_path.name}, {roc_case2_path.name}), "
                f"metrics CSV ({metrics_csv_path.name}), and saved predictions to {preds_dump_path.name}."
            )
            if improvement == float("inf") or improvement >= min_delta:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            early_msg = (
                f"Early stopping at epoch {epoch}: validation loss did not improve by at least "
                f"{min_delta:.3f} for {patience} consecutive epochs."
            )
            print(early_msg)
            with history_log_path.open("a", encoding="utf-8") as log_fp:
                log_fp.write(f"# {early_msg}\n")
            history_plot_path = output_dir / "history_plot.png"
            DisplayUtils.plot_loss_history(
                history_entries,
                save_path=history_plot_path,
                show=False,
                title="Training History (MAE & RMSE)",
            )
            print(f"Saved training history plot to {history_plot_path}")
            break
    # Save per-user age distribution histograms for train/test splits
    train_user_ages = train_meta.groupby("user_id")["age"].mean().to_numpy()
    test_user_ages = test_meta.groupby("user_id")["age"].mean().to_numpy()
    hist_path = output_dir / "age_distribution_users.png"
    saved_hist = DisplayUtils.plot_age_histograms(
        train_user_ages,
        test_user_ages,
        save_path=hist_path,
        show=False,
        title="Per-user age distribution (train vs test)",
    )
    if saved_hist:
        print(f"Saved per-user age histograms to {saved_hist}")
    # Save a single scatter plot for the best epoch (based on validation loss)
    if best_scatter is not None:
        scatter_path = output_dir / "age_val_scatter_best.png"
        if DisplayUtils.save_regression_scatter(
            best_scatter["targets"],
            best_scatter["predictions"],
            save_path=scatter_path,
            title=f"Best Validation Age Predictions (epoch {best_scatter['epoch']})",
            axis_limits=(0.0, 70.0),
            point_size=20,
            alpha=0.6,
        ):
            print(f"Saved best validation scatter plot to {scatter_path}")
    # Final challenge-threshold table (single evaluation pass using best model)
    if best_model_path.exists():
        print("Computing challenge-threshold FPR table on test set using best model...")
        eval_model = model_builder()
        state = torch.load(best_model_path, map_location=DEVICE)
        eval_model.load_state_dict(state)
        eval_model = eval_model.to(DEVICE)
        eval_model.eval()
        all_targets: list[float] = []
        all_means: list[float] = []
        all_log_vars: list[float] = []
        with torch.no_grad():
            for images, ages in test_loader:
                images = images.to(DEVICE)
                ages = ages.to(DEVICE)
                mean, log_var = eval_model(images)
                all_targets.extend(ages.cpu().tolist())
                all_means.extend(mean.cpu().tolist())
                all_log_vars.extend(log_var.cpu().tolist())
        challenge_thresholds = np.arange(20, 30, 1, dtype=float)  # 10 rows: 20..29
        fpr_rows = compute_challenge_fpr_table(
            all_targets,
            all_means,
            all_log_vars,
            thresholds=challenge_thresholds,
            prob_threshold=CHALLENGE_PROB_TAU,
            bins=CHALLENGE_BINS,
        )
        challenge_csv = output_dir / "challenge_fpr_bins.csv"
        with challenge_csv.open("w", encoding="utf-8") as fp:
            header = (
                ["threshold"] + [label for label, _, _ in CHALLENGE_BINS] + ["total"]
            )
            fp.write(",".join(header) + "\n")
            for row in fpr_rows:
                values = (
                    [f"{row['threshold']:.1f}"]
                    + [f"{row[label]:.6f}" for label, _, _ in CHALLENGE_BINS]
                    + [f"{row['total']:.6f}"]
                )
                fp.write(",".join(values) + "\n")
        print(f"Saved challenge FPR table to {challenge_csv}")
    else:
        print("Best model checkpoint not found; skipped challenge-threshold table.")
    print("Training complete. Best model saved on validation improvement.")


if __name__ == "__main__":
    main()
