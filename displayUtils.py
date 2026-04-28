import cv2
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from typing import Iterable, Optional, Tuple


class DisplayUtils:
    # ------------------------------------------------------------------ #
    # connections (unchanged)
    _CONNECTIONS = (
        (0, 1),  (1, 2),  (2, 3),  (3, 4),        # thumb
        (0, 5),  (5, 6),  (6, 7),  (7, 8),        # index
        (0, 9),  (9, 10), (10, 11), (11, 12),     # middle
        (0, 13), (13, 14), (14, 15), (15, 16),    # ring
        (0, 17), (17, 18), (18, 19), (19, 20)     # little
    )

    # ------------------------------------------------------------------ #
    @staticmethod
    def display_with_coords(
        img,
        landmarks,
        *,
        handedness=None,
        palm_side=None,
        colour=(0, 255, 0),
        max_dim=1080,
    ) -> None:
        """Draw hand landmarks (normalised to the FULL image) on *img*."""
        landmarks = np.asarray(landmarks, dtype=np.float32)
        if landmarks.shape != (21, 3):
            raise ValueError("landmarks must have shape (21, 3)")

        work_img = img.copy()
        work_img, scale = DisplayUtils._resize_to_max(work_img, max_dim)
        h_img, w_img = work_img.shape[:2]

        # Points + indices
        for idx, (lx, ly, _lz) in enumerate(landmarks):
            x_px = int(np.clip(lx, 0.0, 1.0) * w_img)
            y_px = int(np.clip(ly, 0.0, 1.0) * h_img)
            cv2.circle(work_img, (x_px, y_px), 4, colour, -1)
            cv2.putText(
                work_img,
                f"{idx}:({x_px},{y_px})",
                (x_px + 5, y_px - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 0, 0),
                1,
            )

        # Skeleton
        for a, b in DisplayUtils._CONNECTIONS:
            xa = int(np.clip(landmarks[a, 0], 0.0, 1.0) * w_img)
            ya = int(np.clip(landmarks[a, 1], 0.0, 1.0) * h_img)
            xb = int(np.clip(landmarks[b, 0], 0.0, 1.0) * w_img)
            yb = int(np.clip(landmarks[b, 1], 0.0, 1.0) * h_img)
            cv2.line(work_img, (xa, ya), (xb, yb), colour, 2)

        # Orientation overlay
        orientation_text = " ".join(t for t in (handedness, palm_side) if t).strip()
        if orientation_text:
            cv2.putText(
                work_img,
                orientation_text,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        title = "Hand Landmarks (image-normalised)"
        if orientation_text:
            title += f" | {orientation_text}"
        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(work_img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title(title)
        plt.show()

    @staticmethod
    def display(
        img,
        *,
        max_dim=1080,
        title: Optional[str] = None,
    ) -> None:
        """Display a BGR image using Matplotlib after optional resizing."""
        work_img = img.copy()
        work_img, _ = DisplayUtils._resize_to_max(work_img, max_dim)

        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(work_img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        if title:
            plt.title(title)
        plt.show()

    @staticmethod
    def display_with_bbox(
        img,
        bbox: Tuple[int, int, int, int],
        *,
        max_dim: int = 1080,
        colour: Tuple[int, int, int] = (255, 255, 0),
        thickness: int = 2,
        title: Optional[str] = None,
    ) -> None:
        """Display an image with a single bounding box overlay."""
        if bbox is None:
            raise ValueError("bbox must not be None")
        if len(bbox) != 4:
            raise ValueError("bbox must be a tuple (xmin, ymin, xmax, ymax)")

        xmin, ymin, xmax, ymax = [int(v) for v in bbox]
        if xmax < xmin or ymax < ymin:
            raise ValueError(f"bbox has invalid coordinates: {bbox}")

        work_img = img.copy()
        work_img, scale = DisplayUtils._resize_to_max(work_img, max_dim)
        xmin_s = int(round(xmin * scale))
        ymin_s = int(round(ymin * scale))
        xmax_s = int(round(xmax * scale))
        ymax_s = int(round(ymax * scale))
        cv2.rectangle(work_img, (xmin_s, ymin_s), (xmax_s, ymax_s), colour, thickness)

        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(work_img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        if title:
            plt.title(title)
        plt.show()

    @staticmethod
    def display_regression_scatter(
        targets: Iterable[float],
        predictions: Iterable[float],
        *,
        title: Optional[str] = None,
        point_size: int = 20,
        alpha: float = 0.6,
    ) -> None:
        """Scatter plot of targets vs predictions with equal axes."""
        targets_arr = np.asarray(list(targets), dtype=float)
        preds_arr = np.asarray(list(predictions), dtype=float)
        if targets_arr.size == 0:
            print("display_regression_scatter: no data to plot.")
            return

        min_val = float(np.min([targets_arr.min(), preds_arr.min()]))
        max_val = float(np.max([targets_arr.max(), preds_arr.max()]))
        padding = max(1.0, 0.05 * (max_val - min_val))
        axis_min = min_val - padding
        axis_max = max_val + padding

        plt.figure(figsize=(6, 6))
        plt.scatter(targets_arr, preds_arr, s=point_size, alpha=alpha, edgecolors='none')
        plt.plot([axis_min, axis_max], [axis_min, axis_max], 'r--', linewidth=1)
        for thr in (18.0,):
            plt.axvline(thr, color="black", linestyle=":", linewidth=1)
        for thr in (20.0, 25.0, 30.0):
            plt.axhline(thr, color="black", linestyle=":", linewidth=1)
        plt.xlabel('True Age')
        plt.ylabel('Predicted Age')
        if title:
            plt.title(title)
        plt.xlim(axis_min, axis_max)
        plt.ylim(axis_min, axis_max)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)

    # ------------------------------------------------------------------ #
    @staticmethod
    def plot_age_histograms(
        train_user_ages: Iterable[float],
        eval_user_ages: Iterable[float],
        *,
        save_path=None,
        show: bool = False,
        title: Optional[str] = None,
        train_title: str = "Train users",
        eval_title: str = "Eval users",
        count_label: str = "Number of users",
        extra_ages: Optional[Iterable[float]] = None,
        extra_title: Optional[str] = None,
    ) -> Optional[Path]:
        """Plot side-by-side histograms of age distributions for train and eval splits."""
        train_arr = np.asarray(list(train_user_ages), dtype=float)
        eval_arr = np.asarray(list(eval_user_ages), dtype=float)
        extra_arr = (
            np.asarray(list(extra_ages), dtype=float)
            if extra_ages is not None
            else np.asarray([], dtype=float)
        )
        if train_arr.size == 0 and eval_arr.size == 0 and extra_arr.size == 0:
            print("plot_age_histograms: no data to plot.")
            return None

        arrays = [arr for arr in (train_arr, eval_arr, extra_arr) if arr.size]
        combined = np.concatenate(arrays)
        min_age = float(np.floor(combined.min()))
        max_age = float(np.ceil(combined.max()))
        bin_edges = np.arange(min_age - 0.5, max_age + 1.5, 1.0)

        train_counts, _ = np.histogram(train_arr, bins=bin_edges) if train_arr.size else (np.array([]), bin_edges)
        eval_counts, _ = np.histogram(eval_arr, bins=bin_edges) if eval_arr.size else (np.array([]), bin_edges)
        extra_counts, _ = np.histogram(extra_arr, bins=bin_edges) if extra_arr.size else (np.array([]), bin_edges)
        y_max = max(
            int(train_counts.max()) if train_counts.size else 0,
            int(eval_counts.max()) if eval_counts.size else 0,
            int(extra_counts.max()) if extra_counts.size else 0,
        )

        plot_specs = [
            (train_arr, train_title, "#1f77b4"),
            (eval_arr, eval_title, "#ff7f0e"),
        ]
        if extra_title is not None:
            plot_specs.append((extra_arr, extra_title, "#2ca02c"))

        fig, axes = plt.subplots(1, len(plot_specs), figsize=(6 * len(plot_specs), 5), sharex=True, sharey=True)
        if not isinstance(axes, np.ndarray):
            axes = np.asarray([axes])

        for ax, (values, subplot_title, colour) in zip(axes, plot_specs):
            ax.hist(values, bins=bin_edges, color=colour, edgecolor="white")
            ax.set_title(subplot_title)
        for ax in axes:
            ax.set_xlabel("Age")
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
        axes[0].set_ylabel(count_label)
        axes[0].set_xlim(bin_edges[0], bin_edges[-1])
        axes[0].set_ylim(0, max(1, y_max + 1))
        if title:
            fig.suptitle(title)
        fig.tight_layout()

        saved_path = None
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path)
            saved_path = save_path

        if show:
            plt.show()
        else:
            plt.close(fig)

        return saved_path

    # ------------------------------------------------------------------ #
    @staticmethod
    def plot_mae_per_bin(
        targets: Iterable[float],
        predictions: Iterable[float],
        bins: Iterable[tuple[str, float, float]],
        *,
        save_path=None,
        show: bool = False,
        title: Optional[str] = None,
    ) -> Optional[Path]:
        """Plot MAE per age bin for evaluation data."""
        targets_arr = np.asarray(list(targets), dtype=float)
        preds_arr = np.asarray(list(predictions), dtype=float)
        if targets_arr.size == 0:
            print("plot_mae_per_bin: no targets to plot.")
            return None

        abs_err = np.abs(preds_arr - targets_arr)
        bin_labels = []
        bin_mae = []
        bin_counts = []
        for label, lower, upper in bins:
            mask = (targets_arr >= lower) & (targets_arr <= upper)
            bin_labels.append(label)
            bin_counts.append(int(mask.sum()))
            bin_mae.append(float(abs_err[mask].mean()) if mask.any() else np.nan)

        fig, ax = plt.subplots(figsize=(8, 5))
        bar_positions = np.arange(len(bin_labels))
        plotted_mae = np.nan_to_num(bin_mae, nan=0.0)
        bars = ax.bar(bar_positions, plotted_mae, color="#9467bd", edgecolor="white")
        ax.set_xticks(bar_positions)
        ax.set_xticklabels(bin_labels, rotation=0)
        ax.set_ylabel("MAE (years)")
        if title:
            ax.set_title(title)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.3)

        y_top = max(1.0, np.nanmax(plotted_mae) * 1.05)
        ax.set_ylim(0, y_top)
        for bar, mae, count in zip(bars, bin_mae, bin_counts):
            height = bar.get_height()
            label_text = f"{mae:.2f} ({count})" if not np.isnan(mae) else f"n/a ({count})"
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.02 * y_top,
                label_text,
                ha="center",
                va="bottom",
                fontsize=8,
            )

        fig.tight_layout()

        saved_path = None
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path)
            saved_path = save_path

        if show:
            plt.show()
        else:
            plt.close(fig)

        return saved_path

    # ------------------------------------------------------------------ #
    @staticmethod
    def make_square_bbox(bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Compute a square bounding box centered on the given bbox."""
        if bbox is None:
            raise ValueError("bbox must not be None")
        if len(bbox) != 4:
            raise ValueError("bbox must be a tuple (xmin, ymin, xmax, ymax)")

        x1, y1, x2, y2 = [int(v) for v in bbox]
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"bbox has invalid coordinates: {bbox}")

        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        side = max(bw, bh)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        sx1 = int(np.floor(cx - side / 2.0))
        sy1 = int(np.floor(cy - side / 2.0))
        sx2 = sx1 + side
        sy2 = sy1 + side
        return sx1, sy1, sx2, sy2

    # ------------------------------------------------------------------ #
    @staticmethod
    def save_regression_scatter(
        targets: Iterable[float],
        predictions: Iterable[float],
        *,
        save_path,
        title: Optional[str] = None,
        axis_limits: Optional[Tuple[float, float]] = None,
        point_size: int = 20,
        alpha: float = 0.6,
    ) -> bool:
        """Save a regression scatter plot without displaying it."""
        targets_arr = np.asarray(list(targets), dtype=float)
        preds_arr = np.asarray(list(predictions), dtype=float)
        if targets_arr.size == 0:
            print("save_regression_scatter: no data to plot.")
            return False

        if axis_limits is not None:
            axis_min, axis_max = axis_limits
        else:
            min_val = float(np.min([targets_arr.min(), preds_arr.min()]))
            max_val = float(np.max([targets_arr.max(), preds_arr.max()]))
            padding = max(1.0, 0.05 * (max_val - min_val))
            axis_min = min_val - padding
            axis_max = max_val + padding

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(targets_arr, preds_arr, s=point_size, alpha=alpha, edgecolors="none")
        ax.plot([axis_min, axis_max], [axis_min, axis_max], "r--", linewidth=1)
        # Minor/adult divide at 18
        ax.axvline(18.0, color="black", linestyle="--", linewidth=1)
        ax.axhline(18.0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("True Age")
        ax.set_ylabel("Predicted Age")
        if title:
            ax.set_title(title)
        ax.set_xlim(axis_min, axis_max)
        ax.set_ylim(axis_min, axis_max)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
        fig.tight_layout()

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)
        return True

    # ------------------------------------------------------------------ #
    @staticmethod
    def save_error_by_age(
        targets: Iterable[float],
        predictions: Iterable[float],
        *,
        save_path,
        title: Optional[str] = None,
    ) -> bool:
        """Save MAE and RMSE as a function of integer target age."""
        targets_arr = np.asarray(list(targets), dtype=float)
        preds_arr = np.asarray(list(predictions), dtype=float)
        if targets_arr.size == 0:
            print("save_error_by_age: no data to plot.")
            return False

        ages_int = targets_arr.astype(int)
        abs_err = np.abs(preds_arr - targets_arr)
        sq_err = (preds_arr - targets_arr) ** 2

        unique_ages = np.unique(ages_int)
        unique_ages.sort()
        mae_vals = []
        rmse_vals = []
        x_vals = []
        for age in unique_ages:
            mask = ages_int == age
            if not np.any(mask):
                continue
            x_vals.append(age)
            mae_vals.append(float(abs_err[mask].mean()))
            rmse_vals.append(float(np.sqrt(sq_err[mask].mean())))

        if not x_vals:
            print("save_error_by_age: no valid age buckets to plot.")
            return False

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x_vals, mae_vals, label="MAE", marker="o", linewidth=1.5)
        ax.plot(x_vals, rmse_vals, label="RMSE", marker="s", linewidth=1.5)
        ax.set_xlabel("Target Age")
        ax.set_ylabel("Error")
        if title:
            ax.set_title(title)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
        ax.legend()
        fig.tight_layout()

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)
        return True

    # ------------------------------------------------------------------ #
    @staticmethod
    def plot_loss_history(
        history: Iterable[dict],
        *,
        save_path=None,
        show: bool = True,
        title: Optional[str] = None,
    ) -> Optional[Path]:
        """Plot training/validation MAE and RMSE curves over epochs.

        Args:
            history: Iterable of dicts with keys: epoch, train_mae, val_mae, train_rmse, val_rmse.
            save_path: Optional path to save the plot. Creates parents as needed.
            show: Whether to display the plot (default True).
            title: Optional plot title.
        Returns:
            Path to the saved plot if saved, otherwise None.
        """
        entries = list(history)
        if not entries:
            print("plot_loss_history: no history entries provided.")
            return None

        epochs = [entry["epoch"] for entry in entries]
        train_mae = [entry["train_mae"] for entry in entries]
        val_mae = [entry["val_mae"] for entry in entries]
        train_rmse = [entry["train_rmse"] for entry in entries]
        val_rmse = [entry["val_rmse"] for entry in entries]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, train_mae, label="Train MAE", color="#1f77b4")
        ax.plot(epochs, val_mae, label="Val MAE", color="#ff7f0e")
        ax.plot(epochs, train_rmse, label="Train RMSE", color="#2ca02c")
        ax.plot(epochs, val_rmse, label="Val RMSE", color="#d62728")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Error")
        if title:
            ax.set_title(title)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
        ax.legend()
        fig.tight_layout()

        saved_path = None
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path)
            saved_path = save_path

        if show:
            plt.show()
        else:
            plt.close(fig)

        return saved_path

    # ------------------------------------------------------------------ #
    @staticmethod
    def plot_roc_curve(
        fprs,
        tprs,
        *,
        thresholds,
        save_path,
        title: Optional[str] = None,
        auc_value: Optional[float] = None,
        highlight_tau: Optional[float] = 0.5,
        show: bool = False,
    ) -> Optional[Path]:
        """Plot an ROC-style curve with thresholds encoded by colour."""
        fprs_arr = np.asarray(list(fprs), dtype=float)
        tprs_arr = np.asarray(list(tprs), dtype=float)
        thresholds_arr = np.asarray(list(thresholds), dtype=float)
        if fprs_arr.size == 0 or tprs_arr.size == 0:
            print("plot_roc_curve: no data to plot.")
            return None

        order = np.argsort(fprs_arr)
        ordered_fprs = fprs_arr[order]
        ordered_tprs = tprs_arr[order]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance")
        label = "ROC"
        if auc_value is not None:
            label += f" (AUC={auc_value:.3f})"
        ax.plot(ordered_fprs, ordered_tprs, color="#1f77b4", label=label)
        scatter = ax.scatter(
            fprs_arr,
            tprs_arr,
            c=thresholds_arr,
            cmap="viridis",
            s=35,
            edgecolors="none",
        )
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Threshold tau", rotation=270, labelpad=15)

        if highlight_tau is not None and thresholds_arr.size > 0:
            idx = int(np.argmin(np.abs(thresholds_arr - float(highlight_tau))))
            tau_fpr = fprs_arr[idx]
            tau_tpr = tprs_arr[idx]
            ax.axvline(tau_fpr, color="black", linestyle=":", linewidth=1)
            ax.axhline(tau_tpr, color="black", linestyle=":", linewidth=1)
            ax.scatter(
                [tau_fpr],
                [tau_tpr],
                color="red",
                s=45,
                edgecolors="white",
                linewidths=0.8,
                zorder=5,
                label=f"tau={thresholds_arr[idx]:.3f}",
            )

        ax.set_xlabel("FPR (undesired risk)")
        ax.set_ylabel("TPR (desired usability)")
        if title:
            ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
        ax.legend(loc="lower right")
        fig.tight_layout()

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        if show:
            plt.show()
        else:
            plt.close(fig)
        return save_path

    # ------------------------------------------------------------------ #
    @staticmethod
    def save_normals_reconstruction_grid(
        rgbs: np.ndarray,
        gt_normals: np.ndarray,
        pred_normals: np.ndarray,
        error_maps: np.ndarray,
        cos_sims: np.ndarray,
        save_path,
        n_samples: int = 8,
        title: str = "Normal Map Reconstruction",
    ) -> None:
        """Grid plot comparing GT vs predicted normal maps.

        Args:
            rgbs:         [N, H, W, 3] float32 in [0, 1] (display-ready RGB).
            gt_normals:   [N, H, W, 3] float32 in [0, 1] (decoded from [-1,1]).
            pred_normals: [N, H, W, 3] float32 in [0, 1] (decoded from [-1,1]).
            error_maps:   [N, H, W] float32, per-pixel 1 − cos(θ).
            cos_sims:     [N] float32, per-sample mean cosine similarity.
        """
        n = min(n_samples, len(rgbs))
        rng = np.random.default_rng(seed=0)
        idx = np.sort(rng.choice(len(rgbs), size=n, replace=False))

        fig, axes = plt.subplots(n, 4, figsize=(16, n * 4))
        if n == 1:
            axes = axes[np.newaxis, :]

        for j, col_title in enumerate(["RGB", "GT Normal", "Pred Normal", "Error (1 − cosθ)"]):
            axes[0, j].set_title(col_title, fontsize=11, fontweight="bold")

        err_vmax = float(np.percentile(error_maps, 95)) if error_maps.size > 0 else 1.0
        im_err = None
        for row, i in enumerate(idx):
            axes[row, 0].imshow(np.clip(rgbs[i], 0.0, 1.0))
            axes[row, 1].imshow(np.clip(gt_normals[i], 0.0, 1.0))
            axes[row, 2].imshow(np.clip(pred_normals[i], 0.0, 1.0))
            im_err = axes[row, 3].imshow(error_maps[i], cmap="hot", vmin=0.0, vmax=err_vmax)
            axes[row, 0].set_ylabel(
                f"cos={cos_sims[i]:.3f}", fontsize=9, rotation=0, labelpad=50, va="center"
            )
            for j in range(4):
                axes[row, j].axis("off")

        mean_cos = float(np.mean(cos_sims))
        fig.suptitle(
            f"{title}  |  mean cos-sim = {mean_cos:.4f}  (N={len(rgbs)})", fontsize=13
        )
        if im_err is not None:
            plt.colorbar(im_err, ax=axes[:, 3], shrink=0.6, label="1 − cosθ")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(Path(save_path), dpi=120, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _resize_to_max(img, max_dim):
        h, w = img.shape[:2]
        scale = 1.0
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return img, scale
