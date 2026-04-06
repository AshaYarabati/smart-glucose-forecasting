# =============================================================================
# Project: Smart Glucose Forecasting (MSc Thesis)
# Author: Asha Deepthi Yarabati
# Description: Creates comparison plots for all available models (baseline and deep) 
#              against the ground truth for each user, mode, and horizon.
#              It ensures all models are plotted on the same y-axis scale for 
#              fair visual comparison. It also generates overview sheets that tile 
#              the individual user plots for each mode+horizon combination.
# Version: 1.0
# Last Updated: 06 Apr 2026
# Usage:
# - Run in project root directory "pwsh ./run_all.ps1"
# =============================================================================

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_csv_list(value: Optional[str]) -> Optional[List[str]]:
    if value is None or str(value).strip() == "":
        return None
    return [v.strip() for v in str(value).split(",") if v.strip()]


def load_baseline_predictions(
    baseline_dir: Path,
    mode: str,
    horizon: int,
    user_id: str,
) -> Optional[pd.DataFrame]:
    """
    Expected file:
    baseline_predictions_{mode}_h{horizon}_{user_id}.csv

    Expected columns:
    timestamp, y_true, y_pred_persistence, y_pred_ridge, y_pred_sarimax
    """
    path = baseline_dir / f"baseline_predictions_{mode}_h{horizon}_{user_id}.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path)
    expected = {"timestamp", "y_true"}
    if not expected.issubset(df.columns):
        raise ValueError(f"Baseline file missing required columns: {path}")

    rename_map = {}
    if "y_pred_persistence" in df.columns:
        rename_map["y_pred_persistence"] = "Persistence"
    if "y_pred_ridge" in df.columns:
        rename_map["y_pred_ridge"] = "Ridge"
    if "y_pred_sarimax" in df.columns:
        rename_map["y_pred_sarimax"] = "SARIMAX"

    df = df.rename(columns=rename_map)
    keep_cols = ["timestamp", "y_true"] + [c for c in ["Persistence", "Ridge", "SARIMAX"] if c in df.columns]
    df = df[keep_cols].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def load_deep_predictions(
    deep_dir: Path,
    model_name: str,
    mode: str,
    horizon: int,
    user_id: str,
) -> Optional[pd.DataFrame]:
    """
    Expected file:
    {model_name}_predictions_{mode}_h{horizon}_{user_id}.csv

    Expected columns:
    timestamp, y_true, y_pred
    """
    path = deep_dir / f"{model_name}_predictions_{mode}_h{horizon}_{user_id}.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path)
    expected = {"timestamp", "y_true", "y_pred"}
    if not expected.issubset(df.columns):
        raise ValueError(f"Deep prediction file missing required columns: {path}")

    label = model_name.upper()
    df = df[["timestamp", "y_true", "y_pred"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.rename(columns={"y_pred": label})
    return df


def merge_prediction_frames(
    baseline_df: Optional[pd.DataFrame],
    lstm_df: Optional[pd.DataFrame],
    tcn_df: Optional[pd.DataFrame],
) -> Optional[pd.DataFrame]:
    """Merge all model outputs on timestamp and y_true."""
    frames = [df for df in [baseline_df, lstm_df, tcn_df] if df is not None]
    if not frames:
        return None

    merged = frames[0].copy()
    for df in frames[1:]:
        merged = merged.merge(
            df,
            on="timestamp",
            how="outer",
            suffixes=("", "_dup"),
        )

        if "y_true_dup" in merged.columns:
            if "y_true" in merged.columns:
                merged["y_true"] = merged["y_true"].combine_first(merged["y_true_dup"])
            else:
                merged["y_true"] = merged["y_true_dup"]
            merged = merged.drop(columns=["y_true_dup"])

    merged = merged.sort_values("timestamp").reset_index(drop=True)
    return merged


def infer_available_users(
    baseline_dir: Path,
    deep_dir: Path,
    modes: List[str],
    horizons: List[int],
) -> List[str]:
    """Infer users by scanning all prediction filenames."""
    user_ids = set()
    patterns = [
        "baseline_predictions_*.csv",
        "lstm_predictions_*.csv",
        "tcn_predictions_*.csv",
    ]

    for pred_dir in [baseline_dir, deep_dir]:
        for pattern in patterns:
            for path in pred_dir.glob(pattern):
                parts = path.stem.split("_")
                if len(parts) >= 5:
                    user_ids.add(parts[-1])

    return sorted(user_ids)


def compute_shared_limits(
    baseline_dir: Path,
    deep_dir: Path,
    users: List[str],
    mode: str,
    horizon: int,
    padding_ratio: float = 0.05,
) -> Tuple[float, float]:
    """Compute shared y-limits across all users for one mode+horizon combination."""
    values = []

    for user_id in users:
        baseline_df = load_baseline_predictions(baseline_dir, mode, horizon, user_id)
        lstm_df = load_deep_predictions(deep_dir, "lstm", mode, horizon, user_id)
        tcn_df = load_deep_predictions(deep_dir, "tcn", mode, horizon, user_id)

        merged = merge_prediction_frames(baseline_df, lstm_df, tcn_df)
        if merged is None:
            continue

        numeric_cols = [c for c in merged.columns if c != "timestamp"]
        for col in numeric_cols:
            series = pd.to_numeric(merged[col], errors="coerce").dropna()
            if not series.empty:
                values.append(series.min())
                values.append(series.max())

    if not values:
        return 0.0, 1.0

    y_min = float(np.min(values))
    y_max = float(np.max(values))

    if np.isclose(y_min, y_max):
        y_min -= 1.0
        y_max += 1.0

    pad = (y_max - y_min) * padding_ratio
    return y_min - pad, y_max + pad


def plot_user_comparison(
    df: pd.DataFrame,
    user_id: str,
    mode: str,
    horizon: int,
    output_path: Path,
    y_limits: Tuple[float, float],
    max_points: Optional[int] = None,
    use_timestamp_x: bool = True,
) -> None:
    """Plot ground truth and all available model predictions on a common y-axis."""
    plot_df = df.copy()

    if max_points is not None and len(plot_df) > max_points:
        plot_df = plot_df.tail(max_points).copy()

    model_cols = [c for c in ["Persistence", "Ridge", "SARIMAX", "LSTM", "TCN"] if c in plot_df.columns]

    plt.figure(figsize=(14, 5.5))

    x = plot_df["timestamp"] if use_timestamp_x else np.arange(len(plot_df))

    plt.plot(x, plot_df["y_true"], label="Ground truth", linewidth=2.4)
    for col in model_cols:
        plt.plot(x, plot_df[col], label=col, linewidth=1.5)

    plt.ylim(*y_limits)
    plt.title(f"Prediction comparison | {user_id} | {mode} | +{horizon} min")
    plt.xlabel("Timestamp" if use_timestamp_x else "Test sample")
    plt.ylabel("Glucose (mg/dL)")
    plt.legend(ncol=3)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def create_overview_sheet(
    combo_out_dir: Path,
    overview_output_path: Path,
    mode: str,
    horizon: int,
    cols: int = 2,
) -> None:
    """Create one overview PNG by tiling the individual user PNGs for a mode+horizon."""
    image_paths = sorted(combo_out_dir.glob(f"comparison_{mode}_h{horizon}_*.png"))
    if not image_paths:
        print(f"[WARN] No individual plots found for overview: {combo_out_dir}")
        return

    n = len(image_paths)
    cols = max(1, cols)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 8.5, rows * 4.8))
    axes = np.array(axes).reshape(-1)

    for ax, image_path in zip(axes, image_paths):
        img = mpimg.imread(image_path)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(image_path.stem.split("_")[-1], fontsize=10)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(f"Overview | {mode} | +{horizon} min", fontsize=16, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(overview_output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote overview: {overview_output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create same-scale comparison plots for baseline and deep model predictions vs ground truth."
    )
    parser.add_argument(
        "--baseline-predictions-dir",
        required=True,
        help="Directory containing baseline prediction CSVs.",
    )
    parser.add_argument(
        "--deep-predictions-dir",
        required=True,
        help="Directory containing deep model prediction CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save comparison plots.",
    )
    parser.add_argument(
        "--modes",
        default="cgm_only,cgm_context",
        help="Comma-separated list of modes, e.g. cgm_only,cgm_context",
    )
    parser.add_argument(
        "--horizons",
        default="30,60",
        help="Comma-separated list of horizons in minutes, e.g. 30,60",
    )
    parser.add_argument(
        "--users",
        default=None,
        help="Optional comma-separated list of user IDs, e.g. U001,U002",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=300,
        help="Optional maximum number of most recent points to plot per user.",
    )
    parser.add_argument(
        "--use-sample-index",
        action="store_true",
        help="Use test sample index on the x-axis instead of timestamp.",
    )
    parser.add_argument(
        "--overview-cols",
        type=int,
        default=2,
        help="Number of columns to use in each overview sheet.",
    )
    parser.add_argument(
        "--skip-overviews",
        action="store_true",
        help="If set, do not create the four overview PNG sheets.",
    )
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_predictions_dir)
    deep_dir = Path(args.deep_predictions_dir)
    output_dir = ensure_dir(args.output_dir)

    modes = parse_csv_list(args.modes) or ["cgm_only", "cgm_context"]
    horizons = [int(h) for h in (parse_csv_list(args.horizons) or ["30", "60"])]
    users = parse_csv_list(args.users)

    if users is None:
        users = infer_available_users(baseline_dir, deep_dir, modes, horizons)

    if not users:
        raise ValueError("No users found in the prediction directories.")

    for mode in modes:
        for horizon in horizons:
            y_limits = compute_shared_limits(
                baseline_dir=baseline_dir,
                deep_dir=deep_dir,
                users=users,
                mode=mode,
                horizon=horizon,
            )

            combo_out_dir = ensure_dir(output_dir / mode / f"h{horizon}")

            for user_id in users:
                baseline_df = load_baseline_predictions(baseline_dir, mode, horizon, user_id)
                lstm_df = load_deep_predictions(deep_dir, "lstm", mode, horizon, user_id)
                tcn_df = load_deep_predictions(deep_dir, "tcn", mode, horizon, user_id)

                merged = merge_prediction_frames(baseline_df, lstm_df, tcn_df)
                if merged is None:
                    print(f"[WARN] No prediction files found for {user_id} | {mode} | h{horizon}")
                    continue

                output_path = combo_out_dir / f"comparison_{mode}_h{horizon}_{user_id}.png"

                plot_user_comparison(
                    df=merged,
                    user_id=user_id,
                    mode=mode,
                    horizon=horizon,
                    output_path=output_path,
                    y_limits=y_limits,
                    max_points=args.max_points,
                    use_timestamp_x=not args.use_sample_index,
                )

                print(f"Wrote plot: {output_path}")

            if not args.skip_overviews:
                overview_output_path = output_dir / f"overview_{mode}_h{horizon}.png"
                create_overview_sheet(
                    combo_out_dir=combo_out_dir,
                    overview_output_path=overview_output_path,
                    mode=mode,
                    horizon=horizon,
                    cols=args.overview_cols,
                )


if __name__ == "__main__":
    main()
