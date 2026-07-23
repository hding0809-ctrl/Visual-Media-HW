from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import ensure_dir


def _as_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    if arr.size and arr.max() <= 1.0:
        arr *= 255.0
    return np.clip(arr, 0, 255).round().astype(np.uint8)


def _error_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred = _as_uint8(pred).astype(np.float32)
    gt = _as_uint8(gt).astype(np.float32)
    return np.mean(np.abs(pred - gt), axis=2)


def _metric_title(metrics: Mapping[str, float] | None) -> str:
    if not metrics:
        return ""
    parts = []
    if "psnr" in metrics:
        parts.append(f"PSNR {metrics['psnr']:.2f}")
    if "ssim" in metrics:
        parts.append(f"SSIM {metrics['ssim']:.4f}")
    return " / ".join(parts)


def make_comparison_panel(
    input_image: np.ndarray,
    pred_image: np.ndarray,
    gt_image: np.ndarray,
    error_map: np.ndarray | None,
    metrics: Mapping[str, float] | None,
    save_path: str | Path,
) -> None:
    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    err = error_map if error_map is not None else _error_map(pred_image, gt_image)
    vmax = max(1.0, float(np.percentile(err, 99)))

    fig, axes = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)
    panels = [
        ("Blurry input", _as_uint8(input_image), None),
        ("NAFNet output", _as_uint8(pred_image), None),
        ("GT", _as_uint8(gt_image), None),
        ("Absolute error", err, "inferno"),
    ]
    for ax, (title, image, cmap) in zip(axes, panels):
        if cmap:
            ax.imshow(image, cmap=cmap, vmin=0, vmax=vmax)
        else:
            ax.imshow(image)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle(_metric_title(metrics), fontsize=11)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def make_before_after_panel(
    input_image: np.ndarray,
    baseline_pred: np.ndarray,
    tta_pred: np.ndarray,
    gt_image: np.ndarray,
    metrics_before: Mapping[str, float] | None,
    metrics_after: Mapping[str, float] | None,
    save_path: str | Path,
) -> None:
    save_path = Path(save_path)
    ensure_dir(save_path.parent)

    title_before = "Baseline"
    if metrics_before:
        title_before += f"\n{_metric_title(metrics_before)}"
    title_after = "D4-TTA"
    if metrics_after:
        title_after += f"\n{_metric_title(metrics_after)}"

    fig, axes = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)
    panels = [
        ("Blurry input", _as_uint8(input_image)),
        (title_before, _as_uint8(baseline_pred)),
        (title_after, _as_uint8(tta_pred)),
        ("GT", _as_uint8(gt_image)),
    ]
    for ax, (title, image) in zip(axes, panels):
        ax.imshow(image)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def make_group_bar_chart(summary_csv: str | Path, save_path: str | Path) -> None:
    df = pd.read_csv(summary_csv)
    if df.empty:
        return
    save_path = Path(save_path)
    ensure_dir(save_path.parent)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    x = np.arange(len(df))
    axes[0].bar(x, df["mean_psnr"], color="#4c78a8")
    axes[0].set_ylabel("Mean PSNR")
    axes[0].set_xticks(x, df["group"], rotation=30, ha="right")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x, df["mean_ssim"], color="#59a14f")
    axes[1].set_ylabel("Mean SSIM")
    axes[1].set_xticks(x, df["group"], rotation=30, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def make_scatter_plot(csv_path: str | Path, x_col: str, y_col: str, save_path: str | Path) -> None:
    df = pd.read_csv(csv_path)
    if df.empty or x_col not in df or y_col not in df:
        return
    save_path = Path(save_path)
    ensure_dir(save_path.parent)

    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    ax.scatter(df[x_col], df[y_col], s=14, alpha=0.65, color="#4c78a8")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col.upper() if y_col in {"psnr", "ssim"} else y_col)
    ax.grid(alpha=0.25)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def make_before_after_bar_chart(before_after_csv: str | Path, save_path: str | Path) -> None:
    df = pd.read_csv(before_after_csv)
    if df.empty:
        return
    save_path = Path(save_path)
    ensure_dir(save_path.parent)

    x = np.arange(len(df))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    axes[0].bar(x - width / 2, df["baseline_psnr_mean"], width, label="Baseline", color="#4c78a8")
    axes[0].bar(x + width / 2, df["tta_psnr_mean"], width, label="D4-TTA", color="#f58518")
    axes[0].set_ylabel("Mean PSNR")
    axes[0].set_xticks(x, df["group"], rotation=30, ha="right")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x - width / 2, df["baseline_ssim_mean"], width, label="Baseline", color="#4c78a8")
    axes[1].bar(x + width / 2, df["tta_ssim_mean"], width, label="D4-TTA", color="#f58518")
    axes[1].set_ylabel("Mean SSIM")
    axes[1].set_xticks(x, df["group"], rotation=30, ha="right")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create report figures from CSV files.")
    parser.add_argument("--summary-csv", default="results/csv/failure_group_summary.csv")
    parser.add_argument("--baseline-csv", default="results/csv/baseline_per_image_metrics.csv")
    parser.add_argument("--before-after-csv", default="results/csv/before_after_summary.csv")
    parser.add_argument("--output-dir", default="results/figures")
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    if Path(args.summary_csv).exists():
        make_group_bar_chart(args.summary_csv, output_dir / "failure_group_bar_chart.png")
    if Path(args.baseline_csv).exists():
        for feature in ["blur_score", "saturation_ratio", "dark_ratio", "edge_density"]:
            make_scatter_plot(args.baseline_csv, feature, "psnr", output_dir / f"scatter_psnr_vs_{feature}.png")
    if Path(args.before_after_csv).exists():
        make_before_after_bar_chart(args.before_after_csv, output_dir / "before_after_bar_chart.png")


if __name__ == "__main__":
    main()

