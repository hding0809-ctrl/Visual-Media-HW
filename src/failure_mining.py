from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from .image_stats import calculate_image_stats
from .metrics import calculate_psnr, calculate_ssim
from .utils import (
    ImageCollection,
    assert_same_shape,
    common_image_ids,
    ensure_dir,
    load_yaml,
    safe_filename,
)
from .visualization import make_comparison_panel


def _quantile_mask(df: pd.DataFrame, col: str, q: float, direction: str) -> pd.Series:
    if direction == "low":
        return df[col] <= df[col].quantile(q)
    if direction == "high":
        return df[col] >= df[col].quantile(1.0 - q)
    raise ValueError(f"Unsupported direction: {direction}")


def build_failure_masks(df: pd.DataFrame, q: float) -> Dict[str, pd.Series]:
    sat_thr = df["saturation_ratio"].quantile(1.0 - q)
    dark_thr = df["dark_ratio"].quantile(1.0 - q)
    return {
        "overall": pd.Series(True, index=df.index),
        "heavy_blur": _quantile_mask(df, "blur_score", q, "low"),
        "light_blur": _quantile_mask(df, "blur_score", q, "high"),
        "high_saturation_or_dark": (df["saturation_ratio"] >= sat_thr) | (df["dark_ratio"] >= dark_thr),
        "high_texture": _quantile_mask(df, "edge_density", q, "high"),
        "low_texture": _quantile_mask(df, "edge_density", q, "low"),
    }


def summarize_groups(df: pd.DataFrame, masks: Dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for group, mask in masks.items():
        part = df[mask].copy()
        if part.empty:
            continue
        rows.append(
            {
                "group": group,
                "n": int(len(part)),
                "mean_psnr": float(part["psnr"].mean()),
                "median_psnr": float(part["psnr"].median()),
                "mean_ssim": float(part["ssim"].mean()),
                "median_ssim": float(part["ssim"].median()),
                "mean_blur_score": float(part["blur_score"].mean()),
                "mean_saturation_ratio": float(part["saturation_ratio"].mean()),
                "mean_dark_ratio": float(part["dark_ratio"].mean()),
                "mean_edge_density": float(part["edge_density"].mean()),
            }
        )
    return pd.DataFrame(rows)


def compute_per_image_metrics(
    input_collection: ImageCollection,
    gt_collection: ImageCollection,
    pred_collection: ImageCollection,
    image_ids: Iterable[str],
) -> pd.DataFrame:
    rows = []
    for image_id in tqdm(list(image_ids), desc="Evaluating baseline images"):
        input_img = input_collection.read(image_id)
        gt_img = gt_collection.read(image_id)
        pred_img = pred_collection.read(image_id)
        assert_same_shape(image_id, input_img, gt_img, pred_img)

        metrics = {
            "image_id": image_id,
            "input_path": input_collection.describe(image_id),
            "gt_path": gt_collection.describe(image_id),
            "pred_path": pred_collection.describe(image_id),
            "psnr": calculate_psnr(pred_img, gt_img, crop_border=0),
            "ssim": calculate_ssim(pred_img, gt_img, crop_border=0),
        }
        metrics.update(calculate_image_stats(input_img, gt_img))
        rows.append(metrics)
    return pd.DataFrame(rows)


def export_failure_panels(
    df: pd.DataFrame,
    masks: Dict[str, pd.Series],
    input_collection: ImageCollection,
    gt_collection: ImageCollection,
    pred_collection: ImageCollection,
    output_dir: Path,
    max_cases: int,
) -> None:
    panel_groups = ["heavy_blur", "high_saturation_or_dark", "high_texture"]
    for group in panel_groups:
        if group not in masks:
            continue
        group_dir = ensure_dir(output_dir / "failure_cases" / group)
        worst = df[masks[group]].sort_values("psnr", ascending=True).head(max_cases)
        for _, row in worst.iterrows():
            image_id = row["image_id"]
            input_img = input_collection.read(image_id)
            gt_img = gt_collection.read(image_id)
            pred_img = pred_collection.read(image_id)
            make_comparison_panel(
                input_img,
                pred_img,
                gt_img,
                error_map=None,
                metrics={"psnr": float(row["psnr"]), "ssim": float(row["ssim"])},
                save_path=group_dir / safe_filename(image_id),
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine NAFNet baseline failure cases.")
    parser.add_argument("--input", required=True, help="Blurry input path, directory or LMDB.")
    parser.add_argument("--gt", required=True, help="Ground-truth path, directory or LMDB.")
    parser.add_argument("--pred", required=True, help="Baseline prediction image directory.")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--config", default="configs/failure_analysis.yaml")
    args = parser.parse_args()

    config = load_yaml(args.config) if Path(args.config).exists() else {}
    q = float(config.get("quantile", 0.25))
    max_cases = int(config.get("max_cases_per_group", 8))
    extensions = config.get("image_extensions")

    input_collection = ImageCollection(args.input, extensions)
    gt_collection = ImageCollection(args.gt, extensions)
    pred_collection = ImageCollection(args.pred, extensions)
    image_ids = common_image_ids(input_collection, gt_collection, pred_collection)

    if not image_ids:
        raise RuntimeError(
            "No common image ids found.\n"
            f"input samples: {input_collection.sample_keys()}\n"
            f"gt samples: {gt_collection.sample_keys()}\n"
            f"pred samples: {pred_collection.sample_keys()}"
        )

    output_dir = Path(args.output_dir)
    csv_dir = ensure_dir(output_dir / "csv")
    ensure_dir(output_dir / "figures")

    df = compute_per_image_metrics(input_collection, gt_collection, pred_collection, image_ids)
    per_image_csv = csv_dir / "baseline_per_image_metrics.csv"
    df.to_csv(per_image_csv, index=False)

    masks = build_failure_masks(df, q)
    summary = summarize_groups(df, masks)
    summary_csv = csv_dir / "failure_group_summary.csv"
    summary.to_csv(summary_csv, index=False)

    export_failure_panels(df, masks, input_collection, gt_collection, pred_collection, output_dir, max_cases)

    print(f"Wrote {per_image_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Exported failure panels under {output_dir / 'failure_cases'}")


if __name__ == "__main__":
    main()
