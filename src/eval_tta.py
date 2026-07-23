from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .metrics import calculate_psnr, calculate_ssim
from .tta import crop_to_original, pad_to_multiple, self_ensemble_forward
from .utils import (
    ImageCollection,
    assert_same_shape,
    common_image_ids,
    ensure_dir,
    load_yaml,
    safe_filename,
    write_image_rgb,
)
from .visualization import make_before_after_panel


def _state_dict_from_checkpoint(checkpoint: object) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("params_ema", "params", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError("Unsupported checkpoint format")


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(k.startswith("module.") for k in state_dict):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def build_nafnet_model(
    nafnet_root: str | Path,
    config_path: str | Path,
    checkpoint_path: str | Path,
    device: torch.device,
) -> torch.nn.Module:
    nafnet_root = Path(nafnet_root)
    sys.path.insert(0, str(nafnet_root))

    try:
        from basicsr.models.archs.NAFNet_arch import NAFNet, NAFNetLocal
    except Exception as exc:
        raise ImportError(
            "Could not import NAFNet architecture. Make sure third_party/NAFNet exists "
            "and its requirements are installed."
        ) from exc

    cfg = load_yaml(config_path)
    net_opt = dict(cfg.get("network_g", {}))
    net_type = net_opt.pop("type", "NAFNetLocal")
    if net_type == "NAFNetLocal":
        model = NAFNetLocal(**net_opt)
    elif net_type == "NAFNet":
        model = NAFNet(**net_opt)
    else:
        raise ValueError(f"Unsupported network_g.type for this TTA script: {net_type}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _strip_module_prefix(_state_dict_from_checkpoint(checkpoint))
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def image_to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    arr = image.astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().float().cpu().clamp(0.0, 1.0)
    arr = tensor.squeeze(0).permute(1, 2, 0).numpy()
    return np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)


@torch.no_grad()
def run_tta_on_image(model: torch.nn.Module, image: np.ndarray, device: torch.device) -> tuple[np.ndarray, float]:
    x = image_to_tensor(image, device)
    h, w = x.shape[-2:]
    x_pad, _ = pad_to_multiple(x, multiple=16)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    pred = self_ensemble_forward(model, x_pad)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    pred = crop_to_original(pred, (h, w))
    return tensor_to_image(pred), elapsed


def build_before_after_summary(baseline_csv: Path, tta_csv: Path, summary_csv: Path) -> pd.DataFrame:
    baseline = pd.read_csv(baseline_csv).rename(columns={"psnr": "baseline_psnr", "ssim": "baseline_ssim"})
    tta = pd.read_csv(tta_csv).rename(columns={"psnr": "tta_psnr", "ssim": "tta_ssim"})
    merged = baseline.merge(tta[["image_id", "tta_psnr", "tta_ssim", "inference_time_seconds"]], on="image_id")
    merged["delta_psnr"] = merged["tta_psnr"] - merged["baseline_psnr"]
    merged["delta_ssim"] = merged["tta_ssim"] - merged["baseline_ssim"]

    q = 0.25
    masks = {
        "overall": pd.Series(True, index=merged.index),
        "heavy_blur": merged["blur_score"] <= merged["blur_score"].quantile(q),
        "high_saturation_or_dark": (merged["saturation_ratio"] >= merged["saturation_ratio"].quantile(1.0 - q))
        | (merged["dark_ratio"] >= merged["dark_ratio"].quantile(1.0 - q)),
        "high_texture": merged["edge_density"] >= merged["edge_density"].quantile(1.0 - q),
    }

    rows = []
    for group, mask in masks.items():
        part = merged[mask]
        if part.empty:
            continue
        rows.append(
            {
                "group": group,
                "n": int(len(part)),
                "baseline_psnr_mean": float(part["baseline_psnr"].mean()),
                "tta_psnr_mean": float(part["tta_psnr"].mean()),
                "delta_psnr_mean": float(part["delta_psnr"].mean()),
                "baseline_ssim_mean": float(part["baseline_ssim"].mean()),
                "tta_ssim_mean": float(part["tta_ssim"].mean()),
                "delta_ssim_mean": float(part["delta_ssim"].mean()),
                "tta_time_seconds_mean": float(part["inference_time_seconds"].mean()),
            }
        )
    summary = pd.DataFrame(rows)
    ensure_dir(summary_csv.parent)
    summary.to_csv(summary_csv, index=False)
    return merged


def export_improved_examples(
    merged: pd.DataFrame,
    input_collection: ImageCollection,
    gt_collection: ImageCollection,
    baseline_pred: str | Path,
    tta_pred: str | Path,
    output_dir: Path,
    max_examples: int,
) -> None:
    baseline_path = Path(baseline_pred)
    if not baseline_path.exists():
        print(f"Skipping before/after panels because baseline predictions were not found: {baseline_path}")
        return

    baseline_collection = ImageCollection(baseline_path)
    tta_collection = ImageCollection(tta_pred)
    common = set(merged["image_id"]) & baseline_collection.keys() & tta_collection.keys()
    if not common:
        print("Skipping before/after panels because no matching baseline/TTA images were found.")
        return

    ensure_dir(output_dir)
    top = merged[merged["image_id"].isin(common)].sort_values("delta_psnr", ascending=False).head(max_examples)
    for _, row in top.iterrows():
        image_id = row["image_id"]
        make_before_after_panel(
            input_collection.read(image_id),
            baseline_collection.read(image_id),
            tta_collection.read(image_id),
            gt_collection.read(image_id),
            {"psnr": float(row["baseline_psnr"]), "ssim": float(row["baseline_ssim"])},
            {"psnr": float(row["tta_psnr"]), "ssim": float(row["tta_ssim"])},
            output_dir / safe_filename(image_id),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NAFNet with D4 test-time augmentation.")
    parser.add_argument("--nafnet-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--output-dir", default="results/tta/images")
    parser.add_argument("--csv", default="results/csv/tta_per_image_metrics.csv")
    parser.add_argument("--summary-csv", default="results/csv/before_after_summary.csv")
    parser.add_argument("--baseline-csv", default="results/csv/baseline_per_image_metrics.csv")
    parser.add_argument("--baseline-pred", default="")
    parser.add_argument("--max-examples", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = ensure_dir(args.output_dir)
    csv_path = Path(args.csv)
    ensure_dir(csv_path.parent)

    input_collection = ImageCollection(args.input)
    gt_collection = ImageCollection(args.gt)
    image_ids = common_image_ids(input_collection, gt_collection)
    if not image_ids:
        raise RuntimeError("No common ids found between input and GT collections.")

    model = build_nafnet_model(args.nafnet_root, args.config, args.checkpoint, device)

    rows: List[dict] = []
    for image_id in tqdm(image_ids, desc="D4-TTA evaluation"):
        input_img = input_collection.read(image_id)
        gt_img = gt_collection.read(image_id)
        assert_same_shape(image_id, input_img, gt_img)

        pred_img, elapsed = run_tta_on_image(model, input_img, device)
        assert_same_shape(image_id, pred_img, gt_img)

        save_path = output_dir / safe_filename(image_id)
        write_image_rgb(save_path, pred_img)

        rows.append(
            {
                "image_id": image_id,
                "pred_path": str(save_path),
                "psnr": calculate_psnr(pred_img, gt_img, crop_border=0),
                "ssim": calculate_ssim(pred_img, gt_img, crop_border=0),
                "inference_time_seconds": elapsed,
            }
        )

    tta_df = pd.DataFrame(rows)
    tta_df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    baseline_csv = Path(args.baseline_csv)
    if baseline_csv.exists():
        merged = build_before_after_summary(baseline_csv, csv_path, Path(args.summary_csv))
        print(f"Wrote {args.summary_csv}")
        if args.baseline_pred:
            export_improved_examples(
                merged,
                input_collection,
                gt_collection,
                args.baseline_pred,
                output_dir,
                Path("results/failure_cases/tta_improved_examples"),
                args.max_examples,
            )
    else:
        print(f"Skipping before/after summary because baseline CSV was not found: {baseline_csv}")


if __name__ == "__main__":
    main()

