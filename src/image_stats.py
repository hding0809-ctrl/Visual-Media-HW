from __future__ import annotations

import cv2
import numpy as np


def _uint8_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape HxWx3, got {arr.shape}")
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    if arr.size and arr.max() <= 1.0:
        arr *= 255.0
    return np.clip(arr, 0, 255).round().astype(np.uint8)


def calculate_image_stats(input_image: np.ndarray, gt_image: np.ndarray | None = None) -> dict:
    """Calculate simple image statistics used for failure mining."""
    rgb = _uint8_rgb(input_image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    blur_score = float(lap.var())

    saturation_ratio = float(np.mean(np.max(rgb, axis=2) > 250))
    dark_ratio = float(np.mean(gray < 30))

    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(np.mean(edges > 0))

    hist = np.bincount(gray.reshape(-1), minlength=256).astype(np.float64)
    prob = hist / max(hist.sum(), 1.0)
    prob = prob[prob > 0]
    entropy = float(-(prob * np.log2(prob)).sum())

    brightness = float(gray.mean() / 255.0)
    contrast = float(gray.std() / 255.0)

    stats = {
        "blur_score": blur_score,
        "saturation_ratio": saturation_ratio,
        "dark_ratio": dark_ratio,
        "edge_density": edge_density,
        "entropy": entropy,
        "brightness": brightness,
        "contrast": contrast,
    }

    if gt_image is not None:
        gt = _uint8_rgb(gt_image)
        gt_gray = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY)
        stats["gt_brightness"] = float(gt_gray.mean() / 255.0)
        stats["gt_contrast"] = float(gt_gray.std() / 255.0)

    return stats

