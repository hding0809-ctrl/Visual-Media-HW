from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from .utils import read_image_rgb


def _to_float_255(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim not in (2, 3):
        raise ValueError(f"Expected a 2D or 3D image array, got shape {arr.shape}")
    arr = arr.astype(np.float64)
    if arr.size and arr.max() <= 1.0:
        arr *= 255.0
    return arr


def _crop_border(image: np.ndarray, crop_border: int) -> np.ndarray:
    if crop_border <= 0:
        return image
    if image.shape[0] <= 2 * crop_border or image.shape[1] <= 2 * crop_border:
        raise ValueError(f"crop_border={crop_border} is too large for shape {image.shape}")
    return image[crop_border:-crop_border, crop_border:-crop_border, ...]


def calculate_psnr(pred: np.ndarray, gt: np.ndarray, crop_border: int = 0) -> float:
    """Calculate RGB PSNR with pixel range [0, 255]."""
    pred = _crop_border(_to_float_255(pred), crop_border)
    gt = _crop_border(_to_float_255(gt), crop_border)
    if pred.shape != gt.shape:
        raise ValueError(f"PSNR shape mismatch: pred={pred.shape}, gt={gt.shape}")
    mse = np.mean((pred - gt) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def calculate_ssim(pred: np.ndarray, gt: np.ndarray, crop_border: int = 0) -> float:
    """Calculate SSIM with the NAFNet/BasicSR metric convention."""
    pred = _crop_border(_to_float_255(pred), crop_border)
    gt = _crop_border(_to_float_255(gt), crop_border)
    if pred.shape != gt.shape:
        raise ValueError(f"SSIM shape mismatch: pred={pred.shape}, gt={gt.shape}")

    if pred.ndim == 2:
        return _ssim_2d(pred, gt, max_value=255.0)
    if pred.ndim == 3 and pred.shape[2] == 1:
        return _ssim_2d(pred[..., 0], gt[..., 0], max_value=255.0)
    if pred.ndim == 3:
        return _ssim_3d_or_channelwise(pred, gt, max_value=255.0)
    raise ValueError(f"Unsupported SSIM image shape: {pred.shape}")


def _ssim_2d(img1: np.ndarray, img2: np.ndarray, max_value: float) -> float:
    c1 = (0.01 * max_value) ** 2
    c2 = (0.03 * max_value) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return float(ssim_map.mean())


def _ssim_3d_or_channelwise(img1: np.ndarray, img2: np.ndarray, max_value: float) -> float:
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        return float(np.mean([_ssim_2d(img1[..., i], img2[..., i], max_value) for i in range(img1.shape[2])]))

    if not torch.cuda.is_available():
        return float(np.mean([_ssim_2d(img1[..., i], img2[..., i], max_value) for i in range(img1.shape[2])]))

    c1 = (0.01 * max_value) ** 2
    c2 = (0.03 * max_value) ** 2
    device = torch.device("cuda")

    kernel_1d = cv2.getGaussianKernel(11, 1.5).astype(np.float32)
    window = np.outer(kernel_1d, kernel_1d.transpose())
    kernel = np.stack([window * k for k in kernel_1d], axis=0)
    kernel_t = torch.from_numpy(kernel).to(device=device, dtype=torch.float32).view(1, 1, 11, 11, 11)

    x1 = torch.from_numpy(img1.astype(np.float32)).to(device).unsqueeze(0).unsqueeze(0)
    x2 = torch.from_numpy(img2.astype(np.float32)).to(device).unsqueeze(0).unsqueeze(0)

    def filt(x: torch.Tensor) -> torch.Tensor:
        return F.conv3d(F.pad(x, (5, 5, 5, 5, 5, 5), mode="replicate"), kernel_t)

    with torch.no_grad():
        mu1 = filt(x1)
        mu2 = filt(x2)
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = filt(x1**2) - mu1_sq
        sigma2_sq = filt(x2**2) - mu2_sq
        sigma12 = filt(x1 * x2) - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        )
        return float(ssim_map.mean().detach().cpu())


def read_image(path: str | Path) -> np.ndarray:
    return read_image_rgb(path)
