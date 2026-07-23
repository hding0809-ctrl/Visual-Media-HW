from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F


D4_TRANSFORMS: Tuple[str, ...] = (
    "identity",
    "hflip",
    "vflip",
    "hvflip",
    "transpose",
    "transpose_hflip",
    "transpose_vflip",
    "transpose_hvflip",
)


def _ops_for_transform(name: str) -> List[str]:
    table = {
        "identity": [],
        "hflip": ["hflip"],
        "vflip": ["vflip"],
        "hvflip": ["hflip", "vflip"],
        "transpose": ["transpose"],
        "transpose_hflip": ["transpose", "hflip"],
        "transpose_vflip": ["transpose", "vflip"],
        "transpose_hvflip": ["transpose", "hflip", "vflip"],
    }
    if name not in table:
        raise ValueError(f"Unknown D4 transform: {name}")
    return table[name]


def _apply_op(x: torch.Tensor, op: str) -> torch.Tensor:
    if op == "hflip":
        return torch.flip(x, dims=(-1,))
    if op == "vflip":
        return torch.flip(x, dims=(-2,))
    if op == "transpose":
        return x.transpose(-1, -2)
    raise ValueError(f"Unknown op: {op}")


def apply_transform(x: torch.Tensor, name: str) -> torch.Tensor:
    y = x
    for op in _ops_for_transform(name):
        y = _apply_op(y, op)
    return y


def invert_transform(x: torch.Tensor, name: str) -> torch.Tensor:
    y = x
    for op in reversed(_ops_for_transform(name)):
        y = _apply_op(y, op)
    return y


def pad_to_multiple(x: torch.Tensor, multiple: int = 16) -> tuple[torch.Tensor, tuple[int, int]]:
    if x.ndim != 4:
        raise ValueError(f"Expected BCHW tensor, got shape {tuple(x.shape)}")
    h, w = x.shape[-2:]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)
    padded = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return padded, (pad_h, pad_w)


def crop_to_original(x: torch.Tensor, original_hw: tuple[int, int]) -> torch.Tensor:
    h, w = original_hw
    return x[..., :h, :w]


@torch.no_grad()
def self_ensemble_forward(
    model: torch.nn.Module,
    x: torch.Tensor,
    transforms: Sequence[str] = D4_TRANSFORMS,
) -> torch.Tensor:
    preds = []
    for name in transforms:
        transformed = apply_transform(x, name)
        pred = model(transformed)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        if isinstance(pred, dict):
            pred = pred.get("output", next(iter(pred.values())))
        preds.append(invert_transform(pred, name))
    out = torch.stack(preds, dim=0).mean(dim=0)
    return out.clamp(0.0, 1.0)

