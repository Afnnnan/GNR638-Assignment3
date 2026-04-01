"""
Utility functions for U-Net training.

Handles:
  - Device auto-detection (MPS / CUDA / CPU)
  - Reproducible seeding
  - Config loading
  - Metric helpers
"""

import os
import random
import yaml
import torch
import numpy as np


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    """Auto-detect the best available accelerator."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    print(f"[INFO] Using device: {dev}")
    return dev


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """Set seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # MPS doesn't have a separate manual_seed, torch.manual_seed covers it.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] Random seed set to {seed}")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_config(path: str = "configs/config.yaml") -> dict:
    """Load YAML config file and return as a dict."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ---------------------------------------------------------------------------
# Metrics (per-batch helpers; used in train.py and evaluate.py)
# ---------------------------------------------------------------------------
def dice_coefficient(pred: torch.Tensor, target: torch.Tensor,
                     smooth: float = 1e-6) -> float:
    """
    Dice coefficient for binary segmentation.

    Args:
        pred:   (N, H, W) predicted binary mask (0 or 1)
        target: (N, H, W) ground-truth binary mask
    Returns:
        Scalar Dice coefficient.
    """
    pred_flat = pred.contiguous().view(-1).float()
    target_flat = target.contiguous().view(-1).float()
    intersection = (pred_flat * target_flat).sum()
    return ((2.0 * intersection + smooth) /
            (pred_flat.sum() + target_flat.sum() + smooth)).item()


def iou_score(pred: torch.Tensor, target: torch.Tensor,
              smooth: float = 1e-6) -> float:
    """
    Intersection over Union (Jaccard Index).
    """
    pred_flat = pred.contiguous().view(-1).float()
    target_flat = target.contiguous().view(-1).float()
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    return ((intersection + smooth) / (union + smooth)).item()


def pixel_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Simple pixel-wise accuracy.
    """
    correct = (pred == target).sum().item()
    total = target.numel()
    return correct / total


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
