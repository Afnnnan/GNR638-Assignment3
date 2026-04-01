"""
Training loop for U-Net.

Follows the paper:
  • SGD with momentum 0.99
  • Initial LR 0.01 with ReduceLROnPlateau scheduler
  • Weighted cross-entropy loss (pixel-wise weight map from Eq. 2)
  • Tracks training/validation Dice, loss per epoch
  • Saves best checkpoint
"""

import os
import time
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (works on servers)
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from src.model import UNet
from src.dataset import build_dataloaders
from src.utils import (
    get_device, set_seed, load_config, ensure_dir,
    dice_coefficient, iou_score, pixel_accuracy,
)


# ═══════════════════════════════════════════════════════════════════════
# Weighted cross-entropy (paper Section 3.1)
# ═══════════════════════════════════════════════════════════════════════

def weighted_cross_entropy(logits: torch.Tensor,
                           targets: torch.Tensor,
                           weight_maps: torch.Tensor) -> torch.Tensor:
    """
    Pixel-wise cross-entropy weighted by the pre-computed weight map.

    Args:
        logits:      (N, C, H, W) raw network output
        targets:     (N, H, W) long tensor with class labels
        weight_maps: (N, H, W) float tensor of per-pixel weights
    Returns:
        Scalar loss.
    """
    ce = nn.functional.cross_entropy(logits, targets, reduction='none')  # (N,H,W)
    weighted = ce * weight_maps
    return weighted.mean()


# ═══════════════════════════════════════════════════════════════════════
# Training & validation steps
# ═══════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, device, use_weight_map):
    model.train()
    epoch_loss = 0.0
    epoch_dice = 0.0
    n_batches = 0

    for images, masks, wmaps in loader:
        images = images.to(device)
        masks = masks.to(device)
        wmaps = wmaps.to(device)

        optimizer.zero_grad()
        logits = model(images)

        if use_weight_map:
            loss = weighted_cross_entropy(logits, masks, wmaps)
        else:
            loss = nn.functional.cross_entropy(logits, masks)

        loss.backward()
        optimizer.step()

        # Metrics
        preds = logits.argmax(dim=1)
        epoch_loss += loss.item()
        epoch_dice += dice_coefficient(preds, masks)
        n_batches += 1

    return epoch_loss / n_batches, epoch_dice / n_batches


@torch.no_grad()
def validate(model, loader, device, use_weight_map):
    model.eval()
    epoch_loss = 0.0
    epoch_dice = 0.0
    epoch_iou = 0.0
    epoch_acc = 0.0
    n_batches = 0

    for images, masks, wmaps in loader:
        images = images.to(device)
        masks = masks.to(device)
        wmaps = wmaps.to(device)

        logits = model(images)

        if use_weight_map:
            loss = weighted_cross_entropy(logits, masks, wmaps)
        else:
            loss = nn.functional.cross_entropy(logits, masks)

        preds = logits.argmax(dim=1)
        epoch_loss += loss.item()
        epoch_dice += dice_coefficient(preds, masks)
        epoch_iou += iou_score(preds, masks)
        epoch_acc += pixel_accuracy(preds, masks)
        n_batches += 1

    return (epoch_loss / n_batches,
            epoch_dice / n_batches,
            epoch_iou / n_batches,
            epoch_acc / n_batches)


# ═══════════════════════════════════════════════════════════════════════
# Main training routine
# ═══════════════════════════════════════════════════════════════════════

def train(cfg: dict):
    """Run full training and return the path to the best checkpoint."""

    # Setup
    set_seed(cfg.get("seed", 42))
    device = get_device()
    out_dir = cfg.get("output_dir", "./outputs")
    ensure_dir(out_dir)

    # Data
    train_loader, val_loader = build_dataloaders(cfg)

    # Model
    model = UNet(in_channels=1, n_classes=cfg.get("n_classes", 2))
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] U-Net created  —  {n_params:,} parameters")

    # Optimizer (paper: SGD, momentum=0.99)
    optimizer = optim.SGD(model.parameters(),
                          lr=cfg.get("lr", 0.01),
                          momentum=cfg.get("momentum", 0.99),
                          weight_decay=cfg.get("weight_decay", 0.0005))

    # Scheduler (verbose removed in PyTorch 2.4+)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=15)

    use_weight_map = cfg.get("use_weight_map", True)

    # Tracking
    history = {
        "train_loss": [], "train_dice": [],
        "val_loss": [], "val_dice": [], "val_iou": [], "val_acc": [],
    }
    best_dice = 0.0
    best_epoch = 0
    ckpt_path = os.path.join(out_dir, cfg.get("checkpoint_name", "unet_best.pth"))

    epochs = cfg.get("epochs", 200)
    print(f"\n{'='*60}")
    print(f"  Training for {epochs} epochs  |  device={device}")
    print(f"{'='*60}\n")

    t_start = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_dice = train_one_epoch(
            model, train_loader, optimizer, device, use_weight_map)

        val_loss, val_dice, val_iou, val_acc = validate(
            model, val_loader, device, use_weight_map)

        lr_before = optimizer.param_groups[0]["lr"]
        scheduler.step(val_dice)
        lr_after = optimizer.param_groups[0]["lr"]
        if lr_after < lr_before:
            print(f"  ↓ LR reduced: {lr_before:.6f} → {lr_after:.6f}")

        # Log
        history["train_loss"].append(train_loss)
        history["train_dice"].append(train_dice)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs}  |  "
                  f"train_loss={train_loss:.4f}  train_dice={train_dice:.4f}  |  "
                  f"val_loss={val_loss:.4f}  val_dice={val_dice:.4f}  "
                  f"val_iou={val_iou:.4f}  val_acc={val_acc:.4f}  |  "
                  f"lr={lr_now:.6f}  time={elapsed:.1f}s")

        # Checkpoint
        if val_dice > best_dice:
            best_dice = val_dice
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": val_dice,
                "val_iou": val_iou,
                "config": cfg,
            }, ckpt_path)

    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Training complete in {total_time/60:.1f} min")
    print(f"  Best val Dice: {best_dice:.4f}  (epoch {best_epoch})")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"{'='*60}\n")

    # Save history
    history_path = os.path.join(out_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Plot training curves
    _plot_curves(history, out_dir)

    return ckpt_path, history


# ═══════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════

def _plot_curves(history: dict, out_dir: str):
    """Save training/validation loss and Dice curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", linewidth=1.5)
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", linewidth=1.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Dice
    axes[1].plot(epochs, history["train_dice"], label="Train Dice", linewidth=1.5)
    axes[1].plot(epochs, history["val_dice"], label="Val Dice", linewidth=1.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice Coefficient")
    axes[1].set_title("Training & Validation Dice")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "training_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Training curves saved to {path}")


# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    cfg = load_config()
    train(cfg)
