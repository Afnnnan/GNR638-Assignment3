"""
Evaluation & Comparison for U-Net.

Generates:
  1. Per-image metrics (Dice, IoU, Pixel Acc, Precision, Recall)
  2. Prediction visualisations  (input | ground truth | prediction)
  3. Architecture comparison table  (our PyTorch vs Caffe prototxt)
  4. Summary comparison with paper's reported results
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import glob

import torch
import torch.nn as nn

from src.model import UNet
from src.dataset import build_dataloaders, load_tiff_stack
from src.utils import (
    get_device, set_seed, load_config, ensure_dir,
    dice_coefficient, iou_score, pixel_accuracy,
)


# ═══════════════════════════════════════════════════════════════════════
# Detailed metrics
# ═══════════════════════════════════════════════════════════════════════

def precision_recall(pred: torch.Tensor, target: torch.Tensor, smooth=1e-6):
    """Compute precision and recall for binary segmentation."""
    pred_flat = pred.view(-1).float()
    target_flat = target.view(-1).float()
    tp = (pred_flat * target_flat).sum()
    precision = (tp + smooth) / (pred_flat.sum() + smooth)
    recall = (tp + smooth) / (target_flat.sum() + smooth)
    return precision.item(), recall.item()


# ═══════════════════════════════════════════════════════════════════════
# ISBI metrics & SEG score
# ═══════════════════════════════════════════════════════════════════════


def compute_isbi_metrics(all_masks, all_preds):
    """
    Compute ISBI 2012 challenge metrics on the validation split:
    - Pixel Error: fraction of misclassified pixels
    - Foreground-restricted Rand Error: via adapted_rand_error

    For Rand Error, we invert membrane predictions to get neuron
    instances (connected components of non-membrane regions).
    """
    from scipy.ndimage import label as ndlabel
    from skimage.metrics import adapted_rand_error

    pixel_errors = []
    rand_errors = []
    rand_precisions = []
    rand_recalls = []

    for i in range(len(all_masks)):
        pred = all_preds[i]
        mask = all_masks[i]

        # Pixel Error = fraction of misclassified pixels
        pixel_error = float(np.mean(pred != mask))
        pixel_errors.append(pixel_error)

        # For Rand Error: invert membrane mask to get neuron interiors,
        # then use connected components to get instance labels.
        # ISBI 2012: class 1 = membrane, class 0 = neuron interior
        gt_instances, _ = ndlabel(mask == 0)
        pred_instances, _ = ndlabel(pred == 0)

        error, precision, recall = adapted_rand_error(
            gt_instances, pred_instances)
        rand_errors.append(error)
        rand_precisions.append(precision)
        rand_recalls.append(recall)

    result = {
        "pixel_error": {
            "mean": float(np.mean(pixel_errors)),
            "std": float(np.std(pixel_errors)),
        },
        "rand_error": {
            "mean": float(np.mean(rand_errors)),
            "std": float(np.std(rand_errors)),
        },
        "rand_precision": {
            "mean": float(np.mean(rand_precisions)),
            "std": float(np.std(rand_precisions)),
        },
        "rand_recall": {
            "mean": float(np.mean(rand_recalls)),
            "std": float(np.std(rand_recalls)),
        },
    }

    print(f"\n[ISBI] Pixel Error:  {result['pixel_error']['mean']:.4f} "
          f"± {result['pixel_error']['std']:.4f}")
    print(f"[ISBI] Rand Error:   {result['rand_error']['mean']:.4f} "
          f"± {result['rand_error']['std']:.4f}")

    return result


@torch.no_grad()
def compute_seg_score(cfg, model, device):
    """
    Compute the Cell Tracking Challenge SEG score (instance-level Jaccard).

    For each GT instance R, find the predicted instance S such that
    |R ∩ S| > 0.5·|R|, then compute J(S,R) = |R ∩ S| / |R ∪ S|.
    Unmatched instances score 0.  SEG = mean J over all GT instances.

    Reference: http://celltrackingchallenge.net/evaluation-methodology/
    """
    from scipy.ndimage import label as ndlabel
    from PIL import Image as PILImage

    dataset_type = cfg.get("dataset_type")
    if dataset_type not in ("phc", "dic"):
        return None

    data_dir = cfg["data_dir"]
    sequences = cfg.get("sequences", ["01", "02"])
    image_size = cfg.get("image_size", 512)

    all_jaccard = []
    n_matched = 0
    n_total = 0

    for seq in sequences:
        gt_seg_dir = os.path.join(data_dir, f"{seq}_GT", "SEG")
        img_dir = os.path.join(data_dir, seq)

        if not os.path.isdir(gt_seg_dir):
            print(f"  [SEG] GT/SEG not found: {gt_seg_dir}, skipping")
            continue

        gt_files = sorted(glob.glob(os.path.join(gt_seg_dir, "*.tif")))

        for gt_path in gt_files:
            basename = os.path.splitext(os.path.basename(gt_path))[0]
            num_str = ''.join(c for c in basename if c.isdigit())
            if not num_str:
                continue
            frame_num = int(num_str)

            # Load GT instance mask (uint16: 0=bg, 1,2,...=instances)
            gt_instance_mask = np.array(PILImage.open(gt_path))
            orig_h, orig_w = gt_instance_mask.shape

            # Find corresponding raw image
            img_path = os.path.join(img_dir, f"t{frame_num:03d}.tif")
            if not os.path.exists(img_path):
                continue

            raw_image = np.array(PILImage.open(img_path))

            # Preprocess (replicate training pipeline)
            img_pil = PILImage.fromarray(raw_image.astype(np.uint8))
            img_resized = img_pil.resize(
                (image_size, image_size), PILImage.BILINEAR)
            img_np = np.array(img_resized).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)
            img_tensor = img_tensor.to(device)

            # Inference
            logits = model(img_tensor)
            pred_binary = logits.argmax(dim=1).cpu().numpy()[0]

            # Resize prediction back to original resolution
            pred_pil = PILImage.fromarray(pred_binary.astype(np.uint8))
            pred_orig = np.array(
                pred_pil.resize((orig_w, orig_h), PILImage.NEAREST))

            # Binary → instance labels via connected components
            from scipy.ndimage import binary_fill_holes, label as ndlabel
            filled = binary_fill_holes(pred_orig > 0)
            pred_instances, n_pred = ndlabel(filled)

            # Remove small segments (paper uses minSegmAreaPx=500)
            if n_pred > 0:
                for inst_id in range(1, n_pred + 1):
                    if (pred_instances == inst_id).sum() < 500:
                        pred_instances[pred_instances == inst_id] = 0
                pred_instances, n_pred = ndlabel(pred_instances > 0)

            # Compute per-instance Jaccard (SEG metric)
            gt_ids = np.unique(gt_instance_mask)
            gt_ids = gt_ids[gt_ids > 0]

            for gt_id in gt_ids:
                gt_region = (gt_instance_mask == gt_id)
                gt_area = gt_region.sum()
                n_total += 1

                best_jaccard = 0.0
                for pred_id in range(1, n_pred + 1):
                    pred_region = (pred_instances == pred_id)
                    intersection = (gt_region & pred_region).sum()
                    if intersection > 0.5 * gt_area:
                        union = (gt_region | pred_region).sum()
                        best_jaccard = float(intersection) / float(union)
                        n_matched += 1
                        break

                all_jaccard.append(best_jaccard)

    if len(all_jaccard) == 0:
        return None

    seg_score = float(np.mean(all_jaccard))
    seg_std = float(np.std(all_jaccard))

    print(f"\n[SEG] CTC SEG Score: {seg_score:.4f} ± {seg_std:.4f}")
    print(f"[SEG] Matched {n_matched}/{n_total} GT instances")

    return {
        "seg_score": {"mean": seg_score, "std": seg_std},
        "n_gt_instances": n_total,
        "n_matched": n_matched,
        "per_instance_jaccard": [float(j) for j in all_jaccard],
    }


# ═══════════════════════════════════════════════════════════════════════
# Evaluation on validation set
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(cfg: dict, checkpoint_path: str = None):
    """
    Load best checkpoint, evaluate on validation set, generate visuals.
    """
    set_seed(cfg.get("seed", 42))
    device = get_device()
    out_dir = cfg.get("output_dir", "./outputs")
    dataset_type = cfg.get("dataset_type", "isbi2012")
    ensure_dir(out_dir)

    # Resolve checkpoint
    if checkpoint_path is None:
        checkpoint_path = os.path.join(
            out_dir, cfg.get("checkpoint_name", "unet_best.pth"))

    # Load model
    model = UNet(in_channels=1, n_classes=cfg.get("n_classes", 2))
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"[EVAL] Loaded checkpoint from epoch {ckpt.get('epoch', '?')} "
          f"(val_dice={ckpt.get('val_dice', '?'):.4f})")

    # Data
    _, val_loader = build_dataloaders(cfg)

    # ---- Per-image metrics ----
    all_metrics = []
    all_images = []
    all_masks = []
    all_preds = []

    for images, masks, wmaps in val_loader:
        images_dev = images.to(device)
        logits = model(images_dev)
        preds = logits.argmax(dim=1).cpu()

        for i in range(images.size(0)):
            img_np = images[i, 0].numpy()
            mask_np = masks[i].numpy()
            pred_np = preds[i].numpy()

            d = dice_coefficient(preds[i:i+1], masks[i:i+1])
            iou = iou_score(preds[i:i+1], masks[i:i+1])
            acc = pixel_accuracy(preds[i:i+1], masks[i:i+1])
            prec, rec = precision_recall(preds[i], masks[i])

            all_metrics.append({
                "dice": d, "iou": iou, "pixel_acc": acc,
                "precision": prec, "recall": rec,
            })
            all_images.append(img_np)
            all_masks.append(mask_np)
            all_preds.append(pred_np)

    # ---- Aggregate ----
    keys = ["dice", "iou", "pixel_acc", "precision", "recall"]
    agg = {k: np.mean([m[k] for m in all_metrics]) for k in keys}
    agg_std = {k: np.std([m[k] for m in all_metrics]) for k in keys}

    print("\n" + "="*60)
    print("  Validation Results (Our PyTorch U-Net)")
    print("="*60)
    for k in keys:
        print(f"  {k:12s}:  {agg[k]:.4f} ± {agg_std[k]:.4f}")
    print("="*60 + "\n")

    # Save metrics
    results = {
        "per_image": all_metrics,
        "aggregate": {k: {"mean": agg[k], "std": agg_std[k]} for k in keys},
        "checkpoint_epoch": ckpt.get("epoch", "?"),
    }
    metrics_path = os.path.join(out_dir, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[EVAL] Metrics saved to {metrics_path}")

    # ---- Prediction visualisations ----
    _plot_predictions(all_images, all_masks, all_preds, out_dir, dataset_type)

    # ---- SEG score (Cell Tracking Challenge metric) ----
    seg_result = None
    if dataset_type in ("phc", "dic"):
        seg_result = compute_seg_score(cfg, model, device)
        if seg_result is not None:
            results["seg_score"] = seg_result
            with open(metrics_path, "w") as f:
                json.dump(results, f, indent=2)

    # ---- ISBI 2012 challenge metrics (Pixel Error, Rand Error) ----
    isbi_result = None
    if dataset_type == "isbi2012":
        isbi_result = compute_isbi_metrics(all_masks, all_preds)
        results["isbi_metrics"] = isbi_result
        with open(metrics_path, "w") as f:
            json.dump(results, f, indent=2)

    # ---- Architecture comparison ----
    _architecture_comparison(out_dir)

    # ---- Paper comparison ----
    _paper_comparison(agg, out_dir, dataset_type,
                      seg_result=seg_result, isbi_result=isbi_result)

    return results


# ═══════════════════════════════════════════════════════════════════════
# Prediction overlay visualisation
# ═══════════════════════════════════════════════════════════════════════

DATASET_NAMES = {
    "isbi2012": "ISBI 2012 EM Segmentation",
    "phc": "PhC-C2DH-U373 (Phase Contrast)",
    "dic": "DIC-C2DH-HeLa (DIC Microscopy)",
}


def _plot_predictions(images, masks, preds, out_dir, dataset_type="isbi2012"):
    """Plot side-by-side: Input | Ground Truth | Prediction for each val image."""
    ds_name = DATASET_NAMES.get(dataset_type, dataset_type)
    n = len(images)
    # Cap at 12 images to keep plots manageable
    n_show = min(n, 12)
    fig, axes = plt.subplots(n_show, 3, figsize=(15, 5 * n_show))
    if n_show == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_show):
        axes[i, 0].imshow(images[i], cmap="gray")
        axes[i, 0].set_title(f"Input #{i+1}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(masks[i], cmap="gray")
        axes[i, 1].set_title(f"Ground Truth #{i+1}")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(preds[i], cmap="gray")
        axes[i, 2].set_title(f"Prediction #{i+1}")
        axes[i, 2].axis("off")

    plt.suptitle(f"U-Net Predictions — {ds_name}", fontsize=16, y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, "predictions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Predictions saved to {path}")

    # Also make an overlay version
    n_overlay = min(n_show, 8)
    fig2, axes2 = plt.subplots(1, n_overlay, figsize=(5 * n_overlay, 5))
    if n_overlay == 1:
        axes2 = [axes2]
    for i in range(n_overlay):
        axes2[i].imshow(images[i], cmap="gray")
        axes2[i].imshow(preds[i], cmap="Reds", alpha=0.35)
        axes2[i].set_title(f"Overlay #{i+1}")
        axes2[i].axis("off")
    plt.suptitle(f"Prediction Overlays — {ds_name}", fontsize=14, y=1.01)
    plt.tight_layout()
    path2 = os.path.join(out_dir, "prediction_overlays.png")
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"[PLOT] Overlays saved to {path2}")


# ═══════════════════════════════════════════════════════════════════════
# Architecture comparison table
# ═══════════════════════════════════════════════════════════════════════

def _architecture_comparison(out_dir):
    """Generate a markdown file comparing our PyTorch arch vs Caffe prototxt."""
    table = """# Architecture Comparison: PyTorch vs Official Caffe

| Component | Paper / Caffe Prototxt | Our PyTorch Implementation | Match? |
|---|---|---|---|
| **Encoder** | | | |
| Level 0 | 2× Conv3×3 (64 ch), ReLU, MaxPool 2×2 | ✓ Same (+ BatchNorm) | ✅ |
| Level 1 | 2× Conv3×3 (128 ch), ReLU, MaxPool 2×2 | ✓ Same (+ BatchNorm) | ✅ |
| Level 2 | 2× Conv3×3 (256 ch), ReLU, MaxPool 2×2 | ✓ Same (+ BatchNorm) | ✅ |
| Level 3 | 2× Conv3×3 (512 ch), ReLU, Dropout 0.5, MaxPool 2×2 | ✓ Same (+ BatchNorm) | ✅ |
| **Bottleneck** | 2× Conv3×3 (1024 ch), ReLU, Dropout 0.5 | ✓ Same (+ BatchNorm) | ✅ |
| **Decoder** | | | |
| Level 3 | UpConv 2×2 (512 ch), Crop+Concat, 2× Conv3×3 (512 ch), ReLU | ✓ Same (+ BatchNorm) | ✅ |
| Level 2 | UpConv 2×2 (256 ch), Crop+Concat, 2× Conv3×3 (256 ch), ReLU | ✓ Same (+ BatchNorm) | ✅ |
| Level 1 | UpConv 2×2 (128 ch), Crop+Concat, 2× Conv3×3 (128 ch), ReLU | ✓ Same (+ BatchNorm) | ✅ |
| Level 0 | UpConv 2×2 (128→64 ch*), Concat, 2× Conv3×3 (64 ch), ReLU | ✓ Same (+ BatchNorm) | ✅ |
| **Final** | 1×1 Conv → 2 classes | ✓ Same | ✅ |
| **Loss** | Softmax + weighted cross-entropy (Eq. 2) | ✓ Same | ✅ |
| **Padding** | Valid (unpadded) convolutions | Same-padding (padding=1) | ⚠️ |
| **Batch Norm** | Not used (2015) | Added (modern best practice) | ⚠️ |
| **Weight Init** | Gaussian, std=√(2/N) | Kaiming He normal (equivalent) | ✅ |
| **Optimizer** | SGD, momentum=0.99 | ✓ Same | ✅ |

*Note: The Caffe prototxt upconvs in level 0 output 128 channels (slight deviation from
Figure 1 which shows 64).  Our implementation follows Figure 1 (64 channels).*

## Intentional Differences

1. **Same-padding vs Valid-padding**: We use `padding=1` in all 3×3 convolutions so
   that the output spatial size equals the input.  The original paper uses unpadded
   convolutions which gradually shrink the feature maps, requiring an overlap-tile
   strategy at inference.  Same-padding is the universally adopted modern approach and
   does not fundamentally alter the architecture.

2. **Batch Normalisation**: Added after each convolution (before ReLU).  This was not
   available / common in 2015 but is standard practice and improves training stability.
   It does not change the fundamental design.
"""
    path = os.path.join(out_dir, "architecture_comparison.md")
    with open(path, "w") as f:
        f.write(table)
    print(f"[EVAL] Architecture comparison saved to {path}")


# ═══════════════════════════════════════════════════════════════════════
# Paper comparison (dataset-aware)
# ═══════════════════════════════════════════════════════════════════════

def _paper_comparison(our_metrics: dict, out_dir: str, dataset_type: str = "isbi2012",
                      seg_result: dict = None, isbi_result: dict = None):
    """
    Generate a comparison summary between our results and the paper's
    reported performance.  Dataset-aware: shows relevant paper results
    for ISBI 2012, PhC-C2DH-U373, or DIC-C2DH-HeLa.
    """
    ds_name = DATASET_NAMES.get(dataset_type, dataset_type)

    # Common header
    comparison = f"""# Results Comparison: Our Implementation vs Paper

## Dataset: {ds_name}

### Our Validation Results

| Metric | Our U-Net (PyTorch) |
|---|---|
| Dice Coefficient | {our_metrics.get('dice', 0):.4f} |
| IoU (Jaccard) | {our_metrics.get('iou', 0):.4f} |
| Pixel Accuracy | {our_metrics.get('pixel_acc', 0):.4f} |
| Precision | {our_metrics.get('precision', 0):.4f} |
| Recall | {our_metrics.get('recall', 0):.4f} |

"""

    if dataset_type == "isbi2012":
        comparison += """### Paper's Reported Results (ISBI 2012 Leaderboard, Table 1)

| Metric | Paper's U-Net | Previous Best |
|---|---|---|
| Warping Error | **0.000353** | 0.000420 |
| Rand Error | **0.0382** | 0.0611 |
| Pixel Error | **0.0611** | 0.0639 |

"""
        if isbi_result is not None:
            pe = isbi_result['pixel_error']
            re = isbi_result['rand_error']
            comparison += f"""### Our ISBI Challenge Metrics (on validation split)

| Metric | Our Score | Paper's Score |
|---|---|---|
| Pixel Error | {pe['mean']:.4f} ± {pe['std']:.4f} | 0.0611 |
| Rand Error | {re['mean']:.4f} ± {re['std']:.4f} | 0.0382 |

"""

        comparison += """### Notes
- The paper evaluates on the held-out ISBI test set (not publicly labelled).
  We evaluate on a random validation split from the training images.
- Our Pixel Error and Rand Error are computed on the same metrics as the
  paper, enabling a more direct comparison than Dice/IoU alone.
- Warping Error requires specialized topology-aware code not included here.
"""
    elif dataset_type == "phc":
        comparison += """### Paper's Reported Results (ISBI Cell Tracking Challenge 2015, Table 2)

| Metric | Paper's U-Net | Second Best |
|---|---|---|
| SEG score (PhC-C2DH-U373) | **0.9203** | 0.2227 |

The U-Net **won by a very large margin** on this dataset.  The SEG score
measures segmentation accuracy at the instance level.

### Notes
- The paper's SEG score is an instance-level metric from the Cell Tracking
  Challenge, not directly comparable to our binary Dice/IoU.
- We convert instance masks to binary (foreground/background) for training,
  which is a simplification compared to the paper's full instance approach.
- Our binary Dice captures the overall cell vs background segmentation quality.
"""
    elif dataset_type == "dic":
        comparison += """### Paper's Reported Results (ISBI Cell Tracking Challenge 2015, Table 2)

| Metric | Paper's U-Net | Second Best |
|---|---|---|
| SEG score (DIC-C2DH-HeLa) | **0.7756** | 0.4583 |

The U-Net **won by a large margin** on this dataset.  DIC microscopy is
particularly challenging due to the halo effect around cell boundaries.

### Notes
- The paper's SEG score is an instance-level metric from the Cell Tracking
  Challenge, not directly comparable to our binary Dice/IoU.
- DIC images have less contrast than phase-contrast, making segmentation harder.
- We convert instance masks to binary for training, which is a simplification.
"""

    if seg_result is not None:
        seg_mean = seg_result['seg_score']['mean']
        seg_std = seg_result['seg_score']['std']
        n_inst = seg_result['n_gt_instances']
        n_match = seg_result['n_matched']
        comparison += f"""
### Our SEG Score (Instance-Level Jaccard)

| Metric | Score |
|---|---|
| **SEG Score** | **{seg_mean:.4f}** ± {seg_std:.4f} |
| GT Instances Evaluated | {n_inst} |
| Instances Matched | {n_match} / {n_inst} |

Computed using the official CTC methodology: binary predictions are
converted to instance labels via connected-component labelling, then
each GT instance is matched to a predicted instance (>50% area overlap)
and the Jaccard index is computed per pair.
"""

    comparison += """
### General Notes on Comparison

1. **Faithful reproduction**: Our architecture, training procedure (SGD
   momentum=0.99, weighted loss, elastic deformation) closely follow the paper.
   The main differences are same-padding and batch normalisation (see
   architecture_comparison.md).

2. **Binary vs Instance**: The paper performs instance-aware segmentation
   for the cell tracking datasets.  Our implementation does binary
   (cell vs background) segmentation, which is simpler but still demonstrates
   the core U-Net capabilities.
"""
    path = os.path.join(out_dir, "results_comparison.md")
    with open(path, "w") as f:
        f.write(comparison)
    print(f"[EVAL] Results comparison saved to {path}")

    # Also generate a visual comparison figure
    chart_metrics = dict(our_metrics)
    if seg_result is not None:
        chart_metrics['SEG'] = seg_result['seg_score']['mean']
    fig, ax = plt.subplots(figsize=(8, 4))
    metrics = list(chart_metrics.keys())
    values = [chart_metrics[k] for k in metrics]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(metrics)))

    bars = ax.bar(metrics, values, color=colors, edgecolor='white', linewidth=1.2)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Our U-Net — {ds_name}")
    ax.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.4f}", ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    path = os.path.join(out_dir, "metrics_bar_chart.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Metrics chart saved to {path}")


# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    cfg = load_config()
    evaluate(cfg)
