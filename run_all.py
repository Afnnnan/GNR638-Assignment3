#!/usr/bin/env python3
"""
run_all.py — Train & evaluate U-Net on all three paper datasets.

Usage (local):
    export DATA_ROOT="./data"
    python run_all.py

Usage (Kaggle):
    import os
    os.environ["DATA_ROOT"] = "/kaggle/input/YOUR_DATASET"
    !python run_all.py

Flags:
    python run_all.py --eval-only             # Skip training
    python run_all.py --smoke-test            # Quick forward-pass check
    python run_all.py --datasets isbi2012     # Run only one dataset
    python run_all.py --datasets phc dic      # Run a subset
"""

import argparse
import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import load_config, set_seed, get_device, ensure_dir


# ═══════════════════════════════════════════════════════════════════════
# All configs — hardcoded, no YAML files needed
# ═══════════════════════════════════════════════════════════════════════

DATASET_NAMES = {
    "isbi2012": "ISBI 2012 EM Segmentation",
    "phc":      "PhC-C2DH-U373 (Phase Contrast)",
    "dic":      "DIC-C2DH-HeLa (DIC Microscopy)",
}

# Subdirectory under DATA_ROOT where each dataset lives
DATASET_SUBDIRS = {
    "isbi2012": "ISBI-2012-Challenge",
    "phc":      "PhC-C2DH-U373_train",
    "dic":      "DIC-C2DH-HeLa_train",
}


def make_config(dataset_type: str, data_root: str) -> dict:
    """Build a config dict for the given dataset. No YAML files needed."""
    data_dir = os.path.join(data_root, DATASET_SUBDIRS[dataset_type])

    # Base config (shared across all datasets — follows the paper)
    cfg = dict(
        # Data
        dataset_type=dataset_type,
        data_dir=data_dir,
        image_size=512,
        val_split=0.2,
        n_classes=2,

        # Training  (paper: SGD, momentum=0.99)
        batch_size=2 if dataset_type == "isbi2012" else 4,
        epochs=200,
        lr=0.01,
        momentum=0.99,
        weight_decay=0.0005,

        # Augmentation  (paper Section 3.1)
        use_elastic_deform=True,
        elastic_alpha=10,
        elastic_sigma=3,

        # Weight map  (paper Eq. 2)
        use_weight_map=True,
        w0=10,
        sigma=5,

        # Misc
        seed=42,
        output_dir=f"./outputs/{dataset_type}",
        checkpoint_name=f"unet_{dataset_type}.pth",
    )

    # Cell tracking datasets have more images → can use bigger batch
    if dataset_type in ("phc", "dic"):
        cfg["mask_subdir"] = "ERR_SEG"
        cfg["sequences"] = ["01", "02"]

    return cfg


# ═══════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    """Quick forward pass to verify the model works."""
    import torch
    from src.model import UNet

    device = get_device()
    model = UNet(in_channels=1, n_classes=2).to(device)
    x = torch.randn(1, 1, 512, 512).to(device)
    with torch.no_grad():
        out = model(x)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n✅  Smoke test passed!")
    print(f"    Device:     {device}")
    print(f"    Input:      {x.shape}")
    print(f"    Output:     {out.shape}")
    print(f"    Parameters: {n_params:,}\n")


# ═══════════════════════════════════════════════════════════════════════
# Run one dataset
# ═══════════════════════════════════════════════════════════════════════

def run_dataset(ds_type: str, data_root: str, eval_only: bool = False):
    """Train and/or evaluate on a single dataset."""
    ds_name = DATASET_NAMES[ds_type]
    cfg = make_config(ds_type, data_root)

    print("\n" + "█" * 70)
    print(f"  {ds_name}")
    print(f"  Data: {cfg['data_dir']}")
    print("█" * 70 + "\n")

    if not os.path.isdir(cfg["data_dir"]):
        print(f"  ❌  Data directory not found: {cfg['data_dir']}")
        print(f"      Check DATA_ROOT={data_root}")
        return None

    ensure_dir(cfg["output_dir"])
    t_start = time.time()

    if not eval_only:
        from src.train import train
        ckpt_path, history = train(cfg)
    else:
        ckpt_path = os.path.join(cfg["output_dir"], cfg["checkpoint_name"])
        if not os.path.exists(ckpt_path):
            print(f"  ⚠  No checkpoint at {ckpt_path}, skipping.")
            return None

    from src.evaluate import evaluate
    results = evaluate(cfg, checkpoint_path=ckpt_path)

    elapsed = time.time() - t_start
    dice = results["aggregate"]["dice"]["mean"]
    print(f"\n  ✅ {ds_name} — Dice={dice:.4f} ({elapsed/60:.1f} min)\n")
    return results


# ═══════════════════════════════════════════════════════════════════════
# Combined summary
# ═══════════════════════════════════════════════════════════════════════

def generate_summary(all_results: dict):
    """Create combined results table across all datasets."""
    ensure_dir("./outputs")

    paper_results = {
        "isbi2012": ("Pixel Error", "0.0611"),
        "phc":      ("SEG score",   "0.9203"),
        "dic":      ("SEG score",   "0.7756"),
    }

    lines = ["# U-Net Results — All Datasets\n"]
    lines.append("| Dataset | Dice | IoU | Pixel Acc | Precision | Recall |")
    lines.append("|---|---|---|---|---|---|")

    for ds, res in all_results.items():
        if res is None:
            continue
        a = res["aggregate"]
        lines.append(
            f"| {DATASET_NAMES[ds]} | "
            f"{a['dice']['mean']:.4f} ± {a['dice']['std']:.4f} | "
            f"{a['iou']['mean']:.4f} ± {a['iou']['std']:.4f} | "
            f"{a['pixel_acc']['mean']:.4f} ± {a['pixel_acc']['std']:.4f} | "
            f"{a['precision']['mean']:.4f} ± {a['precision']['std']:.4f} | "
            f"{a['recall']['mean']:.4f} ± {a['recall']['std']:.4f} |"
        )

    lines.append("\n\n## Paper Comparison\n")
    lines.append("| Dataset | Paper Metric | Paper Score | Our Dice |")
    lines.append("|---|---|---|---|")
    for ds, res in all_results.items():
        if res is None:
            continue
        m, s = paper_results[ds]
        lines.append(
            f"| {DATASET_NAMES[ds]} | {m} | {s} | "
            f"{res['aggregate']['dice']['mean']:.4f} |"
        )

    path = "./outputs/combined_results.md"
    with open(path, "w") as f:
        f.write("\n".join(lines))

    # Also JSON
    json_data = {ds: res["aggregate"] for ds, res in all_results.items()
                 if res is not None}
    with open("./outputs/combined_results.json", "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"\n[SUMMARY] Combined results → {path}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="U-Net — All Datasets")
    parser.add_argument("--datasets", nargs="+",
                        choices=["isbi2012", "phc", "dic"],
                        default=["isbi2012", "phc", "dic"],
                        help="Which datasets to run (default: all)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, evaluate existing checkpoints")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Quick forward-pass sanity check")
    args = parser.parse_args()

    # ---- DATA_ROOT ----
    data_root = os.environ.get("DATA_ROOT", "./data")
    print(f"DATA_ROOT = {data_root}")

    if args.smoke_test:
        smoke_test()
        return

    all_results = {}
    total_start = time.time()

    for ds in args.datasets:
        try:
            results = run_dataset(ds, data_root, eval_only=args.eval_only)
            all_results[ds] = results
        except Exception as e:
            print(f"\n❌ {DATASET_NAMES[ds]}: {e}")
            import traceback
            traceback.print_exc()
            all_results[ds] = None

    if any(v is not None for v in all_results.values()):
        generate_summary(all_results)

    total = time.time() - total_start
    print(f"{'='*70}")
    print(f"  Done!  Total: {total/60:.1f} min")
    print(f"  Outputs: ./outputs/")
    print(f"{'='*70}\n")

    # ---- Zip outputs for Kaggle download ----
    if os.path.isdir("/kaggle/working") and os.path.isdir("./outputs"):
        import shutil
        shutil.make_archive("/kaggle/working/outputs", "zip", ".", "outputs")
        print("✅ Outputs zipped → /kaggle/working/outputs.zip")


if __name__ == "__main__":
    main()
