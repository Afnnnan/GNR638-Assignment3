# U-Net: Convolutional Networks for Biomedical Image Segmentation

PyTorch implementation of [U-Net](https://arxiv.org/abs/1505.04597) (Ronneberger et al., 2015) from scratch, trained and evaluated on all three datasets from the paper.

> **GNR 638 — Assignment 3**: Implement a paper from scratch and compare against the official implementation.

## Results

| Dataset | Dice | IoU | Pixel Acc | Precision | Recall |
|---|---|---|---|---|---|
| ISBI 2012 EM Segmentation | 0.9449 ± 0.0072 | 0.8957 ± 0.0130 | 0.9162 ± 0.0094 | 0.9599 ± 0.0093 | 0.9305 ± 0.0124 |
| PhC-C2DH-U373 | — | — | — | — | — |
| DIC-C2DH-HeLa | — | — | — | — | — |

*(PhC and DIC results to be filled after training on Kaggle.)*

## Project Structure

```
GNR638-Assignment3/
├── run_all.py              # Single entry point — train & evaluate all datasets
├── requirements.txt        # Python dependencies
├── src/
│   ├── model.py            # U-Net architecture (31M params, from scratch)
│   ├── dataset.py          # Data loading, augmentation, weight maps
│   ├── train.py            # Training loop (SGD, momentum=0.99)
│   ├── evaluate.py         # Metrics, visualisations, paper comparison
│   └── utils.py            # Device detection, seeding, helpers
├── data/                   # Datasets (see below)
│   ├── ISBI-2012-Challenge/
│   ├── PhC-C2DH-U373_train/
│   ├── PhC-C2DH-U373_test/
│   ├── DIC-C2DH-HeLa_train/
│   └── DIC-C2DH-HeLa_test/
├── outputs/                # Auto-generated: checkpoints, plots, metrics
│   ├── isbi2012/
│   ├── phc/
│   ├── dic/
│   └── combined_results.md
├── u-net-release/          # Official Caffe implementation (from paper website)
├── 1505.04597v1.pdf        # The paper
└── UnetBlog.pdf            # Blog writeup
```

## Dataset Preparation

Download the following datasets and place them in a `data/` folder:

### 1. ISBI 2012 EM Segmentation Challenge

- **Source**: https://www.kaggle.com/datasets/soumikrakshit/isbi-challenge-dataset
- Download and rename the folder to `ISBI-2012-Challenge`
- Contains: `train-volume.tif` (30 images, 512×512), `train-labels.tif`, `test-volume.tif`

### 2. PhC-C2DH-U373 (Phase Contrast Microscopy)

- **Training**: https://data.celltrackingchallenge.net/training-datasets/PhC-C2DH-U373.zip → rename to `PhC-C2DH-U373_train`
- **Test**: https://data.celltrackingchallenge.net/test-datasets/PhC-C2DH-U373.zip → rename to `PhC-C2DH-U373_test`
- Contains: 2 sequences × 115 frames (520×696), instance segmentation masks

### 3. DIC-C2DH-HeLa (DIC Microscopy)

- **Training**: https://data.celltrackingchallenge.net/training-datasets/DIC-C2DH-HeLa.zip → rename to `DIC-C2DH-HeLa_train`
- **Test**: https://data.celltrackingchallenge.net/test-datasets/DIC-C2DH-HeLa.zip → rename to `DIC-C2DH-HeLa_test`
- Contains: 2 sequences × 84 frames (512×512), instance segmentation masks

### Final folder structure

```
data/
├── ISBI-2012-Challenge/
│   ├── train-volume.tif
│   ├── train-labels.tif
│   └── test-volume.tif
├── PhC-C2DH-U373_train/
│   ├── 01/              # Raw images (t000.tif, t001.tif, ...)
│   ├── 01_ERR_SEG/      # Dense segmentation masks
│   ├── 01_GT/           # Ground truth (sparse)
│   ├── 01_ST/           # Silver truth
│   ├── 02/
│   ├── 02_ERR_SEG/
│   ├── 02_GT/
│   └── 02_ST/
├── PhC-C2DH-U373_test/
│   ├── 01/
│   └── 02/
├── DIC-C2DH-HeLa_train/
│   ├── 01/
│   ├── 01_ERR_SEG/
│   ├── 01_GT/
│   ├── 01_ST/
│   ├── 02/
│   ├── 02_ERR_SEG/
│   ├── 02_GT/
│   └── 02_ST/
└── DIC-C2DH-HeLa_test/
    ├── 01/
    └── 02/
```

## How to Run

### Local (Mac / Linux)

```bash
# Install dependencies
pip install -r requirements.txt

# Set dataset path
export DATA_ROOT="./data"

# Run all 3 datasets (train + evaluate)
python run_all.py

# Or run specific datasets
python run_all.py --datasets isbi2012
python run_all.py --datasets phc dic

# Evaluate existing checkpoints only
python run_all.py --eval-only

# Quick sanity check
python run_all.py --smoke-test
```

### Kaggle

```python
# Clone the repo
!git clone https://github.com/Afnnnan/GNR638-Assignment3.git
%cd GNR638-Assignment3
!pip install tifffile imagecodecs pyyaml scipy -q

# Set dataset path (adjust to your Kaggle dataset name)
import os
os.environ["DATA_ROOT"] = "/kaggle/input/YOUR_DATASET_NAME"  # adjust to your Kaggle dataset

# Run all datasets
!python run_all.py
```

### Google Colab

```python
# Mount Drive (if data is on Drive)
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/Afnnnan/GNR638-Assignment3.git
%cd GNR638-Assignment3
!pip install tifffile imagecodecs pyyaml scipy -q

import os
os.environ["DATA_ROOT"] = "/content/drive/MyDrive/path/to/data"

!python run_all.py
```

## Architecture

Faithful reproduction of the U-Net from the paper, verified against the official Caffe prototxt (`u-net-release/phseg_v5-train.prototxt`).

| Component | Detail |
|---|---|
| Encoder | 4 blocks: 2×(Conv3×3 → BN → ReLU) + MaxPool 2×2. Channels: 64→128→256→512 |
| Bottleneck | 2×(Conv3×3 → BN → ReLU), 1024 channels, Dropout 0.5 |
| Decoder | 4 blocks: TransposedConv 2×2 → Concat skip → 2×(Conv3×3 → BN → ReLU). Channels: 512→256→128→64 |
| Final | 1×1 Conv → n_classes |
| Parameters | 31,036,546 |
| Weight Init | Kaiming He normal (≡ paper's Gaussian with std=√(2/N)) |

### Intentional modern adaptations

1. **Same-padding** (`padding=1`) instead of valid convolutions — avoids overlap-tile inference
2. **Batch Normalisation** — not available in 2015, now standard practice

### Training (paper-faithful)

| Setting | Value |
|---|---|
| Optimizer | SGD, momentum=0.99 |
| Learning rate | 0.01 → ReduceLROnPlateau (factor=0.5, patience=15) |
| Loss | Weighted cross-entropy with border-emphasis weight map (Eq. 2) |
| Augmentation | Elastic deformation, random flips, 90° rotations, intensity jitter |
| Epochs | 200 |

## Outputs

After running, each dataset gets its own output folder:

```
outputs/
├── isbi2012/
│   ├── unet_isbi2012.pth           # Best model checkpoint
│   ├── predictions.png             # Input | Ground Truth | Prediction
│   ├── prediction_overlays.png     # Predictions overlaid on inputs
│   ├── metrics_bar_chart.png       # Dice, IoU, Acc bar chart
│   ├── evaluation_metrics.json     # All metrics (per-image + aggregate)
│   ├── architecture_comparison.md  # Our PyTorch vs Caffe prototxt
│   ├── results_comparison.md       # Our results vs paper's reported results
│   └── training_history.json       # Loss/Dice per epoch
├── phc/
│   └── ... (same structure)
├── dic/
│   └── ... (same structure)
└── combined_results.md             # Summary table across all datasets
```

## Comparison with Official Implementation

The official implementation (`u-net-release/`) is in Caffe and was designed for Ubuntu 14.04 + MATLAB 2014b. It includes a pre-trained model for PhC-C2DH-U373 cell segmentation.

Our comparison approach:
1. **Architecture**: Structural comparison verified against the Caffe prototxt (see `outputs/*/architecture_comparison.md`)
2. **Results**: Our metrics compared against the paper's reported scores (see `outputs/*/results_comparison.md`)
3. **Visual**: Side-by-side prediction visualizations (see `outputs/*/predictions.png`)

## Dependencies

- Python 3.8+
- PyTorch (with MPS/CUDA support)
- tifffile, imagecodecs, scipy, numpy, matplotlib, Pillow, pyyaml, scikit-learn

Device auto-detection: CUDA → MPS → CPU (no code changes needed).

## Reference

```
@inproceedings{ronneberger2015unet,
  title={U-Net: Convolutional Networks for Biomedical Image Segmentation},
  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  booktitle={MICCAI},
  year={2015}
}
```
