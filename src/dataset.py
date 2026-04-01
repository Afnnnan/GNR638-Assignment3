"""
Dataset and augmentation for the ISBI 2012 EM Segmentation Challenge
AND the Cell Tracking Challenge datasets (PhC-C2DH-U373, DIC-C2DH-HeLa).

Supports three dataset formats:
  1. "isbi2012"  — Multi-page TIFF stacks (train-volume.tif, train-labels.tif)
  2. "phc"       — PhC-C2DH-U373 cell tracking (individual TIF files, instance masks)
  3. "dic"       — DIC-C2DH-HeLa cell tracking (individual TIF files, instance masks)

Implements:
  • Multi-page TIFF loading (ISBI) & individual TIFF loading (cell tracking)
  • Instance-mask → binary-mask conversion (cell tracking datasets)
  • Train / val splitting
  • Paper-faithful augmentation:
      – Random elastic deformation  (Section 3.1)
      – Random flips, 90° rotations
      – Intensity jitter
  • Weight-map generation (Eq. 2 of the paper)
"""

import os
import glob
import numpy as np
from PIL import Image
from scipy.ndimage import (
    map_coordinates,
    gaussian_filter,
    distance_transform_edt,
    label as ndlabel,
)

import torch
from torch.utils.data import Dataset


# ═══════════════════════════════════════════════════════════════════════
# TIFF I/O
# ═══════════════════════════════════════════════════════════════════════

def load_tiff_stack(path: str) -> np.ndarray:
    """Load a multi-page TIFF and return (N, H, W) uint8 array."""
    img = Image.open(path)
    frames = []
    for i in range(img.n_frames):
        img.seek(i)
        frames.append(np.array(img))
    return np.stack(frames)


def load_individual_tiffs(folder: str) -> np.ndarray:
    """Load individual TIFF files from a folder, sorted by name.
    Returns (N, H, W) array."""
    patterns = ["*.tif", "*.tiff"]
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(folder, pat)))
    files = sorted(files)
    if len(files) == 0:
        raise FileNotFoundError(f"No .tif files found in {folder}")
    frames = [np.array(Image.open(f)) for f in files]
    return np.stack(frames)


def load_cell_tracking_dataset(data_dir: str, sequences=("01", "02"),
                               mask_subdir="ERR_SEG"):
    """
    Load a Cell Tracking Challenge dataset.

    Args:
        data_dir:    Root of the dataset (e.g., data/PhC-C2DH-U373_train)
        sequences:   Which sequence folders to load (usually "01" and "02")
        mask_subdir: Where to find masks. Options:
                     "ERR_SEG"  — dense masks for every frame (recommended)
                     "ST/SEG"   — silver-truth segmentation
                     "GT/SEG"   — ground truth (sparse, not every frame)

    Returns:
        images: (N, H, W) uint8 array
        masks:  (N, H, W) uint8 binary array (0=bg, 255=fg)
    """
    all_images = []
    all_masks = []

    for seq in sequences:
        img_dir = os.path.join(data_dir, seq)
        if mask_subdir == "ERR_SEG":
            mask_dir = os.path.join(data_dir, f"{seq}_{mask_subdir}")
        elif mask_subdir in ("ST/SEG", "GT/SEG"):
            mask_dir = os.path.join(data_dir, f"{seq}_{mask_subdir.split('/')[0]}", "SEG")
        else:
            mask_dir = os.path.join(data_dir, f"{seq}_{mask_subdir}")

        if not os.path.isdir(img_dir):
            print(f"[WARN] Sequence folder not found: {img_dir}, skipping")
            continue
        if not os.path.isdir(mask_dir):
            print(f"[WARN] Mask folder not found: {mask_dir}, skipping")
            continue

        # Get image files
        img_files = sorted(glob.glob(os.path.join(img_dir, "*.tif")))

        # Get mask files and build an index by frame number
        mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.tif")))
        mask_by_num = {}
        for mf in mask_files:
            basename = os.path.splitext(os.path.basename(mf))[0]
            # Extract trailing digits from names like "mask000", "man_seg001"
            num_str = ''.join(c for c in basename if c.isdigit())
            if num_str:
                mask_by_num[int(num_str)] = mf

        # Match images to masks
        for if_path in img_files:
            basename = os.path.splitext(os.path.basename(if_path))[0]
            num_str = ''.join(c for c in basename if c.isdigit())
            if not num_str:
                continue
            frame_num = int(num_str)
            if frame_num in mask_by_num:
                img = np.array(Image.open(if_path))
                mask = np.array(Image.open(mask_by_num[frame_num]))

                # Convert instance mask to binary: any cell > 0 becomes 255
                mask_binary = np.where(mask > 0, 255, 0).astype(np.uint8)

                all_images.append(img)
                all_masks.append(mask_binary)

    if len(all_images) == 0:
        raise ValueError(f"No image-mask pairs found in {data_dir}")

    images = np.stack(all_images)
    masks = np.stack(all_masks)

    print(f"[DATA] Loaded {len(images)} image-mask pairs from {data_dir}")
    return images, masks


# ═══════════════════════════════════════════════════════════════════════
# Weight map  (Eq. 2 of the paper)
# ═══════════════════════════════════════════════════════════════════════

def compute_weight_map(mask: np.ndarray, w0: float = 10.0,
                       sigma: float = 5.0) -> np.ndarray:
    """
    Compute the pixel-wise weight map described in Section 3.1 of U-Net.

    w(x) = w_c(x) + w0 · exp( -(d1(x) + d2(x))^2 / (2·σ^2) )

    where d1 and d2 are the distances to the border of the nearest and
    second-nearest cell, and w_c is the class-balancing weight.

    Args:
        mask:  (H, W) binary mask  (0 = background, 255 = cell/foreground)
        w0:    weight-map hyper-parameter (paper: 10)
        sigma: weight-map hyper-parameter (paper: 5 px)
    Returns:
        (H, W) float32 weight map.
    """
    # Binarise: cell interior = 1, bg = 0
    binary = (mask > 127).astype(np.uint8)

    # Label connected components (each cell gets a unique integer)
    labelled, n_cells = ndlabel(binary)

    # Class-frequency balancing weight
    n_bg = np.sum(binary == 0)
    n_fg = np.sum(binary == 1)
    total = binary.size
    w_c = np.where(binary == 0, total / (2.0 * n_bg + 1e-8),
                                total / (2.0 * n_fg + 1e-8)).astype(np.float32)

    if n_cells <= 1:
        # No touching-cell boundaries to emphasise
        return w_c

    # Distance transform for each cell
    distances = np.full((n_cells, *mask.shape), np.inf, dtype=np.float32)
    for i in range(1, n_cells + 1):
        cell_mask = (labelled == i)
        distances[i - 1] = distance_transform_edt(~cell_mask)

    # Sort distances along the cell axis and pick d1, d2
    distances.sort(axis=0)
    d1 = distances[0]
    d2 = distances[1] if n_cells >= 2 else distances[0]

    # Border emphasis term
    border_weight = w0 * np.exp(-((d1 + d2) ** 2) / (2 * sigma ** 2))

    weight_map = w_c + border_weight.astype(np.float32)
    return weight_map


# ═══════════════════════════════════════════════════════════════════════
# Elastic deformation  (Section 3.1)
# ═══════════════════════════════════════════════════════════════════════

def elastic_deformation(image: np.ndarray, mask: np.ndarray,
                        alpha: float = 10.0, sigma: float = 3.0,
                        random_state: np.random.RandomState = None):
    """
    Apply random elastic deformation to image and mask jointly.

    The paper describes generating random displacement vectors on a coarse
    3×3 grid and then upsampling with bicubic interpolation to full
    resolution.  We follow the standard implementation: generate full-res
    displacements and smooth with a large Gaussian kernel.
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    h, w = image.shape[:2]

    real_sigma = sigma * min(h, w) / 10.0
    real_alpha = alpha * min(h, w) / 10.0

    dx = gaussian_filter(
        random_state.rand(h, w) * 2 - 1, real_sigma) * real_alpha
    dy = gaussian_filter(
        random_state.rand(h, w) * 2 - 1, real_sigma) * real_alpha

    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    indices = [np.clip(y + dy, 0, h - 1), np.clip(x + dx, 0, w - 1)]

    deformed_image = map_coordinates(
        image, indices, order=1, mode='reflect').reshape(h, w)
    deformed_mask = map_coordinates(
        mask, indices, order=0, mode='reflect').reshape(h, w)

    return deformed_image, deformed_mask


# ═══════════════════════════════════════════════════════════════════════
# PyTorch Dataset (unified for all three dataset types)
# ═══════════════════════════════════════════════════════════════════════

class SegmentationDataset(Dataset):
    """
    Unified segmentation dataset for ISBI 2012, PhC-C2DH-U373, and
    DIC-C2DH-HeLa.

    Args:
        images:         (N, H, W) uint8 array of grayscale images.
        masks:          (N, H, W) uint8 array of binary labels (0 or 255).
        image_size:     Target spatial size (square).
        augment:        Enable augmentation (training only).
        use_elastic:    Enable elastic deformation.
        elastic_alpha:  Intensity of elastic deformation.
        elastic_sigma:  Smoothness of elastic deformation.
        use_weight_map: Compute and return per-pixel weight maps.
        w0:             Weight-map parameter.
        sigma_wm:       Weight-map parameter.
    """

    def __init__(self, images: np.ndarray, masks: np.ndarray,
                 image_size: int = 512,
                 augment: bool = False,
                 use_elastic: bool = True,
                 elastic_alpha: float = 10.0,
                 elastic_sigma: float = 3.0,
                 use_weight_map: bool = True,
                 w0: float = 10.0,
                 sigma_wm: float = 5.0):
        super().__init__()
        self.images = images
        self.masks = masks
        self.image_size = image_size
        self.augment = augment
        self.use_elastic = use_elastic
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.use_weight_map = use_weight_map
        self.w0 = w0
        self.sigma_wm = sigma_wm

        # Pre-compute weight maps for the un-augmented masks
        if use_weight_map and not augment:
            self.weight_maps = []
            for m in masks:
                self.weight_maps.append(compute_weight_map(m, w0, sigma_wm))
        else:
            self.weight_maps = None

    def __len__(self):
        return len(self.images)

    def _resize(self, arr: np.ndarray, order: int = 1) -> np.ndarray:
        """Resize using PIL."""
        h, w = arr.shape
        if h == self.image_size and w == self.image_size:
            return arr
        interp = Image.BILINEAR if order == 1 else Image.NEAREST
        pil = Image.fromarray(arr)
        pil = pil.resize((self.image_size, self.image_size), interp)
        return np.array(pil)

    def __getitem__(self, idx: int):
        image = self.images[idx].astype(np.float64)
        mask = self.masks[idx].copy()

        # ---- Augmentation (training) ----
        if self.augment:
            rng = np.random.RandomState()

            # Elastic deformation
            if self.use_elastic and rng.rand() < 0.5:
                image, mask = elastic_deformation(
                    image, mask,
                    alpha=self.elastic_alpha,
                    sigma=self.elastic_sigma,
                    random_state=rng)

            # Random flips
            if rng.rand() < 0.5:
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()
            if rng.rand() < 0.5:
                image = np.flipud(image).copy()
                mask = np.flipud(mask).copy()

            # Random 90° rotation
            k = rng.randint(0, 4)
            image = np.rot90(image, k).copy()
            mask = np.rot90(mask, k).copy()

            # Intensity jitter
            if rng.rand() < 0.5:
                image = image + rng.uniform(-20, 20)
            if rng.rand() < 0.5:
                factor = rng.uniform(0.8, 1.2)
                mean_val = image.mean()
                image = (image - mean_val) * factor + mean_val

            image = np.clip(image, 0, 255)

        # ---- Resize ----
        image = self._resize(image.astype(np.uint8), order=1).astype(np.float32)
        mask = self._resize(mask, order=0)

        # ---- Normalise image to [0, 1] ----
        image = image.astype(np.float32) / 255.0

        # ---- Binarise mask ----
        mask_binary = (mask > 127).astype(np.int64)

        # ---- Weight map ----
        if self.use_weight_map:
            if self.augment:
                wmap = compute_weight_map(mask, self.w0, self.sigma_wm)
                wmap = self._resize(wmap, order=1)
            else:
                wmap = self.weight_maps[idx]
                wmap = self._resize(wmap, order=1)
            wmap_tensor = torch.from_numpy(wmap.astype(np.float32))
        else:
            wmap_tensor = torch.ones(self.image_size, self.image_size,
                                     dtype=torch.float32)

        # ---- To tensors ----
        image_tensor = torch.from_numpy(image).unsqueeze(0)   # (1, H, W)
        mask_tensor = torch.from_numpy(mask_binary)            # (H, W)

        return image_tensor, mask_tensor, wmap_tensor


# Keep legacy alias for backward compat
ISBIDataset = SegmentationDataset


# ═══════════════════════════════════════════════════════════════════════
# Unified data loading
# ═══════════════════════════════════════════════════════════════════════

def _load_dataset_by_type(cfg: dict):
    """
    Load images and masks based on dataset_type in config.

    Returns:
        images: (N, H, W) uint8
        masks:  (N, H, W) uint8 binary (0 or 255)
    """
    dataset_type = cfg.get("dataset_type", "isbi2012")
    data_dir = cfg["data_dir"]

    if dataset_type == "isbi2012":
        images = load_tiff_stack(os.path.join(data_dir, "train-volume.tif"))
        masks = load_tiff_stack(os.path.join(data_dir, "train-labels.tif"))

    elif dataset_type in ("phc", "dic"):
        mask_sub = cfg.get("mask_subdir", "ERR_SEG")
        sequences = cfg.get("sequences", ["01", "02"])
        images, masks = load_cell_tracking_dataset(
            data_dir, sequences=sequences, mask_subdir=mask_sub)

    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. "
                         f"Use 'isbi2012', 'phc', or 'dic'.")

    return images, masks


def build_dataloaders(cfg: dict):
    """
    Build train and validation DataLoaders from the config dict.
    Works for all three dataset types.
    Returns (train_loader, val_loader).
    """
    from torch.utils.data import DataLoader

    images, masks = _load_dataset_by_type(cfg)

    print(f"[DATA] Loaded {len(images)} images, shape {images.shape}")
    print(f"[DATA] Mask unique values: {np.unique(masks)}")

    # Split
    n = len(images)
    n_val = max(1, int(n * cfg.get("val_split", 0.2)))
    n_train = n - n_val

    # Deterministic split
    idx = np.arange(n)
    np.random.seed(cfg.get("seed", 42))
    np.random.shuffle(idx)

    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    print(f"[DATA] Train: {len(train_idx)} images, Val: {len(val_idx)} images")

    common = dict(
        image_size=cfg.get("image_size", 512),
        use_elastic=cfg.get("use_elastic_deform", True),
        elastic_alpha=cfg.get("elastic_alpha", 10.0),
        elastic_sigma=cfg.get("elastic_sigma", 3.0),
        use_weight_map=cfg.get("use_weight_map", True),
        w0=cfg.get("w0", 10.0),
        sigma_wm=cfg.get("sigma", 5.0),
    )

    train_ds = SegmentationDataset(images[train_idx], masks[train_idx],
                                   augment=True, **common)
    val_ds   = SegmentationDataset(images[val_idx], masks[val_idx],
                                   augment=False, **common)

    train_loader = DataLoader(train_ds,
                              batch_size=cfg.get("batch_size", 2),
                              shuffle=True, num_workers=0,
                              pin_memory=True)
    val_loader = DataLoader(val_ds,
                            batch_size=cfg.get("batch_size", 2),
                            shuffle=False, num_workers=0,
                            pin_memory=True)

    return train_loader, val_loader
