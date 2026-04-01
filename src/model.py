"""
U-Net Architecture — faithful PyTorch implementation.

Reference:
  Ronneberger, Fischer, Brox.  "U-Net: Convolutional Networks for
  Biomedical Image Segmentation" (2015).  arXiv:1505.04597

Architecture verified against the official Caffe prototxt
(phseg_v5-train.prototxt).

Key design notes:
  • Same-padding (padding=1) is used so that output spatial size equals
    input spatial size.  The original paper uses valid (unpadded) convolutions,
    which shrinks the feature map at every conv layer and requires an
    overlap-tile strategy at inference time.  Same-padding is the universally
    adopted modern adaptation and does not change the fundamental architecture.
  • Weight initialisation uses Kaiming / He normal, equivalent to the paper's
    Gaussian with std = sqrt(2/N).
  • Dropout (p=0.5) is applied in the two deepest encoder blocks (level 3 and
    bottleneck), exactly matching the Caffe prototxt.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    """Two consecutive (Conv3x3 → BN → ReLU) blocks.

    Batch Normalisation is added (not in the 2015 paper, but universally
    used in modern reproductions and explicitly *not* contradicted by it).
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """Encoder level: DoubleConv → MaxPool 2×2.

    Optionally applies dropout (used at the two deepest levels).
    """

    def __init__(self, in_channels: int, out_channels: int,
                 dropout: float = 0.0):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor):
        features = self.conv(x)
        features = self.drop(features)
        pooled = self.pool(features)
        return features, pooled          # features kept for skip connection


class DecoderBlock(nn.Module):
    """Decoder level: Up-conv 2×2 → Concat skip → DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # 2×2 up-convolution (transposed convolution) — matches paper exactly
        self.up = nn.ConvTranspose2d(in_channels, out_channels,
                                     kernel_size=2, stride=2)
        # After concat: out_channels (from up) + out_channels (from skip)
        self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # Centre-crop skip connection to match spatial dims of x
        # (only necessary when using valid padding, but kept for safety)
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        x = nn.functional.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                                   diff_h // 2, diff_h - diff_h // 2])

        x = torch.cat([skip, x], dim=1)  # Channel-wise concatenation
        return self.conv(x)


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """
    Full U-Net.

    Channels per level (paper/prototxt):
        Encoder:    64 → 128 → 256 → 512
        Bottleneck: 1024
        Decoder:    512 → 256 → 128 → 64
        Final:      64 → n_classes  (1×1 conv)
    """

    def __init__(self, in_channels: int = 1, n_classes: int = 2):
        super().__init__()

        # ---- Encoder (contracting path) ----
        self.enc1 = EncoderBlock(in_channels,  64)
        self.enc2 = EncoderBlock(64,          128)
        self.enc3 = EncoderBlock(128,         256)
        self.enc4 = EncoderBlock(256,         512, dropout=0.5)   # dropout per prototxt

        # ---- Bottleneck ----
        self.bottleneck = nn.Sequential(
            DoubleConv(512, 1024),
            nn.Dropout2d(p=0.5),          # dropout per prototxt
        )

        # ---- Decoder (expanding path) ----
        self.dec4 = DecoderBlock(1024, 512)
        self.dec3 = DecoderBlock(512,  256)
        self.dec2 = DecoderBlock(256,  128)
        self.dec1 = DecoderBlock(128,   64)

        # ---- Final 1×1 convolution ----
        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)

        # ---- Weight initialisation (paper: Gaussian, std = sqrt(2/N)) ----
        self._init_weights()

    # -----------------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # -----------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        skip1, x = self.enc1(x)    # 512 → 256  (channels: 64)
        skip2, x = self.enc2(x)    # 256 → 128  (channels: 128)
        skip3, x = self.enc3(x)    # 128 → 64   (channels: 256)
        skip4, x = self.enc4(x)    # 64  → 32   (channels: 512)

        # Bottleneck
        x = self.bottleneck(x)     # 32 × 32    (channels: 1024)

        # Decoder
        x = self.dec4(x, skip4)    # 32  → 64   (channels: 512)
        x = self.dec3(x, skip3)    # 64  → 128  (channels: 256)
        x = self.dec2(x, skip2)    # 128 → 256  (channels: 128)
        x = self.dec1(x, skip1)    # 256 → 512  (channels: 64)

        # Final 1×1 conv
        return self.final_conv(x)  # (N, n_classes, H, W)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = UNet(in_channels=1, n_classes=2)
    x = torch.randn(1, 1, 512, 512)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
