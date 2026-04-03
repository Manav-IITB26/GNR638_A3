"""
U-Net: Convolutional Networks for Biomedical Image Segmentation
Implementation from scratch in PyTorch.

Reference: Ronneberger et al. (2015) https://arxiv.org/abs/1505.04597

Architecture faithfully reproduces the paper:
  - Contracting path: 4 encoder blocks with 2x(Conv3x3-BN-ReLU) + MaxPool2x2
  - Bottleneck: 2x(Conv3x3-BN-ReLU) with 1024 channels
  - Expansive path: 4 decoder blocks with UpConv2x2 + skip-connection concat + 2x(Conv3x3-BN-ReLU)
  - Output: Conv1x1 -> num_classes
  - Same padding is used (unlike the paper's valid padding) for convenience with arbitrary input sizes
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────
# Building Blocks
# ─────────────────────────────────────────────

class DoubleConv(nn.Module):
    """Two consecutive Conv3x3 -> BatchNorm -> ReLU blocks."""

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
    """DoubleConv followed by MaxPool2x2 (downsampling)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        skip = self.conv(x)       # kept for skip connection
        pooled = self.pool(skip)  # downsampled
        return skip, pooled


class DecoderBlock(nn.Module):
    """UpConv2x2 -> concatenate skip -> DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Transposed convolution halves the channel count before concat
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                     kernel_size=2, stride=2)
        # After concat with skip (same channel count), double-conv runs
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle potential size mismatch (odd input dimensions)
        if x.shape != skip.shape:
            x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear',
                                          align_corners=False)
        x = torch.cat([skip, x], dim=1)  # channel-wise concatenation (skip first)
        return self.conv(x)


# ─────────────────────────────────────────────
# Full U-Net
# ─────────────────────────────────────────────

class UNet(nn.Module):
    """
    U-Net for image segmentation.

    Args:
        in_channels  : Number of input image channels (1 for grayscale, 3 for RGB).
        num_classes  : Number of output segmentation classes.
        base_features: Number of feature channels in the first encoder block (64 in paper).
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 2,
                 base_features: int = 64):
        super().__init__()

        f = base_features  # shorthand

        # ── Encoder (contracting path) ──────────────────────────────────────
        self.enc1 = EncoderBlock(in_channels, f)       # 64
        self.enc2 = EncoderBlock(f, f * 2)             # 128
        self.enc3 = EncoderBlock(f * 2, f * 4)         # 256
        self.enc4 = EncoderBlock(f * 4, f * 8)         # 512

        # ── Bottleneck ───────────────────────────────────────────────────────
        self.bottleneck = DoubleConv(f * 8, f * 16)    # 1024

        # ── Decoder (expansive path) ─────────────────────────────────────────
        self.dec4 = DecoderBlock(f * 16, f * 8)        # 1024 -> 512
        self.dec3 = DecoderBlock(f * 8, f * 4)         # 512  -> 256
        self.dec2 = DecoderBlock(f * 4, f * 2)         # 256  -> 128
        self.dec1 = DecoderBlock(f * 2, f)             # 128  -> 64

        # ── Output head ──────────────────────────────────────────────────────
        self.head = nn.Conv2d(f, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        skip1, x = self.enc1(x)
        skip2, x = self.enc2(x)
        skip3, x = self.enc3(x)
        skip4, x = self.enc4(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder (skip connections in reverse order)
        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)

        return self.head(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────

if __name__ == "__main__":
    model = UNet(in_channels=1, num_classes=2)
    dummy = torch.randn(2, 1, 256, 256)
    out = model(dummy)
    print(f"Input  : {dummy.shape}")
    print(f"Output : {out.shape}")
    print(f"Params : {model.count_parameters():,}")
    assert out.shape == (2, 2, 256, 256), "Shape mismatch!"
    print("✓ Architecture sanity check passed.")
