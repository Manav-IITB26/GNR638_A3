"""
Synthetic Toy Dataset for U-Net
================================
Generates grayscale images with randomly placed elliptical "cells" on a noisy
background, together with binary segmentation masks — mimicking the biomedical
datasets used in the U-Net paper (EM / light-microscopy images).

Each sample:
  image  : (1, H, W)  float32, values in [0, 1]
  mask   : (H, W)     int64,   0 = background, 1 = foreground (cell)
"""

import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
import os


# ─────────────────────────────────────────────
# Synthetic image generator
# ─────────────────────────────────────────────

def generate_cell_sample(
    height: int = 256,
    width: int = 256,
    n_cells: int = 8,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        image : float32 (H, W), intensity in [0, 1]
        mask  : uint8  (H, W), 0 or 1
    """
    rng = np.random.default_rng(seed)

    # Background: low-intensity Gaussian noise
    image = rng.normal(loc=0.15, scale=0.05, size=(height, width)).astype(np.float32)
    mask = np.zeros((height, width), dtype=np.uint8)

    for _ in range(n_cells):
        cx = rng.integers(20, width - 20)
        cy = rng.integers(20, height - 20)
        rx = rng.integers(10, 30)
        ry = rng.integers(10, 30)
        angle = rng.integers(0, 180)
        intensity = rng.uniform(0.55, 0.95)

        # Draw ellipse on mask
        cv2.ellipse(mask, (cx, cy), (rx, ry), angle, 0, 360, 1, -1)

        # Draw slightly brighter ellipse on image
        cell_img = np.zeros_like(image)
        cv2.ellipse(cell_img, (cx, cy), (rx, ry), angle, 0, 360, intensity, -1)
        image = np.where(mask.astype(bool), np.maximum(image, cell_img), image)

    # Add per-pixel noise on top
    noise = rng.normal(0, 0.04, size=(height, width)).astype(np.float32)
    image = np.clip(image + noise, 0.0, 1.0)

    # Simulate gradient illumination artifact (common in microscopy)
    yy, xx = np.meshgrid(np.linspace(0, 1, height), np.linspace(0, 1, width), indexing='ij')
    vignette = 1.0 - 0.2 * ((yy - 0.5) ** 2 + (xx - 0.5) ** 2)
    image = np.clip(image * vignette, 0.0, 1.0)

    return image, mask


# ─────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────

class SyntheticCellDataset(Dataset):
    """
    In-memory synthetic dataset.

    Args:
        n_samples    : number of images to generate.
        height/width : spatial dimensions.
        n_cells      : approximate number of cells per image.
        augment      : apply random flips & 90° rotations during training.
        seed         : base seed for reproducibility.
    """

    def __init__(
        self,
        n_samples: int = 200,
        height: int = 256,
        width: int = 256,
        n_cells: int = 8,
        augment: bool = False,
        seed: int = 42,
    ):
        self.augment = augment
        self.images: list[np.ndarray] = []
        self.masks: list[np.ndarray] = []

        for i in range(n_samples):
            img, msk = generate_cell_sample(height, width, n_cells, seed=seed + i)
            self.images.append(img)
            self.masks.append(msk)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image = self.images[idx].copy()   # (H, W)
        mask = self.masks[idx].copy()     # (H, W)

        # ── Data augmentation (identical transform applied to image + mask) ──
        if self.augment:
            if random.random() > 0.5:
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()
            if random.random() > 0.5:
                image = np.flipud(image).copy()
                mask = np.flipud(mask).copy()
            k = random.randint(0, 3)
            image = np.rot90(image, k).copy()
            mask = np.rot90(mask, k).copy()

        image_tensor = torch.from_numpy(image).float().unsqueeze(0)  # (1, H, W)
        mask_tensor = torch.from_numpy(mask).long()          # (H, W)

        return image_tensor, mask_tensor


# ─────────────────────────────────────────────
# Visualise a few samples
# ─────────────────────────────────────────────

def visualise_samples(n: int = 4, save_path: str = "sample_data.png"):
    dataset = SyntheticCellDataset(n_samples=n, seed=0)
    fig, axes = plt.subplots(n, 2, figsize=(7, n * 3.2))
    for i in range(n):
        img, msk = dataset[i]
        axes[i, 0].imshow(img.squeeze().numpy(), cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_title(f"Image {i+1}", fontsize=11)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(msk.numpy(), cmap="binary_r")
        axes[i, 1].set_title(f"Ground-Truth Mask {i+1}", fontsize=11)
        axes[i, 1].axis("off")
    plt.suptitle("Synthetic Cell Dataset Samples", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved sample visualisation → {save_path}")


if __name__ == "__main__":
    visualise_samples(n=4, save_path="sample_data.png")
    ds = SyntheticCellDataset(n_samples=10)
    img, msk = ds[0]
    print(f"Image shape : {img.shape}  dtype: {img.dtype}")
    print(f"Mask  shape : {msk.shape}  dtype: {msk.dtype}")
    print(f"Mask unique values: {msk.unique().tolist()}")
