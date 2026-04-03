"""
Training Script for U-Net (from scratch)
=========================================
Trains the custom U-Net on the synthetic cell segmentation dataset.
Logs train/val loss and IoU at every epoch and saves the best checkpoint.

Usage:
    python train.py [--epochs 30] [--lr 1e-3] [--batch 8] [--out_dir runs/scratch]
"""

import argparse
import os
import time
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from unet_scratch import UNet
from dataset import SyntheticCellDataset


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def iou_score(pred_mask: torch.Tensor, true_mask: torch.Tensor,
              num_classes: int = 2) -> float:
    """
    Mean Intersection-over-Union over all classes (ignores empty classes).
    pred_mask, true_mask: (N, H, W) int tensors on CPU.
    """
    ious = []
    for cls in range(num_classes):
        pred_c = (pred_mask == cls)
        true_c = (true_mask == cls)
        intersection = (pred_c & true_c).sum().item()
        union = (pred_c | true_c).sum().item()
        if union == 0:
            continue
        ious.append(intersection / union)
    return float(np.mean(ious)) if ious else 0.0


def dice_score(pred_mask: torch.Tensor, true_mask: torch.Tensor,
               num_classes: int = 2) -> float:
    """Mean Dice coefficient over foreground classes."""
    scores = []
    for cls in range(1, num_classes):   # skip background (cls=0)
        pred_c = (pred_mask == cls).float()
        true_c = (true_mask == cls).float()
        inter = (pred_c * true_c).sum().item()
        denom = pred_c.sum().item() + true_c.sum().item()
        if denom == 0:
            continue
        scores.append(2 * inter / denom)
    return float(np.mean(scores)) if scores else 0.0


# ─────────────────────────────────────────────
# One epoch helpers
# ─────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, total_iou, total_dice = 0.0, 0.0, 0.0
    n_batches = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)           # (B, C, H, W)
            loss = criterion(logits, masks)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            preds = logits.argmax(dim=1).cpu()   # (B, H, W)
            masks_cpu = masks.cpu()

            total_loss += loss.item()
            total_iou += iou_score(preds, masks_cpu)
            total_dice += dice_score(preds, masks_cpu)
            n_batches += 1

    return (total_loss / n_batches,
            total_iou / n_batches,
            total_dice / n_batches)


# ─────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────

def train(
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 8,
    n_samples: int = 200,
    out_dir: str = "runs/scratch",
    seed: int = 42,
):
    torch.manual_seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    full_dataset = SyntheticCellDataset(n_samples=n_samples, height=128, width=128,
                                        augment=False, seed=seed)
    n_val = max(1, int(0.2 * n_samples))
    n_train = n_samples - n_val
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(seed))
    # Enable augmentation on training split
    train_ds.dataset.augment = True   # type: ignore

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=0)

    print(f"Train : {n_train} | Val : {n_val}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = UNet(in_channels=1, num_classes=2, base_features=32).to(device)
    print(f"Parameters : {model.count_parameters():,}")

    # ── Optimiser & Loss ──────────────────────────────────────────────────────
    # Class-weighted cross-entropy to handle foreground/background imbalance
    # (background pixels usually outnumber foreground)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.3, 0.7]).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=True
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    history = {"train_loss": [], "val_loss": [],
               "train_iou": [],  "val_iou": [],
               "train_dice": [], "val_dice": []}
    best_iou = 0.0
    best_path = os.path.join(out_dir, "best_model.pth")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_iou, tr_dice = run_epoch(model, train_loader, criterion,
                                             optimizer, device, train=True)
        vl_loss, vl_iou, vl_dice = run_epoch(model, val_loader, criterion,
                                             optimizer, device, train=False)
        scheduler.step(vl_loss)
        elapsed = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_iou"].append(tr_iou)
        history["val_iou"].append(vl_iou)
        history["train_dice"].append(tr_dice)
        history["val_dice"].append(vl_dice)

        if vl_iou > best_iou:
            best_iou = vl_iou
            torch.save(model.state_dict(), best_path)

        print(f"Epoch {epoch:03d}/{epochs} | "
              f"Loss {tr_loss:.4f}/{vl_loss:.4f} | "
              f"IoU  {tr_iou:.4f}/{vl_iou:.4f} | "
              f"Dice {tr_dice:.4f}/{vl_dice:.4f} | "
              f"{elapsed:.1f}s")

    print(f"\nBest val IoU : {best_iou:.4f}  (checkpoint → {best_path})")

    # ── Save history ──────────────────────────────────────────────────────────
    hist_path = os.path.join(out_dir, "history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    # ── Plot curves ───────────────────────────────────────────────────────────
    plot_training_curves(history, out_dir)

    return model, history, best_iou


def plot_training_curves(history: dict, out_dir: str):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"],   label="Val")
    axes[0].set_title("Cross-Entropy Loss", fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].legend()

    axes[1].plot(epochs, history["train_iou"], label="Train")
    axes[1].plot(epochs, history["val_iou"],   label="Val")
    axes[1].set_title("Mean IoU", fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].legend()

    axes[2].plot(epochs, history["train_dice"], label="Train")
    axes[2].plot(epochs, history["val_dice"],   label="Val")
    axes[2].set_title("Dice Score", fontweight="bold")
    axes[2].set_xlabel("Epoch"); axes[2].legend()

    plt.suptitle("U-Net (From Scratch) — Training Curves", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "training_curves.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves → {path}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",    type=int,   default=30)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--batch",     type=int,   default=8)
    parser.add_argument("--n_samples", type=int,   default=200)
    parser.add_argument("--out_dir",   type=str,   default="runs/scratch")
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch,
        n_samples=args.n_samples,
        out_dir=args.out_dir,
        seed=args.seed,
    )
