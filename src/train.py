"""
train.py — Train the Convolutional Autoencoder
================================================
Trains on tiles from tiles.h5, saves best model weights,
and plots training/validation loss curves.

Usage:
    python train.py

Outputs:
    ../models/best_model.pth        — best model weights (lowest val loss)
    ../models/final_model.pth       — model weights after final epoch
    ../models/loss_curve.png        — training/validation loss plot
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from dataset import get_data_loaders
from model import Autoencoder

# ── Configuration ──────────────────────────────────────────────
H5_PATH = "../data/tiles/tiles.h5"
MODEL_DIR = "../models"

BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
VAL_SPLIT = 0.2
PATIENCE = 5                # early stopping: stop if no improvement for N epochs
# ───────────────────────────────────────────────────────────────


def get_device():
    """Pick the best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")       # Apple Silicon GPU
    elif torch.cuda.is_available():
        return torch.device("cuda")      # NVIDIA GPU
    else:
        return torch.device("cpu")


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run one training epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    total_batches = len(loader)

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if n_batches % 50 == 0 or n_batches == total_batches:
            avg = total_loss / n_batches
            print(f"    batch {n_batches}/{total_batches} — loss: {avg:.6f}", flush=True)

    return total_loss / n_batches


def validate(model, loader, criterion, device):
    """Run validation. Returns average loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches


def plot_loss_curve(train_losses, val_losses, save_path):
    """Save a plot of training and validation loss."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, label="Train Loss", linewidth=2)
        ax.plot(epochs, val_losses, label="Val Loss", linewidth=2)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("MSE Loss", fontsize=12)
        ax.set_title("Autoencoder Training", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"  Loss curve saved to {save_path}")
    except ImportError:
        print("  (Install matplotlib to save loss curve)")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    device = get_device()

    print(f"Autoencoder Training")
    print(f"====================")
    print(f"  Device:     {device}")
    print(f"  Epochs:     {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  LR:         {LEARNING_RATE}")
    print(f"  Patience:   {PATIENCE}")
    print()

    # Data
    train_loader, val_loader, dataset = get_data_loaders(
        H5_PATH, batch_size=BATCH_SIZE, val_split=VAL_SPLIT
    )
    print()

    # Model
    model = Autoencoder().to(device)
    total_params, trainable_params = model.count_params()
    print(f"  Model: {trainable_params:,} trainable parameters")
    print()

    # Loss, optimizer, scheduler
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7
    )

    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_epoch = 0

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        # Check for improvement
        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            # Save best model
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_model.pth"))
            improved = " ★ saved"
        else:
            epochs_no_improve += 1

        print(
            f"  Epoch {epoch:3d}/{EPOCHS} | "
            f"Train: {train_loss:.6f} | "
            f"Val: {val_loss:.6f} | "
            f"LR: {current_lr:.1e} | "
            f"{epoch_time:.1f}s"
            f"{improved}"
        )

        # Early stopping
        if epochs_no_improve >= PATIENCE:
            print(f"\n  Early stopping — no improvement for {PATIENCE} epochs")
            break

    total_time = time.time() - start_time

    # Save final model
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "final_model.pth"))

    # Summary
    print(f"\n{'='*55}")
    print(f"Training complete in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Best val loss: {best_val_loss:.6f} (epoch {best_epoch})")
    print(f"  Models saved to: {os.path.abspath(MODEL_DIR)}/")
    print(f"    best_model.pth  — lowest validation loss")
    print(f"    final_model.pth — last epoch")

    # Plot loss curve
    plot_loss_curve(
        train_losses, val_losses,
        os.path.join(MODEL_DIR, "loss_curve.png")
    )

    # Cleanup
    dataset.close()


if __name__ == "__main__":
    main()