"""
view_anomalies.py — View the Top Anomalies Ranked by MSE
==========================================================
Loads the trained model, computes reconstruction errors,
and saves the top N most anomalous tiles as a ranked grid.

Usage:
    python view_anomalies.py

Output:
    ../models/anomaly_mosaic.png
"""

import os
import numpy as np
import torch
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import Autoencoder

# ── Configuration ──────────────────────────────────────────────
H5_PATH = "../data/tiles/tiles.h5"
MODEL_PATH = "../models/best_model.pth"
OUTPUT_PATH = "../models/anomaly_mosaic.png"

BATCH_SIZE = 128
NUM_SAMPLES = 64             # how many anomalies to show (will be arranged in a grid)
COLS = 8                     # columns in the grid
# ───────────────────────────────────────────────────────────────


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def main():
    device = get_device()

    # Load model
    model = Autoencoder().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    # Load threshold
    threshold = np.load("../models/threshold.npy")
    print(f"Anomaly threshold: {threshold:.6f}")

    # Compute reconstruction errors
    print("Computing reconstruction errors...")
    errors = []
    with h5py.File(H5_PATH, "r") as f:
        tiles = f["tiles"]
        total = tiles.shape[0]

        with torch.no_grad():
            for start in range(0, total, BATCH_SIZE):
                end = min(start + BATCH_SIZE, total)
                batch = tiles[start:end]
                x = torch.from_numpy(batch).unsqueeze(1).float().to(device)
                recon = model(x)
                mse = torch.mean((x - recon) ** 2, dim=(1, 2, 3))
                errors.extend(mse.cpu().numpy())

    errors = np.array(errors)

    # Find anomaly indices
    anomaly_indices = np.where(errors > threshold)[0]
    print(f"Found {len(anomaly_indices)} anomalies")

    # Take the top N by highest MSE
    n = min(NUM_SAMPLES, len(anomaly_indices))
    top_indices = np.argsort(errors)[-n:][::-1]  # highest error first

    rows = int(np.ceil(n / COLS))

    # Build the mosaic
    print(f"Building {rows}x{COLS} mosaic of top {n} anomalies (ranked by MSE)...")
    fig, axes = plt.subplots(rows, COLS, figsize=(COLS * 2.2, rows * 2.4))

    with h5py.File(H5_PATH, "r") as f:
        tiles = f["tiles"]

        for i in range(rows * COLS):
            row = i // COLS
            col = i % COLS
            ax = axes[row, col] if rows > 1 else axes[col]

            if i < n:
                idx = top_indices[i]
                tile = tiles[idx]
                ax.imshow(tile, cmap="gray", origin="lower", vmin=0, vmax=1)
                ax.set_title(f"#{i+1}  MSE: {errors[idx]:.5f}", fontsize=7, pad=2)
            else:
                ax.axis("off")
                continue

            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle(
        f"Top {n} Anomalies Ranked by Reconstruction Error",
        fontsize=14, fontweight="bold", y=1.01
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved to {os.path.abspath(OUTPUT_PATH)}")


if __name__ == "__main__":
    main()