"""
evaluate.py — Evaluate the Trained Autoencoder & Detect Anomalies
==================================================================
Loads the best model, computes reconstruction errors across all tiles,
sets an anomaly threshold, and visualizes the results.

Usage:
    python evaluate.py

Outputs:
    ../models/threshold.npy               — saved anomaly threshold value
    ../models/error_distribution.png      — histogram of reconstruction errors
    ../models/anomalies.png               — top anomalous tiles visualization
    ../models/reconstructions.png         — random input vs reconstruction comparison
"""

import os
import numpy as np
import torch
import h5py
from model import Autoencoder

# ── Configuration ──────────────────────────────────────────────
H5_PATH = "../data/tiles/tiles.h5"
MODEL_PATH = "../models/best_model.pth"
OUTPUT_DIR = "../models"

BATCH_SIZE = 128
ANOMALY_PERCENTILE = 97      # tiles above this percentile are flagged as anomalies
TOP_N_ANOMALIES = 16         # how many top anomalies to visualize
# ───────────────────────────────────────────────────────────────


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def compute_reconstruction_errors(model, h5_path, device, batch_size):
    """
    Run every tile through the model and compute per-tile MSE.
    Returns array of reconstruction errors and indices.
    """
    model.eval()
    errors = []

    with h5py.File(h5_path, "r") as f:
        tiles = f["tiles"]
        total = tiles.shape[0]
        print(f"  Computing reconstruction errors for {total} tiles...")

        with torch.no_grad():
            for start in range(0, total, batch_size):
                end = min(start + batch_size, total)
                batch = tiles[start:end]

                # (N, 128, 128) -> (N, 1, 128, 128)
                x = torch.from_numpy(batch).unsqueeze(1).float().to(device)
                recon = model(x)

                # Per-tile MSE: mean over (C, H, W), keep batch dim
                mse = torch.mean((x - recon) ** 2, dim=(1, 2, 3))
                errors.extend(mse.cpu().numpy())

                if (start // batch_size) % 50 == 0:
                    print(f"    Processed {end}/{total} tiles")

    return np.array(errors)


def plot_error_distribution(errors, threshold, save_path):
    """Plot histogram of reconstruction errors with threshold line."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(errors, bins=200, color="steelblue", alpha=0.8, edgecolor="none")
    ax.axvline(threshold, color="red", linewidth=2, linestyle="--",
               label=f"Threshold ({ANOMALY_PERCENTILE}th pctl): {threshold:.6f}")

    n_anomalies = np.sum(errors > threshold)
    ax.set_xlabel("Reconstruction Error (MSE)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Reconstruction Error Distribution — {n_anomalies} anomalies detected", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Error distribution saved to {save_path}")


def plot_top_anomalies(model, h5_path, errors, device, n=16, save_path=None):
    """Visualize the top N most anomalous tiles with their reconstructions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    top_indices = np.argsort(errors)[-n:][::-1]

    with h5py.File(h5_path, "r") as f:
        tiles = f["tiles"]

        rows = min(n, 8)
        fig, axes = plt.subplots(rows, 3, figsize=(10, rows * 3.2))
        if rows == 1:
            axes = axes[np.newaxis, :]

        axes[0, 0].set_title("Original", fontsize=12, fontweight="bold")
        axes[0, 1].set_title("Reconstruction", fontsize=12, fontweight="bold")
        axes[0, 2].set_title("Error Map", fontsize=12, fontweight="bold")

        for i, idx in enumerate(top_indices[:rows]):
            tile = tiles[idx]
            x = torch.from_numpy(tile).unsqueeze(0).unsqueeze(0).float().to(device)

            with torch.no_grad():
                recon = model(x)

            original = tile
            reconstructed = recon.squeeze().cpu().numpy()
            error_map = (original - reconstructed) ** 2

            axes[i, 0].imshow(original, cmap="gray", origin="lower", vmin=0, vmax=1)
            axes[i, 0].set_ylabel(f"#{i+1}\nMSE: {errors[idx]:.5f}", fontsize=9)
            axes[i, 0].set_xticks([])
            axes[i, 0].set_yticks([])

            axes[i, 1].imshow(reconstructed, cmap="gray", origin="lower", vmin=0, vmax=1)
            axes[i, 1].axis("off")

            im = axes[i, 2].imshow(error_map, cmap="hot", origin="lower")
            axes[i, 2].axis("off")

        plt.suptitle("Top Anomalous Tiles", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"  Top anomalies saved to {save_path}")


def plot_random_reconstructions(model, h5_path, device, n=8, save_path=None):
    """Show random tiles with their reconstructions to check model quality."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()

    with h5py.File(h5_path, "r") as f:
        tiles = f["tiles"]
        total = tiles.shape[0]
        indices = np.random.choice(total, size=n, replace=False)

        fig, axes = plt.subplots(2, n, figsize=(n * 2.5, 5.5))

        axes[0, 0].set_ylabel("Original", fontsize=11, fontweight="bold")
        axes[1, 0].set_ylabel("Reconstructed", fontsize=11, fontweight="bold")

        for i, idx in enumerate(indices):
            tile = tiles[idx]
            x = torch.from_numpy(tile).unsqueeze(0).unsqueeze(0).float().to(device)

            with torch.no_grad():
                recon = model(x)

            original = tile
            reconstructed = recon.squeeze().cpu().numpy()
            mse = np.mean((original - reconstructed) ** 2)

            axes[0, i].imshow(original, cmap="gray", origin="lower", vmin=0, vmax=1)
            axes[0, i].set_title(f"Tile {idx}", fontsize=9)
            axes[0, i].axis("off")

            axes[1, i].imshow(reconstructed, cmap="gray", origin="lower", vmin=0, vmax=1)
            axes[1, i].set_title(f"MSE: {mse:.5f}", fontsize=9)
            axes[1, i].axis("off")

        plt.suptitle("Random Reconstruction Samples", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"  Reconstruction samples saved to {save_path}")


def main():
    device = get_device()

    print(f"Autoencoder Evaluation")
    print(f"======================")
    print(f"  Device: {device}")
    print(f"  Model:  {MODEL_PATH}")
    print()

    # Load model
    model = Autoencoder().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    print(f"  Model loaded successfully")

    # Compute reconstruction errors for all tiles
    errors = compute_reconstruction_errors(model, H5_PATH, device, BATCH_SIZE)

    # Set anomaly threshold
    threshold = np.percentile(errors, ANOMALY_PERCENTILE)
    n_anomalies = np.sum(errors > threshold)

    print(f"\n  Error stats:")
    print(f"    Mean:      {errors.mean():.6f}")
    print(f"    Std:       {errors.std():.6f}")
    print(f"    Min:       {errors.min():.6f}")
    print(f"    Max:       {errors.max():.6f}")
    print(f"    Threshold: {threshold:.6f} ({ANOMALY_PERCENTILE}th percentile)")
    print(f"    Anomalies: {n_anomalies} / {len(errors)} ({100*n_anomalies/len(errors):.1f}%)")

    # Save threshold for later use
    np.save(os.path.join(OUTPUT_DIR, "threshold.npy"), threshold)
    np.save(os.path.join(OUTPUT_DIR, "errors.npy"), errors)
    print(f"\n  Threshold and errors saved to {OUTPUT_DIR}/")

    # Visualizations
    print()
    plot_error_distribution(
        errors, threshold,
        os.path.join(OUTPUT_DIR, "error_distribution.png")
    )

    plot_top_anomalies(
        model, H5_PATH, errors, device,
        n=TOP_N_ANOMALIES,
        save_path=os.path.join(OUTPUT_DIR, "anomalies.png")
    )

    plot_random_reconstructions(
        model, H5_PATH, device,
        n=8,
        save_path=os.path.join(OUTPUT_DIR, "reconstructions.png")
    )

    print(f"\n{'='*55}")
    print(f"Evaluation complete!")
    print(f"  Check {OUTPUT_DIR}/ for visualizations.")
    print(f"\nTo classify a new tile as anomalous:")
    print(f"  threshold = np.load('{OUTPUT_DIR}/threshold.npy')")
    print(f"  if reconstruction_error > threshold:")
    print(f"      print('ANOMALY')")


if __name__ == "__main__":
    main()