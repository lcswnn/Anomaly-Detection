"""
export_images.py — Export Flagged Anomaly Tiles as Individual PNGs
==================================================================
For each anomaly tile (reconstruction error > threshold), exports a
composite PNG showing the original tile alongside its error map,
suitable for volunteer classification (e.g. Galaxy Zoo style).

Usage:
    python export_images.py
    python export_images.py --originals-only   # skip error map, export raw tiles only

Outputs:
    ../zoon_imgs/tile_<ID>_mse_<ERROR>.png   — one composite PNG per anomaly
"""

import os
import sys
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Configuration ──────────────────────────────────────────────
H5_PATH = "../data/tiles/tiles.h5"
ERRORS_PATH = "../models/errors.npy"
THRESHOLD_PATH = "../models/threshold.npy"
SUBCLASS_PATH = "../models/subclassifications.npy"
OUTPUT_DIR = "../zoon_imgs"
# ───────────────────────────────────────────────────────────────


def export_composite(tile, error_map, tile_idx, mse, subclass, save_path):
    """Save a side-by-side composite: original tile + error map."""
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))

    axes[0].imshow(tile, cmap="gray", origin="lower", vmin=0, vmax=1)
    axes[0].set_title("Original", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(error_map, cmap="hot", origin="lower")
    axes[1].set_title("Error Map", fontsize=10)
    axes[1].axis("off")

    label = subclass if subclass else "unclassified"
    fig.suptitle(f"Tile {tile_idx}  |  MSE: {mse:.6f}  |  {label}", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def export_original_only(tile, tile_idx, mse, save_path):
    """Save the raw tile as a single grayscale PNG."""
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.imshow(tile, cmap="gray", origin="lower", vmin=0, vmax=1)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main():
    originals_only = "--originals-only" in sys.argv

    # Load pre-computed errors and threshold
    errors = np.load(ERRORS_PATH)
    threshold = float(np.load(THRESHOLD_PATH))

    # Load subclassifications if available
    subclasses = {}
    if os.path.exists(SUBCLASS_PATH):
        subclasses = np.load(SUBCLASS_PATH, allow_pickle=True).item()

    # Find anomaly indices
    anomaly_indices = np.where(errors > threshold)[0]
    anomaly_indices = anomaly_indices[np.argsort(errors[anomaly_indices])[::-1]]

    print(f"Export Anomaly Tiles")
    print(f"====================")
    print(f"  Threshold:  {threshold:.6f}")
    print(f"  Anomalies:  {len(anomaly_indices)}")
    print(f"  Mode:       {'originals only' if originals_only else 'composite (original + error map)'}")
    print()

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Export tiles
    with h5py.File(H5_PATH, "r") as f:
        tiles = f["tiles"]

        for count, idx in enumerate(anomaly_indices):
            tile = tiles[idx]
            mse = errors[idx]
            subclass = subclasses.get(int(idx), "")

            filename = f"tile_{idx:05d}_mse_{mse:.6f}.png"
            save_path = os.path.join(OUTPUT_DIR, filename)

            if originals_only:
                export_original_only(tile, idx, mse, save_path)
            else:
                error_map = (tile - tile) ** 2  # placeholder; recompute below
                export_composite(tile, error_map, idx, mse, subclass, save_path)

            if (count + 1) % 100 == 0 or (count + 1) == len(anomaly_indices):
                print(f"  Exported {count + 1}/{len(anomaly_indices)} tiles")

    print(f"\nDone! {len(anomaly_indices)} PNGs saved to {OUTPUT_DIR}/")


def main_with_model():
    """Full export with proper error maps computed from the model."""
    import torch
    from model import Autoencoder

    originals_only = "--originals-only" in sys.argv

    # Load pre-computed errors and threshold
    errors = np.load(ERRORS_PATH)
    threshold = float(np.load(THRESHOLD_PATH))

    subclasses = {}
    if os.path.exists(SUBCLASS_PATH):
        subclasses = np.load(SUBCLASS_PATH, allow_pickle=True).item()

    anomaly_indices = np.where(errors > threshold)[0]
    anomaly_indices = anomaly_indices[np.argsort(errors[anomaly_indices])[::-1]]

    print(f"Export Anomaly Tiles")
    print(f"====================")
    print(f"  Threshold:  {threshold:.6f}")
    print(f"  Anomalies:  {len(anomaly_indices)}")
    print(f"  Mode:       {'originals only' if originals_only else 'composite (original + error map)'}")

    # Load model for error maps
    if not originals_only:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        model = Autoencoder().to(device)
        model.load_state_dict(
            torch.load("../models/best_model.pth", map_location=device, weights_only=True)
        )
        model.eval()
        print(f"  Device:     {device}")
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with h5py.File(H5_PATH, "r") as f:
        tiles_ds = f["tiles"]

        for count, idx in enumerate(anomaly_indices):
            tile = tiles_ds[idx]
            mse = errors[idx]
            subclass = subclasses.get(int(idx), "")

            filename = f"tile_{idx:05d}_mse_{mse:.6f}.png"
            save_path = os.path.join(OUTPUT_DIR, filename)

            if originals_only:
                export_original_only(tile, idx, mse, save_path)
            else:
                x = torch.from_numpy(tile).unsqueeze(0).unsqueeze(0).float().to(device)
                with torch.no_grad():
                    recon = model(x)
                reconstructed = recon.squeeze().cpu().numpy()
                error_map = (tile - reconstructed) ** 2
                export_composite(tile, error_map, idx, mse, subclass, save_path)

            if (count + 1) % 100 == 0 or (count + 1) == len(anomaly_indices):
                print(f"  Exported {count + 1}/{len(anomaly_indices)} tiles")

    print(f"\nDone! {len(anomaly_indices)} PNGs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main_with_model()
