"""
FITS Frame Tiling Preprocessor
===============================
Reads raw SDSS .fits.bz2 frames, normalizes pixel values,
cuts them into 128x128 tiles, filters out empty tiles,
and saves everything into a single HDF5 file ready for PyTorch.

Usage:
    pip install astropy numpy h5py matplotlib
    python preprocess_tiles.py

Output:
    ../data/tiles/tiles.h5          — all tiles in one HDF5 file
    ../data/tiles/spot_check/       — sample PNGs for visual verification

Loading in PyTorch:
    import h5py, torch
    from torch.utils.data import Dataset

    class TileDataset(Dataset):
        def __init__(self, h5_path):
            self.file = h5py.File(h5_path, "r")
            self.tiles = self.file["tiles"]

        def __len__(self):
            return self.tiles.shape[0]

        def __getitem__(self, idx):
            tile = self.tiles[idx]
            # Add channel dim: (128, 128) -> (1, 128, 128)
            tensor = torch.from_numpy(tile).unsqueeze(0)
            return tensor, tensor  # input = target for autoencoder

        def close(self):
            self.file.close()
"""


import os
import glob
import numpy as np
import h5py
import itertools
from astropy.io import fits
import warnings

warnings.filterwarnings("ignore", category=fits.verify.VerifyWarning)

# ── Configuration ──────────────────────────────────────────────
RAW_DIR = "../data/raw"
OUTPUT_DIR = "../data/tiles"
OUTPUT_FILE = "tiles.h5"

TILE_SIZE = 128              # tile dimensions (128x128 pixels)
EDGE_CROP = 10               # pixels to trim from frame edges
MIN_STD = 0.01               # minimum std dev to keep a tile (filters blank sky)
ASINH_SCALE = 0.1            # asinh softening parameter
# ───────────────────────────────────────────────────────────────


def load_fits_image(filepath):
    """Load a FITS file and return the image data as a numpy array."""
    try:
        with fits.open(filepath) as hdul:
            data = hdul[0].data
            if data is None:
                data = hdul[1].data
            return None if data is None else data.astype(np.float64)
    except Exception as e:
        print(f"    ✗ Error loading {os.path.basename(filepath)}: {e}")
        return None


def clean_frame(data):
    """Replace NaN/inf values and crop edges."""
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    if EDGE_CROP > 0:
        data = data[EDGE_CROP:-EDGE_CROP, EDGE_CROP:-EDGE_CROP]

    return data


def normalize_frame(data):
    """Apply asinh stretch and min-max normalize to [0, 1]."""
    stretched = np.arcsinh(data / ASINH_SCALE)

    vmin = stretched.min()
    vmax = stretched.max()
    if vmax - vmin == 0:
        return np.zeros_like(stretched, dtype=np.float32)

    normalized = (stretched - vmin) / (vmax - vmin)
    return normalized.astype(np.float32)


def tile_frame(data, tile_size):
    """Cut a 2D array into non-overlapping tiles."""
    h, w = data.shape
    tiles = []

    n_rows = h // tile_size
    n_cols = w // tile_size

    for r, c in itertools.product(range(n_rows), range(n_cols)):
        y = r * tile_size
        x = c * tile_size
        tile = data[y:y + tile_size, x:x + tile_size]
        tiles.append(tile)

    return tiles


def filter_tiles(tiles, min_std):
    """Remove tiles that are mostly empty sky (low standard deviation)."""
    kept = []
    kept.extend(tile for tile in tiles if np.std(tile) >= min_std)
    return kept


def spot_check(h5_path, n=8):  # sourcery skip: extract-method
    """Save sample tiles as PNGs for visual verification."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        check_dir = os.path.join(OUTPUT_DIR, "spot_check")
        os.makedirs(check_dir, exist_ok=True)

        with h5py.File(h5_path, "r") as f:
            total = f["tiles"].shape[0]
            indices = np.random.choice(total, size=min(n, total), replace=False)

            fig, axes = plt.subplots(2, 4, figsize=(14, 7))
            for ax, idx in zip(axes.flat, indices):
                ax.imshow(f["tiles"][idx], cmap="gray", origin="lower", vmin=0, vmax=1)
                ax.set_title(f"Tile {idx}", fontsize=9)
                ax.axis("off")

            plt.suptitle(f"Sample Tiles ({total} total)", fontsize=13)
            plt.tight_layout()
            fig.savefig(os.path.join(check_dir, "samples.png"), dpi=150)
            plt.close(fig)

        print(f"  Spot check saved to {check_dir}/samples.png")

    except ImportError:
        print("  (Install matplotlib for spot-check images)")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find all FITS files
    patterns = [
        os.path.join(RAW_DIR, "*.fits.bz2"),
        os.path.join(RAW_DIR, "*.fits"),
        os.path.join(RAW_DIR, "*.fits.gz"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    files.sort()

    if not files:
        print(f"No FITS files found in {os.path.abspath(RAW_DIR)}")
        return

    print("FITS Tile Preprocessor")
    print("======================")
    print(f"  Input dir:  {os.path.abspath(RAW_DIR)}")
    print(f"  Output:     {os.path.abspath(os.path.join(OUTPUT_DIR, OUTPUT_FILE))}")
    print(f"  FITS files: {len(files)}")
    print(f"  Tile size:  {TILE_SIZE}x{TILE_SIZE}")
    print(f"  Edge crop:  {EDGE_CROP}px")
    print(f"  Min std:    {MIN_STD}")
    print()

    h5_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    total_tiles = 0
    files_processed = 0
    files_skipped = 0

    # Use a resizable HDF5 dataset so we don't need to know total count upfront
    with h5py.File(h5_path, "w") as h5f:
        ds = h5f.create_dataset(
            "tiles",
            shape=(0, TILE_SIZE, TILE_SIZE),
            maxshape=(None, TILE_SIZE, TILE_SIZE),
            dtype=np.float32,
            chunks=(64, TILE_SIZE, TILE_SIZE),
            compression="gzip",
            compression_opts=4,
        )

        for i, filepath in enumerate(files):
            filename = os.path.basename(filepath)

            # Step 1: Load
            data = load_fits_image(filepath)
            if data is None:
                files_skipped += 1
                continue

            # Handle multi-band frames (3D arrays) — take first band
            if data.ndim == 3:
                data = data[0]
            if data.ndim != 2:
                print(f"    ⚠ Skipping {filename} — unexpected shape {data.shape}")
                files_skipped += 1
                continue

            # Step 2: Clean
            data = clean_frame(data)

            # Step 3: Normalize
            data = normalize_frame(data)

            # Step 4: Tile
            tiles = tile_frame(data, TILE_SIZE)

            # Step 5: Filter empty tiles
            tiles = filter_tiles(tiles, MIN_STD)

            if len(tiles) == 0:
                files_skipped += 1
                continue

            # Step 6: Append to HDF5
            tiles_array = np.stack(tiles, axis=0)
            old_size = ds.shape[0]
            new_size = old_size + tiles_array.shape[0]
            ds.resize(new_size, axis=0)
            ds[old_size:new_size] = tiles_array

            total_tiles = new_size
            files_processed += 1

            print(f"  [{i+1}/{len(files)}] {filename} → {len(tiles)} tiles  (total: {total_tiles})")

    # Summary
    print(f"\n{'='*50}")
    print("Done!")
    print(f"  Files processed: {files_processed}")
    print(f"  Files skipped:   {files_skipped}")
    print(f"  Total tiles:     {total_tiles}")

    file_size_mb = os.path.getsize(h5_path) / (1024 * 1024)
    print(f"  Output size:     {file_size_mb:.1f} MB")
    print(f"  Saved to:        {os.path.abspath(h5_path)}")

    # Step 7: Spot check
    if total_tiles > 0:
        print()
        spot_check(h5_path)

    # Print PyTorch usage hint
    print("\n── PyTorch Usage ──────────────────────────────────")
    print("  import h5py, torch")
    print("  from torch.utils.data import Dataset, DataLoader")
    print("")
    print("  class TileDataset(Dataset):")
    print(f"      def __init__(self, path='{h5_path}'):")
    print("          self.file = h5py.File(path, 'r')")
    print("          self.tiles = self.file['tiles']")
    print("      def __len__(self):")
    print("          return self.tiles.shape[0]")
    print("      def __getitem__(self, idx):")
    print("          t = torch.from_numpy(self.tiles[idx]).unsqueeze(0)")
    print("          return t, t  # autoencoder: input = target")
    print("")
    print("  dataset = TileDataset()")
    print("  loader = DataLoader(dataset, batch_size=32, shuffle=True)")


if __name__ == "__main__":
    main()