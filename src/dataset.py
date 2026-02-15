"""
dataset.py — PyTorch Dataset for HDF5 Tiles
=============================================
Loads tiles from tiles.h5 and provides train/val splits.

Usage:
    from dataset import get_data_loaders
    train_loader, val_loader = get_data_loaders("../data/tiles/tiles.h5")
"""

import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split


class TileDataset(Dataset):
    """
    Reads tiles from an HDF5 file.
    Returns (input, target) where both are the same tile —
    this is what an autoencoder needs.
    """

    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.file = h5py.File(h5_path, "r")
        self.tiles = self.file["tiles"]
        self.length = self.tiles.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        tile = self.tiles[idx]
        # (128, 128) -> (1, 128, 128) — add channel dimension
        tensor = torch.from_numpy(tile).unsqueeze(0).float()
        return tensor, tensor

    def close(self):
        self.file.close()


def get_data_loaders(h5_path, batch_size=64, val_split=0.2, num_workers=0, seed=42):
    """
    Create train and validation DataLoaders from an HDF5 tile file.

    Args:
        h5_path:     path to tiles.h5
        batch_size:  batch size for training
        val_split:   fraction of data for validation (0.0 - 1.0)
        num_workers:  DataLoader workers (0 for Mac compatibility)
        seed:        random seed for reproducible splits

    Returns:
        (train_loader, val_loader, dataset)
    """
    dataset = TileDataset(h5_path)
    total = len(dataset)

    val_size = int(total * val_split)
    train_size = total - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Dataset loaded: {total} tiles")
    print(f"  Train: {train_size}  |  Val: {val_size}")
    print(f"  Batch size: {batch_size}")

    return train_loader, val_loader, dataset