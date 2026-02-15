"""
model.py — Convolutional Autoencoder for Anomaly Detection
===========================================================
Encoder compresses 1x128x128 tiles down to a bottleneck,
Decoder reconstructs back to 1x128x128.

Architecture:
    Encoder: 1x128x128 → 32x64x64 → 64x32x32 → 128x16x16 → 256x8x8 → 512x4x4
    Decoder: 512x4x4 → 256x8x8 → 128x16x16 → 64x32x32 → 32x64x64 → 1x128x128
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # 1x128x128 → 32x64x64
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 32x64x64 → 64x32x32
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 64x32x32 → 128x16x16
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 128x16x16 → 256x8x8
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 256x8x8 → 512x4x4
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # 512x4x4 → 256x8x8
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 256x8x8 → 128x16x16
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 128x16x16 → 64x32x32
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 64x32x32 → 32x64x64
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 32x64x64 → 1x128x128
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),  # output in [0, 1] to match normalized input
        )

    def forward(self, x):
        return self.net(x)


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon

    def get_bottleneck(self, x):
        """Extract the compressed representation (useful for analysis)."""
        return self.encoder(x)

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


if __name__ == "__main__":
    # Quick test to verify shapes
    model = Autoencoder()
    total, trainable = model.count_params()
    print(f"Autoencoder — {trainable:,} trainable parameters")

    x = torch.randn(1, 1, 128, 128)
    z = model.encoder(x)
    recon = model(x)
    print(f"  Input:      {list(x.shape)}")
    print(f"  Bottleneck: {list(z.shape)}")
    print(f"  Output:     {list(recon.shape)}")