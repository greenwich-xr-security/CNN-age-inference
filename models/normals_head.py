"""Lightweight auxiliary head for surface normal map reconstruction (Option A)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NormalsHead(nn.Module):
    """Shallow upsampling decoder that maps CNN spatial features to a normal map.

    Takes [B, C, h, w] spatial features from the backbone (before global pooling)
    and progressively upsamples to [B, 3, H, W] with tanh activation so output
    values lie in [-1, 1] (matching the decoded JPEG normal map convention).

    The decoder is intentionally shallow so the geometric signal is forced into
    the shared encoder rather than being reconstructed by the decoder alone.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            # channel compression
            nn.Conv2d(in_channels, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # upsample ×2
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # upsample ×2
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # upsample ×2
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, spatial: torch.Tensor, target_size: int) -> torch.Tensor:
        x = self.blocks(spatial)
        if x.shape[-1] != target_size or x.shape[-2] != target_size:
            x = F.interpolate(
                x,
                size=(target_size, target_size),
                mode="bilinear",
                align_corners=False,
            )
        return x
