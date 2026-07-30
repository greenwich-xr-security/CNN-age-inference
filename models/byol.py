from __future__ import annotations

from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F


class BYOLProjector(nn.Module):
    def __init__(self, in_dim: int, *, hidden_dim: int = 4096, out_dim: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class BYOLPredictor(nn.Module):
    def __init__(self, dim: int, *, hidden_dim: int = 4096) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class BYOLNetwork(nn.Module):
    """backbone -> projector, with an optional online-only predictor.

    Build the online network with a `BYOLPredictor`; build the target network with
    `predictor=None`. BYOL has no target-side predictor, so the EMA update between
    online and target must only track backbone + projector parameters -- use
    `encoder_parameters()` for that rather than `.parameters()`, which would also
    pick up the online network's predictor.
    """

    def __init__(self, backbone: nn.Module, projector: nn.Module, predictor: nn.Module | None = None) -> None:
        super().__init__()
        self.backbone = backbone
        self.projector = projector
        self.predictor = predictor

    def encoder_parameters(self) -> Iterator[nn.Parameter]:
        for module in (self.backbone, self.projector):
            yield from module.parameters()

    def represent(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(self.backbone(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.represent(x)
        if self.predictor is not None:
            return self.predictor(projected)
        return projected


def byol_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample negative cosine similarity, in [0, 4]; 0 when pred and target align.

    `target` is expected to already be detached (stop-gradient). The caller is
    responsible for symmetrizing over both view orderings (BYOL feeds view a through
    online / view b through target, and vice versa, then sums both loss terms).
    """
    pred = F.normalize(pred, dim=-1, p=2)
    target = F.normalize(target, dim=-1, p=2)
    return 2 - 2 * (pred * target).sum(dim=-1)
