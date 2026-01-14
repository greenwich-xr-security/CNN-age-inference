from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightNormLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, *, bias: bool = True, dim: int = 0) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.dim = int(dim)
        self.weight_v = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_g = nn.Parameter(torch.ones(out_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight_v, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_v)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim == 0:
            norm = torch.norm(self.weight_v, dim=0, keepdim=True)
            weight = self.weight_v * (self.weight_g.view(-1, 1) / (norm + 1e-6))
        else:
            norm = torch.norm(self.weight_v, dim=1, keepdim=True)
            weight = self.weight_v * (self.weight_g.view(-1, 1) / (norm + 1e-6))
        return F.linear(x, weight, self.bias)


class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        *,
        out_dim: int = 8192,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        nlayers: int = 3,
    ) -> None:
        super().__init__()
        if nlayers < 1:
            raise ValueError("nlayers must be >= 1")

        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)

        self.last_layer = WeightNormLinear(bottleneck_dim, out_dim, bias=False, dim=0)
        with torch.no_grad():
            self.last_layer.weight_g.fill_(1.0)
        self.last_layer.weight_g.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        return self.last_layer(x)


class DinoNetwork(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)
