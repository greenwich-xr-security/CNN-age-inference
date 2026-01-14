from __future__ import annotations

from typing import Tuple

import torch


def _strip_prefix(state_dict: dict, prefix: str) -> dict:
    if not prefix:
        return state_dict
    return {
        (key[len(prefix):] if key.startswith(prefix) else key): value
        for key, value in state_dict.items()
    }


def _extract_backbone_state(checkpoint: dict) -> dict:
    state = checkpoint
    if "backbone" in checkpoint:
        state = checkpoint["backbone"]
    elif "state_dict" in checkpoint:
        state = checkpoint["state_dict"]

    if any(key.startswith("module.") for key in state.keys()):
        state = _strip_prefix(state, "module.")
    if any(key.startswith("backbone.") for key in state.keys()):
        state = _strip_prefix(state, "backbone.")
    return state


def load_dino_backbone(model, checkpoint_path: str, *, strict: bool = False) -> Tuple[list, list]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = _extract_backbone_state(checkpoint)
    missing, unexpected = model.backbone.load_state_dict(state, strict=strict)
    return missing, unexpected
