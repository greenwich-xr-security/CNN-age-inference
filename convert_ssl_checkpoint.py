"""Convert a DINO/BYOL SSL checkpoint into an age-regressor checkpoint for --init-checkpoint.

SSL checkpoints (train_dino.py / train_byol.py) only contain a feature-only
backbone (plus a projection head with no age-regressor counterpart). This
script builds a fresh age-regressor of the matching architecture, loads only
the backbone weights into it (leaving the age head randomly initialized),
and saves a full model state dict compatible with train_distributed.py's
--init-checkpoint (which loads with strict=True).
"""
import argparse
from pathlib import Path

import torch

from models import resolve_model_builder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ssl_checkpoint", type=str, help="Path to a DINO/BYOL SSL checkpoint (.pth).")
    parser.add_argument("output", type=str, help="Path to write the age-regressor init checkpoint (.pth).")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Backbone variant, e.g. v2_s. Defaults to the 'model' field stored in the SSL checkpoint.",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=0,
        help="Optional age-regressor embedding head dimension; must match downstream fine-tuning.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ssl_checkpoint_path = Path(args.ssl_checkpoint).expanduser()
    ssl_ckpt = torch.load(ssl_checkpoint_path, map_location="cpu", weights_only=False)

    if "backbone" not in ssl_ckpt:
        raise ValueError(f"'{ssl_checkpoint_path}' has no 'backbone' key; is this an SSL checkpoint?")

    model_name = args.model or ssl_ckpt.get("model")
    if not model_name:
        raise ValueError("--model not given and the SSL checkpoint has no 'model' field.")

    model_builder, _, model_desc, _ = resolve_model_builder(model_name, embed_dim=args.embed_dim)
    age_model = model_builder()

    missing, unexpected = age_model.load_state_dict(ssl_ckpt["backbone"], strict=False)
    print(f"Loaded backbone weights from {ssl_checkpoint_path} into fresh {model_desc}.")
    print(f"  Keys left at random init (expected: age head / classifier): {sorted(missing)}")
    if unexpected:
        raise RuntimeError(
            f"Unexpected keys in SSL backbone state dict that don't match {model_desc}: {sorted(unexpected)}"
        )
    if not any(key.startswith("classifier") for key in missing):
        raise RuntimeError(
            "Expected the age classifier head to be among the missing (randomly-initialized) keys, "
            f"but it wasn't. Missing keys were: {sorted(missing)}"
        )

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(age_model.state_dict(), output_path)
    print(f"Saved init-checkpoint-compatible state dict to {output_path}")


if __name__ == "__main__":
    main()
