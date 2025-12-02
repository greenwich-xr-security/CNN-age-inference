"""Backward-compatible shim; dataset utilities moved to dataset.hand_metadata."""
from dataset.hand_metadata import *  # noqa: F401,F403

# Keep CLI functionality available when invoking this module directly.
if __name__ == "__main__":  # pragma: no cover
    from dataset.hand_metadata import _cli_main

    _cli_main()
