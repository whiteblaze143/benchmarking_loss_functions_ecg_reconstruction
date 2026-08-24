"""Backward-compatible entrypoint for the reorganized multi-scale trainer."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from scripts.bootstrap_paths import setup_import_paths

setup_import_paths(include_fairseq=True)

from unified_latents.engineering.trainers.train_multi_scale_vae import *  # noqa: F401,F403


if __name__ == "__main__":
    train(get_args())
