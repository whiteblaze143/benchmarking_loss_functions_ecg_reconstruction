#!/usr/bin/env python3
"""
Prune non-seed-201 checkpoints to free disk space.
Retains all seed 201 checkpoints and baseline checkpoints needed for registry.
"""

import os
from pathlib import Path

CHECKPOINT_DIR = Path("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/checkpoints")

def main():
    if not CHECKPOINT_DIR.exists():
        print(f"Checkpoint directory {CHECKPOINT_DIR} does not exist.")
        return

    all_ckpts = list(CHECKPOINT_DIR.glob("*.pt"))
    print(f"Total checkpoints found in {CHECKPOINT_DIR}: {len(all_ckpts)}")

    freed_bytes = 0
    deleted_count = 0

    for ckpt in all_ckpts:
        name = ckpt.name
        # Keep seed 201 checkpoints (e.g. *_s201.pt)
        if "_s201.pt" in name:
            continue
        # Keep non-factorial baseline checkpoints if any
        if not ("_s" in name and name.endswith(".pt")):
            continue

        size = ckpt.stat().st_size
        try:
            ckpt.unlink()
            freed_bytes += size
            deleted_count += 1
        except Exception as e:
            print(f"Failed to delete {ckpt}: {e}")

    freed_gb = freed_bytes / (1024 ** 3)
    print(f"Deleted {deleted_count} non-201 seed checkpoints.")
    print(f"Freed {freed_gb:.2f} GB of disk space.")

if __name__ == "__main__":
    main()
