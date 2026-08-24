#!/usr/bin/env python3
"""
Generate 3-Architecture Cross-Family Factorial Queue State.
Architectures: UNet, MS-VAE, ECG-AIM.
Single Consolidated Seed: seed=201 (all 128 UNet runs at seed=201 are completed!).
Total jobs: 128 * 3 = 384 jobs.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUEUE_DIR = ROOT / "refine-logs" / "queue_3arch"
QUEUE_DIR.mkdir(parents=True, exist_ok=True)
QUEUE_STATE_FILE = QUEUE_DIR / "queue_state.json"

# All 128 factorial 7-bit masks (0000000 to 1111111)
def generate_all_masks():
    masks = []
    for i in range(128):
        bits = f"{i:07b}"
        # Always enforce anchor loss bit (bit 0 = 1)
        mask = "1" + bits[1:]
        if mask not in masks:
            masks.append(mask)
    return sorted(masks)

MASKS = generate_all_masks()
ARCHITECTURES = ["unet", "msvae", "ecg_aim"]
SEED = 201

def main():
    jobs = []
    for arch in ARCHITECTURES:
        for mask in MASKS:
            job_id = f"{arch}_f_{mask}_s{SEED}"
            ckpt_path = f"checkpoints/factorial_{arch}_{mask}_s{SEED}.pt"
            
            # If architecture is unet, check if checkpoint already exists from previous runs
            if arch == "unet":
                legacy_ckpt = Path(f"checkpoints/factorial_{mask}_s{SEED}.pt")
                if legacy_ckpt.exists():
                    ckpt_path = str(legacy_ckpt)
                    status = "completed"
                else:
                    status = "pending"
            else:
                status = "pending"
                
            cmd = (
                f"~/.venv/bin/python3 scripts/train_factorial_multimodel.py "
                f"--architecture {arch} --factorial_mask {mask} --seed {SEED} "
                f"--run_name {job_id} --checkpoint_path {ckpt_path}"
            )
            
            jobs.append({
                "id": job_id,
                "phase": "default",
                "cmd": cmd,
                "expected_output": None,
                "status": status,
                "gpu": None,
                "screen_name": f"EQ_{job_id}",
                "pid": None,
                "attempts": 1 if status == "completed" else 0,
                "started": None,
                "completed": None,
                "error": None
            })
            
    queue_state = {
        "version": 1,
        "created_at": "2026-08-07T14:16:00Z",
        "lock": {
          "acquired": None,
          "holder": None
        },
        "recovery": {
          "status": "healthy"
        },
        "phases": [
          {
            "name": "default",
            "depends_on": [],
            "status": "running"
          }
        ],
        "jobs": jobs
    }
    
    with open(QUEUE_STATE_FILE, "w") as f:
        json.dump(queue_state, f, indent=2)
        
    comp_count = sum(1 for j in jobs if j["status"] == "completed")
    pend_count = sum(1 for j in jobs if j["status"] == "pending")
    print(f"Generated 3-Architecture Queue at {QUEUE_STATE_FILE}:")
    print(f"  Total Jobs: {len(jobs)}")
    print(f"  UNet Completed (seed {SEED}): {comp_count}")
    print(f"  MS-VAE & ECG-AIM Pending: {pend_count}")

if __name__ == "__main__":
    main()
