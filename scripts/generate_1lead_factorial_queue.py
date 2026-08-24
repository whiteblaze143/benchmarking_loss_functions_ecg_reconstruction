#!/usr/bin/env python3
"""
Generate 1-Lead Factorial Queue State.
Architectures: UNet, MS-VAE, ECG-AIM.
Modalities: Lead I (0) and Lead II (1).
Masks: Pure MSE (1000000), +Temp MMD (1100000), +Deriv L1 (1001000), +Kors (1000100).
Seed: 201
Total jobs: 3 * 2 * 4 = 24 jobs.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUEUE_DIR = ROOT / "refine-logs" / "queue_1lead"
QUEUE_DIR.mkdir(parents=True, exist_ok=True)
QUEUE_STATE_FILE = QUEUE_DIR / "queue_state.json"

MASKS = ["1000000", "1100000", "1001000", "1000100"]
ARCHITECTURES = ["unet", "msvae", "ecg_aim"]
OBSERVED_LEADS_OPTIONS = [[0], [1]]
SEED = 201

def main():
    jobs = []
    for arch in ARCHITECTURES:
        for obs in OBSERVED_LEADS_OPTIONS:
            obs_str = "_".join(map(str, obs))
            for mask in MASKS:
                job_id = f"1lead_{arch}_f_{mask}_s{SEED}_l{obs_str}"
                ckpt_path = f"checkpoints/1lead_{arch}_f_{mask}_s{SEED}_l{obs_str}.pt"
                
                obs_args = " ".join(map(str, obs))
                cmd = (
                    f"~/.venv/bin/python3 scripts/train_1lead_factorial_multimodel.py "
                    f"--architecture {arch} --factorial_mask {mask} --seed {SEED} "
                    f"--observed_leads {obs_args} "
                    f"--run_name {job_id} --checkpoint_path {ckpt_path}"
                )
                
                jobs.append({
                    "id": job_id,
                    "phase": "default",
                    "cmd": cmd,
                    "expected_output": None,
                    "status": "pending",
                    "gpu": None,
                    "screen_name": f"EQ_{job_id}",
                    "pid": None,
                    "attempts": 0,
                    "started": None,
                    "completed": None,
                    "error": None
                })
                
    queue_state = {
        "version": 1,
        "created_at": "2026-08-20T00:00:00Z",
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
        
    print(f"Generated 1-Lead Queue at {QUEUE_STATE_FILE}:")
    print(f"  Total Jobs: {len(jobs)}")

if __name__ == "__main__":
    main()
