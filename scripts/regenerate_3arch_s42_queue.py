#!/usr/bin/env python3
"""
Generate a 3-Architecture Cross-Family Factorial Queue State.
Uses seed 42 as the canonical completed UNet seed.
UNet jobs are automatically skipped as they are already remote_verified in catalog.
Generates jobs for MS-VAE and ECG-AIM for all 160 masks.
"""

import json
import itertools
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUEUE_FILE = ROOT / "refine-logs" / "queue_3arch" / "queue_state.json"
MANIFEST_FILE = ROOT / "refine-logs" / "factorial_manifest.json"

def main():
    QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
    masks = ['1000000', '1000001', '1000002', '1000003', '1000004', '1000010', '1000011', '1000012', '1000013', '1000014', '1000100', '1000101', '1000102', '1000103', '1000104', '1000110', '1000111', '1000112', '1000113', '1000114', '1001000', '1001001', '1001002', '1001003', '1001004', '1001010', '1001011', '1001100', '1001101', '1001102', '1001103', '1001104', '1010000', '1010001', '1010002', '1010003', '1010004', '1010100', '1010101', '1010102', '1010103', '1010104', '1011000', '1011001', '1011002', '1011003', '1011004', '1011100', '1011101', '1011102', '1011103', '1011104', '1100000', '1100001', '1100002', '1100003', '1100004', '1100100', '1100101', '1100102', '1100103', '1100104', '1101000', '1101001', '1101002', '1101003', '1101004', '1101100', '1101101', '1101102', '1101103', '1101104', '1110000', '1110001', '1110002', '1110003', '1110004', '1110100', '1110101', '1110102', '1110103', '1110104', '1111000', '1111001', '1111002', '1111003', '1111004', '1111100', '1111101', '1111102', '1111103', '1111104', '1001012', '1001013', '1001014', '1001110', '1001111', '1001112', '1001113', '1001114', '1010010', '1010011', '1010012', '1010111', '1011012', '1011013', '1011014', '1010013', '1010014', '1010110', '1010112', '1010113', '1010114', '1011010', '1011011', '1011110', '1011111', '1011112', '1011113', '1011114', '1100010', '1100011', '1100012', '1100013', '1100014', '1100110', '1100111', '1100112', '1100113', '1100114', '1101010', '1101011', '1101012', '1101013', '1101014', '1101110', '1101111', '1101112', '1101113', '1101114', '1110010', '1110011', '1110012', '1110013', '1110014', '1110110', '1110111', '1110112', '1110113', '1110114', '1111010', '1111011', '1111012', '1111013', '1111014', '1111110', '1111111', '1111112', '1111113', '1111114']
    
    jobs = []
    
    # 1. MS-VAE (Optimized: batch_size=64, num_workers=2)
    for mask in masks:
        job_id = f"msvae_f_{mask}_s42"
        jobs.append({
            "id": job_id,
            "cmd": f"~/.venv/bin/python3 scripts/train_factorial_multimodel.py --architecture msvae --batch_size 64 --num_workers 2 --factorial_mask {mask} --seed 42 --run_name msvae_f_{mask}_s42 --checkpoint_path checkpoints/factorial_msvae_{mask}_s42.pt",
            "status": "pending",
            "attempts": 0
        })

    # 2. ECG-AIM (Optimized: batch_size=32, num_workers=2)
    for mask in masks:
        job_id = f"ecg_aim_f_{mask}_s42"
        jobs.append({
            "id": job_id,
            "cmd": f"~/.venv/bin/python3 scripts/train_factorial_multimodel.py --architecture ecg_aim --batch_size 32 --num_workers 2 --factorial_mask {mask} --seed 42 --run_name ecg_aim_f_{mask}_s42 --checkpoint_path checkpoints/factorial_ecg_aim_{mask}_s42.pt",
            "status": "pending",
            "attempts": 0
        })

    queue_data = {
        "version": 1,
        "jobs": jobs
    }

    with open(QUEUE_FILE, "w") as f:
        json.dump(queue_data, f, indent=2)

    print(f"Generated 3-Architecture Factorial Queue for seed 42: {len(jobs)} pending jobs.")
    print("UNet baseline jobs skipped (already completed and stored remotely).")

if __name__ == "__main__":
    main()
