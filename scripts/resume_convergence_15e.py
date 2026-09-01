#!/usr/bin/env python3
"""Robust runner for remaining 3 15-epoch convergence runs with transient error auto-recovery:
1. conv15e_tf_sc16_cy4_s42_l1 (resuming from epoch 2/3 checkpoint)
2. conv15e_tf_sc16_cy8_s42_l0
3. conv15e_tf_sc16_cy8_s42_l1
"""

import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.wavelet_ssl_queue import run_queue

MANIFEST = ROOT / "refine-logs/convergence_10e/manifest.json"
DB_PATH = ROOT / "refine-logs/convergence_10e/queue.sqlite"

TARGET_JOBS = [
    'conv15e_tf_sc16_cy4_s42_l1',
    'conv15e_tf_sc16_cy8_s42_l0',
    'conv15e_tf_sc16_cy8_s42_l1',
]

def reset_remaining_jobs():
    con = sqlite3.connect(DB_PATH)
    placeholders = ",".join(f"'{j}'" for j in TARGET_JOBS)
    con.execute(f"""
        UPDATE jobs SET status='pending', attempts=0, error=NULL, returncode=NULL, child_pid=NULL 
        WHERE id IN ({placeholders}) AND status != 'completed'
    """)
    con.commit()
    con.close()

def remaining_count():
    con = sqlite3.connect(DB_PATH)
    placeholders = ",".join(f"'{j}'" for j in TARGET_JOBS)
    cnt = con.execute(f"""
        SELECT count(*) FROM jobs 
        WHERE id IN ({placeholders}) AND status != 'completed'
    """).fetchone()[0]
    con.close()
    return cnt

def main():
    max_outer_loops = 15
    for attempt in range(1, max_outer_loops + 1):
        rem = remaining_count()
        if rem == 0:
            print("All 3 remaining convergence 15e jobs are completed successfully!")
            return 0
        
        print(f"\n=======================================================")
        print(f"Convergence 15e Supervisor: Loop {attempt}/{max_outer_loops} ({rem} jobs remaining)")
        print(f"=======================================================")
        reset_remaining_jobs()
        
        code = run_queue(
            MANIFEST,
            project_root=ROOT,
            max_attempts=3,
            retry_failed=False,
            min_free_gib=5.0,
            min_available_ram_gib=4.0,
            continue_on_error=True,
            max_consecutive_failures=3,
            max_total_failures=15,
            max_gpu_used_mib=1024,
            resource_timeout_seconds=21600,
            job_timeout_seconds=14400,
        )
        print(f"Queue pass finished with code {code}.")
        
        if remaining_count() == 0:
            print("All jobs completed!")
            return 0
        
        print("Cooling down for 15 seconds before next retry pass...")
        time.sleep(15)
        
    return 1

if __name__ == "__main__":
    sys.exit(main())
