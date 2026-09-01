#!/usr/bin/env python3
"""Resume remaining 4 15-epoch convergence runs:
1. conv15e_tf_sc16_cy4_s42_l0 (from epoch 8/9 rolling resume)
2. conv15e_tf_sc16_cy4_s42_l1
3. conv15e_tf_sc16_cy8_s42_l0
4. conv15e_tf_sc16_cy8_s42_l1
"""

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.wavelet_ssl_queue import run_queue

MANIFEST = ROOT / "refine-logs/convergence_10e/manifest.json"
DB_PATH = ROOT / "refine-logs/convergence_10e/queue.sqlite"

def main():
    print("Resetting remaining 4 tf_sc16 convergence jobs in queue.sqlite...")
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        UPDATE jobs SET status='pending', attempts=0, error=NULL, returncode=NULL, child_pid=NULL 
        WHERE id IN (
            'conv15e_tf_sc16_cy4_s42_l0',
            'conv15e_tf_sc16_cy4_s42_l1',
            'conv15e_tf_sc16_cy8_s42_l0',
            'conv15e_tf_sc16_cy8_s42_l1'
        )
    """)
    con.commit()
    con.close()
    
    print("Launching queue runner for remaining convergence 15e jobs...")
    code = run_queue(
        MANIFEST,
        project_root=ROOT,
        max_attempts=3,
        retry_failed=False,
        min_free_gib=5.0,
        min_available_ram_gib=4.0,
        continue_on_error=True,
        max_consecutive_failures=3,
        max_total_failures=10,
        max_gpu_used_mib=1024,
        resource_timeout_seconds=21600,
        job_timeout_seconds=14400,
    )
    print(f"Convergence 15e resume queue finished with return code {code}")
    return code

if __name__ == "__main__":
    sys.exit(main())
