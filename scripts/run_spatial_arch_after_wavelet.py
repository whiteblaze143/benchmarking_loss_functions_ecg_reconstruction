#!/usr/bin/env python3
"""Run Spatial Architecture Search V1 after Wavelet sweeps complete."""

from __future__ import annotations

import json, sys, time, sqlite3, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.wavelet_ssl_queue import run_queue, wait_for_resources, sha256_file

WORK_DIR = ROOT / "refine-logs/spatial_arch_1lead_v1"
PREFLIGHT_MANIFEST = WORK_DIR / "preflight/manifest.json"
FULL_MANIFEST = WORK_DIR / "full/manifest.json"

def main():
    print("="*70)
    print("  EXECUTING SPATIAL ARCHITECTURE SEARCH V1 (48 LEAD-I CELLS)")
    print("="*70)

    # 1. Run Preflight (8 jobs)
    print("\n--- PHASE 1: Running Preflight (8 Representative Cells) ---")
    code = run_queue(
        PREFLIGHT_MANIFEST,
        project_root=ROOT,
        max_attempts=2,
        min_free_gib=8,
        min_available_ram_gib=5,
        continue_on_error=False,
        retry_failed=True,
        max_gpu_used_mib=1024,
        resource_timeout_seconds=21600,
        job_timeout_seconds=3600
    )
    if code != 0:
        raise RuntimeError(f"Spatial architecture preflight exited with error code {code}")
    print("Preflight complete and verified successfully!")

    # 2. Run Full 48-cell Screen
    print("\n--- PHASE 2: Running Full 48-Cell Lead-I Screen (3 Epochs) ---")
    code = run_queue(
        FULL_MANIFEST,
        project_root=ROOT,
        max_attempts=2,
        min_free_gib=8,
        min_available_ram_gib=5,
        continue_on_error=True,
        max_consecutive_failures=2,
        max_total_failures=5,
        max_gpu_used_mib=1024,
        resource_timeout_seconds=21600,
        job_timeout_seconds=14400
    )
    if code != 0:
        raise RuntimeError(f"Spatial architecture full sweep exited with code {code}")
    print("\n" + "="*70)
    print("  SPATIAL ARCHITECTURE SEARCH V1 COMPLETED!")
    print("="*70)

if __name__ == "__main__":
    main()
