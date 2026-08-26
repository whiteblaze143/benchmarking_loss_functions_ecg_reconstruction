#!/usr/bin/env python3
"""Master chained supervisor: Wavelet 1110000 -> Wavelet 1111002 -> Spatial Arch V1 -> 3-Arch Queue."""

from __future__ import annotations

import json, os, signal, sqlite3, subprocess, sys, time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.wavelet_ssl_queue import (
    atomic_json,
    gpu_snapshot,
    run_queue,
    sha256_file,
    wait_for_resources,
)

PYTHON = Path("/home/mithunmanivannan/.venv/bin/python3")
WORK_1110000 = ROOT / "refine-logs/wavelet_ssl_1110000"
WORK_1111002 = ROOT / "refine-logs/wavelet_ssl_1111002"
WORK_SPATIAL_V1 = ROOT / "refine-logs/spatial_arch_1lead_v1"

STATE_FILE = ROOT / "refine-logs/chained_wavelet_supervisor_state.json"
STOP_FILE = ROOT / "refine-logs/STOP_CHAINED_SUPERVISOR"
BARRIER = ROOT / "refine-logs/queue_3arch/WAVELET_PRIORITY_BARRIER.json"

STOP_REQUESTED = False

def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def request_stop(signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(json.dumps({"event": "stop_requested", "signal": signum, "at": utc_now()}), flush=True)

def update_state(phase: str, **extra: Any) -> None:
    atomic_json(STATE_FILE, {"version": 1, "phase": phase, "updated_at": utc_now(), **extra})

def is_queue_complete(manifest_path: Path, expected_jobs: int) -> bool:
    db_path = manifest_path.parent / "queue.sqlite"
    if not db_path.is_file():
        return False
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        completed = c.execute("SELECT count(*) FROM jobs WHERE status = 'completed'").fetchone()[0]
        conn.close()
        return completed == expected_jobs
    except Exception:
        return False

def wait_for_quiescence(timeout_seconds: int = 1800, poll_seconds: int = 5) -> None:
    for _ in range(timeout_seconds // poll_seconds):
        if STOP_REQUESTED or STOP_FILE.exists():
            raise InterruptedError("stopped while waiting for GPU quiescence")
        snapshot = gpu_snapshot()
        if snapshot["used_memory_mib"] <= 1024 and snapshot["utilization_pct"] <= 5:
            return
        time.sleep(poll_seconds)
    raise TimeoutError("GPU did not become quiescent before timeout")

def run_sweep_phase(mask: str, work_dir: Path, expected_preflight: int = 15, expected_full: int = 120) -> None:
    preflight_manifest = work_dir / "preflight/manifest.json"
    full_manifest = work_dir / "full/manifest.json"

    # Preflight
    if not is_queue_complete(preflight_manifest, expected_preflight):
        print(f"[{utc_now()}] Running preflight for {mask} ({expected_preflight} jobs)...", flush=True)
        update_state(f"running_preflight_{mask}")
        code = run_queue(
            preflight_manifest, project_root=ROOT, max_attempts=2, min_free_gib=8,
            min_available_ram_gib=5, continue_on_error=False, retry_failed=True,
            max_gpu_used_mib=1024, resource_timeout_seconds=21600, job_timeout_seconds=3600,
        )
        if code != 0:
            raise RuntimeError(f"Preflight for {mask} failed with code {code}")
        print(f"[{utc_now()}] Preflight for {mask} complete!", flush=True)
    else:
        print(f"[{utc_now()}] Preflight for {mask} already completed.", flush=True)

    # Full Sweep
    if not is_queue_complete(full_manifest, expected_full):
        print(f"[{utc_now()}] Running full sweep for {mask} ({expected_full} jobs)...", flush=True)
        update_state(f"running_full_sweep_{mask}")
        code = run_queue(
            full_manifest, project_root=ROOT, max_attempts=2, min_free_gib=8,
            min_available_ram_gib=5, continue_on_error=True, max_consecutive_failures=2,
            max_total_failures=5, max_gpu_used_mib=1024,
            resource_timeout_seconds=21600, job_timeout_seconds=14400,
        )
        if code != 0:
            raise RuntimeError(f"Full sweep for {mask} exited with code {code}")
        print(f"[{utc_now()}] Full sweep for {mask} complete!", flush=True)
    else:
        print(f"[{utc_now()}] Full sweep for {mask} already completed.", flush=True)

def main() -> None:
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    print("="*70, flush=True)
    print("  LAUNCHING MASTER CHAINED SUPERVISOR (4-STAGE PIPELINE)", flush=True)
    print("="*70, flush=True)

    # Set barrier for 3-architecture queue
    BARRIER.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(BARRIER, {
        "version": 1,
        "purpose": "wavelet_and_spatial_arch_before_3arch",
        "created_at": utc_now(),
        "supervisor_pid": os.getpid(),
        "sweeps": ["1110000", "1111002", "spatial_arch_1lead_v1"]
    })

    try:
        # Phase 1: 1110000 Sweep (120 models)
        print("\n--- PHASE 1: Wavelet SSL 1110000 (120 Models) ---", flush=True)
        run_sweep_phase("1110000", WORK_1110000, expected_preflight=15, expected_full=120)

        # Strategic Pivot: Freeze 1111002 and Spatial Arch V1 until temporal convergence is confirmed
        print("\n" + "="*70, flush=True)
        print("  PHASE 1 COMPLETE: DIVERTING TO 10-EPOCH CONVERGENCE PANEL PIPELINE", flush=True)
        print("  (Freezing 1111002 and Spatial Arch queues per strategic protocol)", flush=True)
        print("="*70, flush=True)
        
        update_state("launching_convergence_10e_pipeline")
        wait_for_quiescence()
        
        # Invoke convergence pipeline
        conv_script = ROOT / "scripts/pipeline_convergence_10e.py"
        if conv_script.is_file():
            print(f"[{utc_now()}] Executing 10-Epoch Convergence Pipeline: {conv_script}", flush=True)
            subprocess.run([str(PYTHON), str(conv_script), "--auto-launch"], check=True)
        else:
            print(f"[{utc_now()}] Convergence script {conv_script} not yet generated. Waiting for definition.", flush=True)

    except InterruptedError as e:
        update_state("paused", error=str(e))
        print(f"Supervisor paused: {e}", flush=True)
        sys.exit(75)
    except Exception as e:
        update_state("failed", error=f"{type(e).__name__}: {e}")
        print(f"Supervisor failed: {e}", flush=True)
        raise

if __name__ == "__main__":
    main()
