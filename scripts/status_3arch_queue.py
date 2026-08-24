#!/usr/bin/env python3
"""Read-only integrity/status report for the three-architecture queue."""

from __future__ import annotations

import json
import shutil
import sqlite3
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.checkpoint_store import archive_row_ready
from scripts.reconcile_3arch_queue import archive_model_id
from scripts.run_3arch_queue import live_training_jobs


STATE = ROOT / "refine-logs" / "queue_3arch" / "queue_state.json"
DB = ROOT / "results" / "checkpoint_store" / "catalog.sqlite"


def main():
    payload = json.loads(STATE.read_text())
    jobs = payload.get("jobs", [])
    ids = [job["id"] for job in jobs]
    live = live_training_jobs()
    connection = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        integrity = connection.execute("PRAGMA quick_check").fetchone()[0]
        ready = {
            row["model_id"]
            for row in connection.execute("SELECT * FROM checkpoints")
            if archive_row_ready(row)
        }
    finally:
        connection.close()

    msvae_suffixes = {
        value.removeprefix("msvae_f_")
        for value in ids if value.startswith("msvae_f_")
    }
    ecg_suffixes = {
        value.removeprefix("ecg_aim_f_")
        for value in ids if value.startswith("ecg_aim_f_")
    }
    completed = [job for job in jobs if job.get("status") == "completed"]
    unbacked = [
        job["id"] for job in completed
        if archive_model_id(job["id"]) not in ready
    ]
    running = {job["id"] for job in jobs if job.get("status") == "running"}
    disk = shutil.disk_usage(ROOT)
    report = {
        "state_valid": True,
        "jobs": len(jobs),
        "unique_job_ids": len(set(ids)),
        "status_counts": dict(Counter(job.get("status", "missing") for job in jobs)),
        "architecture_counts": {
            "msvae": len(msvae_suffixes),
            "ecg_aim": len(ecg_suffixes),
        },
        "missing_ecg_aim_pairs": sorted(msvae_suffixes - ecg_suffixes),
        "missing_msvae_pairs": sorted(ecg_suffixes - msvae_suffixes),
        "live_training_jobs": sorted(live),
        "state_running_without_process": sorted(running - live),
        "process_without_state_running": sorted(live - running),
        "inference_addressable_3arch": sum(
            model_id.startswith(("factorial_msvae_", "factorial_ecg_aim_"))
            for model_id in ready
        ),
        "completed_without_inference_checkpoint_count": len(unbacked),
        "completed_without_inference_checkpoint_examples": unbacked[:20],
        "checkpoint_catalog_integrity": integrity,
        "disk_free_gib": round(disk.free / 1024**3, 3),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
