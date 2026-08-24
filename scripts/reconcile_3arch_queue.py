#!/usr/bin/env python3
"""Reconcile 3-architecture queue labels with inference-addressable storage.

This is intentionally fail-closed: it refuses to mutate state while a trainer
is live.  A historical ``completed`` label is retained only when the exact
checkpoint has passed the remote round-trip and payload validation gates.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sqlite3
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.checkpoint_store import archive_row_ready
from scripts.run_3arch_queue import live_training_jobs


DEFAULT_STATE = ROOT / "refine-logs" / "queue_3arch" / "queue_state.json"
DEFAULT_DB = ROOT / "results" / "checkpoint_store" / "catalog.sqlite"
LEDGER_DIR = ROOT / "refine-logs" / "queue_3arch" / "reconciliation"


def now_utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def archive_model_id(job_id):
    if job_id.startswith("msvae_f_"):
        return "factorial_msvae_" + job_id.removeprefix("msvae_f_")
    if job_id.startswith("ecg_aim_f_"):
        return "factorial_ecg_aim_" + job_id.removeprefix("ecg_aim_f_")
    raise ValueError(f"Unsupported 3-architecture job id: {job_id}")


def restore_missing_architecture_pairs(jobs):
    """Restore cells lost when the disk-full JSON was manually truncated."""
    ids = {job["id"] for job in jobs}
    added = []
    for job in list(jobs):
        if not job["id"].startswith("msvae_f_"):
            continue
        suffix = job["id"].removeprefix("msvae_f_")
        expected = "ecg_aim_f_" + suffix
        if expected in ids:
            continue
        mask, seed = suffix.rsplit("_s", 1)
        restored = {
            "id": expected,
            "cmd": (
                "~/.venv/bin/python3 scripts/train_factorial_multimodel.py "
                f"--architecture ecg_aim --batch_size 32 --num_workers 2 "
                f"--factorial_mask {mask} --seed {seed} --run_name {expected} "
                f"--checkpoint_path checkpoints/factorial_ecg_aim_{mask}_s{seed}.pt"
            ),
            "status": "pending",
            "attempts": 0,
            "restored_at": now_utc(),
            "restore_reason": "missing_architecture_pair_after_disk_full_json_repair",
        }
        jobs.append(restored)
        ids.add(expected)
        added.append(expected)
    return added


def atomic_write_json(path, payload):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def ready_archives(db_path):
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute("SELECT * FROM checkpoints").fetchall()
        return {row["model_id"] for row in rows if archive_row_ready(row)}
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    live = live_training_jobs()
    if live:
        raise SystemExit(
            "Refusing reconciliation while trainers are live: " + ", ".join(sorted(live))
        )
    payload = json.loads(args.state.read_text())
    jobs = payload.get("jobs", [])
    restored_jobs = restore_missing_architecture_pairs(jobs)
    if len({job["id"] for job in jobs}) != len(jobs):
        raise RuntimeError("Queue contains duplicate job IDs")
    ready = ready_archives(args.db)
    observed_at = now_utc()
    changes = []

    for job in jobs:
        model_id = archive_model_id(job["id"])
        before = job.get("status")
        if model_id in ready:
            after = "completed"
            job["archive_model_id"] = model_id
            job["archive_verified_at"] = observed_at
        elif before in {"completed", "failed", "running"}:
            after = "pending"
            job["attempts"] = 0
            job["requeued_at"] = observed_at
            job["requeue_reason"] = "checkpoint_not_inference_addressable_after_disk_full"
            for field in (
                "started", "completed", "failed_at", "returncode", "error",
                "checkpoint_size_bytes", "free_bytes_at_start",
            ):
                job.pop(field, None)
        else:
            after = before
        job["status"] = after
        if before != after:
            changes.append({"id": job["id"], "before": before, "after": after})

    report = {
        "schema_version": 1,
        "observed_at": observed_at,
        "state": str(args.state.resolve()),
        "database": str(args.db.resolve()),
        "before_counts": dict(Counter(change["before"] for change in changes)),
        "after_queue_counts": dict(Counter(job["status"] for job in jobs)),
        "inference_addressable_3arch": sorted(
            model_id for model_id in ready
            if model_id.startswith(("factorial_msvae_", "factorial_ecg_aim_"))
        ),
        "changes": changes,
        "restored_jobs": restored_jobs,
        "applied": bool(args.apply),
    }
    print(json.dumps({key: value for key, value in report.items() if key != "changes"}, indent=2))
    if not args.apply:
        return

    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ledger = LEDGER_DIR / f"reconciliation_{stamp}.json"
    atomic_write_json(ledger, report)
    atomic_write_json(args.state, payload)
    print(f"Applied reconciliation; audit ledger: {ledger}")


if __name__ == "__main__":
    main()
