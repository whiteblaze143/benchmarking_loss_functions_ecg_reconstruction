#!/usr/bin/env python3
"""Crash-safe sequential worker for the priority spatial one-lead screen."""

from __future__ import annotations

import fcntl
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
QUEUE_DIR = ROOT / "refine-logs" / "queue_spatial_1lead"
QUEUE_FILE = QUEUE_DIR / "queue_state.json"
LOCK_FILE = QUEUE_DIR / "queue_worker.lock"
LOG_DIR = QUEUE_DIR / "jobs"
MIN_FREE_BYTES = 4 * 1024**3


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_queue() -> dict:
    return json.loads(QUEUE_FILE.read_text())


def save_queue(data: dict) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=QUEUE_DIR, prefix=".queue_state.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(data, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, QUEUE_FILE)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def singleton_worker():
    with LOCK_FILE.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} started={utc_now()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def checkpoint_path(job: dict) -> Path:
    tokens = shlex.split(job["cmd"])
    value = tokens[tokens.index("--checkpoint_path") + 1]
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def validate_checkpoint(job: dict) -> Path:
    path = checkpoint_path(job)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"missing checkpoint: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload) if isinstance(payload, dict) else None
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"invalid checkpoint state: {path}")
    provenance = payload.get("provenance", {})
    tokens = shlex.split(job["cmd"])
    expected_mask = tokens[tokens.index("--factorial_mask") + 1]
    if provenance.get("factorial_mask") != expected_mask:
        raise RuntimeError(f"wrong factorial mask in {path}")
    if len(provenance.get("preprocessing", {}).get("observed_leads", [])) != 1:
        raise RuntimeError(f"checkpoint is not one-lead: {path}")
    return path


def main() -> None:
    print("Starting priority spatial one-lead queue.", flush=True)
    with singleton_worker():
        while True:
            data = load_queue()
            pending = [job for job in data["jobs"] if job.get("status") == "pending"]
            if not pending:
                print("Priority spatial one-lead queue complete.", flush=True)
                return
            free = shutil.disk_usage(ROOT).free
            if free < MIN_FREE_BYTES:
                print(f"Disk gate: {free / 1024**3:.2f} GiB free; retrying in 60s", flush=True)
                time.sleep(60)
                continue
            job = pending[0]
            job["status"] = "running"
            for stale_field in ("completed", "checkpoint_size_bytes", "failed_at", "returncode", "error"):
                job.pop(stale_field, None)
            job["attempts"] = int(job.get("attempts", 0)) + 1
            job["started"] = utc_now()
            save_queue(data)
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            log_path = LOG_DIR / f"{job['id']}.log"
            print(f"[{utc_now()}] Starting {job['id']}", flush=True)
            with log_path.open("a", buffering=1) as log:
                process = subprocess.Popen(
                    job["cmd"], shell=True, cwd=ROOT, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, bufsize=1,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    sys.stdout.write(line)
                    log.write(line)
                returncode = process.wait()
            if returncode == 0:
                try:
                    path = validate_checkpoint(job)
                except Exception as error:
                    returncode = 70
                    job["error"] = repr(error)
                else:
                    job["status"] = "completed"
                    job["completed"] = utc_now()
                    job["checkpoint_size_bytes"] = path.stat().st_size
            if returncode != 0:
                job["status"] = "failed"
                job["failed_at"] = utc_now()
                job["returncode"] = returncode
                job.setdefault("error", f"training exited with status {returncode}")
            save_queue(data)


if __name__ == "__main__":
    main()
