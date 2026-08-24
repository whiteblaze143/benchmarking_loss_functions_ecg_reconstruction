#!/usr/bin/env python3
"""Crash-safe sequential worker for the three-architecture factorial queue."""

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
QUEUE_DIR = ROOT / "refine-logs" / "queue_3arch"
QUEUE_FILE = QUEUE_DIR / "queue_state.json"
LOCK_FILE = QUEUE_DIR / "queue_worker.lock"
WAVELET_PRIORITY_BARRIER = QUEUE_DIR / "WAVELET_PRIORITY_BARRIER.json"
LOG_DIR = QUEUE_DIR / "jobs"
MIN_FREE_BYTES = 4 * 1024**3
DISK_POLL_SECONDS = 60


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_queue():
    return json.loads(QUEUE_FILE.read_text())


def save_queue(qdata):
    """Atomically replace state so ENOSPC cannot leave truncated JSON."""
    QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=QUEUE_FILE.parent,
            prefix=f".{QUEUE_FILE.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(qdata, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, QUEUE_FILE)
        directory_fd = os.open(QUEUE_FILE.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


@contextmanager
def singleton_worker():
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_FILE.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another 3-architecture queue worker holds the lock") from error
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} started={utc_now()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def live_training_jobs() -> set[str]:
    jobs = set()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = proc.joinpath("cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        decoded = [value.decode(errors="replace") for value in args if value]
        if not any("train_factorial_multimodel.py" in value for value in decoded):
            continue
        if "--run_name" in decoded:
            index = decoded.index("--run_name")
            if index + 1 < len(decoded):
                jobs.add(decoded[index + 1])
    return jobs


def reconcile_orphaned_running(qdata):
    running = [job for job in qdata.get("jobs", []) if job.get("status") == "running"]
    if not running:
        return False
    live = live_training_jobs()
    still_live = [job["id"] for job in running if job["id"] in live]
    if still_live:
        raise RuntimeError(
            "refusing to start beside live queue trainers: " + ", ".join(still_live)
        )
    for job in running:
        job["status"] = "pending"
        job["recovered_at"] = utc_now()
        job["recovery_reason"] = "orphaned_running_state_without_live_training_process"
        job.pop("started", None)
    save_queue(qdata)
    print(f"Recovered {len(running)} orphaned running job(s) to pending.", flush=True)
    return True


def wait_for_disk_headroom():
    while True:
        free = shutil.disk_usage(ROOT).free
        if free >= MIN_FREE_BYTES:
            return free
        print(
            f"Disk gate waiting: {free / 1024**3:.2f} GiB free; "
            f"requires {MIN_FREE_BYTES / 1024**3:.2f} GiB.",
            flush=True,
        )
        time.sleep(DISK_POLL_SECONDS)


def checkpoint_path(job) -> Path:
    tokens = shlex.split(job["cmd"])
    try:
        value = tokens[tokens.index("--checkpoint_path") + 1]
    except (ValueError, IndexError) as error:
        raise RuntimeError(f"{job['id']} command lacks --checkpoint_path") from error
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def validate_checkpoint(job):
    path = checkpoint_path(job)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"expected checkpoint was not created: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload) if isinstance(payload, dict) else None
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"checkpoint has no non-empty model state: {path}")
    if not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise RuntimeError(f"checkpoint state contains non-tensor values: {path}")
    return path


def run_job(job) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{job['id']}.log"
    with log_path.open("a", buffering=1) as log:
        log.write(f"\n[{utc_now()}] attempt={job['attempts']} cmd={job['cmd']}\n")
        process = subprocess.Popen(
            job["cmd"],
            shell=True,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
        return process.wait()


def main():
    if WAVELET_PRIORITY_BARRIER.exists():
        print(
            f"Wavelet priority barrier is active at {WAVELET_PRIORITY_BARRIER}; "
            "the 3-architecture queue remains paused.",
            flush=True,
        )
        raise SystemExit(75)
    print("Starting 3-Architecture Queue Worker...", flush=True)
    with singleton_worker():
        qdata = load_queue()
        reconcile_orphaned_running(qdata)
        while True:
            qdata = load_queue()
            jobs = qdata.get("jobs", [])
            pending_jobs = [job for job in jobs if job.get("status") == "pending"]
            if not pending_jobs:
                print("No pending jobs remain in the 3-architecture queue.", flush=True)
                break

            free_bytes = wait_for_disk_headroom()
            job = pending_jobs[0]
            job_id = job["id"]
            job["status"] = "running"
            job["attempts"] = int(job.get("attempts", 0)) + 1
            job["started"] = utc_now()
            job["free_bytes_at_start"] = free_bytes
            job.pop("error", None)
            save_queue(qdata)
            print(f"\n[{utc_now()}] Starting Job: {job_id}", flush=True)

            returncode = run_job(job)
            if returncode == 0:
                try:
                    path = validate_checkpoint(job)
                except Exception as error:
                    returncode = 70
                    job["error"] = f"checkpoint_validation: {error!r}"
                else:
                    job["status"] = "completed"
                    job["completed"] = utc_now()
                    job["checkpoint_size_bytes"] = path.stat().st_size
                    print(f"Job {job_id} COMPLETED SUCCESSFULLY.", flush=True)
            if returncode != 0:
                job["status"] = "failed"
                job["failed_at"] = utc_now()
                job["returncode"] = returncode
                job.setdefault("error", f"training exited with status {returncode}")
                print(f"Job {job_id} FAILED with exit code {returncode}", flush=True)
            save_queue(qdata)


if __name__ == "__main__":
    main()
