#!/usr/bin/env python3
"""Guarded spatial -> wavelet (1110000 & 1111002) -> three-architecture queue handoff."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.wavelet_ssl_queue import (  # noqa: E402
    atomic_json,
    gpu_snapshot,
    run_queue,
    sha256_file,
    wait_for_resources,
)


PYTHON = Path("/home/mithunmanivannan/.venv/bin/python3")
SPATIAL_DIR = ROOT / "refine-logs/queue_spatial_1lead"
SPATIAL_STATE = SPATIAL_DIR / "queue_state.json"
SPATIAL_LOCK = SPATIAL_DIR / "queue_worker.lock"
EXPECTED_SPATIAL_JOBS = 84
EXPECTED_SPATIAL_PAIR_SHA256 = "9b6bd307c75dc097f9f4b5c72770c7f15025e01640f5f599e66bfb39be7dd065"
KNOWN_SPATIAL_STATUSES = {
    "pending", "running", "completed", "cancelled", "failed",
    "failed_oom", "failed_transient", "stuck",
}
THREE_ARCH_DIR = ROOT / "refine-logs/queue_3arch"
THREE_ARCH_LOCK = THREE_ARCH_DIR / "queue_worker.lock"
BARRIER = THREE_ARCH_DIR / "WAVELET_PRIORITY_BARRIER.json"

SWEEP_MASKS = ("1110000", "1111002")
MASTER_WORK_DIR = ROOT / "refine-logs/wavelet_ssl_1110000"
SUPERVISOR_LOCK = MASTER_WORK_DIR / "supervisor.lock"
SUPERVISOR_STATE = MASTER_WORK_DIR / "supervisor_state.json"
SUPERVISOR_STOP = MASTER_WORK_DIR / "STOP_SUPERVISOR"

RDB_CACHE = ROOT / "data/rdb_wavelet_delineation_cache"
PTB_MANIFEST = ROOT / "refine-logs/ptbxl_tensor_content_manifest.json"

PREFLIGHT_CELLS = (
    "A0_raw",
    "E1_raw",
    "A0_wave_noSSL_gated_add",
    "A0_wave_noSSL_cross_attn",
    "ssl_magnitude_phase_sin_local_gated_add",
    "ssl_log_magnitude_phase_sin_both_cross_attn",
    "P1_A0_morlet_mag_phase_noSSL",
    "R0_morlet_mag_morlet_phase",
    "R1_morlet_mag_ueg_phase",
    "C1_E1_morlet_mag_morlet_phase",
    "ssl_global",
    "tf_sc48_cy6",
    "conv_control",
    "del_boundary",
    "del_fid",
)
STOP_REQUESTED = False


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def request_stop(signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(json.dumps({"event": "supervisor_stop_requested", "signal": signum, "at": utc_now()}), flush=True)


def update_state(phase: str, **extra: Any) -> None:
    atomic_json(SUPERVISOR_STATE, {"version": 1, "phase": phase, "updated_at": utc_now(), **extra})


@contextmanager
def exclusive_lock(path: Path, label: str) -> Iterator[Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"{label} lock is already held: {path}") from error
        handle.seek(0);handle.truncate();handle.write(f"pid={os.getpid()} at={utc_now()} owner=wavelet_handoff\n");handle.flush()
        try:
            yield handle
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def try_lock(path: Path) -> Any | None:
    handle = path.open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def release_lock(handle: Any) -> None:
    fcntl.flock(handle, fcntl.LOCK_UN)
    handle.close()


def spatial_identity(data: dict[str, Any]) -> str:
    jobs = data.get("jobs", [])
    commands = [str(job.get("command")) for job in jobs]
    return hashlib.sha256("\n".join(commands).encode()).hexdigest()


def relevant_processes() -> list[dict[str, Any]]:
    output = subprocess.run(["ps", "-eo", "pid,args"], check=True, capture_output=True, text=True).stdout
    matches: list[dict[str, Any]] = []
    for line in output.splitlines():
        if "train_1lead_spatial_ecg_aim.py" in line or "run_1lead_spatial_queue.py" in line:
            parts = line.strip().split(maxsplit=1)
            matches.append({"pid": int(parts[0]), "cmd": parts[1]})
    return matches


def wait_for_gpu_quiescence(timeout_seconds: int = 1800, poll_seconds: int = 5) -> None:
    for _ in range(timeout_seconds // poll_seconds):
        if STOP_REQUESTED or SUPERVISOR_STOP.exists():
            raise InterruptedError("stopped while waiting for GPU quiescence")
        snapshot = gpu_snapshot()
        if snapshot["used_memory_mib"] <= 1024 and snapshot["utilization_pct"] <= 5 and not relevant_processes():
            return
        time.sleep(poll_seconds)
    raise TimeoutError("GPU did not become quiescent before timeout")


def validate_spatial_state() -> tuple[dict[str, Any], Counter[str]]:
    if not SPATIAL_STATE.exists():
        return {}, Counter({"completed": EXPECTED_SPATIAL_JOBS})
    data = json.loads(SPATIAL_STATE.read_text())
    jobs = data.get("jobs", [])
    if len(jobs) != EXPECTED_SPATIAL_JOBS or len({job.get("id") for job in jobs}) != EXPECTED_SPATIAL_JOBS:
        raise RuntimeError("spatial queue does not contain the expected 84 unique jobs")
    identity = spatial_identity(data)
    if identity != EXPECTED_SPATIAL_PAIR_SHA256:
        raise RuntimeError(f"spatial job identity changed: {identity}")
    statuses = Counter(str(job.get("status")) for job in jobs)
    unknown = set(statuses) - KNOWN_SPATIAL_STATUSES
    if unknown:
        raise RuntimeError(f"spatial queue has unknown statuses: {sorted(unknown)}")
    return data, statuses


def wait_for_spatial(poll_seconds: int) -> Any:
    while True:
        if STOP_REQUESTED or SUPERVISOR_STOP.exists():
            update_state("paused", reason="supervisor stop requested")
            raise InterruptedError("supervisor stopped while waiting for spatial queue")
        _, statuses = validate_spatial_state()
        active = relevant_processes()
        terminal = statuses.get("pending", 0) == 0 and statuses.get("running", 0) == 0
        if terminal and not active:
            handle = try_lock(SPATIAL_LOCK)
            if handle is not None:
                try:
                    _, rechecked = validate_spatial_state()
                    if rechecked.get("pending", 0) == 0 and rechecked.get("running", 0) == 0 and not relevant_processes():
                        update_state("spatial_complete", spatial_counts=dict(rechecked))
                        return handle
                except BaseException:
                    release_lock(handle)
                    raise
                release_lock(handle)
        update_state("waiting_for_spatial", spatial_counts=dict(statuses), live_processes=active)
        print(json.dumps({"event": "waiting_for_spatial", "counts": dict(statuses), "live": active}), flush=True)
        for _ in range(poll_seconds):
            if STOP_REQUESTED or SUPERVISOR_STOP.exists():
                break
            time.sleep(1)


def verify_ptb_content() -> dict[str, Any]:
    contract = json.loads(PTB_MANIFEST.read_text())
    tensor_root = Path(contract["tensor_root"])
    if not tensor_root.is_absolute():
        tensor_root = ROOT / tensor_root
    result: dict[str, Any] = {}
    for split in ("train", "val"):
        entries = contract["splits"][split]["entries"]
        expected = {entry["relative_path"] for entry in entries}
        actual = {str(path.relative_to(tensor_root)) for path in (tensor_root / split).glob("*.pt")}
        if actual != expected:
            raise RuntimeError(f"PTB-XL {split} inventory differs from its content manifest")
        root_digest = hashlib.sha256()
        for entry in entries:
            path = tensor_root / entry["relative_path"]
            if not path.is_file():
                raise RuntimeError(f"missing PTB-XL tensor: {path}")
            if entry["size_bytes"] != path.stat().st_size:
                raise RuntimeError(f"size mismatch: {path}")
            root_digest.update(f"{entry['relative_path']}:{entry['size_bytes']}:{entry['sha256']}\n".encode())
        computed_root = root_digest.hexdigest()
        declared_root = contract["splits"][split]["content_root_sha256"]
        if computed_root != declared_root:
            raise RuntimeError(f"PTB-XL {split} root digest mismatch")
        result[split] = {"records": len(entries), "root_sha256": computed_root}
    return result


def verify_rdb_cache() -> dict[str, Any]:
    manifest = json.loads((RDB_CACHE / "manifest.json").read_text())
    entries = manifest.get("records", [])
    if len(entries) != 2398:
        raise RuntimeError("RDB cache must contain exactly 2,398 tensor records")
    split_counts = Counter(entry["split"] for entry in entries)
    if split_counts != Counter({"train": 1678, "val": 360, "test": 360}):
        raise RuntimeError(f"unexpected RDB cache split counts: {split_counts}")
    rhythm_counts = Counter(entry["canonical_rhythm"] for entry in entries)
    if len(rhythm_counts) != 8:
        raise RuntimeError("RDB cache must contain 8 canonical rhythm strata")
    seen_patients: set[str] = set()
    split_patients: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for entry in entries:
        path = RDB_CACHE / entry["output"]
        if not path.is_file():
            raise RuntimeError(f"missing RDB tensor: {path}")
        patient = entry.get("patient_id")
        if not patient:
            raise RuntimeError(f"RDB cache entry lacks patient_id: {entry}")
        seen_patients.add(patient)
        split_patients[entry["split"]].add(patient)
    if split_patients["train"] & split_patients["val"]:
        raise RuntimeError("RDB cache train/val patient overlap detected")
    if split_patients["train"] & split_patients["test"] or split_patients["val"] & split_patients["test"]:
        raise RuntimeError("RDB cache test patient overlap detected")
    return {"records": len(entries), "splits": dict(split_counts), "unique_patients": len(seen_patients)}


def replace_option(command: list[str], flag: str, value: str) -> None:
    for index, part in enumerate(command):
        if part == flag and index + 1 < len(command):
            command[index + 1] = value
            return
    command.extend([flag, value])


def ensure_preflight_manifest(work_dir: Path) -> dict[str, Any]:
    preflight_manifest_path = work_dir / "preflight/manifest.json"
    full_manifest_path = work_dir / "full/manifest.json"
    if preflight_manifest_path.is_file():
        return json.loads(preflight_manifest_path.read_text())
    preflight_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    full = json.loads(full_manifest_path.read_text())
    full_sha = sha256_file(full_manifest_path)
    selected: dict[str, dict[str, Any]] = {}
    for job in full["jobs"]:
        name = job["cell"]["name"]
        if name in PREFLIGHT_CELLS and job["id"].endswith("_l0") and name not in selected:
            selected[name] = job
    if set(selected) != set(PREFLIGHT_CELLS):
        raise RuntimeError(f"full manifest lacks preflight cells: {sorted(set(PREFLIGHT_CELLS)-set(selected))}")
    jobs = []
    for ordinal, cell_name in enumerate(PREFLIGHT_CELLS):
        source = selected[cell_name]
        command = list(source["command"])
        run_name = f"preflight_{cell_name}_{ordinal:02d}"
        replace_option(command, "--run-name", run_name)
        replace_option(command, "--output-dir", str(work_dir / "preflight/runs" / run_name))
        command.append("--quick-verify")
        jobs.append({"id": run_name, "status": "pending", "command": command, "cell": source["cell"]})
    payload = {
        key: value for key, value in full.items()
        if key not in {"created_at", "cells", "jobs", "delineation_audit"}
    }
    payload.update({
        "version": 2, "created_at": utc_now(), "cells": len(jobs), "jobs": jobs,
        "source_manifest_sha256": full_sha,
        "gate": "fifteen one-epoch/two-batch GPU branch-coverage configurations",
    })
    atomic_json(preflight_manifest_path, payload)
    return payload


def verify_queue_success(manifest_path: Path, expected_jobs: int) -> None:
    marker = manifest_path.parent / "_QUEUE_SUCCESS.json"
    if not marker.is_file():
        raise RuntimeError(f"queue success marker is missing: {marker}")
    payload = json.loads(marker.read_text())
    if payload.get("manifest_sha256") != sha256_file(manifest_path):
        raise RuntimeError("queue success marker has the wrong manifest identity")
    connection = sqlite3.connect(manifest_path.parent / "queue.sqlite")
    try:
        counts = dict(connection.execute("SELECT status,count(*) FROM jobs GROUP BY status"))
        if counts != {"completed": expected_jobs}:
            raise RuntimeError(f"queue DB is incomplete: {counts}")
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("queue SQLite integrity check failed")
    finally:
        connection.close()


def max_peak_memory(manifest_path: Path) -> int:
    connection = sqlite3.connect(manifest_path.parent / "queue.sqlite")
    try:
        rows = connection.execute("SELECT summary_json FROM jobs WHERE status='completed'").fetchall()
        return max(int(json.loads(row[0])["peak_gpu_memory_bytes"]) for row in rows)
    finally:
        connection.close()


def run_sweep(mask: str, ptb: dict[str, Any], rdb: dict[str, Any]) -> None:
    work_dir = ROOT / f"refine-logs/wavelet_ssl_{mask}"
    full_manifest = work_dir / "full/manifest.json"
    preflight_manifest = work_dir / "preflight/manifest.json"
    
    ensure_preflight_manifest(work_dir)
    
    # Check if preflight is already done
    preflight_success_marker = work_dir / "preflight/_QUEUE_SUCCESS.json"
    if not preflight_success_marker.is_file():
        baseline = gpu_snapshot()
        update_state(f"running_preflight_{mask}", ptb=ptb, rdb=rdb, ecc_baseline=baseline)
        preflight_code = run_queue(
            preflight_manifest, project_root=ROOT, max_attempts=2, min_free_gib=8,
            min_available_ram_gib=5, continue_on_error=False, retry_failed=True,
            max_gpu_used_mib=1024,
            resource_timeout_seconds=21600, job_timeout_seconds=3600,
        )
        if preflight_code != 0:
            raise RuntimeError(f"wavelet GPU preflight for {mask} exited {preflight_code}")
        verify_queue_success(preflight_manifest, len(PREFLIGHT_CELLS))
        peak = max_peak_memory(preflight_manifest)
        if peak > 36 * 1024**3:
            raise RuntimeError(f"preflight peak GPU memory exceeded 36 GiB: {peak}")
        after_preflight = gpu_snapshot()
        if after_preflight["ecc_uncorrected"] > baseline["ecc_uncorrected"]:
            raise RuntimeError(f"uncorrected GPU ECC increased across preflight {mask}: {baseline} -> {after_preflight}")
        update_state(
            f"preflight_complete_{mask}", ptb=ptb, rdb=rdb, peak_gpu_memory_bytes=peak,
            corrected_ecc_delta=max(0, after_preflight["ecc_corrected"]-baseline["ecc_corrected"]),
            uncorrected_ecc_delta=max(0, after_preflight["ecc_uncorrected"]-baseline["ecc_uncorrected"]),
        )
    else:
        print(f"Preflight for {mask} already complete, proceeding to full sweep.", flush=True)

    full = json.loads(full_manifest.read_text())
    full_success_marker = work_dir / "full/_QUEUE_SUCCESS.json"
    if not full_success_marker.is_file():
        update_state(f"running_full_sweep_{mask}")
        full_code = run_queue(
            full_manifest, project_root=ROOT, max_attempts=2, min_free_gib=8,
            min_available_ram_gib=5, continue_on_error=True, max_consecutive_failures=2,
            max_total_failures=5, max_gpu_used_mib=1024,
            resource_timeout_seconds=21600, job_timeout_seconds=14400,
        )
        if full_code != 0:
            raise RuntimeError(f"wavelet full sweep {mask} exited {full_code}")
        verify_queue_success(full_manifest, len(full["jobs"]))
    else:
        print(f"Full sweep for {mask} already complete.", flush=True)


def run_all_wavelet_sweeps() -> None:
    update_state("verifying_inputs")
    ptb = verify_ptb_content()
    rdb = verify_rdb_cache()
    env = os.environ.copy();env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    subprocess.run([
        str(PYTHON), "-m", "py_compile",
        "unified_latents/engineering/experimental/wavelet_ssl_ecg_aim.py",
        "scripts/train_1lead_wavelet_ssl_mtl.py", "scripts/wavelet_ssl_queue.py",
    ], cwd=ROOT, env=env, check=True)
    subprocess.run([
        str(PYTHON), "unified_latents/engineering/experimental/wavelet_ssl_ecg_aim.py",
        "--self-test", "--device", "cpu",
    ], cwd=ROOT, env=env, check=True)

    for mask in SWEEP_MASKS:
        print(f"===========================================================", flush=True)
        print(f"  EXECUTING WAVELET SWEEP: {mask}", flush=True)
        print(f"===========================================================", flush=True)
        run_sweep(mask, ptb, rdb)


def ensure_barrier() -> None:
    payload = {
        "version": 1, "purpose": "spatial_then_wavelet_before_3arch",
        "created_or_refreshed_at": utc_now(), "supervisor_pid": os.getpid(),
        "expected_spatial_pair_sha256": EXPECTED_SPATIAL_PAIR_SHA256,
        "sweeps": list(SWEEP_MASKS),
    }
    atomic_json(BARRIER, payload)


def remove_barrier() -> None:
    if BARRIER.exists():
        BARRIER.unlink()
        directory_fd = os.open(BARRIER.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--prepare-only", action="store_true")
    result.add_argument("--poll-seconds", type=int, default=60)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.poll_seconds < 1:
        raise ValueError("--poll-seconds must be positive")
    
    for mask in SWEEP_MASKS:
        ensure_preflight_manifest(ROOT / f"refine-logs/wavelet_ssl_{mask}")

    if args.prepare_only:
        print(json.dumps({"status": "prepared", "sweeps": list(SWEEP_MASKS)}, indent=2))
        return

    signal.signal(signal.SIGINT, request_stop);signal.signal(signal.SIGTERM, request_stop)
    resume_three_arch = False
    with exclusive_lock(SUPERVISOR_LOCK, "wavelet supervisor"):
        ensure_barrier()
        with exclusive_lock(THREE_ARCH_LOCK, "3-architecture worker"):
            spatial_handle = None
            try:
                update_state("waiting_for_spatial")
                spatial_handle = wait_for_spatial(args.poll_seconds)
                wait_for_gpu_quiescence()
                wait_for_resources(
                    ROOT, 8, 5, 1024, 21600,
                    cancelled=lambda: STOP_REQUESTED or SUPERVISOR_STOP.exists(),
                )
                run_all_wavelet_sweeps()
                update_state("wavelet_all_sweeps_complete", sweeps=list(SWEEP_MASKS))
                remove_barrier()
                resume_three_arch = True
            except InterruptedError as error:
                update_state("paused", error=str(error), barrier_retained=True)
                raise SystemExit(75)
            except BaseException as error:
                update_state(
                    "failed", error=f"{type(error).__name__}: {error}", barrier_retained=True,
                )
                raise
            finally:
                if spatial_handle is not None:
                    release_lock(spatial_handle)
    if resume_three_arch:
        update_state("handoff_to_3arch", barrier_retained=False)
        os.execv(str(PYTHON), [str(PYTHON), str(ROOT / "scripts/run_3arch_queue.py")])


if __name__ == "__main__":
    main()
