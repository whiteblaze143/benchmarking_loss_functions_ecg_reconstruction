#!/usr/bin/env python3
"""Guarded spatial -> wavelet -> three-architecture queue handoff."""

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
WORK_DIR = ROOT / "refine-logs/wavelet_ssl_1110000"
SUPERVISOR_LOCK = WORK_DIR / "supervisor.lock"
SUPERVISOR_STATE = WORK_DIR / "supervisor_state.json"
SUPERVISOR_STOP = WORK_DIR / "STOP_SUPERVISOR"
FULL_MANIFEST = WORK_DIR / "full/manifest.json"
PREFLIGHT_MANIFEST = WORK_DIR / "preflight/manifest.json"
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
    pairs = sorted([[job.get("id"), job.get("cmd")] for job in data.get("jobs", [])])
    raw = json.dumps(pairs, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def relevant_processes() -> list[dict[str, Any]]:
    targets = {
        "run_spatial_1lead_queue.py", "train_1lead_spatial_ecg_aim.py",
        "run_spatial_1lead_then_resume_3lead.sh", "run_3arch_queue.py",
    }
    found = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = [part.decode(errors="replace") for part in proc.joinpath("cmdline").read_bytes().split(b"\0") if part]
        except (OSError, ProcessLookupError):
            continue
        matches = sorted({Path(arg).name for arg in args if Path(arg).name in targets})
        if matches:
            found.append({"pid": int(proc.name), "matches": matches, "args": args})
    return found


def gpu_compute_processes() -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            "nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits", "-i", "0",
        ], text=True, capture_output=True, check=True,
    )
    processes = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) != 3:
            raise RuntimeError(f"unexpected nvidia-smi compute row: {line!r}")
        processes.append({"pid": int(parts[0]), "name": parts[1], "used_memory_mib": int(parts[2])})
    return processes


def wait_for_gpu_quiescence(timeout_seconds: int = 21600) -> dict[str, int]:
    deadline = time.monotonic() + timeout_seconds
    stable = 0
    while time.monotonic() < deadline:
        if STOP_REQUESTED or SUPERVISOR_STOP.exists():
            raise InterruptedError("GPU quiescence wait cancelled")
        snapshot = gpu_snapshot()
        compute = gpu_compute_processes()
        quiet = not compute and snapshot["memory_used_mib"] <= 1024 and snapshot["utilization_pct"] <= 5
        stable = stable + 1 if quiet else 0
        if stable >= 2:
            return snapshot
        update_state("waiting_for_gpu_quiescence", gpu=snapshot, compute_processes=compute, stable=stable)
        for _ in range(15):
            if STOP_REQUESTED or SUPERVISOR_STOP.exists():
                raise InterruptedError("GPU quiescence wait cancelled")
            time.sleep(1)
    raise TimeoutError("GPU did not become quiescent before timeout")


def validate_spatial_state() -> tuple[dict[str, Any], Counter[str]]:
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
            digest = sha256_file(path)
            if path.stat().st_size != entry["size_bytes"] or digest != entry["sha256"]:
                raise RuntimeError(f"PTB-XL tensor differs from manifest: {path}")
            root_digest.update(f"{entry['record_id']}:{entry['size_bytes']}:{digest}\n".encode())
        if root_digest.hexdigest() != contract["splits"][split]["content_root_sha256"]:
            raise RuntimeError(f"PTB-XL {split} root digest mismatch")
        result[split] = {"records": len(entries), "content_root_sha256": root_digest.hexdigest()}
    return result


def verify_rdb_cache() -> dict[str, Any]:
    manifest_path = RDB_CACHE / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    inventories: dict[str, dict[str, str]] = {split: {} for split in ("train", "val", "test")}
    patients: dict[str, set[str]] = {split: set() for split in inventories}
    for record in manifest["records"]:
        split = record["split"]
        inventories[split][record["output"]] = record["output_sha256"]
        patients[split].add(record["patient_id"])
    if any(patients[left] & patients[right] for left, right in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise RuntimeError("RDB cache patient identities overlap across splits")
    for split in ("train", "val"):
        expected = inventories[split]
        actual = {str(path.relative_to(RDB_CACHE)) for path in (RDB_CACHE / split).glob("*.pt")}
        if actual != set(expected):
            raise RuntimeError(f"RDB {split} cache inventory differs from manifest")
        for relative, digest in expected.items():
            if sha256_file(RDB_CACHE / relative) != digest:
                raise RuntimeError(f"RDB cache tensor digest mismatch: {relative}")
    if manifest["split"]["test_role"] != "untouched; excluded from architecture selection and sweep training":
        raise RuntimeError("RDB held-out test role changed")
    return {
        "manifest_sha256": sha256_file(manifest_path),
        "counts": manifest["split"]["counts"],
        "source_dataset_sha256": manifest["source"]["dataset_sha256"],
    }


def replace_option(command: list[str], option: str, value: str) -> None:
    index = command.index(option)
    command[index + 1] = value


def ensure_preflight_manifest() -> dict[str, Any]:
    full = json.loads(FULL_MANIFEST.read_text())
    full_sha = sha256_file(FULL_MANIFEST)
    if PREFLIGHT_MANIFEST.exists():
        existing = json.loads(PREFLIGHT_MANIFEST.read_text())
        if existing.get("source_manifest_sha256") != full_sha:
            raise RuntimeError("preflight manifest is bound to a different full manifest")
        return existing
    selected: dict[str, dict[str, Any]] = {}
    for job in full["jobs"]:
        cell_name = job.get("cell", {}).get("name")
        command = job["command"]
        lead = command[command.index("--observed-leads") + 1]
        if cell_name in PREFLIGHT_CELLS and lead == "0" and cell_name not in selected:
            selected[cell_name] = job
    if set(selected) != set(PREFLIGHT_CELLS):
        raise RuntimeError(f"full manifest lacks preflight cells: {sorted(set(PREFLIGHT_CELLS)-set(selected))}")
    jobs = []
    for ordinal, cell_name in enumerate(PREFLIGHT_CELLS):
        source = selected[cell_name]
        command = list(source["command"])
        run_name = f"preflight_{cell_name}_{ordinal:02d}"
        replace_option(command, "--run-name", run_name)
        replace_option(command, "--output-dir", str(WORK_DIR / "preflight/runs" / run_name))
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
    atomic_json(PREFLIGHT_MANIFEST, payload)
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


def run_preflight_and_full() -> None:
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
    ensure_preflight_manifest()
    baseline = gpu_snapshot()
    update_state("running_preflight", ptb=ptb, rdb=rdb, ecc_baseline=baseline)
    preflight_code = run_queue(
        PREFLIGHT_MANIFEST, project_root=ROOT, max_attempts=2, min_free_gib=8,
        min_available_ram_gib=5, continue_on_error=False, retry_failed=True,
        max_gpu_used_mib=1024,
        resource_timeout_seconds=21600, job_timeout_seconds=3600,
    )
    if preflight_code != 0:
        raise RuntimeError(f"wavelet GPU preflight exited {preflight_code}")
    verify_queue_success(PREFLIGHT_MANIFEST, len(PREFLIGHT_CELLS))
    peak = max_peak_memory(PREFLIGHT_MANIFEST)
    if peak > 36 * 1024**3:
        raise RuntimeError(f"preflight peak GPU memory exceeded 36 GiB: {peak}")
    after_preflight = gpu_snapshot()
    if after_preflight["ecc_uncorrected"] > baseline["ecc_uncorrected"]:
        raise RuntimeError(f"uncorrected GPU ECC increased across preflight: {baseline} -> {after_preflight}")
    update_state(
        "preflight_complete", ptb=ptb, rdb=rdb, peak_gpu_memory_bytes=peak,
        corrected_ecc_delta=max(0, after_preflight["ecc_corrected"]-baseline["ecc_corrected"]),
        uncorrected_ecc_delta=max(0, after_preflight["ecc_uncorrected"]-baseline["ecc_uncorrected"]),
    )
    update_state("running_full_sweep", preflight_peak_gpu_memory_bytes=peak, ecc=after_preflight)
    full = json.loads(FULL_MANIFEST.read_text())
    full_code = run_queue(
        FULL_MANIFEST, project_root=ROOT, max_attempts=2, min_free_gib=8,
        min_available_ram_gib=5, continue_on_error=True, max_consecutive_failures=2,
        max_total_failures=5, max_gpu_used_mib=1024,
        resource_timeout_seconds=21600, job_timeout_seconds=14400,
    )
    if full_code != 0:
        raise RuntimeError(f"wavelet full sweep exited {full_code}")
    verify_queue_success(FULL_MANIFEST, len(full["jobs"]))


def ensure_barrier() -> None:
    payload = {
        "version": 1, "purpose": "spatial_then_wavelet_before_3arch",
        "created_or_refreshed_at": utc_now(), "supervisor_pid": os.getpid(),
        "expected_spatial_pair_sha256": EXPECTED_SPATIAL_PAIR_SHA256,
        "full_manifest": str(FULL_MANIFEST), "full_manifest_sha256": sha256_file(FULL_MANIFEST),
    }
    if BARRIER.exists():
        existing = json.loads(BARRIER.read_text())
        if existing.get("purpose") != payload["purpose"] or existing.get("full_manifest_sha256") != payload["full_manifest_sha256"]:
            raise RuntimeError("an incompatible 3-architecture barrier already exists")
    atomic_json(BARRIER, payload)


def remove_barrier() -> None:
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
    ensure_preflight_manifest()
    if args.prepare_only:
        print(json.dumps({
            "full_manifest": str(FULL_MANIFEST), "full_sha256": sha256_file(FULL_MANIFEST),
            "preflight_manifest": str(PREFLIGHT_MANIFEST),
            "preflight_sha256": sha256_file(PREFLIGHT_MANIFEST),
        }, indent=2))
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
                run_preflight_and_full()
                update_state("wavelet_complete", full_manifest_sha256=sha256_file(FULL_MANIFEST))
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
