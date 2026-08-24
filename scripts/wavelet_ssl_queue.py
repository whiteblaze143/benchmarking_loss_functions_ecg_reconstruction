#!/usr/bin/env python3
"""Crash-safe, resource-gated queue for the wavelet SSL smoke screen."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import shlex
import shutil
import signal
import sqlite3
import subprocess
import tempfile
import time
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any, Callable


TERMINAL = {"completed", "cancelled", "failed", "failed_oom", "failed_transient", "stuck"}
FAILURES = {"failed", "failed_oom", "failed_transient", "stuck"}
OOM_MARKERS = ("torch.outofmemoryerror", "cuda out of memory")
TRANSIENT_MARKERS = (
    "invalid access of peer gpu memory",
    "uncorrectable ecc error",
    "nvlink",
    "xid",
)
NON_TRAINING_ARGS = {
    "audit_delineation_dir", "audit_output", "emit_sweep_manifest", "run_sweep_manifest",
    "summarize_sweep", "summary_csv", "sweep_output_root", "sweep_epochs", "sweep_leads",
    "sweep_masks", "quick_verify", "retry_failed", "queue_max_attempts",
    "queue_min_free_gib", "queue_min_available_ram_gib", "queue_continue_on_error",
    "rolling_resume", "resume_min_free_gib",
}
NEGATABLE_OPTIONS = {
    "init_strict", "require_cuda", "rolling_resume", "seg_missing_only",
    "use_relative_geometry", "use_spatial_film", "use_wavelet_branch",
}


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=60)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata(
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS jobs(
            id TEXT PRIMARY KEY,
            ordinal INTEGER NOT NULL UNIQUE,
            command_json TEXT NOT NULL,
            command_sha256 TEXT NOT NULL,
            cell_json TEXT NOT NULL,
            status TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            child_pid INTEGER,
            started_at TEXT,
            completed_at TEXT,
            returncode INTEGER,
            error TEXT,
            log_path TEXT,
            output_dir TEXT NOT NULL,
            summary_json TEXT,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS attempts(
            job_id TEXT NOT NULL,
            attempt INTEGER NOT NULL,
            started_at TEXT NOT NULL,
            completed_at TEXT,
            returncode INTEGER,
            status TEXT NOT NULL,
            error TEXT,
            log_path TEXT NOT NULL,
            ecc_before_json TEXT,
            ecc_after_json TEXT,
            PRIMARY KEY(job_id, attempt),
            FOREIGN KEY(job_id) REFERENCES jobs(id)
        );
        """
    )
    return connection


def command_options(command: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    index = 2
    while index < len(command):
        token = command[index]
        if token.startswith("--"):
            key = token[2:].replace("-", "_")
            if key.startswith("no_") and key[3:] in NEGATABLE_OPTIONS:
                result[key[3:]] = False
                index += 1
                continue
            if index + 1 < len(command) and not command[index + 1].startswith("--"):
                values = []
                index += 1
                while index < len(command) and not command[index].startswith("--"):
                    values.append(command[index])
                    index += 1
                result[key] = values[0] if len(values) == 1 else values
                continue
            result[key] = True
        index += 1
    return result


def command_output_dir(command: list[str], project_root: Path) -> Path:
    options = command_options(command)
    if "output_dir" not in options or isinstance(options["output_dir"], list):
        raise ValueError("job command must contain one --output-dir")
    path = Path(str(options["output_dir"])).expanduser()
    return path if path.is_absolute() else project_root / path


def finite_tree(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    return True


def resolve_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root / path


def current_input_fingerprints(
    config: dict[str, Any], command: list[str], project_root: Path
) -> dict[str, str]:
    if len(command) < 2:
        raise RuntimeError("training command is missing its program path")
    paths = {
        "trainer_sha256": resolve_path(command[1], project_root),
        "model_sha256": project_root / "unified_latents/engineering/experimental/wavelet_ssl_ecg_aim.py",
        "data_manifest_sha256": resolve_path(config["data_manifest"], project_root),
    }
    if config.get("delineation_dir"):
        paths["delineation_manifest_sha256"] = (
            resolve_path(config["delineation_dir"], project_root) / "manifest.json"
        )
    for bank_key, asset_key in (
        ("wavelet_bank", "custom_wavelet_asset"),
        ("view_a_bank", "view_a_custom_wavelet_asset"),
        ("view_b_bank", "view_b_custom_wavelet_asset"),
    ):
        if config.get(bank_key) == "custom_asset":
            value = config.get(asset_key)
            if not value:
                raise RuntimeError(f"{asset_key} is required when {bank_key}=custom_asset")
            paths[f"{asset_key}_sha256"] = resolve_path(value, project_root)
    if config.get("init_checkpoint"):
        paths["init_checkpoint_sha256"] = resolve_path(config["init_checkpoint"], project_root)
    result: dict[str, str] = {}
    for key, path in paths.items():
        if not path.is_file():
            raise RuntimeError(f"required input is missing: {path}")
        result[key] = sha256_file(path)
    return result


def coerce_expected(raw: Any, actual: Any) -> Any:
    if isinstance(actual, list):
        values = raw if isinstance(raw, list) else [raw]
        if not actual:
            return values
        return [coerce_expected(value, actual[0]) for value in values]
    if isinstance(actual, bool):
        if isinstance(raw, bool):
            return raw
        return str(raw).lower() in {"1", "true", "yes"}
    if isinstance(actual, int) and not isinstance(actual, bool):
        return int(raw)
    if isinstance(actual, float):
        return float(raw)
    if actual is None:
        return raw
    return str(raw)


def validate_success(command: list[str], project_root: Path) -> dict[str, Any]:
    output = command_output_dir(command, project_root)
    config_path = output / "config.json"
    metrics_path = output / "metrics.jsonl"
    summary_path = output / "summary.json"
    success_path = output / "_SUCCESS.json"
    for path in (config_path, metrics_path, summary_path, success_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"missing completion artifact: {path}")
    config = json.loads(config_path.read_text())
    summary = json.loads(summary_path.read_text())
    success = json.loads(success_path.read_text())
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    if not finite_tree(config) or not finite_tree(summary) or not finite_tree(success) or not finite_tree(rows):
        raise RuntimeError("completion artifacts contain NaN or infinity")
    options = command_options(command)
    expected_name = str(options["run_name"])
    expected_epochs = 1 if options.get("quick_verify") else int(options["epochs"])
    if summary.get("run_name") != expected_name or success.get("run_name") != expected_name:
        raise RuntimeError("run-name mismatch in completion artifacts")
    if summary.get("epochs_completed") != expected_epochs or success.get("epochs_completed") != expected_epochs:
        raise RuntimeError("epoch-count mismatch in completion artifacts")
    if [row.get("epoch") for row in rows] != list(range(1, expected_epochs + 1)):
        raise RuntimeError("metrics are not a consecutive, complete epoch sequence")
    for key, path in (
        ("config_sha256", config_path),
        ("metrics_sha256", metrics_path),
        ("summary_sha256", summary_path),
    ):
        if success.get(key) != sha256_file(path):
            raise RuntimeError(f"artifact digest mismatch for {path.name}")
    if success.get("training_config_sha256") != summary.get("training_config_sha256"):
        raise RuntimeError("training-config hash mismatch")
    training = success.get("training_config")
    inputs = success.get("input_fingerprints")
    if not isinstance(training, dict) or not isinstance(inputs, dict):
        raise RuntimeError("completion marker lacks the canonical training contract")
    filtered_config = {key: value for key, value in config.items() if key not in NON_TRAINING_ARGS}
    if training != filtered_config:
        raise RuntimeError("completion training contract does not match config.json")
    protocol_raw = json.dumps(
        {"config": training, "input_fingerprints": inputs},
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    )
    if hashlib.sha256(protocol_raw.encode()).hexdigest() != success.get("training_config_sha256"):
        raise RuntimeError("canonical training-contract digest is invalid")
    if summary.get("input_fingerprints") != inputs:
        raise RuntimeError("summary input fingerprints do not match completion marker")
    current_inputs = current_input_fingerprints(config, command, project_root)
    if inputs != current_inputs:
        raise RuntimeError("completion artifacts were produced from different code or data manifests")
    important = {
        "run_name": expected_name,
        "factorial_mask": str(options["factorial_mask"]),
        "seed": int(options["seed"]),
        "batch_size": int(options["batch_size"]),
    }
    if config.get("observed_leads") != [int(options["observed_leads"])]:
        raise RuntimeError("observed-lead mismatch")
    for key, expected in important.items():
        if config.get(key) != expected:
            raise RuntimeError(f"config mismatch for {key}")
    for key, raw_expected in options.items():
        if key not in training:
            continue
        expected = 1 if key == "epochs" and options.get("quick_verify") else coerce_expected(
            raw_expected, training[key]
        )
        if training[key] != expected:
            raise RuntimeError(f"command/config mismatch for {key}: {training[key]!r} != {expected!r}")
    checkpoint_artifacts = success.get("checkpoint_artifacts")
    if not isinstance(checkpoint_artifacts, dict):
        raise RuntimeError("completion marker lacks checkpoint artifact inventory")
    policy = config.get("checkpoint_policy", "none")
    expected_checkpoints = {
        "none": set(), "best": {"best.pt"}, "last": {"last.pt"},
        "all": {"best.pt", "last.pt"},
    }[policy]
    if set(checkpoint_artifacts) != expected_checkpoints:
        raise RuntimeError("checkpoint artifact inventory does not match retention policy")
    for name, digest in checkpoint_artifacts.items():
        path = output / name
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"checkpoint artifact is missing or corrupt: {name}")
    return summary


def cleanup_resume(command: list[str], project_root: Path) -> None:
    output = command_output_dir(command, project_root)
    resume_path = output / "resume.pt"
    if not resume_path.exists():
        return
    resume_path.unlink()
    directory_fd = os.open(output, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def verify_manifest_inputs(manifest: dict[str, Any], jobs: list[dict[str, Any]], project_root: Path) -> None:
    commands = [job["command"] for job in jobs]
    outputs = [str(command_output_dir(command, project_root).resolve()) for command in commands]
    if len(set(outputs)) != len(outputs):
        raise ValueError("manifest jobs must have distinct output directories")
    reference: dict[str, str] | None = None
    for command in commands:
        options = command_options(command)
        minimal_config = {
            "data_manifest": options.get("data_manifest"),
            "delineation_dir": options.get("delineation_dir"),
            "wavelet_bank": options.get("wavelet_bank", "morlet"),
            "custom_wavelet_asset": options.get("custom_wavelet_asset"),
            "view_a_bank": options.get("view_a_bank", "inherit"),
            "view_b_bank": options.get("view_b_bank", "inherit"),
            "view_a_custom_wavelet_asset": options.get("view_a_custom_wavelet_asset"),
            "view_b_custom_wavelet_asset": options.get("view_b_custom_wavelet_asset"),
            "init_checkpoint": options.get("init_checkpoint"),
        }
        if not minimal_config["data_manifest"]:
            raise ValueError("job command lacks --data-manifest")
        fingerprints = current_input_fingerprints(minimal_config, command, project_root)
        common = {key: value for key, value in fingerprints.items()
                  if not key.endswith("custom_wavelet_asset_sha256")}
        if reference is None:
            reference = common
        elif common != reference:
            raise ValueError("manifest jobs do not share one immutable code/data contract")
        for key, value in fingerprints.items():
            if key.endswith("custom_wavelet_asset_sha256"):
                asset_key = key.removesuffix("_sha256")
                asset_value = minimal_config[asset_key]
                asset_path = str(resolve_path(asset_value, project_root).resolve())
                if manifest.get("custom_wavelet_assets", {}).get(asset_path) != value:
                    raise RuntimeError(f"manifest custom-wavelet inventory does not match {asset_path}")
            elif manifest.get(key) != value:
                raise RuntimeError(f"manifest {key} does not match the current input")


def initialize(connection: sqlite3.Connection, manifest_path: Path, project_root: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("manifest must contain at least one job")
    ids = [job.get("id") for job in jobs]
    if any(not isinstance(job_id, str) or not job_id for job_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("manifest job IDs must be non-empty and unique")
    manifest_sha = sha256_file(manifest_path)
    verify_manifest_inputs(manifest, jobs, project_root)
    stored = connection.execute("SELECT value FROM metadata WHERE key='manifest_sha256'").fetchone()
    if stored and stored[0] != manifest_sha:
        raise RuntimeError("manifest changed after queue initialization")
    connection.execute(
        "INSERT OR IGNORE INTO metadata(key,value) VALUES('manifest_sha256',?)", (manifest_sha,)
    )
    connection.execute("INSERT OR IGNORE INTO metadata(key,value) VALUES('created_at',?)", (utc_now(),))
    for ordinal, job in enumerate(jobs):
        command = job.get("command")
        if not isinstance(command, list) or not all(isinstance(token, str) and token for token in command):
            raise ValueError(f"{job['id']}: command must be a non-empty string list")
        raw = json.dumps(command, separators=(",", ":"))
        digest = hashlib.sha256(raw.encode()).hexdigest()
        output_dir = str(command_output_dir(command, project_root).resolve())
        existing = connection.execute("SELECT command_sha256 FROM jobs WHERE id=?", (job["id"],)).fetchone()
        if existing and existing[0] != digest:
            raise RuntimeError(f"{job['id']}: command changed after queue initialization")
        connection.execute(
            """
            INSERT OR IGNORE INTO jobs(
                id,ordinal,command_json,command_sha256,cell_json,status,attempts,
                output_dir,updated_at
            ) VALUES(?,?,?,?,?,'pending',0,?,?)
            """,
            (
                job["id"], ordinal, raw, digest,
                json.dumps(job.get("cell", {}), sort_keys=True, separators=(",", ":")),
                output_dir, utc_now(),
            ),
        )
    if connection.execute("SELECT count(*) FROM jobs").fetchone()[0] != len(jobs):
        raise RuntimeError("queue DB and manifest job counts differ")
    connection.commit()
    return manifest


def process_matches(pid: int | None, command: list[str]) -> bool:
    if not pid or not Path(f"/proc/{pid}/cmdline").is_file():
        return False
    try:
        actual = Path(f"/proc/{pid}/cmdline").read_bytes().rstrip(b"\0").split(b"\0")
        expected = [os.fsencode(token) for token in command]
        return actual == expected
    except OSError:
        return False


def reconcile(connection: sqlite3.Connection, project_root: Path, max_attempts: int, retry_failed: bool) -> None:
    for row in connection.execute("SELECT * FROM jobs ORDER BY ordinal").fetchall():
        command = json.loads(row["command_json"])
        status = row["status"]
        try:
            summary = validate_success(command, project_root)
        except Exception as error:
            summary = None
            validation_error = str(error)
        if summary is not None:
            cleanup_resume(command, project_root)
            connection.execute(
                """UPDATE jobs SET status='completed',child_pid=NULL,returncode=0,summary_json=?,
                   error=NULL,completed_at=coalesce(completed_at,?),updated_at=? WHERE id=?""",
                (json.dumps(summary, sort_keys=True, allow_nan=False), utc_now(), utc_now(), row["id"]),
            )
            continue
        if status == "completed":
            replacement = "pending" if retry_failed and row["attempts"] < max_attempts else "failed"
            connection.execute(
                """UPDATE jobs SET status=?,child_pid=NULL,completed_at=NULL,returncode=NULL,
                   summary_json=NULL,error=?,updated_at=? WHERE id=?""",
                (replacement, f"completed artifact invalid: {validation_error}", utc_now(), row["id"]),
            )
        elif status == "running":
            if process_matches(row["child_pid"], command):
                raise RuntimeError(
                    f"{row['id']}: orphan child is still alive at PID {row['child_pid']}; "
                    "refusing to schedule beside it"
                )
            elif row["attempts"] < max_attempts:
                connection.execute(
                    """UPDATE jobs SET status='pending',child_pid=NULL,completed_at=NULL,
                       returncode=NULL,summary_json=NULL,error=?,updated_at=? WHERE id=?""",
                    ("runner interrupted; retrying current job from rolling resume", utc_now(), row["id"]),
                )
                connection.execute(
                    """UPDATE attempts SET completed_at=?,returncode=130,status='interrupted',error=?
                       WHERE job_id=? AND attempt=? AND status='running'""",
                    (utc_now(), "queue runner disappeared; rolling resume retained", row["id"], row["attempts"]),
                )
            else:
                connection.execute(
                    "UPDATE jobs SET status='stuck',child_pid=NULL,error=?,updated_at=? WHERE id=?",
                    ("runner interrupted and attempt budget exhausted", utc_now(), row["id"]),
                )
                connection.execute(
                    """UPDATE attempts SET completed_at=?,returncode=130,status='stuck',error=?
                       WHERE job_id=? AND attempt=? AND status='running'""",
                    (utc_now(), "queue runner disappeared and attempt budget exhausted", row["id"], row["attempts"]),
                )
        elif status in FAILURES and retry_failed and row["attempts"] < max_attempts:
            connection.execute(
                """UPDATE jobs SET status='pending',child_pid=NULL,completed_at=NULL,
                   returncode=NULL,summary_json=NULL,error=?,updated_at=? WHERE id=?""",
                (f"explicit retry requested after {status}", utc_now(), row["id"]),
            )
    connection.commit()


def export_state(connection: sqlite3.Connection, manifest: dict[str, Any], path: Path, queue_status: str) -> None:
    rows = connection.execute("SELECT * FROM jobs ORDER BY ordinal").fetchall()
    jobs = []
    for row in rows:
        item = {
            "id": row["id"], "command": json.loads(row["command_json"]),
            "cell": json.loads(row["cell_json"]), "status": row["status"],
            "attempts": row["attempts"],
        }
        for key in ("child_pid", "started_at", "completed_at", "returncode", "error", "log_path"):
            if row[key] is not None:
                item[key] = row[key]
        jobs.append(item)
    counts = {status: 0 for status in sorted({row["status"] for row in rows})}
    for row in rows:
        counts[row["status"]] += 1
    atomic_json(
        path,
        {
            "version": 2, "queue_status": queue_status, "updated_at": utc_now(),
            "manifest_sha256": connection.execute(
                "SELECT value FROM metadata WHERE key='manifest_sha256'"
            ).fetchone()[0],
            "counts": counts, "cells": manifest.get("cells"), "jobs": jobs,
        },
    )


def available_ram_bytes() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is absent from /proc/meminfo")


def gpu_snapshot() -> dict[str, int]:
    fields = (
        "memory.used,memory.free,utilization.gpu,"
        "ecc.errors.corrected.volatile.total,"
        "ecc.errors.uncorrected.volatile.total"
    )
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits", "-i", "0"],
        text=True, capture_output=True, check=True,
    )
    values = [value.strip() for value in result.stdout.strip().split(",")]
    if len(values) != 5:
        raise RuntimeError(f"unexpected nvidia-smi output: {result.stdout!r}")
    parsed = []
    for value in values:
        parsed.append(-1 if value in {"N/A", "[N/A]"} else int(value))
    return dict(zip(("memory_used_mib", "memory_free_mib", "utilization_pct", "ecc_corrected", "ecc_uncorrected"), parsed))


def wait_for_resources(
    project_root: Path,
    min_free_gib: float,
    min_available_ram_gib: float,
    max_gpu_used_mib: int,
    timeout_seconds: int,
    cancelled: Callable[[], bool] | None = None,
) -> dict[str, int]:
    deadline = time.monotonic() + timeout_seconds
    stable = 0
    last_report = 0.0
    while time.monotonic() < deadline:
        if cancelled is not None and cancelled():
            raise InterruptedError("resource wait cancelled")
        free = shutil.disk_usage(project_root).free
        ram = available_ram_bytes()
        gpu = gpu_snapshot()
        ok = (
            free >= min_free_gib * 1024**3
            and ram >= min_available_ram_gib * 1024**3
            and gpu["memory_used_mib"] <= max_gpu_used_mib
            and gpu["utilization_pct"] <= 5
        )
        stable = stable + 1 if ok else 0
        if stable >= 2:
            return gpu
        if time.monotonic() - last_report >= 60:
            print(
                f"resource gate: disk={free/1024**3:.2f} GiB RAM={ram/1024**3:.2f} GiB "
                f"GPU-used={gpu['memory_used_mib']} MiB util={gpu['utilization_pct']}% stable={stable}/2",
                flush=True,
            )
            last_report = time.monotonic()
        # Two consecutive samples still protect against racing another GPU
        # process, without imposing a fixed 30-second bubble between our own
        # serial jobs after CUDA memory has already been released.
        for _ in range(2):
            if cancelled is not None and cancelled():
                raise InterruptedError("resource wait cancelled")
            time.sleep(1)
    raise TimeoutError("resource gate did not become stable before timeout")


def classify_failure(log_path: Path) -> str:
    tail = log_path.read_text(errors="replace")[-2_000_000:].lower()
    if any(marker in tail for marker in OOM_MARKERS):
        return "failed_oom"
    if any(marker in tail for marker in TRANSIENT_MARKERS):
        return "failed_transient"
    return "failed"


@contextmanager
def singleton(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        handle.seek(0); handle.truncate(); handle.write(f"pid={os.getpid()} started={utc_now()}\n"); handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _run_queue_impl(
    manifest_path: str | Path,
    *,
    project_root: str | Path,
    max_attempts: int = 2,
    retry_failed: bool = False,
    min_free_gib: float = 8.0,
    min_available_ram_gib: float = 5.0,
    continue_on_error: bool = False,
    max_consecutive_failures: int = 2,
    max_total_failures: int = 5,
    max_gpu_used_mib: int = 1024,
    resource_timeout_seconds: int = 21600,
    job_timeout_seconds: int = 14400,
) -> int:
    manifest_path = Path(manifest_path).resolve()
    project_root = Path(project_root).resolve()
    queue_dir = manifest_path.parent
    state_path = manifest_path.with_suffix(".state.json")
    db_path = queue_dir / "queue.sqlite"
    stop_path = queue_dir / "STOP"
    lock_path = queue_dir / "queue_worker.lock"
    log_dir = queue_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    with singleton(lock_path), closing(connect(db_path)) as connection:
        manifest = initialize(connection, manifest_path, project_root)
        reconcile(connection, project_root, max_attempts, retry_failed)
        export_state(connection, manifest, state_path, "running")
        (queue_dir / "_QUEUE_SUCCESS.json").unlink(missing_ok=True)
        (queue_dir / "_QUEUE_FAILED.json").unlink(missing_ok=True)
        consecutive_failures = 0
        total_failures = connection.execute(
            "SELECT count(*) FROM jobs WHERE status IN ('failed','failed_oom','failed_transient','stuck')"
        ).fetchone()[0]
        interrupted = False
        child: subprocess.Popen[Any] | None = None

        def handle_signal(signum: int, _frame: Any) -> None:
            nonlocal interrupted
            interrupted = True
            print(f"received signal {signum}; stopping after child termination", flush=True)
            if child is not None and child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

        old_handlers = {
            signum: signal.signal(signum, handle_signal) for signum in (signal.SIGINT, signal.SIGTERM)
        }
        try:
            while True:
                if interrupted:
                    export_state(connection, manifest, state_path, "interrupted")
                    return 130
                if stop_path.exists():
                    export_state(connection, manifest, state_path, "paused")
                    print(f"STOP sentinel present at {stop_path}; remove it and restart to continue", flush=True)
                    return 75
                row = connection.execute(
                    "SELECT * FROM jobs WHERE status='pending' ORDER BY ordinal LIMIT 1"
                ).fetchone()
                if row is None:
                    break
                if row["attempts"] >= max_attempts:
                    connection.execute(
                        "UPDATE jobs SET status='stuck',error=?,updated_at=? WHERE id=?",
                        ("attempt budget exhausted", utc_now(), row["id"]),
                    ); connection.commit(); export_state(connection, manifest, state_path, "running")
                    continue
                try:
                    baseline = wait_for_resources(
                        project_root, min_free_gib, min_available_ram_gib,
                        max_gpu_used_mib, resource_timeout_seconds,
                        cancelled=lambda: interrupted or stop_path.exists(),
                    )
                except InterruptedError:
                    queue_status = "interrupted" if interrupted else "paused"
                    export_state(connection, manifest, state_path, queue_status)
                    return 130 if interrupted else 75
                except Exception as resource_error:
                    failure = {
                        "version": 1, "status": "infrastructure_failure", "at": utc_now(),
                        "stage": "resource_gate", "error": f"{type(resource_error).__name__}: {resource_error}",
                    }
                    atomic_json(queue_dir / "_QUEUE_FAILED.json", failure)
                    export_state(connection, manifest, state_path, "failed")
                    return 74
                command = json.loads(row["command_json"])
                attempt = int(row["attempts"]) + 1
                log_path = log_dir / f"{row['ordinal']:03d}_{row['id']}.log"
                started = utc_now()
                connection.execute(
                    """UPDATE jobs SET status='running',attempts=?,started_at=?,completed_at=NULL,
                       returncode=NULL,error=NULL,summary_json=NULL,log_path=?,updated_at=? WHERE id=?""",
                    (attempt, started, str(log_path), started, row["id"]),
                )
                connection.execute(
                    """INSERT OR REPLACE INTO attempts(
                       job_id,attempt,started_at,status,log_path,ecc_before_json
                       ) VALUES(?,?,?,'running',?,?)""",
                    (row["id"], attempt, started, str(log_path), json.dumps(baseline, sort_keys=True)),
                )
                connection.commit(); export_state(connection, manifest, state_path, "running")
                print(f"[{started}] starting {row['id']} attempt {attempt}: {shlex.join(command)}", flush=True)
                try:
                    with log_path.open("a", buffering=1) as log:
                        log.write(f"\n===== attempt {attempt} started {started} =====\n")
                        env = os.environ.copy()
                        env.update(PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES="0")
                        child = subprocess.Popen(
                            command, cwd=project_root, env=env, stdout=log,
                            stderr=subprocess.STDOUT, text=True, start_new_session=True,
                        )
                        connection.execute(
                            "UPDATE jobs SET child_pid=?,updated_at=? WHERE id=?",
                            (child.pid, utc_now(), row["id"]),
                        ); connection.commit(); export_state(connection, manifest, state_path, "running")
                        deadline = time.monotonic() + job_timeout_seconds
                        while child.poll() is None and not interrupted and time.monotonic() < deadline:
                            time.sleep(10)
                        timed_out = child.poll() is None and not interrupted and time.monotonic() >= deadline
                        if child.poll() is None:
                            try: os.killpg(child.pid, signal.SIGTERM)
                            except ProcessLookupError: pass
                            try: child.wait(timeout=30)
                            except subprocess.TimeoutExpired:
                                try: os.killpg(child.pid, signal.SIGKILL)
                                except ProcessLookupError: pass
                                child.wait()
                        returncode = child.returncode
                        child = None
                    ended = utc_now()
                    after = gpu_snapshot()
                except Exception as infrastructure_error:
                    if child is not None and child.poll() is None:
                        try: os.killpg(child.pid, signal.SIGTERM)
                        except ProcessLookupError: pass
                        try: child.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            try: os.killpg(child.pid, signal.SIGKILL)
                            except ProcessLookupError: pass
                            child.wait()
                    child = None
                    ended = utc_now()
                    status = "pending" if attempt < max_attempts else "stuck"
                    error = f"queue infrastructure failure: {type(infrastructure_error).__name__}: {infrastructure_error}"
                    connection.execute(
                        """UPDATE jobs SET status=?,child_pid=NULL,completed_at=NULL,returncode=74,
                           summary_json=NULL,error=?,updated_at=? WHERE id=?""",
                        (status, error, ended, row["id"]),
                    )
                    connection.execute(
                        """UPDATE attempts SET completed_at=?,returncode=74,status='infrastructure_failure',error=?
                           WHERE job_id=? AND attempt=?""",
                        (ended, error, row["id"], attempt),
                    )
                    connection.commit()
                    atomic_json(queue_dir / "_QUEUE_FAILED.json", {
                        "version": 1, "status": "infrastructure_failure", "at": ended,
                        "job_id": row["id"], "error": error,
                    })
                    export_state(connection, manifest, state_path, "failed")
                    return 74
                summary: dict[str, Any] | None = None
                corrected_delta = (
                    max(0, after["ecc_corrected"] - baseline["ecc_corrected"])
                    if baseline["ecc_corrected"] >= 0 and after["ecc_corrected"] >= 0 else 0
                )
                # Corrected ECC is retained in attempt telemetry and warned,
                # but by definition the hardware correction succeeded. Any
                # uncorrected increment remains a hard transient failure.
                ecc_increase = (
                    baseline["ecc_uncorrected"] >= 0
                    and after["ecc_uncorrected"] > baseline["ecc_uncorrected"]
                )
                if corrected_delta:
                    print(
                        f"warning: corrected GPU ECC increased by {corrected_delta} during {row['id']}",
                        flush=True,
                    )
                error: str | None = None
                if interrupted:
                    status = "pending" if attempt < max_attempts else "stuck"
                    error = "queue interrupted during child; rolling resume retained"
                elif timed_out:
                    status = "failed"
                    error = f"job exceeded timeout of {job_timeout_seconds} seconds"
                    returncode = 124
                elif returncode != 0:
                    status = classify_failure(log_path)
                    error = f"child exited with status {returncode}"
                elif ecc_increase:
                    status = "failed_transient"
                    error = f"GPU ECC counters increased: before={baseline}, after={after}"
                    returncode = 75
                else:
                    try:
                        summary = validate_success(command, project_root)
                    except Exception as validation_error:
                        status = "failed"; error = f"completion validation failed: {validation_error}"; returncode = 70
                    else:
                        cleanup_resume(command, project_root)
                        status = "completed"; error = None; returncode = 0
                connection.execute(
                    """UPDATE jobs SET status=?,child_pid=NULL,completed_at=?,returncode=?,error=?,
                       summary_json=?,updated_at=? WHERE id=?""",
                    (
                        status, ended if status != "pending" else None, returncode, error,
                        json.dumps(summary, sort_keys=True, allow_nan=False) if status == "completed" else None,
                        ended, row["id"],
                    ),
                )
                connection.execute(
                    """UPDATE attempts SET completed_at=?,returncode=?,status=?,error=?,ecc_after_json=?
                       WHERE job_id=? AND attempt=?""",
                    (ended, returncode, status, error, json.dumps(after, sort_keys=True), row["id"], attempt),
                )
                connection.commit(); export_state(connection, manifest, state_path, "running")
                print(f"[{ended}] {status}: {row['id']} ({error or 'validated success'})", flush=True)
                if status == "completed":
                    consecutive_failures = 0
                elif status in FAILURES:
                    consecutive_failures += 1; total_failures += 1
                    if (
                        not continue_on_error
                        or consecutive_failures >= max_consecutive_failures
                        or total_failures >= max_total_failures
                    ):
                        print("queue circuit breaker opened", flush=True)
                        break
            counts = dict(connection.execute("SELECT status,count(*) FROM jobs GROUP BY status"))
            pending = counts.get("pending", 0) + counts.get("running", 0)
            failures = sum(counts.get(status, 0) for status in FAILURES)
            if pending == 0 and failures == 0:
                result = {
                    "version": 1, "status": "completed", "completed_at": utc_now(),
                    "manifest_sha256": sha256_file(manifest_path), "counts": counts,
                    "database_sha256_note": "SQLite is live canonical state; use backup API for snapshots",
                }
                atomic_json(queue_dir / "_QUEUE_SUCCESS.json", result)
                (queue_dir / "_QUEUE_FAILED.json").unlink(missing_ok=True)
                export_state(connection, manifest, state_path, "completed")
                return 0
            result = {
                "version": 1, "status": "incomplete", "stopped_at": utc_now(),
                "manifest_sha256": sha256_file(manifest_path), "counts": counts,
            }
            atomic_json(queue_dir / "_QUEUE_FAILED.json", result)
            export_state(connection, manifest, state_path, "failed")
            return 1
        finally:
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)


def run_queue(manifest_path: str | Path, **kwargs: Any) -> int:
    """Run the queue and leave a durable failure marker for outer failures."""
    try:
        return _run_queue_impl(manifest_path, **kwargs)
    except Exception as error:
        path = Path(manifest_path).resolve()
        failure = {
            "version": 1,
            "status": "infrastructure_failure",
            "at": utc_now(),
            "stage": "queue_outer",
            "error": f"{type(error).__name__}: {error}",
        }
        try:
            atomic_json(path.parent / "_QUEUE_FAILED.json", failure)
        except Exception:
            pass
        print(json.dumps(failure, sort_keys=True), flush=True)
        return 74
