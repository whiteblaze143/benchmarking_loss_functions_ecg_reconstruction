#!/usr/bin/env python3
"""Exact, content-verified storage for factorial model checkpoints.

Large tensor blobs live in a private GitHub draft release.  A small local
SQLite catalog maps a queue/model id to its immutable SHA-256 digest and
release asset.  Checkpoints are only evicted after a byte-identical remote
round trip has been verified.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_DB = ROOT / "results/checkpoint_store/catalog.sqlite"
DEFAULT_STATE = ROOT / "refine-logs/queue/queue_state.json"
DEFAULT_CHECKPOINT_DIR = ROOT / "checkpoints"
DEFAULT_CACHE_DIR = ROOT / "checkpoints/cache"
DEFAULT_REPO = "whiteblaze143/benchmarking_loss_functions_ecg_reconstruction"
DEFAULT_TAG = "factorial-checkpoints-v1"
CHECKPOINT_RE = re.compile(
    r"^factorial_(?:(?P<architecture>msvae|ecg_aim)_)?"
    r"(?P<mask>\d{7})_s(?P<seed>\d+)\.pt$"
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], *, capture: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=capture,
        check=True,
    )


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path, timeout=60)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS checkpoints (
            model_id TEXT PRIMARY KEY,
            factorial_mask TEXT NOT NULL,
            seed INTEGER NOT NULL,
            filename TEXT NOT NULL UNIQUE,
            local_path TEXT,
            local_mtime_ns INTEGER,
            size_bytes INTEGER NOT NULL,
            sha256 TEXT NOT NULL,
            metadata_json TEXT,
            queue_completed_at TEXT,
            release_tag TEXT,
            asset_id INTEGER,
            asset_name TEXT,
            asset_size_bytes INTEGER,
            asset_digest TEXT,
            asset_state TEXT,
            uploaded_at TEXT,
            remote_verified_at TEXT,
            round_trip_verified_at TEXT,
            payload_validated_at TEXT,
            payload_tensor_count INTEGER,
            payload_factorial_mask TEXT,
            payload_seed INTEGER,
            payload_state_schema_sha256 TEXT,
            last_materialized_at TEXT,
            status TEXT NOT NULL CHECK (
                status IN ('local', 'uploading', 'remote_verified', 'cached', 'error')
            ),
            error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS checkpoints_status_idx
            ON checkpoints(status);
        """
    )
    columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(checkpoints)")
    }
    if "local_mtime_ns" not in columns:
        connection.execute(
            "ALTER TABLE checkpoints ADD COLUMN local_mtime_ns INTEGER"
        )
        connection.commit()
    migrations = {
        "round_trip_verified_at": "TEXT",
        "payload_validated_at": "TEXT",
        "payload_tensor_count": "INTEGER",
        "payload_factorial_mask": "TEXT",
        "payload_seed": "INTEGER",
        "payload_state_schema_sha256": "TEXT",
    }
    for name, kind in migrations.items():
        if name not in columns:
            connection.execute(
                f"ALTER TABLE checkpoints ADD COLUMN {name} {kind}"
            )
    connection.commit()
    return connection


@contextlib.contextmanager
def store_lock(db_path: Path) -> Iterator[None]:
    lock_path = db_path.with_suffix(db_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def set_settings(connection: sqlite3.Connection, repo: str, tag: str) -> None:
    connection.executemany(
        """
        INSERT INTO settings(key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """,
        [("repo", repo), ("release_tag", tag)],
    )


def read_settings(connection: sqlite3.Connection) -> tuple[str, str]:
    values = {
        row["key"]: row["value"]
        for row in connection.execute("SELECT key, value FROM settings")
    }
    return values.get("repo", DEFAULT_REPO), values.get("release_tag", DEFAULT_TAG)


def completed_jobs(state_path: Path) -> dict[str, dict[str, Any]]:
    target = state_path if state_path.is_absolute() else ROOT / state_path
    jobs: dict[str, dict[str, Any]] = {}
    if target.exists():
        payload = json.loads(target.read_text())
        for job in payload.get("jobs", []):
            if job.get("status") == "completed":
                jobs[job["id"]] = job
    alt_state = ROOT / "refine-logs/queue_3arch/queue_state.json"
    if alt_state != target and alt_state.exists():
        payload_alt = json.loads(alt_state.read_text())
        for job in payload_alt.get("jobs", []):
            if job.get("status") == "completed":
                jobs[job["id"]] = job
    return jobs


def parse_checkpoint(path: Path) -> tuple[str, str, int]:
    match = CHECKPOINT_RE.match(path.name)
    if not match:
        raise ValueError(f"Not a factorial checkpoint filename: {path.name}")
    mask = match.group("mask")
    seed = int(match.group("seed"))
    architecture = match.group("architecture")
    if architecture:
        return f"factorial_{architecture}_{mask}_s{seed}", mask, seed
    return f"f_{mask}_s{seed}", mask, seed


def queue_job_id(model_id: str, mask: str, seed: int) -> str:
    if model_id.startswith("factorial_msvae_"):
        return f"msvae_f_{mask}_s{seed}"
    if model_id.startswith("factorial_ecg_aim_"):
        return f"ecg_aim_f_{mask}_s{seed}"
    return model_id


def catalog_completed(
    connection: sqlite3.Connection,
    state_path: Path,
    checkpoint_dir: Path,
) -> int:
    jobs = completed_jobs(state_path)
    count = 0
    now = utc_now()
    for path in sorted(checkpoint_dir.glob("factorial_*.pt")):
        model_id, mask, seed = parse_checkpoint(path)
        job = jobs.get(queue_job_id(model_id, mask, seed))
        if job is None:
            continue
        existing = connection.execute(
            """
            SELECT sha256, size_bytes, status, local_mtime_ns
            FROM checkpoints WHERE model_id=?
            """,
            (model_id,),
        ).fetchone()
        file_stat = path.stat()
        size = file_stat.st_size
        if (
            existing
            and existing["size_bytes"] == size
            and existing["local_mtime_ns"] == file_stat.st_mtime_ns
        ):
            digest = existing["sha256"]
        else:
            digest = sha256_file(path)
        if existing and existing["status"] in {"remote_verified", "cached"}:
            if existing["sha256"] != digest or existing["size_bytes"] != size:
                raise RuntimeError(
                    f"Refusing changed bytes for verified {model_id}: "
                    f"{existing['sha256']} != {digest}"
                )
        metadata_path = path.with_suffix(".metadata.json")
        metadata = metadata_path.read_text() if metadata_path.exists() else None
        connection.execute(
            """
            INSERT INTO checkpoints(
                model_id, factorial_mask, seed, filename, local_path,
                local_mtime_ns, size_bytes, sha256, metadata_json, queue_completed_at,
                status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'local', ?, ?)
            ON CONFLICT(model_id) DO UPDATE SET
                local_path=excluded.local_path,
                local_mtime_ns=excluded.local_mtime_ns,
                size_bytes=excluded.size_bytes,
                sha256=excluded.sha256,
                metadata_json=COALESCE(excluded.metadata_json, checkpoints.metadata_json),
                queue_completed_at=excluded.queue_completed_at,
                status=CASE
                    WHEN checkpoints.sha256 != excluded.sha256
                      OR checkpoints.size_bytes != excluded.size_bytes
                    THEN 'local'
                    WHEN checkpoints.status IN ('remote_verified', 'cached')
                    THEN 'cached'
                    ELSE 'local'
                END,
                release_tag=CASE
                    WHEN checkpoints.sha256 != excluded.sha256
                      OR checkpoints.size_bytes != excluded.size_bytes
                    THEN NULL ELSE checkpoints.release_tag END,
                asset_id=CASE
                    WHEN checkpoints.sha256 != excluded.sha256
                      OR checkpoints.size_bytes != excluded.size_bytes
                    THEN NULL ELSE checkpoints.asset_id END,
                asset_name=CASE
                    WHEN checkpoints.sha256 != excluded.sha256
                      OR checkpoints.size_bytes != excluded.size_bytes
                    THEN NULL ELSE checkpoints.asset_name END,
                asset_size_bytes=CASE
                    WHEN checkpoints.sha256 != excluded.sha256
                      OR checkpoints.size_bytes != excluded.size_bytes
                    THEN NULL ELSE checkpoints.asset_size_bytes END,
                asset_digest=CASE
                    WHEN checkpoints.sha256 != excluded.sha256
                      OR checkpoints.size_bytes != excluded.size_bytes
                    THEN NULL ELSE checkpoints.asset_digest END,
                asset_state=CASE
                    WHEN checkpoints.sha256 != excluded.sha256
                      OR checkpoints.size_bytes != excluded.size_bytes
                    THEN NULL ELSE checkpoints.asset_state END,
                uploaded_at=CASE
                    WHEN checkpoints.sha256 != excluded.sha256
                      OR checkpoints.size_bytes != excluded.size_bytes
                    THEN NULL ELSE checkpoints.uploaded_at END,
                remote_verified_at=CASE
                    WHEN checkpoints.sha256 != excluded.sha256
                      OR checkpoints.size_bytes != excluded.size_bytes
                    THEN NULL ELSE checkpoints.remote_verified_at END,
                round_trip_verified_at=CASE
                    WHEN checkpoints.sha256 != excluded.sha256
                      OR checkpoints.size_bytes != excluded.size_bytes
                    THEN NULL ELSE checkpoints.round_trip_verified_at END,
                payload_validated_at=CASE
                    WHEN checkpoints.sha256 != excluded.sha256
                      OR checkpoints.size_bytes != excluded.size_bytes
                    THEN NULL ELSE checkpoints.payload_validated_at END,
                payload_tensor_count=CASE
                    WHEN checkpoints.sha256 != excluded.sha256
                      OR checkpoints.size_bytes != excluded.size_bytes
                    THEN NULL ELSE checkpoints.payload_tensor_count END,
                payload_factorial_mask=CASE
                    WHEN checkpoints.sha256 != excluded.sha256
                      OR checkpoints.size_bytes != excluded.size_bytes
                    THEN NULL ELSE checkpoints.payload_factorial_mask END,
                payload_seed=CASE
                    WHEN checkpoints.sha256 != excluded.sha256
                      OR checkpoints.size_bytes != excluded.size_bytes
                    THEN NULL ELSE checkpoints.payload_seed END,
                payload_state_schema_sha256=CASE
                    WHEN checkpoints.sha256 != excluded.sha256
                      OR checkpoints.size_bytes != excluded.size_bytes
                    THEN NULL ELSE checkpoints.payload_state_schema_sha256 END,
                error=NULL,
                updated_at=excluded.updated_at
            """,
            (
                model_id,
                mask,
                seed,
                path.name,
                str(path.resolve()),
                file_stat.st_mtime_ns,
                size,
                digest,
                metadata,
                job.get("completed"),
                now,
                now,
            ),
        )
        count += 1
    connection.commit()
    return count


def release_json(repo: str, tag: str) -> dict[str, Any]:
    releases = json.loads(
        run(
            [
                "gh",
                "api",
                "--paginate",
                f"repos/{repo}/releases?per_page=100",
            ]
        ).stdout
    )
    for release in releases:
        if release["tag_name"] == tag:
            return release
    raise RuntimeError(f"Release tag {tag!r} not found in {repo}")


def release_asset(repo: str, tag: str, name: str) -> dict[str, Any] | None:
    release = release_json(repo, tag)
    for asset in release.get("assets", []):
        if asset["name"] == name:
            return asset
    return None


def verify_asset(row: sqlite3.Row, asset: dict[str, Any]) -> None:
    expected_digest = f"sha256:{row['sha256']}"
    actual_digest = asset.get("digest")
    if asset.get("state") != "uploaded":
        raise RuntimeError(f"Asset {asset['name']} state is {asset.get('state')!r}")
    if int(asset["size"]) != int(row["size_bytes"]):
        raise RuntimeError(
            f"Asset size mismatch for {row['model_id']}: "
            f"{asset['size']} != {row['size_bytes']}"
        )
    if actual_digest != expected_digest:
        raise RuntimeError(
            f"Asset digest mismatch for {row['model_id']}: "
            f"{actual_digest} != {expected_digest}"
        )


def download_asset(repo: str, asset_id: int, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "gh",
            "api",
            "-H",
            "Accept: application/octet-stream",
            f"repos/{repo}/releases/assets/{asset_id}",
        ],
        capture=False,
    )


def download_asset_to_file(repo: str, asset_id: int, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        subprocess.run(
            [
                "gh",
                "api",
                "-H",
                "Accept: application/octet-stream",
                f"repos/{repo}/releases/assets/{asset_id}",
            ],
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.PIPE,
            check=True,
        )


def verify_remote_round_trip(
    repo: str,
    row: sqlite3.Row,
    asset: dict[str, Any],
    connection: sqlite3.Connection,
) -> None:
    with tempfile.TemporaryDirectory(prefix="checkpoint-verify-") as temporary:
        downloaded = Path(temporary) / row["filename"]
        download_asset_to_file(repo, int(asset["id"]), downloaded)
        downloaded_size = downloaded.stat().st_size
        downloaded_digest = sha256_file(downloaded)
        if downloaded_size != row["size_bytes"] or downloaded_digest != row["sha256"]:
            raise RuntimeError(
                f"Downloaded bytes failed verification for {row['model_id']}: "
                f"{downloaded_size}/{downloaded_digest}"
            )
        if (
            row["payload_validated_at"] is None
            or row["payload_state_schema_sha256"] is None
        ):
            enrich_sidecar(connection, row, downloaded)


def enrich_sidecar(
    connection: sqlite3.Connection,
    row: sqlite3.Row,
    local_path: Path,
) -> None:
    """Validate embedded identity/provenance and persist a compact audit sidecar."""
    import torch

    payload = torch.load(local_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{row['model_id']} checkpoint payload is not a mapping")
    structured = isinstance(payload.get("model_state_dict"), dict)
    provenance = payload.get("provenance", {}) if structured else {}
    if provenance and not isinstance(provenance, dict):
        raise RuntimeError(f"{row['model_id']} has invalid embedded provenance")
    embedded_mask = provenance.get("factorial_mask", row["factorial_mask"])
    embedded_seed = provenance.get("seed", row["seed"])
    if str(embedded_mask) != row["factorial_mask"]:
        raise RuntimeError(f"Embedded factorial mask mismatch for {row['model_id']}")
    if int(embedded_seed) != int(row["seed"]):
        raise RuntimeError(f"Embedded seed mismatch for {row['model_id']}")
    states = payload["model_state_dict"] if structured else payload
    if not isinstance(states, dict) or not states:
        raise RuntimeError(f"{row['model_id']} has an empty model state")
    states = {
        key.removeprefix("_orig_mod."): tensor
        for key, tensor in states.items()
    }
    if not all(isinstance(tensor, torch.Tensor) for tensor in states.values()):
        raise RuntimeError(f"{row['model_id']} state contains non-tensor values")
    allowed_dtypes = {torch.float16, torch.float32}
    invalid_dtypes = {
        str(tensor.dtype)
        for tensor in states.values()
        if tensor.dtype not in allowed_dtypes
    }
    if invalid_dtypes:
        raise RuntimeError(
            f"{row['model_id']} has unsupported state dtypes: {sorted(invalid_dtypes)}"
        )
    if not all(torch.isfinite(tensor).all().item() for tensor in states.values()):
        raise RuntimeError(f"{row['model_id']} state contains non-finite weights")
    model_id = row["model_id"]
    if model_id.startswith("factorial_msvae_"):
        from unified_latents.engineering.experimental.Multi_Scale_VAE import WearECGVAE

        architecture = "msvae"
        reference_model = WearECGVAE(
            latent_channels=int(payload.get("latent_channels", 4)),
            target_len=int(payload.get("target_len", 5000)),
            beta_kl=float(payload.get("beta_kl", 1e-4)),
            missing_lead_weight=float(payload.get("missing_lead_weight", 1.0)),
        )
    elif model_id.startswith("factorial_ecg_aim_"):
        from unified_latents.engineering.experimental.alitok_vae_exp import build_alitok_vae_1d

        architecture = "ecg_aim"
        reference_model = build_alitok_vae_1d(
            architecture="ecg_aim_v1",
            target_len=int(payload.get("target_len", 5000)),
            patch_size=int(payload.get("alitok_patch_size", 25)),
            encoder_depth=int(payload.get("alitok_encoder_depth", 8)),
            decoder_depth=int(payload.get("alitok_decoder_depth", 4)),
        )
    else:
        from scripts.train_mcma_3lead import MCMAModel

        architecture = "unet"
        reference_model = MCMAModel(in_channels=3, out_channels=12)
    reference_state = reference_model.state_dict()
    reference_model.load_state_dict(states, strict=True)
    state_schema = [
        {
            "key": key,
            "shape": list(states[key].shape),
            "stored_dtype": str(states[key].dtype),
            "reference_shape": list(reference_state[key].shape),
            "reference_dtype": str(reference_state[key].dtype),
        }
        for key in sorted(states)
    ]
    state_schema_sha256 = hashlib.sha256(
        json.dumps(
            state_schema, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    sidecar_path = DEFAULT_CHECKPOINT_DIR / Path(row["filename"]).with_suffix(
        ".metadata.json"
    )
    metadata: dict[str, Any] = {}
    if sidecar_path.exists():
        metadata = json.loads(sidecar_path.read_text())
    metadata.update(
        {
            "schema_version": 3,
            "run_name": row["model_id"],
            "family": metadata.get("family", architecture),
            "seed": row["seed"],
            "factorial_mask": row["factorial_mask"],
            "checkpoint": str((DEFAULT_CHECKPOINT_DIR / row["filename"]).resolve()),
            "checkpoint_size_bytes": row["size_bytes"],
            "checkpoint_sha256": row["sha256"],
            "state_precision": sorted({str(tensor.dtype) for tensor in states.values()}),
            "state_tensor_count": len(states),
            "state_schema_sha256": state_schema_sha256,
            "structured_checkpoint": structured,
            "split_inventory_name_size_sha256": provenance.get(
                "split_inventory_name_size_sha256"
            ),
            "split_content_roots": provenance.get("split_content_roots"),
            "preprocessing": provenance.get("preprocessing"),
            "architecture_revision": provenance.get("architecture_revision"),
            "architecture_provenance": provenance.get("architecture_provenance"),
            "checkpoint_selector": provenance.get("checkpoint_selector"),
            "training_contract": provenance.get("training_contract"),
            "provenance": provenance,
        }
    )
    temporary = sidecar_path.with_suffix(sidecar_path.suffix + ".tmp")
    temporary.write_text(json.dumps(metadata, indent=2, allow_nan=False))
    os.replace(temporary, sidecar_path)
    connection.execute(
        """
        UPDATE checkpoints
        SET metadata_json=?, payload_validated_at=?, payload_tensor_count=?,
            payload_factorial_mask=?, payload_seed=?,
            payload_state_schema_sha256=?, updated_at=?
        WHERE model_id=?
        """,
        (
            json.dumps(metadata, allow_nan=False),
            utc_now(),
            len(states),
            str(embedded_mask),
            int(embedded_seed),
            state_schema_sha256,
            utc_now(),
            row["model_id"],
        ),
    )
    connection.commit()


def upload_one(
    connection: sqlite3.Connection,
    model_id: str,
    *,
    evict: bool,
) -> None:
    repo, tag = read_settings(connection)
    row = connection.execute(
        "SELECT * FROM checkpoints WHERE model_id=?", (model_id,)
    ).fetchone()
    if row is None:
        raise KeyError(f"Unknown model id {model_id!r}; run catalog first")
    local_path = Path(row["local_path"]) if row["local_path"] else None
    if local_path is not None and local_path.exists():
        enrich_sidecar(connection, row, local_path)
    asset = release_asset(repo, tag, row["asset_name"] or row["filename"])
    if asset is None:
        if local_path is None or not local_path.exists():
            raise FileNotFoundError(f"No local or remote bytes for {model_id}")
        connection.execute(
            "UPDATE checkpoints SET status='uploading', error=NULL, updated_at=? WHERE model_id=?",
            (utc_now(), model_id),
        )
        connection.commit()
        run(
            [
                "gh",
                "release",
                "upload",
                tag,
                str(local_path),
                "--repo",
                repo,
            ]
        )
        asset = release_asset(repo, tag, row["filename"])
        if asset is None:
            raise RuntimeError(f"Uploaded asset cannot be found for {model_id}")
    verify_asset(row, asset)
    verify_remote_round_trip(repo, row, asset, connection)
    now = utc_now()
    connection.execute(
        """
        UPDATE checkpoints SET
            release_tag=?, asset_id=?, asset_name=?, asset_size_bytes=?,
            asset_digest=?, asset_state=?, uploaded_at=COALESCE(uploaded_at, ?),
            remote_verified_at=?, round_trip_verified_at=?,
            status='remote_verified', error=NULL, updated_at=?
        WHERE model_id=?
        """,
        (
            tag,
            asset["id"],
            asset["name"],
            asset["size"],
            asset.get("digest"),
            asset.get("state"),
            now,
            now,
            now,
            now,
            model_id,
        ),
    )
    connection.commit()
    if evict:
        if local_path is None or not local_path.exists():
            return
        if sha256_file(local_path) != row["sha256"]:
            raise RuntimeError(f"Local bytes changed before eviction: {local_path}")
        local_path.unlink()
        connection.execute(
            """
            UPDATE checkpoints
            SET local_path=NULL, local_mtime_ns=NULL,
                status='remote_verified', updated_at=?
            WHERE model_id=?
            """,
            (utc_now(), model_id),
        )
        connection.commit()


def export_catalog(connection: sqlite3.Connection, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        dict(row)
        for row in connection.execute(
            """
            SELECT model_id, factorial_mask, seed, filename, size_bytes, sha256,
                   metadata_json, queue_completed_at, release_tag, asset_id, asset_name,
                   asset_size_bytes, asset_digest, remote_verified_at, status
                   , round_trip_verified_at, payload_validated_at,
                   payload_tensor_count, payload_factorial_mask, payload_seed
                   , payload_state_schema_sha256
            FROM checkpoints ORDER BY factorial_mask, seed
            """
        )
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, path)


def publish_catalog_snapshot(connection: sqlite3.Connection, path: Path) -> None:
    """Back up the small recovery index beside the exact checkpoint assets."""
    repo, tag = read_settings(connection)
    export_catalog(connection, path)
    run(
        [
            "gh",
            "release",
            "upload",
            tag,
            str(path),
            "--repo",
            repo,
            "--clobber",
        ]
    )


def archive_row_ready(row: sqlite3.Row) -> bool:
    common_ready = (
        row["status"] in {"remote_verified", "cached"}
        and row["asset_id"] is not None
        and row["asset_state"] == "uploaded"
        and row["asset_size_bytes"] == row["size_bytes"]
        and row["asset_digest"] == f"sha256:{row['sha256']}"
        and row["remote_verified_at"] is not None
        and row["round_trip_verified_at"] is not None
        and row["payload_validated_at"] is not None
        and row["payload_factorial_mask"] == row["factorial_mask"]
        and row["payload_seed"] == row["seed"]
        and row["payload_tensor_count"] is not None
        and row["payload_tensor_count"] > 0
        and row["payload_state_schema_sha256"] is not None
    )
    if not row["model_id"].startswith("f_"):
        return common_ready
    contract = json.loads(
        (ROOT / "refine-logs/factorial_training_contract.json").read_text()
    )
    return (
        common_ready
        and row["payload_tensor_count"] == 162
        and row["payload_state_schema_sha256"] == contract["state_schema_sha256"]
    )


def materialize(
    connection: sqlite3.Connection,
    model_id: str,
    cache_dir: Path,
) -> Path:
    repo, _ = read_settings(connection)
    row = connection.execute(
        "SELECT * FROM checkpoints WHERE model_id=?", (model_id,)
    ).fetchone()
    if row is None:
        raise KeyError(f"Unknown model id {model_id!r}")
    if row["status"] == "local":
        candidate = Path(row["local_path"]) if row["local_path"] else None
        if (
            candidate is None
            or not candidate.is_file()
            or candidate.stat().st_size != row["size_bytes"]
            or sha256_file(candidate) != row["sha256"]
        ):
            raise RuntimeError(f"{model_id} local bytes do not match the catalog")
        enrich_sidecar(connection, row, candidate)
        connection.execute(
            """
            UPDATE checkpoints
            SET local_path=?, local_mtime_ns=?, last_materialized_at=?,
                status='local', updated_at=?
            WHERE model_id=? AND sha256=?
            """,
            (
                str(candidate.resolve()),
                candidate.stat().st_mtime_ns,
                utc_now(),
                utc_now(),
                model_id,
                row["sha256"],
            ),
        )
        if connection.execute("SELECT changes()").fetchone()[0] != 1:
            connection.rollback()
            raise RuntimeError(f"{model_id} generation changed during materialization")
        connection.commit()
        return candidate.resolve()
    if not archive_row_ready(row):
        raise RuntimeError(
            f"{model_id} is not eligible for remote materialization "
            f"(status={row['status']!r})"
        )
    for candidate in (
        Path(row["local_path"]) if row["local_path"] else None,
        cache_dir / row["filename"],
    ):
        if candidate and candidate.exists():
            if (
                candidate.stat().st_size == row["size_bytes"]
                and sha256_file(candidate) == row["sha256"]
            ):
                connection.execute(
                    """
                    UPDATE checkpoints
                    SET local_path=?, local_mtime_ns=?, last_materialized_at=?,
                        status='cached',
                        error=NULL, updated_at=?
                    WHERE model_id=? AND sha256=?
                    """,
                    (
                        str(candidate.resolve()),
                        candidate.stat().st_mtime_ns,
                        utc_now(),
                        utc_now(),
                        model_id,
                        row["sha256"],
                    ),
                )
                if connection.execute("SELECT changes()").fetchone()[0] != 1:
                    connection.rollback()
                    if candidate.parent.resolve() == cache_dir.resolve():
                        candidate.unlink(missing_ok=True)
                    raise RuntimeError(
                        f"{model_id} generation changed during materialization"
                    )
                connection.commit()
                return candidate.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / row["filename"]
    temporary = cache_dir / f".{row['filename']}.{os.getpid()}.part"
    try:
        download_asset_to_file(repo, int(row["asset_id"]), temporary)
        if (
            temporary.stat().st_size != row["size_bytes"]
            or sha256_file(temporary) != row["sha256"]
        ):
            raise RuntimeError(f"Downloaded cache failed SHA-256 verification: {model_id}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    connection.execute(
        """
        UPDATE checkpoints
        SET local_path=?, local_mtime_ns=?, last_materialized_at=?, status='cached',
            remote_verified_at=COALESCE(remote_verified_at, ?),
            round_trip_verified_at=COALESCE(round_trip_verified_at, ?),
            error=NULL, updated_at=?
        WHERE model_id=? AND sha256=?
        """,
        (
            str(destination.resolve()),
            destination.stat().st_mtime_ns,
            utc_now(),
            utc_now(),
            utc_now(),
            utc_now(),
            model_id,
            row["sha256"],
        ),
    )
    if connection.execute("SELECT changes()").fetchone()[0] != 1:
        connection.rollback()
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"{model_id} generation changed during materialization")
    connection.commit()
    return destination.resolve()


def prune_cache(connection: sqlite3.Connection, cache_dir: Path, max_gib: float) -> int:
    limit = int(max_gib * 1024**3)
    files = [path for path in cache_dir.glob("factorial_*.pt") if path.is_file()]
    total = sum(path.stat().st_size for path in files)
    removed = 0
    rows = connection.execute(
        """
        SELECT *
        FROM checkpoints
        WHERE status='cached'
        ORDER BY COALESCE(last_materialized_at, created_at) ASC
        """
    ).fetchall()
    for row in rows:
        path = Path(row["local_path"]) if row["local_path"] else None
        if path is None or not path.exists():
            if archive_row_ready(row):
                connection.execute(
                    """
                    UPDATE checkpoints
                    SET local_path=NULL, local_mtime_ns=NULL,
                        status='remote_verified', updated_at=?
                    WHERE model_id=?
                    """,
                    (utc_now(), row["model_id"]),
                )
            continue
        if total <= limit:
            continue
        if path.parent.resolve() != cache_dir.resolve():
            continue
        if not archive_row_ready(row):
            raise RuntimeError(f"Refusing to prune unverified cache row {row['model_id']}")
        if path.stat().st_size != row["size_bytes"] or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"Refusing to prune changed cache bytes for {row['model_id']}")
        size = path.stat().st_size
        path.unlink()
        total -= size
        removed += 1
        connection.execute(
            """
            UPDATE checkpoints
            SET local_path=NULL, local_mtime_ns=NULL,
                status='remote_verified', updated_at=?
            WHERE model_id=?
            """,
            (utc_now(), row["model_id"]),
        )
    connection.commit()
    return removed


def metadata_is_complete(row: sqlite3.Row) -> bool:
    if not row["metadata_json"]:
        return False
    try:
        metadata = json.loads(row["metadata_json"])
    except json.JSONDecodeError:
        return False
    return (
        metadata.get("schema_version", 0) >= 3
        and metadata.get("checkpoint_sha256") == row["sha256"]
        and bool(metadata.get("split_inventory_name_size_sha256"))
        and bool(metadata.get("split_content_roots"))
        and bool(metadata.get("preprocessing"))
        and bool(metadata.get("architecture_provenance"))
    )


def backfill_metadata(
    connection: sqlite3.Connection,
    cache_dir: Path,
    limit: int | None = None,
) -> dict[str, int]:
    rows = connection.execute(
        "SELECT * FROM checkpoints ORDER BY factorial_mask, seed"
    ).fetchall()
    pending = [row for row in rows if not metadata_is_complete(row)]
    if limit is not None:
        pending = pending[:limit]
    enriched = 0
    downloaded = 0
    for index, row in enumerate(pending, start=1):
        print(
            f"[{index}/{len(pending)}] enriching metadata {row['model_id']}",
            flush=True,
        )
        original_local = (
            Path(row["local_path"])
            if row["local_path"] and Path(row["local_path"]).exists()
            else None
        )
        if original_local is None:
            local_path = materialize(connection, row["model_id"], cache_dir)
            downloaded += 1
        else:
            local_path = original_local
        current = connection.execute(
            "SELECT * FROM checkpoints WHERE model_id=?", (row["model_id"],)
        ).fetchone()
        enrich_sidecar(connection, current, local_path)
        enriched += 1
        if original_local is None:
            current = connection.execute(
                "SELECT * FROM checkpoints WHERE model_id=?", (row["model_id"],)
            ).fetchone()
            if current["asset_id"] is None or current["remote_verified_at"] is None:
                raise RuntimeError(
                    f"Cannot discard metadata cache without verified asset: {row['model_id']}"
                )
            if sha256_file(local_path) != current["sha256"]:
                raise RuntimeError(
                    f"Metadata cache changed before removal: {row['model_id']}"
                )
            local_path.unlink()
            connection.execute(
                """
                UPDATE checkpoints
                SET local_path=NULL, local_mtime_ns=NULL,
                    status='remote_verified', updated_at=?
                WHERE model_id=?
                """,
                (utc_now(), row["model_id"]),
            )
            connection.commit()
    return {"pending": len(pending), "enriched": enriched, "downloaded": downloaded}


def status(connection: sqlite3.Connection) -> dict[str, Any]:
    counts = {
        row["status"]: row["count"]
        for row in connection.execute(
            "SELECT status, COUNT(*) AS count FROM checkpoints GROUP BY status"
        )
    }
    totals = connection.execute(
        """
        SELECT COUNT(*) AS models,
               COALESCE(SUM(size_bytes), 0) AS logical_bytes,
               COALESCE(SUM(CASE
                   WHEN status IN ('remote_verified', 'cached') THEN size_bytes
                   ELSE 0 END), 0) AS inference_addressable_bytes,
               COALESCE(SUM(CASE
                   WHEN status = 'error' THEN size_bytes ELSE 0 END), 0)
                   AS error_generation_bytes,
               COALESCE(SUM(CASE WHEN local_path IS NOT NULL THEN size_bytes ELSE 0 END), 0)
                   AS local_bytes
        FROM checkpoints
        """
    ).fetchone()
    return {"counts": counts, **dict(totals)}


def load_checkpoint(
    model_id: str,
    *,
    db_path: str | Path = DEFAULT_DB,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    map_location: str = "cpu",
    weights_only: bool = False,
) -> Any:
    """Materialize, verify, and torch-load an exact checkpoint by model id."""
    payload, _ = load_checkpoint_with_identity(
        model_id,
        db_path=db_path,
        cache_dir=cache_dir,
        map_location=map_location,
        weights_only=weights_only,
    )
    return payload


def load_checkpoint_with_identity(
    model_id: str,
    *,
    db_path: str | Path = DEFAULT_DB,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    map_location: str = "cpu",
    weights_only: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Load one immutable generation and return its lock-consistent identity."""
    import torch

    resolved_db = Path(db_path)
    with store_lock(resolved_db):
        connection = connect(resolved_db)
        try:
            path = materialize(connection, model_id, Path(cache_dir))
            row = connection.execute(
                """
                SELECT model_id, sha256, size_bytes, status
                FROM checkpoints WHERE model_id=?
                """,
                (model_id,),
            ).fetchone()
            if (
                row is None
                or path.stat().st_size != row["size_bytes"]
                or sha256_file(path) != row["sha256"]
            ):
                raise RuntimeError(
                    f"{model_id} catalog generation changed before deserialization"
                )
            payload = torch.load(
                path, map_location=map_location, weights_only=weights_only
            )
            identity = dict(row)
        finally:
            connection.close()
    return payload, identity


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init")

    catalog = subparsers.add_parser("catalog")
    catalog.add_argument("--state", type=Path, default=DEFAULT_STATE)
    catalog.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)

    upload = subparsers.add_parser("upload")
    upload.add_argument("model_id")
    upload.add_argument("--evict", action="store_true")

    archive = subparsers.add_parser("archive-completed")
    archive.add_argument("--state", type=Path, default=DEFAULT_STATE)
    archive.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    archive.add_argument("--evict", action="store_true")
    archive.add_argument("--limit", type=int)
    archive.add_argument("--watch", action="store_true")
    archive.add_argument("--poll-seconds", type=int, default=60)

    get = subparsers.add_parser("materialize")
    get.add_argument("model_id")
    get.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)

    prune = subparsers.add_parser("prune-cache")
    prune.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    prune.add_argument("--max-gib", type=float, default=2.0)

    backfill = subparsers.add_parser("backfill-metadata")
    backfill.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    backfill.add_argument("--limit", type=int)

    subparsers.add_parser("status")
    return parser


def archive_completed_pass(
    connection: sqlite3.Connection, args: argparse.Namespace
) -> None:
    """Archive one queue snapshot while the caller holds the writer lock."""
    catalog_completed(connection, args.state, args.checkpoint_dir)
    if args.evict:
        selection = """
            SELECT model_id FROM checkpoints
            WHERE status='local'
               OR (
                   status='error'
                   AND (
                       local_path IS NOT NULL
                       OR (
                           asset_id IS NOT NULL
                           AND asset_name NOT LIKE 'QUARANTINED_%'
                       )
                   )
               )
               OR (
                   status='remote_verified'
                   AND (
                       round_trip_verified_at IS NULL
                       OR payload_state_schema_sha256 IS NULL
                   )
                   AND asset_id IS NOT NULL
               )
            ORDER BY factorial_mask, seed
        """
    else:
        selection = """
            SELECT model_id FROM checkpoints
            WHERE status IN ('local', 'error')
            ORDER BY factorial_mask, seed
        """
    rows = connection.execute(selection).fetchall()
    if args.limit is not None:
        rows = rows[: args.limit]
    for index, row in enumerate(rows, start=1):
        model_id = row["model_id"]
        print(f"[{index}/{len(rows)}] archiving {model_id}", flush=True)
        try:
            upload_one(connection, model_id, evict=args.evict)
        except Exception as error:
            connection.execute(
                """
                UPDATE checkpoints
                SET status='error', error=?, updated_at=?
                WHERE model_id=?
                """,
                (repr(error), utc_now(), model_id),
            )
            connection.commit()
            print(
                f"archive error for {model_id}: {error!r}",
                file=sys.stderr,
                flush=True,
            )
            if not args.watch:
                raise
        export_catalog(connection, args.db.with_name("catalog.jsonl"))
        if index % 10 == 0 or index == len(rows):
            publish_catalog_snapshot(connection, args.db.with_name("catalog.jsonl"))
    if rows:
        # The compatibility audit is a reporting gate over the entire catalog.
        # It deliberately exits non-zero when *any* historical checkpoint is
        # incompatible.  That verdict must not terminate watch mode after the
        # newly archived checkpoint has already passed its own byte-identical
        # remote and payload validation; otherwise one quarantined/legacy
        # catalog entry prevents every later checkpoint from being archived and
        # eventually fills the training filesystem.
        try:
            run(
                [
                    sys.executable,
                    str(ROOT / "scripts/audit_factorial_compatibility.py"),
                    "--db",
                    str(args.db),
                    "--output",
                    str(args.db.with_name("compatibility_audit.json")),
                ],
                capture=False,
            )
        except subprocess.CalledProcessError as error:
            if not args.watch:
                raise
            print(
                "compatibility audit reported incompatible catalog entries "
                f"(exit {error.returncode}); archive watch will continue. "
                "See compatibility_audit.json for the model-level verdicts.",
                file=sys.stderr,
                flush=True,
            )
    print(json.dumps(status(connection)), flush=True)


def archive_watch(args: argparse.Namespace) -> None:
    """Poll without monopolizing the writer lock between archive passes."""
    while True:
        with store_lock(args.db):
            connection = connect(args.db)
            try:
                set_settings(connection, args.repo, args.tag)
                connection.commit()
                archive_completed_pass(connection, args)
            finally:
                connection.close()
        time.sleep(args.poll_seconds)


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "archive-completed" and args.watch:
        archive_watch(args)
        return
    # Reads and inference materialization stay available while the one archival
    # writer holds its dedicated process lock during remote transfers.
    needs_writer_lock = args.command != "status"
    lock_context = store_lock(args.db) if needs_writer_lock else contextlib.nullcontext()
    with lock_context:
        connection = connect(args.db)
        try:
            set_settings(connection, args.repo, args.tag)
            connection.commit()
            if args.command == "init":
                print(json.dumps({"db": str(args.db), "repo": args.repo, "tag": args.tag}))
            elif args.command == "catalog":
                count = catalog_completed(connection, args.state, args.checkpoint_dir)
                export_catalog(connection, args.db.with_name("catalog.jsonl"))
                print(json.dumps({"cataloged_local_completed": count, **status(connection)}))
            elif args.command == "upload":
                upload_one(
                    connection,
                    args.model_id,
                    evict=args.evict,
                )
                export_catalog(connection, args.db.with_name("catalog.jsonl"))
                publish_catalog_snapshot(
                    connection, args.db.with_name("catalog.jsonl")
                )
                print(json.dumps(status(connection)))
            elif args.command == "archive-completed":
                archive_completed_pass(connection, args)
            elif args.command == "materialize":
                print(materialize(connection, args.model_id, args.cache_dir))
            elif args.command == "prune-cache":
                removed = prune_cache(connection, args.cache_dir, args.max_gib)
                print(json.dumps({"removed": removed, **status(connection)}))
            elif args.command == "backfill-metadata":
                result = backfill_metadata(connection, args.cache_dir, args.limit)
                export_catalog(connection, args.db.with_name("catalog.jsonl"))
                print(json.dumps({**result, **status(connection)}))
            elif args.command == "status":
                print(json.dumps(status(connection), indent=2))
        finally:
            connection.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("checkpoint-store command stopped cleanly", file=sys.stderr)
