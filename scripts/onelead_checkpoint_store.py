#!/usr/bin/env python3
"""Verified, separately cataloged archive for one-lead experiment checkpoints."""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import os
import shlex
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_DB = ROOT / "results/onelead_checkpoint_store/catalog.sqlite"
DEFAULT_EXPORT = ROOT / "results/onelead_checkpoint_store/catalog.jsonl"
DEFAULT_STATE = ROOT / "refine-logs/queue_spatial_1lead/queue_state.json"
DEFAULT_CACHE = ROOT / "checkpoints/onelead_cache"
DEFAULT_REPO = "whiteblaze143/benchmarking_loss_functions_ecg_reconstruction"
DEFAULT_TAG = "spatial-1lead-checkpoints-v1"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=60)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS checkpoints(
            model_id TEXT PRIMARY KEY,
            factorial_mask TEXT NOT NULL,
            seed INTEGER NOT NULL,
            observed_leads_json TEXT NOT NULL,
            architecture TEXT NOT NULL,
            filename TEXT NOT NULL UNIQUE,
            local_path TEXT,
            size_bytes INTEGER NOT NULL,
            sha256 TEXT NOT NULL,
            queue_completed_at TEXT,
            queue_command TEXT NOT NULL,
            payload_tensor_count INTEGER,
            payload_state_schema_sha256 TEXT,
            payload_validated_at TEXT,
            release_tag TEXT,
            asset_id INTEGER,
            asset_name TEXT,
            asset_size_bytes INTEGER,
            asset_digest TEXT,
            remote_verified_at TEXT,
            round_trip_verified_at TEXT,
            status TEXT NOT NULL CHECK(status IN ('local','uploading','remote_verified','cached','error')),
            error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS onelead_checkpoint_status_idx ON checkpoints(status);
        """
    )
    return connection


def command_options(command: str) -> dict[str, Any]:
    tokens = shlex.split(command)
    result: dict[str, Any] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.startswith("--"):
            key = token[2:].replace("-", "_")
            if index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
                result[key] = tokens[index + 1]
                index += 2
                continue
            result[key] = True
        index += 1
    return result


def completed_jobs(state: Path) -> list[dict[str, Any]]:
    payload = json.loads(state.read_text())
    return [job for job in payload.get("jobs", []) if job.get("status") == "completed"]


def set_settings(connection: sqlite3.Connection, repo: str, tag: str) -> None:
    connection.executemany(
        "INSERT INTO settings VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (("repo", repo), ("release_tag", tag)),
    )
    connection.commit()


def settings(connection: sqlite3.Connection) -> tuple[str, str]:
    values = dict(connection.execute("SELECT key,value FROM settings"))
    return values.get("repo", DEFAULT_REPO), values.get("release_tag", DEFAULT_TAG)


def validate_payload(path: Path, row: sqlite3.Row | dict[str, Any]) -> tuple[int, str]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model_state_dict"), dict):
        raise RuntimeError(f"{path.name}: checkpoint is not a structured model payload")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise RuntimeError(f"{path.name}: missing provenance")
    if str(provenance.get("factorial_mask")) != str(row["factorial_mask"]):
        raise RuntimeError(f"{path.name}: factorial mask mismatch")
    if int(provenance.get("seed")) != int(row["seed"]):
        raise RuntimeError(f"{path.name}: seed mismatch")
    observed = provenance.get("preprocessing", {}).get("observed_leads")
    if observed != json.loads(row["observed_leads_json"]):
        raise RuntimeError(f"{path.name}: observed-lead mismatch")
    if payload.get("architecture") != row["architecture"]:
        raise RuntimeError(f"{path.name}: architecture mismatch")
    states = {
        key.removeprefix("_orig_mod."): value
        for key, value in payload["model_state_dict"].items()
    }
    if not states or not all(isinstance(value, torch.Tensor) for value in states.values()):
        raise RuntimeError(f"{path.name}: invalid tensor state")
    if not all(torch.isfinite(value).all().item() for value in states.values()):
        raise RuntimeError(f"{path.name}: non-finite model weights")
    model_architecture = str(payload["architecture"])
    if model_architecture == "unet":
        from scripts.train_mcma_3lead import MCMAModel

        model = MCMAModel(in_channels=12, out_channels=12)
    elif model_architecture == "msvae":
        from unified_latents.engineering.experimental.Multi_Scale_VAE import WearECGVAE

        model = WearECGVAE(
            latent_channels=int(payload.get("latent_channels", 4)),
            target_len=int(payload.get("target_len", 5000)),
            beta_kl=1e-4,
            missing_lead_weight=1.0,
        )
    else:
        from unified_latents.engineering.experimental.aim_1_lead import build_alitok_vae_1d

        architecture = payload.get("alitok_architecture")
        if not isinstance(architecture, str):
            raise RuntimeError(f"{path.name}: missing AliTok architecture identifier")
        model = build_alitok_vae_1d(
            architecture=architecture,
            target_len=int(payload.get("target_len", 5000)),
            patch_size=int(payload.get("alitok_patch_size", 25)),
            encoder_depth=int(payload.get("alitok_encoder_depth", 8)),
            decoder_depth=int(payload.get("alitok_decoder_depth", 4)),
            lead_conditioning_mode=str(payload.get("lead_conditioning_mode", "learned")),
            use_learned_lead_id=bool(payload.get("use_learned_lead_id", False)),
            use_relative_geometry=bool(payload.get("use_relative_geometry", False)),
            use_spatial_film=bool(payload.get("use_spatial_film", False)),
            spatial_gain_init=float(payload.get("spatial_gain_init", 0.1)),
            geometry_control=str(payload.get("geometry_control", "standard")),
        )
    model.load_state_dict(states, strict=True)
    schema = [
        (key, list(value.shape), str(value.dtype))
        for key, value in sorted(states.items())
    ]
    schema_hash = hashlib.sha256(
        json.dumps(schema, separators=(",", ":")).encode()
    ).hexdigest()
    count = len(states)
    del model, states, payload
    gc.collect()
    return count, schema_hash


def catalog(connection: sqlite3.Connection, state: Path) -> int:
    count = 0
    now = utc_now()
    for job in completed_jobs(state):
        options = command_options(job["cmd"])
        required = ("checkpoint_path", "factorial_mask", "seed", "observed_leads", "architecture")
        missing = [key for key in required if key not in options]
        if missing:
            raise RuntimeError(f"{job['id']}: missing command options {missing}")
        path = (ROOT / options["checkpoint_path"]).resolve()
        existing = connection.execute(
            "SELECT * FROM checkpoints WHERE model_id=?", (job["id"],)
        ).fetchone()
        if not path.is_file():
            if existing and existing["status"] in {"remote_verified", "cached"}:
                continue
            raise FileNotFoundError(f"completed job has no checkpoint: {path}")
        size = path.stat().st_size
        digest = sha256_file(path)
        if existing and existing["status"] in {"remote_verified", "cached"} and (
            existing["size_bytes"] != size or existing["sha256"] != digest
        ):
            raise RuntimeError(f"refusing changed bytes for verified {job['id']}")
        observed = [int(value) for value in str(options["observed_leads"]).split(",")]
        row = {
            "factorial_mask": str(options["factorial_mask"]),
            "seed": int(options["seed"]),
            "observed_leads_json": json.dumps(observed, separators=(",", ":")),
            "architecture": str(options["architecture"]),
        }
        tensors, schema_hash = validate_payload(path, row)
        connection.execute(
            """
            INSERT INTO checkpoints(
                model_id,factorial_mask,seed,observed_leads_json,architecture,
                filename,local_path,size_bytes,sha256,queue_completed_at,queue_command,
                payload_tensor_count,payload_state_schema_sha256,payload_validated_at,
                status,created_at,updated_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,'local',?,?)
            ON CONFLICT(model_id) DO UPDATE SET
                local_path=excluded.local_path,size_bytes=excluded.size_bytes,
                sha256=excluded.sha256,queue_completed_at=excluded.queue_completed_at,
                queue_command=excluded.queue_command,
                payload_tensor_count=excluded.payload_tensor_count,
                payload_state_schema_sha256=excluded.payload_state_schema_sha256,
                payload_validated_at=excluded.payload_validated_at,
                status=CASE WHEN checkpoints.sha256=excluded.sha256 AND checkpoints.status IN ('remote_verified','cached') THEN 'cached' ELSE 'local' END,
                error=NULL,updated_at=excluded.updated_at
            """,
            (job["id"], row["factorial_mask"], row["seed"], row["observed_leads_json"],
             row["architecture"], path.name, str(path), size, digest, job.get("completed"),
             job["cmd"], tensors, schema_hash, now, now, now),
        )
        count += 1
    connection.commit()
    return count


def catalog_legacy(connection: sqlite3.Connection, checkpoint_dir: Path) -> int:
    """Catalog completed seed-201 one-lead checkpoints using embedded identity."""
    import torch

    count = 0
    now = utc_now()
    for path in sorted(checkpoint_dir.glob("1lead_*_s201_l*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        provenance = payload.get("provenance", {}) if isinstance(payload, dict) else {}
        preprocessing = provenance.get("preprocessing", {}) if isinstance(provenance, dict) else {}
        observed = preprocessing.get("observed_leads")
        architecture = payload.get("architecture") if isinstance(payload, dict) else None
        if (
            not isinstance(observed, list)
            or len(observed) != 1
            or architecture not in {"unet", "msvae", "ecg_aim"}
            or int(provenance.get("seed", -1)) != 201
        ):
            raise RuntimeError(f"{path.name}: incomplete legacy one-lead identity")
        model_id = path.stem
        existing = connection.execute(
            "SELECT * FROM checkpoints WHERE model_id=?", (model_id,)
        ).fetchone()
        size = path.stat().st_size
        digest = sha256_file(path)
        if existing and existing["status"] in {"remote_verified", "cached"} and (
            existing["size_bytes"] != size or existing["sha256"] != digest
        ):
            raise RuntimeError(f"refusing changed bytes for verified {model_id}")
        identity = {
            "factorial_mask": str(provenance["factorial_mask"]),
            "seed": 201,
            "observed_leads_json": json.dumps([int(observed[0])], separators=(",", ":")),
            "architecture": str(architecture),
        }
        tensors, schema_hash = validate_payload(path, identity)
        connection.execute(
            """
            INSERT INTO checkpoints(
                model_id,factorial_mask,seed,observed_leads_json,architecture,
                filename,local_path,size_bytes,sha256,queue_completed_at,queue_command,
                payload_tensor_count,payload_state_schema_sha256,payload_validated_at,
                status,created_at,updated_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,'local',?,?)
            ON CONFLICT(model_id) DO UPDATE SET
                local_path=excluded.local_path,size_bytes=excluded.size_bytes,
                sha256=excluded.sha256,queue_command=excluded.queue_command,
                payload_tensor_count=excluded.payload_tensor_count,
                payload_state_schema_sha256=excluded.payload_state_schema_sha256,
                payload_validated_at=excluded.payload_validated_at,
                status=CASE WHEN checkpoints.sha256=excluded.sha256 AND checkpoints.status IN ('remote_verified','cached') THEN 'cached' ELSE 'local' END,
                error=NULL,updated_at=excluded.updated_at
            """,
            (model_id, identity["factorial_mask"], identity["seed"],
             identity["observed_leads_json"], identity["architecture"], path.name,
             str(path.resolve()), size, digest, None,
             "legacy seed-201 checkpoint; completion and identity verified from embedded payload",
             tensors, schema_hash, now, now, now),
        )
        connection.commit()
        count += 1
        del payload
        gc.collect()
    return count


def release(repo: str, tag: str) -> dict[str, Any]:
    output = subprocess.run(
        ["gh", "api", "--paginate", f"repos/{repo}/releases?per_page=100"],
        cwd=ROOT, text=True, capture_output=True, check=True,
    ).stdout
    for candidate in json.loads(output):
        if candidate["tag_name"] == tag:
            return candidate
    raise RuntimeError(f"release tag {tag!r} not found in {repo}")


def asset_for(repo: str, tag: str, name: str) -> dict[str, Any] | None:
    for asset in release(repo, tag).get("assets", []):
        if asset["name"] == name:
            return asset
    return None


def download(repo: str, asset_id: str, destination: Path) -> None:
    with destination.open("wb") as handle:
        subprocess.run(
            ["gh", "api", "-H", "Accept: application/octet-stream",
             f"repos/{repo}/releases/assets/{asset_id}"],
            cwd=ROOT, stdout=handle, stderr=subprocess.PIPE, check=True,
        )


def verify_asset(row: sqlite3.Row, asset: dict[str, Any]) -> None:
    if asset.get("state") != "uploaded":
        raise RuntimeError(f"{row['model_id']}: remote asset is not uploaded")
    if int(asset["size"]) != int(row["size_bytes"]):
        raise RuntimeError(f"{row['model_id']}: remote size mismatch")
    if asset.get("digest") != f"sha256:{row['sha256']}":
        raise RuntimeError(f"{row['model_id']}: remote digest mismatch")


def upload(connection: sqlite3.Connection, model_id: str, evict: bool) -> None:
    repo, tag = settings(connection)
    row = connection.execute("SELECT * FROM checkpoints WHERE model_id=?", (model_id,)).fetchone()
    if row is None:
        raise KeyError(model_id)
    path = Path(row["local_path"]) if row["local_path"] else None
    asset = asset_for(repo, tag, row["filename"])
    if asset is None:
        if path is None or not path.is_file():
            raise FileNotFoundError(f"{model_id}: no local checkpoint")
        connection.execute("UPDATE checkpoints SET status='uploading',updated_at=? WHERE model_id=?", (utc_now(), model_id))
        connection.commit()
        subprocess.run(
            ["gh", "release", "upload", tag, str(path), "--repo", repo],
            cwd=ROOT, text=True, check=True,
        )
        asset = asset_for(repo, tag, row["filename"])
        if asset is None:
            raise RuntimeError(f"{model_id}: upload is not visible")
    verify_asset(row, asset)
    with tempfile.TemporaryDirectory(prefix="onelead-roundtrip-") as directory:
        downloaded = Path(directory) / row["filename"]
        download(repo, str(asset["id"]), downloaded)
        if downloaded.stat().st_size != row["size_bytes"] or sha256_file(downloaded) != row["sha256"]:
            raise RuntimeError(f"{model_id}: downloaded bytes failed verification")
        tensors, schema_hash = validate_payload(downloaded, row)
        if tensors != row["payload_tensor_count"] or schema_hash != row["payload_state_schema_sha256"]:
            raise RuntimeError(f"{model_id}: downloaded payload schema mismatch")
    now = utc_now()
    connection.execute(
        """UPDATE checkpoints SET release_tag=?,asset_id=?,asset_name=?,asset_size_bytes=?,
           asset_digest=?,remote_verified_at=?,round_trip_verified_at=?,status='remote_verified',
           error=NULL,updated_at=? WHERE model_id=?""",
        (tag, asset["id"], asset["name"], asset["size"], asset.get("digest"), now, now, now, model_id),
    )
    connection.commit()
    if evict and path and path.is_file():
        if path.stat().st_size != row["size_bytes"] or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"{model_id}: local bytes changed before eviction")
        path.unlink()
        connection.execute(
            "UPDATE checkpoints SET local_path=NULL,status='remote_verified',updated_at=? WHERE model_id=?",
            (utc_now(), model_id),
        )
        connection.commit()


def export(connection: sqlite3.Connection, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [dict(row) for row in connection.execute("SELECT * FROM checkpoints ORDER BY model_id")]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, path)


def publish_export(connection: sqlite3.Connection, path: Path) -> None:
    """Keep a small remote recovery index beside the exact checkpoint assets."""
    repo, tag = settings(connection)
    subprocess.run(
        ["gh", "release", "upload", tag, str(path), "--repo", repo, "--clobber"],
        cwd=ROOT, text=True, check=True,
    )


def archive_pass(connection: sqlite3.Connection, args: argparse.Namespace) -> None:
    catalog(connection, args.state)
    archive_pending(connection, args)


def archive_pending(connection: sqlite3.Connection, args: argparse.Namespace) -> None:
    rows = connection.execute(
        "SELECT model_id FROM checkpoints WHERE status IN ('local','uploading','error') ORDER BY queue_completed_at"
    ).fetchall()
    for row in rows:
        print(json.dumps({"event": "archive_started", "model_id": row["model_id"], "at": utc_now()}), flush=True)
        try:
            upload(connection, row["model_id"], args.evict)
        except Exception as error:
            connection.execute(
                "UPDATE checkpoints SET status='error',error=?,updated_at=? WHERE model_id=?",
                (str(error), utc_now(), row["model_id"]),
            )
            connection.commit()
            export(connection, args.export)
            raise
        export(connection, args.export)
        print(json.dumps({"event": "archive_complete", "model_id": row["model_id"], "at": utc_now()}), flush=True)
    if rows:
        export(connection, args.export)
        publish_export(connection, args.export)


def materialize(connection: sqlite3.Connection, model_id: str, cache: Path) -> Path:
    repo, _ = settings(connection)
    row = connection.execute("SELECT * FROM checkpoints WHERE model_id=?", (model_id,)).fetchone()
    if row is None or row["status"] not in {"remote_verified", "cached"}:
        raise RuntimeError(f"{model_id}: no verified archive generation")
    asset = asset_for(repo, row["release_tag"], row["asset_name"])
    if asset is None:
        raise RuntimeError(f"{model_id}: archive asset is missing")
    verify_asset(row, asset)
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / row["filename"]
    temporary = cache / f".{row['filename']}.{os.getpid()}.part"
    download(repo, str(asset["id"]), temporary)
    if temporary.stat().st_size != row["size_bytes"] or sha256_file(temporary) != row["sha256"]:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"{model_id}: materialized bytes failed verification")
    os.replace(temporary, destination)
    connection.execute(
        "UPDATE checkpoints SET local_path=?,status='cached',updated_at=? WHERE model_id=?",
        (str(destination.resolve()), utc_now(), model_id),
    )
    connection.commit()
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    archive = sub.add_parser("archive-completed")
    archive.add_argument("--state", type=Path, default=DEFAULT_STATE)
    archive.add_argument("--export", type=Path, default=DEFAULT_EXPORT)
    archive.add_argument("--evict", action="store_true")
    archive.add_argument("--watch", action="store_true")
    archive.add_argument("--poll-seconds", type=int, default=30)
    legacy = sub.add_parser("archive-legacy")
    legacy.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    legacy.add_argument("--export", type=Path, default=DEFAULT_EXPORT)
    legacy.add_argument("--evict", action="store_true")
    status = sub.add_parser("status")
    get = sub.add_parser("materialize")
    get.add_argument("model_id")
    get.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()
    connection = connect(args.db)
    set_settings(connection, args.repo, args.tag)
    if args.command == "init":
        print(json.dumps({"db": str(args.db), "repo": args.repo, "tag": args.tag}))
    elif args.command == "status":
        counts = dict(connection.execute("SELECT status,count(*) FROM checkpoints GROUP BY status"))
        totals = connection.execute("SELECT count(*),coalesce(sum(size_bytes),0),coalesce(sum(CASE WHEN local_path IS NOT NULL THEN size_bytes ELSE 0 END),0) FROM checkpoints").fetchone()
        print(json.dumps({"counts": counts, "models": totals[0], "logical_bytes": totals[1], "local_bytes": totals[2]}))
    elif args.command == "materialize":
        print(materialize(connection, args.model_id, args.cache))
    elif args.command == "archive-legacy":
        catalog_legacy(connection, args.checkpoint_dir)
        archive_pending(connection, args)
    else:
        while True:
            archive_pass(connection, args)
            if not args.watch:
                break
            time.sleep(args.poll_seconds)
    connection.close()


if __name__ == "__main__":
    main()
