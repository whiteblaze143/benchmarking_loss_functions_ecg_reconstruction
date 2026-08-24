#!/usr/bin/env python3
"""Quarantine nonzero-exit partial checkpoints and requeue false completions."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "whiteblaze143/benchmarking_loss_functions_ecg_reconstruction"
DEFAULT_TAG = "factorial-checkpoints-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def gh_json(arguments: list[str]) -> Any:
    result = subprocess.run(
        ["gh", *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout)


def release_assets(repo: str, tag: str) -> list[dict[str, Any]]:
    releases = gh_json(
        ["api", "--paginate", f"repos/{repo}/releases?per_page=100"]
    )
    release = next(item for item in releases if item["tag_name"] == tag)
    return release["assets"]


def download_asset(repo: str, asset_id: int, destination: Path) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state",
        type=Path,
        default=ROOT / "refine-logs/queue/queue_state.json",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=ROOT / "results/checkpoint_store/catalog.sqlite",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=ROOT / "refine-logs/queue/logs",
    )
    parser.add_argument(
        "--recovery-root",
        type=Path,
        default=ROOT / "refine-logs/queue/recovery",
    )
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    state = json.loads(args.state.read_text())
    suspects = []
    for job in state["jobs"]:
        if job["status"] != "completed":
            continue
        exitcode_path = args.log_dir / f"{job['id']}.log.exitcode"
        if not exitcode_path.exists():
            continue
        try:
            exitcode = int(exitcode_path.read_text().strip())
        except ValueError:
            exitcode = -999
        if exitcode == 0:
            continue
        log_path = args.log_dir / f"{job['id']}.log"
        text = log_path.read_text(errors="replace") if log_path.exists() else ""
        epochs = [int(value) for value in re.findall(r"Epoch (\d+)/10", text)]
        suspects.append(
            {
                "job_id": job["id"],
                "exitcode": exitcode,
                "maximum_started_epoch": max(epochs) if epochs else 0,
                "queue_row_before": dict(job),
                "log_path": str(log_path),
                "exitcode_path": str(exitcode_path),
            }
        )

    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    recovery = args.recovery_root / f"{timestamp}_false_completed_bulk_repair"
    recovery.mkdir(parents=True, exist_ok=False)
    (recovery / "queue_state.before.json").write_text(json.dumps(state, indent=2))
    (recovery / "suspects.before.json").write_text(json.dumps(suspects, indent=2))
    if args.dry_run:
        print(json.dumps({"dry_run": True, "suspects": suspects}, indent=2))
        return

    now = dt.datetime.now(dt.timezone.utc).isoformat()
    quarantined = []
    with sqlite3.connect(args.db) as connection:
        connection.row_factory = sqlite3.Row
        catalog_rows = {}
        for suspect in suspects:
            row = connection.execute(
                "SELECT * FROM checkpoints WHERE model_id=?",
                (suspect["job_id"],),
            ).fetchone()
            if row is None:
                raise RuntimeError(f"Missing catalog row: {suspect['job_id']}")
            catalog_rows[suspect["job_id"]] = dict(row)
        (recovery / "checkpoint_catalog.before.json").write_text(
            json.dumps(catalog_rows, indent=2)
        )

        for index, suspect in enumerate(suspects, start=1):
            job_id = suspect["job_id"]
            row = catalog_rows[job_id]
            expected_digest = f"sha256:{row['sha256']}"
            quarantine_name = (
                f"QUARANTINED_exit{suspect['exitcode']}_{job_id}_"
                f"sha{row['sha256'][:16]}.pt"
            )
            print(f"[{index}/{len(suspects)}] quarantining {job_id}", flush=True)

            assets = release_assets(args.repo, args.tag)
            asset = next(
                (item for item in assets if item["id"] == row["asset_id"]),
                None,
            )
            local_path = (
                Path(row["local_path"]) if row["local_path"] else
                ROOT / "checkpoints" / row["filename"]
            )
            if asset is None:
                asset = next(
                    (item for item in assets if item["name"] == row["filename"]),
                    None,
                )
            if asset is None:
                if not local_path.is_file():
                    raise RuntimeError(f"No local or remote bytes for {job_id}")
                if (
                    local_path.stat().st_size != row["size_bytes"]
                    or sha256_file(local_path) != row["sha256"]
                ):
                    raise RuntimeError(f"Local catalog mismatch for {job_id}")
                subprocess.run(
                    [
                        "gh",
                        "release",
                        "upload",
                        args.tag,
                        str(local_path),
                        "--repo",
                        args.repo,
                    ],
                    cwd=ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                asset = next(
                    item
                    for item in release_assets(args.repo, args.tag)
                    if item["name"] == row["filename"]
                )
            if (
                asset["state"] != "uploaded"
                or asset["size"] != row["size_bytes"]
                or asset.get("digest") != expected_digest
            ):
                raise RuntimeError(f"Remote catalog mismatch for {job_id}")

            renamed = gh_json(
                [
                    "api",
                    "--method",
                    "PATCH",
                    f"repos/{args.repo}/releases/assets/{asset['id']}",
                    "-f",
                    f"name={quarantine_name}",
                ]
            )
            with tempfile.TemporaryDirectory(prefix="quarantine-verify-") as tmp:
                downloaded = Path(tmp) / quarantine_name
                download_asset(args.repo, int(renamed["id"]), downloaded)
                if (
                    downloaded.stat().st_size != row["size_bytes"]
                    or sha256_file(downloaded) != row["sha256"]
                ):
                    raise RuntimeError(f"Quarantine round trip failed for {job_id}")

            local_removed = False
            if local_path.is_file():
                if sha256_file(local_path) != row["sha256"]:
                    raise RuntimeError(f"Local bytes changed before eviction: {job_id}")
                local_path.unlink()
                local_removed = True

            reason = (
                f"Historical manager falsely marked completed despite exit code "
                f"{suspect['exitcode']}; maximum started epoch "
                f"{suspect['maximum_started_epoch']}/10. Partial bytes retained "
                f"as private quarantined asset {quarantine_name}."
            )
            connection.execute(
                """
                UPDATE checkpoints
                SET status='error', local_path=NULL, local_mtime_ns=NULL,
                    asset_id=?, asset_name=?, asset_size_bytes=?,
                    asset_digest=?, asset_state=?, error=?,
                    remote_verified_at=NULL, round_trip_verified_at=NULL,
                    updated_at=?
                WHERE model_id=?
                """,
                (
                    renamed["id"],
                    renamed["name"],
                    renamed["size"],
                    renamed.get("digest"),
                    renamed.get("state"),
                    reason,
                    now,
                    job_id,
                ),
            )
            connection.commit()
            suspect.update(
                {
                    "quarantined_asset_id": renamed["id"],
                    "quarantined_asset_name": renamed["name"],
                    "sha256": row["sha256"],
                    "size_bytes": row["size_bytes"],
                    "local_partial_removed": local_removed,
                    "reason": reason,
                }
            )
            quarantined.append(suspect)

    suspect_ids = {item["job_id"] for item in suspects}
    for job in state["jobs"]:
        if job["id"] not in suspect_ids:
            continue
        job.update(
            {
                "status": "pending",
                "gpu": None,
                "screen_name": None,
                "pid": None,
                "attempts": 0,
                "started": None,
                "completed": None,
                "error": None,
            }
        )
    temporary = args.state.with_suffix(args.state.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2))
    os.replace(temporary, args.state)
    (recovery / "quarantine_events.json").write_text(
        json.dumps(quarantined, indent=2)
    )
    (recovery / "queue_state.after.json").write_text(json.dumps(state, indent=2))
    print(
        json.dumps(
            {
                "suspects_requeued": len(suspects),
                "local_partials_removed_after_remote_round_trip": sum(
                    item["local_partial_removed"] for item in quarantined
                ),
                "recovery_directory": str(recovery),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
