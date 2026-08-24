#!/usr/bin/env python3
"""Quarantine release-incompatible checkpoints and requeue their jobs."""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def gh(*arguments: str) -> dict:
    result = subprocess.run(
        ["gh", "api", *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("results/checkpoint_store/compatibility_audit.json"),
    )
    parser.add_argument(
        "--db", type=Path, default=Path("results/checkpoint_store/catalog.sqlite")
    )
    parser.add_argument(
        "--state", type=Path, default=Path("refine-logs/queue/queue_state.json")
    )
    parser.add_argument("--recovery-dir", type=Path, required=True)
    parser.add_argument(
        "--repo",
        default="whiteblaze143/benchmarking_loss_functions_ecg_reconstruction",
    )
    args = parser.parse_args()

    audit = json.loads(args.audit.read_text())
    incompatible = {
        row["model_id"]: row
        for row in audit["models"]
        if not row["compatible"]
    }
    if not incompatible:
        print('{"requeued": 0}')
        return

    recovery = args.recovery_dir.resolve()
    if recovery.is_dir():
        state = json.loads(args.state.read_text())
        candidates = [
            event
            for event in state.get("meta", {}).get("manual_repairs", [])
            if event.get("reason") == "source/precision compatibility audit"
            and event.get("count") == len(incompatible)
        ]
        if not candidates:
            raise RuntimeError("Existing recovery directory has no matching event")
        event = candidates[-1]
        (recovery / "event.json").write_text(
            json.dumps(event, indent=2) + "\n"
        )
        from scripts.checkpoint_store import export_catalog

        connection = sqlite3.connect(args.db)
        connection.row_factory = sqlite3.Row
        export_catalog(connection, args.db.with_name("catalog.jsonl"))
        connection.close()
        print(
            json.dumps(
                {
                    "requeued": len(incompatible),
                    "recovery": str(recovery),
                    "resumed": True,
                }
            )
        )
        return
    recovery.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.audit, recovery / "compatibility_audit.json")
    shutil.copy2(args.db, recovery / "catalog.before.sqlite")
    shutil.copy2(args.state, recovery / "queue_state.before.json")

    connection = sqlite3.connect(args.db)
    connection.row_factory = sqlite3.Row
    catalog = {
        row["model_id"]: row
        for row in connection.execute(
            "SELECT * FROM checkpoints WHERE model_id IN (%s)"
            % ",".join("?" for _ in incompatible),
            tuple(incompatible),
        )
    }
    missing = sorted(set(incompatible) - set(catalog))
    if missing:
        raise RuntimeError(f"Incompatible ids absent from catalog: {missing}")

    remote_events = []
    for model_id in sorted(incompatible):
        row = catalog[model_id]
        if row["asset_id"] is None:
            raise RuntimeError(f"{model_id} has no remote asset id")
        quarantine_name = (
            f"QUARANTINED_release_policy_{model_id}_"
            f"sha{row['sha256'][:12]}.pt"
        )
        asset = gh(f"repos/{args.repo}/releases/assets/{row['asset_id']}")
        if asset["name"] != quarantine_name:
            if asset["name"] != row["asset_name"]:
                raise RuntimeError(
                    f"{model_id} remote asset has unexpected name {asset['name']}"
                )
            asset = gh(
                "--method",
                "PATCH",
                f"repos/{args.repo}/releases/assets/{row['asset_id']}",
                "-f",
                f"name={quarantine_name}",
            )
        if asset.get("digest") != f"sha256:{row['sha256']}":
            raise RuntimeError(f"{model_id} digest changed during quarantine")
        remote_events.append(
            {
                "model_id": model_id,
                "asset_id": row["asset_id"],
                "old_name": row["asset_name"],
                "quarantine_name": quarantine_name,
                "sha256": row["sha256"],
                "reasons": incompatible[model_id]["reasons"],
            }
        )

    state = json.loads(args.state.read_text())
    jobs = {job["id"]: job for job in state["jobs"]}
    sidecar_dir = recovery / "sidecars"
    sidecar_dir.mkdir()
    now = utc_now()
    for event in remote_events:
        model_id = event["model_id"]
        job = jobs[model_id]
        if job["status"] != "completed":
            raise RuntimeError(f"{model_id} queue status is {job['status']}")
        mask = incompatible[model_id]["factorial_mask"]
        seed = incompatible[model_id]["seed"]
        sidecar = ROOT / "checkpoints" / f"factorial_{mask}_s{seed}.metadata.json"
        if sidecar.is_file():
            shutil.move(str(sidecar), sidecar_dir / sidecar.name)
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
        connection.execute(
            """
            UPDATE checkpoints
            SET status='error', asset_name=?, error=?, updated_at=?
            WHERE model_id=?
            """,
            (
                event["quarantine_name"],
                "release-incompatible generation quarantined: "
                + ",".join(event["reasons"]),
                now,
                model_id,
            ),
        )

    event = {
        "schema_version": 1,
        "timestamp": now,
        "reason": "source/precision compatibility audit",
        "count": len(remote_events),
        "models": remote_events,
    }
    state.setdefault("meta", {}).setdefault("manual_repairs", []).append(event)
    state_tmp = args.state.with_suffix(args.state.suffix + ".policy.tmp")
    state_tmp.write_text(json.dumps(state, indent=2) + "\n")
    state_tmp.replace(args.state)
    connection.commit()
    from scripts.checkpoint_store import export_catalog

    export_catalog(connection, args.db.with_name("catalog.jsonl"))
    connection.close()
    (recovery / "event.json").write_text(json.dumps(event, indent=2) + "\n")
    print(json.dumps({"requeued": len(remote_events), "recovery": str(recovery)}))


if __name__ == "__main__":
    main()
