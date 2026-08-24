#!/usr/bin/env python3
"""Audit and requeue one falsely completed factorial job with corrupt bytes."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id")
    parser.add_argument("--reason", required=True)
    parser.add_argument("--quarantined-asset-name", required=True)
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
        "--recovery-root",
        type=Path,
        default=ROOT / "refine-logs/queue/recovery",
    )
    args = parser.parse_args()

    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    recovery = args.recovery_root / f"{timestamp}_{args.job_id}_corrupt_checkpoint"
    recovery.mkdir(parents=True, exist_ok=False)

    state = json.loads(args.state.read_text())
    matches = [job for job in state["jobs"] if job["id"] == args.job_id]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one state row for {args.job_id}, found {len(matches)}")
    job = matches[0]
    if job["status"] != "completed":
        raise RuntimeError(
            f"Refusing to requeue {args.job_id} from status {job['status']!r}"
        )
    (recovery / "queue_state.before.json").write_text(
        json.dumps(state, indent=2)
    )

    with sqlite3.connect(args.db) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT * FROM checkpoints WHERE model_id=?", (args.job_id,)
        ).fetchone()
        if row is None:
            raise RuntimeError(f"Checkpoint catalog row missing for {args.job_id}")
        (recovery / "checkpoint_catalog.before.json").write_text(
            json.dumps(dict(row), indent=2)
        )
        connection.execute(
            """
            UPDATE checkpoints
            SET status='error', asset_name=?, error=?, updated_at=?
            WHERE model_id=?
            """,
            (
                args.quarantined_asset_name,
                args.reason,
                dt.datetime.now(dt.timezone.utc).isoformat(),
                args.job_id,
            ),
        )
        connection.commit()

    before = dict(job)
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
    event = {
        "timestamp_utc": timestamp,
        "job_id": args.job_id,
        "reason": args.reason,
        "quarantined_asset_name": args.quarantined_asset_name,
        "queue_row_before": before,
        "queue_row_after": job,
        "recovery_directory": str(recovery),
    }
    (recovery / "event.json").write_text(json.dumps(event, indent=2))
    print(json.dumps(event, indent=2))


if __name__ == "__main__":
    main()
