#!/usr/bin/env python3
"""Print a compact JSON status for the one-lead blinded RDB workflow."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "results/onelead_rdb_semiseg_blinded/compact.sqlite"


def main() -> None:
    output: dict[str, object] = {"database": str(DB), "exists": DB.is_file(), "frozen_models": 60}
    if not DB.is_file():
        print(json.dumps(output, indent=2)); return
    connection = sqlite3.connect(DB); connection.row_factory = sqlite3.Row
    try:
        output["evaluations"] = [dict(row) for row in connection.execute(
            "SELECT stage,status,count(*) AS models,round(avg(duration_seconds),1) AS mean_seconds FROM evaluations GROUP BY stage,status ORDER BY stage,status"
        )]
        output["ceilings"] = [dict(row) for row in connection.execute(
            "SELECT model_id,status,records,round(duration_seconds,1) AS seconds,round(primary_mean_micro_f1_20ms,4) AS primary_f1 FROM evaluations WHERE model_id LIKE '__original%' ORDER BY model_id"
        )]
        output["thresholds"] = [dict(row) for row in connection.execute(
            "SELECT input_lead,status,round(boundary_cutoff,4) AS boundary_cutoff,round(signal_cutoff,4) AS signal_cutoff,competitive_anchors,calibration_anchors FROM thresholds ORDER BY input_lead"
        )]
        output["screening"] = [dict(row) for row in connection.execute(
            "SELECT decision,count(*) AS models FROM screening_decisions GROUP BY decision ORDER BY decision"
        )]
        terminal = connection.execute(
            "SELECT count(DISTINCT model_id) FROM screening_decisions"
        ).fetchone()[0]
        output["screening_terminal_models"] = terminal
        output["screening_remaining"] = 60 - terminal
    finally:
        connection.close()
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
