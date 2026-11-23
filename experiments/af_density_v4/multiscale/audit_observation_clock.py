"""Observation Clock & Continuity Auditor for T-CAIREM V4.1.

Audits physical sample offsets, prefix exclusions, file boundaries, and
nominal vs. actual recording durations across development recordings.
Strictly excludes protected held-out test records.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction")
WAVEFORM_DIR = Path("/data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1")
REPORTS_DIR = PROJECT_ROOT / "experiments" / "af_density_v4" / "reports"
CODEX_DIR = Path("/data/mithunmanivannan/codex_artifacts/af_density_v4")

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
CODEX_DIR.mkdir(parents=True, exist_ok=True)


def audit_clocks() -> pd.DataFrame:
    print("=" * 80)
    print("T-CAIREM V4.1: Auditing Physical Observation Clocks and Continuity")
    print("=" * 80)

    manifest_csv = WAVEFORM_DIR / "file_manifest.csv"
    if not manifest_csv.exists():
        raise FileNotFoundError(f"File manifest not found: {manifest_csv}")

    manifest = list(csv.DictReader(manifest_csv.open()))
    records: List[Dict[str, Any]] = []

    # Group by patient
    patient_files: Dict[str, List[Dict[str, Any]]] = {}
    for r in manifest:
        # Enforce quarantine on test split
        if r["partition"] == "test":
            continue
        pid = r["patient_id"]
        if pid not in patient_files:
            patient_files[pid] = []
        patient_files[pid].append(r)

    print(f"Auditing {len(patient_files)} authorized development patients...")

    for pid, files in sorted(patient_files.items()):
        # Sort files by file_index
        files.sort(key=lambda x: int(x["file_index"]))
        n_files = len(files)
        total_windows = sum(int(f["windows"]) for f in files)
        actual_seconds = total_windows * 30.0
        actual_hours = actual_seconds / 3600.0

        # Check offsets and continuous timeline alignment
        file_boundaries = []
        gaps_detected = False
        gap_total_s = 0.0

        for idx, f in enumerate(files):
            fi = int(f["file_index"])
            win = int(f["windows"])
            offset = float(f.get("record_file_offset_samples", 0.0))
            prefix = float(f.get("excluded_prefix_samples", 0.0))
            dur_s = win * 30.0
            file_boundaries.append((fi, offset / 200.0, (offset / 200.0) + dur_s))

        # Check for gaps between consecutive files of the same patient
        for i in range(len(file_boundaries) - 1):
            curr_end = file_boundaries[i][2]
            next_start = file_boundaries[i+1][1]
            if next_start > curr_end + 1.0:
                gaps_detected = True
                gap_total_s += (next_start - curr_end)

        # Check nominal 24h multiples
        nominal_days = int(np.round(actual_hours / 24.0))
        nominal_hours = float(nominal_days * 24.0)
        hour_discrepancy = actual_hours - nominal_hours

        records.append({
            "patient_id": pid,
            "partition": files[0]["partition"],
            "num_files": n_files,
            "total_30s_windows": total_windows,
            "actual_duration_seconds": actual_seconds,
            "actual_duration_hours": round(actual_hours, 4),
            "nominal_target_hours": nominal_hours,
            "duration_discrepancy_hours": round(hour_discrepancy, 4),
            "has_interfile_gaps": gaps_detected,
            "total_gap_seconds": round(gap_total_s, 2),
            "eligible_for_24h_exact": bool(actual_hours >= 24.0),
            "eligible_for_48h_exact": bool(actual_hours >= 48.0),
            "eligible_for_72h_exact": bool(actual_hours >= 72.0),
        })

    df = pd.DataFrame(records)
    out_csv = REPORTS_DIR / "v41_clock_and_continuity_audit.csv"
    df.to_csv(out_csv, index=False)
    df.to_csv(CODEX_DIR / "v41_clock_and_continuity_audit.csv", index=False)
    print(f"Audit complete for {len(df)} patients. Output saved to {out_csv}")
    print(f"Summary: {df['eligible_for_24h_exact'].sum()} patients >= 24h, {df['eligible_for_48h_exact'].sum()} >= 48h, {df['eligible_for_72h_exact'].sum()} >= 72h.")
    return df


if __name__ == "__main__":
    audit_clocks()
