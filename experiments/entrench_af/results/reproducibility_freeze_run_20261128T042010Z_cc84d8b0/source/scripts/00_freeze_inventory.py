#!/usr/bin/env python3
"""Stage 0: Freeze sample inventory, verify sample-disjointness, and hash record files."""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from common import (
    REPO_ROOT,
    canonical_cohort,
    compute_definitions_hash,
    create_run_dir,
    load_cohorts,
    load_definitions,
    resolve_run_dir,
    sha256_file,
    stamp_dataframe,
    utc_timestamp_iso,
    get_git_hash,
    write_json,
)


def _record_primary_file(cohort: str, record_id: str, cohorts_cfg: dict) -> Path:
    base = Path(cohorts_cfg["cohorts"][cohort]["base_dir"])
    if cohort in ("LTAFDB", "SHDB"):
        return base / f"{record_id}.hea"
    if cohort == "IRIDIA":
        h5_dir = base / "ecg_hdf5"
        matches = sorted(h5_dir.glob(f"{record_id}*.h5"))
        if matches:
            return matches[0]
        return Path(cohorts_cfg["cohorts"]["IRIDIA"]["metadata_csv"])
    raise ValueError(f"Unsupported cohort: {cohort}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 0: Freeze inventory")
    parser.add_argument("--run-dir", type=str, default=None, help="Existing run dir (optional)")
    args = parser.parse_args()

    defs = load_definitions()
    cohorts_cfg = load_cohorts()
    defs_hash = compute_definitions_hash()

    if args.run_dir:
        run_dir = resolve_run_dir(args.run_dir)
        git_hash = get_git_hash()
        run_ts = utc_timestamp_iso()
    else:
        run_dir, git_hash, run_ts = create_run_dir()

    stage_dir = run_dir / "stage0"
    stage_dir.mkdir(parents=True, exist_ok=True)

    v8_parquet = REPO_ROOT / cohorts_cfg["upstream_manifests"]["v8_inventory_parquet"]
    v5_runs_parquet = REPO_ROOT / cohorts_cfg["upstream_manifests"]["v5_runs_parquet"]

    inv = pd.read_parquet(v8_parquet).copy()
    inv["cohort"] = inv["dataset"].map(canonical_cohort)

    # 1. Sample-disjoint check: raw == known + unknown, discrepancy == 0
    discrepancy = inv["raw_samples"] - (inv["known_samples"] + inv["unknown_samples"])
    max_disc = int(discrepancy.abs().max())
    assert max_disc == 0, f"Sample-disjoint check failed: max discrepancy = {max_disc}"
    assert int(inv["sample_discrepancy"].abs().max()) == 0, "V8 sample_discrepancy != 0"

    # 2. Exact inventory counts
    n_records = len(inv)
    n_patients = int(inv["patient_id"].nunique())
    ann = inv[inv["is_annotated"]]
    n_ann_records = len(ann)
    n_ann_patients = int(ann["patient_id"].nunique())

    assert n_records == 379, f"Expected 379 records, got {n_records}"
    assert n_patients == 358, f"Expected 358 patients, got {n_patients}"
    assert n_ann_records == 349, f"Expected 349 annotated records, got {n_ann_records}"
    assert n_ann_patients == 329, f"Expected 329 annotated patients, got {n_ann_patients}"

    # 3. Unannotated SHDB IDs exact match
    expected_unann = set(cohorts_cfg["cohorts"]["SHDB"]["unannotated_ids"])
    observed_unann = set(inv.loc[~inv["is_annotated"], "record_id"].astype(str))
    assert observed_unann == expected_unann, (
        f"Unannotated SHDB IDs mismatch: extra={observed_unann - expected_unann}, "
        f"missing={expected_unann - observed_unann}"
    )

    # 4. Pre-harmonisation run count check
    v5_runs = pd.read_parquet(v5_runs_parquet)
    n_v5_runs = len(v5_runs)
    assert n_v5_runs == 8540, f"Expected 8,540 pre-harmonisation runs, got {n_v5_runs}"

    # 5. Compute SHA256 of every record's primary file (metadata/header)
    file_paths = []
    file_hashes = []
    for row in inv.itertuples(index=False):
        fpath = _record_primary_file(row.cohort, str(row.record_id), cohorts_cfg)
        assert fpath.exists(), f"Missing record file: {fpath}"
        file_paths.append(str(fpath))
        file_hashes.append(sha256_file(fpath))

    inv["primary_file_path"] = file_paths
    inv["primary_file_sha256"] = file_hashes
    inv = stamp_dataframe(inv, git_hash=git_hash, definitions_hash=defs_hash, run_timestamp=run_ts)

    out_csv = stage_dir / "inventory_frozen.csv"
    out_parquet = stage_dir / "inventory_frozen.parquet"
    inv.to_csv(out_csv, index=False)
    inv.to_parquet(out_parquet, index=False)

    (stage_dir / "definitions_hash.txt").write_text(defs_hash + "\n", encoding="utf-8")

    summary = {
        "run_dir": str(run_dir.resolve()),
        "git_hash": git_hash,
        "definitions_hash": defs_hash,
        "run_timestamp": run_ts,
        "seed": defs["seed"],
        "n_records": n_records,
        "n_patients": n_patients,
        "n_annotated_records": n_ann_records,
        "n_annotated_patients": n_ann_patients,
        "n_unannotated_shdb": len(observed_unann),
        "n_runs_pre_harmonisation": n_v5_runs,
        "max_sample_discrepancy": max_disc,
        "total_af_hours": float(inv["af_h"].sum()),
        "total_known_non_af_hours": float(inv["known_non_af_h"].sum()),
        "total_unknown_hours": float(inv["unknown_h"].sum()),
    }
    write_json(stage_dir / "stage0_summary.json", summary)

    print(f"[Stage 0] OK | run_dir={run_dir}")
    print(f"  records={n_records} (patients={n_patients}), annotated={n_ann_records} (patients={n_ann_patients})")
    print(f"  unannotated_shdb={len(observed_unann)}, pre_harmonisation_runs={n_v5_runs}, max_disc={max_disc}")
    print(f"  definitions_hash={defs_hash}")


if __name__ == "__main__":
    main()
