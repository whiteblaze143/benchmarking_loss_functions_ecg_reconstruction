#!/usr/bin/env python3
"""Master pipeline runner for E01 and E02 under af_density_temporal_phenotypes_v7.
Executes run_e01_temporal_neighbourhoods.py and run_e02_historical_electrophysiology.py,
verifies all 22 required artifact deliverables, computes checksums, and mirrors artifacts.
"""

import os
import sys
import time
import shutil
import hashlib
from pathlib import Path

base_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(base_dir))

E01_REQUIRED = [
    "E01_duration_class_inventory.csv",
    "E01_patient_feasibility.csv",
    "E01_index_episode_manifest.parquet",
    "E01_neighbourhood_intervals.parquet",
    "E01_neighbourhood_event_assignments.parquet",
    "E01_risk_time_summary.csv",
    "E01_remote_comparison_manifest.parquet",
    "E01_neighbourhood_rate_ratios.csv",
    "E01_patient_level_effects.csv",
    "E01_conditional_null_results.parquet",
    "E01_density_architecture_results.csv",
    "E01_sensitivity_analysis.csv"
]

E02_REQUIRED = [
    "E02_historical_exposure_manifest.parquet",
    "E02_landmark_feasibility.csv",
    "E02_conditional_support.csv",
    "E02_fixed_elapsed_rr_features.parquet",
    "E02_ventricular_trajectory_models.csv",
    "E02_within_patient_effects.csv",
    "E02_between_patient_effects.csv",
    "E02_survival_risk_sets.csv",
    "E02_subsequent_episode_models.csv",
    "E02_nested_model_comparisons.csv"
]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    t_start = time.time()
    print("=" * 80)
    print("MASTER EXECUTION PIPELINE: EXPERIMENTS E01 & E02 (DIRECTIVE V7)")
    print("=" * 80)

    # 1. Run E01
    print("\n>>> LAUNCHING EXPERIMENT E01...")
    from scripts.run_e01_temporal_neighbourhoods import main as run_e01
    run_e01()

    # 2. Run E02
    print("\n>>> LAUNCHING EXPERIMENT E02...")
    from scripts.run_e02_historical_electrophysiology import main as run_e02
    run_e02()

    # 3. Verification of all 22 deliverables
    print("\n" + "=" * 80)
    print("AUDITING REQUIRED DELIVERABLES IN results/...")
    print("=" * 80)
    
    results_dir = base_dir / "results"
    mirror_dir = Path("/data/mithunmanivannan/codex_artifacts/af_density_temporal_phenotypes_v7/results")
    mirror_dir.mkdir(parents=True, exist_ok=True)
    
    missing = []
    manifest = []
    
    for f_name in E01_REQUIRED + E02_REQUIRED:
        src = results_dir / f_name
        if not src.exists():
            missing.append(f_name)
            print(f"  [MISSING] {f_name}")
        else:
            sz = src.stat().st_size
            sha = file_sha256(src)
            # Copy to mirror
            dst = mirror_dir / f_name
            shutil.copy2(src, dst)
            manifest.append({
                "filename": f_name,
                "experiment": "E01" if f_name.startswith("E01") else "E02",
                "size_bytes": sz,
                "sha256": sha
            })
            print(f"  [OK] {f_name:45s} ({sz:10,d} bytes) -> Mirrored")

    if missing:
        raise RuntimeError(f"Missing required deliverables: {missing}")

    # Write receipt
    import json
    receipt = {
        "execution_status": "SUCCESS",
        "total_deliverables": len(manifest),
        "e01_deliverables": len(E01_REQUIRED),
        "e02_deliverables": len(E02_REQUIRED),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_runtime_seconds": time.time() - t_start,
        "files": manifest
    }
    
    receipt_path = results_dir / "e01_e02_execution_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2))
    shutil.copy2(receipt_path, mirror_dir / "e01_e02_execution_receipt.json")
    print(f"\nExecution receipt written to {receipt_path}")
    print(f"Total pipeline runtime: {time.time() - t_start:.1f} seconds.")
    print("=" * 80)


if __name__ == "__main__":
    main()
