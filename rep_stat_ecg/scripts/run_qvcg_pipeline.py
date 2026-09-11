"""Master Autonomous Pipeline Orchestrator for Q-VCG.

Sequentially executes:
- M0: Contract verification via pytest.
- M1: Patient-balanced microstate census on Folds 1-7.
- M2: 3D lattice discretization & CAHC contiguous domain discovery.
- M3: Calibrated MMD & patient block permutation quotient closure.
- M4: Transparent feature extraction (R0, R1, R2, R3) across Folds 1-9.
- M5 & M6: Linear probing, sample efficiency, retrieval, and kill-gate audit.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def run_stage(cmd: list[str], stage_name: str) -> None:
    print(f"\n{'='*70}")
    print(f"   STARTING STAGE: {stage_name}")
    print(f"   COMMAND: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    start_time = time.time()

    env = os.environ.copy()
    env["PYTHONPATH"] = f".:{env.get('PYTHONPATH', '')}"

    proc = subprocess.run(cmd, env=env)
    elapsed = time.time() - start_time

    if proc.returncode != 0:
        print(f"\n[FATAL] Stage {stage_name} FAILED with exit code {proc.returncode} after {elapsed:.1f}s")
        sys.exit(proc.returncode)

    print(f"\n[SUCCESS] Stage {stage_name} COMPLETED in {elapsed:.1f}s\n")


def main():
    parser = argparse.ArgumentParser(description="Run Q-VCG autonomous pipeline.")
    parser.add_argument("--data-dir", type=str, default="data/ptb_xl")
    parser.add_argument("--out-dir", type=str, default="refine-logs/qvcg")
    parser.add_argument("--max-patients", type=int, default=2500)
    parser.add_argument("--n-domains", type=int, default=64)
    parser.add_argument("--skip-unit-tests", action="store_true")
    args = parser.parse_args()

    python_bin = sys.executable

    # 1. Milestone M0: Unit Verification
    if not args.skip_unit_tests:
        run_stage(
            [
                python_bin, "-m", "pytest", "-v",
                "tests/test_vcg_geometry.py",
                "tests/test_vcg_dynamics.py",
                "tests/test_motif_discovery.py",
                "tests/test_qvcg_representation.py",
            ],
            "M0: Unit Verification (21/21 contracts)",
        )

    # 2. Milestone M1: Microstates Census
    run_stage(
        [
            python_bin, "rep_stat_ecg/scripts/build_vcg_microstates.py",
            "--data-dir", args.data_dir,
            "--out-dir", args.out_dir,
            "--max-patients", str(args.max_patients),
        ],
        "M1: Patient-Balanced VCG Microstate Extraction",
    )

    # 3. Milestone M2: Spatial Domain Discovery
    run_stage(
        [
            python_bin, "rep_stat_ecg/scripts/discover_vcg_domains.py",
            "--micro-table", f"{args.out_dir}/VCG_MICROSTATE_TABLE.parquet",
            "--std-json", f"{args.out_dir}/VCG_COORDINATE_STANDARDIZER.json",
            "--out-dir", args.out_dir,
            "--n-domains", str(args.n_domains),
        ],
        "M2: 3D Lattice CAHC Contiguous Spatial Domains",
    )

    # 4. Milestone M3: MMD & Quotient Closure
    run_stage(
        [
            python_bin, "rep_stat_ecg/scripts/discover_repeated_motifs.py",
            "--micro-table", f"{args.out_dir}/VCG_MICROSTATE_TABLE.parquet",
            "--domain-stats", f"{args.out_dir}/VCG_DOMAIN_STATS.parquet",
            "--domain-registry", f"{args.out_dir}/VCG_SPATIAL_DOMAIN_REGISTRY.parquet",
            "--out-dir", args.out_dir,
            "--n-perm", "999",
        ],
        "M3: repSpat IMQ Attribute-Block Permutation + BH Similarity Graph",
    )

    # 5. Milestone M4: Feature Vector Serialization
    run_stage(
        [
            python_bin, "rep_stat_ecg/scripts/build_qvcg_features.py",
            "--data-dir", args.data_dir,
            "--qvcg-dir", args.out_dir,
            "--max-records-train", "4000",
        ],
        "M4: Extract Representations (R0, R1, R2, R3) on Folds 1-9",
    )

    # 6. Milestone M5 & M6: Evaluation & Kill-Gate Audit
    run_stage(
        [
            python_bin, "rep_stat_ecg/scripts/evaluate_qvcg.py",
            "--qvcg-dir", args.out_dir,
            "--data-dir", args.data_dir,
            "--out-dir", args.out_dir,
        ],
        "M5 & M6: Linear Probes, Sample Efficiency, Retrieval, Kill Gates",
    )

    print("\n=======================================================")
    print("   Q-VCG PIPELINE COMPLETE — INSPECT QVCG_TRANSPARENT_GATE.json")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
