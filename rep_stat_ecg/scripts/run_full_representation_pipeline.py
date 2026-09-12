"""End-to-End Orchestrator for PRD — QVCG-H / GRAIL-H Full-ECG Representation.

Pipeline Execution Ladder (PRD Section 48):
1. Monitor / Complete Stage M3H-CAL
2. Audit CAL-FAMILY graph topology and binary-graph gate (PRD Section 32, 33)
3. Run Stage M3H-F8: Fold-8 Hilbert Atlas Replication (PRD Section 28, 29)
4. Run Milestone M4A: Full-ECG Deterministic Representation Qualification R0-R4 (PRD Section 12-14)
5. Run Milestone M4B: Neural Reference Representation B0-B3-M (PRD Section 15-25, 38-40)
6. Verify and output Consolidated Verification Report
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def run_cmd(cmd: list[str], desc: str) -> None:
    print("\n" + "=" * 80)
    print(f"PIPELINE STEP: {desc}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 80)
    t0 = time.time()
    res = subprocess.run(cmd, check=True)
    t1 = time.time()
    print(f"Completed {desc} in {t1 - t0:.2f}s (Exit code: {res.returncode})")


def main():
    parser = argparse.ArgumentParser(description="End-to-End Representation Pipeline Orchestrator")
    parser.add_argument("--skip-cal", action="store_true", help="Skip waiting for calibration if already done")
    args = parser.parse_args()

    python_bin = sys.executable

    # Step 1: Wait for Stage M3H-CAL to complete
    cal_verdict_path = Path("refine-logs/qvcg/m3h_calibration/CAL_VERDICT.json")
    if not args.skip_cal and not cal_verdict_path.exists():
        print("Waiting for Stage M3H-CAL calibration to finish in tmux...")
        while not cal_verdict_path.exists():
            time.sleep(15)
            # Check if calibration log has finished
            cal_log = Path("refine-logs/qvcg/m3h_calibration/calibration.log")
            if cal_log.exists():
                text = cal_log.read_text()
                if "CALIBRATION VERDICT" in text or "CAL_VERDICT.json" in text:
                    break
        print("Stage M3H-CAL completed.")

    # Step 2: Audit CAL-FAMILY graph topology
    if Path("refine-logs/qvcg/m3h_calibration/CALBH_FAMILY_RESULTS.parquet").exists():
        run_cmd(
            [python_bin, "rep_stat_ecg/scripts/evaluate_cal_family_graph_topology.py", "--cal-dir", "refine-logs/qvcg/m3h_calibration"],
            "Audit CAL-FAMILY Graph Topology & Binary Overlay Gate",
        )

    # Step 3: Run Stage M3H-F8: Fold 8 Hilbert Atlas Replication
    run_cmd(
        [python_bin, "rep_stat_ecg/scripts/replicate_fold8_hilbert_atlas.py", "--data-dir", "data/ptb_xl", "--out-dir", "refine-logs/qvcg/fold8_atlas"],
        "Stage M3H-F8: Fold-8 Hilbert Atlas Replication",
    )

    # Step 4: Run Milestone M4A: Deterministic Representation Baselines & Kill Gate
    run_cmd(
        [python_bin, "rep_stat_ecg/scripts/evaluate_deterministic_representations.py", "--data-dir", "data/ptb_xl", "--out-dir", "refine-logs/qvcg/deterministic_representation"],
        "Milestone M4A: Deterministic Representation Baselines (R0-R4) & Kill Gate",
    )

    # Check Deterministic Kill Gate
    m4a_report_path = Path("refine-logs/qvcg/deterministic_representation/M4A_DETERMINISTIC_REPORT.json")
    with open(m4a_report_path) as f:
        m4a_data = json.load(f)

    if m4a_data.get("deterministic_kill_gate_verdict") != "PASS":
        print("\n" + "!" * 80)
        print("DETERMINISTIC KILL GATE TRIGGERED: R3 does not add value over R0.")
        print("Stopping prior to neural representation per PRD Section 14.")
        print("!" * 80)
        return

    # Step 5: Run Milestone M4B: Neural Reference Representation (B0-B3-M)
    run_cmd(
        [python_bin, "rep_stat_ecg/scripts/train_evaluate_neural_representations.py", "--data-dir", "data/ptb_xl", "--out-dir", "refine-logs/qvcg/neural_representation"],
        "Milestone M4B: Neural Reference Representation (B0-B3-M) & Gates A-D",
    )

    print("\n" + "=" * 80)
    print("END-TO-END REPRESENTATION PIPELINE COMPLETED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    main()
