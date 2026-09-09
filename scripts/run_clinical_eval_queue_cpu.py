#!/usr/bin/env python3
"""Autonomous CPU Clinical Biomarker & Foundation Model Evaluation Queue.

Evaluates all un-evaluated models across:
1. 6 Un-evaluated Models from the 55-Model Roster
2. 15 Round-1 Kill-Gate Ablation Suite Models
3. Round-2 Lean Ablation Suite Models (completed + dynamic watcher for new runs)

Features:
- Strictly executes on CPU (CUDA_VISIBLE_DEVICES="", device="cpu") to ensure 0% contention with GPU training
- Ephemeral on-demand materialization & immediate eviction for catalog checkpoints
- Automatic cleanup of large resume.pt files to protect disk buffer (>5 GB)
- Live database insertion into clinical_metrics.db (evaluation_version='1lead_clinical_v1')
- Rebuilds FINAL_CLINICAL_BENCHMARK_METRICS_MASTER.csv after each model completes
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import subprocess
import sys
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.evaluate_1lead_clinical_classifier_suite import (
    EVALUATION_VERSION,
    evaluate_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [CPU-EVAL] %(message)s",
    datefmt="%H:%M:%S"
)

DB_PATH = _ROOT / "results/clinical_biomarkers_multids/clinical_metrics.db"
REINTEGRATE_SCRIPT = _ROOT / "scripts/reintegrate_master_results.py"
BUILD_MASTER_SCRIPT = _ROOT / "scripts/build_final_master_clinical_metrics.py"


def get_already_evaluated(db_path: Path) -> Set[str]:
    if not db_path.exists():
        return set()
    try:
        with sqlite3.connect(db_path, timeout=60) as con:
            cur = con.cursor()
            cur.execute("""
                SELECT model_id FROM clinical_metrics WHERE evaluation_version = ? AND target = 'ECGFounder_Macro_150'
                INTERSECT
                SELECT model_id FROM clinical_metrics WHERE evaluation_version = ? AND target = 'QRS_Duration_SemiSeg'
                INTERSECT
                SELECT model_id FROM clinical_metrics WHERE evaluation_version = ? AND target = 'EchoNextSHD_Macro_12'
            """, (EVALUATION_VERSION, EVALUATION_VERSION, EVALUATION_VERSION))
            return {row[0] for row in cur.fetchall()}
    except Exception as e:
        logging.error("Error querying DB: %s", e)
        return set()


def get_static_candidates() -> List[Dict[str, Any]]:
    candidates = []

    # 1. 6 Un-evaluated models from 55-roster
    roster_6 = [
        ("conv15e_A0_raw_s42_l0", _ROOT / "refine-logs/convergence_10e/runs/conv15e_A0_raw_s42_l0/best.pt", False, "15e-Convergence Baseline"),
        ("conv15e_conv_control_s42_l0", _ROOT / "refine-logs/convergence_10e/runs/conv15e_conv_control_s42_l0/best.pt", False, "15e-Convergence Control"),
        ("conv15e_tf_sc16_cy4_s42_l0", _ROOT / "refine-logs/convergence_10e/runs/conv15e_tf_sc16_cy4_s42_l0/best.pt", False, "15e-TimeSformer sc16 cy4"),
        ("conv15e_tf_sc16_cy8_s42_l0", _ROOT / "refine-logs/convergence_10e/runs/conv15e_tf_sc16_cy8_s42_l0/best.pt", False, "15e-TimeSformer sc16 cy8"),
        ("D0_current_id_currentloss_s42_l0", _ROOT / "refine-logs/convergence_10e/runs/D0_current_id_currentloss_s42_l0/best.pt", False, "Stage A 3D-Theta D0"),
        ("spatial_1lead_a0_1000000_s42_l0", _ROOT / "checkpoints/onelead_cache/spatial_1lead_a0_1000000_s42_l0.pt", True, "3-Epoch Spatial A0 Single-Lead"),
    ]
    for mid, path, ephem, desc in roster_6:
        candidates.append({"model_id": mid, "path": path, "is_ephemeral": ephem, "group": "55-Roster Backfill", "description": desc})

    # 2. 15 Kill-Gate Suite Models
    killgate_mids = [
        "conv15e_K1_nold_s42_l0", "conv15e_K2_noart_s42_l0", "conv15e_K3_nodel_s42_l0",
        "conv15e_K4_l1_s42_l0", "conv15e_K5_adaptive_s42_l0", "conv15e_B1_hardbasis_s42_l0",
        "conv15e_B2_hardbasis_nodel_s42_l0", "conv15e_B3_hardbasis_adaptive_s42_l0",
        "conv15e_B4_hardbasis_l1_s42_l0", "conv15e_C1_enc4_s42_l0", "conv15e_C2_dec2_s42_l0",
        "conv15e_C3_width512_s42_l0", "conv15e_C4_minimal_s42_l0", "conv15e_Z1_zscore_s42_l0",
        "conv15e_Z2_hardbasis_zscore_s42_l0"
    ]
    for mid in killgate_mids:
        p = _ROOT / f"refine-logs/killgate/runs/{mid}/best.pt"
        candidates.append({"model_id": mid, "path": p, "is_ephemeral": False, "group": "Kill-Gate Ablations", "description": mid})

    # 3. Completed Lean Suite Models
    lean_dir = _ROOT / "refine-logs/lean_abl2/runs"
    if lean_dir.exists():
        for rdir in sorted(lean_dir.glob("lean2_*_s42_l0")):
            bp = rdir / "best.pt"
            if bp.exists():
                candidates.append({"model_id": rdir.name, "path": bp, "is_ephemeral": False, "group": "Lean Ablations", "description": rdir.name})

    return candidates


def clean_disk_buffer():
    # Purge any stale resume.pt files that are >500 MB if best.pt exists
    for resume_file in _ROOT.glob("refine-logs/**/resume.pt"):
        parent = resume_file.parent
        if (parent / "best.pt").exists() and (parent / "summary.json").exists():
            try:
                sz_mb = resume_file.stat().st_size / (1024 * 1024)
                if sz_mb > 500:
                    logging.info("Purging completed optimizer state: %s (%.1f MB)", resume_file, sz_mb)
                    resume_file.unlink(missing_ok=True)
            except Exception:
                pass


def sync_master_csv():
    try:
        logging.info("Synchronizing Master Results CSVs...")
        subprocess.run([sys.executable, str(REINTEGRATE_SCRIPT)], check=True, capture_output=True, text=True)
        subprocess.run([sys.executable, str(BUILD_MASTER_SCRIPT)], check=True, capture_output=True, text=True)
        logging.info("Master Results CSV synchronized successfully!")
    except Exception as e:
        logging.error("Failed to synchronize Master Results CSV: %s", e)


def main():
    parser = argparse.ArgumentParser(description="Autonomous CPU Clinical Evaluation Queue")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--smoke", action="store_true", help="Run in smoke test mode")
    parser.add_argument("--watch", action="store_true", default=True, help="Continuously poll for new completed training runs")
    parser.add_argument("--poll-interval", type=int, default=60, help="Seconds to sleep between polling sweeps")
    args = parser.parse_args()

    # Enforce CPU execution
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "2"
    device = torch.device("cpu")

    logging.info("=" * 80)
    logging.info("AUTONOMOUS CPU CLINICAL EVALUATION QUEUE")
    logging.info("Device: CPU (Forced) | OMP_NUM_THREADS=2 | Batch Size: %d", args.batch_size)
    logging.info("=" * 80)

    clean_disk_buffer()

    while True:
        already_done = get_already_evaluated(DB_PATH)
        candidates = get_static_candidates()

        todo = [c for c in candidates if c["model_id"] not in already_done and (c["path"].exists() or c["is_ephemeral"])]
        logging.info("Status: %d already evaluated in DB | %d candidates ready for evaluation", len(already_done), len(todo))

        if not todo:
            if not args.watch:
                logging.info("No remaining models to evaluate and watch mode disabled. Exiting.")
                break
            logging.info("All currently available models evaluated. Waiting %d seconds for new checkpoints from training queue...", args.poll_interval)
            time.sleep(args.poll_interval)
            continue

        for idx, item in enumerate(todo, 1):
            mid = item["model_id"]
            ckpt_path = item["path"]
            is_ephem = item["is_ephemeral"]
            grp = item["group"]

            logging.info("-" * 80)
            logging.info("[%d/%d] [%s] Starting Clinical Evaluation: %s", idx, len(todo), grp, mid)
            logging.info("-" * 80)

            if is_ephem:
                logging.info("Materializing ephemeral checkpoint for %s...", mid)
                res = subprocess.run(
                    [sys.executable, str(_ROOT / "scripts/onelead_checkpoint_store.py"), "materialize", mid],
                    capture_output=True,
                    text=True
                )
                if res.returncode != 0:
                    logging.error("Failed to materialize %s: %s", mid, res.stderr)
                    continue
                ckpt_path = Path(res.stdout.strip())

            if not ckpt_path.exists():
                logging.warning("Checkpoint path does not exist: %s. Skipping.", ckpt_path)
                continue

            t0 = time.time()
            try:
                eval_cmd = [
                    sys.executable,
                    str(_ROOT / "scripts/evaluate_1lead_clinical_classifier_suite.py"),
                    "--model-id", mid,
                    "--checkpoint-path", str(ckpt_path),
                    "--device", "cpu",
                    "--batch-size", str(args.batch_size),
                    "--db-path", str(DB_PATH)
                ]
                if args.smoke:
                    eval_cmd.append("--smoke")

                env = dict(os.environ)
                env["CUDA_VISIBLE_DEVICES"] = ""
                env["OMP_NUM_THREADS"] = "2"

                res_eval = subprocess.run(eval_cmd, env=env)
                if res_eval.returncode != 0:
                    logging.error("Evaluation subprocess failed for %s with code %d", mid, res_eval.returncode)
                else:
                    elapsed = time.time() - t0
                    logging.info("Successfully evaluated %s in %.1f seconds!", mid, elapsed)
                    clean_disk_buffer()
                    sync_master_csv()
            except Exception as e:
                logging.exception("Error evaluating %s: %s", mid, e)
            finally:
                if is_ephem and ckpt_path.exists():
                    logging.info("Evicting ephemeral checkpoint %s...", ckpt_path.name)
                    ckpt_path.unlink(missing_ok=True)

        if not args.watch:
            break


if __name__ == "__main__":
    main()
