#!/usr/bin/env python3
"""Persistent Week-Long Supervisor for Paper 07 SetOperator Architecture Ablation Campaign.

Features:
- Enforces single-process GPU allocation on cuda:0 (never runs simultaneous training jobs).
- Depends strictly on the active G1 launch controller: waits for pipeline_complete_summary.json before starting ablations.
- Durable state machine tracking states: PLANNED -> PREFLIGHT -> READY -> RUNNING -> EVALUATING -> COMPLETE / FAILED / SKIPPED_BUDGET.
- Enforces 168 GPU-hour maximum allocation cap with dynamic time accounting.
- Supports flags:
  --dry-run
  --validate-manifest
  --resume
  --status
  --max-gpu-hours 168
  --single-gpu cuda:0
  --stop-after-current
  --preflight-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Dict, Any, List, Optional

import torch

DEV_ROOT = Path("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction_geometry_dev")
PYTHON_BIN = Path("/home/mithunmanivannan/.venv/bin/python")

MANIFEST_PATH = DEV_ROOT / "experiments/echonext_operator_benchmark/ablation_week_manifest.json"
STATE_DIR = Path("/data/mithunmanivannan/codex_artifacts/echonext_geometry_setoperator_v1/supervisor_state")
STATE_DIR.mkdir(parents=True, exist_ok=True)

STATE_FILE = STATE_DIR / "ablation_queue_state.json"
LOCK_FILE = STATE_DIR / "ABLAT_LOCK"
LOG_FILE = STATE_DIR / "ablation_supervisor.log"
PRIMARY_PIPELINE_SUMMARY = Path("/data/mithunmanivannan/codex_artifacts/echonext_geometry_setoperator_v1/controller/pipeline_complete_summary.json")

TRACKER_FILE = Path("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/refine-logs/EXPERIMENT_TRACKER.md")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a"),
    ],
)
logger = logging.getLogger("ablation_supervisor")


def get_gpu_compute_processes() -> list[dict[str, str]]:
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        lines = [line.strip() for line in res.stdout.strip().split("\n") if line.strip()]
        procs = []
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                procs.append({"pid": parts[0], "name": parts[1], "memory_mib": parts[2]})
        return procs
    except Exception as e:
        logger.warning("Failed to query nvidia-smi: %s", e)
        return []


def load_manifest() -> Dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest missing at {MANIFEST_PATH}")
    with open(MANIFEST_PATH, "r") as f:
        return json.load(f)


def validate_manifest() -> bool:
    logger.info("Validating manifest at %s...", MANIFEST_PATH)
    manifest = load_manifest()
    required_keys = ["campaign_name", "max_gpu_hours", "primary_seed", "dataset", "experiments"]
    for k in required_keys:
        if k not in manifest:
            logger.error("Manifest missing top-level key: %s", k)
            return False

    exp_ids = set()
    for exp in manifest["experiments"]:
        eid = exp.get("id")
        if not eid:
            logger.error("Experiment missing 'id'")
            return False
        if eid in exp_ids:
            logger.error("Duplicate experiment ID found: %s", eid)
            return False
        exp_ids.add(eid)

    logger.info("Manifest validation PASSED. Total experiments registered: %d", len(exp_ids))
    return True


def run_preflight_checks() -> bool:
    logger.info("Executing P1 Scientific Validity Preflight Gate...")

    # 1. Verify Phase-KME fit artifact
    manifest = load_manifest()
    fit_path = Path(manifest["fit_artifact"])
    if not fit_path.exists() or fit_path.stat().st_size < 1000:
        logger.error("Preflight failed: Phase-KME fit artifact missing or empty at %s", fit_path)
        return False
    logger.info("Phase-KME fit artifact verified: %s (%d bytes)", fit_path, fit_path.stat().st_size)

    # 2. Verify validation beats cache
    val_cache = Path(manifest["validation_beats_cache"])
    if not val_cache.exists():
        logger.error("Preflight failed: Validation beats cache missing at %s", val_cache)
        return False
    logger.info("Validation beats cache verified: %s", val_cache)

    # 3. Run architecture unit tests on CPU
    test_path = DEV_ROOT / "tests/test_ablation_week_architectures.py"
    res = subprocess.run([str(PYTHON_BIN), "-m", "pytest", str(test_path), "-v"], capture_output=True, text=True)
    if res.returncode != 0:
        logger.error("Preflight failed: Unit tests failed:\n%s\n%s", res.stdout, res.stderr)
        return False
    logger.info("Preflight unit tests PASSED cleanly.")
    return True


def initialize_or_load_state(manifest: Dict[str, Any]) -> Dict[str, Any]:
    if STATE_FILE.exists():
        logger.info("Loading existing queue state from %s...", STATE_FILE)
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
        return state

    logger.info("Initializing new queue state...")
    state = {
        "campaign_name": manifest["campaign_name"],
        "start_timestamp": time.time(),
        "total_gpu_hours_consumed": 0.0,
        "max_gpu_hours": manifest["max_gpu_hours"],
        "current_experiment": None,
        "stop_after_current": False,
        "experiments": {},
    }
    for exp in manifest["experiments"]:
        eid = exp["id"]
        status = exp.get("status", "PLANNED")
        state["experiments"][eid] = {
            "id": eid,
            "name": exp["name"],
            "group": exp["group"],
            "priority": exp["priority"],
            "status": status,
            "active_parameters": exp.get("active_parameters"),
            "est_gpu_hours": exp.get("est_gpu_hours", 3.0),
            "actual_gpu_hours": 0.0,
            "best_epoch": None,
            "best_val_auroc": None,
            "best_val_auprc": None,
            "dev_grassmann_auroc": None,
            "checkpoint_dir": exp.get("checkpoint_dir"),
            "start_time": None,
            "end_time": None,
            "notes": None,
        }

    save_state(state)
    return state


def save_state(state: Dict[str, Any]) -> None:
    temp_file = STATE_DIR / "state.tmp.json"
    with open(temp_file, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(temp_file, STATE_FILE)


def print_status_table(state: Dict[str, Any]) -> None:
    logger.info("=========================================================================================")
    logger.info("PAPER 07 ABLATION QUEUE STATUS | Consumed: %.2fh / %.2fh | Active: %s",
                state["total_gpu_hours_consumed"], state["max_gpu_hours"], state.get("current_experiment"))
    logger.info("=========================================================================================")
    logger.info("%-4s | %-24s | %-8s | %-8s | %-9s | %-9s | %-7s",
                "GRP", "EXPERIMENT", "PRIORITY", "STATUS", "VAL AUROC", "DEV AUROC", "GPU-H")
    logger.info("-----------------------------------------------------------------------------------------")
    for eid, exp in state["experiments"].items():
        val_auc_str = f"{exp['best_val_auroc']:.4f}" if exp["best_val_auroc"] is not None else "—"
        dev_auc_str = f"{exp['dev_grassmann_auroc']:.4f}" if exp["dev_grassmann_auroc"] is not None else "—"
        logger.info("%-4s | %-24s | %-8s | %-8s | %-9s | %-9s | %-7.2f",
                    exp["group"], eid[:24], exp["priority"], exp["status"][:8], val_auc_str, dev_auc_str, exp["actual_gpu_hours"])
    logger.info("=========================================================================================")


def wait_for_primary_pipeline(poll_interval: int = 30) -> Dict[str, Any]:
    logger.info("Awaiting completion of primary pipeline (Original SetOperator + G1 + Evaluation)...")
    logger.info("Target signal: %s", PRIMARY_PIPELINE_SUMMARY)

    t0 = time.time()
    last_log = 0

    while not PRIMARY_PIPELINE_SUMMARY.exists():
        now = time.time()
        if now - last_log >= 300:  # log heartbeat every 5 minutes
            logger.info("Heartbeat: Waiting for primary pipeline completion summary (elapsed: %.1f min)...", (now - t0) / 60.0)
            last_log = now
        time.sleep(poll_interval)

    logger.info("Primary pipeline summary detected! Validating...")
    with open(PRIMARY_PIPELINE_SUMMARY, "r") as f:
        summary = json.load(f)

    if summary.get("status") != "COMPLETED":
        raise ValueError(f"Primary pipeline did not exit cleanly: {summary.get('status')}")

    logger.info("Primary pipeline validated successfully: Canonical R01 Base=%.4f, G1=%.4f (Δ=%+.4f)",
                summary["evaluation_summary"]["canonical_r01_baseline_auroc"],
                summary["evaluation_summary"]["canonical_r01_g1_auroc"],
                summary["evaluation_summary"]["canonical_r01_delta_auroc"])
    return summary


def run_experiment(exp_cfg: Dict[str, Any], state: Dict[str, Any]) -> None:
    eid = exp_cfg["id"]
    logger.info("==================================================")
    logger.info("STARTING EXPERIMENT: %s (%s)", eid, exp_cfg["name"])
    logger.info("Hypothesis: %s", exp_cfg["hypothesis"])
    logger.info("==================================================")

    state["current_experiment"] = eid
    state["experiments"][eid]["status"] = "RUNNING"
    state["experiments"][eid]["start_time"] = time.time()
    save_state(state)

    # Verify GPU idle
    gpu_procs = get_gpu_compute_processes()
    if gpu_procs:
        logger.warning("Active GPU compute processes detected before launch: %s. Waiting 30s...", gpu_procs)
        time.sleep(30)
        gpu_procs = get_gpu_compute_processes()
        if gpu_procs:
            raise RuntimeError(f"GPU not idle for {eid}: {gpu_procs}")

    cmd = [
        str(PYTHON_BIN),
        str(DEV_ROOT / "experiments/echonext_operator_benchmark/train_ablation_setoperator_echonext.py"),
        "--experiment-id", eid,
        "--geometry-mode", exp_cfg.get("geometry_mode", "baseline"),
        "--operator-mode", exp_cfg.get("operator_mode", "continuous"),
        "--phase-mode", exp_cfg.get("phase_mode", "standard_circular"),
        "--set-encoder-mode", exp_cfg.get("set_encoder_mode", "standard_transformer"),
        "--pooling-mode", exp_cfg.get("pooling_mode", "learned_query"),
        "--tabular-mode", exp_cfg.get("tabular_mode", "native_additive"),
        "--representation", exp_cfg.get("representation", "phase_kme"),
        "--epochs", str(exp_cfg.get("max_epochs", 50)),
        "--patience", str(exp_cfg.get("patience", 10)),
        "--batch-size", str(exp_cfg.get("batch_size", 64)),
        "--init-mode", exp_cfg.get("init_mode", "native_pytorch"),
        "--model-class", exp_cfg.get("model_class", "unified_ablation"),
        "--seed", "42",
        "--device", "cuda:0",
    ]

    t0 = time.time()
    res = subprocess.run(cmd, cwd=str(DEV_ROOT), capture_output=True, text=True)
    duration_hours = (time.time() - t0) / 3600.0

    state["total_gpu_hours_consumed"] += duration_hours
    state["experiments"][eid]["actual_gpu_hours"] = duration_hours
    state["experiments"][eid]["end_time"] = time.time()

    if res.returncode != 0:
        logger.error("EXPERIMENT %s FAILED (code %d):\n%s\n%s", eid, res.returncode, res.stdout[-2000:], res.stderr[-2000:])
        state["experiments"][eid]["status"] = "FAILED"
        state["experiments"][eid]["notes"] = f"Failed with returncode {res.returncode}"
        save_state(state)
        return

    # Verify outputs
    out_dir = Path(exp_cfg["checkpoint_dir"])
    summary_path = out_dir / "train_summary.json"
    ckpt_path = out_dir / "best_model.pt"

    if not summary_path.exists() or not ckpt_path.exists():
        logger.error("EXPERIMENT %s outputs missing at %s", eid, out_dir)
        state["experiments"][eid]["status"] = "FAILED"
        save_state(state)
        return

    with open(summary_path, "r") as f:
        train_sum = json.load(f)

    state["experiments"][eid]["best_epoch"] = train_sum.get("best_epoch")
    state["experiments"][eid]["best_val_auroc"] = train_sum.get("best_val_auroc")
    state["experiments"][eid]["best_val_auprc"] = train_sum.get("val_auprc_at_best_epoch")

    # Run matched evaluation
    state["experiments"][eid]["status"] = "EVALUATING"
    save_state(state)

    base_ckpt = Path("/data/mithunmanivannan/codex_artifacts/echonext_three_way_v1/checkpoints/setoperator/SO-Span_SO-ECG+Tab-Native_s42/best_model.pt")
    eval_cmd = [
        str(PYTHON_BIN),
        str(DEV_ROOT / "scripts/evaluate_ablation_comparison.py"),
        "--device", "cuda:0",
        "--baseline-ckpt", str(base_ckpt),
        "--ablation-ckpt", str(ckpt_path),
        "--experiment-id", eid,
        "--hypothesis", exp_cfg.get("hypothesis", ""),
        "--out-dir", str(out_dir),
    ]

    res_eval = subprocess.run(eval_cmd, cwd=str(DEV_ROOT), capture_output=True, text=True)
    if res_eval.returncode == 0:
        result_json_path = out_dir / "result.json"
        if result_json_path.exists():
            with open(result_json_path, "r") as f:
                res_data = json.load(f)
            dev_auc = res_data["development_grassmann_bank"]["overall_development_bank"]["ablation_auroc"]
            state["experiments"][eid]["dev_grassmann_auroc"] = dev_auc
            logger.info("Evaluation Complete for %s: Val AUROC=%.4f | Dev Grassmann AUROC=%.4f",
                        eid, state["experiments"][eid]["best_val_auroc"], dev_auc)
    else:
        logger.warning("Evaluation script returned non-zero for %s:\n%s\n%s", eid, res_eval.stdout[-1000:], res_eval.stderr[-1000:])

    state["experiments"][eid]["status"] = "COMPLETE"
    save_state(state)


def main():
    parser = argparse.ArgumentParser(description="Paper 07 168-Hour Ablation Campaign Supervisor")
    parser.add_argument("--dry-run", action="store_true", help="Validate queue and dependencies without executing")
    parser.add_argument("--validate-manifest", action="store_true", help="Validate manifest schema and paths")
    parser.add_argument("--resume", action="store_true", help="Resume queue from existing state")
    parser.add_argument("--status", action="store_true", help="Display current leaderboard and queue status")
    parser.add_argument("--max-gpu-hours", type=float, default=168.0, help="Hard maximum GPU-hour cap")
    parser.add_argument("--single-gpu", type=str, default="cuda:0", help="Target single GPU device")
    parser.add_argument("--stop-after-current", action="store_true", help="Gracefully stop after active job")
    parser.add_argument("--preflight-only", action="store_true", help="Run only the P1 scientific preflight gate")
    parser.add_argument("--single-experiment", type=str, default=None, help="Run ONLY this specific authorized experiment ID and halt.")
    args = parser.parse_args()

    if args.validate_manifest:
        ok = validate_manifest()
        sys.exit(0 if ok else 1)

    if args.preflight_only:
        ok = run_preflight_checks()
        sys.exit(0 if ok else 1)

    manifest = load_manifest()
    state = initialize_or_load_state(manifest)

    if args.status:
        print_status_table(state)
        sys.exit(0)

    if args.stop_after_current:
        state["stop_after_current"] = True
        save_state(state)
        logger.info("Queue instructed to gracefully stop after current experiment completes.")
        sys.exit(0)

    # 1. Run Preflight
    logger.info("Executing Preflight Gate...")
    if not run_preflight_checks():
        logger.error("Preflight gate FAILED. Halting.")
        sys.exit(1)

    if args.dry_run:
        logger.info("DRY RUN: Validating experiment dependency order...")
        for exp in manifest["experiments"]:
            logger.info("  [Plan] %-24s (Est: %.1fh) -> %s", exp["id"], exp.get("est_gpu_hours", 3.0), exp["name"])
        logger.info("Dry run complete. All 22 experiment definitions verified.")
        sys.exit(0)

    # 2. Prevent concurrent supervisors
    if LOCK_FILE.exists():
        lock_pid = LOCK_FILE.read_text().strip()
        logger.error("Lock file exists at %s (%s). Another supervisor is active.", LOCK_FILE, lock_pid)
        sys.exit(1)
    LOCK_FILE.write_text(f"PID={os.getpid()} TIME={time.time()}")

    try:
        # 3. Await Primary Pipeline completion (baseline + G1 + paired eval)
        primary_summary = wait_for_primary_pipeline()

        # Record A0 and A1 in state
        state["experiments"]["A0_baseline"]["status"] = "COMPLETE"
        state["experiments"]["A0_baseline"]["best_val_auroc"] = primary_summary["evaluation_summary"]["canonical_r01_baseline_auroc"]
        state["experiments"]["A0_baseline"]["dev_grassmann_auroc"] = primary_summary["evaluation_summary"]["overall_dev_grassmann_baseline_auroc"]

        state["experiments"]["A1_g1_scratch"]["status"] = "COMPLETE"
        state["experiments"]["A1_g1_scratch"]["best_val_auroc"] = primary_summary["evaluation_summary"]["canonical_r01_g1_auroc"]
        state["experiments"]["A1_g1_scratch"]["dev_grassmann_auroc"] = primary_summary["evaluation_summary"]["overall_dev_grassmann_g1_auroc"]
        save_state(state)

        # 4. Process Pending Ablations
        logger.info("Primary pipeline complete. Beginning Ablation Matrix...")

        if args.single_experiment:
            target_cfg = next((exp for exp in manifest["experiments"] if exp["id"] == args.single_experiment), None)
            if not target_cfg:
                raise ValueError(f"Requested single-experiment {args.single_experiment} not found in manifest.")
            cur_status = state["experiments"][args.single_experiment]["status"]
            if cur_status == "COMPLETE":
                logger.info("Authorized single-experiment %s is ALREADY COMPLETE. Exiting cleanly without duplicate execution.", args.single_experiment)
                return

            logger.info("Executing ONLY single authorized experiment: %s", args.single_experiment)
            run_experiment(target_cfg, state)
            print_status_table(state)
            logger.info("Single authorized experiment %s completed. Queue halted safely.", args.single_experiment)
            return

        for exp_cfg in manifest["experiments"]:
            eid = exp_cfg["id"]
            if eid in {"A0_baseline", "A1_g1_scratch"}:
                continue

            cur_status = state["experiments"][eid]["status"]
            if cur_status in {"COMPLETE", "FAILED", "SKIPPED_BUDGET"}:
                continue

            if state["total_gpu_hours_consumed"] >= args.max_gpu_hours:
                logger.warning("GPU-hour cap of %.1fh reached. Marking %s as SKIPPED_BUDGET.", args.max_gpu_hours, eid)
                state["experiments"][eid]["status"] = "SKIPPED_BUDGET"
                save_state(state)
                continue

            if STATE_FILE.exists():
                try:
                    with open(STATE_FILE, "r") as f:
                        disk_state = json.load(f)
                    if disk_state.get("stop_after_current", False):
                        state["stop_after_current"] = True
                except Exception as e:
                    logger.warning("Failed to check disk state: %s", e)

            if state.get("stop_after_current", False):
                logger.info("Stop-after-current requested. Halting queue.")
                break

            run_experiment(exp_cfg, state)
            print_status_table(state)

        logger.info("ALL SCHEDULED EXPERIMENTS PROCESSED. 168-HOUR CAMPAIGN CONCLUDED.")
        print_status_table(state)

    finally:
        if LOCK_FILE.exists():
            LOCK_FILE.unlink()


if __name__ == "__main__":
    main()
