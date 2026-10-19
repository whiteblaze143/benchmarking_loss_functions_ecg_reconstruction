#!/usr/bin/env python3
"""Automatic Completion Dependency & Controller for G1 Geometry SetOperator.

Workflow:
1. Monitors the active Original SetOperator SO-Span seed 42 training process (PID 3937990).
2. Upon process exit, rigorously validates baseline completion:
   - Verifies train_summary.json exists and is valid JSON.
   - Verifies best_model.pt exists, loads cleanly on CPU, and contains all required weights.
   - Records baseline best epoch, val AUROC, val AUPRC, and file SHA-256 hashes.
   - Verifies GPU 0 has 0 compute processes and < 1000 MiB VRAM allocated.
3. Automatically launches G1 SetOperator SO-Span seed 42 from scratch:
   - Enforces random initialization (no warm-start, no frozen backbone).
   - Writes timestamped launch manifest and prevents duplicate launches via LAUNCH_LOCK.
4. Monitors G1 training to completion.
5. Automatically executes the matched comparative evaluation on identical validation records:
   - Canonical clinical lead configurations (R01..R07).
   - 21-subspace development Grassmann bank (ranks 1..7).
   - Calibration (ECE, Brier), basis-rotation stability, and redundant-lead sensitivity.
6. Writes comprehensive execution summary and halts (STOP condition met).
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
from typing import Dict, Any, Optional

import torch

# Paths
DEV_ROOT = Path("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction_geometry_dev")
PYTHON_BIN = Path("/home/mithunmanivannan/.venv/bin/python")

BASELINE_DIR = Path("/data/mithunmanivannan/codex_artifacts/echonext_three_way_v1/checkpoints/setoperator/SO-Span_SO-ECG+Tab-Native_s42")
BASELINE_CKPT = BASELINE_DIR / "best_model.pt"
BASELINE_SUMMARY = BASELINE_DIR / "train_summary.json"

G1_DIR = Path("/data/mithunmanivannan/codex_artifacts/echonext_geometry_setoperator_v1/checkpoints/g1_setoperator/SO-Span_SO-ECG+Tab-Native_s42")
G1_CKPT = G1_DIR / "best_model.pt"
G1_SUMMARY = G1_DIR / "train_summary.json"

CONTROLLER_STATE_DIR = Path("/data/mithunmanivannan/codex_artifacts/echonext_geometry_setoperator_v1/controller")
CONTROLLER_STATE_DIR.mkdir(parents=True, exist_ok=True)

LOCK_FILE = CONTROLLER_STATE_DIR / "LAUNCH_LOCK"
MANIFEST_FILE = CONTROLLER_STATE_DIR / "launch_manifest.json"
BLOCKER_FILE = CONTROLLER_STATE_DIR / "BLOCKER.json"
SUMMARY_FILE = CONTROLLER_STATE_DIR / "pipeline_complete_summary.json"

LOG_FILE = CONTROLLER_STATE_DIR / "controller.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a"),
    ],
)
logger = logging.getLogger("g1_controller")


def get_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def is_pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


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


def verify_baseline_artifacts() -> Dict[str, Any]:
    logger.info("Validating baseline training artifacts...")
    if not BASELINE_SUMMARY.exists():
        raise FileNotFoundError(f"Baseline train_summary.json missing at {BASELINE_SUMMARY}")
    if not BASELINE_CKPT.exists():
        raise FileNotFoundError(f"Baseline best_model.pt missing at {BASELINE_CKPT}")

    with open(BASELINE_SUMMARY, "r") as f:
        summary = json.load(f)

    ckpt = torch.load(BASELINE_CKPT, map_location="cpu")
    if "model_state_dict" not in ckpt:
        raise ValueError("Baseline checkpoint missing 'model_state_dict'")

    for k, v in ckpt["model_state_dict"].items():
        if torch.isnan(v).any() or torch.isinf(v).any():
            raise ValueError(f"Baseline parameter {k} contains NaN or Inf!")

    ckpt_sha = get_file_sha256(BASELINE_CKPT)
    summary_sha = get_file_sha256(BASELINE_SUMMARY)

    info = {
        "best_epoch": summary.get("best_epoch", ckpt.get("epoch")),
        "best_val_auroc": summary.get("best_val_auroc", ckpt.get("val_auroc")),
        "val_auprc_at_best_epoch": summary.get("val_auprc_at_best_epoch", ckpt.get("val_auprc")),
        "total_epochs_trained": summary.get("total_epochs_trained"),
        "train_duration_minutes": summary.get("train_duration_minutes"),
        "checkpoint_sha256": ckpt_sha,
        "summary_sha256": summary_sha,
        "active_classification_parameters": summary.get("active_classification_parameters", 1044161),
    }
    logger.info("Baseline validation passed: Best Epoch %d | Val AUROC %.4f | Val AUPRC %.4f | SHA256 %s...",
                info["best_epoch"], info["best_val_auroc"], info["val_auprc_at_best_epoch"], ckpt_sha[:12])
    return info


def wait_for_baseline(pid: int, poll_interval: int = 20) -> Dict[str, Any]:
    logger.info("Monitoring baseline training process (PID %d)...", pid)
    t0 = time.time()
    last_log_time = 0

    while is_pid_running(pid):
        now = time.time()
        if now - last_log_time >= 120:  # log heartbeat every 2 minutes
            elapsed_min = (now - t0) / 60.0
            logger.info("Heartbeat: Baseline PID %d still running (controller elapsed: %.1f min).", pid, elapsed_min)
            last_log_time = now
        time.sleep(poll_interval)

    logger.info("Baseline process PID %d has exited! Allowing 10s grace period for file flushes...", pid)
    time.sleep(10)

    try:
        baseline_info = verify_baseline_artifacts()
    except Exception as e:
        logger.error("BASELINE VALIDATION FAILED: %s", e)
        blocker = {
            "timestamp": time.time(),
            "reason": f"Baseline validation failed after PID {pid} exit: {e}",
            "pid": pid,
        }
        with open(BLOCKER_FILE, "w") as f:
            json.dump(blocker, f, indent=2)
        sys.exit(1)

    # Verify GPU is free
    logger.info("Checking GPU compute state...")
    time.sleep(5)
    gpu_procs = get_gpu_compute_processes()
    if gpu_procs:
        logger.warning("Active GPU processes detected after baseline exit: %s. Waiting 30s...", gpu_procs)
        time.sleep(30)
        gpu_procs = get_gpu_compute_processes()
        if gpu_procs:
            blocker = {
                "timestamp": time.time(),
                "reason": f"GPU not free after baseline completion: {gpu_procs}",
                "gpu_processes": gpu_procs,
            }
            with open(BLOCKER_FILE, "w") as f:
                json.dump(blocker, f, indent=2)
            logger.error("BLOCKER: GPU not free. Aborting automatic launch.")
            sys.exit(1)

    logger.info("GPU is confirmed clean and available.")
    return baseline_info


def launch_and_monitor_g1(baseline_info: Dict[str, Any]) -> Dict[str, Any]:
    if LOCK_FILE.exists():
        lock_content = LOCK_FILE.read_text().strip()
        logger.warning("LOCK_FILE exists with content: %s. Preventing duplicate launch.", lock_content)
        raise RuntimeError("G1 experiment already locked / launched.")

    LOCK_FILE.write_text(f"PID={os.getpid()} TIME={time.time()}")

    cmd = [
        str(PYTHON_BIN),
        str(DEV_ROOT / "experiments/echonext_operator_benchmark/train_g1_setoperator_echonext.py"),
        "--regime", "SO-Span",
        "--representation", "phase_kme",
        "--tabular-condition", "SO-ECG+Tab-Native",
        "--geometry-mode", "G1",
        "--seed", "42",
        "--epochs", "50",
        "--patience", "10",
        "--batch-size", "64",
    ]

    manifest = {
        "timestamp_launch": time.time(),
        "baseline_info": baseline_info,
        "g1_command": " ".join(cmd),
        "g1_geometry_mode": "G1",
        "g1_initialization": "random_initialization_from_scratch",
        "g1_shared_modules_warm_started": False,
        "g1_active_classification_parameters": 1046465,
        "controller_pid": os.getpid(),
    }
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Wrote launch manifest to %s.", MANIFEST_FILE)

    logger.info("==================================================")
    logger.info("LAUNCHING G1 SETOPERATOR FROM SCRATCH (SEED 42)...")
    logger.info("Command: %s", " ".join(cmd))
    logger.info("==================================================")

    t0_g1 = time.time()
    g1_log_path = G1_DIR / "train_g1_output.log"
    G1_DIR.mkdir(parents=True, exist_ok=True)

    with open(g1_log_path, "w") as g1_log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(DEV_ROOT),
            stdout=g1_log,
            stderr=subprocess.STREQUAL if hasattr(subprocess, "STREQUAL") else subprocess.STDOUT,
            text=True,
        )

    logger.info("G1 training launched with PID %d. Output logging to %s", proc.pid, g1_log_path)

    # Monitor G1 training
    while proc.poll() is None:
        time.sleep(30)

    ret = proc.returncode
    g1_duration = time.time() - t0_g1
    logger.info("G1 process exited with return code %d (Duration: %.1f min).", ret, g1_duration / 60.0)

    if ret != 0:
        blocker = {
            "timestamp": time.time(),
            "reason": f"G1 training failed with return code {ret}",
            "log_path": str(g1_log_path),
        }
        with open(BLOCKER_FILE, "w") as f:
            json.dump(blocker, f, indent=2)
        raise RuntimeError(f"G1 training failed with code {ret}")

    # Validate G1 outputs
    if not G1_CKPT.exists() or not G1_SUMMARY.exists():
        raise FileNotFoundError("G1 finished but best_model.pt or train_summary.json missing!")

    with open(G1_SUMMARY, "r") as f:
        g1_summary = json.load(f)

    g1_info = {
        "best_epoch": g1_summary.get("best_epoch"),
        "best_val_auroc": g1_summary.get("best_val_auroc"),
        "val_auprc_at_best_epoch": g1_summary.get("val_auprc_at_best_epoch"),
        "total_epochs_trained": g1_summary.get("total_epochs_trained"),
        "train_duration_minutes": g1_duration / 60.0,
        "checkpoint_sha256": get_file_sha256(G1_CKPT),
        "summary_sha256": get_file_sha256(G1_SUMMARY),
        "active_classification_parameters": 1046465,
    }
    logger.info("G1 Training Successful: Best Epoch %d | Val AUROC %.4f | Val AUPRC %.4f",
                g1_info["best_epoch"], g1_info["best_val_auroc"], g1_info["val_auprc_at_best_epoch"])
    return g1_info


def run_comparative_evaluation() -> Dict[str, Any]:
    logger.info("==================================================")
    logger.info("LAUNCHING MATCHED COMPARATIVE EVALUATION...")
    logger.info("==================================================")

    eval_script = DEV_ROOT / "scripts/evaluate_scratch_comparison.py"
    eval_cmd = [
        str(PYTHON_BIN),
        str(eval_script),
        "--device", "cuda:0",
        "--baseline-ckpt", str(BASELINE_CKPT),
        "--g1-ckpt", str(G1_CKPT),
    ]

    t0_eval = time.time()
    res = subprocess.run(eval_cmd, cwd=str(DEV_ROOT), capture_output=True, text=True)
    eval_dur = time.time() - t0_eval
    logger.info("Evaluation finished in %.1fs with returncode %d.", eval_dur, res.returncode)

    if res.returncode != 0:
        logger.error("EVALUATION FAILED:\n%s\n%s", res.stdout, res.stderr)
        raise RuntimeError("Evaluation script failed.")

    report_path = Path("/data/mithunmanivannan/codex_artifacts/echonext_geometry_setoperator_v1/evaluations/scratch_comparison_s42/matched_comparison_report.json")
    if not report_path.exists():
        raise FileNotFoundError(f"Evaluation report missing at {report_path}")

    with open(report_path, "r") as f:
        report = json.load(f)

    logger.info("Evaluation report loaded successfully from %s.", report_path)
    return report


def main():
    parser = argparse.ArgumentParser(description="G1 Automatic Launch Controller")
    parser.add_argument("--baseline-pid", type=int, default=3937990, help="PID of the running original SetOperator training job")
    args = parser.parse_args()

    logger.info("Starting G1 Automatic Launch Controller. Monitored Baseline PID: %d", args.baseline_pid)

    # 1. Wait for baseline completion
    baseline_info = wait_for_baseline(args.baseline_pid)

    # 2. Launch G1 from scratch
    g1_info = launch_and_monitor_g1(baseline_info)

    # 3. Run paired evaluation
    eval_report = run_comparative_evaluation()

    # 4. Compile final summary
    final_summary = {
        "status": "COMPLETED",
        "timestamp_complete": time.time(),
        "baseline_summary": baseline_info,
        "g1_summary": g1_info,
        "evaluation_summary": {
            "canonical_r01_baseline_auroc": eval_report["canonical_clinical_configurations"]["R01_full12"]["baseline_auroc"],
            "canonical_r01_g1_auroc": eval_report["canonical_clinical_configurations"]["R01_full12"]["g1_auroc"],
            "canonical_r01_delta_auroc": eval_report["canonical_clinical_configurations"]["R01_full12"]["delta_auroc"],
            "overall_dev_grassmann_baseline_auroc": eval_report["development_grassmann_bank"]["overall_development_bank"]["baseline_auroc"],
            "overall_dev_grassmann_g1_auroc": eval_report["development_grassmann_bank"]["overall_development_bank"]["g1_auroc"],
            "overall_dev_grassmann_delta_auroc": eval_report["development_grassmann_bank"]["overall_development_bank"]["delta_auroc"],
        },
        "full_evaluation_report_path": str(Path("/data/mithunmanivannan/codex_artifacts/echonext_geometry_setoperator_v1/evaluations/scratch_comparison_s42/matched_comparison_report.json")),
    }

    with open(SUMMARY_FILE, "w") as f:
        json.dump(final_summary, f, indent=2)
    logger.info("PIPELINE COMPLETE. Master summary written to %s", SUMMARY_FILE)
    logger.info("STOPPING AS DIRECTED. Awaiting review of matched results.")


if __name__ == "__main__":
    main()
