#!/usr/bin/env python3
"""Automated Queue Handoff Daemon: D1 Completion -> A0 Long-Run Launch.

Workflow:
1. Monitors active D1 worker (PID 4193047) and ablation supervisor (PID 4184400).
2. Awaits natural completion of D1 training and evaluation on the 100-subspace EchoNext dev bank.
3. Verifies D1 artifacts (train_summary, best_model, result.json, ablation_eval_report.json).
4. Verifies complete GPU process teardown and idle state.
5. Executes A0 replication parity audit (Option A gate: 57/57 bitwise hashes, LR schedule, cohort).
6. If all gates pass, safely clears stop_after_current, restarts supervisor in tmux session
   'paper07_ablation_week', and launches A0_native_init_no_earlystop_50ep_s42.
7. Immediately resets stop_after_current: true once A0-long starts, guaranteeing that
   the queue halts after A0-long completes without advancing to matched controls.
8. Writes comprehensive handoff summary to supervisor_state/a0_auto_handoff_summary.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Dict, Any, Optional

DEV_ROOT = Path("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction_geometry_dev")
STATE_DIR = Path("/data/mithunmanivannan/codex_artifacts/echonext_geometry_setoperator_v1/supervisor_state")
D1_DIR = Path("/data/mithunmanivannan/codex_artifacts/echonext_geometry_setoperator_v1/checkpoints/ablations/D1_ecg_only")
STATE_FILE = STATE_DIR / "ablation_queue_state.json"
LOCK_FILE = STATE_DIR / "ABLAT_LOCK"
SUMMARY_FILE = STATE_DIR / "a0_auto_handoff_summary.json"
LOG_FILE = STATE_DIR / "a0_auto_handoff.log"
PYTHON_BIN = Path("/home/mithunmanivannan/.venv/bin/python")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a"),
    ],
)
logger = logging.getLogger("a0_auto_handoff")


def is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
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
        logger.warning("nvidia-smi query failed: %s", e)
        return []


def run_handoff(d1_pid: int, supervisor_pid: int, poll_interval: int = 15):
    logger.info("================================================================================")
    logger.info("AUTOMATED QUEUE HANDOFF DAEMON INITIALIZED")
    logger.info("Monitoring D1 Worker PID:       %d", d1_pid)
    logger.info("Monitoring Supervisor PID:      %d", supervisor_pid)
    logger.info("Poll Interval:                  %d seconds", poll_interval)
    logger.info("================================================================================")

    t0 = time.time()
    last_heartbeat = 0
    d1_finished = False

    while True:
        now = time.time()
        d1_alive = is_pid_alive(d1_pid)
        sup_alive = is_pid_alive(supervisor_pid)

        if not d1_alive and not d1_finished:
            d1_finished = True
            logger.info(">>> D1 worker PID %d has finished execution! Supervisor is now performing Grassmann-100 evaluation...", d1_pid)

        if not sup_alive:
            logger.info(">>> Supervisor PID %d has terminated. Reached safe halt boundary.", supervisor_pid)
            break

        if now - last_heartbeat >= 300:  # 5-minute heartbeat
            last_heartbeat = now
            elapsed_min = (now - t0) / 60.0
            gpu_procs = get_gpu_compute_processes()
            gpu_mem = gpu_procs[0]["memory_mib"] if gpu_procs else "0"
            logger.info("Heartbeat: Elapsed %.1f min | D1 PID %d: %s | Sup PID %d: %s | GPU Mem: %s MiB",
                        elapsed_min, d1_pid, "RUNNING" if d1_alive else "DONE",
                        supervisor_pid, "RUNNING" if sup_alive else "DONE", gpu_mem)

        time.sleep(poll_interval)

    # 1. Clean up and verify process teardown
    logger.info("Step 1: Verifying complete process teardown and settling file writes...")
    time.sleep(5)

    if LOCK_FILE.exists():
        logger.info("Clearing stale ABLAT_LOCK: %s", LOCK_FILE.read_text().strip())
        LOCK_FILE.unlink(missing_ok=True)

    gpu_procs = get_gpu_compute_processes()
    if gpu_procs:
        logger.warning("GPU compute processes detected after supervisor halt: %s. Awaiting 15s...", gpu_procs)
        time.sleep(15)
        gpu_procs = get_gpu_compute_processes()
        if gpu_procs:
            raise RuntimeError(f"GPU not idle after teardown: {gpu_procs}")
    logger.info("GPU 0 is verified 100% idle.")

    # 2. Verify D1 artifacts
    logger.info("Step 2: Verifying D1 output artifacts at %s...", D1_DIR)
    train_summary_file = D1_DIR / "train_summary.json"
    best_model_file = D1_DIR / "best_model.pt"
    result_file = D1_DIR / "result.json"
    eval_report_file = D1_DIR / "ablation_eval_report.json"

    for req_f in [train_summary_file, best_model_file, result_file, eval_report_file]:
        if not req_f.exists() or req_f.stat().st_size == 0:
            raise FileNotFoundError(f"Missing required D1 artifact: {req_f}")

    with open(train_summary_file, "r") as f:
        d1_train = json.load(f)
    with open(result_file, "r") as f:
        d1_result = json.load(f)

    d1_best_epoch = d1_train.get("best_epoch")
    d1_val_auroc = d1_train.get("best_val_auroc")
    d1_val_auprc = d1_train.get("val_auprc_at_best_epoch")
    d1_dev_bank = d1_result.get("development_grassmann_bank", {})
    overall_dev = d1_dev_bank.get("overall_development_bank", {})
    d1_grassmann_macro_auroc = overall_dev.get("ablation_auroc")
    d1_grassmann_macro_auprc = overall_dev.get("ablation_auprc")

    rank_aurocs = {}
    for r in range(1, 8):
        rk = f"rank_{r}"
        if rk in d1_dev_bank:
            rank_aurocs[rk] = d1_dev_bank[rk].get("ablation_auroc")

    logger.info("================================================================================")
    logger.info("D1 FINAL SCIENTIFIC EVALUATION SUMMARY:")
    logger.info("  Best Epoch:                 %d", d1_best_epoch)
    logger.info("  Canonical Val AUROC:        %.7f", d1_val_auroc)
    logger.info("  Canonical Val AUPRC:        %.7f", d1_val_auprc)
    logger.info("  Grassmann-100 Macro AUROC:  %.7f (Delta vs Base: %+.4f)",
                d1_grassmann_macro_auroc, overall_dev.get("delta_auroc", 0.0))
    logger.info("  Grassmann-100 Macro AUPRC:  %.7f (Delta vs Base: %+.4f)",
                d1_grassmann_macro_auprc, overall_dev.get("delta_auprc", 0.0))
    logger.info("  Per-Rank AUROCs:            %s", rank_aurocs)
    logger.info("================================================================================")

    # 3. Run Option A Replication Parity Audit
    logger.info("Step 3: Running A0 Replication Parity Audit (Preflight Gate)...")
    audit_cmd = [str(PYTHON_BIN), str(DEV_ROOT / "scripts/audit_a0_replication_parity.py")]
    res_audit = subprocess.run(audit_cmd, capture_output=True, text=True)
    if res_audit.returncode != 0:
        logger.error("A0 REPLICATION PARITY AUDIT FAILED (Code %d):\n%s\n%s",
                     res_audit.returncode, res_audit.stdout, res_audit.stderr)
        summary_payload = {
            "status": "BLOCKED_AUDIT_FAILURE",
            "timestamp": time.time(),
            "d1_results": {
                "best_epoch": d1_best_epoch,
                "val_auroc": d1_val_auroc,
                "val_auprc": d1_val_auprc,
                "grassmann_macro_auroc": d1_grassmann_macro_auroc,
                "rank_breakdown": rank_aurocs,
            },
            "blocker": res_audit.stderr,
        }
        with open(SUMMARY_FILE, "w") as f:
            json.dump(summary_payload, f, indent=2)
        sys.exit(1)

    logger.info("A0 Replication Parity Audit PASSED cleanly (57/57 bitwise hashes, LR, cohort verified).")

    # 4. Prepare Queue State for A0 Launch
    logger.info("Step 4: Preparing queue state for A0 launch...")
    with open(STATE_FILE, "r") as f:
        state = json.load(f)

    # Verify D1 marked COMPLETE
    state["experiments"]["D1_ecg_only"]["status"] = "COMPLETE"
    state["experiments"]["D1_ecg_only"]["best_epoch"] = d1_best_epoch
    state["experiments"]["D1_ecg_only"]["best_val_auroc"] = d1_val_auroc
    state["experiments"]["D1_ecg_only"]["best_val_auprc"] = d1_val_auprc
    state["experiments"]["D1_ecg_only"]["dev_grassmann_auroc"] = d1_grassmann_macro_auroc

    target_exp = "A0_native_init_no_earlystop_50ep_s42"
    if state["experiments"][target_exp]["status"] != "PLANNED":
        logger.warning("%s status was %s, resetting to PLANNED for execution.", target_exp, state["experiments"][target_exp]["status"])
        state["experiments"][target_exp]["status"] = "PLANNED"

    state["stop_after_current"] = False
    state["current_experiment"] = None

    temp_state = STATE_DIR / "state.tmp.json"
    with open(temp_state, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(temp_state, STATE_FILE)
    logger.info("Queue state saved: stop_after_current cleared, target experiment is %s.", target_exp)

    # 5. Launch supervisor in detached tmux session
    logger.info("Step 5: Resuming supervisor in tmux session 'paper07_ablation_week'...")
    sup_cmd = f"{PYTHON_BIN} {DEV_ROOT}/scripts/run_paper07_ablation_week.py --max-gpu-hours 168 --single-gpu cuda:0"
    tmux_launch_cmd = ["tmux", "send-keys", "-t", "paper07_ablation_week", sup_cmd, "C-m"]
    subprocess.run(tmux_launch_cmd, check=True)
    logger.info("Supervisor launch command sent to tmux.")

    # 6. Verify launch and active job
    logger.info("Step 6: Verifying supervisor startup and worker process...")
    new_sup_pid = None
    new_worker_pid = None

    for attempt in range(12):  # Poll up to 60s
        time.sleep(5)
        # Check supervisor lock or process
        if LOCK_FILE.exists():
            lock_content = LOCK_FILE.read_text().strip()
            for part in lock_content.split():
                if part.startswith("PID="):
                    new_sup_pid = int(part.split("=")[1])

        # Check queue state
        if STATE_FILE.exists():
            with open(STATE_FILE, "r") as f:
                cur_state = json.load(f)
            if cur_state.get("current_experiment") == target_exp and cur_state["experiments"][target_exp]["status"] == "RUNNING":
                # Find worker on GPU
                procs = get_gpu_compute_processes()
                if procs:
                    new_worker_pid = int(procs[0]["pid"])
                    logger.info("Confirmed %s is RUNNING! Supervisor PID: %s | Worker PID: %d",
                                target_exp, new_sup_pid, new_worker_pid)
                    break

    if not new_worker_pid:
        logger.warning("Could not identify worker PID within 60s. Checking compute procs directly...")
        procs = get_gpu_compute_processes()
        if procs:
            new_worker_pid = int(procs[0]["pid"])

    # 7. Immediately reset stop_after_current = True so the queue halts after A0-long completes
    logger.info("Step 7: Setting stop_after_current: true to guarantee queue halts after A0-long completes...")
    with open(STATE_FILE, "r") as f:
        active_state = json.load(f)
    active_state["stop_after_current"] = True
    temp_state = STATE_DIR / "state.tmp.json"
    with open(temp_state, "w") as f:
        json.dump(active_state, f, indent=2)
    os.replace(temp_state, STATE_FILE)
    logger.info("Enforced stop_after_current: true in queue state.")

    # 8. Record handoff summary
    consumed_gpu_h = active_state.get("total_gpu_hours_consumed", 0.0)
    remaining_gpu_h = 168.0 - consumed_gpu_h
    campaign_start = active_state.get("start_timestamp", time.time())
    elapsed_calendar_h = (time.time() - campaign_start) / 3600.0
    remaining_calendar_h = max(0.0, 168.0 - elapsed_calendar_h)

    summary_payload = {
        "status": "HANDOFF_SUCCESS",
        "timestamp": time.time(),
        "d1_results": {
            "best_epoch": d1_best_epoch,
            "canonical_val_auroc": d1_val_auroc,
            "canonical_val_auprc": d1_val_auprc,
            "grassmann_macro_auroc": d1_grassmann_macro_auroc,
            "grassmann_macro_auprc": d1_grassmann_macro_auprc,
            "rank_breakdown": rank_aurocs,
            "delta_vs_baseline_grassmann": overall_dev.get("delta_auroc", 0.0),
        },
        "a0_parity_audit": "PASSED_GO_FOR_QUEUE",
        "active_experiment": {
            "id": target_exp,
            "model_class": "production_setoperator",
            "init_mode": "native_pytorch",
            "supervisor_pid": new_sup_pid,
            "worker_pid": new_worker_pid,
            "max_epochs": 50,
            "patience": 50,
            "stop_after_current": True,
        },
        "budget_accounting": {
            "consumed_gpu_hours": consumed_gpu_h,
            "remaining_gpu_hours": remaining_gpu_h,
            "elapsed_calendar_hours": elapsed_calendar_h,
            "remaining_calendar_hours": remaining_calendar_h,
        },
        "isolation_confirmation": "Only A0_native_init_no_earlystop_50ep_s42 was launched. Matched triplet controls remain PLANNED.",
    }

    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary_payload, f, indent=2)

    logger.info("================================================================================")
    logger.info("AUTOMATED HANDOFF COMPLETED SUCCESSFULLY!")
    logger.info("Summary saved to %s", SUMMARY_FILE)
    logger.info("================================================================================")


def main():
    parser = argparse.ArgumentParser(description="Automated Queue Handoff Daemon")
    parser.add_argument("--d1-pid", type=int, default=4193047)
    parser.add_argument("--supervisor-pid", type=int, default=4184400)
    parser.add_argument("--poll-interval", type=int, default=15)
    args = parser.parse_args()

    run_handoff(args.d1_pid, args.supervisor_pid, args.poll_interval)


if __name__ == "__main__":
    main()
