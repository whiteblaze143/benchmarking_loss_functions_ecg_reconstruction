#!/usr/bin/env python3
"""EchoNext Master Training Queue.

Orchestrates sequential training across architectures and seeds:
1. GraphECG (Seeds 42, 43, 44, 45, 46)
2. SetOperator (SO-Span, SO-Basis, SO-12 across Seeds 42, 43, 44, 45, 46)
3. SetONet (SetONet-Span, SetONet-Basis, SetONet-12 across Seeds 42, 43, 44, 45, 46)

Features:
- Sequential execution to prevent GPU memory thrashing on A100-40GB.
- Real-time logging to /data/mithunmanivannan/codex_artifacts/echonext_three_way_v1/logs/training_queue.log.
- Persistent checkpoint verification.
- Skips already completed runs.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import subprocess
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("echonext_queue")

PYTHON_EXEC = "/home/mithunmanivannan/.venv/bin/python"
BENCHMARK_DIR = Path("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/echonext_operator_benchmark")
LOG_DIR = Path("/data/mithunmanivannan/codex_artifacts/echonext_three_way_v1/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
QUEUE_LOG = LOG_DIR / "training_queue.log"

CHECKPOINT_BASE = Path("/data/mithunmanivannan/codex_artifacts/echonext_three_way_v1/checkpoints")


def is_job_completed(job_type: str, seed: int, regime: str = "", tabular: str = "") -> bool:
    if job_type == "graphecg":
        ckpt = CHECKPOINT_BASE / "graphecg" / f"seed_{seed}" / "graphecg_best.pt"
        summary = CHECKPOINT_BASE / "graphecg" / f"seed_{seed}" / "train_summary.json"
        return ckpt.exists() and summary.exists()
    elif job_type == "setoperator":
        ckpt = CHECKPOINT_BASE / "setoperator" / f"{regime}_{tabular}_s{seed}" / "best_model.pt"
        summary = CHECKPOINT_BASE / "setoperator" / f"{regime}_{tabular}_s{seed}" / "train_summary.json"
        return ckpt.exists() and summary.exists()
    elif job_type == "setonet":
        ckpt = CHECKPOINT_BASE / "setonet" / f"{regime}_{tabular}_s{seed}" / "best_model.pt"
        summary = CHECKPOINT_BASE / "setonet" / f"{regime}_{tabular}_s{seed}" / "train_summary.json"
        return ckpt.exists() and summary.exists()
    return False


def run_command(cmd: list[str], desc: str) -> bool:
    logger.info(">>> STARTING: %s", desc)
    logger.info("Command: %s", " ".join(cmd))
    t0 = time.time()

    with open(QUEUE_LOG, "a") as f_log:
        f_log.write(f"\n{'='*80}\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] START: {desc}\n{'='*80}\n")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            f_log.write(line)
            f_log.flush()

        ret = proc.wait()

    duration = time.time() - t0
    if ret == 0:
        logger.info("<<< COMPLETED: %s in %.2f min", desc, duration / 60)
        return True
    else:
        logger.error("!!! FAILED (exit %d): %s in %.2f min", ret, desc, duration / 60)
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--graphecg-only", action="store_true")
    parser.add_argument("--setoperator-only", action="store_true")
    parser.add_argument("--setonet-only", action="store_true")
    args = parser.parse_args()

    logger.info("==================================================")
    logger.info("ECHONEXT REPRODUCTION & OPERATOR BENCHMARK QUEUE")
    logger.info("Seeds: %s | Epochs: %d | Patience: %d", args.seeds, args.epochs, args.patience)
    logger.info("Queue Log: %s", QUEUE_LOG)
    logger.info("==================================================")

    # 1. GraphECG Reproduction
    if not (args.setoperator_only or args.setonet_only):
        for seed in args.seeds:
            if is_job_completed("graphecg", seed):
                logger.info("GraphECG Seed %d already complete. Skipping.", seed)
                continue

            desc = f"GraphECG EchoNext Reproduction | Seed {seed}"
            cmd = [
                PYTHON_EXEC,
                str(BENCHMARK_DIR / "train_graphecg_echonext.py"),
                "--seed", str(seed),
                "--epochs", str(args.epochs),
                "--patience", str(args.patience),
                "--batch-size", "32",
            ]
            success = run_command(cmd, desc)
            if not success:
                logger.error("Queue halted on GraphECG failure.")
                return

    # 2. SetOperator (SO-Span, SO-Basis, SO-12)
    if not (args.graphecg_only or args.setonet_only):
        for regime in ["SO-Span", "SO-Basis", "SO-12"]:
            for seed in args.seeds:
                tabular = "SO-ECG+Tab-Native"
                if is_job_completed("setoperator", seed, regime=regime, tabular=tabular):
                    logger.info("SetOperator %s Seed %d already complete. Skipping.", regime, seed)
                    continue

                desc = f"SetOperator EchoNext {regime} | Seed {seed}"
                cmd = [
                    PYTHON_EXEC,
                    str(BENCHMARK_DIR / "train_setoperator_echonext.py"),
                    "--regime", regime,
                    "--tabular-condition", tabular,
                    "--seed", str(seed),
                    "--epochs", str(args.epochs),
                    "--patience", str(args.patience),
                    "--batch-size", "64",
                ]
                success = run_command(cmd, desc)
                if not success:
                    logger.error("Queue halted on SetOperator failure.")
                    return

    # 3. SetONet (SetONet-Span, SetONet-Basis, SetONet-12)
    if not (args.graphecg_only or args.setoperator_only):
        for regime in ["SetONet-Span", "SetONet-Basis", "SetONet-12"]:
            for seed in args.seeds:
                tabular = "SO-ECG+Tab-Trunk"
                if is_job_completed("setonet", seed, regime=regime, tabular=tabular):
                    logger.info("SetONet %s Seed %d already complete. Skipping.", regime, seed)
                    continue

                desc = f"SetONet EchoNext {regime} | Seed {seed}"
                cmd = [
                    PYTHON_EXEC,
                    str(BENCHMARK_DIR / "train_setonet_echonext.py"),
                    "--regime", regime,
                    "--tabular-condition", tabular,
                    "--seed", str(seed),
                    "--epochs", str(args.epochs),
                    "--patience", str(args.patience),
                    "--batch-size", "64",
                ]
                success = run_command(cmd, desc)
                if not success:
                    logger.error("Queue halted on SetONet failure.")
                    return

    logger.info("🎉 ALL ECHONEXT QUEUE JOBS SUCCESSFULLY COMPLETED!")


if __name__ == "__main__":
    main()
