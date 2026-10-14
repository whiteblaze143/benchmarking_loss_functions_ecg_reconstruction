#!/usr/bin/env python3
"""Overnight Definitive Promotion Chain Orchestrator.

Automated Stage Progression:
1. Stage 1 (Dataset Gate): Waits for build_final_r028_pipeline.py to finish materializing
   all 4 representation files and verified Grassmann bank.
2. Stage 2 (5-Seed Training): Runs train_final_5seeds.py across seeds {42, 43, 44, 45, 46}
   for both Basis and Span with Fold-9 checkpoint selection.
3. Stage 3 (Manifest Freeze): Runs freeze_submission_manifest.py to freeze all artifact SHA-256 hashes.
4. Stage 4 (Fold-10 Unsealing): Runs evaluate_final_fold10_locked.py under the preregistered protocol.
5. Stage 5 (Definitive Tables): Runs generate_cvpr_definitive_tables.py to compile publication tables.
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("overnight_promotion_chain")

REPO_DIR = Path(__file__).resolve().parents[4]
EXPERIMENT_DIR = REPO_DIR / "experiments/ptbxl_distributional_repecg"
SCRIPTS_DIR = EXPERIMENT_DIR / "scripts/paper07"
PYTHON_BIN = Path(sys.executable)

CODEX_DIR = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg")
OUT_DIR = CODEX_DIR / "paper07_operator/final_representations"
CHECKPOINTS_DIR = CODEX_DIR / "paper07_operator/final_checkpoints"
MANIFEST_PATH = CODEX_DIR / "paper07_operator/frozen_submission_manifest.json"
FOLD10_OUT_DIR = CODEX_DIR / "paper07_operator/final_fold10_evaluation"
TABLES_OUT_DIR = CODEX_DIR / "paper07_operator/final_tables"


def wait_for_stage1_datasets(poll_interval: int = 60) -> None:
    """Wait for all 4 Stage 1 representation files to exist and be non-empty."""
    logger.info("=== Stage 1: Waiting for Dataset Materialization (Folds 1-8 representations & Fold 9) ===")
    required_files = [
        OUT_DIR / "representation_final_train_basis_only.npz",
        OUT_DIR / "representation_final_train_operator_span.npz",
        OUT_DIR / "representation_final_select_waveforms.npz",
        OUT_DIR / "expanded_grassmann_subspace_bank.npz",
        OUT_DIR / "final_scaler.json",
        OUT_DIR / "response_fit_final.npz",
    ]

    while True:
        missing = [str(f) for f in required_files if not (f.exists() and f.stat().st_size > 1024)]
        if not missing:
            logger.info("All Stage 1 representation files verified on disk!")
            break
        logger.info("Waiting for %d files to finish materialization (next check in %ds)...", len(missing), poll_interval)
        time.sleep(poll_interval)


def run_stage2_training(seeds: list[int] = [42, 43, 44, 45, 46]) -> None:
    """Run train_final_5seeds.py for Basis and Span."""
    logger.info("=== Stage 2: Launching Five-Seed Training (Seeds: %s) ===", seeds)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(PYTHON_BIN),
        str(SCRIPTS_DIR / "train_final_5seeds.py"),
        "--mode", "both",
        "--seeds", *[str(s) for s in seeds],
        "--train-basis-npz", str(OUT_DIR / "representation_final_train_basis_only.npz"),
        "--train-span-npz", str(OUT_DIR / "representation_final_train_operator_span.npz"),
        "--val-npz", str(OUT_DIR / "representation_final_select_waveforms.npz"),
        "--output-dir", str(CHECKPOINTS_DIR),
        "--epochs", "25",
        "--device", "cuda",
    ]
    logger.info("Executing command: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    logger.info("Stage 2: Five-seed training completed successfully!")


def run_stage3_manifest_freeze() -> None:
    """Run freeze_submission_manifest.py to freeze all SHA-256 hashes."""
    logger.info("=== Stage 3: Freezing Submission Manifest ===")
    cmd = [
        str(PYTHON_BIN),
        str(SCRIPTS_DIR / "freeze_submission_manifest.py"),
        "--scaler-path", str(OUT_DIR / "final_scaler.json"),
        "--response-fit-path", str(OUT_DIR / "response_fit_final.npz"),
        "--checkpoints-dir", str(CHECKPOINTS_DIR),
        "--grassmann-bank-path", str(OUT_DIR / "expanded_grassmann_subspace_bank.npz"),
        "--output-path", str(MANIFEST_PATH),
    ]
    logger.info("Executing command: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    logger.info("Stage 3: Submission manifest frozen at %s", MANIFEST_PATH)


def run_stage4_fold10_evaluation() -> None:
    """Run evaluate_final_fold10_locked.py to evaluate on Fold 10."""
    logger.info("=== Stage 4: Unsealing Fold 10 for Confirmatory Evaluation ===")
    FOLD10_OUT_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(PYTHON_BIN),
        str(SCRIPTS_DIR / "evaluate_final_fold10_locked.py"),
        "--manifest-path", str(MANIFEST_PATH),
        "--checkpoints-dir", str(CHECKPOINTS_DIR),
        "--grassmann-bank-path", str(OUT_DIR / "expanded_grassmann_subspace_bank.npz"),
        "--output-dir", str(FOLD10_OUT_DIR),
        "--auth-token", "CVPR2027_CONFIRMATORY_UNSEAL_AUTHORIZED",
        "--device", "cuda",
    ]
    logger.info("Executing command: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    logger.info("Stage 4: Fold 10 confirmatory evaluation completed!")


def run_stage5_table_generation() -> None:
    """Run generate_cvpr_definitive_tables.py to compile publication tables."""
    logger.info("=== Stage 5: Generating Definitive CVPR Benchmark Tables ===")
    TABLES_OUT_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(PYTHON_BIN),
        str(SCRIPTS_DIR / "generate_cvpr_definitive_tables.py"),
        "--manifest-path", str(MANIFEST_PATH),
        "--fold10-results-dir", str(FOLD10_OUT_DIR),
        "--output-dir", str(TABLES_OUT_DIR),
    ]
    logger.info("Executing command: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    logger.info("Stage 5: Definitive CVPR benchmark tables generated at %s", TABLES_OUT_DIR)


def main():
    logger.info("****************************************************************")
    logger.info("Starting Overnight Definitive Promotion Chain for SetOperator CVPR 2027")
    logger.info("****************************************************************")

    # 1. Wait for Stage 1 dataset materialization
    wait_for_stage1_datasets(poll_interval=120)

    # 2. Run 5-seed training
    run_stage2_training()

    # 3. Freeze submission manifest
    run_stage3_manifest_freeze()

    # 4. Run locked Fold 10 evaluation
    run_stage4_fold10_evaluation()

    # 5. Generate definitive benchmark tables
    run_stage5_table_generation()

    logger.info("****************************************************************")
    logger.info("OVERNIGHT DEFINITIVE PROMOTION CHAIN COMPLETED SUCCESSFULLY!")
    logger.info("****************************************************************")


if __name__ == "__main__":
    main()
