#!/usr/bin/env python3
"""Master Orchestrator for the Authoritative PTB-XL Three-Way Model Comparison.

Rigorous comparison across:
  1. SetOperator (Frozen continuous operator learning architecture)
  2. SetONet (Branch-trunk operator network adapted from st-ep/SetONet)
  3. GraphECG (Electrode-centric GNN from Ansari et al., Stanford 2026)

Locked Experimental Protocol:
- Task: 5-class multilabel ECG diagnosis (NORM, MI, STTC, CD, HYP)
- Cohort: PTB-XL Folds 1-8 Train (17,417 records), Fold 9 Validation (2,183 records)
- Confirmation: Fold 10 strictly SEALED during development
- Loss: Class-weighted BCEWithLogitsLoss
- Optimizer: AdamW, lr=3e-4, wd=1e-3, gradient norm clipping 1.0
- Scheduler: CosineAnnealingLR
- Epochs: 80 max, early stopping patience 10 on Fold 9 macro AUROC
- Seeds: 42, 43, 44, 45, 46 (matched paired initializations where applicable)
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List
import zipfile

import numpy as np
import torch

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("three_way_orchestrator")

torch.backends.cudnn.enabled = False
torch.set_float32_matmul_precision("high")

REPO_DIR = Path(__file__).resolve().parents[4]
EXPERIMENT_DIR = REPO_DIR / "experiments/ptbxl_distributional_repecg"
SCRIPTS_DIR = EXPERIMENT_DIR / "scripts/paper07"
PYTHON_BIN = Path(sys.executable)

CODEX_DIR = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg")
FINAL_REPS_DIR = CODEX_DIR / "paper07_operator/final_representations"
ARTIFACTS_DIR = CODEX_DIR / "ptbxl_three_way_v1"

# Destination directories
CHECKPOINTS_SO = ARTIFACTS_DIR / "checkpoints/setoperator"
CHECKPOINTS_SN = ARTIFACTS_DIR / "checkpoints/setonet"
CHECKPOINTS_GE = ARTIFACTS_DIR / "checkpoints/graphecg"
PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"
RESULTS_DIR = ARTIFACTS_DIR / "results"
LOGS_DIR = ARTIFACTS_DIR / "logs"

EXPECTED_MODEL_SHA256 = "66dae0d122266271e4b5b080e3e7048e4ed4ad3bea7d1f96e7bc8fd99b85227e"


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def get_npz_array_header(npz_path: Path, array_name: str) -> tuple[tuple[int, ...], np.dtype]:
    """Inspect array shape and dtype directly from zip member header without decompressing data."""
    with zipfile.ZipFile(npz_path, "r") as z:
        with z.open(f"{array_name}.npy") as f:
            version = np.lib.format.read_magic(f)
            shape, fortran, dtype = (
                np.lib.format.read_array_header_1_0(f)
                if version == (1, 0)
                else np.lib.format.read_array_header_2_0(f)
            )
            return shape, dtype


# ==============================================================================
# STAGE 0: Population & Representation Verification Gate
# ==============================================================================

def run_stage0_verification() -> Dict[str, Any]:
    logger.info("==================================================================")
    logger.info("STAGE 0: Verifying Input Datasets, Cohort Alignment, & Fold Isolation")
    logger.info("==================================================================")

    assert torch.cuda.is_available(), "CUDA GPU required!"
    gpu_name = torch.cuda.get_device_name(0)
    logger.info("Verified GPU: %s", gpu_name)

    basis_path = FINAL_REPS_DIR / "representation_final_train_basis_only.npz"
    span_path = FINAL_REPS_DIR / "representation_final_train_operator_span.npz"
    val_path = FINAL_REPS_DIR / "representation_final_select_waveforms.npz"
    grassmann_path = FINAL_REPS_DIR / "expanded_grassmann_subspace_bank.npz"

    for p in (basis_path, span_path, val_path, grassmann_path):
        assert p.is_file() and p.stat().st_size > 1024, f"Missing required file: {p}"

    # Fast header inspection
    basis_shape, _ = get_npz_array_header(basis_path, "responses")
    span_shape, _ = get_npz_array_header(span_path, "responses")
    val_shape, _ = get_npz_array_header(val_path, "responses")

    assert basis_shape == (17417, 8, 16, 128), f"Basis shape mismatch: {basis_shape}"
    assert span_shape == (17417, 16, 16, 128), f"Span shape mismatch: {span_shape}"
    assert val_shape == (2183, 16, 16, 128), f"Val shape mismatch: {val_shape}"
    logger.info("Verified array dimensions: Basis %s | Span %s | Val %s", basis_shape, span_shape, val_shape)

    # Verify zero overlap from ID arrays
    with zipfile.ZipFile(basis_path) as zb, zipfile.ZipFile(val_path) as zv:
        train_ids = np.load(zb.open("ecg_ids.npy"))
        val_ids = np.load(zv.open("ecg_ids.npy"))
        overlap = set(train_ids).intersection(set(val_ids))
        assert len(overlap) == 0, f"LEAKAGE DETECTED: {len(overlap)} records overlap!"
        logger.info("Verified ZERO record overlap between Folds 1-8 (%d) and Fold 9 (%d).", len(train_ids), len(val_ids))

    # Verify Grassmann 350-subspace bank
    g_bank = np.load(grassmann_path)
    total_subs = 0
    for r in range(1, 8):
        k = f"rank_{r}"
        assert k in g_bank, f"Missing {k} in Grassmann bank"
        subs = g_bank[k]
        assert subs.shape == (50, r, 8), f"Unexpected shape for {k}: {subs.shape}"
        total_subs += len(subs)
    assert total_subs == 350, f"Expected 350 subspaces, found {total_subs}"
    logger.info("Verified all 350 Grassmann subspaces (ranks 1-7, 50 per rank).")

    # Confirm Fold 10 lockbox is sealed
    lockbox = CODEX_DIR / "paper07_operator/final_fold10_evaluation"
    assert not (lockbox / "fold10_access.lock").exists(), "Fold 10 lockbox is not sealed!"
    logger.info("Confirmed Fold 10 remains strictly sealed.")

    audit_payload = {
        "status": "STAGE0_VERIFIED",
        "gpu": gpu_name,
        "n_train": 17417,
        "n_val": 2183,
        "total_grassmann_subspaces": 350,
        "fold10_sealed": True,
        "timestamp": time.time(),
    }
    (ARTIFACTS_DIR / "stage0_audit_manifest.json").write_text(json.dumps(audit_payload, indent=2))
    return audit_payload


# ==============================================================================
# STAGE 1: Real GPU Smoke Test
# ==============================================================================

def run_stage1_smoke_test() -> None:
    logger.info("==================================================================")
    logger.info("STAGE 1: Executing Pre-Flight Smoke Test")
    logger.info("==================================================================")
    cmd = [str(PYTHON_BIN), str(SCRIPTS_DIR / "test_setoperator_smoke.py")]
    t0 = time.time()
    res = subprocess.run(cmd, check=True)
    logger.info("Stage 1 Smoke Test passed in %.2f seconds!", time.time() - t0)


# ==============================================================================
# STAGE 2: SetOperator Five-Seed Training (Basis & Span)
# ==============================================================================

def run_stage2_setoperator_training(
    seeds: List[int],
    epochs: int = 80,
    patience: int = 10,
    lr: float = 3e-4,
    wd: float = 1e-3,
    device: str = "cuda",
) -> Dict[str, Any]:
    logger.info("==================================================================")
    logger.info("STAGE 2: Training SetOperator (Basis & Span) across seeds %s", seeds)
    logger.info("==================================================================")
    CHECKPOINTS_SO.mkdir(parents=True, exist_ok=True)

    # Check which seeds are already trained
    ledger_path = CHECKPOINTS_SO / "checkpoints_selection_ledger.json"
    existing_ledger = {}
    if ledger_path.exists():
        try:
            existing_ledger = json.loads(ledger_path.read_text())
        except Exception:
            existing_ledger = {}

    missing_seeds = []
    for s in seeds:
        for mode in ("basis", "span"):
            key = f"{mode}_s{s}"
            chk = CHECKPOINTS_SO / f"{mode}_s{s}.pt"
            if key not in existing_ledger or not chk.is_file():
                missing_seeds.append(s)
                break

    if not missing_seeds:
        logger.info("All SetOperator seeds %s already completed! Skipping training.", seeds)
        return existing_ledger

    logger.info("Training required for seeds: %s", missing_seeds)
    cmd = [
        str(PYTHON_BIN),
        "-u",
        str(SCRIPTS_DIR / "train_final_5seeds.py"),
        "--mode", "both",
        "--seeds", *[str(s) for s in missing_seeds],
        "--train-basis-npz", str(FINAL_REPS_DIR / "representation_final_train_basis_only.npz"),
        "--train-span-npz", str(FINAL_REPS_DIR / "representation_final_train_operator_span.npz"),
        "--val-npz", str(FINAL_REPS_DIR / "representation_final_select_waveforms.npz"),
        "--output-dir", str(CHECKPOINTS_SO),
        "--epochs", str(epochs),
        "--patience", str(patience),
        "--lr", str(lr),
        "--wd", str(wd),
        "--device", device,
    ]

    t0 = time.time()
    subprocess.run(cmd, check=True)
    logger.info("Stage 2 SetOperator Training completed in %.1f minutes!", (time.time() - t0) / 60.0)

    # Load and return final ledger
    final_ledger = json.loads(ledger_path.read_text())
    return final_ledger


# ==============================================================================
# STAGE 3: SetONet Five-Seed Training (Basis & Span)
# ==============================================================================

def run_stage3_setonet_training(
    seeds: List[int],
    epochs: int = 80,
    patience: int = 10,
    lr: float = 3e-4,
    wd: float = 1e-3,
    device: str = "cuda",
) -> Dict[str, Any]:
    logger.info("==================================================================")
    logger.info("STAGE 3: Training SetONet (Basis & Span) across seeds %s", seeds)
    logger.info("==================================================================")
    CHECKPOINTS_SN.mkdir(parents=True, exist_ok=True)

    ledger_path = CHECKPOINTS_SN / "checkpoints_selection_ledger.json"
    existing_ledger = {}
    if ledger_path.exists():
        try:
            existing_ledger = json.loads(ledger_path.read_text())
        except Exception:
            existing_ledger = {}

    missing_seeds = [s for s in seeds if f"span_s{s}" not in existing_ledger]
    if not missing_seeds:
        logger.info("All SetONet seeds %s already completed! Skipping.", seeds)
        return existing_ledger

    cmd = [
        str(PYTHON_BIN),
        "-u",
        str(SCRIPTS_DIR / "train_setonet_ptbxl.py"),
        "--mode", "both",
        "--seeds", *[str(s) for s in missing_seeds],
        "--train-basis-npz", str(FINAL_REPS_DIR / "representation_final_train_basis_only.npz"),
        "--train-span-npz", str(FINAL_REPS_DIR / "representation_final_train_operator_span.npz"),
        "--val-npz", str(FINAL_REPS_DIR / "representation_final_select_waveforms.npz"),
        "--output-dir", str(CHECKPOINTS_SN),
        "--epochs", str(epochs),
        "--patience", str(patience),
        "--lr", str(lr),
        "--wd", str(wd),
        "--device", device,
    ]
    t0 = time.time()
    subprocess.run(cmd, check=True)
    logger.info("Stage 3 SetONet Training completed in %.1f minutes!", (time.time() - t0) / 60.0)
    return json.loads(ledger_path.read_text())


# ==============================================================================
# STAGE 4: GraphECG Split-Corrected Five-Seed Training
# ==============================================================================

def run_stage4_graphecg_training(
    seeds: List[int],
    epochs: int = 50,
    patience: int = 10,
    lr: float = 3e-4,
    wd: float = 1e-4,
    device: str = "cuda",
) -> Dict[str, Any]:
    logger.info("==================================================================")
    logger.info("STAGE 4: Training GraphECG across seeds %s", seeds)
    logger.info("==================================================================")
    CHECKPOINTS_GE.mkdir(parents=True, exist_ok=True)

    ledger_path = CHECKPOINTS_GE / "checkpoints_selection_ledger.json"
    existing_ledger = {}
    if ledger_path.exists():
        try:
            existing_ledger = json.loads(ledger_path.read_text())
        except Exception:
            existing_ledger = {}

    missing_seeds = [s for s in seeds if f"ge12_s{s}" not in existing_ledger]
    if not missing_seeds:
        logger.info("All GraphECG seeds %s already completed! Skipping.", seeds)
        return existing_ledger

    cmd = [
        str(PYTHON_BIN),
        "-u",
        str(REPO_DIR / "experiments/ptbxl_distributional_repecg/scripts/graphecg/train_graphecg_ptbxl_three_way.py"),
        "--seeds", *[str(s) for s in missing_seeds],
        "--output-dir", str(CHECKPOINTS_GE),
        "--epochs", str(epochs),
        "--patience", str(patience),
        "--lr", str(lr),
        "--wd", str(wd),
        "--device", device,
    ]
    t0 = time.time()
    subprocess.run(cmd, check=True)
    logger.info("Stage 4 GraphECG Training completed in %.1f minutes!", (time.time() - t0) / 60.0)
    return json.loads(ledger_path.read_text())


# ==============================================================================
# STAGE 5: Comprehensive Three-Way Evaluation Suite
# ==============================================================================

def run_stage5_evaluations(device: str = "cuda") -> None:
    logger.info("==================================================================")
    logger.info("STAGE 5: Running Comprehensive Three-Way Evaluation Suite on Fold 9")
    logger.info("==================================================================")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(PYTHON_BIN),
        "-u",
        str(SCRIPTS_DIR / "evaluate_ptbxl_three_way.py"),
        "--output-dir", str(RESULTS_DIR),
        "--device", device,
    ]
    t0 = time.time()
    subprocess.run(cmd, check=True)
    logger.info("Stage 5 Evaluation Suite completed in %.1f minutes!", (time.time() - t0) / 60.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Master PTB-XL Three-Way Experiment Launcher.")
    parser.add_argument("--stage", choices=["all", "audit", "smoke", "setoperator", "setonet", "graphecg", "eval"], default="all")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    logger.info("******************************************************************")
    logger.info("STARTING PTB-XL THREE-WAY REPRODUCIBLE EXPERIMENTAL PROGRAM")
    logger.info("******************************************************************")

    # Stage 0: Verification
    if args.stage in ("all", "audit"):
        run_stage0_verification()

    # Stage 1: Smoke Test
    if args.stage in ("all", "smoke"):
        run_stage1_smoke_test()

    # Stage 2: SetOperator Training
    if args.stage in ("all", "setoperator"):
        run_stage2_setoperator_training(
            seeds=args.seeds,
            epochs=args.epochs,
            patience=args.patience,
            device=args.device,
        )

    # Stage 3: SetONet Training
    if args.stage in ("all", "setonet"):
        run_stage3_setonet_training(
            seeds=args.seeds,
            epochs=args.epochs,
            patience=args.patience,
            device=args.device,
        )

    # Stage 4: GraphECG Training
    if args.stage in ("all", "graphecg"):
        run_stage4_graphecg_training(
            seeds=args.seeds,
            epochs=min(args.epochs, 50),
            patience=args.patience,
            device=args.device,
        )

    # Stage 5: Evaluation Suite
    if args.stage in ("all", "eval"):
        run_stage5_evaluations(device=args.device)

    logger.info("Pipeline execution complete in %.1f minutes.", (time.time() - t_start) / 60.0)


if __name__ == "__main__":
    main()
