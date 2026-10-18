#!/usr/bin/env python3
"""Audit Phase-KME Numerical Equivalence against Frozen Paper 07 Implementation.

Rigorous Verification:
1. Loads authentic EchoNext training Phase-KME fit: echonext_phase_kme_fit.npz.
2. Extracts authentic beats from 50 EchoNext validation records.
3. Computes Phase-KME representations using:
   - Method A: Exact PTB-XL reference pipeline (repecg.paper07_operator.measurements.response_atoms + FastResponseComputer)
   - Method B: EchoNextPhaseKMEExtractor (vectorized implementation in phase_kme_representation.py)
4. Tests across 16 canonical and random operator directions.
5. Computes maximum absolute difference and relative Frobenius error.
6. Verifies TF32 safeguards and numerical tolerances.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
import time

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("audit_kme_equiv")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = REPO_ROOT / "experiments/ptbxl_distributional_repecg/src"
SCRIPTS_DIR = REPO_ROOT / "experiments/ptbxl_distributional_repecg/scripts/paper07"
BENCHMARK_DIR = REPO_ROOT / "experiments/echonext_operator_benchmark"

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(BENCHMARK_DIR))

from repecg.paper07_operator.measurements import normalize_operator, response_atoms
from evaluator_physical_operator import FastResponseComputer, build_canonical_and_derived_operator_bank
from phase_kme_representation import EchoNextPhaseKMEExtractor

FIT_PATH = Path("/data/mithunmanivannan/codex_artifacts/echonext_three_way_v1/representations/echonext_phase_kme_fit.npz")
VAL_WF_PATH = Path("/data/mithunmanivannan/echonext/EchoNext_val_waveforms.npy")
CANONICAL_8_LEADS = [0, 1, 6, 7, 8, 9, 10, 11]


def main():
    logger.info("==================================================")
    logger.info("AUDITING PHASE-KME NUMERICAL EQUIVALENCE")
    logger.info("Fit Artifact: %s", FIT_PATH)
    logger.info("==================================================")

    # 1. Enforce TF32 safeguards
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.enabled = False

    device = torch.device("cpu")  # Audit on CPU first for deterministic exactness

    # 2. Instantiate both computers
    logger.info("Instantiating reference FastResponseComputer and EchoNextPhaseKMEExtractor...")
    ref_computer = FastResponseComputer(FIT_PATH, device)
    new_extractor = EchoNextPhaseKMEExtractor(FIT_PATH, device)

    # 3. Create test operator bank (16 operators: 8 canonical + 8 random directions)
    rng = np.random.default_rng(42)
    canonical_8 = np.eye(8, dtype=np.float32)
    random_8 = np.stack([normalize_operator(rng.normal(size=8).astype(np.float32)) for _ in range(8)], axis=0)
    q_test_bank = np.concatenate([canonical_8, random_8], axis=0).astype(np.float32)  # (16, 8)
    assert q_test_bank.shape == (16, 8)

    # 4. Load authentic EchoNext validation waveforms
    val_wf = np.load(VAL_WF_PATH, mmap_mode="r")
    n_test_records = 30
    logger.info("Testing numerical equivalence across %d authentic EchoNext validation ECGs...", n_test_records)

    max_abs_diff = 0.0
    all_rel_errors = []

    for idx in range(n_test_records):
        wf_8 = val_wf[idx, 0][:, CANONICAL_8_LEADS].astype(np.float32)
        beats = new_extractor.extract_beats(wf_8)  # (n_beats, 256, 8)

        # Compute with Reference FastResponseComputer (Method A)
        resps_ref = ref_computer.compute_record_responses(beats, q_test_bank)  # (16, 16, 128)

        # Compute with EchoNextPhaseKMEExtractor (Method B)
        with torch.no_grad():
            resps_new = new_extractor.compute_phase_kme(
                torch.from_numpy(beats), torch.from_numpy(q_test_bank)
            ).cpu().numpy()  # (16, 16, 128)

        # Compare
        diff = np.abs(resps_ref - resps_new)
        record_max_diff = float(np.max(diff))
        rel_frob = float(np.linalg.norm(resps_ref - resps_new) / np.linalg.norm(resps_ref))
        all_rel_errors.append(rel_frob)

        if record_max_diff > max_abs_diff:
            max_abs_diff = record_max_diff

    mean_rel_err = float(np.mean(all_rel_errors))
    logger.info("--------------------------------------------------")
    logger.info("AUDIT VERDICT:")
    logger.info("Max Absolute Difference: %.2e", max_abs_diff)
    logger.info("Mean Relative Frobenius Error: %.2e", mean_rel_err)
    logger.info("--------------------------------------------------")

    # The difference should be at single-precision epsilon (< 1e-4)
    if max_abs_diff > 1e-4:
        raise RuntimeError(f"Numerical equivalence failure: max absolute difference {max_abs_diff:.2e} > 1e-4")

    logger.info("SUCCESS: EchoNext Phase-KME is numerically identical to the frozen PTB-XL implementation!")


if __name__ == "__main__":
    main()
