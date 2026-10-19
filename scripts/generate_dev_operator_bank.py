#!/usr/bin/env python3
"""Generates a separate development operator bank and precomputes authentic Phase-KME responses.

Scientific Principle:
1. Strictly distinct random seed (2026) from the frozen 350-subspace Grassmann reporting bank (seed 42).
2. Contains 3 random orthonormal subspaces per rank r in 1..7 (total 21 subspaces).
3. Precomputes authentic Phase-KME responses on GPU using FastResponseComputer and fold9_authentic_beats.npz.
4. Allows sub-second development evaluation during G1 checkpoint adaptation without touching the reporting bank.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import sys
import time
from typing import Dict, Any, List

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_dev_bank")

DEV_ROOT = Path("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction_geometry_dev")
REPO_ROOT = Path("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction")
SRC_DIR = REPO_ROOT / "experiments/ptbxl_distributional_repecg/scripts/paper07"
sys.path.insert(0, str(SRC_DIR))

from evaluator_physical_operator import FastResponseComputer

PTBXL_DIR = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg")
FINAL_REPS_DIR = PTBXL_DIR / "paper07_operator/final_representations"
BEATS_PATH = FINAL_REPS_DIR / "fold9_authentic_beats.npz"
FIT_PATH = FINAL_REPS_DIR / "response_fit_final.npz"
GRASSMANN_REPORTING_PATH = FINAL_REPS_DIR / "expanded_grassmann_subspace_bank.npz"

OUT_DIR = Path("/data/mithunmanivannan/codex_artifacts/echonext_geometry_setoperator_v1/runs/g1_checkpoint_adapted_s42/dev_bank")


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Initializing Dev Bank Generator on %s...", device)

    # 1. Generate Dev Bank with seed 2026
    seed = 2026
    n_per_rank = 3
    rng = np.random.default_rng(seed)

    bank_subspaces: Dict[str, np.ndarray] = {}
    flat_subspaces: List[np.ndarray] = []
    flat_ranks: List[int] = []

    for r in range(1, 8):
        rank_key = f"rank_{r}"
        sub_list = []
        for i in range(n_per_rank):
            G = rng.normal(size=(8, r))
            Q, _ = np.linalg.qr(G)
            sub = Q.T.astype(np.float64)  # (r, 8)
            # Verify orthonormality
            err = np.max(np.abs(sub @ sub.T - np.eye(r)))
            assert err < 1e-6, f"Orthonormality violation in dev bank: {err}"
            sub_list.append(sub)
            flat_subspaces.append(sub)
            flat_ranks.append(r)
        bank_subspaces[rank_key] = np.stack(sub_list, axis=0)

    bank_file = OUT_DIR / "ptbxl_dev_operator_bank.npz"
    np.savez_compressed(bank_file, **bank_subspaces)
    logger.info("Saved %d development subspaces to %s", len(flat_subspaces), bank_file)

    # 2. Verify distinction from Reporting Bank (seed 42)
    rep_bank = np.load(GRASSMANN_REPORTING_PATH)
    logger.info("Verifying orthogonality / distinctness from reporting bank at %s...", GRASSMANN_REPORTING_PATH)
    min_dist = float("inf")
    for r in range(1, 8):
        dev_subs = bank_subspaces[f"rank_{r}"]
        rep_subs = rep_bank[f"rank_{r}"]
        for d in dev_subs:
            for rep in rep_subs:
                # Projection distance: ||d d^T - rep rep^T||_F
                dist = np.linalg.norm(d.T @ d - rep.T @ rep, "fro")
                if dist < min_dist:
                    min_dist = dist
    logger.info("Minimum projection distance between dev bank and reporting bank: %.4f (strictly > 0)", min_dist)
    assert min_dist > 0.05, f"Dev bank too close to reporting bank: {min_dist}"

    # 3. Precompute responses on authentic patient beats
    logger.info("Loading authentic Fold-9 beats from %s...", BEATS_PATH)
    d_beats = np.load(BEATS_PATH, allow_pickle=True)
    beats_list = list(d_beats["beats"])
    ecg_ids = d_beats["ecg_ids"]
    patient_ids = d_beats["patient_ids"]
    n_records = len(beats_list)
    logger.info("Loaded %d patient records.", n_records)

    computer = FastResponseComputer(FIT_PATH, device)

    # Precompute responses for each subspace
    logger.info("Precomputing authentic Phase-KME responses for %d subspaces...", len(flat_subspaces))
    t0 = time.time()
    responses_dict: Dict[str, np.ndarray] = {}

    for sub_idx, (sub, r) in enumerate(zip(flat_subspaces, flat_ranks)):
        t_sub = time.time()
        # Shape for this subspace: (n_records, r, 16, 128)
        sub_res = np.empty((n_records, r, 16, 128), dtype=np.float32)
        for rec_idx, beats in enumerate(beats_list):
            sub_res[rec_idx] = computer.compute_record_responses(beats, sub)
        responses_dict[f"sub_{sub_idx}"] = sub_res
        logger.info("Subspace %02d/%02d (Rank %d) computed in %.2fs",
                    sub_idx + 1, len(flat_subspaces), r, time.time() - t_sub)

    total_time = time.time() - t0
    logger.info("All %d dev subspace responses computed in %.2fs (%.2fs per subspace)",
                len(flat_subspaces), total_time, total_time / len(flat_subspaces))

    resp_file = OUT_DIR / "ptbxl_dev_responses.npz"
    np.savez_compressed(
        resp_file,
        ecg_ids=ecg_ids,
        patient_ids=patient_ids,
        ranks=np.array(flat_ranks, dtype=np.int32),
        **responses_dict
    )
    logger.info("Saved development responses to %s", resp_file)

    manifest = {
        "dataset": "ptbxl_dev_operator_bank",
        "seed": seed,
        "n_subspaces": len(flat_subspaces),
        "ranks": list(range(1, 8)),
        "n_per_rank": n_per_rank,
        "bank_file": str(bank_file),
        "bank_sha256": compute_sha256(bank_file),
        "responses_file": str(resp_file),
        "responses_sha256": compute_sha256(resp_file),
        "min_dist_to_reporting_bank": min_dist,
        "n_records": n_records,
        "total_compute_seconds": total_time,
        "timestamp": time.time(),
    }
    with open(OUT_DIR / "dev_bank_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Saved manifest to %s", OUT_DIR / "dev_bank_manifest.json")
    logger.info("DEV BANK GENERATION COMPLETE!")


if __name__ == "__main__":
    main()
