#!/usr/bin/env python3
"""Milestone R024: Build Datasets for Continuous Operator-Support Training.

Materializes:
1. representation_train_basis_only.npz:
   Candidate operators: strictly the 8 canonical basis leads {e_1, ..., e_8}
   Shape: operators [15244, 8, 8], responses [15244, 8, 16, 128]
2. representation_train_operator_span.npz:
   Candidate operators: 8 canonical basis leads + 8 continuous normalized linear combinations
   sampled from S^7 with held-out exclusion filter.
   Shape: operators [15244, 16, 8], responses [15244, 16, 16, 128]
3. representation_val_real_waveforms.npz:
   Validation set with real physical waveforms for all evaluation configurations:
   Q8, S6_limb, derived_limb, S_icm (V3 - V2), oblique_1 (V1 - V6), oblique_2 (I + V4),
   oblique_3 (II - V5), S6_precordial, S3_icu_v1, S3_icu_v5, S2_bipolar, S1_smartwatch, S1_rhythm.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

from repecg.paper07_operator import normalize_operator, response_atoms

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_r024_datasets")

CODEX_DIR = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg")
TRAIN_CACHE = CODEX_DIR / "paper02_kernel_mean/development_train"
VAL_CACHE = CODEX_DIR / "paper02_kernel_mean/development_select"
RESPONSE_FIT = CODEX_DIR / "paper07_operator/development_representations/response_fit.npz"
OUT_DIR = CODEX_DIR / "paper07_operator/r024_operator_span"


def get_heldout_test_operators() -> np.ndarray:
    """Define operators that MUST NEVER be seen during training."""
    # Basis definitions in 8-lead coordinate: [I, II, V1, V2, V3, V4, V5, V6]
    e = np.eye(8, dtype=np.float64)
    # Einthoven / Goldberger
    q_III = normalize_operator(e[1] - e[0])
    q_aVR = normalize_operator(-e[0] - e[1])
    q_aVL = normalize_operator(e[0] - 0.5 * e[1])
    q_aVF = normalize_operator(e[1] - 0.5 * e[0])
    # Oblique subcutaneous / precordial subtraction
    q_V3_V2 = normalize_operator(e[4] - e[3])
    # Additional test oblique vectors
    q_V1_V6 = normalize_operator(e[2] - e[7])
    q_I_V4 = normalize_operator(e[0] + e[5])
    q_II_V5 = normalize_operator(e[1] - e[6])

    heldout = np.stack([q_III, q_aVR, q_aVL, q_aVF, q_v3_v2 if (q_v3_v2 := q_V3_V2) is not None else None, q_V1_V6, q_I_V4, q_II_V5])
    return heldout


def is_safe_operator(q: np.ndarray, heldout: np.ndarray, min_dist: float = 0.08) -> bool:
    """Ensure candidate q does not overlap with any heldout test operator up to sign."""
    q_norm = normalize_operator(q)
    for h in heldout:
        if np.linalg.norm(q_norm - h) < min_dist or np.linalg.norm(q_norm + h) < min_dist:
            return False
    return True


def sample_training_operators(rng: np.random.Generator, heldout: np.ndarray, n_span: int = 8) -> np.ndarray:
    """Sample continuous normalized linear combinations from S^7 excluding held-out vectors."""
    span_ops = []
    attempts = 0
    while len(span_ops) < n_span:
        attempts += 1
        if attempts > 2000:
            raise RuntimeError("Exceeded max attempts to sample safe operators")
        mode = len(span_ops) % 2
        if mode == 0:
            # Dense random linear combination
            alpha = rng.normal(size=8)
        else:
            # Sparse random combination (2 or 3 active leads)
            k = rng.choice([2, 3])
            active = rng.choice(8, size=k, replace=False)
            alpha = np.zeros(8)
            alpha[active] = rng.normal(size=k)
        if np.linalg.norm(alpha) < 1e-6:
            continue
        q = normalize_operator(alpha)
        if is_safe_operator(q, heldout):
            span_ops.append(q)
    return np.stack(span_ops, axis=0)


class FastResponseComputer:
    def __init__(self, fit_path: Path, device: torch.device):
        data = np.load(fit_path)
        self.v_scale = float(data["voltage_scale"])
        self.mean = data["physical_mean"].astype(np.float32)
        self.std = data["physical_std"].astype(np.float32)

        self.landmarks = torch.as_tensor(data["landmarks"], device=device, dtype=torch.float32)
        self.inv_root = torch.as_tensor(data["inverse_root"], device=device, dtype=torch.float32)
        self.c2 = float(data["c2"])
        self.w_mean = torch.as_tensor(data["whitening_mean"], device=device, dtype=torch.float32)
        self.w_comp = torch.as_tensor(data["whitening_components"], device=device, dtype=torch.float32)
        self.w_scale = torch.as_tensor(data["whitening_scales"], device=device, dtype=torch.float32)
        self.device = device

    def compute_record_responses(self, beats_mv: np.ndarray, q_bank: np.ndarray) -> np.ndarray:
        """Compute responses for one record across K operators."""
        # beats_mv is (beat, 256, 8)
        n_ops = len(q_bank)
        atoms = [response_atoms(beats_mv.astype(np.float64), q, self.v_scale) for q in q_bank]
        atoms = np.stack(atoms, axis=0)  # (K, beat, 256, 2)
        cells = atoms.reshape(n_ops, len(beats_mv), 16, 16, 2).transpose(0, 2, 1, 3, 4).reshape(n_ops, 16, -1, 2)
        t_cells = torch.as_tensor(cells, device=self.device, dtype=torch.float32)
        white = (t_cells - self.w_mean) @ self.w_comp * self.w_scale
        d2 = torch.cdist(white, self.landmarks.expand(n_ops, 16, -1, -1)).square() + self.c2
        feat = torch.rsqrt(d2) @ self.inv_root
        return feat.mean(dim=2).cpu().numpy().astype(np.float32)  # (K, 16, 128)


def build_train_splits(device: torch.device):
    logger.info("=== Building Milestone R024 Training Splits ===")
    heldout = get_heldout_test_operators()
    qc_frame = pd.read_csv(TRAIN_CACHE / "qc.csv").query("eligible").reset_index(drop=True)
    n_records = len(qc_frame)
    logger.info(f"Loaded {n_records} eligible training records from {TRAIN_CACHE}.")

    computer = FastResponseComputer(RESPONSE_FIT, device)

    # Pre-allocate output arrays
    # 1. Basis-Only: 8 basis leads
    basis_ops_all = np.empty((n_records, 8, 8), dtype=np.float32)
    basis_resps_all = np.empty((n_records, 8, 16, 128), dtype=np.float32)

    # 2. Operator-Span: 8 basis leads + 8 continuous span leads = 16 operators
    span_ops_all = np.empty((n_records, 16, 8), dtype=np.float32)
    span_resps_all = np.empty((n_records, 16, 16, 128), dtype=np.float32)

    labels_all = np.empty((n_records, 5), dtype=np.float32)
    ecg_ids_all = qc_frame.ecg_id.to_numpy(dtype=np.int64)
    patient_ids_all = qc_frame.patient_id.to_numpy(dtype=np.int64)

    basis_8 = np.eye(8, dtype=np.float32)

    chunk_size = 500
    t0_start = time.time()

    for chunk_start in range(0, n_records, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_records)
        chunk_files = [TRAIN_CACHE / qc_frame.iloc[i].artifact for i in range(chunk_start, chunk_end)]

        def _load_beat(p):
            with np.load(p) as item:
                b = np.asarray(item["beats"], dtype=np.float32) * computer.std + computer.mean
                return b, np.asarray(item["labels"], dtype=np.float32)

        with ThreadPoolExecutor(max_workers=16) as ex:
            chunk_data = list(ex.map(_load_beat, chunk_files))

        for idx_in_chunk, (beats, lbl) in enumerate(chunk_data):
            rec_idx = chunk_start + idx_in_chunk
            labels_all[rec_idx] = lbl

            # Seed per record
            rng = np.random.default_rng(np.random.SeedSequence([42, int(ecg_ids_all[rec_idx])]))

            # 8 continuous span operators
            span_8 = sample_training_operators(rng, heldout, n_span=8).astype(np.float32)
            bank_16 = np.concatenate([basis_8, span_8], axis=0)

            # Compute responses
            resps_16 = computer.compute_record_responses(beats, bank_16)

            # Store Basis-Only
            basis_ops_all[rec_idx] = basis_8
            basis_resps_all[rec_idx] = resps_16[:8]

            # Store Operator-Span
            span_ops_all[rec_idx] = bank_16
            span_resps_all[rec_idx] = resps_16

        elapsed = time.time() - t0_start
        rate = chunk_end / elapsed
        logger.info(f"Processed {chunk_end}/{n_records} training records ({rate:.1f} rec/s, elapsed: {elapsed:.1f}s)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save Basis-Only
    p_basis = OUT_DIR / "representation_train_basis_only.npz"
    logger.info(f"Saving basis-only dataset to {p_basis}...")
    np.savez_compressed(
        p_basis,
        operators=basis_ops_all,
        responses=basis_resps_all,
        labels=labels_all,
        ecg_ids=ecg_ids_all,
        patient_ids=patient_ids_all,
    )

    # Save Operator-Span
    p_span = OUT_DIR / "representation_train_operator_span.npz"
    logger.info(f"Saving operator-span dataset to {p_span}...")
    np.savez_compressed(
        p_span,
        operators=span_ops_all,
        responses=span_resps_all,
        labels=labels_all,
        ecg_ids=ecg_ids_all,
        patient_ids=patient_ids_all,
    )


def build_validation_split(device: torch.device):
    logger.info("=== Building Milestone R024 Validation Split on Real Physical Waveforms ===")
    qc_frame = pd.read_csv(VAL_CACHE / "qc.csv").query("eligible").reset_index(drop=True)
    n_records = len(qc_frame)
    logger.info(f"Loaded {n_records} eligible validation records from {VAL_CACHE}.")

    computer = FastResponseComputer(RESPONSE_FIT, device)

    # Operators bank:
    # 0..7: Basis (I, II, V1, V2, V3, V4, V5, V6)
    # 8: Lead III = (II - I) / sqrt(2)
    # 9: Lead aVR = (-I - II) / sqrt(2)
    # 10: Lead aVL = (I - 0.5 II) / sqrt(1.25)
    # 11: Lead aVF = (II - 0.5 I) / sqrt(1.25)
    # 12: S_icm = (V3 - V2) / sqrt(2)
    # 13: Oblique 1 = (V1 - V6) / sqrt(2)
    # 14: Oblique 2 = (I + V4) / sqrt(2)
    # 15: Oblique 3 = (II - V5) / sqrt(2)
    e = np.eye(8, dtype=np.float32)
    q_bank = np.stack([
        e[0],                                       # 0: I
        e[1],                                       # 1: II
        e[2],                                       # 2: V1
        e[3],                                       # 3: V2
        e[4],                                       # 4: V3
        e[5],                                       # 5: V4
        e[6],                                       # 6: V5
        e[7],                                       # 7: V6
        normalize_operator(e[1] - e[0]),             # 8: III
        normalize_operator(-e[0] - e[1]),            # 9: aVR
        normalize_operator(e[0] - 0.5 * e[1]),       # 10: aVL
        normalize_operator(e[1] - 0.5 * e[0]),       # 11: aVF
        normalize_operator(e[4] - e[3]),             # 12: V3 - V2 (S_icm)
        normalize_operator(e[2] - e[7]),             # 13: V1 - V6
        normalize_operator(e[0] + e[5]),             # 14: I + V4
        normalize_operator(e[1] - e[6]),             # 15: II - V5
    ]).astype(np.float32)

    val_resps_all = np.empty((n_records, 16, 16, 128), dtype=np.float32)
    labels_all = np.empty((n_records, 5), dtype=np.float32)
    ecg_ids_all = qc_frame.ecg_id.to_numpy(dtype=np.int64)
    patient_ids_all = qc_frame.patient_id.to_numpy(dtype=np.int64)

    chunk_size = 500
    t0_start = time.time()

    for chunk_start in range(0, n_records, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_records)
        chunk_files = [VAL_CACHE / qc_frame.iloc[i].artifact for i in range(chunk_start, chunk_end)]

        def _load_beat(p):
            with np.load(p) as item:
                b = np.asarray(item["beats"], dtype=np.float32) * computer.std + computer.mean
                return b, np.asarray(item["labels"], dtype=np.float32)

        with ThreadPoolExecutor(max_workers=16) as ex:
            chunk_data = list(ex.map(_load_beat, chunk_files))

        for idx_in_chunk, (beats, lbl) in enumerate(chunk_data):
            rec_idx = chunk_start + idx_in_chunk
            labels_all[rec_idx] = lbl
            resps = computer.compute_record_responses(beats, q_bank)
            val_resps_all[rec_idx] = resps

        elapsed = time.time() - t0_start
        logger.info(f"Processed {chunk_end}/{n_records} validation records ({chunk_end/elapsed:.1f} rec/s)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p_val = OUT_DIR / "representation_val_real_waveforms.npz"
    logger.info(f"Saving real physical validation dataset to {p_val}...")
    np.savez_compressed(
        p_val,
        operators=np.tile(q_bank[None, :, :], (n_records, 1, 1)),
        responses=val_resps_all,
        labels=labels_all,
        ecg_ids=ecg_ids_all,
        patient_ids=patient_ids_all,
    )
    logger.info("=== Dataset generation complete! ===")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    build_validation_split(device)
    build_train_splits(device)


if __name__ == "__main__":
    main()
