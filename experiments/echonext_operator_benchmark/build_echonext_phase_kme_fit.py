#!/usr/bin/env python3
"""Fit authentic Phase-KME representation parameters on the EchoNext Training Set.

Strict Protocol:
1. Uses exclusively EchoNext_train_waveforms.npy (Train split, N=72,475).
   Val (N=4,626) and Test (N=5,442) partitions remain strictly untouched.
2. Slices the 8 independent canonical leads [0, 1, 6, 7, 8, 9, 10, 11] (I, II, V1..V6).
3. Detects R-peaks using detect_rpeaks(basis, fs=250) and extracts beats via phase_normalize(basis, valid_intervals, samples=256).
4. Determines voltage scale from training beats.
5. Computes response atoms (y_q / v_scale, dy_q / v_scale).
6. Fits WhiteningTransform (mean, components, scales).
7. Fits NystromMap with 128 landmarks (MiniBatchKMeans, c2=1.0, seed=42).
8. Audits kernel representation quality against exact IMQ MMD^2.
9. Saves fitted parameters to /data/mithunmanivannan/codex_artifacts/echonext_three_way_v1/representations/echonext_phase_kme_fit.npz.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import MiniBatchKMeans
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_echonext_kme_fit")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = REPO_ROOT / "experiments/ptbxl_distributional_repecg/src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from repecg.common.beats import detect_rpeaks
from repecg.common.phase import phase_normalize
from repecg.common.kernels import WhiteningTransform, NystromMap, imq_kernel, biased_mmd2
from repecg.paper07_operator.measurements import normalize_operator

DATA_DIR = Path("/data/mithunmanivannan/echonext")
OUT_DIR = Path("/data/mithunmanivannan/codex_artifacts/echonext_three_way_v1/representations")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CANONICAL_8_LEADS = [0, 1, 6, 7, 8, 9, 10, 11]


def extract_training_beats(
    waveforms: np.ndarray,
    n_records_to_sample: int = 500,
    seed: int = 42,
) -> np.ndarray:
    """Sample records from training split and extract phase-normalized beats."""
    rng = np.random.default_rng(seed)
    total_train = len(waveforms)
    indices = rng.choice(total_train, size=n_records_to_sample, replace=False)
    indices.sort()

    logger.info("Extracting beats from %d sampled EchoNext training records...", n_records_to_sample)
    all_beats = []
    eligible_count = 0

    t0 = time.time()
    for idx in indices:
        sig_8 = waveforms[idx, 0][:, CANONICAL_8_LEADS].astype(np.float64)  # (2500, 8)
        det = detect_rpeaks(sig_8, fs=250)
        beats = phase_normalize(sig_8, det.valid_intervals, samples=256)    # (n_beats, 256, 8)
        if len(beats) >= 2:
            eligible_count += 1
            all_beats.append(beats)

    logger.info(
        "Extracted %d total beats across %d/%d eligible records in %.1fs",
        sum(len(b) for b in all_beats),
        eligible_count,
        n_records_to_sample,
        time.time() - t0,
    )
    return np.concatenate(all_beats, axis=0)  # (Total_Beats, 256, 8)


def sample_response_atoms(
    beats: np.ndarray,
    voltage_scale: float,
    n_atoms_target: int = 50000,
    seed: int = 42,
) -> np.ndarray:
    """Sample response atoms (v, s) across random operators and cardiac phases."""
    rng = np.random.default_rng(seed)
    n_beats = len(beats)
    atoms_list = []
    atoms_collected = 0

    # 8 canonical basis operators + random directions
    canonical_ops = [np.eye(8, dtype=np.float64)[i] for i in range(8)]

    for i in range(n_beats):
        b = beats[i]  # (256, 8)
        # Select 2 operators per beat: 1 canonical, 1 random
        q_can = canonical_ops[rng.integers(0, 8)]
        q_rand = normalize_operator(rng.normal(size=8)).astype(np.float64)

        for q in [q_can, q_rand]:
            y = b @ q  # (256,)
            y_scaled = y / voltage_scale
            dy_scaled = np.gradient(y_scaled)
            atom_pair = np.stack([y_scaled, dy_scaled], axis=-1)  # (256, 2)
            atoms_list.append(atom_pair)
            atoms_collected += len(atom_pair)
            if atoms_collected >= n_atoms_target:
                break
        if atoms_collected >= n_atoms_target:
            break

    atoms_pool = np.concatenate(atoms_list, axis=0)
    indices = rng.choice(len(atoms_pool), size=min(n_atoms_target, len(atoms_pool)), replace=False)
    return atoms_pool[indices]


def main():
    parser = argparse.ArgumentParser(description="Build EchoNext Phase-KME Nystrom fit.")
    parser.add_argument("--train-waveforms", type=Path, default=DATA_DIR / "EchoNext_train_waveforms.npy")
    parser.add_argument("--output-fit", type=Path, default=OUT_DIR / "echonext_phase_kme_fit.npz")
    parser.add_argument("--n-sample-records", type=int, default=500)
    parser.add_argument("--n-landmarks", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger.info("==================================================")
    logger.info("BUILDING ECHONEXT PHASE-KME NYSTROM FIT")
    logger.info("Source: %s", args.train_waveforms)
    logger.info("Destination: %s", args.output_fit)
    logger.info("==================================================")

    # 1. Load training waveforms using mmap
    wf_train = np.load(args.train_waveforms, mmap_mode="r")
    logger.info("Train waveforms shape: %s, dtype: %s", wf_train.shape, wf_train.dtype)
    assert wf_train.shape[0] == 72475, f"Unexpected training count: {wf_train.shape[0]}"

    # 2. Extract authentic training beats
    beats = extract_training_beats(wf_train, n_records_to_sample=args.n_sample_records, seed=args.seed)
    logger.info("Beats array shape: %s", beats.shape)

    # 3. Compute physical scale
    physical_mean = np.mean(beats, axis=(0, 1))
    physical_std = np.std(beats, axis=(0, 1))
    voltage_scale = float(np.std(beats))
    logger.info("Calculated voltage scale: %.4f (physical std: %s)", voltage_scale, np.round(physical_std, 3))

    # 4. Sample response atoms
    atoms_reservoir = sample_response_atoms(beats, voltage_scale=voltage_scale, n_atoms_target=50000, seed=args.seed)
    logger.info("Atoms reservoir shape: %s", atoms_reservoir.shape)

    # 5. Fit 2D Whitening Transform
    logger.info("Fitting WhiteningTransform on training atoms...")
    whitening = WhiteningTransform.fit(atoms_reservoir)
    whitened_atoms = whitening.transform(atoms_reservoir)
    logger.info("Whitening mean: %s", whitening.mean)
    logger.info("Whitening scales: %s", whitening.scales)

    # 6. Fit Nystrom Map
    logger.info("Fitting NystromMap with %d landmarks...", args.n_landmarks)
    nystrom = NystromMap.fit(whitened_atoms, landmarks=args.n_landmarks, c2=1.0, seed=args.seed)
    logger.info("Landmarks shape: %s", nystrom.landmarks.shape)
    logger.info("Inverse root condition number: %.2e", np.linalg.cond(nystrom.inverse_root))

    # 7. Audit Nystrom Map Precision vs Exact IMQ MMD2
    logger.info("Auditing Nystrom approximation fidelity against exact IMQ kernel...")
    rng = np.random.default_rng(args.seed)
    n_pairs = 100
    exact_mmd2 = np.empty(n_pairs)
    approx_dist2 = np.empty(n_pairs)

    # Group atoms into phase cells (16 atoms per cell)
    cell_size = 16
    n_cells = len(whitened_atoms) // cell_size
    cells = [whitened_atoms[i * cell_size : (i + 1) * cell_size] for i in range(n_cells)]

    for p in range(n_pairs):
        i1, i2 = rng.choice(len(cells), size=2, replace=False)
        c1, c2 = cells[i1], cells[i2]
        exact_mmd2[p] = biased_mmd2(c1, c2, c2=1.0)
        phi_c1 = (1.0 / np.sqrt(((c1[:, None, :] - nystrom.landmarks[None, :, :]) ** 2).sum(axis=-1) + 1.0)) @ nystrom.inverse_root
        phi_c2 = (1.0 / np.sqrt(((c2[:, None, :] - nystrom.landmarks[None, :, :]) ** 2).sum(axis=-1) + 1.0)) @ nystrom.inverse_root
        approx_dist2[p] = np.sum((phi_c1.mean(axis=0) - phi_c2.mean(axis=0)) ** 2)

    corr, pval = spearmanr(exact_mmd2, approx_dist2)
    logger.info("Audit Result: Spearman Rank Correlation r = %.4f (p = %.2e)", corr, pval)
    if corr < 0.90:
        raise RuntimeError(f"Nystrom audit failed: rank correlation {corr:.4f} < 0.90")

    # 8. Save fitted parameters to NPZ
    args.output_fit.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_fit,
        physical_mean=physical_mean.astype(np.float32),
        physical_std=physical_std.astype(np.float32),
        voltage_scale=np.float32(voltage_scale),
        whitening_mean=whitening.mean.astype(np.float32),
        whitening_components=whitening.components.astype(np.float32),
        whitening_scales=whitening.scales.astype(np.float32),
        landmarks=nystrom.landmarks.astype(np.float32),
        inverse_root=nystrom.inverse_root.astype(np.float32),
        c2=np.float32(1.0),
        dt=np.float32(1.0 / 250.0),
        sampling_rate=np.int32(250),
        spearman_audit_r=np.float32(corr),
        num_train_records_sampled=np.int32(args.n_sample_records),
    )
    logger.info("Saved verified EchoNext Phase-KME fit to %s", args.output_fit)


if __name__ == "__main__":
    main()
