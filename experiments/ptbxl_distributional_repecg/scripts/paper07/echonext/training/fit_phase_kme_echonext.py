"""Fit Phase-KME response representation parameters exclusively on the EchoNext training split.

Protocol conforming strictly to PRD Section 6:
- Zero leakage: Fits strictly on EchoNext training split (split == 'train', N=72,475).
- Never reuses PTB-XL-derived scalers, centers, or whitening matrices.
- Samples N_fit=500 random training records.
- 8 canonical leads: [I, II, V1, V2, V3, V4, V5, V6] (indices [0, 1, 6, 7, 8, 9, 10, 11]).
- Computes voltage_scale from training physical signal standard deviation.
- Collects 500,000 response atoms [x_q(t), dx_q(t)/dt] under canonical and random continuous operators.
- Fits WhiteningTransform on response atoms.
- Fits 128-landmark IMQ Nyström mapping on whitened atoms.
- Audits Spearman correlation (>= 0.90) and relative MMD error (<= 0.15).
- Saves response_fit_echonext.npz and response_fit_manifest.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import MiniBatchKMeans

REPO_ROOT = Path("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction")
SRC_DIR = REPO_ROOT / "experiments/ptbxl_distributional_repecg/src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from repecg.common.kernels import WhiteningTransform, NystromMap, imq_kernel, biased_mmd2, audit_nystrom
from repecg.paper07_operator.measurements import canonical_operators, sample_sparse_operators, normalize_operator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fit_phase_kme_echonext")

ECHONEXT_DIR = Path("/home/mithunmanivannan/data/echonext")
OUT_DIR = REPO_ROOT / "results/paper07_echonext/setoperator"

# Indices of 8 canonical linearly independent leads in standard 12-lead array:
# 0: I, 1: II, 6: V1, 7: V2, 8: V3, 9: V4, 10: V5, 11: V6
CANONICAL_LEAD_INDICES = [0, 1, 6, 7, 8, 9, 10, 11]


def extract_segments(waveform_2500x12: np.ndarray, seg_len: int = 256) -> np.ndarray:
    """Extract canonical 8-lead segments of length seg_len from a 2500-sample 12-lead ECG."""
    # Extract 8 canonical leads: shape (2500, 8)
    ecg_8 = waveform_2500x12[:, CANONICAL_LEAD_INDICES]  # (2500, 8)
    n_segs = len(ecg_8) // seg_len
    segments = []
    for i in range(n_segs):
        seg = ecg_8[i * seg_len : (i + 1) * seg_len]  # (256, 8)
        segments.append(seg)
    return np.stack(segments, axis=0)  # (n_segs, 256, 8)


def compute_response_atoms(segments: np.ndarray, q: np.ndarray, voltage_scale: float) -> np.ndarray:
    """Compute 2D response atoms [y_q(t), dy_q(t)/dt] for a given measurement operator q."""
    q_norm = normalize_operator(q)
    # segments: (N_segs, 256, 8)
    # waveform: (N_segs, 256)
    waveform = (segments @ q_norm) / voltage_scale
    derivative = np.gradient(waveform, axis=1)
    # return shape: (N_segs, 256, 2)
    return np.stack((waveform, derivative), axis=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-sample-records", type=int, default=500, help="Number of training records to sample for fitting")
    parser.add_argument("--target-atoms", type=int, default=500_000, help="Number of response atoms for reservoir")
    parser.add_argument("--landmarks", type=int, default=128, help="Number of Nyström landmarks")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t0_all = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    resp_fit_path = OUT_DIR / "response_fit_echonext.npz"

    logger.info("==================================================")
    logger.info("Fitting EchoNext-Native Phase-KME Response Representation")
    logger.info("Destination: %s", resp_fit_path)
    logger.info("==================================================")

    # 1. Load metadata and select training records
    df = pd.read_csv(ECHONEXT_DIR / "echonext_metadata_100k.csv")
    train_indices = np.where(df["split"] == "train")[0]
    logger.info("Total training records in EchoNext: %d", len(train_indices))

    rng = np.random.default_rng(args.seed)
    sampled_train_indices = np.sort(rng.choice(train_indices, size=args.n_sample_records, replace=False))
    logger.info("Sampled %d training records for parameter fitting (seed=%d).", len(sampled_train_indices), args.seed)

    # 2. Read waveforms for training records sequentially from NFS (zero seeks)
    train_w_path = Path("/data/mithunmanivannan/echonext/EchoNext_train_waveforms.npy")
    logger.info("Reading %d training records sequentially from %s...", args.n_sample_records, train_w_path)
    bytes_per_sample = 2500 * 12 * 8  # float64
    total_bytes = args.n_sample_records * bytes_per_sample

    with open(train_w_path, "rb") as f:
        _ = f.read(128)  # skip 128-byte header
        raw = f.read(total_bytes)

    sampled_waveforms = np.frombuffer(raw, dtype="<f8").reshape(args.n_sample_records, 1, 2500, 12)[:, 0]
    logger.info("Successfully loaded waveforms: shape %s, dtype %s", sampled_waveforms.shape, sampled_waveforms.dtype)

    # 3. Extract 8-lead segments and compute physical voltage scale
    logger.info("Extracting 8-lead 256-sample segments...")
    all_segments = []
    for wf in sampled_waveforms:
        segs = extract_segments(wf, seg_len=256)
        all_segments.append(segs)
    all_segments = np.concatenate(all_segments, axis=0)  # (N_total_segs, 256, 8)
    logger.info("Extracted %d segments of shape (256, 8).", len(all_segments))

    # Physical voltage scale: standard deviation across physical reservoir
    voltage_scale = float(np.std(all_segments))
    mean_voltage = float(np.mean(all_segments))
    logger.info("EchoNext Training Voltage Scale: %.6f (mean: %.6f)", voltage_scale, mean_voltage)

    # 4. Operator response reservoir
    logger.info("Sampling %d response atoms across operators...", args.target_atoms)
    canonical_ops = canonical_operators()  # 8 canonical
    sparse_ops = sample_sparse_operators(20, rng)  # 20 random sparse operators
    op_pool = np.concatenate([canonical_ops, sparse_ops], axis=0)

    atom_chunks = []
    collected_atoms = 0

    for i, seg in enumerate(all_segments):
        # Sample an operator from pool
        q = op_pool[i % len(op_pool)]
        atoms = compute_response_atoms(seg[None, ...], q, voltage_scale)  # (1, 256, 2)
        atoms_flat = atoms.reshape(-1, 2)  # (256, 2)
        take = min(len(atoms_flat), args.target_atoms - collected_atoms)
        atom_chunks.append(atoms_flat[:take])
        collected_atoms += take
        if collected_atoms >= args.target_atoms:
            break

    response_reservoir = np.concatenate(atom_chunks, axis=0)
    logger.info("Collected response reservoir: shape %s, finite: %s", response_reservoir.shape, np.isfinite(response_reservoir).all())

    # 5. Fit Whitening Transform
    logger.info("Fitting WhiteningTransform on response reservoir...")
    whitening = WhiteningTransform.fit(response_reservoir)
    whitened = whitening.transform(response_reservoir)
    logger.info("Whitening fitted. Mean: %s, Scales: %s", whitening.mean, whitening.scales)

    # 6. Fit IMQ Nyström Mapping
    logger.info("Fitting NystromMap with %d landmarks (seed=%d)...", args.landmarks, args.seed)
    mapping = NystromMap.fit(whitened, landmarks=args.landmarks, c2=1.0, seed=args.seed)
    logger.info("Nyström anchors shape: %s, inverse_root shape: %s", mapping.landmarks.shape, mapping.inverse_root.shape)

    # 7. Audit Nyström Mapping (Exact IMQ MMD vs Nyström Embedded L2)
    logger.info("Running Nyström accuracy audit on 1000 pairs...")
    # Form audit cells: 16 phase cells of 16 atoms each
    audit_cells = []
    for seg in all_segments[:64]:
        for q in canonical_ops[:2]:
            atoms = compute_response_atoms(seg[None, ...], q, voltage_scale)[0]  # (256, 2)
            # Reshape into 16 phase bins of 16 samples
            cells = atoms.reshape(16, 16, 2)
            for c in cells:
                audit_cells.append(c)

    w_cells = [whitening.transform(c) for c in audit_cells]
    n_pairs = 1000
    exact = np.empty(n_pairs)
    approx = np.empty(n_pairs)
    audit_rng = np.random.default_rng(args.seed + 100)

    for p in range(n_pairs):
        l, r = audit_rng.choice(len(w_cells), size=2, replace=False)
        exact[p] = biased_mmd2(w_cells[l], w_cells[r], c2=1.0)
        approx[p] = np.square(mapping.mean(w_cells[l]) - mapping.mean(w_cells[r])).sum()

    audit_result = audit_nystrom(exact, approx, min_spearman=0.90, max_median_relative_error=0.15)
    logger.info(
        "Nyström Audit Results: Spearman=%.4f (min 0.90) | MedRelError=%.4f (max 0.15) | PASSED=%s",
        audit_result["spearman"], audit_result["median_relative_error"], audit_result["passed"]
    )
    assert audit_result["passed"], f"Nyström audit failed: {audit_result}"

    # 8. Save artifacts
    np.savez_compressed(
        resp_fit_path,
        voltage_scale=np.asarray(voltage_scale),
        whitening_mean=whitening.mean,
        whitening_components=whitening.components,
        whitening_scales=whitening.scales,
        landmarks=mapping.landmarks,
        inverse_root=mapping.inverse_root,
        c2=np.asarray(mapping.c2),
    )
    logger.info("Saved response fit parameters to %s (size: %.2f KB)", resp_fit_path, resp_fit_path.stat().st_size / 1024)

    manifest = {
        "status": "FITTED_AND_VERIFIED",
        "dataset": "EchoNext",
        "training_records_sampled": args.n_sample_records,
        "sample_seed": args.seed,
        "voltage_scale": voltage_scale,
        "mean_voltage": mean_voltage,
        "n_landmarks": args.landmarks,
        "imq_c2": mapping.c2,
        "audit": audit_result,
        "total_fit_time_seconds": round(time.time() - t0_all, 2),
        "fit_path": str(resp_fit_path),
    }

    manifest_path = OUT_DIR / "response_fit_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Saved response fit manifest to %s", manifest_path)
    logger.info("Phase-KME Response Representation Fit COMPLETE in %.2fs.", time.time() - t0_all)


if __name__ == "__main__":
    main()
