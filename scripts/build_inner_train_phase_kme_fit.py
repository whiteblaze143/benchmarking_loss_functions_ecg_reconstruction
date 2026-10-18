#!/usr/bin/env python3
"""Fit authentic Phase-KME representation strictly on Inner-Train Patients."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.cluster import MiniBatchKMeans
import torch

REPO_ROOT = Path("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction")
SRC_DIR = REPO_ROOT / "experiments/ptbxl_distributional_repecg/src"
sys.path.insert(0, str(SRC_DIR))

from repecg.common.beats import detect_rpeaks
from repecg.common.phase import phase_normalize
from repecg.common.kernels import WhiteningTransform, NystromMap
from repecg.paper07_operator.measurements import normalize_operator

split_indices_path = Path("/data/mithunmanivannan/codex_artifacts/echonext_geometry_setoperator_v1/echonext_inner_dev_split_indices.npz")
split_data = np.load(split_indices_path)
inner_train_indices = split_data["inner_train_indices"]

waveforms = np.load("/data/mithunmanivannan/echonext/EchoNext_train_waveforms.npy", mmap_mode="r")
CANONICAL_8_LEADS = [0, 1, 6, 7, 8, 9, 10, 11]

rng = np.random.default_rng(42)
sampled_train_idx = rng.choice(inner_train_indices, size=500, replace=False)
sampled_train_idx.sort()

all_beats = []
eligible_count = 0
t0 = time.time()
print(f"Sampling 500 records strictly from {len(inner_train_indices)} inner-train records...", flush=True)

for i, idx in enumerate(sampled_train_idx):
    sig_8 = waveforms[idx, 0][:, CANONICAL_8_LEADS].astype(np.float64)  # (2500, 8)
    det = detect_rpeaks(sig_8, fs=250)
    beats = phase_normalize(sig_8, det.valid_intervals, samples=256)
    if len(beats) >= 2:
        eligible_count += 1
        all_beats.append(beats)
    if (i + 1) % 100 == 0:
        print(f"  Processed {i+1}/500 records ({eligible_count} eligible) in {time.time()-t0:.1f}s", flush=True)

beats_cat = np.concatenate(all_beats, axis=0)
print(f"Extracted {len(beats_cat)} beats across {eligible_count}/500 records in {time.time()-t0:.1f}s.", flush=True)

# Voltage scale
v_scale = float(np.percentile(np.abs(beats_cat), 95))
if v_scale <= 0.0:
    v_scale = 1.0
print(f"Computed voltage scale: {v_scale:.6f}", flush=True)

# Sample response atoms
atoms_list = []
atoms_collected = 0
n_atoms_target = 50000
canonical_ops = [np.eye(8, dtype=np.float64)[i] for i in range(8)]
sample_rng = np.random.default_rng(42)

for i in range(len(beats_cat)):
    b = beats_cat[i]
    q_can = canonical_ops[sample_rng.integers(0, 8)]
    q_rand = normalize_operator(sample_rng.normal(size=8)).astype(np.float64)
    for q in [q_can, q_rand]:
        y = b @ q
        y_scaled = y / v_scale
        dy_scaled = np.gradient(y_scaled)
        atom_pair = np.stack([y_scaled, dy_scaled], axis=-1)
        atoms_list.append(atom_pair)
        atoms_collected += len(atom_pair)
        if atoms_collected >= n_atoms_target:
            break
    if atoms_collected >= n_atoms_target:
        break

atom_pool = np.concatenate(atoms_list, axis=0)
print(f"Atom pool size: {len(atom_pool)}", flush=True)

whitener = WhiteningTransform.fit(atom_pool)
whitened_atoms = whitener.transform(atom_pool)

mapping = NystromMap.fit(whitened_atoms, landmarks=128, c2=1.0)
print(f"Fitted NystromMap with {len(mapping.landmarks)} landmarks and inverse_root shape {mapping.inverse_root.shape}.", flush=True)

out_dir = Path("/data/mithunmanivannan/codex_artifacts/echonext_geometry_setoperator_v1/representations")
out_dir.mkdir(parents=True, exist_ok=True)
fit_path = out_dir / "echonext_inner_train_phase_kme_fit.npz"

np.savez_compressed(
    fit_path,
    landmarks=mapping.landmarks.astype(np.float32),
    inverse_root=mapping.inverse_root.astype(np.float32),
    voltage_scale=np.float32(v_scale),
    whitening_mean=whitener.mean.astype(np.float32),
    whitening_components=whitener.components.astype(np.float32),
    whitening_scales=whitener.scales.astype(np.float32),
    c2=np.float32(1.0),
    sampling_rate=np.int32(250),
    landmark_dim=np.int32(128),
    phase_dim=np.int32(16),
    n_sample_records=np.int32(500),
    sample_seed=np.int32(42),
    isolation="STRICT_INNER_TRAIN_ONLY"
)

with open(fit_path, "rb") as f:
    sha = hashlib.sha256(f.read()).hexdigest()
print(f"Saved strictly isolated Phase-KME fit to {fit_path}", flush=True)
print(f"SHA-256: {sha}", flush=True)
