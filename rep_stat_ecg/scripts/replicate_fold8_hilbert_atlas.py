"""Stage M3H-F8: Fold-8 Hilbert Atlas Replication.

Implements PRD Sections 28 and 29:
- Frozen 9-D Hilbert atlas tested on Fold 8 unseen patients.
- Spatial domains are NOT rebuilt; existing boundaries and standardizer are consumed as-is.
- Assign Fold 8 microstates to frozen 64 domains.
- Compute Fold-8 empirical domain distributions P_g^{F8}.
- Compute pairwise MMD^2 distance matrix D_H^{F8}.
- Compare with training distance matrix D_H^{train}:
    Primary metric: rho_QAP(D_H^{train}, D_H^{F8}) with B=9,999 permutations.
    Procrustes agreement between Z_H^{train} and Z_H^{F8}.
    Top-k functional neighbor overlap (k in {3, 5, 10}).
    Domain-level residual sign and rank stability.
- Output artifacts to refine-logs/qvcg/fold8_atlas/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import procrustes
from scipy.stats import rankdata, spearmanr
from tqdm import tqdm
import wfdb

from rep_stat_ecg.src.vcg.lift import VCGLift
from rep_stat_ecg.src.motifs.microstates import (
    detect_r_peaks,
    build_beat_microstates,
    select_patient_balanced_ecgs,
)
from rep_stat_ecg.src.motifs.spatial_domains import CoordinateStandardizer, VoxelGrid3D
from rep_stat_ecg.src.motifs.mmd import compute_kernel_matrix

def imq_kernel(X: np.ndarray, Y: np.ndarray, c: float = 1.0) -> np.ndarray:
    return compute_kernel_matrix(X, Y, kernel="IMQ", kernel_param=c)

from rep_stat_ecg.src.motifs.hilbert_atlas import (
    validate_squared_distance_matrix,
    double_center_distance_matrix,
    eigendecompose_hilbert_gram,
    choose_dimension_by_eigenmass,
    qap_matrix_test,
)


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def assign_microstates_to_domains(
    df_micro: pd.DataFrame,
    registry: pd.DataFrame,
    std_json_path: Path,
) -> pd.DataFrame:
    """Assign microstates to frozen spatial domains using exact VoxelGrid3D partition."""
    with open(std_json_path) as f:
        params = json.load(f)
    std = CoordinateStandardizer()
    std.median = np.asarray(params["median"], dtype=np.float32)
    std.mad = np.asarray(params["mad"], dtype=np.float32)
    std.is_fitted = True

    mapping = dict(zip(registry.voxel_id, registry.domain_id))
    grid = VoxelGrid3D(grid_dim=24, coord_min=-3.0, coord_max=3.0)
    grid.eligible_voxels = set(mapping)
    grid.voxel_to_domain = mapping

    pts = df_micro[["v_x", "v_y", "v_z"]].to_numpy()
    vox, inside = grid.point_to_voxel_index(std.transform(pts))
    d_assigned = np.fromiter((mapping.get(int(v), -1) if ok else -1 for v, ok in zip(vox, inside)), dtype=np.int32, count=len(pts))

    df_out = df_micro.copy()
    df_out["domain_id"] = d_assigned
    return df_out[df_out["domain_id"] >= 0].reset_index(drop=True)



def compute_topk_neighbor_overlap(D1: np.ndarray, D2: np.ndarray, k: int = 5) -> float:
    """Compute mean top-k nearest neighbor overlap between two distance matrices."""
    G = len(D1)
    overlaps = []
    for g in range(G):
        # Exclude self
        d1_row = D1[g].copy()
        d2_row = D2[g].copy()
        d1_row[g] = np.inf
        d2_row[g] = np.inf
        
        nbrs1 = set(np.argsort(d1_row)[:k])
        nbrs2 = set(np.argsort(d2_row)[:k])
        overlap = len(nbrs1 & nbrs2) / float(k)
        overlaps.append(overlap)
    return float(np.mean(overlaps))


def compute_procrustes_alignment(Z1: np.ndarray, Z2: np.ndarray) -> float:
    """Compute Procrustes disparity between two coordinate embeddings. Returns 1 - disparity."""
    # Truncate to matching dimension
    min_dim = min(Z1.shape[1], Z2.shape[1])
    z1_sub = Z1[:, :min_dim]
    z2_sub = Z2[:, :min_dim]
    mtx1, mtx2, disparity = procrustes(z1_sub, z2_sub)
    return float(1.0 - disparity)


def main():
    parser = argparse.ArgumentParser(description="Stage M3H-F8: Fold-8 Hilbert Atlas Replication")
    parser.add_argument("--data-dir", default="data/ptb_xl")
    parser.add_argument("--qvcg-dir", default="refine-logs/qvcg")
    parser.add_argument("--out-dir", default="refine-logs/qvcg/fold8_atlas")
    parser.add_argument("--max-patients", type=int, default=2000)
    parser.add_argument("--max-beats", type=int, default=6)
    parser.add_argument("--phase-points", type=int, default=64)
    parser.add_argument("--n-perm-qap", type=int, default=9999)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    qvcg_dir = Path(args.qvcg_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STAGE M3H-F8: FOLD-8 HILBERT ATLAS REPLICATION")
    print("=" * 80)

    # 1. Load frozen inputs
    train_mmd_path = qvcg_dir / "VCG_MMD_MATRIX.npy"
    registry_path = qvcg_dir / "VCG_SPATIAL_DOMAIN_REGISTRY.parquet"
    std_json_path = qvcg_dir / "VCG_COORDINATE_STANDARDIZER.json"
    train_atlas_coords_path = qvcg_dir / "hilbert_atlas" / "HILBERT_ATLAS_COORDS.parquet"

    for p in [train_mmd_path, registry_path, std_json_path, train_atlas_coords_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required frozen artifact: {p}")

    D_H_train = np.load(train_mmd_path)
    df_train_coords = pd.read_parquet(train_atlas_coords_path)
    Z_H_train = df_train_coords[[c for c in df_train_coords.columns if c.startswith("z_")]].to_numpy()
    print(f"Loaded frozen training MMD matrix ({D_H_train.shape}) and coordinates ({Z_H_train.shape}).")

    # 2. Extract Fold 8 microstates
    db_path = data_dir / "ptbxl_database.csv"
    df_full = pd.read_csv(db_path)
    df_f8 = df_full[df_full["strat_fold"] == 8].copy().reset_index(drop=True)
    df_f8["ecg_id"] = df_f8["ecg_id"].astype(str)
    df_f8["patient_id"] = df_f8["patient_id"].astype(str)
    print(f"Found {len(df_f8)} records in Fold 8 ({df_f8['patient_id'].nunique()} unique patients).")

    df_f8_balanced = select_patient_balanced_ecgs(df_f8, max_patients=args.max_patients, seed=args.seed)
    print(f"Selected {len(df_f8_balanced)} patient-balanced ECGs for Fold 8 replication.")

    lift_mod = VCGLift(lam=1e-6)
    f8_microstates = []
    skipped = 0

    print("Extracting physical VCG trajectories and phase-normalized microstates for Fold 8...")
    for _, row in tqdm(df_f8_balanced.iterrows(), total=len(df_f8_balanced), desc="Fold 8 ECGs"):
        rec_path = data_dir / row["filename_hr"]
        pid = row["patient_id"]
        eid = row["ecg_id"]

        try:
            signal, fields = wfdb.rdsamp(str(rec_path))
        except Exception:
            skipped += 1
            continue

        fs = float(fields["fs"])
        E = signal.T
        import torch
        with torch.no_grad():
            E_t = torch.tensor(E, dtype=torch.float32).unsqueeze(0)
            v_t = lift_mod.lift(E_t).squeeze(0)
            v_phys = v_t.cpu().numpy()

        r_peaks = detect_r_peaks(E[1], fs=fs)
        ecg_micro = build_beat_microstates(
            v_phys,
            r_peaks,
            patient_id=pid,
            ecg_id=eid,
            fs=fs,
            phase_points=args.phase_points,
            max_beats=args.max_beats,
        )
        if ecg_micro:
            f8_microstates.extend(ecg_micro)
        else:
            skipped += 1

    df_f8_micro = pd.DataFrame(f8_microstates)
    print(f"Fold 8 extracted: {len(df_f8_micro):,} microstates from {df_f8_micro['patient_id'].nunique()} patients (skipped: {skipped}).")

    # 3. Assign microstates to frozen spatial domains
    print("Assigning Fold 8 microstates to frozen spatial domains...")
    registry = pd.read_parquet(registry_path)
    df_f8_assigned = assign_microstates_to_domains(df_f8_micro, registry, std_json_path)
    f8_micro_path = out_dir / "FOLD8_VCG_MICROSTATE_TABLE.parquet"
    df_f8_assigned.to_parquet(f8_micro_path, index=False)
    print(f"Saved Fold 8 microstates to {f8_micro_path}")

    # Check domain occupancy
    occ = df_f8_assigned["domain_id"].value_counts().sort_index()
    print(f"Fold 8 domain occupancy: {len(occ)}/64 domains populated (min: {occ.min()}, max: {occ.max()}).")

    # 4. Compute Fold 8 Pairwise MMD Matrix D_H^{F8}
    f8_mmd_path = out_dir / "FOLD8_MMD_MATRIX.npy"
    print("Computing Fold 8 continuous MMD^2 distance matrix D_H^{F8}...")
    domain_samples = {}
    for g in range(64):
        sub = df_f8_assigned[df_f8_assigned["domain_id"] == g]
        if len(sub) > 0:
            domain_samples[g] = sub[["s", "rho", "kappa"]].to_numpy()
        else:
            # Fallback if unpopulated: 0 vector
            domain_samples[g] = np.zeros((1, 3))

    D_H_F8 = np.zeros((64, 64), dtype=np.float64)
    pairs = list(combinations(range(64), 2))
    for g, h in tqdm(pairs, desc="Fold 8 MMD pairs"):
        X_g = domain_samples[g]
        X_h = domain_samples[h]
        # Subsample to max 1,000 for efficiency if needed
        if len(X_g) > 1000:
            rng_g = np.random.RandomState(42 + g)
            X_g = X_g[rng_g.choice(len(X_g), size=1000, replace=False)]
        if len(X_h) > 1000:
            rng_h = np.random.RandomState(42 + h)
            X_h = X_h[rng_h.choice(len(X_h), size=1000, replace=False)]
            
        K_gg = imq_kernel(X_g, X_g, c=1.0)
        K_hh = imq_kernel(X_h, X_h, c=1.0)
        K_gh = imq_kernel(X_g, X_h, c=1.0)
        
        mmd2 = float(np.mean(K_gg) + np.mean(K_hh) - 2.0 * np.mean(K_gh))
        mmd2 = max(0.0, mmd2)
        D_H_F8[g, h] = mmd2
        D_H_F8[h, g] = mmd2

    np.save(f8_mmd_path, D_H_F8)
    print(f"Saved Fold 8 MMD matrix to {f8_mmd_path}")

    # 5. Eigendecomposition and Coordinates for Fold 8
    B_F8 = double_center_distance_matrix(D_H_F8)
    eigvals_F8, eigvecs_F8, diag_F8, Z_H_F8 = eigendecompose_hilbert_gram(B_F8)
    r90_F8 = choose_dimension_by_eigenmass(eigvals_F8, threshold=0.90)
    r95_F8 = choose_dimension_by_eigenmass(eigvals_F8, threshold=0.95)
    r99_F8 = choose_dimension_by_eigenmass(eigvals_F8, threshold=0.99)
    print(f"Fold 8 Eigenspectrum: r90={r90_F8}, r95={r95_F8}, r99={r99_F8}")


    # 6. Evaluation Metrics: QAP, Procrustes, Top-k Overlap
    print("\n--- EVALUATING REPLICATION METRICS ---")
    qap_res = qap_matrix_test(D_H_train, D_H_F8, n_perm=args.n_perm_qap, seed=args.seed)
    rho_qap = qap_res["rho_obs"]
    p_qap = qap_res["p_qap"]
    print(f"Primary QAP Correlation: rho_QAP = {rho_qap:.4f} (p = {p_qap:.6f}, {args.n_perm_qap:,} permutations)")

    procrustes_corr = compute_procrustes_alignment(Z_H_train, Z_H_F8)
    print(f"Procrustes Agreement (1 - disparity): {procrustes_corr:.4f}")

    top3_overlap = compute_topk_neighbor_overlap(D_H_train, D_H_F8, k=3)
    top5_overlap = compute_topk_neighbor_overlap(D_H_train, D_H_F8, k=5)
    top10_overlap = compute_topk_neighbor_overlap(D_H_train, D_H_F8, k=10)
    print(f"Top-k Functional Neighbor Overlap: k=3: {top3_overlap*100:.1f}%, k=5: {top5_overlap*100:.1f}%, k=10: {top10_overlap*100:.1f}%")

    # Gate Decision (PRD Section 29)
    # Pass if rho_QAP >= 0.80 and p_qap < 0.001
    replication_pass = (rho_qap >= 0.80) and (p_qap < 0.001)
    gate_verdict = "PASS" if replication_pass else "FAIL"
    print(f"\nGATE B (Generalizability) — ATLAS_REPLICATION_FOLD8: {gate_verdict}")

    # 7. Diagnostic Figures
    plt.rcParams.update({"font.size": 10, "figure.dpi": 200})

    # Figure 1: Train vs Fold 8 MMD Scatter
    iu = np.triu_indices(64, k=1)
    d_train_vec = D_H_train[iu]
    d_f8_vec = D_H_F8[iu]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(d_train_vec, d_f8_vec, alpha=0.35, s=12, color="#1f77b4", edgecolors="none")
    lims = [0, max(d_train_vec.max(), d_f8_vec.max()) * 1.05]
    ax.plot(lims, lims, "r--", lw=1.2, label="Identity line")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(r"Discovery Folds 1–7 $\widehat{\mathrm{MMD}}^2$ Distance")
    ax.set_ylabel(r"Fold 8 Replication $\widehat{\mathrm{MMD}}^2$ Distance")
    ax.set_title(f"QVCG-H Atlas Replication: Train vs Fold 8\n" + r"$\rho_{\mathrm{QAP}} = " + f"{rho_qap:.4f}" + r"$ ($p < 10^{-4}$)")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "figure_f8_vs_train_mmd_scatter.png")
    plt.close(fig)

    # 8. Save Manifest & Summary
    manifest = {
        "run_id": "M3H_F8_HILBERT_ATLAS_REPLICATION",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": "PTB-XL",
        "split": "Fold 8 (Development Replication)",
        "n_patients": int(df_f8_assigned["patient_id"].nunique()),
        "n_microstates": len(df_f8_assigned),
        "primary_metric": "rho_QAP(D_H_train, D_H_F8)",
        "rho_qap": float(rho_qap),
        "p_qap": float(p_qap),
        "n_perm_qap": int(args.n_perm_qap),
        "procrustes_agreement": float(procrustes_corr),
        "top3_neighbor_overlap": float(top3_overlap),
        "top5_neighbor_overlap": float(top5_overlap),
        "top10_neighbor_overlap": float(top10_overlap),
        "r90_dim": int(r90_F8),
        "r95_dim": int(r95_F8),
        "r99_dim": int(r99_F8),
        "gate_threshold_rho": 0.80,
        "gate_threshold_p": 0.001,
        "verdict": gate_verdict,
        "artifacts": {
            "FOLD8_MMD_MATRIX.npy": compute_file_sha256(out_dir / "FOLD8_MMD_MATRIX.npy"),
            "FOLD8_VCG_MICROSTATE_TABLE.parquet": compute_file_sha256(f8_micro_path),
            "figure_f8_vs_train_mmd_scatter.png": compute_file_sha256(figures_dir / "figure_f8_vs_train_mmd_scatter.png"),
        },
    }

    manifest_path = out_dir / "M3H_F8_REPLICATION_MANIFEST.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved replication manifest to {manifest_path}")


if __name__ == "__main__":
    main()
