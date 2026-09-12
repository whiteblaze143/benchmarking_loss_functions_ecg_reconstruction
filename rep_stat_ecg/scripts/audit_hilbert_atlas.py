"""Audits QVCG-H Hilbert Atlas artifacts against declared PRD invariants."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


MANDATORY_NON_FIGURE_ARTIFACTS = [
    "HILBERT_ATLAS_COORDS.parquet",
    "HILBERT_GRAM_MATRIX.npy",
    "HILBERT_EIGENSPECTRUM.csv",
    "HILBERT_ATLAS_SUMMARY.json",
    "RSP_COMPATIBILITY_SETS.parquet",
    "RSP_COMPATIBILITY_MEMBERSHIP.parquet",
    "RSP_COMPATIBILITY_OVERLAP.npy",
    "PHYSICAL_FUNCTIONAL_PAIR_TABLE.parquet",
    "FUNCTIONAL_FOLD_RANKING.parquet",
    "FUNCTIONAL_BOUNDARY_RANKING.parquet",
    "QAP_PHYSICAL_FUNCTIONAL.json",
]

MANDATORY_FIGURES = [
    "figure_h1_vcg_centroids_z1.png",
    "figure_h2_hilbert_coords_z1_z2.png",
    "figure_h3_physical_vs_hilbert_isotonic.png",
    "figure_h4_vcg_nonlocal_folds.png",
    "figure_h5_vcg_local_boundaries.png",
    "figure_h6_eigenspectrum_cumulative_mass.png",
    "figure_h7_compatibility_overlap_heatmap.png",
]


def audit_atlas(atlas_dir: Path) -> dict:
    results = {}
    errors = []

    # 0. Artifact Completeness (11 non-figure + 7 figures = 18 total)
    missing_artifacts = [f for f in MANDATORY_NON_FIGURE_ARTIFACTS if not (atlas_dir / f).exists()]
    if missing_artifacts:
        errors.append(f"Missing non-figure artifacts: {missing_artifacts}")
    results["n_non_figure_artifacts"] = len(MANDATORY_NON_FIGURE_ARTIFACTS) - len(missing_artifacts)

    missing_figures = [f for f in MANDATORY_FIGURES if not (atlas_dir / f).exists()]
    if missing_figures:
        errors.append(f"Missing primary figures: {missing_figures}")
    results["n_primary_figures"] = len(MANDATORY_FIGURES) - len(missing_figures)

    summary_file = atlas_dir / "HILBERT_ATLAS_SUMMARY.json"
    if not summary_file.exists():
        errors.append(f"Missing summary file: {summary_file}")
        return {"audit_status": "FAIL", "errors": errors}

    summary = json.loads(summary_file.read_text())

    # 1. Geometry Invariants
    gram_file = atlas_dir / "HILBERT_GRAM_MATRIX.npy"
    if gram_file.exists():
        B = np.load(gram_file)
        if B.shape != (64, 64):
            errors.append(f"Gram matrix shape {B.shape} != (64, 64)")
        if not np.allclose(B, B.T, atol=1e-12):
            errors.append("Gram matrix is not numerically symmetric")
        if not np.allclose(B @ np.ones(64), np.zeros(64), atol=1e-12):
            errors.append("Centering invariant failed: B @ 1 != 0")

    coords_file = atlas_dir / "HILBERT_ATLAS_COORDS.parquet"
    if coords_file.exists():
        coords_df = pd.read_parquet(coords_file)
        if len(coords_df) != 64:
            errors.append(f"Coords rows {len(coords_df)} != 64")
        if coords_df.isna().any().any():
            errors.append("Coords dataframe contains NaN values")
        # Check coordinate columns: domain_id, z_1 .. z_rplus, centroids
        z_cols = [c for c in coords_df.columns if c.startswith("z_")]
        results["n_coord_dimensions"] = len(z_cols)
        if len(z_cols) > 63:
            errors.append(f"Mathematical violation: Coordinate columns {len(z_cols)} > 63")

    eigenspec_file = atlas_dir / "HILBERT_EIGENSPECTRUM.csv"
    if eigenspec_file.exists():
        eigenspec_df = pd.read_csv(eigenspec_file)
        if len(eigenspec_df) != 64:
            errors.append(f"Eigenspectrum length {len(eigenspec_df)} != 64")
        if "status" not in eigenspec_df.columns:
            errors.append("Eigenspectrum missing 'status' column")

    # 2. Tolerances & Rank Correctness
    pos_rank = summary.get("positive_rank", 64)
    if pos_rank > 63:
        errors.append(f"Mathematical violation: Positive rank {pos_rank} > 63")
    results["positive_rank"] = pos_rank

    n_pos = summary.get("n_positive", 64)
    if n_pos > 63:
        errors.append(f"Mathematical violation: n_positive {n_pos} > 63")
    results["n_positive"] = n_pos
    results["n_numerical_zero"] = summary.get("n_numerical_zero", 0)
    results["n_negative"] = summary.get("n_negative", 0)

    neg_mass = summary.get("negative_eigenmass", 1.0)
    if neg_mass > 1e-6:
        errors.append(f"Material negative eigenmass {neg_mass:.6e} > 1e-6")
    results["negative_eigenmass"] = neg_mass

    recon_err = summary.get("distance_reconstruction_max_error", 1.0)
    if recon_err > 1e-12:
        errors.append(f"Distance reconstruction error {recon_err:.6e} > 1e-12")
    results["distance_reconstruction_max_error"] = recon_err

    # 3. Compatibility Cover Invariants
    sets_file = atlas_dir / "RSP_COMPATIBILITY_SETS.parquet"
    if sets_file.exists():
        sets_df = pd.read_parquet(sets_file)
        results["observed_n_cliques"] = len(sets_df)
        results["observed_n_nonlocal"] = int(sets_df["is_nonlocal"].sum())

    overlap_file = atlas_dir / "RSP_COMPATIBILITY_OVERLAP.npy"
    if overlap_file.exists():
        overlap = np.load(overlap_file)
        if overlap.shape[0] != overlap.shape[1]:
            errors.append("Overlap matrix is not square")
        if not np.allclose(overlap, overlap.T, atol=1e-12):
            errors.append("Overlap matrix is not symmetric")
        if not np.allclose(np.diag(overlap), 1.0, atol=1e-12):
            errors.append("Overlap matrix diagonal is not 1.0")

    # 4. Pair Table & Rankings
    pairs_file = atlas_dir / "PHYSICAL_FUNCTIONAL_PAIR_TABLE.parquet"
    if pairs_file.exists():
        pairs_df = pd.read_parquet(pairs_file)
        if len(pairs_df) != 2016:
            errors.append(f"Pair table length {len(pairs_df)} != 2016")

    folds_file = atlas_dir / "FUNCTIONAL_FOLD_RANKING.parquet"
    if folds_file.exists():
        folds_df = pd.read_parquet(folds_file)
        results["observed_n_supported_folds"] = len(folds_df)
        if len(folds_df) > 0 and (folds_df["fold_residual"] >= 0).any():
            errors.append("Supported fold ranking contains non-negative residuals")

    bounds_file = atlas_dir / "FUNCTIONAL_BOUNDARY_RANKING.parquet"
    if bounds_file.exists():
        bounds_df = pd.read_parquet(bounds_file)
        results["observed_n_local_boundaries"] = len(bounds_df)
        if len(bounds_df) > 0 and (bounds_df["fold_residual"] <= 0).any():
            errors.append("Local boundary ranking contains non-positive residuals")

    # 5. QAP Association (Software Gate: valid Monte Carlo execution, p in [0.0001, 1.0])
    qap_file = atlas_dir / "QAP_PHYSICAL_FUNCTIONAL.json"
    if not qap_file.exists():
        errors.append(f"Missing QAP results: {qap_file}")
    else:
        qap = json.loads(qap_file.read_text())
        results["qap_n_perm"] = qap.get("n_perm")
        if qap.get("n_perm") != 9999:
            errors.append(f"QAP permutations {qap.get('n_perm')} != 9999")
        p_val = qap.get("p_qap", -1.0)
        if not (0.0001 <= p_val <= 1.0):
            errors.append(f"QAP p-value {p_val} outside valid range [0.0001, 1.0]")
        results["observed_qap_rho"] = qap.get("rho_obs")
        results["observed_qap_p"] = p_val

    # 6. Leakage Check
    if summary.get("clinical_labels_used", True) is not False:
        errors.append("clinical_labels_used is True or missing")

    results["errors"] = errors
    results["audit_status"] = "PASS" if not errors else "FAIL"
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit QVCG-H Hilbert Atlas artifacts")
    parser.add_argument("--atlas-dir", default="refine-logs/qvcg/hilbert_atlas")
    args = parser.parse_args()

    atlas_dir = Path(args.atlas_dir)
    audit = audit_atlas(atlas_dir)

    print("=== QVCG-H Atlas Software Qualification Gates ===")
    print(f"  Artifact Count: {audit.get('n_non_figure_artifacts')}/11 non-figure, {audit.get('n_primary_figures')}/7 figures")
    print(f"  Positive Rank: {audit.get('positive_rank')} (<= 63 required)")
    print(f"  Negative Eigenmass: {audit.get('negative_eigenmass'):.6e} (<= 1e-6 required)")
    print(f"  Distance Reconstruction Max Error: {audit.get('distance_reconstruction_max_error'):.6e} (<= 1e-12 required)")
    print(f"  QAP Execution: {audit.get('qap_n_perm')} permutations completed, p in [0.0001, 1.0]")
    print(f"  Software Audit Status: {audit.get('audit_status')}")

    print("\n=== Observed Scientific Findings (Descriptive / Not Failure Gates) ===")
    print(f"  Observed Maximal Cliques: {audit.get('observed_n_cliques')}")
    print(f"  Observed Nonlocal Compatibility Sets: {audit.get('observed_n_nonlocal')}")
    print(f"  Observed Supported Negative Folds: {audit.get('observed_n_supported_folds')}")
    print(f"  Observed Local Functional Boundaries: {audit.get('observed_n_local_boundaries')}")
    print(f"  Observed Physical-Functional Association: rho = {audit.get('observed_qap_rho'):.4f}, Monte Carlo p = {audit.get('observed_qap_p'):.4f}")

    if audit["audit_status"] == "PASS":
        print("\nAll software qualification gates PASSED.")
        print("M3H_M1_IMPLEMENTATION=PASS")
        print("HILBERT_GEOMETRY=PASS")
        print("HILBERT_DISTANCE_RECONSTRUCTION=PASS")
    else:
        print("\nAudit FAILED with the following errors:")
        for err in audit["errors"]:
            print(f"  - {err}")
        raise RuntimeError("QVCG-H Atlas Audit Failed")


if __name__ == "__main__":
    main()
