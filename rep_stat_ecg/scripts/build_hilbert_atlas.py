"""Builds the QVCG-H Hilbert Atlas of Cardiac Electrical Dynamics from repSpat.

Implements Milestone 1 pipeline per PRD v1.0 Section 51:
1. Frozen M3/M3R input audit and SHA-256 verification
2. Double-centered Hilbert Gram matrix B and continuous coordinates Z_H
3. Eigenspectrum, cumulative eigenmass, and dimension selection
4. Maximal compatibility cliques and physical recurrence classification
5. Physical-to-functional mapping, isotonic regression, fold and boundary rankings
6. QAP matrix permutation test (B=9,999)
7. Publication figures H1-H7
8. Exports all required artifacts to refine-logs/qvcg/hilbert_atlas/
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from rep_stat_ecg.src.motifs.hilbert_atlas import (
    validate_squared_distance_matrix,
    double_center_distance_matrix,
    eigendecompose_hilbert_gram,
    choose_dimension_by_eigenmass,
    reconstruct_squared_distances,
    enumerate_compatibility_cliques,
    build_physical_adjacency,
    classify_physical_recurrence,
    fit_physical_functional_isotonic,
    compute_fold_residuals,
    qap_matrix_test,
)


EXPECTED_HASHES = {
    "VCG_MMD_MATRIX.npy": "890e77082220f165fef03850324e4abe6fd69d4137cdfdad44845e17ef5e7fd5",
    "VCG_REPSPAT_PAIR_TESTS.parquet": "a6a9c93dceb008b2e6fe984016afd05939746eae8d16cf2697e0e489c1a2edfb",
    "VCG_DOMAIN_STATS.parquet": "b973995cb3a266e7edbbbe084cc4b12a8943cbe3961dd72b80b8183d62ad3e1e",
    "VCG_SPATIAL_DOMAIN_REGISTRY.parquet": "e2f805f40583aa00e7b8687d23cd8fdd946de38874922f270f9f61d5929826b2",
    "QVCG_CC_MOTIF_GATE.json": "b0211cc29ee0e2744fb9d1f0078bb9a20a6536eadf7b989b1fc0cfe87a7436e3",
    "M3R_MMD_MATRIX.npy": "320fce9e38950e94dde10b36678abb11c6f01bbd6277cd5b1f427986cd9f259e",
    "M3R_PAIR_TESTS.parquet": "80c579d6aa9845b076aed323f0f3503bcb9a499e6e37115ffef9e7953b351929",
    "M3R_VS_M3_COMPARISON.json": "6a8bab4ea044b9a33e66106e6d0a8a9544e26fdb6d13d8a00b75bfeca13fc27e",
}


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_inputs(qvcg_dir: Path, m3r_dir: Path) -> dict[str, str]:
    hashes = {}
    # Primary M3
    for name in [
        "VCG_MMD_MATRIX.npy",
        "VCG_REPSPAT_PAIR_TESTS.parquet",
        "VCG_DOMAIN_STATS.parquet",
        "VCG_SPATIAL_DOMAIN_REGISTRY.parquet",
        "QVCG_CC_MOTIF_GATE.json",
    ]:
        p = qvcg_dir / name
        if not p.exists():
            raise RuntimeError(f"M3H_INPUT_AUDIT=FAIL: Missing {p}")
        h = compute_sha256(p)
        if h != EXPECTED_HASHES[name]:
            raise RuntimeError(f"M3H_INPUT_AUDIT=FAIL: Hash mismatch for {name}: {h} != {EXPECTED_HASHES[name]}")
        hashes[name] = h

    # Reference M3R
    for name in ["M3R_MMD_MATRIX.npy", "M3R_PAIR_TESTS.parquet", "M3R_VS_M3_COMPARISON.json"]:
        p = m3r_dir / name
        if not p.exists():
            raise RuntimeError(f"M3H_INPUT_AUDIT=FAIL: Missing {p}")
        h = compute_sha256(p)
        if h != EXPECTED_HASHES[name]:
            raise RuntimeError(f"M3H_INPUT_AUDIT=FAIL: Hash mismatch for {name}: {h} != {EXPECTED_HASHES[name]}")
        hashes[name] = h

    return hashes


def plot_primary_figures(
    out_dir: Path,
    centroids: np.ndarray,
    Z_H: np.ndarray,
    D_P_mat: np.ndarray,
    D_H_mat: np.ndarray,
    d_p_vec: np.ndarray,
    d_h_vec: np.ndarray,
    d_h_pred: np.ndarray,
    pairs_df: pd.DataFrame,
    eigvals: np.ndarray,
    cum_mass: np.ndarray,
    r90: int,
    r95: int,
    r99: int,
    overlap_mat: np.ndarray,
    physical_adj: nx.Graph,
) -> None:
    plt.style.use("default")

    # Figure H1: Physical VCG centroids colored by Hilbert coordinate 1
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(
        centroids[:, 0], centroids[:, 1], c=Z_H[:, 0], cmap="viridis", s=70, edgecolors="k", linewidth=0.5
    )
    plt.colorbar(scatter, ax=ax, label="Hilbert Coordinate 1 ($z_1$)")
    ax.set_xlabel("VCG Centroid X ($vx$)")
    ax.set_ylabel("VCG Centroid Y ($vy$)")
    ax.set_title("Figure H1: Physical VCG Centroids Colored by Hilbert Coord 1")
    plt.tight_layout()
    plt.savefig(out_dir / "figure_h1_vcg_centroids_z1.png", dpi=300)
    plt.close()

    # Figure H2: Functional Hilbert coordinates z1, z2
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(Z_H[:, 0], Z_H[:, 1], c="tab:blue", s=50, edgecolors="k", linewidth=0.5)
    for i in range(len(Z_H)):
        ax.annotate(str(i), (Z_H[i, 0], Z_H[i, 1]), fontsize=7, alpha=0.7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Hilbert Dimension 1 ($z_1$)")
    ax.set_ylabel("Hilbert Dimension 2 ($z_2$)")
    ax.set_title("Figure H2: Functional Hilbert Coordinates ($z_1, z_2$)")
    plt.tight_layout()
    plt.savefig(out_dir / "figure_h2_hilbert_coords_z1_z2.png", dpi=300)
    plt.close()

    # Figure H3: D_P vs D_H with isotonic fit
    order = np.argsort(d_p_vec)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(d_p_vec, d_h_vec, alpha=0.25, s=15, color="darkgray", label="Observed Pairs (N=2,016)")
    ax.plot(d_p_vec[order], d_h_pred[order], color="crimson", linewidth=2.5, label=r"Isotonic Fit $\hat{f}(D_P)$")
    ax.set_xlabel(r"Physical Distance $D_P$")
    ax.set_ylabel(r"Functional Hilbert Distance $D_H$")
    ax.set_title("Figure H3: Physical vs Functional Distance with Isotonic Fit")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_dir / "figure_h3_physical_vs_hilbert_isotonic.png", dpi=300)
    plt.close()

    # Figure H4: Physical VCG map with top negative fold residual pairs connected
    top_folds = pairs_df[pairs_df.is_supported_fold].sort_values("fold_residual").head(25)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(centroids[:, 0], centroids[:, 1], c="gray", s=40, alpha=0.6)
    for _, row in top_folds.iterrows():
        d1, d2 = int(row.domain_1), int(row.domain_2)
        ax.plot([centroids[d1, 0], centroids[d2, 0]], [centroids[d1, 1], centroids[d2, 1]], "b-", alpha=0.7, linewidth=1.5)
    ax.set_xlabel("VCG Centroid X")
    ax.set_ylabel("VCG Centroid Y")
    ax.set_title(r"Figure H4: Top Supported Functional Folds ($F_{gh} < 0, A_{gh}=1$)")
    plt.tight_layout()
    plt.savefig(out_dir / "figure_h4_vcg_nonlocal_folds.png", dpi=300)
    plt.close()

    # Figure H5: Physical adjacency graph highlighting top positive boundary residuals
    top_boundaries = pairs_df[pairs_df.is_local_boundary].sort_values("fold_residual", ascending=False).head(25)
    fig, ax = plt.subplots(figsize=(6, 5))
    # Draw all physical contact edges faintly
    for u, v in physical_adj.edges():
        ax.plot([centroids[u, 0], centroids[v, 0]], [centroids[u, 1], centroids[v, 1]], color="lightgray", linewidth=0.8, alpha=0.5)
    # Draw top boundaries in bright red
    for _, row in top_boundaries.iterrows():
        d1, d2 = int(row.domain_1), int(row.domain_2)
        ax.plot([centroids[d1, 0], centroids[d2, 0]], [centroids[d1, 1], centroids[d2, 1]], "r-", linewidth=2.0, alpha=0.8)
    ax.scatter(centroids[:, 0], centroids[:, 1], c="black", s=30, zorder=5)
    ax.set_xlabel("VCG Centroid X")
    ax.set_ylabel("VCG Centroid Y")
    ax.set_title(r"Figure H5: Local Functional Boundaries ($F_{gh} > 0, (g,h) \in E_P$)")
    plt.tight_layout()
    plt.savefig(out_dir / "figure_h5_vcg_local_boundaries.png", dpi=300)
    plt.close()

    # Figure H6: Positive eigenspectrum and cumulative eigenmass
    fig, ax1 = plt.subplots(figsize=(6, 5))
    pos_eig = eigvals[:len(cum_mass)]
    dims = np.arange(1, len(cum_mass) + 1)
    color = "tab:blue"
    ax1.set_xlabel("Dimension Rank")
    ax1.set_ylabel(r"Eigenvalue $\lambda_r$", color=color)
    ax1.plot(dims, pos_eig, "o-", color=color, markersize=3)
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:orange"
    ax2.set_ylabel(r"Cumulative Eigenmass $C(r)$", color=color)
    ax2.plot(dims, cum_mass, "s--", color=color, markersize=3)
    ax2.axhline(0.90, color="gray", linestyle=":", alpha=0.7)
    ax2.axhline(0.95, color="red", linestyle=":", alpha=0.7)
    ax2.axhline(0.99, color="gray", linestyle=":", alpha=0.7)
    ax2.scatter([r90, r95, r99], [cum_mass[r90 - 1], cum_mass[r95 - 1], cum_mass[r99 - 1]], color="red", zorder=10)
    ax2.annotate(rf"95% ($r^\star={r95}$)", (r95, cum_mass[r95 - 1]), xytext=(r95 + 4, cum_mass[r95 - 1] - 0.08),
                 arrowprops=dict(arrowstyle="->", color="red"), fontsize=8, color="red")
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_ylim([0, 1.05])
    plt.title("Figure H6: Positive Eigenspectrum & Cumulative Mass")
    plt.tight_layout()
    plt.savefig(out_dir / "figure_h6_eigenspectrum_cumulative_mass.png", dpi=300)
    plt.close()

    # Figure H7: Compatibility-set overlap heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(overlap_mat, cmap="magma", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Jaccard Overlap $J(C_i, C_j)$")
    ax.set_xlabel("Compatibility Set Index")
    ax.set_ylabel("Compatibility Set Index")
    ax.set_title(f"Figure H7: Compatibility-Set Overlap ({len(overlap_mat)} Sets)")
    plt.tight_layout()
    plt.savefig(out_dir / "figure_h7_compatibility_overlap_heatmap.png", dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build QVCG-H Hilbert Atlas")
    parser.add_argument("--qvcg-dir", default="refine-logs/qvcg")
    parser.add_argument("--m3r-dir", default="refine-logs/qvcg/m3r_reference")
    parser.add_argument("--out-dir", default="refine-logs/qvcg/hilbert_atlas")
    parser.add_argument("--qap-perms", type=int, default=9999)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    qvcg_dir = Path(args.qvcg_dir)
    m3r_dir = Path(args.m3r_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Step 1: Auditing Frozen M3 and M3R Inputs ===")
    hashes = verify_inputs(qvcg_dir, m3r_dir)
    print("All input SHA-256 hashes successfully verified.")

    mmd2_m3 = np.load(qvcg_dir / "VCG_MMD_MATRIX.npy")
    mmd2_m3r = np.load(m3r_dir / "M3R_MMD_MATRIX.npy")
    pairs_m3 = pd.read_parquet(qvcg_dir / "VCG_REPSPAT_PAIR_TESTS.parquet")
    stats = pd.read_parquet(qvcg_dir / "VCG_DOMAIN_STATS.parquet").sort_values("domain_id")
    registry = pd.read_parquet(qvcg_dir / "VCG_SPATIAL_DOMAIN_REGISTRY.parquet")

    # Invariant audits
    validate_squared_distance_matrix(mmd2_m3)
    validate_squared_distance_matrix(mmd2_m3r)
    m3_vs_m3r_diff = float(np.max(np.abs(mmd2_m3 - mmd2_m3r)))
    if m3_vs_m3r_diff > 1e-12:
        raise RuntimeError(f"M3 vs M3R MMD mismatch: {m3_vs_m3r_diff:.6e} > 1e-12")
    if len(stats) != 64 or len(pairs_m3) != 2016:
        raise RuntimeError(f"Expected 64 domains and 2,016 pairs, got {len(stats)} domains and {len(pairs_m3)} pairs")
    print(f"Verified 64 domains, 2,016 pairs, symmetric zero-diagonal MMD (M3 vs M3R diff: {m3_vs_m3r_diff:.2e}).")

    print("=== Step 2: Constructing Centered Hilbert Gram Matrix & Coordinates ===")
    B = double_center_distance_matrix(mmd2_m3)
    eigvals, eigvecs, diag, Z_H = eigendecompose_hilbert_gram(B, tolerance=1e-6)
    positive_rank = diag["positive_rank"]
    print(f"Eigendecomposition complete. Negative eigenmass eta_- = {diag['negative_eigenmass']:.6e}")
    print(f"Eigenvalue classification: {diag['n_positive']} positive, {diag['n_numerical_zero']} numerical zero, {diag['n_negative']} negative")
    if positive_rank > 63:
        raise RuntimeError(f"Theoretical violation: positive rank {positive_rank} > 63")

    max_recon_err = reconstruct_squared_distances(mmd2_m3, Z_H)
    print(f"Max distance reconstruction error on full positive spectrum: {max_recon_err:.6e}")
    if max_recon_err > 1e-12:
        raise RuntimeError(f"HILBERT_DISTANCE_RECONSTRUCTION=FAIL: Error {max_recon_err:.6e} > 1e-12")

    pos_eigvals = eigvals[:positive_rank]
    cum_mass = np.cumsum(pos_eigvals) / np.sum(pos_eigvals)
    r90 = choose_dimension_by_eigenmass(eigvals, threshold=0.90)
    r95 = choose_dimension_by_eigenmass(eigvals, threshold=0.95)
    r99 = choose_dimension_by_eigenmass(eigvals, threshold=0.99)
    print(f"Dimensions: r90={r90}, r95={r95} (r*), r99={r99}, full_pos_rank={positive_rank} (<= 63)")

    # Export Geometry Artifacts
    np.save(out_dir / "HILBERT_GRAM_MATRIX.npy", B)

    tau = diag["eigenvalue_tolerance"]
    status = [
        "positive" if v > tau else ("negative" if v < -tau else "numerical_zero")
        for v in eigvals
    ]
    eigenspec_df = pd.DataFrame({
        "dimension": np.arange(1, len(eigvals) + 1),
        "eigenvalue": eigvals,
        "status": status,
        "is_positive": np.array(status) == "positive",
        "cumulative_positive_eigenmass": np.pad(cum_mass, (0, len(eigvals) - len(pos_eigvals)), constant_values=1.0),
    })
    eigenspec_df.to_csv(out_dir / "HILBERT_EIGENSPECTRUM.csv", index=False)

    coords_data = {"domain_id": np.arange(64)}
    for dim_idx in range(Z_H.shape[1]):
        coords_data[f"z_{dim_idx + 1}"] = Z_H[:, dim_idx]
    coords_data["centroid_vx"] = stats["centroid_vx"].to_numpy()
    coords_data["centroid_vy"] = stats["centroid_vy"].to_numpy()
    coords_data["centroid_vz"] = stats["centroid_vz"].to_numpy()
    coords_df = pd.DataFrame(coords_data)
    coords_df.to_parquet(out_dir / "HILBERT_ATLAS_COORDS.parquet", index=False)
    print("Geometry artifacts exported.")

    print("=== Step 3: Enumerating Compatibility Sets & Physical Recurrence ===")
    physical_adj = build_physical_adjacency(registry)
    cliques = enumerate_compatibility_cliques(pairs_m3, n_domains=64)
    print(f"Found {len(cliques)} maximal compatibility cliques.")

    centroids = stats[["centroid_vx", "centroid_vy", "centroid_vz"]].to_numpy(float)
    diff_centroids = centroids[:, None, :] - centroids[None, :, :]
    D_P_mat = np.linalg.norm(diff_centroids, axis=-1)
    D_H_mat = np.sqrt(np.maximum(mmd2_m3, 0.0))

    clique_records = []
    membership_records = []
    for c_id, clique in enumerate(cliques):
        for member in clique:
            membership_records.append({"domain_id": member, "clique_id": c_id})
        is_nonlocal, n_cc = classify_physical_recurrence(clique, physical_adj)
        # Compute diameter and distances
        clique_d_h = [D_H_mat[i, j] for i in clique for j in clique if i < j]
        clique_d_p = [D_P_mat[i, j] for i in clique for j in clique if i < j]
        clique_records.append({
            "clique_id": c_id,
            "size": len(clique),
            "n_phys_cc": n_cc,
            "is_nonlocal": is_nonlocal,
            "hilbert_diameter": float(np.max(clique_d_h)) if clique_d_h else 0.0,
            "hilbert_mean_dist": float(np.mean(clique_d_h)) if clique_d_h else 0.0,
            "hilbert_median_dist": float(np.median(clique_d_h)) if clique_d_h else 0.0,
            "physical_diameter": float(np.max(clique_d_p)) if clique_d_p else 0.0,
            "members": json.dumps(clique),
        })

    clique_df = pd.DataFrame(clique_records)
    clique_df.to_parquet(out_dir / "RSP_COMPATIBILITY_SETS.parquet", index=False)

    membership_df = pd.DataFrame(membership_records)
    membership_df.to_parquet(out_dir / "RSP_COMPATIBILITY_MEMBERSHIP.parquet", index=False)

    # Compute overlap matrix
    K = len(cliques)
    overlap_mat = np.zeros((K, K), dtype=float)
    clique_set_list = [set(c) for c in cliques]
    for i in range(K):
        for j in range(K):
            inter = len(clique_set_list[i] & clique_set_list[j])
            union = len(clique_set_list[i] | clique_set_list[j])
            overlap_mat[i, j] = inter / union if union > 0 else 0.0
    np.save(out_dir / "RSP_COMPATIBILITY_OVERLAP.npy", overlap_mat)

    n_nonlocal = int(clique_df["is_nonlocal"].sum())
    max_clique_size = int(clique_df["size"].max())
    print(f"Cover exported: {K} cliques, {n_nonlocal} nonlocal compatibility sets, max size {max_clique_size}.")

    print("=== Step 4: Physical-Functional Mapping & Fold Residuals ===")
    iu = np.triu_indices(64, k=1)
    d_p_vec = D_P_mat[iu]
    d_h_vec = D_H_mat[iu]

    iso_model = fit_physical_functional_isotonic(d_p_vec, d_h_vec)
    d_h_pred = iso_model.predict(d_p_vec)
    residuals = d_h_vec - d_h_pred

    # Build pair table aligned with pairs_m3
    # Build lookup for pairs
    pair_dict = {}
    for idx in range(len(iu[0])):
        u, v = int(iu[0][idx]), int(iu[1][idx])
        pair_dict[(u, v)] = {
            "d_p": float(d_p_vec[idx]),
            "d_h": float(d_h_vec[idx]),
            "d_h_pred": float(d_h_pred[idx]),
            "fold_residual": float(residuals[idx]),
        }

    phys_adj_set = set(tuple(sorted((u, v))) for u, v in physical_adj.edges())
    q75_dp = float(np.quantile(d_p_vec, 0.75))

    pairs_table_rows = []
    for row in pairs_m3.itertuples(index=False):
        d1 = int(getattr(row, "domain_1"))
        d2 = int(getattr(row, "domain_2"))
        u, v = min(d1, d2), max(d1, d2)
        geom = pair_dict[(u, v)]
        sim_edge = bool(getattr(row, "similarity_edge"))
        q_bh = float(getattr(row, "q_bh_imq"))
        is_adj = (u, v) in phys_adj_set
        is_distant = geom["d_p"] >= q75_dp
        res = geom["fold_residual"]
        is_supported_fold = bool(res < 0.0 and sim_edge)
        is_local_boundary = bool(res > 0.0 and is_adj)

        pairs_table_rows.append({
            "domain_1": d1,
            "domain_2": d2,
            "d_p": geom["d_p"],
            "d_h": geom["d_h"],
            "d_h_pred": geom["d_h_pred"],
            "fold_residual": res,
            "q_bh_imq": q_bh,
            "similarity_edge": sim_edge,
            "is_physically_adjacent": is_adj,
            "is_physically_distant_q75": is_distant,
            "is_supported_fold": is_supported_fold,
            "is_local_boundary": is_local_boundary,
        })

    pair_table_df = pd.DataFrame(pairs_table_rows)
    pair_table_df.to_parquet(out_dir / "PHYSICAL_FUNCTIONAL_PAIR_TABLE.parquet", index=False)

    # Rankings
    fold_ranking = pair_table_df[pair_table_df.is_supported_fold].sort_values("fold_residual", ascending=True)
    fold_ranking.to_parquet(out_dir / "FUNCTIONAL_FOLD_RANKING.parquet", index=False)

    boundary_ranking = pair_table_df[pair_table_df.is_local_boundary].sort_values("fold_residual", ascending=False)
    boundary_ranking.to_parquet(out_dir / "FUNCTIONAL_BOUNDARY_RANKING.parquet", index=False)
    print(f"Folding exported: {len(fold_ranking)} supported folds, {len(boundary_ranking)} local boundaries.")

    print(f"=== Step 5: Running QAP Matrix Permutation Test (B={args.qap_perms}) ===")
    qap_results = qap_matrix_test(D_P_mat, D_H_mat, n_perm=args.qap_perms, seed=args.seed)
    with (out_dir / "QAP_PHYSICAL_FUNCTIONAL.json").open("w") as f:
        json.dump(qap_results, f, indent=2)
    print(f"QAP test finished: observed rho = {qap_results['rho_obs']:.4f}, p_qap = {qap_results['p_qap']:.6e}")

    print("=== Step 6: Generating Primary Figures H1-H7 ===")
    plot_primary_figures(
        out_dir=out_dir,
        centroids=centroids,
        Z_H=Z_H,
        D_P_mat=D_P_mat,
        D_H_mat=D_H_mat,
        d_p_vec=d_p_vec,
        d_h_vec=d_h_vec,
        d_h_pred=d_h_pred,
        pairs_df=pair_table_df,
        eigvals=eigvals,
        cum_mass=cum_mass,
        r90=r90,
        r95=r95,
        r99=r99,
        overlap_mat=overlap_mat,
        physical_adj=physical_adj,
    )
    print("Figures H1-H7 successfully generated.")

    print("=== Step 7: Writing Initial Summary Artifact ===")
    summary = {
        "run_id": "qvcgh_m1_discovery_f17",
        "source_m3_sha256": hashes["VCG_MMD_MATRIX.npy"],
        "source_m3r_sha256": hashes["M3R_MMD_MATRIX.npy"],
        "n_domains": 64,
        "n_pairs": 2016,
        "kernel": "IMQ",
        "kernel_parameter": 1.0,
        "lambda_min": diag["lambda_min"],
        "lambda_max": diag["lambda_max"],
        "lambda_abs_min": diag["lambda_abs_min"],
        "eigenvalue_tolerance": diag["eigenvalue_tolerance"],
        "n_positive": diag["n_positive"],
        "n_numerical_zero": diag["n_numerical_zero"],
        "n_negative": diag["n_negative"],
        "positive_rank": diag["positive_rank"],
        "negative_eigenmass": diag["negative_eigenmass"],
        "r90": int(r90),
        "r95": int(r95),
        "r99": int(r99),
        "distance_reconstruction_max_error": float(max_recon_err),
        "qap_rho": float(qap_results["rho_obs"]),
        "qap_p": float(qap_results["p_qap"]),
        "n_maximal_cliques": int(K),
        "maximum_clique_size": int(max_clique_size),
        "n_nonlocal_compatibility_sets": int(n_nonlocal),
        "physical_distance_q75": float(q75_dp),
        "fold_residual_min": float(residuals.min()),
        "fold_residual_max": float(residuals.max()),
        "clinical_labels_used": False,
    }
    with (out_dir / "HILBERT_ATLAS_SUMMARY.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Artifact generation complete. Summary written to {out_dir / 'HILBERT_ATLAS_SUMMARY.json'}")


if __name__ == "__main__":
    main()
