"""Milestone M2: Discover M Contiguous Spatial VCG Domains via 3D Lattice & CAHC.

Adheres strictly to PRD Sections 15, 16, 17, 18, 19, 20:
- Standardizes coordinates using training-only parameters.
- Discretizes onto 24^3 occupancy lattice.
- Filters eligible voxels based on support (n_sample_min, n_patient_min).
- Constructs 26-neighbor spatial connectivity graph.
- Summarizes voxel dynamic attributes: a_c = [median(s), median(rho), median(kappa)].
- Fits Constrained Agglomerative Hierarchical Clustering (CAHC) to discover M=64 contiguous domains.
- Serializes VCG_SPATIAL_DOMAIN_REGISTRY.parquet and VCG_DOMAIN_STATS.parquet.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from rep_stat_ecg.src.motifs.spatial_domains import CoordinateStandardizer, VoxelGrid3D


def main():
    parser = argparse.ArgumentParser(description="Discover contiguous VCG spatial domains.")
    parser.add_argument("--micro-table", type=str, default="refine-logs/qvcg/VCG_MICROSTATE_TABLE.parquet")
    parser.add_argument("--std-json", type=str, default="refine-logs/qvcg/VCG_COORDINATE_STANDARDIZER.json")
    parser.add_argument("--out-dir", type=str, default="refine-logs/qvcg")
    parser.add_argument("--grid-dim", type=int, default=24)
    parser.add_argument("--n-domains", type=int, default=64)
    parser.add_argument("--n-sample-min", type=int, default=30)
    parser.add_argument("--n-patient-min", type=int, default=5)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading microstates from {args.micro_table}...")
    df_micro = pd.read_parquet(args.micro_table)
    with open(args.std_json) as f:
        std_dict = json.load(f)

    standardizer = CoordinateStandardizer()
    standardizer.median = np.array(std_dict["median"], dtype=np.float32)
    standardizer.mad = np.array(std_dict["mad"], dtype=np.float32)
    standardizer.is_fitted = True

    v_physical = df_micro[["v_x", "v_y", "v_z"]].values
    xi = df_micro[["s", "rho", "kappa"]].values
    patient_ids = df_micro["patient_id"].astype(str).tolist()

    v_tilde = standardizer.transform(v_physical)

    grid = VoxelGrid3D(
        grid_dim=args.grid_dim,
        coord_min=-3.0,
        coord_max=3.0,
        n_sample_min=args.n_sample_min,
        n_patient_min=args.n_patient_min,
    )

    print(f"Fitting {args.n_domains} contiguous spatial domains on {args.grid_dim}^3 lattice...")
    voxel_to_domain = grid.fit_spatial_domains(
        v_tilde=v_tilde,
        xi=xi,
        patient_ids=patient_ids,
        n_domains=args.n_domains,
        linkage="ward",
    )

    actual_m = len(grid.domain_to_voxels)
    print(f"Discovered {actual_m} contiguous spatial domains across {len(grid.eligible_voxels):,} eligible voxels.")

    # Serialize Voxel -> Domain registry
    registry_rows = []
    for vx, dom_id in voxel_to_domain.items():
        ix, iy, iz = grid.linear_to_3d_index(vx)
        center_x = float(grid.bin_centers[ix])
        center_y = float(grid.bin_centers[iy])
        center_z = float(grid.bin_centers[iz])
        registry_rows.append({
            "voxel_id": int(vx),
            "ix": int(ix),
            "iy": int(iy),
            "iz": int(iz),
            "center_tilde_x": center_x,
            "center_tilde_y": center_y,
            "center_tilde_z": center_z,
            "domain_id": int(dom_id),
        })

    df_registry = pd.DataFrame(registry_rows)
    registry_parquet = out_dir / "VCG_SPATIAL_DOMAIN_REGISTRY.parquet"
    df_registry.to_parquet(registry_parquet, index=False)
    print(f"Saved domain registry to {registry_parquet}")

    # Map microstates to domains and compute statistics
    voxel_indices, in_bounds = grid.point_to_voxel_index(v_tilde)
    domain_col = np.full(len(df_micro), fill_value=-1, dtype=np.int32)
    for i in range(len(df_micro)):
        if in_bounds[i]:
            vx = voxel_indices[i]
            if vx in voxel_to_domain:
                domain_col[i] = voxel_to_domain[vx]

    df_micro["domain_id"] = domain_col
    assigned_mask = df_micro["domain_id"] >= 0
    assigned_rate = float(np.mean(assigned_mask))
    print(f"Microstate domain assignment rate: {assigned_rate:.1%}")

    stats_rows = []
    for d in range(actual_m):
        d_mask = df_micro["domain_id"] == d
        d_df = df_micro[d_mask]
        n_pts = len(d_df)
        n_pats = d_df["patient_id"].nunique() if n_pts > 0 else 0
        mean_v = d_df[["v_x", "v_y", "v_z"]].mean().values if n_pts > 0 else np.zeros(3)
        mean_xi = d_df[["s", "rho", "kappa"]].mean().values if n_pts > 0 else np.zeros(3)

        stats_rows.append({
            "domain_id": int(d),
            "num_voxels": len(grid.domain_to_voxels.get(d, [])),
            "num_microstates": int(n_pts),
            "num_patients": int(n_pats),
            "centroid_vx": float(mean_v[0]),
            "centroid_vy": float(mean_v[1]),
            "centroid_vz": float(mean_v[2]),
            "mean_speed": float(mean_xi[0]),
            "mean_radial_vel": float(mean_xi[1]),
            "mean_curvature": float(mean_xi[2]),
        })

    df_stats = pd.DataFrame(stats_rows)
    stats_parquet = out_dir / "VCG_DOMAIN_STATS.parquet"
    df_stats.to_parquet(stats_parquet, index=False)
    print(f"Saved domain statistics to {stats_parquet}")


if __name__ == "__main__":
    main()
