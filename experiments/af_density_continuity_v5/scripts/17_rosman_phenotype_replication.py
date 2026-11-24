#!/usr/bin/env python3
"""Script 17: Rosman 2026 Methodological Replication (R01) & Horizon Migration (R02).
Generates:
  - results/R01_rosman_patient_matrix.csv
  - results/R01_rosman_recording_matrix.csv
  - results/R01_rosman_density_comparison.csv
  - results/R02_horizon_migration_matrix.csv
  - results/R02_horizon_agreement_metrics.csv
  - results/R02_start_time_sensitivity.csv
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir))

from src.rosman_phenotypes import (
    ROSMAN_TIERS,
    assign_rosman_tier,
    compute_rosman_phenotype_matrix,
    compute_weighted_cohen_kappa,
    compute_rank_stability
)
from src.density_continuous import compute_continuous_density

def main():
    features_dir = root_dir / "features"
    results_dir = root_dir / "results"
    manifests_dir = root_dir / "manifests"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ROSMAN 2026 REPLICATION: R01 (4x4 PHENOTYPES) & R02 (HORIZON MIGRATION)")
    print("=" * 80)

    # 1. Load density reference features and windows
    df_feats = pd.read_parquet(features_dir / "density_reference_features.parquet")
    df_runs = pd.read_parquet(manifests_dir / "canonical_af_runs.parquet")
    df_intervals = pd.read_parquet(manifests_dir / "canonical_rhythm_intervals.parquet")

    # Primary 12h horizon for benchmark phenotype matrix
    df_12h = df_feats[
        (df_feats["horizon"] == "12h") &
        (df_feats["window_type"] == "OBSERVED_MONITORING_ANCHORED")
    ].copy()

    # R01: Patient-level and Recording-level 4x4 Matrices
    print("\n--- R01: Computing 4x4 Burden-Density Phenotype Matrices ---")
    rec_matrix, rec_diag = compute_rosman_phenotype_matrix(
        df_12h, burden_col="af_burden", density_col="density_continuous"
    )
    pat_matrix, pat_diag = compute_rosman_phenotype_matrix(
        df_12h, burden_col="af_burden", density_col="density_continuous", group_col="patient_id"
    )

    rec_matrix.to_csv(results_dir / "R01_rosman_recording_matrix.csv")
    pat_matrix.to_csv(results_dir / "R01_rosman_patient_matrix.csv")

    df_comp = pd.DataFrame(pat_diag["density_comparison"])
    df_comp.to_csv(results_dir / "R01_rosman_density_comparison.csv", index=False)

    print(f"Patient-level 4x4 Matrix (N={pat_diag['n_valid_joint_instances']} valid, N={pat_diag['n_undefined_instances']} undefined):")
    print(pat_matrix.to_string())

    print("\nComparison with Published Rosman et al. (2026) Marginal Density Distribution:")
    print(df_comp.to_string(index=False))

    # R02: Observation-Horizon Phenotype Migration
    print("\n--- R02: Observation-Horizon Phenotype Migration Across Nested Horizons ---")
    # We evaluate nested horizons anchored at t0: 1h, 3h, 6h, 12h, 24h, 48h, 72h, full_recording
    # First, let's build nested windows from canonical intervals for all multi-hour recordings
    target_horizons_s = {
        "1h": 3600.0,
        "3h": 10800.0,
        "6h": 21600.0,
        "12h": 43200.0,
        "24h": 86400.0,
        "48h": 172800.0,
        "72h": 259200.0
    }

    # Group by recording and find t0 and t_end
    rec_meta = df_intervals.groupby(["dataset", "record_id", "patient_id"]).agg(
        t_start=("start_time_s", "min"),
        t_end=("end_time_s", "max")
    ).reset_index()
    rec_meta["monitoring_dur_s"] = rec_meta["t_end"] - rec_meta["t_start"]

    nested_records = []
    print(f"Evaluating nested horizons on {len(rec_meta)} recordings...")

    for idx, rec in rec_meta.iterrows():
        ds = rec["dataset"]
        r_id = rec["record_id"]
        p_id = rec["patient_id"]
        t0 = rec["t_start"]
        max_dur = rec["monitoring_dur_s"]

        rec_rhythm = df_intervals[(df_intervals["dataset"] == ds) & (df_intervals["record_id"] == r_id)]
        rec_af = rec_rhythm[rec_rhythm["rhythm_label_harmonized"] == "AF"]

        # Full recording measurement
        full_af_s = rec_af["duration_s"].sum()
        full_b = full_af_s / max_dur if max_dur > 0 else 0.0
        full_runs = [(r["start_time_s"] - t0, r["end_time_s"] - t0) for _, r in rec_af.iterrows()]
        full_runs.sort()
        full_d = compute_continuous_density(full_runs, max_dur) if (0.0 < full_b < 1.0) else np.nan

        row = {
            "dataset": ds,
            "record_id": r_id,
            "patient_id": p_id,
            "max_monitoring_dur_h": max_dur / 3600.0,
            "full_burden": full_b,
            "full_density": full_d,
            "full_burden_tier": assign_rosman_tier(full_b),
            "full_density_tier": assign_rosman_tier(full_d)
        }

        # Sub-horizons
        for h_name, h_dur in target_horizons_s.items():
            if max_dur >= h_dur:
                w_start = t0
                w_end = t0 + h_dur
                # Clip AF runs to [w_start, w_end]
                sub_runs = []
                sub_af_s = 0.0
                for _, r in rec_af.iterrows():
                    cs = max(w_start, float(r["start_time_s"]))
                    ce = min(w_end, float(r["end_time_s"]))
                    if ce > cs:
                        sub_runs.append((cs - w_start, ce - w_start))
                        sub_af_s += (ce - cs)
                sub_b = sub_af_s / h_dur
                sub_d = compute_continuous_density(sub_runs, h_dur) if (0.0 < sub_b < 1.0) else np.nan
                row[f"{h_name}_burden"] = sub_b
                row[f"{h_name}_density"] = sub_d
                row[f"{h_name}_burden_tier"] = assign_rosman_tier(sub_b)
                row[f"{h_name}_density_tier"] = assign_rosman_tier(sub_d)
            else:
                row[f"{h_name}_burden"] = np.nan
                row[f"{h_name}_density"] = np.nan
                row[f"{h_name}_burden_tier"] = "Not_Supported"
                row[f"{h_name}_density_tier"] = "Not_Supported"

        nested_records.append(row)

    df_nested = pd.DataFrame(nested_records)

    # Calculate agreement metrics and migration
    agreement_records = []
    migration_records = []

    # Compare 12h vs 24h on identical recordings (paired subset supporting >= 24h)
    p24 = df_nested[df_nested["24h_burden"].notna()].copy()
    print(f"Recordings supporting paired >= 24h comparisons: {len(p24)}")

    for h in ["1h", "3h", "6h", "12h", "24h"]:
        sub_paired = df_nested[df_nested[f"{h}_burden"].notna() & df_nested["full_burden"].notna()].copy()
        if len(sub_paired) < 5:
            continue

        # Disagreement
        delta_b = (sub_paired[f"{h}_burden"] - sub_paired["full_burden"]).abs()
        dens_valid = sub_paired[sub_paired[f"{h}_density"].notna() & sub_paired["full_density"].notna()]
        delta_d = (dens_valid[f"{h}_density"] - dens_valid["full_density"]).abs() if len(dens_valid) > 0 else pd.Series([np.nan])

        # Rank stability
        rho_b, p_b = compute_rank_stability(sub_paired[f"{h}_burden"], sub_paired["full_burden"])
        rho_d, p_d = compute_rank_stability(dens_valid[f"{h}_density"], dens_valid["full_density"]) if len(dens_valid) >= 5 else (np.nan, np.nan)

        # Weighted kappa for density tiers
        dens_tier_valid = sub_paired[
            sub_paired[f"{h}_density_tier"].isin(ROSMAN_TIERS) &
            sub_paired["full_density_tier"].isin(ROSMAN_TIERS)
        ]
        kappa_lin = compute_weighted_cohen_kappa(
            dens_tier_valid["full_density_tier"].tolist(),
            dens_tier_valid[f"{h}_density_tier"].tolist(),
            weight="linear"
        ) if len(dens_tier_valid) >= 5 else np.nan

        kappa_quad = compute_weighted_cohen_kappa(
            dens_tier_valid["full_density_tier"].tolist(),
            dens_tier_valid[f"{h}_density_tier"].tolist(),
            weight="quadratic"
        ) if len(dens_tier_valid) >= 5 else np.nan

        # High-density saturation fraction
        sat_frac = (dens_valid[f"{h}_density"] >= 0.90).mean() if len(dens_valid) > 0 else np.nan

        agreement_records.append({
            "horizon": h,
            "n_recordings_paired": len(sub_paired),
            "n_density_valid_paired": len(dens_valid),
            "median_abs_burden_diff": float(delta_b.median()),
            "mean_abs_burden_diff": float(delta_b.mean()),
            "median_abs_density_diff": float(delta_d.median()) if delta_d.notna().any() else np.nan,
            "mean_abs_density_diff": float(delta_d.mean()) if delta_d.notna().any() else np.nan,
            "burden_rank_stability_rho": rho_b,
            "density_rank_stability_rho": rho_d,
            "density_weighted_kappa_linear": kappa_lin,
            "density_weighted_kappa_quadratic": kappa_quad,
            "high_density_saturation_fraction": sat_frac
        })

    df_agree = pd.DataFrame(agreement_records)
    df_agree.to_csv(results_dir / "R02_horizon_agreement_metrics.csv", index=False)
    print("\nObservation-Horizon Agreement and Saturation Metrics:")
    print(df_agree.to_string(index=False))

    # Category migration: 12h tier -> full recording tier
    valid_migration = df_nested[
        df_nested["12h_density_tier"].isin(ROSMAN_TIERS) &
        df_nested["full_density_tier"].isin(ROSMAN_TIERS)
    ]
    mig_crosstab = pd.crosstab(
        valid_migration["12h_density_tier"],
        valid_migration["full_density_tier"],
        rownames=["12h_Density"],
        colnames=["Full_Density"]
    ).reindex(index=ROSMAN_TIERS, columns=ROSMAN_TIERS, fill_value=0)
    mig_crosstab.to_csv(results_dir / "R02_horizon_migration_matrix.csv")
    print("\n12-Hour to Full-Recording Density Category Migration Matrix:")
    print(mig_crosstab.to_string())

    # Start-time sensitivity on 12h horizon: t0 vs t0 + 6h
    start_time_records = []
    p18 = df_nested[df_nested["max_monitoring_dur_h"] >= 18.0].copy()
    for idx, rec in p18.iterrows():
        ds = rec["dataset"]
        r_id = rec["record_id"]
        p_id = rec["patient_id"]
        t0 = rec_meta.loc[(rec_meta["record_id"] == r_id), "t_start"].iloc[0]

        rec_af = df_intervals[(df_intervals["dataset"] == ds) & (df_intervals["record_id"] == r_id) & (df_intervals["rhythm_label_harmonized"] == "AF")]

        # Window 1: [t0, t0 + 12h]
        w1_s, w1_e = t0, t0 + 43200.0
        # Window 2: [t0 + 6h, t0 + 18h]
        w2_s, w2_e = t0 + 21600.0, t0 + 64800.0

        for w_label, ws, we in [("win_t0", w1_s, w1_e), ("win_t0_plus_6h", w2_s, w2_e)]:
            sub_runs = []
            sub_af_s = 0.0
            for _, r in rec_af.iterrows():
                cs = max(ws, float(r["start_time_s"]))
                ce = min(we, float(r["end_time_s"]))
                if ce > cs:
                    sub_runs.append((cs - ws, ce - ws))
                    sub_af_s += (ce - cs)
            b = sub_af_s / 43200.0
            d = compute_continuous_density(sub_runs, 43200.0) if (0.0 < b < 1.0) else np.nan
            start_time_records.append({
                "dataset": ds,
                "record_id": r_id,
                "patient_id": p_id,
                "start_anchor": w_label,
                "af_burden": b,
                "density_continuous": d,
                "density_tier": assign_rosman_tier(d)
            })

    df_st = pd.DataFrame(start_time_records)
    # Pivot to compare anchors
    piv_st = df_st.pivot(index=["dataset", "record_id", "patient_id"], columns="start_anchor", values=["af_burden", "density_continuous", "density_tier"]).reset_index()
    piv_st.columns = [f"{c[0]}_{c[1]}" if c[1] else c[0] for c in piv_st.columns]
    piv_st["abs_delta_burden"] = (piv_st["af_burden_win_t0"] - piv_st["af_burden_win_t0_plus_6h"]).abs()
    piv_st["abs_delta_density"] = (piv_st["density_continuous_win_t0"] - piv_st["density_continuous_win_t0_plus_6h"]).abs()
    piv_st["same_density_tier"] = piv_st["density_tier_win_t0"] == piv_st["density_tier_win_t0_plus_6h"]

    piv_st.to_csv(results_dir / "R02_start_time_sensitivity.csv", index=False)
    print(f"\nStart-Time Sensitivity (t0 vs t0+6h on 12h horizon, N={len(piv_st)} recordings):")
    print(f"Mean |Delta Burden|: {piv_st['abs_delta_burden'].mean():.4f} | Mean |Delta Density|: {piv_st['abs_delta_density'].mean():.4f}")
    print(f"Percentage in identical density tier: {piv_st['same_density_tier'].mean()*100:.1f}%")

if __name__ == "__main__":
    main()
