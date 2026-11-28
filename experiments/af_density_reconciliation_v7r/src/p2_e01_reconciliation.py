"""Priority 2: Independent E01 Reconstruction, Risk vs Physical Time & LOPO Sensitivity (V7-R)."""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

OUT_DIR = Path("experiments/af_density_reconciliation_v7r/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fit_poisson_cluster(df_target, df_ref, exposure_col, outcome_col_t="resolved_brief_af_count", outcome_col_r="brief_af_onset_count"):
    t = df_target[df_target[exposure_col] > 0].copy()
    r = df_ref[df_ref[exposure_col] > 0].copy()
    t["y"] = t[outcome_col_t].astype(float)
    r["y"] = r[outcome_col_r].astype(float)
    t["is_target"] = 1.0
    r["is_target"] = 0.0
    comb = pd.concat([
        t[["patient_id", "is_target", exposure_col, "y"]],
        r[["patient_id", "is_target", exposure_col, "y"]],
    ], ignore_index=True).sort_values("patient_id")
    if comb.empty or comb["is_target"].nunique() < 2:
        return {"rr": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p_val": np.nan, "n_pts": 0}
    offset = np.log(comb[exposure_col].values)
    X = sm.add_constant(comb[["is_target"]])
    glm = sm.GLM(comb["y"], X, family=sm.families.Poisson(), offset=offset).fit(
        cov_type="cluster", cov_kwds={"groups": comb["patient_id"]}
    )
    b = float(glm.params["is_target"])
    se = float(glm.bse["is_target"])
    return {
        "rr": float(np.exp(b)),
        "ci_lo": float(np.exp(b - 1.96 * se)),
        "ci_hi": float(np.exp(b + 1.96 * se)),
        "p_val": float(glm.pvalues["is_target"]),
        "n_pts": int(comb["patient_id"].nunique()),
        "target_events": int(t["y"].sum()),
        "ref_events": int(r["y"].sum()),
        "target_exp_h": round(float(t[exposure_col].sum()), 4),
        "ref_exp_h": round(float(r[exposure_col].sum()), 4),
        "target_rate_per_h": round(float(t["y"].sum() / t[exposure_col].sum()), 4),
        "ref_rate_per_h": round(float(r["y"].sum() / r[exposure_col].sum()), 4),
    }


def main():
    print("=== Priority 2: E01 Event-Level Reconstruction & LOPO Sensitivity ===")
    nb_iv = pd.read_parquet("experiments/af_density_temporal_phenotypes_v7/results/E01_neighbourhood_intervals.parquet")
    rem_iv = pd.read_parquet("experiments/af_density_temporal_phenotypes_v7/results/E01_remote_comparison_manifest.parquet")
    idx_ep = pd.read_parquet("experiments/af_density_temporal_phenotypes_v7/results/E01_index_episode_manifest.parquet")
    runs = pd.read_parquet("experiments/af_density_continuity_v5/manifests/canonical_af_runs.parquet")
    sens_orig = pd.read_csv("experiments/af_density_temporal_phenotypes_v7/results/E01_sensitivity_analysis.csv")

    print("Original E01_sensitivity_analysis.csv:")
    print(sens_orig.to_string(index=False))

    # Build exposure columns cleanly in hours
    nb_iv["physical_time_h"] = nb_iv["known_rhythm_s"] / 3600.0
    rem_iv["physical_time_h"] = rem_iv["known_rhythm_s"] / 3600.0

    # Aggregate 0-15m composite windows per index episode (0-5m + 5-15m)
    nb_15 = nb_iv[nb_iv["band_name"].isin(["pre_0_5m", "pre_5_15m", "post_0_5m", "post_5_15m"])].copy()
    nb_15["comp_side"] = np.where(nb_15["side"] == "pre", "pre_0_15m", "post_0_15m")
    nb_15_agg = nb_15.groupby(["dataset", "patient_id", "record_id", "index_episode_id", "comp_side"], as_index=False).agg(
        resolved_brief_af_count=("resolved_brief_af_count", "sum"),
        risk_time_h=("risk_time_h", "sum"),
        physical_time_h=("physical_time_h", "sum"),
    )

    rows = []
    for ds in ["ALL", "LTAFDB", "SHDB-AF"]:
        sub_agg = nb_15_agg if ds == "ALL" else nb_15_agg[nb_15_agg["dataset"] == ds]
        sub_rem = rem_iv if ds == "ALL" else rem_iv[rem_iv["dataset"] == ds]
        pre15 = sub_agg[sub_agg["comp_side"] == "pre_0_15m"]
        post15 = sub_agg[sub_agg["comp_side"] == "post_0_15m"]

        for exp in ["risk_time_h", "physical_time_h"]:
            r_pre = fit_poisson_cluster(pre15, sub_rem, exp)
            rows.append({"dataset": ds, "comparison": "pre_0_15m_vs_remote", "exposure_type": exp, **r_pre})
            r_post = fit_poisson_cluster(post15, sub_rem, exp)
            rows.append({"dataset": ds, "comparison": "post_0_15m_vs_remote", "exposure_type": exp, **r_post})
            r_pvp = fit_poisson_cluster(post15, pre15, exp, outcome_col_r="resolved_brief_af_count")
            rows.append({"dataset": ds, "comparison": "post_0_15m_vs_pre_0_15m", "exposure_type": exp, **r_pvp})

    rec_df = pd.DataFrame(rows)
    rec_df.to_csv(OUT_DIR / "V7R_e01_reconciled.csv", index=False)
    print("\nReconstructed E01 Rate Ratios:")
    print(rec_df[["dataset", "comparison", "exposure_type", "rr", "ci_lo", "ci_hi", "p_val", "target_events", "ref_events", "target_exp_h", "ref_exp_h"]].to_string(index=False))

    # Leave-One-Patient-Out (LOPO) sensitivity across top prolific brief-run LTAFDB contributors
    ltaf_brief = runs[(runs["dataset"] == "LTAFDB") & (runs["duration_s"] < 30.0)]
    top_pts = ltaf_brief.groupby("patient_id").size().sort_values(ascending=False).head(10).index.tolist()

    lopo_rows = []
    for drop_label, drop_pts in [("None (Full Pooled)", []), ("Exclude Top-1 LTAFDB Pt", top_pts[:1]),
                                 ("Exclude Top-3 LTAFDB Pts", top_pts[:3]), ("Exclude Top-5 LTAFDB Pts", top_pts[:5]),
                                 ("Exclude Top-10 LTAFDB Pts", top_pts[:10])]:
        sub_agg = nb_15_agg[~nb_15_agg["patient_id"].isin(drop_pts)]
        sub_rem = rem_iv[~rem_iv["patient_id"].isin(drop_pts)]
        pre15 = sub_agg[sub_agg["comp_side"] == "pre_0_15m"]
        post15 = sub_agg[sub_agg["comp_side"] == "post_0_15m"]
        r_pre = fit_poisson_cluster(pre15, sub_rem, "risk_time_h")
        r_post = fit_poisson_cluster(post15, sub_rem, "risk_time_h")
        r_pvp = fit_poisson_cluster(post15, pre15, "risk_time_h", outcome_col_r="resolved_brief_af_count")
        lopo_rows.append({
            "sensitivity_scenario": drop_label,
            "excluded_patients": ",".join(drop_pts) if drop_pts else "None",
            "pre_vs_remote_rr": round(r_pre["rr"], 3),
            "pre_vs_remote_p": r_pre["p_val"],
            "post_vs_remote_rr": round(r_post["rr"], 3),
            "post_vs_remote_p": r_post["p_val"],
            "post_vs_pre_rr": round(r_pvp["rr"], 3),
            "post_vs_pre_p": r_pvp["p_val"],
        })
    lopo_df = pd.DataFrame(lopo_rows)
    lopo_df.to_csv(OUT_DIR / "V7R_e01_lopo_sensitivity.csv", index=False)
    print("\nE01 LOPO Prolific-Patient Sensitivity:")
    print(lopo_df.to_string(index=False))


if __name__ == "__main__":
    main()
