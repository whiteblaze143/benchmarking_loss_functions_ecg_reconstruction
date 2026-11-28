"""AF-DENSITY V8: E03 Common Support + E04 Annotation Harmonization + E05 Prospective Blocks + E06 Within/Between Patient."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

sys.path.insert(0, "experiments/af_density_temporal_phenotypes_v7/src")
from density_continuous import compute_continuous_density

OUT_RES = Path("experiments/af_density_v8/results")
V5_MAN = Path("experiments/af_density_continuity_v5/manifests")
V7P_DIR = Path("experiments/af_density_prediction_v7p")


def merge_ivs(ivs):
    if not ivs:
        return []
    s = sorted((float(a), float(b)) for a, b in ivs if b > a + 1e-9)
    if not s:
        return []
    m = [[s[0][0], s[0][1]]]
    for a, b in s[1:]:
        if a <= m[-1][1] + 1e-9:
            m[-1][1] = max(m[-1][1], b)
        else:
            m.append([a, b])
    return [(x, y) for x, y in m]


def apply_harmonization(af_ivs, min_af_s=0.0, min_gap_s=0.0, known_T=None):
    """Delete short AF runs (< min_af_s) first, then bridge short non-AF gaps (< min_gap_s)."""
    ivs = merge_ivs(af_ivs)
    if min_af_s > 0.0:
        ivs = [(a, b) for a, b in ivs if (b - a) >= min_af_s]
    if min_gap_s > 0.0 and ivs:
        bridged = [list(ivs[0])]
        for a, b in ivs[1:]:
            if (a - bridged[-1][1]) < min_gap_s:
                bridged[-1][1] = b
            else:
                bridged.append([a, b])
        ivs = [(a, b) for a, b in bridged]
    if known_T is not None:
        ivs = [(max(0.0, a), min(known_T, b)) for a, b in ivs if min(known_T, b) > max(0.0, a) + 1e-9]
    return merge_ivs(ivs)


def compute_axes(af_ivs, T_s):
    """Compute canonical 4 axes: exposure (burden), concentration (density), continuity (s_max/M), transition dynamics (lambda_on, lambda_off)."""
    ivs = merge_ivs([(max(0.0, a), min(T_s, b)) for a, b in af_ivs])
    M = sum(b - a for a, b in ivs)
    B = M / T_s if T_s > 0 else np.nan
    non_af_s = T_s - M
    dens = compute_continuous_density(ivs, T_s) if (0 < B < 1.0 - 1e-9 and ivs) else np.nan
    s_max = max((b - a for a, b in ivs), default=0.0)
    continuity = (s_max / M) if M > 0 else np.nan
    n_runs = len(ivs)
    # Observed onsets (not starting at 0) and observed terminations (not ending at T_s)
    n_onsets = sum(1 for a, _ in ivs if a > 1e-6)
    n_terms = sum(1 for _, b in ivs if b < T_s - 1e-6)
    lam_on_per_nonaf_h = (n_onsets / (non_af_s / 3600.0)) if non_af_s > 60.0 else np.nan
    lam_off_per_af_h = (n_terms / (M / 3600.0)) if M > 60.0 else np.nan
    lam_on_per_elapsed_h = n_onsets / (T_s / 3600.0)
    return {
        "af_s": M, "non_af_s": non_af_s, "burden": B, "density": dens if dens is not None else np.nan,
        "longest_ep_s": s_max, "longest_share": continuity, "n_runs": n_runs,
        "n_onsets": n_onsets, "n_terms": n_terms,
        "lam_on_risk_h": lam_on_per_nonaf_h, "lam_off_af_h": lam_off_per_af_h,
        "lam_on_phys_h": lam_on_per_elapsed_h,
    }


def extract_known_timeline(dataset, rid, rhy, runs, cal_grp, rec_row):
    if dataset in ["LTAFDB", "SHDB-AF"]:
        sub_r = rhy[(rhy["dataset"] == dataset) & (rhy["record_id"].astype(str) == rid) & (rhy["rhythm_known"])]
        known_ivs = sorted(zip(sub_r["start_time_s"], sub_r["end_time_s"]))
        af_raw = list(zip(runs[(runs["dataset"] == dataset) & (runs["record_id"].astype(str) == rid)]["start_time_s"],
                          runs[(runs["dataset"] == dataset) & (runs["record_id"].astype(str) == rid)]["end_time_s"]))
    else:
        raw_end = rec_row.raw_samples / rec_row.fs
        if rid in cal_grp.groups:
            cs = cal_grp.get_group(rid)
            ex_ivs = [(s / 200.0, e / 200.0) for s, e in zip(cs["excluded_start_sample"], cs["excluded_end_sample_exclusive"])]
        else:
            ex_ivs = [(0.0, min(raw_end, 30.0))]
        # Subtract exclusions cleanly
        cur_k = [(0.0, raw_end)]
        for ex_a, ex_b in merge_ivs(ex_ivs):
            nxt = []
            for a, b in cur_k:
                if b <= ex_a + 1e-9 or a >= ex_b - 1e-9:
                    nxt.append((a, b))
                else:
                    if a < ex_a - 1e-9:
                        nxt.append((a, ex_a))
                    if b > ex_b + 1e-9:
                        nxt.append((ex_b, b))
            cur_k = nxt
        known_ivs = merge_ivs(cur_k)
        sub_af = runs[(runs["dataset"] == "IRIDIA-AF") & (runs["record_id"].astype(str) == rid)]
        cur_af = list(zip(sub_af["start_time_s"], sub_af["end_time_s"]))
        for ex_a, ex_b in merge_ivs(ex_ivs):
            nxt = []
            for a, b in cur_af:
                if b <= ex_a + 1e-9 or a >= ex_b - 1e-9:
                    nxt.append((a, b))
                else:
                    if a < ex_a - 1e-9:
                        nxt.append((a, ex_a))
                    if b > ex_b + 1e-9:
                        nxt.append((ex_b, b))
            cur_af = nxt
        af_raw = merge_ivs(cur_af)

    # Concatenate known_ivs into contiguous coordinate [0, T_known]
    mapped_af = []
    t_acc = 0.0
    for ka, kb in known_ivs:
        for aa, ab in af_raw:
            oa, ob = max(aa, ka), min(ab, kb)
            if ob > oa + 1e-9:
                mapped_af.append((t_acc + oa - ka, t_acc + ob - ka))
        t_acc += (kb - ka)
    return merge_ivs(mapped_af), t_acc


def run_e03_and_e04():
    print("=== V8-E03 Common Support & V8-E04 Annotation Harmonization ===")
    s_df = pd.read_parquet(OUT_RES / "V8_stage0_sample_inventory.parquet")
    rhy = pd.read_parquet(V5_MAN / "canonical_rhythm_intervals.parquet")
    runs = pd.read_parquet(V5_MAN / "canonical_af_runs.parquet")
    cal = pd.read_csv("/data/mithunmanivannan/datasets/iridia_af_8405941_v1.0.1/calibration_exclusion_intervals.csv")
    cal_grp = cal.groupby("record_id")

    # 1. Fixed 24h known-horizon axes per patient (1st annotated recording per patient)
    ann_pts = s_df[s_df["is_annotated"]].sort_values(["dataset", "patient_id", "record_id"]).drop_duplicates(["dataset", "patient_id"])
    horizon_24h = 86400.0
    horizon_12h = 43200.0

    # V8-E03: Common support across cohorts (12h & 24h)
    cs_rows = []
    # V8-E04: Harmonization grid across 12h/24h
    harm_rows = []
    for rec in ann_pts.itertuples():
        ds, rid, pid = rec.dataset, str(rec.record_id), rec.patient_id
        af_k, T_k = extract_known_timeline(ds, rid, rhy, runs, cal_grp, rec)
        for h_s in [horizon_12h, horizon_24h]:
            if T_k + 1e-6 < h_s:
                continue
            af_win = [(a, min(b, h_s)) for a, b in af_k if a < h_s]
            ax = compute_axes(af_win, h_s)
            ax.update({"dataset": ds, "patient_id": pid, "record_id": rid, "horizon_h": h_s / 3600.0})
            cs_rows.append(ax)
            if h_s == horizon_12h:
                for min_af in [0.0, 10.0, 30.0, 60.0, 120.0, 360.0]:
                    h_af = apply_harmonization(af_win, min_af_s=min_af, min_gap_s=0.0, known_T=h_s)
                    hx = compute_axes(h_af, h_s)
                    hx.update({"dataset": ds, "patient_id": pid, "horizon_h": 12.0, "harm_rule": f"min_af_{int(min_af)}s"})
                    harm_rows.append(hx)
                for min_gap in [10.0, 30.0, 60.0, 120.0]:
                    h_af = apply_harmonization(af_win, min_af_s=0.0, min_gap_s=min_gap, known_T=h_s)
                    hx = compute_axes(h_af, h_s)
                    hx.update({"dataset": ds, "patient_id": pid, "horizon_h": 12.0, "harm_rule": f"min_gap_{int(min_gap)}s"})
                    harm_rows.append(hx)

    cs_df = pd.DataFrame(cs_rows)
    cs_df.to_csv(OUT_RES / "V8_e03_common_support_axes.csv", index=False)

    # Cohort-specific 6-axis distribution summary (12h & 24h independent patients, IRIDIA separate)
    ax_rows = []
    for (h_val, ds), grp in cs_df.groupby(["horizon_h", "dataset"]):
        multi = grp[grp["n_runs"] >= 2]
        ax_rows.append({
            "horizon_h": h_val, "dataset": ds, "n_pts": len(grp),
            "n_any_af": int((grp["n_runs"] >= 1).sum()),
            "n_dens_defined": int(grp["density"].notna().sum()),
            "n_multi_run": len(multi),
            "n_ge10_runs": int((grp["n_runs"] >= 10).sum()),
            "med_burden_all": round(float(grp["burden"].median()), 4),
            "med_dens_defined": round(float(grp["density"].median()), 4) if grp["density"].notna().any() else np.nan,
            "iqr_dens_defined": round(float(grp["density"].quantile(0.75) - grp["density"].quantile(0.25)), 4) if grp["density"].notna().any() else np.nan,
            "med_longest_share_def": round(float(grp.loc[grp["density"].notna(), "longest_share"].median()), 4) if grp["density"].notna().any() else np.nan,
            "med_n_runs_def": float(grp.loc[grp["density"].notna(), "n_runs"].median()) if grp["density"].notna().any() else np.nan,
            "med_lam_on_risk_h": round(float(grp.loc[grp["density"].notna(), "lam_on_risk_h"].median()), 4) if grp["density"].notna().any() else np.nan,
            "med_lam_off_af_h": round(float(grp.loc[grp["density"].notna(), "lam_off_af_h"].median()), 4) if grp["density"].notna().any() else np.nan,
        })
    ax_df = pd.DataFrame(ax_rows)
    ax_df.to_csv(OUT_RES / "V8_e03_cohort_axes_summary.csv", index=False)
    print("\nV8-E03 Cohort-Specific 6-Axis Availability & Distributions (Independent Patients):")
    print(ax_df.to_string(index=False))

    # Check common support in joint Burden-Density space (12h horizon) inside high-density (d>=0.80) AND matched 10-50% burden!
    h12 = cs_df[(cs_df["horizon_h"] == 12.0) & (cs_df["density"].notna()) & (cs_df["n_runs"] >= 2)].copy()
    hi_d = h12[h12["density"] >= 0.80].copy()
    # Joint support check: within burden [0.05, 0.50] & density [0.80, 1.00], how many pts have <=5 runs vs >=10 runs per cohort?
    joint_band = hi_d[(hi_d["burden"] >= 0.05) & (hi_d["burden"] <= 0.50)]
    support_tbl = []
    for band_lbl, sub in [("All_MultiRun_12h", h12), ("HighDens_ge0.80_12h", hi_d), ("JointBand_B05_50_D80_100_12h", joint_band)]:
        for ds in ["ALL", "IRIDIA-AF", "LTAFDB", "SHDB-AF"]:
            dsub = sub if ds == "ALL" else sub[sub["dataset"] == ds]
            low_f = dsub[dsub["n_runs"] <= 5]
            hi_f = dsub[dsub["n_runs"] >= 10]
            support_tbl.append({
                "stratum": band_lbl, "cohort": ds, "n_pts_total": len(dsub),
                "n_low_frag_le5_runs": len(low_f),
                "med_burden_low_frag": round(float(low_f["burden"].median()), 4) if len(low_f) else np.nan,
                "n_high_frag_ge10_runs": len(hi_f),
                "med_burden_high_frag": round(float(hi_f["burden"].median()), 4) if len(hi_f) else np.nan,
                "common_support_feasible": bool(len(low_f) >= 5 and len(hi_f) >= 5),
            })
    sup_df = pd.DataFrame(support_tbl)
    sup_df.to_csv(OUT_RES / "V8_e03_joint_support_summary.csv", index=False)
    print("\nV8-E03 Joint Burden-Density Common Support (12h Independent Patients):")
    print(sup_df.to_string(index=False))

    harm_df = pd.DataFrame(harm_rows)
    harm_summary = harm_df[harm_df["n_runs"] >= 1].groupby(["harm_rule", "dataset"]).agg(
        n_af_pts=("patient_id", "count"),
        n_multi_run_pts=("n_runs", lambda s: int((s >= 2).sum())),
        n_ge10_run_pts=("n_runs", lambda s: int((s >= 10).sum())),
        median_burden=("burden", "median"),
        median_density=("density", "median"),
        median_runs=("n_runs", "median"),
        median_lam_on_risk=("lam_on_risk_h", "median"),
    ).reset_index()
    harm_summary.to_csv(OUT_RES / "V8_e04_harmonization_summary.csv", index=False)
    print("\nV8-E04 Annotation Harmonization Sensitivity (12h Independent Patients):")
    print(harm_summary[harm_summary["harm_rule"].isin(["min_af_0s", "min_af_30s", "min_af_60s", "min_gap_30s", "min_gap_60s"])].to_string(index=False))


def run_e05_prospective():
    print("\n=== V8-E05 Prospective Nonoverlapping 6h -> Next 6h Transition Prediction ===")
    s_df = pd.read_parquet(OUT_RES / "V8_stage0_sample_inventory.parquet")
    rhy = pd.read_parquet(V5_MAN / "canonical_rhythm_intervals.parquet")
    runs = pd.read_parquet(V5_MAN / "canonical_af_runs.parquet")
    cal = pd.read_csv("/data/mithunmanivannan/datasets/iridia_af_8405941_v1.0.1/calibration_exclusion_intervals.csv")
    cal_grp = cal.groupby("record_id")

    ann_pts = s_df[s_df["is_annotated"]].sort_values(["dataset", "patient_id", "record_id"]).drop_duplicates(["dataset", "patient_id"])
    # 1 independent pair per patient: [0, 6h) -> [6h, 12h) (plus [6h, 12h) -> [12h, 18h) / [12h, 18h) -> [18h, 24h) as repeated longitudinal sensitivity, keeping primary 1 pair/pt!)
    blk_s = 6 * 3600.0
    rows = []
    for rec in ann_pts.itertuples():
        ds, rid, pid = rec.dataset, str(rec.record_id), rec.patient_id
        af_k, T_k = extract_known_timeline(ds, rid, rhy, runs, cal_grp, rec)
        if T_k + 1e-6 < 2 * blk_s:
            continue
        hist_af = [(a, min(b, blk_s)) for a, b in af_k if a < blk_s]
        fut_af = [(max(0.0, a - blk_s), min(blk_s, b - blk_s)) for a, b in af_k if b > blk_s and a < 2 * blk_s]
        hx = compute_axes(hist_af, blk_s)
        fx = compute_axes(fut_af, blk_s)
        # Future prolonged episode (>=30m inside future 6h block)
        fut_prolonged = int(any((b - a) >= 1800.0 for a, b in fut_af))
        rows.append({
            "dataset": ds, "patient_id": pid,
            "hist_burden": hx["burden"],
            "hist_density": hx["density"],
            "hist_density_defined": int(np.isfinite(hx["density"])),
            "hist_longest_share": hx["longest_share"],
            "hist_n_runs": hx["n_runs"],
            "hist_lam_on_risk_h": hx["lam_on_risk_h"],
            "hist_lam_off_af_h": hx["lam_off_af_h"],
            "fut_n_onsets": fx["n_onsets"],
            "fut_n_terms": fx["n_terms"],
            "fut_burden": fx["burden"],
            "fut_nonaf_h": max(1e-4, fx["non_af_s"] / 3600.0),
            "fut_prolonged": fut_prolonged,
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_RES / "V8_e05_prospective_6h_6h_cohort.csv", index=False)

    # Evaluate M0 -> M4 on patients with DEFINED historical density & valid future non-AF offset (>=60s future non-AF for conditional rate model, AND unconditional count model on all patients with defined hist_density)
    sub_def = df[df["hist_density_defined"] == 1].copy().reset_index(drop=True)
    # Standardize predictors for GLM interpretability
    for col in ["hist_burden", "hist_density", "hist_longest_share", "hist_n_runs", "hist_lam_on_risk_h"]:
        sub_def[f"z_{col}"] = (sub_def[col] - sub_def[col].mean()) / (sub_def[col].std() + 1e-9)

    results = []
    # Unconditional count of future AF onsets (Negative Binomial GLM with log link) + Conditional rate (offset log(fut_nonaf_h))
    sub_cond = sub_def[sub_def["fut_nonaf_h"] >= (60.0 / 3600.0)].copy().reset_index(drop=True)
    models = {
        "M0_Cohort": ["C(dataset)"],
        "M1_Burden": ["C(dataset)", "z_hist_burden"],
        "M2_Burden_Density": ["C(dataset)", "z_hist_burden", "z_hist_density"],
        "M3_M2_Plus_Continuity": ["C(dataset)", "z_hist_burden", "z_hist_density", "z_hist_longest_share"],
        "M4_M3_Plus_RunCount": ["C(dataset)", "z_hist_burden", "z_hist_density", "z_hist_longest_share", "z_hist_n_runs"],
        "M5_M4_Plus_HistOnsetRate": ["C(dataset)", "z_hist_burden", "z_hist_density", "z_hist_longest_share", "z_hist_n_runs", "z_hist_lam_on_risk_h"],
    }

    for estimand, edf, offset_col in [
        ("Unconditional_Future_Onsets_NB", sub_def, None),
        ("Conditional_Onset_Rate_Per_NonAF_H_NB", sub_cond, "fut_nonaf_h"),
    ]:
        edf_use = edf.dropna(subset=["z_hist_longest_share", "z_hist_lam_on_risk_h"]).copy()
        prev_ll = None
        for mname, terms in models.items():
            formula = "fut_n_onsets ~ " + " + ".join(terms)
            off = np.log(edf_use[offset_col].values) if offset_col else None
            try:
                glm = smf.glm(formula=formula, data=edf_use, family=sm.families.NegativeBinomial(alpha=1.0), offset=off).fit()
                ll = float(glm.llf)
                aic = float(glm.aic)
                lr_stat = 2.0 * (ll - prev_ll) if prev_ll is not None else np.nan
                results.append({
                    "estimand": estimand, "model": mname, "n_patients": len(edf_use),
                    "log_lik": round(ll, 3), "aic": round(aic, 2),
                    "delta_ll_vs_prev": round(ll - prev_ll, 3) if prev_ll is not None else np.nan,
                    "coef_hist_density": round(float(glm.params.get("z_hist_density", np.nan)), 4),
                    "p_hist_density": round(float(glm.pvalues.get("z_hist_density", np.nan)), 4),
                    "coef_hist_lam_on": round(float(glm.params.get("z_hist_lam_on_risk_h", np.nan)), 4),
                    "p_hist_lam_on": round(float(glm.pvalues.get("z_hist_lam_on_risk_h", np.nan)), 4),
                })
                prev_ll = ll
            except Exception as e:
                results.append({"estimand": estimand, "model": mname, "error": str(e)})

    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT_RES / "V8_e05_prospective_models.csv", index=False)
    print(f"\nProspective 6h->6h Models (Defined-density independent patients n={len(sub_def)}):")
    print(res_df.to_string(index=False))

    # V8-E05 Sensitivity: (a) 3h->3h and 12h->12h Cond/Uncond NB, (b) LTAFDB+SHDB vs IRIDIA separately, (c) fut_burden & fut_prolonged
    sens_rows = []
    for h_lbl, b_sec in [("3h_to_3h", 3 * 3600.0), ("6h_to_6h", 6 * 3600.0), ("12h_to_12h", 12 * 3600.0)]:
        b_rows = []
        for rec in ann_pts.itertuples():
            ds, rid, pid = rec.dataset, str(rec.record_id), rec.patient_id
            af_k, T_k = extract_known_timeline(ds, rid, rhy, runs, cal_grp, rec)
            if T_k + 1e-6 < 2 * b_sec:
                continue
            hist_af = [(a, min(b, b_sec)) for a, b in af_k if a < b_sec]
            fut_af = [(max(0.0, a - b_sec), min(b_sec, b - b_sec)) for a, b in af_k if b > b_sec and a < 2 * b_sec]
            hx = compute_axes(hist_af, b_sec)
            fx = compute_axes(fut_af, b_sec)
            if not np.isfinite(hx["density"]) or not np.isfinite(hx["longest_share"]) or not np.isfinite(hx["lam_on_risk_h"]):
                continue
            b_rows.append({
                "dataset": ds, "patient_id": pid,
                "hist_burden": hx["burden"], "hist_density": hx["density"],
                "hist_longest_share": hx["longest_share"], "hist_n_runs": hx["n_runs"],
                "hist_lam_on_risk_h": hx["lam_on_risk_h"],
                "fut_n_onsets": fx["n_onsets"], "fut_burden": fx["burden"],
                "fut_nonaf_h": max(1e-4, fx["non_af_s"] / 3600.0),
                "fut_prolonged": int(any((b - a) >= 1800.0 for a, b in fut_af)),
            })
        bdf = pd.DataFrame(b_rows)
        for col in ["hist_burden", "hist_density", "hist_longest_share", "hist_n_runs", "hist_lam_on_risk_h"]:
            bdf[f"z_{col}"] = (bdf[col] - bdf[col].mean()) / (bdf[col].std() + 1e-9)
        for grp_lbl, gdf in [("ALL", bdf), ("LTAFDB_SHDB", bdf[bdf["dataset"] != "IRIDIA-AF"]), ("IRIDIA_ONLY", bdf[bdf["dataset"] == "IRIDIA-AF"])]:
            g_cond = gdf[gdf["fut_nonaf_h"] >= (60.0 / 3600.0)].copy()
            if len(g_cond) < 20:
                continue
            cterm = "C(dataset) + " if g_cond["dataset"].nunique() > 1 else ""
            try:
                m1 = smf.glm(f"fut_n_onsets ~ {cterm}z_hist_burden", data=g_cond, family=sm.families.NegativeBinomial(alpha=1.0), offset=np.log(g_cond["fut_nonaf_h"].values)).fit()
                m2 = smf.glm(f"fut_n_onsets ~ {cterm}z_hist_burden + z_hist_density", data=g_cond, family=sm.families.NegativeBinomial(alpha=1.0), offset=np.log(g_cond["fut_nonaf_h"].values)).fit()
                m4 = smf.glm(f"fut_n_onsets ~ {cterm}z_hist_burden + z_hist_density + z_hist_longest_share + z_hist_n_runs", data=g_cond, family=sm.families.NegativeBinomial(alpha=1.0), offset=np.log(g_cond["fut_nonaf_h"].values)).fit()
                sens_rows.append({
                    "horizon": h_lbl, "cohort_subset": grp_lbl, "n_pts": len(g_cond),
                    "ll_M1_burden": round(float(m1.llf), 2), "ll_M2_plus_dens": round(float(m2.llf), 2),
                    "ll_M4_plus_arch": round(float(m4.llf), 2),
                    "p_dens_M2": round(float(m2.pvalues.get("z_hist_density", np.nan)), 4),
                    "p_dens_M4": round(float(m4.pvalues.get("z_hist_density", np.nan)), 4),
                    "p_nruns_M4": round(float(m4.pvalues.get("z_hist_n_runs", np.nan)), 4),
                })
            except Exception as e:
                sens_rows.append({"horizon": h_lbl, "cohort_subset": grp_lbl, "n_pts": len(g_cond), "error": str(e)})
    pd.DataFrame(sens_rows).to_csv(OUT_RES / "V8_e05_horizon_cohort_sensitivity.csv", index=False)
    print("\nV8-E05 Horizon & Cohort Sensitivity (Conditional Onset Rate):")
    print(pd.DataFrame(sens_rows).to_string(index=False))


def run_e06_within_between():
    print("\n=== V8-E06 Within-Patient vs Between-Patient Regime Propensity Audit ===")
    oof_df = pd.read_parquet(OUT_RES / "V8_stage0_oof_predictions.parquet")
    # Also load P02 & P01 manifests to build a training-fold-only patient propensity / cohort+rate baseline AND within-patient mean-centered decomposition!
    rows = []
    for task_lbl in ["P01_CleanAnnotated", "P02_Persistence60s", "P03_Survival15m"]:
        sub = oof_df[oof_df["task"] == task_lbl]
        for mk in ["logistic", "lgbm"]:
            sm = sub[sub["model"] == mk]
            # Equal-patient AUROC + counts of pos/neg per patient
            eq_aucs_s2, eq_aucs_s3, pos_counts, neg_counts = [], [], [], []
            for pt, grp in sm.groupby("patient_id"):
                if grp["y_true"].nunique() == 2:
                    eq_aucs_s2.append(roc_auc_score(grp["y_true"], grp["prob_s2_matched"]))
                    eq_aucs_s3.append(roc_auc_score(grp["y_true"], grp["prob_s3a_matched"]))
                    pos_counts.append(int(grp["y_true"].sum()))
                    neg_counts.append(int((1 - grp["y_true"]).sum()))
            # Compare against a chronological prior-only patient propensity baseline (mean y_true observed BEFORE current landmark in SAME patient!)
            sm_sorted = sm.sort_values(["patient_id", "landmark_time_s"]).copy()
            prior_prop = np.zeros(len(sm_sorted))
            fold_prior = sm_sorted.groupby("fold")["y_true"].transform(lambda s: s)  # placeholder
            for f in sm_sorted["fold"].unique():
                tr_mean = float(sm_sorted[sm_sorted["fold"] != f]["y_true"].mean())
                va_idx = sm_sorted["fold"] == f
                # Causal cumulative prior y inside same patient (0 prior -> fold_tr_mean)
                va_sub = sm_sorted[va_idx]
                preds_va = []
                for pt, grp in va_sub.groupby("patient_id", sort=False):
                    ys = grp["y_true"].values
                    # Laplace-smoothed causal running rate from prior episodes of this patient ONLY
                    csum = 0.0
                    for idx_i, y_val in enumerate(ys):
                        preds_va.append((csum + 2.0 * tr_mean) / (idx_i + 2.0))
                        csum += y_val
                prior_prop[va_idx] = preds_va

            # Wait: assign carefully by original row order
            sm_sorted["causal_pt_prop"] = np.nan
            for f in sm_sorted["fold"].unique():
                tr_mean = float(sm_sorted[sm_sorted["fold"] != f]["y_true"].mean())
                for pt, grp in sm_sorted[sm_sorted["fold"] == f].groupby("patient_id"):
                    ys = grp["y_true"].values
                    csum = 0.0
                    vals = []
                    for idx_i, y_val in enumerate(ys):
                        vals.append((csum + 2.0 * tr_mean) / (idx_i + 2.0))
                        csum += y_val
                    sm_sorted.loc[grp.index, "causal_pt_prop"] = vals

            auc_causal_prop = float(roc_auc_score(sm_sorted["y_true"], sm_sorted["causal_pt_prop"]))
            # Cohort-stratified + pooled
            rows.append({
                "task": task_lbl, "model": mk,
                "pooled_s2m_auroc": round(float(roc_auc_score(sm["y_true"], sm["prob_s2_matched"])), 4),
                "pooled_s3am_auroc": round(float(roc_auc_score(sm["y_true"], sm["prob_s3a_matched"])), 4),
                "causal_pt_propensity_only_auroc": round(auc_causal_prop, 4),
                "n_both_outcome_pts": len(eq_aucs_s2),
                "med_pos_per_eq_pt": float(np.median(pos_counts)),
                "med_neg_per_eq_pt": float(np.median(neg_counts)),
                "within_pt_s2m_auroc": round(float(np.mean(eq_aucs_s2)), 4),
                "within_pt_s3am_auroc": round(float(np.mean(eq_aucs_s3)), 4),
                "within_pt_causal_prop_auroc": round(float(np.mean([
                    roc_auc_score(g["y_true"], g["causal_pt_prop"])
                    for _, g in sm_sorted.groupby("patient_id") if g["y_true"].nunique() == 2
                ])), 4),
            })
    e6_df = pd.DataFrame(rows)
    e6_df.to_csv(OUT_RES / "V8_e06_within_between_summary.csv", index=False)
    print(e6_df.to_string(index=False))

    # Matched LTAFDB+SHDB (and All Cohorts) Valid-RR Contrast: History(S3a) vs S3a+CleanRR vs S3a+CleanRR+ECGFounder_10s(PCA-32)
    s_df = pd.read_parquet(OUT_RES / "V8_stage0_sample_inventory.parquet")
    unann_shdb = set(s_df[(s_df["dataset"] == "SHDB-AF") & (~s_df["is_annotated"])]["record_id"].astype(str))
    p1 = pd.read_parquet(V7P_DIR / "P01_landmark_manifest.parquet")
    p1_mask = (~((p1["dataset"] == "SHDB-AF") & (p1["record_id"].astype(str).isin(unann_shdb)))) & (
        ~((p1["dataset"] == "LTAFDB") & (p1["current_rhythm"] == "UNKNOWN"))
    )
    p2 = pd.read_parquet(V7P_DIR / "P02_episode_persistence_manifest.parquet")
    p3 = pd.read_parquet(V7P_DIR / "P03_survival_manifest.parquet")

    f_s3a = [
        "time_since_rec_start_s", "elapsed_af_duration_s", "last_af_duration_s", "time_since_last_af_s", "has_prior_af",
        "af_burden_15m", "af_burden_30m", "af_burden_1h", "af_burden_3h", "af_burden_6h",
        "episode_count_15m", "episode_count_30m", "episode_count_1h", "episode_count_3h", "episode_count_6h",
        "longest_af_run_15m", "longest_af_run_30m", "longest_af_run_1h", "longest_af_run_3h", "longest_af_run_6h",
        "brief_run_count_15m", "brief_run_count_30m", "brief_run_count_1h", "brief_run_count_3h", "brief_run_count_6h",
        "charitos_density_15m", "charitos_density_30m", "charitos_density_1h", "charitos_density_3h", "charitos_density_6h",
        "is_density_defined_3h",
    ]
    rr_cols = ["mean_hr_bpm", "std_hr_bpm", "rmssd_ms", "cvrr", "pnn50", "num_beats"]

    def boot_auc_diff(y_true, pa, pb, pts, n_boot=2000, seed=42):
        rng = np.random.RandomState(seed)
        upts = np.unique(pts)
        pt_idx = {p: np.where(pts == p)[0] for p in upts}
        obs = float(roc_auc_score(y_true, pb) - roc_auc_score(y_true, pa))
        ds = []
        for _ in range(n_boot):
            idx = np.concatenate([pt_idx[p] for p in rng.choice(upts, size=len(upts), replace=True)])
            yb = y_true[idx]
            if len(np.unique(yb)) < 2:
                continue
            ra = rankdata(pa[idx], method="average")
            rb = rankdata(pb[idx], method="average")
            n1 = int(yb.sum()); n0 = len(yb) - n1
            ds.append((rb[yb == 1].sum() - ra[yb == 1].sum()) / (n1 * n0))
        arr = np.array(ds)
        p_lo = (np.sum(arr <= 0) + 1) / (len(arr) + 1)
        p_hi = (np.sum(arr >= 0) + 1) / (len(arr) + 1)
        return obs, float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)), float(min(1.0, 2 * min(p_lo, p_hi)))

    def eq_pt_auc(y_true, probs, pts):
        vals = [float(roc_auc_score(y_true[pts == p], probs[pts == p])) for p in np.unique(pts) if len(np.unique(y_true[pts == p])) == 2]
        return float(np.mean(vals)) if vals else np.nan, len(vals)

    matched_rows = []
    for pcode, task_lbl, df_m, row_mask, t_col in [
        ("P01", "P01_CleanAnnotated", p1, p1_mask.values, "target_sustained_30m"),
        ("P02", "P02_Persistence60s", p2, np.ones(len(p2), dtype=bool), "target_persists_30m"),
        ("P03", "P03_Survival15m", p3, np.ones(len(p3), dtype=bool), "target_terminate_in_15m"),
    ]:
        f4 = pd.read_parquet(V7P_DIR / f"embeddings/{pcode}_f4_rr_features.parquet")
        ef = np.load(V7P_DIR / f"embeddings/{pcode}_ecgfounder_10s.npz", allow_pickle=True)["embeddings"]
        for csub_lbl, coh_filter in [
            ("LTAFDB_SHDB_ValidRR", df_m["dataset"].isin(["LTAFDB", "SHDB-AF"]).values),
            ("ALL_Cohorts_ValidRR", np.ones(len(df_m), dtype=bool)),
        ]:
            keep = row_mask & coh_filter & f4["is_rr_valid"].astype(bool).values & np.all(np.isfinite(ef), axis=1)
            sub = df_m[keep].reset_index(drop=True)
            sub_rr = f4.loc[keep, rr_cols].values.astype(float)
            sub_ef = ef[keep].astype(np.float32)
            y = sub[t_col].astype(int).values
            folds = sub["fold"].values
            pts = sub["patient_id"].values
            if len(np.unique(y)) < 2:
                continue
            cols_hist = [c for c in f_s3a if c in sub.columns]
            X_hist = sub[cols_hist].values.astype(float)

            for mk in ["logistic", "lgbm"]:
                oof_h = np.zeros(len(y))
                oof_hr = np.zeros(len(y))
                oof_hrf = np.zeros(len(y))
                for f in sorted(np.unique(folds)):
                    tr, va = folds != f, folds == f
                    if len(np.unique(y[tr])) < 2:
                        continue
                    def fit_pred(Xtr_raw, Xva_raw):
                        imp = SimpleImputer(strategy="median")
                        sc = StandardScaler()
                        Xtr_s = sc.fit_transform(imp.fit_transform(Xtr_raw))
                        Xva_s = sc.transform(imp.transform(Xva_raw))
                        clf = LogisticRegression(C=1.0, max_iter=500) if mk == "logistic" else LGBMClassifier(
                            n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=42, n_jobs=1, verbose=-1
                        )
                        return clf.fit(Xtr_s, y[tr]).predict_proba(Xva_s)[:, 1]
                    oof_h[va] = fit_pred(X_hist[tr], X_hist[va])
                    oof_hr[va] = fit_pred(np.hstack([X_hist[tr], sub_rr[tr]]), np.hstack([X_hist[va], sub_rr[va]]))
                    # PCA-32 on ECGFounder fitted strictly inside fold!
                    sc_f = StandardScaler()
                    ef_tr = sc_f.fit_transform(sub_ef[tr])
                    ef_va = sc_f.transform(sub_ef[va])
                    pca = PCA(n_components=min(32, ef_tr.shape[0] - 1, ef_tr.shape[1]), random_state=42)
                    Z_tr = pca.fit_transform(ef_tr)
                    Z_va = pca.transform(ef_va)
                    oof_hrf[va] = fit_pred(np.hstack([X_hist[tr], sub_rr[tr], Z_tr]), np.hstack([X_hist[va], sub_rr[va], Z_va]))

                d_rr, lo_rr, hi_rr, pv_rr = boot_auc_diff(y, oof_h, oof_hr, pts)
                d_ef, lo_ef, hi_ef, pv_ef = boot_auc_diff(y, oof_hr, oof_hrf, pts)
                eq_h, n_eq = eq_pt_auc(y, oof_h, pts)
                eq_hr, _ = eq_pt_auc(y, oof_hr, pts)
                eq_hrf, _ = eq_pt_auc(y, oof_hrf, pts)
                matched_rows.append({
                    "task": task_lbl, "subset": csub_lbl, "model": mk,
                    "n_rows": len(y), "n_pos": int(y.sum()), "n_patients": len(np.unique(pts)),
                    "auc_hist_s3a": round(float(roc_auc_score(y, oof_h)), 4),
                    "auc_hist_rr": round(float(roc_auc_score(y, oof_hr)), 4),
                    "auc_hist_rr_ecgf": round(float(roc_auc_score(y, oof_hrf)), 4),
                    "delta_rr_over_hist": round(d_rr, 4), "ci_rr": f"[{lo_rr:.4f},{hi_rr:.4f}]", "p_rr": round(pv_rr, 4),
                    "delta_ecgf_over_rr": round(d_ef, 4), "ci_ecgf": f"[{lo_ef:.4f},{hi_ef:.4f}]", "p_ecgf": round(pv_ef, 4),
                    "eq_pt_hist": round(eq_h, 4), "eq_pt_hist_rr": round(eq_hr, 4), "eq_pt_hist_rr_ecgf": round(eq_hrf, 4), "n_eq_pts": n_eq,
                })
    m_df = pd.DataFrame(matched_rows)
    m_df.to_csv(OUT_RES / "V8_e06_matched_rr_ecgfounder.csv", index=False)
    print("\n=== V8-E06 Matched Valid-RR Ablation (History vs +CleanRR vs +ECGFounder) ===")
    print(m_df.to_string(index=False))


if __name__ == "__main__":
    run_e03_and_e04()
    run_e05_prospective()
    run_e06_within_between()
