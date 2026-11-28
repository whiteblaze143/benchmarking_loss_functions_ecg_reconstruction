#!/usr/bin/env python3
"""
Stage 11 — Secondary Subgroup & Circadian Analyses (Revision).

1. SHDB post-ablation vs palpitation: adds `p_reinit_60s_between_groups` (Fisher + GEE).
2. SHDB `non-AF` labelled patients (all 33 records): full audit with `Annotated` status,
   `n_raw_af_runs`, `n_g0_af_runs`, `total_af_sec`, plus PAF/PerAF summary counts.
3. IRIDIA repeat-recording patients: ICC(1,1) of reinit-<360s, mean AFR, mean PMV_A.
4. Circadian: per-cohort (`IRIDIA`, `SHDB`, `LTAFDB`) and `POOLED` (with cohort fixed effects),
   using patient-clustered GEE binomial logit cosinor AND random-intercept mixed logit.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from lifelines import WeibullFitter
from scipy import stats

SEED = 20260101
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import common  # type: ignore


def get_provenance() -> dict[str, str]:
    return {
        "git_hash": common.get_git_hash(),
        "definitions_hash": common.compute_definitions_hash(),
        "run_timestamp": common.utc_timestamp_iso(),
    }


def add_provenance(df: pd.DataFrame, prov: dict[str, str]) -> pd.DataFrame:
    df = df.copy()
    for k, v in prov.items():
        df[k] = v
    return df


def compute_icc_oneway(df: pd.DataFrame, group_col: str, val_col: str) -> tuple[float, int, int]:
    sub = df.dropna(subset=[group_col, val_col]).copy()
    counts = sub.groupby(group_col)[val_col].count()
    valid_groups = counts[counts >= 2].index
    sub = sub[sub[group_col].isin(valid_groups)]
    k, N = len(valid_groups), len(sub)
    if k < 2 or N <= k:
        return np.nan, k, N
    grand_mean = sub[val_col].mean()
    grp = sub.groupby(group_col)[val_col]
    ni, means = grp.count().values, grp.mean().values
    ss_between = float(np.sum(ni * (means - grand_mean) ** 2))
    ss_within = float(np.sum((sub[val_col] - sub.groupby(group_col)[val_col].transform("mean")) ** 2))
    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (N - k)
    k0 = (N - np.sum(ni ** 2) / N) / (k - 1)
    denom = ms_between + (k0 - 1.0) * ms_within
    return float((ms_between - ms_within) / denom) if denom > 0 else np.nan, int(k), int(N)


def load_merged_data(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = pd.read_parquet(run_dir / "stage2" / "runs_g0.parquet")
    events = pd.read_parquet(run_dir / "stage4" / "events_g0.parquet")
    for sub_rel in ["stage5/terminal_markers.parquet", "stage6/post_markers.parquet"]:
        p = run_dir / sub_rel
        if p.exists():
            m = pd.read_parquet(p)
            keys = ["event_id"] if ("event_id" in events.columns and "event_id" in m.columns) else [
                c for c in ["record", "run_id"] if c in events.columns and c in m.columns
            ]
            if keys:
                new_cols = [c for c in m.columns if c not in events.columns or c in keys]
                events = events.merge(m[new_cols].drop_duplicates(subset=keys), on=keys, how="left")
    return runs, events


def analyze_shdb_subgroups(runs: pd.DataFrame, events: pd.DataFrame, prov: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    add_path = Path("/data/mithunmanivannan/datasets/shdb_af/AdditionalData.csv")
    meta = pd.read_csv(add_path, dtype={"Data_ID": str})
    meta["record_key"] = meta["Data_ID"].str.zfill(3)

    shdb_runs = runs[runs["cohort"] == "SHDB"].copy()
    shdb_runs["record_key"] = shdb_runs["record"].astype(str).str.extract(r"(\d+)")[0].str.zfill(3)
    shdb_runs = shdb_runs.merge(meta, on="record_key", how="left")

    shdb_ev = events[events["cohort"] == "SHDB"].copy()
    shdb_ev["record_key"] = shdb_ev["record"].astype(str).str.extract(r"(\d+)")[0].str.zfill(3)
    shdb_ev = shdb_ev.merge(meta, on="record_key", how="left", suffixes=("", "_meta"))

    # 1. Post-ablation vs Palpitation
    groups = ["AF monitoring after ablation", "Palpitation"]
    rows = []
    for ind in groups:
        g_meta = meta[meta["Indication_Holter"] == ind]
        g_runs = shdb_runs[shdb_runs["Indication_Holter"] == ind]
        g_ev = shdb_ev[shdb_ev["Indication_Holter"] == ind]
        wb_shape = np.nan
        if len(g_runs) >= 10:
            g_known = g_runs[~g_runs["boundary_censored_start"].astype(bool)] if "boundary_censored_start" in g_runs.columns else g_runs
            if len(g_known) >= 10:
                wf = WeibullFitter()
                ev_obs = (~g_known["boundary_censored_end"].astype(bool)).astype(int) if "boundary_censored_end" in g_known.columns else np.ones(len(g_known))
                wf.fit(g_known["dur_sec"].clip(lower=1e-3), event_observed=ev_obs)
                wb_shape = float(wf.rho_)
        rows.append({
            "subgroup": ind,
            "n_records_meta": int(len(g_meta)),
            "n_records_with_af": int(g_runs["record"].nunique()),
            "n_runs": int(len(g_runs)),
            "n_termination_events": int(len(g_ev)),
            "reinit_60s_prop": float((g_ev["time_to_reinit_sec"] < 60).mean()) if len(g_ev) else np.nan,
            "reinit_360s_prop": float((g_ev["time_to_reinit_sec"] < 360).mean()) if len(g_ev) else np.nan,
            "afr_df_mean_hz": float(g_ev["AFR_DF_mean"].mean()) if ("AFR_DF_mean" in g_ev.columns and g_ev["AFR_DF_mean"].notna().any()) else np.nan,
            "afr_df_sd_hz": float(g_ev["AFR_DF_mean"].std()) if ("AFR_DF_mean" in g_ev.columns and g_ev["AFR_DF_mean"].notna().any()) else np.nan,
            "weibull_shape": wb_shape,
        })

    g1_ev = shdb_ev[shdb_ev["Indication_Holter"] == groups[0]]
    g2_ev = shdb_ev[shdb_ev["Indication_Holter"] == groups[1]]
    p_60_fisher = p_60_gee = p_360 = p_afr = np.nan
    if len(g1_ev) > 0 and len(g2_ev) > 0:
        _, p_60_fisher = stats.fisher_exact([
            [int((g1_ev["time_to_reinit_sec"] < 60).sum()), int((g1_ev["time_to_reinit_sec"] >= 60).sum())],
            [int((g2_ev["time_to_reinit_sec"] < 60).sum()), int((g2_ev["time_to_reinit_sec"] >= 60).sum())],
        ])
        _, p_360 = stats.fisher_exact([
            [int((g1_ev["time_to_reinit_sec"] < 360).sum()), int((g1_ev["time_to_reinit_sec"] >= 360).sum())],
            [int((g2_ev["time_to_reinit_sec"] < 360).sum()), int((g2_ev["time_to_reinit_sec"] >= 360).sum())],
        ])
        # Patient-clustered GEE logit for <60s
        comb = pd.concat([g1_ev, g2_ev], ignore_index=True).dropna(subset=["time_to_reinit_sec", "patient"])
        comb["y60"] = (comb["time_to_reinit_sec"] < 60).astype(float)
        comb["is_ablation"] = (comb["Indication_Holter"] == groups[0]).astype(float)
        try:
            gee = sm.GEE(
                comb["y60"],
                sm.add_constant(comb[["is_ablation"]]),
                groups=comb["patient"],
                family=sm.families.Binomial(),
                cov_struct=sm.cov_struct.Exchangeable(),
            ).fit()
            p_60_gee = float(gee.pvalues["is_ablation"])
        except Exception:
            pass
        if "AFR_DF_mean" in g1_ev.columns:
            a1, a2 = g1_ev["AFR_DF_mean"].dropna(), g2_ev["AFR_DF_mean"].dropna()
            if len(a1) >= 3 and len(a2) >= 3:
                _, p_afr = stats.mannwhitneyu(a1, a2, alternative="two-sided")

    sub_df = pd.DataFrame(rows)
    sub_df["p_reinit_60s_between_groups"] = float(p_60_fisher) if np.isfinite(p_60_fisher) else np.nan
    sub_df["p_reinit_60s_gee_clustered"] = float(p_60_gee) if np.isfinite(p_60_gee) else np.nan
    sub_df["p_reinit_360s_between_groups"] = float(p_360) if np.isfinite(p_360) else np.nan
    sub_df["p_afr_between_groups"] = float(p_afr) if np.isfinite(p_afr) else np.nan

    # 2. Full audit of ALL 33 SHDB non-AF records (strictly distinguish Annotated vs Unannotated -> NA, never fill 0!)
    run_counts = shdb_runs.groupby("record_key").agg(n_g0_af_runs=("dur_sec", "count"), total_af_sec=("dur_sec", "sum")).reset_index()
    non_af_meta = meta[meta["AF_Type"].astype(str).str.lower().isin(["non-af", "non af"])].copy()
    non_af_meta = non_af_meta.merge(run_counts, on="record_key", how="left")
    is_ann = non_af_meta["Annotated"].astype(str).str.lower() == "true"

    non_af_meta["n_raw_af_runs"] = pd.array([pd.NA] * len(non_af_meta), dtype="Int64")
    non_af_meta["n_g0_af_runs_exact"] = pd.array([pd.NA] * len(non_af_meta), dtype="Int64")
    non_af_meta.loc[is_ann, "n_raw_af_runs"] = non_af_meta.loc[is_ann, "n_g0_af_runs"].fillna(0).astype(int)
    non_af_meta.loc[is_ann, "n_g0_af_runs_exact"] = non_af_meta.loc[is_ann, "n_g0_af_runs"].fillna(0).astype(int)
    non_af_meta["n_g0_af_runs"] = non_af_meta["n_g0_af_runs_exact"]
    non_af_meta.loc[is_ann, "total_af_sec"] = non_af_meta.loc[is_ann, "total_af_sec"].fillna(0.0)
    non_af_meta.loc[~is_ann, "total_af_sec"] = np.nan

    # PAF & PerAF reference counts
    ann_meta = meta[meta["Annotated"].astype(str).str.lower() == "true"]
    paf_ann = ann_meta[ann_meta["AF_Type"] == "PAF"]
    peraf_ann = ann_meta[ann_meta["AF_Type"].astype(str).str.upper().str.contains("PER")]
    paf_with_af = int(paf_ann["record_key"].isin(run_counts["record_key"]).sum())
    peraf_with_af = int(peraf_ann["record_key"].isin(run_counts["record_key"]).sum())

    non_af_out = pd.DataFrame({
        "Data_ID": non_af_meta["record_key"],
        "Subject_ID": non_af_meta["Subject_ID"],
        "Annotated": non_af_meta["Annotated"],
        "AF_Type": non_af_meta["AF_Type"],
        "Indication_Holter": non_af_meta["Indication_Holter"],
        "Age_at_Holter": non_af_meta["Age_at_Holter"],
        "Sex": non_af_meta["Sex"],
        "n_raw_af_runs": non_af_meta["n_raw_af_runs"],
        "n_g0_af_runs": non_af_meta["n_g0_af_runs"],
        "total_af_sec": non_af_meta["total_af_sec"],
        "shdb_paf_annotated_with_af": f"{paf_with_af}/{len(paf_ann)}",
        "shdb_peraf_annotated_with_af": f"{peraf_with_af}/{len(peraf_ann)}",
    })
    return add_provenance(sub_df, prov), add_provenance(non_af_out, prov)


def analyze_iridia_repeats(runs: pd.DataFrame, events: pd.DataFrame, prov: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Comprehensive Priority 6 audit of IRIDIA repeat-recording patients:
    - All multi-record patients from metadata (`record_start_time`, gap in days/years)
    - Exact runs and qualifying termination denominators per visit
    - `reinit_30s`, `reinit_60s`, `reinit_360s` per visit
    - Patient-cluster bootstrap 95% CI on ICC(1,1)
    - Leave-one-patient-out ICC sensitivity (exposing single-patient driver `patient_045`)
    """
    ir_ev = events[events["cohort"] == "IRIDIA"].copy()
    ir_runs = runs[runs["cohort"] == "IRIDIA"].copy()
    if len(ir_ev) == 0:
        return add_provenance(pd.DataFrame(), prov), add_provenance(pd.DataFrame(), prov)

    ir_meta = pd.read_csv("/data/mithunmanivannan/datasets/iridia_af_8405941_v1.0.1/iridia-af-metadata-v1.0.1.csv")
    rep_meta = ir_meta.groupby("patient_id").filter(lambda g: len(g) >= 2).sort_values(["patient_id", "record_start_time"]).copy()
    rep_meta["dt_start"] = pd.to_datetime(rep_meta["record_start_time"])
    rep_meta["days_since_first_visit"] = (
        rep_meta["dt_start"] - rep_meta.groupby("patient_id")["dt_start"].transform("min")
    ).dt.total_seconds() / 86400.0
    rep_meta["days_since_prev_visit"] = rep_meta.groupby("patient_id")["dt_start"].diff().dt.total_seconds() / 86400.0

    run_agg = ir_runs.groupby("record", as_index=False).agg(
        n_runs=("run_id", "count"),
        total_af_sec=("dur_sec", "sum"),
    )
    ev_agg = ir_ev.groupby("record", as_index=False).agg(
        n_qualifying_terminations=("time_to_reinit_sec", "count"),
        n_observed_reinit=("event", "sum"),
        n_reinit_30s=("reinit_60s", lambda s: 0),  # overwritten below from exact time
        n_reinit_60s=("time_to_reinit_sec", lambda s: int((s < 60.0).sum())),
        n_reinit_360s=("time_to_reinit_sec", lambda s: int((s < 360.0).sum())),
        reinit_360s_prop=("time_to_reinit_sec", lambda s: float((s < 360.0).mean())),
        AFR_DF_mean=("AFR_DF_mean", "mean") if "AFR_DF_mean" in ir_ev.columns else ("time_to_reinit_sec", lambda s: np.nan),
        PMV_A=("PMV_A", "mean") if "PMV_A" in ir_ev.columns else ("time_to_reinit_sec", lambda s: np.nan),
    )
    # Exact 30s reinit count using event==1 & time_to_reinit_sec < 30
    r30_map = ir_ev.groupby("record").apply(lambda g: int(((g["event"] == 1) & (g["time_to_reinit_sec"] < 30.0)).sum()), include_groups=False).to_dict()
    ev_agg["n_reinit_30s"] = ev_agg["record"].map(r30_map).fillna(0).astype(int)

    audit_df = rep_meta.merge(run_agg, left_on="record_id", right_on="record", how="left").merge(ev_agg, left_on="record_id", right_on="record", how="left", suffixes=("", "_ev"))
    audit_cols = [
        "patient_id", "record_id", "record_start_time", "record_end_time", "record_seconds",
        "days_since_first_visit", "days_since_prev_visit", "n_runs", "total_af_sec",
        "n_qualifying_terminations", "n_observed_reinit", "n_reinit_30s", "n_reinit_60s",
        "n_reinit_360s", "reinit_360s_prop", "AFR_DF_mean", "PMV_A",
    ]
    visit_audit = audit_df[[c for c in audit_cols if c in audit_df.columns]].copy()

    # Visit table on qualifying termination events
    agg_dict: dict[str, tuple[str, Any]] = {
        "n_events": ("time_to_reinit_sec", "count"),
        "reinit_360s_prop": ("time_to_reinit_sec", lambda s: float((s < 360).mean())),
    }
    if "AFR_DF_mean" in ir_ev.columns:
        agg_dict["AFR_DF_mean"] = ("AFR_DF_mean", "mean")
    if "PMV_A" in ir_ev.columns:
        agg_dict["PMV_A"] = ("PMV_A", "mean")
    visit_df = ir_ev.groupby(["patient", "record"], as_index=False).agg(**agg_dict)

    rng = np.random.default_rng(SEED)
    rows = []
    for metric in ["reinit_360s_prop", "AFR_DF_mean", "PMV_A"]:
        if metric not in visit_df.columns:
            continue
        icc, n_pts, n_visits = compute_icc_oneway(visit_df, "patient", metric)

        # Patient-bootstrap 95% CI: explicitly audit undefined (zero-variance) resamples!
        sub_m = visit_df.dropna(subset=["patient", metric])
        rep_pts_m = sub_m.groupby("patient").filter(lambda g: len(g) >= 2)["patient"].unique()
        grouped_m = {p: g for p, g in sub_m.groupby("patient") if p in rep_pts_m}
        boot_iccs = []
        n_boot_tot = 500
        n_boot_undef = 0
        if len(rep_pts_m) >= 3:
            for b in range(n_boot_tot):
                sampled = rng.choice(rep_pts_m, size=len(rep_pts_m), replace=True)
                bdf = pd.concat([grouped_m[p].assign(boot_pt=f"{i}_{p}") for i, p in enumerate(sampled)], ignore_index=True)
                b_icc, _, _ = compute_icc_oneway(bdf, "boot_pt", metric)
                if np.isfinite(b_icc):
                    boot_iccs.append(b_icc)
                else:
                    n_boot_undef += 1
        # Do NOT report a narrow finite CI if >5% of bootstrap samples have undefined zero variance
        if len(boot_iccs) >= 20 and n_boot_undef <= 0.05 * n_boot_tot:
            ci_lo = float(np.percentile(boot_iccs, 2.5))
            ci_hi = float(np.percentile(boot_iccs, 97.5))
        else:
            ci_lo = np.nan
            ci_hi = np.nan

        # Leave-one-patient-out ICC sensitivity
        lopo_vals = {}
        for p_excl in rep_pts_m:
            l_icc, _, _ = compute_icc_oneway(sub_m[sub_m["patient"] != p_excl], "patient", metric)
            lopo_vals[str(p_excl)] = l_icc
        finite_lopo = [v for v in lopo_vals.values() if np.isfinite(v)]
        icc_no_pt045 = lopo_vals.get("patient_045", np.nan)

        rows.append({
            "metric": metric,
            "icc_1_1": icc,
            "icc_boot_ci_lo": ci_lo,
            "icc_boot_ci_hi": ci_hi,
            "n_boot_total": n_boot_tot if len(rep_pts_m) >= 3 else 0,
            "n_boot_defined": len(boot_iccs),
            "n_boot_undefined_zero_var": n_boot_undef,
            "n_repeat_patients": n_pts,
            "n_visits": n_visits,
            "n_nonzero_prop_patients": int((sub_m[sub_m["patient"].isin(rep_pts_m)].groupby("patient")[metric].max() > 0).sum()) if (metric == "reinit_360s_prop" and len(rep_pts_m)) else np.nan,
            "icc_excluding_patient_045": icc_no_pt045,
            "lopo_icc_min": float(min(finite_lopo)) if finite_lopo else np.nan,
            "lopo_icc_max": float(max(finite_lopo)) if finite_lopo else np.nan,
            "n_lopo_zero_variance_nan": int(sum(not np.isfinite(v) for v in lopo_vals.values())),
        })
    return add_provenance(pd.DataFrame(rows), prov), add_provenance(visit_audit, prov)


def analyze_circadian(events: pd.DataFrame, prov: dict[str, str]) -> pd.DataFrame:
    hour_col = "clock_hour_t0" if "clock_hour_t0" in events.columns else "clock_hour"
    ev = events.dropna(subset=[hour_col, "time_to_reinit_sec", "patient", "cohort"]).copy()
    ev["hour_bin"] = pd.cut(
        ev[hour_col] % 24.0,
        bins=[0, 4, 8, 12, 16, 20, 24],
        labels=["00-04", "04-08", "08-12", "12-16", "16-20", "20-24"],
        include_lowest=True,
        right=False,
    )
    ev["reinit_360s_bin"] = (ev["time_to_reinit_sec"] < 360).astype(float)
    rad = 2.0 * np.pi * (ev[hour_col] % 24.0) / 24.0
    ev["sin_h"] = np.sin(rad)
    ev["cos_h"] = np.cos(rad)

    rows = []
    for cohort_name in ["POOLED", "IRIDIA", "SHDB", "LTAFDB"]:
        sub = ev if cohort_name == "POOLED" else ev[ev["cohort"] == cohort_name].copy()
        if len(sub) < 20:
            continue
        X_cols = ["sin_h", "cos_h"]
        X_df = sub[X_cols].copy()
        if cohort_name == "POOLED":
            for ch in ["SHDB", "LTAFDB"]:
                X_df[f"cohort_{ch}"] = (sub["cohort"] == ch).astype(float)
        X_df = sm.add_constant(X_df)

        # 1. Patient-clustered GEE binomial logit
        gee_amp = gee_acro = gee_p = np.nan
        try:
            gee = sm.GEE(
                sub["reinit_360s_bin"],
                X_df,
                groups=sub["patient"],
                family=sm.families.Binomial(),
                cov_struct=sm.cov_struct.Exchangeable(),
            ).fit()
            bs, bc = float(gee.params["sin_h"]), float(gee.params["cos_h"])
            gee_amp = float(np.hypot(bs, bc))
            gee_acro = float((np.arctan2(bs, bc) * 24.0 / (2.0 * np.pi)) % 24.0)
            gee_p = float(gee.wald_test("sin_h = 0, cos_h = 0", scalar=True).pvalue)
        except Exception:
            pass

        # 2. Random-intercept mixed logit (BinomialBayesMixedGLM)
        glmm_amp = glmm_acro = np.nan
        try:
            from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
            exog_vc = pd.get_dummies(sub["patient"].astype(str), dtype=float)
            ident = np.zeros(exog_vc.shape[1], dtype=int)
            glmm = BinomialBayesMixedGLM(sub["reinit_360s_bin"].values, X_df.values, exog_vc.values, ident).fit_vb()
            bs_g, bc_g = float(glmm.fe_mean[1]), float(glmm.fe_mean[2])
            glmm_amp = float(np.hypot(bs_g, bc_g))
            glmm_acro = float((np.arctan2(bs_g, bc_g) * 24.0 / (2.0 * np.pi)) % 24.0)
        except Exception:
            pass

        for hbin, g in sub.groupby("hour_bin", observed=True):
            rows.append({
                "cohort": cohort_name,
                "hour_bin": str(hbin),
                "n_events": int(len(g)),
                "n_patients": int(g["patient"].nunique()),
                "reinit_360s_prop": float(g["reinit_360s_bin"].mean()),
                "median_dur_sec": float(g["dur_sec"].median()) if "dur_sec" in g.columns else np.nan,
                "gee_cosinor_amplitude": gee_amp,
                "gee_cosinor_acrophase_h": gee_acro,
                "gee_cosinor_p": gee_p,
                "glmm_cosinor_amplitude": glmm_amp,
                "glmm_cosinor_acrophase_h": glmm_acro,
            })
    return add_provenance(pd.DataFrame(rows), prov)


def main() -> None:
    prov = get_provenance()
    run_dir = common.resolve_run_dir()
    out_dir = run_dir / "secondary"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs, events = load_merged_data(run_dir)
    shdb_sub, shdb_nonaf = analyze_shdb_subgroups(runs, events, prov)
    shdb_sub.to_csv(out_dir / "shdb_subgroups.csv", index=False)
    shdb_nonaf.to_csv(out_dir / "shdb_non_af_with_af.csv", index=False)

    rep_icc, rep_audit = analyze_iridia_repeats(runs, events, prov)
    rep_icc.to_csv(out_dir / "iridia_repeat_icc.csv", index=False)
    rep_audit.to_csv(out_dir / "iridia_repeat_visits_audit.csv", index=False)
    analyze_circadian(events, prov).to_csv(out_dir / "circadian_analysis.csv", index=False)
    print(f"[Secondary] Revision complete. Outputs written to {out_dir}")


if __name__ == "__main__":
    main()
