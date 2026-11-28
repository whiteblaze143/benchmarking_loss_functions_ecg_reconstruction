#!/usr/bin/env python3
"""
Stage 9 — Aim 3 (Revision): Clean Complete-Case Predictors of Early AF Reinitiation,
Eligibility Flow, Collinearity VIF, Cohort-Stratified AFT, Validation, Decisive Contrast,
and Frailty-Adjusted Two-State Summary.

Writes to <run_dir>/aim3/:
  - eligibility_flow.csv
  - collinearity_vif.csv
  - table3_primary_hrs.csv
  - stratified_aft_sensitivity.csv
  - tableS3_loco_validation.csv
  - loco_calibration_bins.csv
  - decisive_contrast.csv
  - two_state_summary.csv
  - descriptive_reinit_km.csv
  - shdb_clinical_model.csv
  - sensitivity_models.csv
  - km_reinit_curves.parquet
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from lifelines import CoxPHFitter, KaplanMeierFitter, WeibullAFTFitter, WeibullFitter
from lifelines.statistics import proportional_hazard_test
from lifelines.utils import concordance_index
from scipy import optimize, stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

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


def get_latest_run_dir() -> Path:
    return common.resolve_run_dir()


def add_provenance(df: pd.DataFrame, prov: dict[str, str]) -> pd.DataFrame:
    df = df.copy()
    for k, v in prov.items():
        df[k] = v
    return df


def patient_cluster_bootstrap_ci(
    df: pd.DataFrame,
    patient_col: str,
    stat_fn,
    n_boot: int = 1000,
    seed: int = SEED,
) -> tuple[float, float, float]:
    point = float(stat_fn(df))
    patients = df[patient_col].dropna().unique()
    if len(patients) < 2:
        return point, np.nan, np.nan
    rng = np.random.default_rng(seed)
    grouped = {p: g for p, g in df.groupby(patient_col, sort=False)}
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        sampled_pts = rng.choice(patients, size=len(patients), replace=True)
        boot_df = pd.concat([grouped[p] for p in sampled_pts], ignore_index=True)
        boots[b] = float(stat_fn(boot_df))
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def merge_event_markers(run_dir: Path, events_file: str = "events_g0.parquet") -> pd.DataFrame:
    ev_path = run_dir / "stage4" / events_file
    if not ev_path.exists():
        raise FileNotFoundError(f"Missing event table: {ev_path}")
    df = pd.read_parquet(ev_path)

    for sub_rel in ["stage5/terminal_markers.parquet", "stage6/post_markers.parquet"]:
        p = run_dir / sub_rel
        if not p.exists():
            continue
        m = pd.read_parquet(p)
        if "event_id" in df.columns and "event_id" in m.columns:
            keys = ["event_id"]
        else:
            keys = [c for c in ["record", "run_id"] if c in df.columns and c in m.columns]
        if not keys:
            continue
        new_cols = [c for c in m.columns if c not in df.columns or c in keys]
        m_sub = m[new_cols].drop_duplicates(subset=keys)
        df = df.merge(m_sub, on=keys, how="left")

    if "time_to_reinit_sec" not in df.columns and "next_gap_sec" in df.columns:
        df["time_to_reinit_sec"] = df["next_gap_sec"]
    if "event" not in df.columns:
        df["event"] = df["time_to_reinit_sec"].notna().astype(int)
    if "log_dur" not in df.columns and "dur_sec" in df.columns:
        df["log_dur"] = np.log1p(df["dur_sec"].clip(lower=0))

    hour_col = "clock_hour_t0" if "clock_hour_t0" in df.columns else ("clock_hour" if "clock_hour" in df.columns else None)
    if hour_col is not None:
        if "clock_sin" not in df.columns:
            df["clock_sin"] = df["cosinor_sin"] if "cosinor_sin" in df.columns else np.sin(2.0 * np.pi * df[hour_col] / 24.0)
        if "clock_cos" not in df.columns:
            df["clock_cos"] = df["cosinor_cos"] if "cosinor_cos" in df.columns else np.cos(2.0 * np.pi * df[hour_col] / 24.0)

    return df


PRIMARY_PREDICTORS = [
    "log_dur",
    "AFR_DF_mean",
    "AFR_DF_slope",
    "fwave_SampEn_mean",
    "PMV_A",
    "first_PAC_coupling_ms",
    "clock_sin",
    "clock_cos",
    "cum_af_sec_before_t0",
    "n_prior_terminations",
    "VCC",
    "RR_mean",
]

PAC_COUNT_PREDICTORS = [
    "PAC_count_A" if c == "first_PAC_coupling_ms" else c for c in PRIMARY_PREDICTORS
]

ECG_MARKERS_ONLY = [
    "AFR_DF_mean",
    "AFR_DF_slope",
    "fwave_SampEn_mean",
    "PMV_A",
    "PAC_count_A",
    "VCC",
    "RR_mean",
]

HISTORY_PREDICTORS = [
    "log_dur",
    "cum_af_sec_before_t0",
    "n_prior_terminations",
    "clock_sin",
    "clock_cos",
]


def compute_eligibility_flow(run_dir: Path, df: pd.DataFrame, prov: dict[str, str]) -> pd.DataFrame:
    """Strictly nested, monotone eligibility flow separating post-A beat sufficiency from 120-s landmarking."""
    runs_g0 = pd.read_parquet(run_dir / "stage2" / "runs_g0.parquet")

    sqi_t = df["sqi_pass_terminal"].eq(1) if "sqi_pass_terminal" in df.columns else pd.Series(False, index=df.index)
    sqi_a = df["sqi_pass_postA"].eq(1) if "sqi_pass_postA" in df.columns else pd.Series(False, index=df.index)
    insuf_a = df["postA_insufficient"].eq(0) if "postA_insufficient" in df.columns else pd.Series(False, index=df.index)
    untrunc_a = (df["postA_truncated"].eq(0) & (df["time_to_reinit_sec"] > 120.0)) if "postA_truncated" in df.columns else pd.Series(False, index=df.index)
    ge300 = df["dur_sec"] >= 300.0

    gate_3 = ge300
    gate_4 = gate_3 & sqi_t
    gate_5 = gate_4 & sqi_a & insuf_a
    gate_5b = gate_5 & df["PMV_A"].notna() if "PMV_A" in df.columns else gate_5
    gate_6_lm120 = gate_5b & untrunc_a

    cc12_mask = gate_5b.copy()
    for c in PRIMARY_PREDICTORS:
        if c in df.columns:
            cc12_mask &= df[c].notna()

    has_d = df["has_postD_window"].fillna(False).astype(bool) if "has_postD_window" in df.columns else pd.Series(False, index=df.index)
    sqi_d = df["sqi_pass_postD"].eq(1) if "sqi_pass_postD" in df.columns else pd.Series(False, index=df.index)
    pmvd_ok = df["PMV_D"].notna() if "PMV_D" in df.columns else pd.Series(False, index=df.index)
    lm720 = df["time_to_reinit_sec"] > 720.0
    win_d_lm720 = gate_5b & has_d & sqi_d & pmvd_ok & lm720

    gates = [
        ("1_stage2_runs_g0", runs_g0, None),
        ("2_stage4_qualifying_terminations_ge30s", df, pd.Series(True, index=df.index)),
        ("3_full_terminal_window_ge300s", df, gate_3),
        ("4_terminal_sqi_pass", df, gate_4),
        ("5_postA_sqi_and_sufficient_sinus_beats", df, gate_5),
        ("5b_marker_eligible_pmva_nonnull", df, gate_5b),
        ("6_landmark_120s_untruncated_postA", df, gate_6_lm120),
        ("7_complete_case_with_pac_coupling", df, cc12_mask),
        ("8_landmark_720s_window_D_h4b", df, win_d_lm720),
    ]

    rows = []
    for gate_name, base_df, mask in gates:
        sub = base_df if mask is None else base_df[mask]
        for cohort in ["POOLED", "IRIDIA", "SHDB", "LTAFDB"]:
            c_sub = sub if cohort == "POOLED" else sub[sub["cohort"] == cohort]
            rows.append({
                "gate": gate_name,
                "cohort": cohort,
                "n_events": int(len(c_sub)),
                "n_patients": int(c_sub["patient"].nunique()) if len(c_sub) else 0,
            })
    return add_provenance(pd.DataFrame(rows), prov)


def compute_collinearity_vif(df: pd.DataFrame, prov: dict[str, str]) -> pd.DataFrame:
    cols = [c for c in ["cum_af_sec_before_t0", "n_prior_terminations", "position_in_record_frac", "log_dur"] if c in df.columns]
    sub = df[cols].dropna().copy()
    if len(sub) < 10 or len(cols) < 2:
        return add_provenance(pd.DataFrame(), prov)
    # Standardize for stable VIF
    X = (sub - sub.mean()) / sub.std(ddof=0)
    X_const = sm.add_constant(X)
    rows = []
    for i, c in enumerate(cols):
        vif = float(variance_inflation_factor(X_const.values, i + 1))
        rec: dict[str, Any] = {"variable": c, "n": int(len(sub)), "vif": vif}
        for other in cols:
            if other == c:
                continue
            r_p, _ = stats.pearsonr(sub[c], sub[other])
            r_s, _ = stats.spearmanr(sub[c], sub[other])
            rec[f"pearson_{other}"] = float(r_p)
            rec[f"spearman_{other}"] = float(r_s)
        rows.append(rec)
    return add_provenance(pd.DataFrame(rows), prov)


def compute_descriptive_km(
    df: pd.DataFrame,
    prov: dict[str, str],
    df_cs: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    km_curves = []
    thresholds = [30, 60, 360, 3600]

    def _fast_km_at_thresholds(times: np.ndarray, events: np.ndarray, thrs: list[int]) -> np.ndarray:
        order = np.argsort(times, kind="mergesort")
        t_s, e_s = times[order], events[order]
        at_risk = np.arange(len(t_s), 0, -1, dtype=float)
        mask = (t_s <= max(thrs)) & (e_s == 1)
        t_ev, ar_ev = t_s[mask], at_risk[mask]
        if len(t_ev) == 0:
            return np.zeros(len(thrs), dtype=float)
        uniq_t, idx, counts = np.unique(t_ev, return_index=True, return_counts=True)
        surv_u = np.cumprod(1.0 - counts / ar_ev[idx])
        out = np.empty(len(thrs), dtype=float)
        for i, thr in enumerate(thrs):
            pos = np.searchsorted(uniq_t, thr, side="right") - 1
            out[i] = 1.0 - (surv_u[pos] if pos >= 0 else 1.0)
        return out

    def _patient_weighted_reinit(sub_df: pd.DataFrame, thr: int) -> tuple[float, int]:
        t = sub_df["time_to_reinit_sec"].to_numpy(float)
        ev = sub_df["event"].to_numpy(int)
        # Decidable at thr: either observed reinit <= thr, OR followed past thr
        dec = ((ev == 1) & (t <= thr)) | (t > thr)
        if not np.any(dec):
            return np.nan, 0
        s_dec = sub_df.loc[dec].copy()
        s_dec["y"] = ((s_dec["event"] == 1) & (s_dec["time_to_reinit_sec"] <= thr)).astype(float)
        pt_m = s_dec.groupby("patient")["y"].mean()
        return float(pt_m.mean()), int(len(pt_m))

    cohort_groups: list[tuple[str, str, pd.DataFrame]] = [("harmonized_annotation", "POOLED", df)] + [
        ("harmonized_annotation", str(c), g) for c, g in df.groupby("cohort", sort=True)
    ]
    if df_cs is not None and len(df_cs) > 0:
        for c in ["SHDB", "LTAFDB", "POOLED"]:
            sub_c = df_cs if c == "POOLED" else df_cs[df_cs["cohort"] == c]
            if len(sub_c) >= 10:
                cohort_groups.append(("continuity_nonsinus_gaps", str(c), sub_c))

    for tier_name, cohort_name, sub_raw in cohort_groups:
        sub = sub_raw.dropna(subset=["time_to_reinit_sec", "event"]).copy()
        sub = sub[sub["time_to_reinit_sec"] >= 0]
        if len(sub) == 0:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(sub["time_to_reinit_sec"], event_observed=sub["event"])
        if tier_name == "harmonized_annotation":
            grid = np.unique(np.concatenate([
                np.array([0, 10, 30, 60, 120, 300, 360, 600, 1200, 1800, 3600, 7200, 14400, 28800, 43200, 86400]),
                np.quantile(sub["time_to_reinit_sec"], np.linspace(0, 1, 50)),
            ]))
            grid = grid[grid <= sub["time_to_reinit_sec"].max()]
            sf = kmf.survival_function_at_times(grid).values
            for t, s in zip(grid, sf):
                km_curves.append({"cohort": cohort_name, "stratum": "all", "time_sec": float(t), "survival_prob": float(s), "reinit_prob": float(1.0 - s)})

        rec: dict[str, Any] = {
            "rhythm_tier": tier_name,
            "cohort": cohort_name,
            "n_events": int(len(sub)),
            "n_patients": int(sub["patient"].nunique()),
            "n_reinit_observed": int(sub["event"].sum()),
            "median_reinit_sec": float(kmf.median_survival_time_) if np.isfinite(kmf.median_survival_time_) else np.nan,
        }
        pt_arrays = [(g["time_to_reinit_sec"].to_numpy(float), g["event"].to_numpy(int)) for _, g in sub.groupby("patient", sort=False)]
        point_vals = _fast_km_at_thresholds(sub["time_to_reinit_sec"].to_numpy(float), sub["event"].to_numpy(int), thresholds)
        rng = np.random.default_rng(SEED)
        n_pts = len(pt_arrays)
        boots = np.empty((1000, len(thresholds)), dtype=float)
        if n_pts >= 2:
            for b in range(1000):
                idx_pts = rng.integers(0, n_pts, size=n_pts)
                boots[b] = _fast_km_at_thresholds(np.concatenate([pt_arrays[i][0] for i in idx_pts]), np.concatenate([pt_arrays[i][1] for i in idx_pts]), thresholds)
            lo_vals = np.nanpercentile(boots, 2.5, axis=0)
            hi_vals = np.nanpercentile(boots, 97.5, axis=0)
        else:
            lo_vals = hi_vals = np.full(len(thresholds), np.nan)
        for i, thr in enumerate(thresholds):
            rec[f"reinit_{thr}s_km"] = float(point_vals[i])
            rec[f"reinit_{thr}s_ci_lo"] = float(lo_vals[i])
            rec[f"reinit_{thr}s_ci_hi"] = float(hi_vals[i])
            pw_val, pw_n = _patient_weighted_reinit(sub, thr)
            rec[f"reinit_{thr}s_patient_weighted"] = pw_val
            rec[f"reinit_{thr}s_n_patients_decidable"] = pw_n
        rows.append(rec)

    # KM curves by PMV_A and AFR_DF_mean tertiles on clean marker-eligible events
    sqi_t = df["sqi_pass_terminal"].eq(1) if "sqi_pass_terminal" in df.columns else pd.Series(False, index=df.index)
    sqi_a = df["sqi_pass_postA"].eq(1) if "sqi_pass_postA" in df.columns else pd.Series(False, index=df.index)
    insuf_a = df["postA_insufficient"].eq(0) if "postA_insufficient" in df.columns else pd.Series(False, index=df.index)
    clean_df = df[(df["dur_sec"] >= 300.0) & sqi_t & sqi_a & insuf_a].copy()
    for marker in ["PMV_A", "AFR_DF_mean"]:
        if marker in clean_df.columns and clean_df[marker].notna().sum() >= 30:
            sub = clean_df.dropna(subset=["time_to_reinit_sec", "event", marker]).copy()
            try:
                sub["tertile"] = pd.qcut(sub[marker], q=3, labels=["T1_low", "T2_mid", "T3_high"], duplicates="drop")
            except Exception:
                continue
            for tlabel, g in sub.groupby("tertile", observed=True):
                kmf = KaplanMeierFitter()
                kmf.fit(g["time_to_reinit_sec"], event_observed=g["event"])
                grid = np.array([0, 15, 30, 60, 120, 180, 300, 360, 600, 1200, 1800, 3600, 7200, 14400, 28800])
                grid = grid[grid <= g["time_to_reinit_sec"].max()]
                sf = kmf.survival_function_at_times(grid).values
                for t, s in zip(grid, sf):
                    km_curves.append({"cohort": "POOLED", "stratum": f"{marker}:{tlabel}", "time_sec": float(t), "survival_prob": float(s), "reinit_prob": float(1.0 - s)})

    return add_provenance(pd.DataFrame(rows), prov), pd.DataFrame(km_curves)


def prepare_complete_case_df(
    df: pd.DataFrame,
    predictors: list[str],
    standardize: bool = True,
    landmark_offset_sec: float = 0.0,
) -> tuple[pd.DataFrame, list[str]]:
    """Strict complete-case preparation (ZERO imputation)."""
    present = [c for c in predictors if c in df.columns]
    sub = df.dropna(subset=["time_to_reinit_sec", "event", "patient", "cohort"]).copy()
    if landmark_offset_sec > 0.0:
        sub = sub[sub["time_to_reinit_sec"] > landmark_offset_sec].copy()
        sub["time_to_reinit_sec"] = sub["time_to_reinit_sec"] - landmark_offset_sec
    else:
        sub = sub[sub["time_to_reinit_sec"] > 0].copy()

    for col in present:
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
    sub = sub.dropna(subset=present)

    usable = []
    for col in present:
        std = float(sub[col].std(ddof=0)) if len(sub) > 1 else 0.0
        if not np.isfinite(std) or std < 1e-12:
            continue
        if standardize:
            sub[col] = (sub[col] - float(sub[col].mean())) / std
        usable.append(col)
    return sub, usable


def fit_cox_clean(
    df: pd.DataFrame,
    predictors: list[str],
    model_name: str,
    strata: list[str] | None = None,
    penalizer: float = 0.01,
    landmark_offset_sec: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, CoxPHFitter | None, list[str]]:
    """
    Fit stratified, patient-clustered Cox PH on complete cases (zero imputation).
    Returns (cox_df, stratified_aft_df, cph, usable_predictors).
    """
    sub, usable = prepare_complete_case_df(
        df, predictors, standardize=True, landmark_offset_sec=landmark_offset_sec
    )
    if len(sub) < 25 or len(usable) == 0:
        return pd.DataFrame(), pd.DataFrame(), None, []

    cols = ["time_to_reinit_sec", "event", "patient"] + usable
    if strata:
        cols += [s for s in strata if s not in cols]
    fit_df = sub[cols].copy()

    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(fit_df, duration_col="time_to_reinit_sec", event_col="event", strata=strata, cluster_col="patient", robust=True)

    # Schoenfeld PH test (per-term and global)
    ph_pvals: dict[str, float] = {}
    global_ph_p = np.nan
    any_ph_violated = False
    try:
        ph_res = proportional_hazard_test(cph, fit_df, time_transform="rank")
        for var in usable:
            if var in ph_res.summary.index:
                pv = float(ph_res.summary.loc[var, "p"])
                ph_pvals[var] = pv
                if pv < 0.05:
                    any_ph_violated = True
        if len(ph_pvals):
            # Fisher's method / min-p Bonferroni proxy for global PH p-value
            chi2_stat = float(ph_res.summary["test_statistic"].sum())
            global_ph_p = float(stats.chi2.sf(chi2_stat, df=len(ph_res.summary)))
    except Exception:
        pass

    # Cohort-stratified Weibull AFT sensitivity (cohort dummies in both mu_ and ancillary!)
    aft_rows = []
    always_aft = model_name in ("primary_marker_eligible_complete_case", "marker_eligible_449_pac_count")
    if (any_ph_violated or always_aft) and strata and "cohort" in fit_df.columns:
        try:
            aft_df = fit_df[["time_to_reinit_sec", "event"] + usable].copy()
            cohorts = sorted(fit_df["cohort"].dropna().unique())
            dummy_cols = []
            for ch in cohorts[1:]:
                dcol = f"cohort_{ch}"
                aft_df[dcol] = (fit_df["cohort"] == ch).astype(float).values
                dummy_cols.append(dcol)
            aft = WeibullAFTFitter(penalizer=penalizer)
            aft.fit(aft_df, duration_col="time_to_reinit_sec", event_col="event", ancillary=True)
            lam_sum = aft.summary.loc["lambda_"]
            for var in usable + dummy_cols:
                if var in lam_sum.index:
                    r = lam_sum.loc[var]
                    aft_rows.append({
                        "model": model_name,
                        "predictor": var,
                        "n_events": int(len(aft_df)),
                        "n_reinit": int(aft_df["event"].sum()),
                        "aft_coef": float(r["coef"]),
                        "aft_time_ratio": float(r["exp(coef)"]),
                        "aft_tr_ci_lo": float(r["exp(coef) lower 95%"]),
                        "aft_tr_ci_hi": float(r["exp(coef) upper 95%"]),
                        "aft_p": float(r["p"]),
                    })
        except Exception:
            pass

    c_idx = float(cph.concordance_index_)
    rows = []
    for var in usable:
        row = cph.summary.loc[var]
        pv = ph_pvals.get(var, np.nan)
        rows.append({
            "model": model_name,
            "predictor": var,
            "n_events": int(len(fit_df)),
            "n_reinit": int(fit_df["event"].sum()),
            "n_patients": int(fit_df["patient"].nunique()),
            "coef": float(row["coef"]),
            "se_robust": float(row["se(coef)"]),
            "hr_per_sd": float(row["exp(coef)"]),
            "hr_ci_lo": float(row["exp(coef) lower 95%"]),
            "hr_ci_hi": float(row["exp(coef) upper 95%"]),
            "wald_p": float(row["p"]),
            "schoenfeld_p": pv,
            "term_ph_violated": bool(pv < 0.05) if np.isfinite(pv) else False,
            "global_ph_p": global_ph_p,
            "c_index": c_idx,
        })
    return pd.DataFrame(rows), pd.DataFrame(aft_rows), cph, usable


def fit_cohort_x_logdur_spline(df: pd.DataFrame) -> pd.DataFrame:
    """Natural cubic spline basis for log_dur + cohort x log_dur interaction."""
    sub = df.dropna(subset=["time_to_reinit_sec", "event", "patient", "cohort", "log_dur"]).copy()
    sub = sub[sub["time_to_reinit_sec"] > 0].copy()
    if len(sub) < 50:
        return pd.DataFrame()
    ld = sub["log_dur"].values
    q25, q50, q75 = np.percentile(ld, [25, 50, 75])
    # Truncated power spline basis (2 nonlinear terms)
    sub["log_dur_sp1"] = np.maximum(0.0, ld - q25) ** 3
    sub["log_dur_sp2"] = np.maximum(0.0, ld - q75) ** 3
    sub["log_dur_x_SHDB"] = ld * (sub["cohort"] == "SHDB").astype(float)
    sub["log_dur_x_LTAFDB"] = ld * (sub["cohort"] == "LTAFDB").astype(float)
    preds = ["log_dur", "log_dur_sp1", "log_dur_sp2", "log_dur_x_SHDB", "log_dur_x_LTAFDB", "cum_af_sec_before_t0", "clock_sin", "clock_cos"]
    res, _, _, _ = fit_cox_clean(sub, preds, "cohort_x_log_dur_and_spline", strata=["cohort"], penalizer=0.05)
    return res


def run_validation_clean(
    df: pd.DataFrame,
    model_specs: list[tuple[str, pd.DataFrame, list[str]]],
    prov: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    val_rows = []
    cal_rows = []

    for spec_name, spec_df, preds in model_specs:
        sub, usable = prepare_complete_case_df(spec_df, preds, standardize=True)
        if len(sub) < 40 or len(usable) == 0:
            continue

        def _eval_split(train_df: pd.DataFrame, test_df: pd.DataFrame, split_type: str, split_name: str):
            if len(train_df) < 25 or len(test_df) < 10 or test_df["event"].sum() < 5:
                return
            cols = ["time_to_reinit_sec", "event"] + usable
            cph = CoxPHFitter(penalizer=0.05)
            try:
                cph.fit(train_df[cols], duration_col="time_to_reinit_sec", event_col="event")
                lp = cph.predict_log_partial_hazard(test_df[usable]).values
                c_index = float(concordance_index(test_df["time_to_reinit_sec"], -lp, test_df["event"]))
                cal_df = pd.DataFrame({"time_to_reinit_sec": test_df["time_to_reinit_sec"].values, "event": test_df["event"].values, "lp": lp})
                cal_cph = CoxPHFitter(penalizer=0.001)
                cal_cph.fit(cal_df, duration_col="time_to_reinit_sec", event_col="event")
                val_rows.append({
                    "model_spec": spec_name,
                    "validation_type": split_type,
                    "split": split_name,
                    "n_train": int(len(train_df)),
                    "n_test": int(len(test_df)),
                    "n_test_events": int(test_df["event"].sum()),
                    "n_test_patients": int(test_df["patient"].nunique()),
                    "harrell_c": c_index,
                    "calibration_slope": float(cal_cph.params_["lp"]),
                    "calibration_slope_se": float(cal_cph.standard_errors_["lp"]),
                })
                if split_type == "LOCO" and spec_name == "marker_eligible_449_pac_count":
                    surv_360 = cph.predict_survival_function(test_df[usable], times=[360.0]).values.ravel()
                    pred_risk = 1.0 - surv_360
                    tmp = pd.DataFrame({"pred_risk_360s": pred_risk, "time_to_reinit_sec": test_df["time_to_reinit_sec"].values, "event": test_df["event"].values})
                    try:
                        tmp["bin"] = pd.qcut(tmp["pred_risk_360s"], q=5, labels=False, duplicates="drop")
                    except Exception:
                        tmp["bin"] = 0
                    for b, g in tmp.groupby("bin"):
                        kmf = KaplanMeierFitter()
                        kmf.fit(g["time_to_reinit_sec"], event_observed=g["event"])
                        cal_rows.append({
                            "held_out_cohort": split_name,
                            "bin": int(b),
                            "n": int(len(g)),
                            "pred_risk_360s": float(g["pred_risk_360s"].mean()),
                            "obs_risk_360s": 1.0 - float(kmf.predict(360.0)),
                        })
            except Exception:
                pass

        rng = np.random.default_rng(SEED)
        patients = sub["patient"].unique()
        rng.shuffle(patients)
        for k, test_pts in enumerate(np.array_split(patients, 5)):
            mask = sub["patient"].isin(test_pts)
            _eval_split(sub[~mask], sub[mask], "5fold_patient_CV", f"fold_{k+1}")
        for cohort in sorted(sub["cohort"].unique()):
            mask = sub["cohort"] == cohort
            _eval_split(sub[~mask], sub[mask], "LOCO", str(cohort))

    return add_provenance(pd.DataFrame(val_rows), prov), add_provenance(pd.DataFrame(cal_rows), prov)


def compute_decisive_contrast(run_dir: Path, prov: dict[str, str]) -> pd.DataFrame:
    rec_path = run_dir / "aim2" / "reconstructed_episodes.parquet"
    if not rec_path.exists():
        return add_provenance(pd.DataFrame(), prov)
    rec = pd.read_parquet(rec_path)
    sub = rec[rec["gap_sec"] == 0].copy() if "gap_sec" in rec.columns else rec.copy()
    sub = sub[sub["dur_sec"] >= 3600.0].copy()
    if len(sub) == 0:
        return add_provenance(pd.DataFrame(), prov)

    rows = []
    for cohort_name, g in [("POOLED", sub)] + list(sub.groupby("cohort", sort=True)):
        def _share(d):
            tot = d["dur_sec"].sum()
            return float(d["af_sec_reinit_chain"].sum() / tot) if tot > 0 else np.nan
        est, lo, hi = patient_cluster_bootstrap_ci(g, "patient", _share, n_boot=1000, seed=SEED)
        rows.append({
            "cohort": cohort_name,
            "n_episodes_ge_1h": int(len(g)),
            "n_patients": int(g["patient"].nunique()),
            "total_af_hours": float(g["dur_sec"].sum() / 3600.0),
            "chain_af_hours": float(g["af_sec_reinit_chain"].sum() / 3600.0),
            "uninterrupted_af_hours": float((g["dur_sec"].sum() - g["af_sec_reinit_chain"].sum()) / 3600.0),
            "share_reinit_chain_360s": est,
            "ci_lo": lo,
            "ci_hi": hi,
        })
    return add_provenance(pd.DataFrame(rows), prov)


def fit_alternating_hawkes(event_times: np.ndarray, T_max: float) -> dict[str, Any]:
    t = np.sort(event_times[(event_times > 0) & (event_times < T_max)])
    n = len(t)
    if n < 5 or T_max <= 0:
        return {"converged": False, "branching_ratio": np.nan}

    def neg_loglik(params):
        log_mu, log_alpha, log_beta = params
        mu, alpha, beta = np.exp(log_mu), np.exp(log_alpha), np.exp(log_beta)
        if alpha >= beta * 0.999:
            return 1e12
        R, ll, prev_t = 0.0, 0.0, 0.0
        for ti in t:
            dt = ti - prev_t
            R = np.exp(-beta * dt) * (R + (1.0 if prev_t > 0 else 0.0))
            lam = mu + alpha * R
            if lam <= 1e-15:
                return 1e12
            ll += np.log(lam)
            prev_t = ti
        integral = mu * T_max + (alpha / beta) * np.sum(1.0 - np.exp(-beta * (T_max - t)))
        return -(ll - integral)

    init_mu = max(n / T_max * 0.5, 1e-6)
    res = optimize.minimize(neg_loglik, np.log([init_mu, 0.002, 0.005]), method="Nelder-Mead", options={"maxiter": 1000})
    mu, alpha, beta = np.exp(res.x)
    conv = bool(res.success and np.isfinite(res.fun) and res.fun < 1e11 and beta > 0)
    return {"converged": conv, "branching_ratio": float(alpha / beta) if conv else np.nan}


def compute_two_state_summary(run_dir: Path, df: pd.DataFrame, prov: dict[str, str]) -> pd.DataFrame:
    aim1_mod = importlib.import_module("07_aim1_termination_hazard")
    fit_frailty = aim1_mod.fit_shared_gamma_frailty_weibull

    aim1_path = run_dir / "aim1" / "tableS1_weibull_frailty.csv"
    aim1 = pd.read_csv(aim1_path) if aim1_path.exists() else pd.DataFrame()

    rows = []
    for cohort_name, sub in [("POOLED", df)] + list(df.groupby("cohort", sort=True)):
        sub_valid = sub.dropna(subset=["time_to_reinit_sec", "event", "patient"])
        sub_valid = sub_valid[sub_valid["time_to_reinit_sec"] > 0]
        if len(sub_valid) < 10:
            continue

        # Unadjusted Weibull
        wf = WeibullFitter()
        wf.fit(sub_valid["time_to_reinit_sec"], event_observed=sub_valid["event"])
        sr_unadj_shape = float(wf.rho_)

        # Patient-level shared-gamma-frailty Weibull on SR-reinitiation times
        fr = fit_frailty(
            sub_valid["time_to_reinit_sec"].to_numpy(float),
            sub_valid["event"].to_numpy(int),
            sub_valid["patient"].to_numpy(),
        )
        sr_fr_k = float(fr.get("shape_k", np.nan))
        sr_fr_lo = float(fr.get("shape_ci_low", np.nan))
        sr_fr_hi = float(fr.get("shape_ci_high", np.nan))
        sr_fr_theta = float(fr.get("frailty_theta", np.nan))

        # Left-truncated sensitivity at >= 60 s (t_min=60.0)
        sub_lt60 = sub_valid[sub_valid["time_to_reinit_sec"] > 60.0]
        if len(sub_lt60) >= 10:
            try:
                fr60 = fit_frailty(
                    sub_lt60["time_to_reinit_sec"].to_numpy(float),
                    sub_lt60["event"].to_numpy(int),
                    sub_lt60["patient"].to_numpy(),
                    t_min=60.0,
                )
            except TypeError:
                fr60 = fit_frailty(
                    sub_lt60["time_to_reinit_sec"].to_numpy(float),
                    sub_lt60["event"].to_numpy(int),
                    sub_lt60["patient"].to_numpy(),
                )
            lt60_k = float(fr60.get("shape_k", np.nan))
            lt60_lo = float(fr60.get("shape_ci_low", np.nan))
            lt60_hi = float(fr60.get("shape_ci_high", np.nan))
            lt60_theta = float(fr60.get("frailty_theta", np.nan))
        else:
            lt60_k = lt60_lo = lt60_hi = lt60_theta = np.nan

        # Aim 1 AF-termination frailty Weibull shape
        af_shape = np.nan
        if not aim1.empty and "cohort" in aim1.columns:
            m = aim1[aim1["cohort"].astype(str).str.upper() == str(cohort_name).upper()]
            if "sensitivity" in m.columns and (m["sensitivity"] == "primary_g0").any():
                m = m[m["sensitivity"] == "primary_g0"]
            if not m.empty:
                af_shape = float(m["shape_k"].iloc[0])

        # Hawkes audit (Henriksson 2021)
        n_rec_total = int(sub["record"].nunique())
        n_rec_ge10 = 0
        n_rec_conv = 0
        n_rec_fail = 0
        br_list = []
        t_col = "t0" if "t0" in sub.columns else "offset_sec"
        for _, g in sub.groupby("record"):
            if len(g) >= 10:
                n_rec_ge10 += 1
                times = g[t_col].dropna().to_numpy(float)
                hk = fit_alternating_hawkes(times, float(times.max() + 3600.0))
                if hk["converged"] and np.isfinite(hk["branching_ratio"]):
                    n_rec_conv += 1
                    br_list.append(hk["branching_ratio"])
                else:
                    n_rec_fail += 1

        rows.append({
            "cohort": cohort_name,
            "n_events": int(len(sub_valid)),
            "af_termination_frailty_shape_k": af_shape,
            "sr_reinit_unadj_shape": sr_unadj_shape,
            "sr_reinit_frailty_shape_k": sr_fr_k,
            "sr_reinit_shape_ci_lo": sr_fr_lo,
            "sr_reinit_shape_ci_hi": sr_fr_hi,
            "sr_reinit_frailty_theta": sr_fr_theta,
            "sr_reinit_lt60_n_runs": int(len(sub_lt60)),
            "sr_reinit_lt60_n_events": int(sub_lt60["event"].sum()),
            "sr_reinit_lt60_shape_k": lt60_k,
            "sr_reinit_lt60_ci_lo": lt60_lo,
            "sr_reinit_lt60_ci_hi": lt60_hi,
            "sr_reinit_lt60_frailty_theta": lt60_theta,
            "both_decreasing_hazard": bool(np.isfinite(af_shape) and np.isfinite(sr_fr_k) and af_shape < 1.0 and sr_fr_k < 1.0),
            "hawkes_n_records_total": n_rec_total,
            "hawkes_n_records_ge10_events": n_rec_ge10,
            "hawkes_n_records_converged": n_rec_conv,
            "hawkes_n_fit_failures": n_rec_fail,
            "hawkes_median_branching_ratio": float(np.median(br_list)) if br_list else np.nan,
        })
    return add_provenance(pd.DataFrame(rows), prov)


def run_shdb_clinical_model(df: pd.DataFrame, prov: dict[str, str]) -> pd.DataFrame:
    shdb = df[df["cohort"] == "SHDB"].copy()
    if len(shdb) < 30:
        return add_provenance(pd.DataFrame(), prov)
    if "shdb_sex" in shdb.columns:
        shdb["sex_male"] = (shdb["shdb_sex"].astype(str).str.upper() == "M").astype(float)
    if "shdb_prev_ablation" in shdb.columns:
        shdb["prior_ablation"] = shdb["shdb_prev_ablation"].astype(str).str.lower().isin(["true", "1", "yes"]).astype(float)
    if "shdb_af_type" in shdb.columns:
        shdb["af_type_persistent"] = shdb["shdb_af_type"].astype(str).str.upper().str.contains("PERS|PERM").astype(float)

    base_preds = [c for c in HISTORY_PREDICTORS if c in shdb.columns]
    clin_preds = [c for c in ["shdb_age", "sex_male", "shdb_bmi", "prior_ablation", "af_type_persistent", "shdb_echo_lad"] if c in shdb.columns]
    base_res, _, base_cph, _ = fit_cox_clean(shdb, base_preds, "SHDB_history_base", strata=None)
    full_res, _, full_cph, _ = fit_cox_clean(shdb, base_preds + clin_preds, "SHDB_history_plus_clinical", strata=None)
    if not full_res.empty and base_cph is not None and full_cph is not None:
        full_res["delta_c_index_vs_base"] = float(full_cph.concordance_index_ - base_cph.concordance_index_)
    return add_provenance(pd.concat([base_res, full_res], ignore_index=True), prov)


def compute_within_between_decomposition(
    df: pd.DataFrame,
    marker_clean_df: pd.DataFrame,
    prov: dict[str, str],
    df_cs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Output #1: Patient-stratified and within/between patient-mean-centered Cox for log_dur & cum_af_sec_before_t0."""
    cc336, _ = prepare_complete_case_df(marker_clean_df, PRIMARY_PREDICTORS, standardize=False)
    cc449, _ = prepare_complete_case_df(marker_clean_df, PAC_COUNT_PREDICTORS, standardize=False)
    lm120_cc449 = cc449[(cc449["postA_truncated"].eq(0)) & (cc449["time_to_reinit_sec"] > 120.0)].copy() if "postA_truncated" in cc449.columns else cc449.iloc[0:0].copy()
    if len(lm120_cc449):
        lm120_cc449["time_to_reinit_sec"] = lm120_cc449["time_to_reinit_sec"] - 120.0
    subsets = [
        ("clean_336", cc336),
        ("clean_449", cc449),
        ("landmark_120s_matched", lm120_cc449),
        ("clean_IRIDIA", marker_clean_df[marker_clean_df["cohort"] == "IRIDIA"].copy()),
        ("all_events_g0", df.copy()),
        ("all_IRIDIA", df[df["cohort"] == "IRIDIA"].copy()),
    ]
    if df_cs is not None and len(df_cs) >= 20:
        subsets.append(("continuity_nonsinus_gaps", df_cs.copy()))
    vars_pair = ["log_dur", "cum_af_sec_before_t0"]
    rows = []

    for sub_name, raw_sub in subsets:
        sub = raw_sub.dropna(subset=["time_to_reinit_sec", "event", "patient", "cohort"] + vars_pair).copy()
        sub = sub[sub["time_to_reinit_sec"] > 0].copy()
        if len(sub) < 20:
            continue

        # Scale by subset overall SD
        sds = {v: float(sub[v].std(ddof=0)) for v in vars_pair}
        means = {v: float(sub[v].mean()) for v in vars_pair}
        for v in vars_pair:
            if sds[v] <= 1e-12:
                continue
            sub[f"{v}_z"] = (sub[v] - means[v]) / sds[v]
            pt_mean = sub.groupby("patient")[v].transform("mean")
            sub[f"{v}_between"] = (pt_mean - means[v]) / sds[v]
            sub[f"{v}_within"] = (sub[v] - pt_mean) / sds[v]

        # (a) Patient-stratified Cox (patients with >= 2 events)
        pt_counts = sub["patient"].value_counts()
        multi_pts = pt_counts[pt_counts >= 2].index
        sub_ps = sub[sub["patient"].isin(multi_pts)].copy()
        if len(sub_ps) >= 15 and sub_ps["event"].sum() >= 5:
            for spec_name, cols_fit in [
                ("joint", ["log_dur_z", "cum_af_sec_before_t0_z"]),
                ("single_log_dur", ["log_dur_z"]),
                ("single_cum_af", ["cum_af_sec_before_t0_z"]),
            ]:
                try:
                    cph = CoxPHFitter(penalizer=0.01)
                    fit_df = sub_ps[["time_to_reinit_sec", "event", "patient"] + cols_fit].copy()
                    cph.fit(fit_df, duration_col="time_to_reinit_sec", event_col="event", strata=["patient"])
                    ph_p = {}
                    try:
                        ph_res = proportional_hazard_test(cph, fit_df, time_transform="rank")
                        for c in cols_fit:
                            if c in ph_res.summary.index:
                                ph_p[c] = float(ph_res.summary.loc[c, "p"])
                    except Exception:
                        pass
                    for c in cols_fit:
                        r = cph.summary.loc[c]
                        rows.append({
                            "subset": sub_name,
                            "model_type": "patient_stratified",
                            "spec": spec_name,
                            "predictor": c.replace("_z", ""),
                            "n_events": int(len(fit_df)),
                            "n_reinit": int(fit_df["event"].sum()),
                            "n_patients": int(fit_df["patient"].nunique()),
                            "hr_per_sd": float(r["exp(coef)"]),
                            "hr_ci_lo": float(r["exp(coef) lower 95%"]),
                            "hr_ci_hi": float(r["exp(coef) upper 95%"]),
                            "wald_p": float(r["p"]),
                            "schoenfeld_p": ph_p.get(c, np.nan),
                        })
                except Exception:
                    pass

        # (b) Within-between patient-mean-centered Cox
        use_strata = ["cohort"] if sub["cohort"].nunique() > 1 else None
        for spec_name, cols_fit in [
            ("joint", ["log_dur_between", "log_dur_within", "cum_af_sec_before_t0_between", "cum_af_sec_before_t0_within"]),
            ("single_log_dur", ["log_dur_between", "log_dur_within"]),
            ("single_cum_af", ["cum_af_sec_before_t0_between", "cum_af_sec_before_t0_within"]),
        ]:
            try:
                cph = CoxPHFitter(penalizer=0.01)
                fit_cols = ["time_to_reinit_sec", "event", "patient"] + (use_strata or []) + cols_fit
                fit_df = sub[fit_cols].copy()
                cph.fit(fit_df, duration_col="time_to_reinit_sec", event_col="event", strata=use_strata, cluster_col="patient", robust=True)
                ph_p = {}
                try:
                    ph_res = proportional_hazard_test(cph, fit_df, time_transform="rank")
                    for c in cols_fit:
                        if c in ph_res.summary.index:
                            ph_p[c] = float(ph_res.summary.loc[c, "p"])
                except Exception:
                    pass
                for c in cols_fit:
                    r = cph.summary.loc[c]
                    rows.append({
                        "subset": sub_name,
                        "model_type": "within_between",
                        "spec": spec_name,
                        "predictor": c,
                        "n_events": int(len(fit_df)),
                        "n_reinit": int(fit_df["event"].sum()),
                        "n_patients": int(fit_df["patient"].nunique()),
                        "hr_per_sd": float(r["exp(coef)"]),
                        "hr_ci_lo": float(r["exp(coef) lower 95%"]),
                        "hr_ci_hi": float(r["exp(coef) upper 95%"]),
                        "wald_p": float(r["p"]),
                        "schoenfeld_p": ph_p.get(c, np.nan),
                    })
            except Exception:
                pass

    return add_provenance(pd.DataFrame(rows), prov)


def build_pmva_specification_curve(table3: pd.DataFrame, sens_df: pd.DataFrame, prov: dict[str, str]) -> pd.DataFrame:
    """Output #5: PMV_A / PMV_D across all specifications."""
    comb = pd.concat([table3, sens_df], ignore_index=True)
def build_pmva_specification_curve(table3: pd.DataFrame, sens_df: pd.DataFrame, prov: dict[str, str]) -> pd.DataFrame:
    """Output #5: PMV_A / PMV_D across all strictly non-imputed complete-case and landmark specifications."""
    comb = pd.concat([table3, sens_df], ignore_index=True)
    pmv = comb[comb["predictor"].isin(["PMV_A", "PMV_D"])].copy()
    keep_models = [
        "primary_marker_eligible_complete_case",
        "marker_eligible_449_pac_count",
        "ecg_markers_only_449",
        "landmark_120s_untruncated",
        "landmark_720s_H4b_PMV_A",
        "landmark_720s_H4b_PMV_D",
        "H4b_ge30s_landmark720_PMV_A",
        "H4b_ge30s_landmark720_PMV_D",
        "no_super_contributors_clean",
    ]
    pmv = pmv[pmv["model"].isin(keep_models)].copy()
    pmv["inference_status"] = np.where(
        pmv["model"].isin(["landmark_120s_untruncated", "landmark_720s_H4b_PMV_A", "landmark_720s_H4b_PMV_D"]),
        "unbiased_landmark_exploratory",
        "secondary_exploratory",
    )
    cols = ["model", "predictor", "n_events", "n_reinit", "n_patients", "hr_per_sd", "hr_ci_lo", "hr_ci_hi", "wald_p", "schoenfeld_p", "inference_status"]
    return add_provenance(pmv[cols], prov)


def main() -> None:
    prov = get_provenance()
    run_dir = get_latest_run_dir()
    out_dir = run_dir / "aim3"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = merge_event_markers(run_dir, "events_g0.parquet")
    df_cs_path = run_dir / "stage4" / "events_g0_continuity_nonsinus_gaps.parquet"
    df_cs = pd.read_parquet(df_cs_path) if df_cs_path.exists() else pd.DataFrame()
    print(f"[Aim 3] Loaded {len(df)} events across {df['patient'].nunique()} patients.")

    # 1. Eligibility flow & Collinearity VIF
    elig_df = compute_eligibility_flow(run_dir, df, prov)
    elig_df.to_csv(out_dir / "eligibility_flow.csv", index=False)

    vif_df = compute_collinearity_vif(df, prov)
    vif_df.to_csv(out_dir / "collinearity_vif.csv", index=False)

    # 2. Descriptive KM (native + confirmed-sinus sensitivity, episode-weighted & patient-weighted at 30s, 1m, 6m, 1h)
    desc_df, km_curves = compute_descriptive_km(df, prov, df_cs=df_cs)
    desc_df.to_csv(out_dir / "descriptive_reinit_km.csv", index=False)
    km_curves.to_parquet(out_dir / "km_reinit_curves.parquet", index=False)

    # Marker-eligible masks (strict non-null SQI checks, no fallback filling)
    sqi_t = df["sqi_pass_terminal"].eq(1) if "sqi_pass_terminal" in df.columns else pd.Series(False, index=df.index)
    sqi_a = df["sqi_pass_postA"].eq(1) if "sqi_pass_postA" in df.columns else pd.Series(False, index=df.index)
    insuf_a = df["postA_insufficient"].eq(0) if "postA_insufficient" in df.columns else pd.Series(False, index=df.index)
    untrunc_a = (df["postA_truncated"].eq(0)) if "postA_truncated" in df.columns else pd.Series(False, index=df.index)
    ge300 = df["dur_sec"] >= 300.0
    marker_clean_df = df[ge300 & sqi_t & sqi_a & insuf_a].copy()
    landmark_df = marker_clean_df[untrunc_a[marker_clean_df.index] & (marker_clean_df["time_to_reinit_sec"] > 120.0)].copy()

    # Exact matched complete-case 449-row DataFrame for fair incremental validation (history vs ECG vs combined)
    matched_449_df, _ = prepare_complete_case_df(marker_clean_df, PAC_COUNT_PREDICTORS, standardize=False)

    # 3. Primary & Companion Models (strictly complete-case, ZERO imputation, explicit landmark origin offsets)
    t3_frames = []
    aft_frames = []

    for mname, mdf, mpreds, lm_off in [
        ("primary_marker_eligible_complete_case", marker_clean_df, PRIMARY_PREDICTORS, 0.0),
        ("marker_eligible_449_pac_count", marker_clean_df, PAC_COUNT_PREDICTORS, 0.0),
        ("ecg_markers_only_449", matched_449_df, ECG_MARKERS_ONLY, 0.0),
        ("history_only_matched_449", matched_449_df, HISTORY_PREDICTORS, 0.0),
        ("landmark_120s_untruncated", landmark_df, PAC_COUNT_PREDICTORS, 120.0),
        ("all_events_history_only", df, HISTORY_PREDICTORS, 0.0),
        ("history_no_prior_count", df, [c for c in HISTORY_PREDICTORS if c != "n_prior_terminations"], 0.0),
    ]:
        res, aft_res, _, _ = fit_cox_clean(mdf, mpreds, mname, strata=["cohort"], landmark_offset_sec=lm_off)
        if not res.empty:
            t3_frames.append(res)
        if not aft_res.empty:
            aft_frames.append(aft_res)

    if not df_cs.empty:
        res_cs, _, _, _ = fit_cox_clean(df_cs, HISTORY_PREDICTORS, "continuity_nonsinus_history_only", strata=["cohort"])
        if not res_cs.empty:
            t3_frames.append(res_cs)

    spline_res = fit_cohort_x_logdur_spline(df)
    if not spline_res.empty:
        t3_frames.append(spline_res)

    # H4b: Window A vs Window D evaluated STRICTLY on 720-s (12-min) landmark origin
    has_d = df["has_postD_window"].fillna(False).astype(bool) if "has_postD_window" in df.columns else pd.Series(False, index=df.index)
    sqi_d = df["sqi_pass_postD"].eq(1) if "sqi_pass_postD" in df.columns else pd.Series(False, index=df.index)
    lm720_mask = df["time_to_reinit_sec"] > 720.0
    for m_prefix, base_mask in [
        ("landmark_720s_H4b", ge300 & sqi_t & sqi_a & insuf_a & has_d & sqi_d & lm720_mask),
        ("H4b_ge30s_landmark720", sqi_a & insuf_a & has_d & sqi_d & lm720_mask),
    ]:
        sub_d = df[base_mask].dropna(subset=["PMV_A", "PMV_D"]).copy()
        preds_a = PAC_COUNT_PREDICTORS
        preds_d = ["PMV_D" if c == "PMV_A" else c for c in PAC_COUNT_PREDICTORS]
        res_a, _, _, _ = fit_cox_clean(sub_d, preds_a, f"{m_prefix}_PMV_A", strata=["cohort"], landmark_offset_sec=720.0)
        res_d, _, _, _ = fit_cox_clean(sub_d, preds_d, f"{m_prefix}_PMV_D", strata=["cohort"], landmark_offset_sec=720.0)
        if not res_a.empty and not res_d.empty:
            ca = res_a.loc[res_a["predictor"] == "PMV_A", "coef"].values
            cd = res_d.loc[res_d["predictor"] == "PMV_D", "coef"].values
            atten = float(cd[0] / ca[0]) if (len(ca) and len(cd) and abs(ca[0]) > 1e-9) else np.nan
            res_a["h4b_attenuation_ratio"] = atten
            res_d["h4b_attenuation_ratio"] = atten
            t3_frames.extend([res_a, res_d])

    table3 = pd.concat(t3_frames, ignore_index=True)
    add_provenance(table3, prov).to_csv(out_dir / "table3_primary_hrs.csv", index=False)

    aft_all = pd.concat(aft_frames, ignore_index=True) if aft_frames else pd.DataFrame()
    add_provenance(aft_all, prov).to_csv(out_dir / "stratified_aft_sensitivity.csv", index=False)

    # 4. Validation (strictly non-imputed, includes matched history-only baseline on identical rows!)
    val_df, cal_df = run_validation_clean(
        df,
        [
            ("history_only_matched_449", matched_449_df, HISTORY_PREDICTORS),
            ("ecg_markers_only_449", matched_449_df, ECG_MARKERS_ONLY),
            ("marker_eligible_449_pac_count", matched_449_df, PAC_COUNT_PREDICTORS),
            ("primary_marker_eligible_complete_case", marker_clean_df, PRIMARY_PREDICTORS),
            ("all_events_history_only", df, HISTORY_PREDICTORS),
        ],
        prov,
    )
    val_df.to_csv(out_dir / "tableS3_loco_validation.csv", index=False)
    cal_df.to_csv(out_dir / "loco_calibration_bins.csv", index=False)

    # 5. Sensitivity models (super-contributors excluded; zero legacy imputation!)
    sens_frames = []
    top12_overall = df["patient"].value_counts().head(12).index
    top10_ltafdb = df[df["cohort"] == "LTAFDB"]["patient"].value_counts().head(10).index
    excl_pts = set(top12_overall).union(set(top10_ltafdb))
    s_nosuper, _, _, _ = fit_cox_clean(marker_clean_df[~marker_clean_df["patient"].isin(excl_pts)], PAC_COUNT_PREDICTORS, "no_super_contributors_clean", strata=["cohort"])
    if not s_nosuper.empty:
        sens_frames.append(s_nosuper)
    sens_all = pd.concat(sens_frames, ignore_index=True) if sens_frames else pd.DataFrame()
    add_provenance(sens_all, prov).to_csv(out_dir / "sensitivity_models.csv", index=False)

    # 5b. Output #1 (Within-Between Decomposition) & Output #5 (PMV_A Specification Curve)
    compute_within_between_decomposition(df, marker_clean_df, prov, df_cs=df_cs).to_csv(out_dir / "within_between_decomposition.csv", index=False)
    build_pmva_specification_curve(table3, sens_all, prov).to_csv(out_dir / "pmva_specification_curve.csv", index=False)

    # 6. Decisive contrast, SHDB clinical, Two-state frailty + Hawkes
    compute_decisive_contrast(run_dir, prov).to_csv(out_dir / "decisive_contrast.csv", index=False)
    run_shdb_clinical_model(df, prov).to_csv(out_dir / "shdb_clinical_model.csv", index=False)
    compute_two_state_summary(run_dir, df, prov).to_csv(out_dir / "two_state_summary.csv", index=False)

    print(f"[Aim 3] Revision complete. Outputs written to {out_dir}")


if __name__ == "__main__":
    main()
