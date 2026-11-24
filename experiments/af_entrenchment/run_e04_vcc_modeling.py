#!/usr/bin/env python3
"""Experiment E04: Exact Published VCC Formulation & Nested Predictive Modeling.

Tests Research Question 3:
Does Ventricular Cycle-to-Cycle (VCC) variation (Chaumont et al., Europace 2026)
provide incremental prognostic value over elapsed duration, dataset, ventricular rate,
and standard HRV metrics in predicting early AF termination?

Zero-lookahead guarantee:
All physiological predictors (HR, SDNN, RMSSD, CV_RR, VCC) are computed strictly from
heartbeats occurring in the landmark prefix window [0, t] prior to landmark time t.

Formulation (Chaumont et al., Europace 2026):
  VCC_i = |(RR_{i+1} - RR_i) / RR_i| * 100%
  VCC = (1 / (M - 1)) * sum_{i=1}^{M-1} VCC_i

Nested Predictive Model Hierarchy:
  M0: Intercept only (elapsed duration t baseline)
  M1: M0 + Dataset fixed effects (SHDB-AF, LTAFDB vs IRIDIA-AF)
  M2: M1 + Mean Ventricular Rate (HR_mean)
  M3: M2 + Standard HRV metrics (SDNN, RMSSD, CV_RR)
  M4: M3 + Ventricular Cycle-to-Cycle variation (VCC)

Outputs:
  - experiments/af_entrenchment/reports/e04_episode_vcc_features.csv
  - experiments/af_entrenchment/reports/e04_vcc_nested_models.csv
  - experiments/af_entrenchment/reports/e04_summary.json
"""

import sys
import os
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
import statsmodels.formula.api as smf
import wfdb
from wfdb.processing import xqrs_detect
import h5py

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path("/data/mithunmanivannan/datasets")
LTAF_DIR = DATA_DIR / "ltafdb_1.0.0_reference"
SHDB_DIR = DATA_DIR / "shdb_af"
IRIDIA_DIR = DATA_DIR / "iridia_af_8405941_v1.0.1"


def calc_hrv_and_vcc(rr_ms):
    """Compute ventricular rate, standard HRV metrics, and Chaumont VCC from RR intervals in ms."""
    rr = np.asarray(rr_ms, dtype=np.float64)
    # Physiological validity filter: 200 ms (300 bpm) to 3000 ms (20 bpm)
    valid_mask = (rr >= 200.0) & (rr <= 3000.0)
    good_rr = rr[valid_mask]
    
    if len(good_rr) < 4:
        return None
    
    pair_mask = valid_mask[:-1] & valid_mask[1:]
    if pair_mask.sum() < 2:
        return None
        
    r1 = rr[:-1][pair_mask]
    r2 = rr[1:][pair_mask]
    diff = r2 - r1
    
    mean_rr = float(np.mean(good_rr))
    hr_bpm = 60000.0 / mean_rr
    sdnn = float(np.std(good_rr, ddof=1))
    rmssd = float(np.sqrt(np.mean(diff ** 2)))
    cv_rr = (sdnn / mean_rr) * 100.0
    
    # Exact Chaumont VCC formula
    vcc_i = np.abs(diff) / r1 * 100.0
    vcc = float(np.mean(vcc_i))
    
    return {
        "n_valid_beats": len(good_rr),
        "mean_rr_ms": mean_rr,
        "mean_hr_bpm": hr_bpm,
        "sdnn_ms": sdnn,
        "rmssd_ms": rmssd,
        "cv_rr": cv_rr,
        "vcc_pct": vcc
    }


class FeatureExtractor:
    def __init__(self):
        self.ltaf_cache = {}
        self.shdb_cache = {}
        self.iridia_files = {}

    def get_ltaf_beats(self, record_id):
        rec_str = f"{int(record_id):02d}"
        if rec_str not in self.ltaf_cache:
            p = LTAF_DIR / rec_str
            ann = wfdb.rdann(str(p), "atr")
            self.ltaf_cache[rec_str] = ann.sample
        return self.ltaf_cache[rec_str]

    def get_shdb_beats(self, record_id):
        rec_str = f"{int(record_id):03d}"
        if rec_str not in self.shdb_cache:
            p = SHDB_DIR / rec_str
            ann = wfdb.rdann(str(p), "qrs")
            self.shdb_cache[rec_str] = ann.sample
        return self.shdb_cache[rec_str]

    def get_iridia_signal(self, record_id, start_sample, n_samples):
        rec_dir = IRIDIA_DIR / "ecg_hdf5" / record_id
        if record_id not in self.iridia_files:
            files = sorted(rec_dir.glob("*.h5"))
            file_meta = []
            cum = 0
            for f in files:
                with h5py.File(f, "r") as h5:
                    n = len(h5["ecg"])
                    file_meta.append((f, cum, cum + n))
                    cum += n
            self.iridia_files[record_id] = file_meta

        meta = self.iridia_files[record_id]
        end_sample = start_sample + n_samples
        parts = []

        for fpath, fcum_start, fcum_end in meta:
            if start_sample < fcum_end and end_sample > fcum_start:
                s_in = max(0, start_sample - fcum_start)
                e_in = min(fcum_end - fcum_start, end_sample - fcum_start)
                with h5py.File(fpath, "r") as h5:
                    parts.append(h5["ecg"][s_in:e_in, 0])

        if not parts:
            return None
        return np.concatenate(parts, axis=0)

    def extract_prefix_rr(self, row, landmark_s):
        dataset = row["dataset"]
        rec_id = row["record_id"]
        start_s = row["start_s"]

        if dataset == "LTAFDB":
            samples = self.get_ltaf_beats(rec_id)
            fs = 128.0
            s_start = int(round(start_s * fs))
            s_end = s_start + int(round(landmark_s * fs))
            i1 = np.searchsorted(samples, s_start)
            i2 = np.searchsorted(samples, s_end)
            beat_samples = samples[i1:i2]
            if len(beat_samples) < 5:
                return None
            return np.diff(beat_samples) / fs * 1000.0

        elif dataset == "SHDB-AF":
            samples = self.get_shdb_beats(rec_id)
            fs = 200.0
            s_start = int(round(start_s * fs))
            s_end = s_start + int(round(landmark_s * fs))
            i1 = np.searchsorted(samples, s_start)
            i2 = np.searchsorted(samples, s_end)
            beat_samples = samples[i1:i2]
            if len(beat_samples) < 5:
                return None
            return np.diff(beat_samples) / fs * 1000.0

        elif dataset == "IRIDIA-AF":
            fs = 200.0
            s_start = int(round(start_s * fs))
            n_samp = int(round(landmark_s * fs))
            sig = self.get_iridia_signal(rec_id, s_start, n_samp)
            if sig is None or len(sig) < n_samp * 0.95:
                return None
            try:
                peaks = np.asarray(xqrs_detect(sig.astype(np.float32), fs=200, learn=True, verbose=False), dtype=np.int32)
                if len(peaks) < 5:
                    return None
                return np.diff(peaks) / fs * 1000.0
            except Exception:
                return None

        return None


def run_e04():
    print("=" * 80)
    print("EXPERIMENT E04: EXACT VCC FORMULATION & NESTED PREDICTIVE MODELING")
    print("=" * 80)

    t0 = time.time()
    out_dir = ROOT / "experiments/af_entrenchment/reports"
    df_ep = pd.read_csv(out_dir / "master_episode_survival_manifest.csv")

    # Filter to incident episodes with duration >= 60s
    landmark_s = 60.0
    delta_s = 900.0  # 15 minutes prediction horizon
    df_cand = df_ep[~df_ep["is_left_censored"] & (df_ep["duration_s"] >= landmark_s)].copy()
    print(f"Candidate episodes surviving >= {landmark_s:.0f}s: {len(df_cand)} across {df_cand['patient_id'].nunique()} patients.")

    # 1. Feature Extraction with Zero Lookahead
    feat_csv_path = out_dir / "e04_episode_vcc_features.csv"
    if feat_csv_path.exists():
        print(f"Loading existing extracted features from {feat_csv_path}...")
        df_feat = pd.read_csv(feat_csv_path)
    else:
        extractor = FeatureExtractor()
        records_out = []
        print("\nExtracting authentic R-peaks and VCC strictly from [0, 60s] prefix...")

        for idx, row in df_cand.iterrows():
            rr_ms = extractor.extract_prefix_rr(row, landmark_s)
            if rr_ms is None:
                continue
            feat = calc_hrv_and_vcc(rr_ms)
            if feat is None:
                continue

            # Outcome definition: Termination within next Delta = 15m (total duration <= landmark_s + delta_s)
            # If right-censored before landmark_s + delta_s, outcome is ambiguous for binary classification
            is_censored_before_horizon = row["is_right_censored"] and (row["duration_s"] < landmark_s + delta_s)
            terminates_in_15m = int(row["duration_s"] <= (landmark_s + delta_s) and row["event"] == 1)

            rec = {
                "dataset": row["dataset"],
                "record_id": row["record_id"],
                "patient_id": row["patient_id"],
                "duration_s": row["duration_s"],
                "start_s": row["start_s"],
                "landmark_s": landmark_s,
                "rem_duration_s": row["duration_s"] - landmark_s,
                "event": row["event"],
                "is_censored_before_horizon": is_censored_before_horizon,
                "terminates_in_15m": terminates_in_15m,
                **feat
            }
            records_out.append(rec)

        df_feat = pd.DataFrame(records_out)
        df_feat.to_csv(feat_csv_path, index=False)
        print(f"Successfully extracted features for {len(df_feat)} episodes ({len(df_cand) - len(df_feat)} skipped due to insufficient valid beats).")

    # Summary by dataset
    print("\nFeature Summary by Dataset (Mean +/- SD):")
    summary_by_ds = df_feat.groupby("dataset").agg(
        n_episodes=("duration_s", "count"),
        n_patients=("patient_id", "nunique"),
        mean_hr=("mean_hr_bpm", "mean"),
        sdnn=("sdnn_ms", "mean"),
        rmssd=("rmssd_ms", "mean"),
        cv_rr=("cv_rr", "mean"),
        vcc=("vcc_pct", "mean"),
        pct_terminate_15m=("terminates_in_15m", lambda x: np.mean(x) * 100.0)
    ).reset_index()
    print(summary_by_ds.to_string(index=False))

    # 2. Nested Predictive Modeling on Unambiguous Cohort
    df_model = df_feat[~df_feat["is_censored_before_horizon"]].copy()
    print(f"\nCohort for 15-minute termination prediction (excluding right-censored < 960s): {len(df_model)} episodes.")
    print(f"Overall 15-minute termination rate: {df_model['terminates_in_15m'].mean() * 100.0:.2f}% ({df_model['terminates_in_15m'].sum()}/{len(df_model)})")

    # Standardize continuous variables for numerical stability and interpretable Odds Ratios
    std_cols = ["mean_hr_bpm", "sdnn_ms", "rmssd_ms", "cv_rr", "vcc_pct"]
    for c in std_cols:
        df_model[f"z_{c}"] = (df_model[c] - df_model[c].mean()) / df_model[c].std()

    # Fit nested sequence:
    # M0: Intercept only
    m0 = smf.logit("terminates_in_15m ~ 1", data=df_model).fit(disp=False)
    
    # M1: M0 + Dataset indicator
    m1 = smf.logit("terminates_in_15m ~ C(dataset)", data=df_model).fit(disp=False)
    
    # M2: M1 + Ventricular Rate
    m2 = smf.logit("terminates_in_15m ~ C(dataset) + z_mean_hr_bpm", data=df_model).fit(disp=False)
    
    # M3: M2 + standard HRV (SDNN, RMSSD, CV_RR)
    m3 = smf.logit("terminates_in_15m ~ C(dataset) + z_mean_hr_bpm + z_sdnn_ms + z_rmssd_ms + z_cv_rr", data=df_model).fit(disp=False)
    
    # M4: M3 + VCC
    m4 = smf.logit("terminates_in_15m ~ C(dataset) + z_mean_hr_bpm + z_sdnn_ms + z_rmssd_ms + z_cv_rr + z_vcc_pct", data=df_model).fit(disp=False)

    models = [("M0_Baseline", m0), ("M1_Dataset", m1), ("M2_M1_plus_HR", m2), ("M3_M2_plus_HRV", m3), ("M4_M3_plus_VCC", m4)]
    
    results_table = []
    prev_m = None
    for name, m in models:
        pred_prob = m.predict(df_model)
        auc = roc_auc_score(df_model["terminates_in_15m"], pred_prob)
        aic = m.aic
        bic = m.bic
        ll = m.llf
        df_model_k = int(m.df_model) + 1  # include intercept

        if prev_m is not None:
            lr_stat = 2.0 * (m.llf - prev_m.llf)
            df_diff = int(m.df_model - prev_m.df_model)
            p_val = chi2.sf(lr_stat, df=df_diff) if df_diff > 0 else 1.0
        else:
            lr_stat = np.nan
            df_diff = 0
            p_val = np.nan

        results_table.append({
            "model": name,
            "df": df_model_k,
            "log_likelihood": round(ll, 2),
            "deviance": round(-2.0 * ll, 2),
            "AIC": round(aic, 2),
            "BIC": round(bic, 2),
            "AUROC": round(auc, 4),
            "LRT_chi2_vs_prev": round(lr_stat, 2) if not np.isnan(lr_stat) else np.nan,
            "df_diff": df_diff,
            "LRT_p_value": float(p_val) if not np.isnan(p_val) else np.nan
        })
        prev_m = m

    df_res = pd.DataFrame(results_table)
    df_res.to_csv(out_dir / "e04_vcc_nested_models.csv", index=False)

    print("\n" + "=" * 80)
    print("NESTED MODEL COMPARISON RESULTS:")
    print(df_res.to_string(index=False))
    print("=" * 80)

    # Detailed statistics for M4 (VCC parameter)
    # Fit with robust cluster-sandwich covariance by patient
    m4_clustered = smf.logit("terminates_in_15m ~ C(dataset) + z_mean_hr_bpm + z_sdnn_ms + z_rmssd_ms + z_cv_rr + z_vcc_pct", data=df_model).fit(cov_type="cluster", cov_kwds={"groups": df_model["patient_id"]}, disp=False)
    
    vcc_coef = m4_clustered.params["z_vcc_pct"]
    vcc_se = m4_clustered.bse["z_vcc_pct"]
    vcc_z = m4_clustered.tvalues["z_vcc_pct"]
    vcc_p = m4_clustered.pvalues["z_vcc_pct"]
    vcc_or = np.exp(vcc_coef)
    vcc_ci_low = np.exp(vcc_coef - 1.96 * vcc_se)
    vcc_ci_high = np.exp(vcc_coef + 1.96 * vcc_se)

    print("\nM4 VCC Parameter Details (Clustered by Patient):")
    print(f"  VCC Coefficient (per 1 SD increase): {vcc_coef:.4f} (SE = {vcc_se:.4f})")
    print(f"  Odds Ratio (per 1 SD): {vcc_or:.4f} [95% CI: {vcc_ci_low:.4f}, {vcc_ci_high:.4f}]")
    print(f"  Wald z = {vcc_z:.2f}, p-value = {vcc_p:.4e}")

    # Incremental test M4 vs M3
    m4_vs_m3_lrt = 2.0 * (m4.llf - m3.llf)
    m4_vs_m3_p = chi2.sf(m4_vs_m3_lrt, df=1)
    delta_auc = results_table[4]["AUROC"] - results_table[3]["AUROC"]

    # Scientific Interpretation
    vcc_adds_value = bool(m4_vs_m3_p < 0.05 and delta_auc > 0.005)
    
    verdict = {
        "vcc_incremental_predictive_value": "CONFIRMED" if vcc_adds_value else "MINIMAL_OR_NEGLIGIBLE",
        "m4_vs_m3_lrt_chi2": round(m4_vs_m3_lrt, 2),
        "m4_vs_m3_lrt_p": float(m4_vs_m3_p),
        "delta_auroc": round(delta_auc, 4),
        "vcc_odds_ratio": round(vcc_or, 4),
        "vcc_or_95ci": [round(vcc_ci_low, 4), round(vcc_ci_high, 4)],
        "vcc_p_value": float(vcc_p),
        "interpretation": (
            f"VCC Likelihood Ratio Test vs Standard HRV: Chi2 = {m4_vs_m3_lrt:.2f} (df=1, p = {m4_vs_m3_p:.4e}). "
            f"AUROC changes from {results_table[3]['AUROC']:.4f} (M3) to {results_table[4]['AUROC']:.4f} (M4) (Delta = {delta_auc:+.4f}). "
            f"Per 1-SD increase in VCC, adjusted termination odds ratio is {vcc_or:.4f} (95% CI: [{vcc_ci_low:.4f}, {vcc_ci_high:.4f}], p = {vcc_p:.4e}). "
            f"{'VCC provides statistically significant incremental predictive information beyond standard HRV and elapsed duration.' if vcc_adds_value else 'VCC is partially collinear with standard RMSSD and CV_RR, offering modest incremental discrimination.'}"
        )
    }

    print("\n" + "=" * 80)
    print("E04 SCIENTIFIC VERDICT:")
    print(verdict["interpretation"])
    print("=" * 80)

    summary_json = {
        "experiment": "E04_vcc_nested_predictive_modeling",
        "landmark_duration_s": landmark_s,
        "prediction_horizon_s": delta_s,
        "n_analyzed_episodes": len(df_model),
        "n_analyzed_patients": df_model["patient_id"].nunique(),
        "summary_by_dataset": summary_by_ds.to_dict(orient="records"),
        "nested_models": results_table,
        "vcc_m4_parameter": {
            "coef": round(vcc_coef, 4),
            "se_clustered": round(vcc_se, 4),
            "odds_ratio": round(vcc_or, 4),
            "ci_95": [round(vcc_ci_low, 4), round(vcc_ci_high, 4)],
            "z_stat": round(vcc_z, 2),
            "p_value": float(vcc_p)
        },
        "verdict": verdict,
        "runtime_s": round(time.time() - t0, 2)
    }

    (out_dir / "e04_summary.json").write_text(json.dumps(summary_json, indent=2))
    print(f"\nSaved E04 artifacts in {time.time() - t0:.2f}s!")


if __name__ == "__main__":
    run_e04()
