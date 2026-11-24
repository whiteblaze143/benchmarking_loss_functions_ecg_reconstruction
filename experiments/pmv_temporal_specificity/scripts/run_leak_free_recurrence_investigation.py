"""Leak-Free Landmark Recurrence Investigation.

Primary Clinical Research Question:
Is early post-termination PMV independently associated with subsequent AF recurrence
after accounting for patient-specific arrhythmia propensity and conventional ECG features?

Design:
- Population: Observed AF terminations followed by >= 120s of verified continuous sinus rhythm.
- Landmark: Exactly t_landmark = 120s (2.0 minutes).
- Predictor: PMV extracted strictly from the single [t_0, t_0 + 120s] window (no data past 120s).
- Arrhythmia History: Past-only cumulative AF burden, prior AF episode count, preceding AF duration.
- Conventional ECG: Mean HR, SDNN, RMSSD from the [0, 120s] window.
- Primary Outcome: Time to subsequent AF recurrence landmarked at t = 120s.
- Horizon: 60-minute prespecified truncated follow-up and full recording follow-up.
- Model 1: Clinical AF History + Conventional ECG.
- Model 2: Clinical AF History + Conventional ECG + Landmark PMV.
- Cohorts: Evaluated pooled (patient-clustered), and individually for IRIDIA-AF, SHDB-AF, LTAFDB.
"""

import sys
from pathlib import Path
import json
import time
import concurrent.futures
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.duration.hazard_regression import PHReg
import wfdb
import wfdb.processing
import h5py

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.pmv_temporal_specificity.scripts.pmv_core import PMVExtractor

MANIFESTS_DIR = PROJECT_ROOT / "experiments" / "pmv_temporal_specificity" / "manifests"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "pmv_temporal_specificity" / "results"
LTAFDB_DIR = Path("/data/mithunmanivannan/datasets/ltafdb_1.0.0_reference")
SHDB_DIR = Path("/data/mithunmanivannan/datasets/shdb_af")
IRIDIA_DIR = Path("/data/mithunmanivannan/datasets/iridia_af_8405941_v1.0.1/ecg_hdf5")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_ecg_and_qrs_slice(
    dataset: str,
    record_id: str,
    start_s: float,
    end_s: float
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    """Load ECG slice [start_s, end_s] and relative QRS indices."""
    duration_s = end_s - start_s
    if duration_s < 100.0:
        return None, None, 0.0

    if dataset == "LTAFDB":
        fs = 128.0
        rec_base = str(LTAFDB_DIR / record_id)
        s_idx = max(0, int(round(start_s * fs)))
        e_idx = max(s_idx + 1, int(round(end_s * fs)))
        try:
            rec = wfdb.rdrecord(rec_base, sampfrom=s_idx, sampto=e_idx)
            ecg = rec.p_signal.T
            atr = wfdb.rdann(rec_base, "atr", sampfrom=s_idx, sampto=e_idx)
            qrs_list = [s - s_idx for s, sym in zip(atr.sample, atr.symbol) if sym in ["N", "B", "L", "R", "j"]]
            qrs = np.array(qrs_list, dtype=int)
            qrs = qrs[(qrs >= 0) & (qrs < ecg.shape[1])]
            return ecg, qrs, fs
        except Exception:
            return None, None, fs

    elif dataset == "SHDB-AF":
        fs = 200.0
        rec_base = str(SHDB_DIR / record_id)
        s_idx = max(0, int(round(start_s * fs)))
        e_idx = max(s_idx + 1, int(round(end_s * fs)))
        try:
            rec = wfdb.rdrecord(rec_base, sampfrom=s_idx, sampto=e_idx)
            ecg = rec.p_signal.T
            qrs_ann = wfdb.rdann(rec_base, "qrs", sampfrom=s_idx, sampto=e_idx)
            rel_qrs = qrs_ann.sample - s_idx
            rel_qrs = rel_qrs[(rel_qrs >= 0) & (rel_qrs < ecg.shape[1])]
            return ecg, rel_qrs, fs
        except Exception:
            return None, None, fs

    elif dataset == "IRIDIA-AF":
        fs = 200.0
        rec_dir = IRIDIA_DIR / record_id
        h5_files = list(rec_dir.glob("*.h5"))
        if not h5_files:
            return None, None, fs
        h5_path = h5_files[0]
        s_idx = max(0, int(round(start_s * fs)))
        e_idx = max(s_idx + 1, int(round(end_s * fs)))
        try:
            with h5py.File(h5_path, "r") as f:
                ecg = f["ecg"][s_idx:e_idx].T
            if ecg.shape[1] < int(100.0 * fs):
                return None, None, fs
            qrs = wfdb.processing.xqrs_detect(ecg[0], fs=fs, verbose=False)
            return ecg, qrs, fs
        except Exception:
            return None, None, fs

    return None, None, 0.0


def process_single_landmark_episode(job: Tuple[Dict, Dict, Dict]) -> Optional[Dict]:
    """Process exactly one 120s window [t_0, t_0 + 120s] for an episode."""
    row, ref, history = job
    dataset = row["dataset"]
    record_id = row["record_id"]
    patient_id = row["patient_id"]
    af_term_s = row["end_s"]
    af_dur_s = row["duration_s"]
    sinus_dur_s = row["subsequent_sinus_dur_s"]
    time_until_next_af_s = row["time_until_next_af_s"]

    # Landmark strictly at t = 120s
    if sinus_dur_s < 120.0:
        return None

    # Load strictly the 120.0 seconds of ECG: [af_term_s, af_term_s + 120.0]
    ecg, qrs, fs = load_ecg_and_qrs_slice(dataset, record_id, af_term_s, af_term_s + 120.0)
    if ecg is None or qrs is None or len(qrs) < 20:
        return None

    extractor = PMVExtractor(fs=fs, subsegment_len_s=120.0, subsegment_step_s=60.0, min_beats_per_subsegment=20)
    try:
        ecg_filt = extractor.filter_signal(ecg)
    except Exception:
        return None

    Ei_rec = ref["Ei_rec"]
    Ei_lead0 = ref["Ei_lead0"]
    Ei_lead1 = ref["Ei_lead1"]
    u1 = np.array([ref["u1_0"], ref["u1_1"]])

    # Single 120s window centered at 60s
    res_pca = extractor.compute_subsegment_pmv(
        ecg_filt, qrs, 60.0, fixed_eigenvector=u1, fixed_reference_energy=Ei_rec
    )
    res_l0 = extractor.compute_subsegment_pmv(
        ecg_filt[0], qrs, 60.0, fixed_reference_energy=Ei_lead0
    )
    res_l1 = extractor.compute_subsegment_pmv(
        ecg_filt[1], qrs, 60.0, fixed_reference_energy=Ei_lead1
    )

    if not res_pca.is_valid or np.isnan(res_pca.pmv_V) or np.isinf(res_pca.pmv_V):
        return None

    # Follow-up from landmark t = 120s
    time_to_rec_s = sinus_dur_s - 120.0
    time_to_rec_m = time_to_rec_s / 60.0

    # Event indicator: did AF recur within recording?
    event_rec = int(pd.notna(time_until_next_af_s) and time_until_next_af_s <= sinus_dur_s + 5.0)

    # 60-minute truncated horizon
    event_rec_60m = int(event_rec == 1 and time_to_rec_m <= 60.0)
    time_to_rec_60m = min(time_to_rec_m, 60.0)

    # 30-minute truncated horizon
    event_rec_30m = int(event_rec == 1 and time_to_rec_m <= 30.0)
    time_to_rec_30m = min(time_to_rec_m, 30.0)

    return {
        "dataset": dataset,
        "record_id": record_id,
        "patient_id": patient_id,
        "episode_id": f"{dataset}_{record_id}_{int(af_term_s)}",
        "af_term_s": af_term_s,
        # Predictors from [0, 120s]
        "pmv_V": res_pca.pmv_V,
        "mad_energy_D": res_pca.mad_energy_D,
        "pmv_lead0": res_l0.pmv_V,
        "pmv_lead1": res_l1.pmv_V,
        "pmv_aligned_V": res_pca.pmv_aligned_V,
        "pmv_aligned_D": res_pca.pmv_aligned_D,
        "pr_aligned": res_pca.pr_aligned,
        "mean_hr_bpm": res_pca.mean_hr_bpm,
        "sdnn_ms": res_pca.sdnn_ms,
        "rmssd_ms": res_pca.rmssd_ms,
        "num_beats": res_pca.num_beats,
        "ref_energy_Ei": Ei_rec,
        # Past-only Arrhythmia History at t_0
        "af_duration_s": af_dur_s,
        "af_duration_m": af_dur_s / 60.0,
        "preceding_sinus_dur_s": row.get("preceding_sinus_dur_s", np.nan),
        "cum_prior_af_time_s": history["cum_prior_af_time_s"],
        "cum_prior_af_episodes": history["cum_prior_af_episodes"],
        "prior_af_burden_pct": history["prior_af_burden_pct"],
        # Follow-up Outcomes landmarked at t = 120s
        "time_to_recurrence_s": time_to_rec_s,
        "time_to_recurrence_m": time_to_rec_m,
        "event_recurrent_af": event_rec,
        "time_to_rec_60m": time_to_rec_60m,
        "event_rec_60m": event_rec_60m,
        "time_to_rec_30m": time_to_rec_30m,
        "event_rec_30m": event_rec_30m,
    }


def compute_harrell_c_index(times: np.ndarray, events: np.ndarray, risk_scores: np.ndarray) -> float:
    """Calculate Harrell C-index with high risk corresponding to shorter time."""
    n = len(times)
    concordant, discordant, tied_risk = 0, 0, 0
    permissible = 0
    for i in range(n):
        if events[i] != 1:
            continue
        for j in range(n):
            if times[j] > times[i]:
                permissible += 1
                if risk_scores[i] > risk_scores[j]:
                    concordant += 1
                elif risk_scores[i] < risk_scores[j]:
                    discordant += 1
                else:
                    tied_risk += 1
    if permissible == 0:
        return 0.5
    return float((concordant + 0.5 * tied_risk) / permissible)


def evaluate_recurrence_models(df: pd.DataFrame, cohort_name: str) -> Dict:
    """Fit Model 1 (Clinical AF History + Conventional ECG) vs Model 2 (+ PMV)."""
    n_episodes = len(df)
    n_patients = df["patient_id"].nunique()
    n_events = int(df["event_recurrent_af"].sum())
    n_events_60m = int(df["event_rec_60m"].sum())

    if n_episodes < 20 or n_events < 10:
        return {"cohort": cohort_name, "n_episodes": n_episodes, "status": "INSUFFICIENT_EVENTS"}

    df_fit = df.copy()
    df_fit["log_pmv_V"] = np.log(df_fit["pmv_V"] + 1e-6)
    df_fit["log_raw_D"] = np.log(df_fit["mad_energy_D"] + 1e-9)
    df_fit["log_af_dur"] = np.log(df_fit["af_duration_s"] + 1.0)
    df_fit["log_prior_af_time"] = np.log(df_fit["cum_prior_af_time_s"] + 1.0)
    df_fit["log_hr"] = np.log(df_fit["mean_hr_bpm"] + 1e-6)
    df_fit["sdnn_scaled"] = df_fit["sdnn_ms"] / 100.0

    # 1. Unadjusted PMV Cox Model
    try:
        ph_unadj = PHReg.from_formula("time_to_rec_60m ~ log_pmv_V", data=df_fit, status="event_rec_60m")
        res_unadj = ph_unadj.fit()
        hr_unadj = float(np.exp(res_unadj.params[0]))
        p_unadj = float(res_unadj.pvalues[0])
        ci_unadj_low = float(np.exp(res_unadj.conf_int()[0, 0]))
        ci_unadj_high = float(np.exp(res_unadj.conf_int()[0, 1]))
    except Exception as e:
        hr_unadj, p_unadj, ci_unadj_low, ci_unadj_high = np.nan, np.nan, np.nan, np.nan

    # 2. Model 1: Clinical AF History + Conventional ECG
    # Predictors: log_af_dur, cum_prior_af_episodes, prior_af_burden_pct, mean_hr_bpm, sdnn_scaled
    formula_m1 = "time_to_rec_60m ~ log_af_dur + cum_prior_af_episodes + prior_af_burden_pct + mean_hr_bpm + sdnn_scaled"
    try:
        ph_m1 = PHReg.from_formula(formula_m1, data=df_fit, status="event_rec_60m")
        res_m1 = ph_m1.fit()
        # Risk scores for C-index: linear predictor X * beta
        pred_m1 = np.dot(ph_m1.exog, res_m1.params)
        c_m1 = compute_harrell_c_index(df_fit["time_to_rec_60m"].values, df_fit["event_rec_60m"].values, pred_m1)
    except Exception as e:
        c_m1 = np.nan

    # 3. Model 2: Clinical AF History + Conventional ECG + Landmark PMV
    formula_m2 = formula_m1 + " + log_pmv_V"
    try:
        ph_m2 = PHReg.from_formula(formula_m2, data=df_fit, status="event_rec_60m")
        res_m2 = ph_m2.fit()
        pred_m2 = np.dot(ph_m2.exog, res_m2.params)
        c_m2 = compute_harrell_c_index(df_fit["time_to_rec_60m"].values, df_fit["event_rec_60m"].values, pred_m2)

        # Adjusted HR for log_pmv_V (last parameter in exog)
        pmv_idx = list(ph_m2.exog_names).index("log_pmv_V")
        hr_adj = float(np.exp(res_m2.params[pmv_idx]))
        p_adj = float(res_m2.pvalues[pmv_idx])
        ci_adj_low = float(np.exp(res_m2.conf_int()[pmv_idx, 0]))
        ci_adj_high = float(np.exp(res_m2.conf_int()[pmv_idx, 1]))
    except Exception as e:
        c_m2, hr_adj, p_adj, ci_adj_low, ci_adj_high = np.nan, np.nan, np.nan, np.nan, np.nan

    delta_c = c_m2 - c_m1 if not np.isnan(c_m2) and not np.isnan(c_m1) else np.nan

    # 4. Logistic regression for 60-minute recurrence outcome
    try:
        logit_m1 = smf.logit("event_rec_60m ~ log_af_dur + cum_prior_af_episodes + prior_af_burden_pct + mean_hr_bpm + sdnn_scaled", data=df_fit).fit(disp=False)
        logit_m2 = smf.logit("event_rec_60m ~ log_af_dur + cum_prior_af_episodes + prior_af_burden_pct + mean_hr_bpm + sdnn_scaled + log_pmv_V", data=df_fit).fit(disp=False)
        from sklearn.metrics import roc_auc_score
        auc_m1 = float(roc_auc_score(df_fit["event_rec_60m"], logit_m1.predict(df_fit)))
        auc_m2 = float(roc_auc_score(df_fit["event_rec_60m"], logit_m2.predict(df_fit)))
        delta_auc = auc_m2 - auc_m1
    except Exception:
        auc_m1, auc_m2, delta_auc = np.nan, np.nan, np.nan

    # 5. Within-Patient Sensitivity Analysis (for patients with >= 2 episodes)
    multi_pts = df_fit.groupby("patient_id").filter(lambda g: len(g) >= 2)
    if len(multi_pts["patient_id"].unique()) >= 5:
        # Spearman correlation within patient
        pt_corrs = []
        for pid, grp in multi_pts.groupby("patient_id"):
            if grp["pmv_V"].nunique() >= 2 and grp["time_to_rec_60m"].nunique() >= 2:
                r_pt, _ = stats.spearmanr(grp["pmv_V"], grp["time_to_rec_60m"])
                if not np.isnan(r_pt):
                    pt_corrs.append(r_pt)
        median_within_pt_r = float(np.median(pt_corrs)) if len(pt_corrs) > 0 else np.nan
        pct_positive_within_pt = float(np.mean(np.array(pt_corrs) > 0) * 100.0) if len(pt_corrs) > 0 else np.nan
    else:
        median_within_pt_r, pct_positive_within_pt = np.nan, np.nan

    return {
        "cohort": cohort_name,
        "n_episodes": n_episodes,
        "n_patients": n_patients,
        "n_events_total": n_events,
        "n_events_60m": n_events_60m,
        "unadjusted_hr_log_pmv": round(hr_unadj, 4),
        "unadjusted_ci95_low": round(ci_unadj_low, 4),
        "unadjusted_ci95_high": round(ci_unadj_high, 4),
        "unadjusted_p": round(p_unadj, 4),
        "adjusted_hr_log_pmv": round(hr_adj, 4),
        "adjusted_ci95_low": round(ci_adj_low, 4),
        "adjusted_ci95_high": round(ci_adj_high, 4),
        "adjusted_p": round(p_adj, 4),
        "c_index_model1_clinical": round(c_m1, 4),
        "c_index_model2_with_pmv": round(c_m2, 4),
        "delta_c_index_pmv": round(delta_c, 4),
        "auroc_model1_clinical": round(auc_m1, 4),
        "auroc_model2_with_pmv": round(auc_m2, 4),
        "delta_auroc_pmv": round(delta_auc, 4),
        "within_patient_median_spearman_r": round(median_within_pt_r, 4),
        "within_patient_pct_positive_r": round(pct_positive_within_pt, 1),
    }


def main():
    print("=" * 70)
    print("LEAK-FREE LANDMARK RECURRENCE INVESTIGATION")
    print("=" * 70)

    # 1. Load references
    df_refs = pd.read_parquet(MANIFESTS_DIR / "recording_level_references.parquet")
    valid_refs = df_refs[df_refs["has_valid_remote_ref"]].copy()
    ref_dict = {row["record_id"]: row.to_dict() for _, row in valid_refs.iterrows()}
    print(f"Loaded {len(ref_dict)} verified recording references.")

    # 2. Load inventory and compute past-only arrhythmia history
    df_inv = pd.read_parquet(MANIFESTS_DIR / "af_episode_inventory.parquet")
    df_inv = df_inv.sort_values(["record_id", "end_s"]).copy()

    # Precompute past-only cumulative AF history per record up to each episode
    history_dict = {}
    for rec_id, grp in df_inv.groupby("record_id"):
        cum_af_time = 0.0
        cum_episodes = 0
        for idx, row in grp.iterrows():
            history_dict[row["episode_id"] if "episode_id" in row else f"{row['dataset']}_{rec_id}_{int(row['end_s'])}"] = {
                "cum_prior_af_time_s": cum_af_time,
                "cum_prior_af_episodes": cum_episodes,
                "prior_af_burden_pct": (cum_af_time / (row["end_s"] + 1e-6)) * 100.0,
            }
            cum_af_time += row["duration_s"]
            cum_episodes += 1

    eligible_episodes = df_inv[(df_inv["record_id"].isin(ref_dict)) & (df_inv["subsequent_sinus_dur_s"] >= 120.0)].copy()
    print(f"Eligible landmark episodes (subsequent sinus >= 120s): {len(eligible_episodes)} across {eligible_episodes['patient_id'].nunique()} patients.")

    jobs = []
    for _, row in eligible_episodes.iterrows():
        ep_key = row["episode_id"] if "episode_id" in row else f"{row['dataset']}_{row['record_id']}_{int(row['end_s'])}"
        hist = history_dict.get(ep_key, {"cum_prior_af_time_s": 0.0, "cum_prior_af_episodes": 0, "prior_af_burden_pct": 0.0})
        jobs.append((row.to_dict(), ref_dict[row["record_id"]], hist))

    print(f"Extracting leak-free [0, 120s] landmark features across 16 workers...")
    t0 = time.time()
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
        for res in executor.map(process_single_landmark_episode, jobs):
            if res is not None:
                results.append(res)
    t1 = time.time()
    print(f"Extracted {len(results)} valid landmark episodes in {t1 - t0:.1f} seconds.")

    df_landmark = pd.DataFrame(results)
    out_landmark_parquet = RESULTS_DIR / "leak_free_landmark_recurrence_features.parquet"
    df_landmark.to_parquet(out_landmark_parquet, index=False)
    print(f"Saved feature dataset to {out_landmark_parquet}")

    # 3. Evaluate Models: Pooled, IRIDIA-AF, SHDB-AF, LTAFDB
    eval_results = []

    # Pooled
    eval_pooled = evaluate_recurrence_models(df_landmark, "Pooled (All Cohorts)")
    eval_results.append(eval_pooled)

    # IRIDIA-AF (Primary prolonged-episode cohort)
    df_iridia = df_landmark[df_landmark["dataset"] == "IRIDIA-AF"].copy()
    eval_iridia = evaluate_recurrence_models(df_iridia, "IRIDIA-AF (Primary Cohort)")
    eval_results.append(eval_iridia)

    # SHDB-AF (Replication cohort 1)
    df_shdb = df_landmark[df_landmark["dataset"] == "SHDB-AF"].copy()
    eval_shdb = evaluate_recurrence_models(df_shdb, "SHDB-AF (Replication 1)")
    eval_results.append(eval_shdb)

    # LTAFDB (Replication cohort 2)
    df_ltaf = df_landmark[df_landmark["dataset"] == "LTAFDB"].copy()
    eval_ltaf = evaluate_recurrence_models(df_ltaf, "LTAFDB (Replication 2)")
    eval_results.append(eval_ltaf)

    df_eval = pd.DataFrame(eval_results)
    out_eval_csv = RESULTS_DIR / "leak_free_recurrence_model_comparison.csv"
    df_eval.to_csv(out_eval_csv, index=False)

    out_summary_json = RESULTS_DIR / "leak_free_recurrence_summary.json"
    with open(out_summary_json, "w") as f:
        json.dump(eval_results, f, indent=2)

    print("\n" + "=" * 70)
    print("LEAK-FREE RECURRENCE MODEL COMPARISON RESULTS")
    print("=" * 70)
    print(df_eval.to_string())


if __name__ == "__main__":
    main()
