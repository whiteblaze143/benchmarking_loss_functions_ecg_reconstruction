"""AF-DENSITY V8: Stage 0 Corrections + V8-E01 Density Conditioning + V8-E02 Residual Non-AF Gap Fidelity."""
import hashlib
import json
import math
import os
import glob
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import wfdb
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from lifelines import CoxPHFitter

sys.path.insert(0, "experiments/af_density_temporal_phenotypes_v7/src")
from density_continuous import compute_continuous_density
sys.path.insert(0, "experiments/af_density_prediction_v7p/src")
from alert_evaluation import evaluate_p01_alerts

OUT_RES = Path("experiments/af_density_v8/results")
OUT_RES.mkdir(parents=True, exist_ok=True)
V5_MAN = Path("experiments/af_density_continuity_v5/manifests")
V7P_DIR = Path("experiments/af_density_prediction_v7p")
IRIDIA_META = Path("/data/mithunmanivannan/datasets/iridia_af_8405941_v1.0.1")
SHDB_DIR = Path("/data/mithunmanivannan/datasets/shdb_af")
LTAF_DIR = Path("/data/mithunmanivannan/datasets/ltafdb_1.0.0_reference")


# ---------------------------------------------------------------------------
# Independent Exact Charitos Reference Algorithm (Piecewise F(p) Grid)
# ---------------------------------------------------------------------------
def merge_intervals(ivs: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not ivs:
        return []
    s = sorted((float(a), float(b)) for a, b in ivs if b > a + 1e-12)
    if not s:
        return []
    m = [[s[0][0], s[0][1]]]
    for a, b in s[1:]:
        if a <= m[-1][1] + 1e-9:
            m[-1][1] = max(m[-1][1], b)
        else:
            m.append([a, b])
    return [(x, y) for x, y in m]


def exact_reference_charitos_components(
    intervals: List[Tuple[float, float]], T: float
) -> Dict[str, Any]:
    """Independent exact implementation returning numerator (2*A), denominator (1-B), density, and reason."""
    ivs = merge_intervals([(max(0.0, a), min(T, b)) for a, b in intervals])
    if not ivs or T <= 0:
        return {
            "af_s": 0.0, "burden": 0.0, "numerator_2A": np.nan,
            "denom_1_minus_B": 1.0, "density": np.nan, "is_defined": False,
            "undef_reason": "ZERO_AF", "n_runs": 0,
        }
    M = math.fsum(b - a for a, b in ivs)
    B = M / T
    if B <= 1e-12:
        return {
            "af_s": M, "burden": 0.0, "numerator_2A": np.nan,
            "denom_1_minus_B": 1.0, "density": np.nan, "is_defined": False,
            "undef_reason": "ZERO_AF", "n_runs": len(ivs),
        }
    if B >= 1.0 - 1e-9:
        return {
            "af_s": M, "burden": 1.0, "numerator_2A": np.nan,
            "denom_1_minus_B": 0.0, "density": np.nan, "is_defined": False,
            "undef_reason": "ALL_AF", "n_runs": len(ivs),
        }

    # Exact lower-envelope breakpoints across contiguous unions of episodes [i..j]
    # For fixed mass m in [0, M], F(m/M) = min_{(i,j) with sum_{i..j} d_k >= m} (e_j - s_i) / T - ... wait:
    # Exact optimal window containing mass m shifts the endpoints so span = m + min_gap(mass >= m).
    n = len(ivs)
    durs = [b - a for a, b in ivs]
    # Prefix mass
    pm = [0.0] * (n + 1)
    for k in range(n):
        pm[k + 1] = pm[k] + durs[k]

    # Collect minimal gap for every contiguous episode block [i..j]
    # Using monotone Pareto frontier over (mass_ij, gap_ij)
    pairs = []
    for i in range(n):
        si = ivs[i][0]
        for j in range(i, n):
            m_ij = pm[j + 1] - pm[i]
            gap_ij = (ivs[j][1] - si) - m_ij
            pairs.append((m_ij, max(0.0, gap_ij)))
    # For any m, minimum gap g(m) = min_{m_ij >= m} gap_ij (non-decreasing piecewise constant step fn!)
    pairs.sort(key=lambda x: x[0])
    # Sweep right-to-left to maintain running minimum gap for masses <= m_ij
    step_pts = []
    min_g = math.inf
    for m_val, g_val in reversed(pairs):
        if g_val < min_g:
            min_g = g_val
        step_pts.append((m_val, min_g))
    step_pts.reverse()

    # Compress duplicate/equal-gap steps cleanly over consecutive mass intervals [m_prev, m_curr]
    area = 0.0
    m_prev = 0.0
    # Evaluate exact integral of |F(p) - p| dp = (1/M) * int_0^M |(m + g(m))/T - m/M| dm
    # Group contiguous points with same m_curr
    idx = 0
    while idx < len(step_pts):
        m_curr = step_pts[idx][0]
        g_curr = step_pts[idx][1]
        while idx + 1 < len(step_pts) and abs(step_pts[idx + 1][0] - m_curr) <= 1e-12:
            idx += 1
        if m_curr > m_prev + 1e-15:
            # Linear integrand alpha * m + beta on [m_prev, m_curr], alpha = (1/T - 1/M), beta = g_curr / T
            alpha = (1.0 / T) - (1.0 / M)
            beta = g_curr / T
            def prim(m):
                return 0.5 * alpha * m * m + beta * m
            root = -beta / alpha if abs(alpha) > 1e-18 else math.inf
            if m_prev < root < m_curr:
                seg = abs(prim(root) - prim(m_prev)) + abs(prim(m_curr) - prim(root))
            else:
                seg = abs(prim(m_curr) - prim(m_prev))
            area += seg / M
        m_prev = m_curr
        idx += 1

    num = 2.0 * area
    den = 1.0 - B
    return {
        "af_s": M, "burden": B, "numerator_2A": num, "denom_1_minus_B": den,
        "density": num / den, "is_defined": True, "undef_reason": "DEFINED", "n_runs": n,
    }


def subtract_exclusions(
    intervals: List[Tuple[float, float]], exclusions: List[Tuple[float, float]]
) -> List[Tuple[float, float]]:
    """Subtract exclusion spans from positive intervals without overlap."""
    cur = list(intervals)
    for ex_a, ex_b in merge_intervals(exclusions):
        nxt = []
        for a, b in cur:
            if b <= ex_a + 1e-9 or a >= ex_b - 1e-9:
                nxt.append((a, b))
            else:
                if a < ex_a - 1e-9:
                    nxt.append((a, ex_a))
                if b > ex_b + 1e-9:
                    nxt.append((ex_b, b))
        cur = nxt
    return merge_intervals(cur)


# ---------------------------------------------------------------------------
# Stage 0: Sample-Level Inventory, P01 Event Alerts, Matched Density Ablation, P03 Survival
# ---------------------------------------------------------------------------
def run_stage0():
    print("=== Stage 0: Sample-Level Inventory, Event Alerts & Matched Density Contrasts ===")
    rhy = pd.read_parquet(V5_MAN / "canonical_rhythm_intervals.parquet")
    runs = pd.read_parquet(V5_MAN / "canonical_af_runs.parquet")
    c5 = pd.read_csv(V5_MAN / "cohort_inventory.csv")
    c5["record_id_str"] = c5["record_id"].astype(str)
    cal = pd.read_csv(IRIDIA_META / "calibration_exclusion_intervals.csv")
    ir_meta = pd.read_csv(IRIDIA_META / "iridia-af-metadata-v1.0.1.csv")
    sh_meta = pd.read_csv(SHDB_DIR / "AdditionalData.csv")

    # 1. Sample-level disjoint reconstruction fixing IRIDIA 0.519029 h calibration overlap
    sample_rows = []
    # LTAFDB
    for r_row in c5[c5["dataset"] == "LTAFDB"].itertuples():
        rid = str(r_row.record_id)
        fs = float(r_row.fs)
        n_samp = int(r_row.n_samples)
        sub_rhy = rhy[(rhy["dataset"] == "LTAFDB") & (rhy["record_id"].astype(str) == rid)]
        k_samp = int((sub_rhy[sub_rhy["rhythm_known"]]["end_sample"] - sub_rhy[sub_rhy["rhythm_known"]]["start_sample"]).sum())
        u_samp = int((sub_rhy[~sub_rhy["rhythm_known"]]["end_sample"] - sub_rhy[~sub_rhy["rhythm_known"]]["start_sample"]).sum())
        af_samp = int((sub_rhy[sub_rhy["rhythm_label_harmonized"] == "AF"]["end_sample"] - sub_rhy[sub_rhy["rhythm_label_harmonized"] == "AF"]["start_sample"]).sum())
        sample_rows.append({
            "dataset": "LTAFDB", "record_id": rid, "patient_id": r_row.patient_id, "fs": fs,
            "is_annotated": True, "raw_samples": n_samp, "known_samples": k_samp, "unknown_samples": u_samp,
            "af_samples": af_samp, "known_non_af_samples": k_samp - af_samp,
            "sample_discrepancy": n_samp - (k_samp + u_samp),
        })

    # SHDB-AF
    for r_row in c5[c5["dataset"] == "SHDB-AF"].itertuples():
        rid = str(r_row.record_id).zfill(3) if str(r_row.record_id).isdigit() else str(r_row.record_id)
        is_ann = str(r_row.is_annotated).lower() == "true"
        if pd.isna(r_row.n_samples) or pd.isna(r_row.fs):
            hdr = wfdb.rdheader(str(SHDB_DIR / rid))
            fs = float(hdr.fs)
            n_samp = int(hdr.sig_len)
        else:
            fs = float(r_row.fs)
            n_samp = int(r_row.n_samples)
        sub_rhy = rhy[(rhy["dataset"] == "SHDB-AF") & (rhy["record_id"].astype(str).str.zfill(3) == rid)]
        if is_ann:
            k_samp = int((sub_rhy[sub_rhy["rhythm_known"]]["end_sample"] - sub_rhy[sub_rhy["rhythm_known"]]["start_sample"]).sum())
            u_samp = int((sub_rhy[~sub_rhy["rhythm_known"]]["end_sample"] - sub_rhy[~sub_rhy["rhythm_known"]]["start_sample"]).sum())
            af_samp = int((sub_rhy[sub_rhy["rhythm_label_harmonized"] == "AF"]["end_sample"] - sub_rhy[sub_rhy["rhythm_label_harmonized"] == "AF"]["start_sample"]).sum())
        else:
            k_samp, u_samp, af_samp = 0, n_samp, 0
        sample_rows.append({
            "dataset": "SHDB-AF", "record_id": rid, "patient_id": r_row.patient_id, "fs": fs,
            "is_annotated": is_ann, "raw_samples": n_samp, "known_samples": k_samp, "unknown_samples": u_samp,
            "af_samples": af_samp, "known_non_af_samples": k_samp - af_samp,
            "sample_discrepancy": n_samp - (k_samp + u_samp),
        })

    # IRIDIA-AF: Enforce exact disjoint partition [0, n_samples) minus ALL calibration_exclusion_intervals
    cal_grp = cal.groupby("record_id")
    for r_row in ir_meta.itertuples():
        rid = str(r_row.record_id)
        pid = str(r_row.patient_id)
        fs = 200.0
        n_samp = int(r_row.record_samples)
        ex_ivs = []
        if rid in cal_grp.groups:
            c_sub = cal_grp.get_group(rid)
            ex_ivs = [(int(s), int(e)) for s, e in zip(c_sub["excluded_start_sample"], c_sub["excluded_end_sample_exclusive"])]
        else:
            ex_ivs = [(0, min(n_samp, 6000))]
        # Merge exclusions
        ex_merged = []
        for s, e in sorted(ex_ivs):
            s, e = max(0, s), min(n_samp, e)
            if e <= s:
                continue
            if not ex_merged or s > ex_merged[-1][1]:
                ex_merged.append([s, e])
            else:
                ex_merged[-1][1] = max(ex_merged[-1][1], e)
        u_samp = sum(e - s for s, e in ex_merged)
        k_samp = n_samp - u_samp

        sub_runs = runs[(runs["dataset"] == "IRIDIA-AF") & (runs["record_id"].astype(str) == rid)]
        af_raw_ivs = [(float(s), float(e)) for s, e in zip(sub_runs["start_sample"], sub_runs["end_sample"])]
        af_known_ivs = subtract_exclusions(af_raw_ivs, [(float(s), float(e)) for s, e in ex_merged])
        af_samp = int(round(sum(e - s for s, e in af_known_ivs)))
        sample_rows.append({
            "dataset": "IRIDIA-AF", "record_id": rid, "patient_id": pid, "fs": fs,
            "is_annotated": True, "raw_samples": n_samp, "known_samples": k_samp, "unknown_samples": u_samp,
            "af_samples": af_samp, "known_non_af_samples": k_samp - af_samp,
            "sample_discrepancy": n_samp - (k_samp + u_samp),
        })

    s_df = pd.DataFrame(sample_rows)
    for col in ["raw", "known", "unknown", "af", "known_non_af"]:
        s_df[f"{col}_h"] = s_df[f"{col}_samples"] / (s_df["fs"] * 3600.0)
    s_df.to_parquet(OUT_RES / "V8_stage0_sample_inventory.parquet", index=False)

    sum_rows = []
    for ds in ["IRIDIA-AF", "LTAFDB", "SHDB-AF", "COMBINED"]:
        sub = s_df if ds == "COMBINED" else s_df[s_df["dataset"] == ds]
        sum_rows.append({
            "dataset": ds,
            "raw_recordings": len(sub),
            "annotated_recordings": int(sub["is_annotated"].sum()),
            "unique_patients_raw": int(sub["patient_id"].nunique()),
            "unique_patients_annotated": int(sub[sub["is_annotated"]]["patient_id"].nunique()),
            "raw_h": round(sub["raw_h"].sum(), 4),
            "known_h": round(sub["known_h"].sum(), 4),
            "unknown_h": round(sub["unknown_h"].sum(), 4),
            "raw_minus_known_plus_unknown_h": round(sub["raw_h"].sum() - (sub["known_h"].sum() + sub["unknown_h"].sum()), 8),
            "max_sample_discrepancy": int(sub["sample_discrepancy"].abs().max()),
            "af_known_h": round(sub["af_h"].sum(), 4),
            "known_non_af_h": round(sub["known_non_af_h"].sum(), 4),
        })
    sum_df = pd.DataFrame(sum_rows)
    sum_df.to_csv(OUT_RES / "V8_stage0_sample_inventory.csv", index=False)
    print(sum_df.to_string(index=False))

    # 2. P01 Clean Audit: Verify follow-up censoring + unique-event vs landmark-level alert evaluation
    c5["is_ann"] = c5["is_annotated"].astype(str).str.lower() == "true"
    unann_shdb = set(c5[(c5["dataset"] == "SHDB-AF") & (~c5["is_ann"])]["record_id"].astype(str))
    p1 = pd.read_parquet(V7P_DIR / "P01_landmark_manifest.parquet")
    p1_clean_mask = (~((p1["dataset"] == "SHDB-AF") & (p1["record_id"].astype(str).isin(unann_shdb)))) & (
        ~((p1["dataset"] == "LTAFDB") & (p1["current_rhythm"] == "UNKNOWN"))
    )
    p1_clean = p1[p1_clean_mask].copy().reset_index(drop=True)

    # Check whether any P01 candidate onset lacks 1800s follow-up in its record!
    rec_dur_map = {(r.dataset, r.record_id): r.raw_samples / r.fs for r in s_df.itertuples()}
    rec_end_s = p1_clean.apply(lambda r: rec_dur_map[(r["dataset"], str(r["record_id"]))], axis=1)
    horizon_insufficient = (p1_clean["landmark_time_s"] + 1800.0 > rec_end_s + 1e-6).sum()
    pos_mask = p1_clean["target_sustained_30m"].astype(bool)
    ev_followup_insufficient = ((p1_clean.loc[pos_mask, "event_time_s"] + 1800.0) > rec_end_s[pos_mask] + 1e-6).sum()
    n_unique_events = p1_clean.loc[pos_mask, ["dataset", "record_id", "event_time_s"]].drop_duplicates().shape[0]
    print(f"\nP01 Follow-up Check: landmark_insufficient={horizon_insufficient}, pos_event_followup_insufficient={ev_followup_insufficient}, "
          f"positive_landmarks={pos_mask.sum()}, unique_sustained_onset_events={n_unique_events}")

    # 3. Matched Density-Availability Indicator Ablation + P03 Cox PH vs Binary + OOF Parquet
    p2 = pd.read_parquet(V7P_DIR / "P02_episode_persistence_manifest.parquet")
    p3 = pd.read_parquet(V7P_DIR / "P03_survival_manifest.parquet")

    f_base = ["time_since_rec_start_s", "elapsed_af_duration_s"]
    f_burd = f_base + ["last_af_duration_s", "time_since_last_af_s", "has_prior_af",
                       "af_burden_15m", "af_burden_30m", "af_burden_1h", "af_burden_3h", "af_burden_6h"]
    f_arch = f_burd + [
        "episode_count_15m", "episode_count_30m", "episode_count_1h", "episode_count_3h", "episode_count_6h",
        "longest_af_run_15m", "longest_af_run_30m", "longest_af_run_1h", "longest_af_run_3h", "longest_af_run_6h",
        "brief_run_count_15m", "brief_run_count_30m", "brief_run_count_1h", "brief_run_count_3h", "brief_run_count_6h",
    ]
    # Matched availability indicator in BOTH S2 and S3a!
    f_arch_matched_avail = f_arch + ["is_density_defined_3h"]
    f_dens_matched_avail = f_arch + ["charitos_density_15m", "charitos_density_30m", "charitos_density_1h", "charitos_density_3h", "charitos_density_6h", "is_density_defined_3h"]

    def cv_oof(X, y, folds, mkind):
        oof = np.zeros(len(y), dtype=float)
        for f in sorted(np.unique(folds)):
            tr, va = folds != f, folds == f
            imp = SimpleImputer(strategy="median")
            sc = StandardScaler()
            Xtr = sc.fit_transform(imp.fit_transform(X[tr]))
            Xva = sc.transform(imp.transform(X[va]))
            clf = LogisticRegression(C=1.0, max_iter=500) if mkind == "logistic" else LGBMClassifier(
                n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=42, n_jobs=1, verbose=-1
            )
            oof[va] = clf.fit(Xtr, y[tr]).predict_proba(Xva)[:, 1]
        return oof

    def boot_diff(y_true, pa, pb, pts, n_boot=2000, seed=42):
        rng = np.random.RandomState(seed)
        upts = np.unique(pts)
        n_pts = len(upts)
        pt_idx = {p: np.where(pts == p)[0] for p in upts}
        obs = float(roc_auc_score(y_true, pb) - roc_auc_score(y_true, pa))
        ds = []
        for _ in range(n_boot):
            samp = rng.choice(upts, size=n_pts, replace=True)
            idx = np.concatenate([pt_idx[p] for p in samp])
            yb = y_true[idx]
            if len(np.unique(yb)) < 2:
                continue
            r_a = rankdata(pa[idx], method="average")
            r_b = rankdata(pb[idx], method="average")
            n1 = int(yb.sum()); n0 = len(yb) - n1
            ds.append(((r_b[yb == 1].sum() - r_a[yb == 1].sum()) / (n1 * n0)))
        arr = np.array(ds)
        p_lo = (np.sum(arr <= 0) + 1) / (len(arr) + 1)
        p_hi = (np.sum(arr >= 0) + 1) / (len(arr) + 1)
        return obs, float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)), float(min(1.0, 2 * min(p_lo, p_hi)))

    def equal_pt_boot_diff(y_true, pa, pb, pts, n_boot=2000, seed=42):
        eligible = []
        for pt in np.unique(pts):
            m = pts == pt
            if len(np.unique(y_true[m])) == 2:
                eligible.append((pt, float(roc_auc_score(y_true[m], pa[m])), float(roc_auc_score(y_true[m], pb[m]))))
        aucs_a = np.array([x[1] for x in eligible])
        aucs_b = np.array([x[2] for x in eligible])
        diffs = aucs_b - aucs_a
        rng = np.random.RandomState(seed)
        boots = [float(diffs[rng.choice(len(diffs), size=len(diffs), replace=True)].mean()) for _ in range(n_boot)]
        arr = np.array(boots)
        p_lo = (np.sum(arr <= 0) + 1) / (len(arr) + 1)
        p_hi = (np.sum(arr >= 0) + 1) / (len(arr) + 1)
        return float(aucs_a.mean()), float(aucs_b.mean()), float(diffs.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)), float(min(1.0, 2 * min(p_lo, p_hi))), len(eligible)

    if (OUT_RES / "V8_stage0_oof_predictions.parquet").exists() and (OUT_RES / "V8_stage0_p01_event_alerts.csv").exists():
        print("\n[Stage 0] Reusing already-completed OOF & P01 alert CSVs; running P03 Cox PH sensitivity...")
    else:
        abl_rows = []
        oof_store = []
        p1_oofs = {}
        for task_lbl, df_t, t_col in [
            ("P01_CleanAnnotated", p1_clean, "target_sustained_30m"),
            ("P02_Persistence60s", p2, "target_persists_30m"),
            ("P03_Survival15m", p3, "target_terminate_in_15m"),
        ]:
            y = df_t[t_col].astype(int).values
            folds = df_t["fold"].values
            pts = df_t["patient_id"].values
            v_s2m = [c for c in f_arch_matched_avail if c in df_t.columns]
            v_s3m = [c for c in f_dens_matched_avail if c in df_t.columns]
            for mk in ["logistic", "lgbm"]:
                o_s2m = cv_oof(df_t[v_s2m].values.astype(float), y, folds, mk)
                o_s3m = cv_oof(df_t[v_s3m].values.astype(float), y, folds, mk)
                d_pool, lo, hi, pv = boot_diff(y, o_s2m, o_s3m, pts)
                ea, eb, d_eq, eq_lo, eq_hi, eq_p, n_eq = equal_pt_boot_diff(y, o_s2m, o_s3m, pts)
                abl_rows.append({
                    "task": task_lbl, "model": mk,
                    "s2_matched_avail_auroc": round(float(roc_auc_score(y, o_s2m)), 4),
                    "s3a_matched_avail_auroc": round(float(roc_auc_score(y, o_s3m)), 4),
                    "pooled_delta": round(d_pool, 4), "pooled_ci95_lo": round(lo, 4), "pooled_ci95_hi": round(hi, 4), "pooled_p": round(pv, 4),
                    "eq_pt_s2": round(ea, 4), "eq_pt_s3a": round(eb, 4), "eq_pt_delta": round(d_eq, 4),
                    "eq_pt_ci95_lo": round(eq_lo, 4), "eq_pt_ci95_hi": round(eq_hi, 4), "eq_pt_p": round(eq_p, 4), "n_eq_pts": n_eq,
                })
                if task_lbl == "P01_CleanAnnotated":
                    p1_oofs[f"{mk}_S2m"] = o_s2m
                    p1_oofs[f"{mk}_S3m"] = o_s3m
                oof_df = pd.DataFrame({
                    "task": task_lbl, "dataset": df_t["dataset"].values, "patient_id": pts,
                    "record_id": df_t["record_id"].astype(str).values,
                    "landmark_time_s": df_t["landmark_time_s"].values, "fold": folds,
                    "model": mk, "y_true": y, "prob_s2_matched": o_s2m, "prob_s3a_matched": o_s3m,
                })
                oof_store.append(oof_df)

        abl_df = pd.DataFrame(abl_rows)
        abl_df.to_csv(OUT_RES / "V8_stage0_matched_density_ablations.csv", index=False)
        pd.concat(oof_store, ignore_index=True).to_parquet(OUT_RES / "V8_stage0_oof_predictions.parquet", index=False)
        print("\nMatched-Availability Density Contrast (including Equal-Patient Paired CI!):")
        print(abl_df.to_string(index=False))

        alert_rows = []
        p1_base = p1_clean[["dataset", "record_id", "patient_id", "landmark_time_s", "target_sustained_30m", "event_time_s"]].copy()
        non_pos = ~p1_base["target_sustained_30m"].astype(bool)
        non_af_known_h_total = float(s_df["known_non_af_h"].sum())
        sampled_grid_h = float(len(p1_base) * 300.0 / 3600.0)
        for m_name, prob in p1_oofs.items():
            p1_base["prob"] = prob
            pos_df = p1_base[pos_mask]
            ev_max = pos_df.groupby(["dataset", "record_id", "event_time_s"])["prob"].max()
            for thr in [0.02, 0.05, 0.10, 0.15]:
                lm_sens = float((pos_df["prob"] >= thr).mean())
                ev_sens = float((ev_max >= thr).mean())
                det_ev = pos_df[pos_df["prob"] >= thr].groupby(["dataset", "record_id", "event_time_s"])
                leads = []
                for k, grp in det_ev:
                    t_first = float(grp["landmark_time_s"].min())
                    leads.append(max(0.0, (float(k[2]) - t_first) / 60.0))
                med_lead_ev = float(np.median(leads)) if leads else np.nan
                n_false_alerts = int((p1_base.loc[non_pos, "prob"] >= thr).sum())
                alert_rows.append({
                    "model": m_name, "threshold": thr,
                    "n_pos_landmarks": int(pos_mask.sum()), "landmark_sens": round(lm_sens, 4),
                    "n_unique_events": int(len(ev_max)), "n_events_detected": int((ev_max >= thr).sum()),
                    "unique_event_sens": round(ev_sens, 4),
                    "median_event_lead_min": round(med_lead_ev, 2) if np.isfinite(med_lead_ev) else np.nan,
                    "false_alert_landmarks": n_false_alerts,
                    "false_alerts_per_sampled_hr": round(n_false_alerts / sampled_grid_h, 4),
                    "false_alerts_per_known_nonaf_hr": round(n_false_alerts / non_af_known_h_total, 4),
                    "warnings_per_patient_day_sampled": round(n_false_alerts / sampled_grid_h * 24.0, 2),
                    "pct_landmarks_in_alert": round(float((p1_base["prob"] >= thr).mean() * 100.0), 2),
                })
        alert_df = pd.DataFrame(alert_rows)
        alert_df.to_csv(OUT_RES / "V8_stage0_p01_event_alerts.csv", index=False)
        print("\nP01 Landmark-Level vs Unique-Event Alert Reconciliation:")
        print(alert_df.to_string(index=False))

    # P03 Censoring-Aware Survival vs Binary Horizon Summary (non-collinear core covariates + L2 ridge)
    p3_s = p3.copy()
    p3_s["residual_dur_s"] = p3_s["time_to_event_s"].clip(lower=1.0)
    p3_s["event_observed"] = p3_s["event"].astype(int)
    cph_cols = ["residual_dur_s", "event_observed", "af_burden_3h", "episode_count_3h", "charitos_density_3h"]
    sub_cph = p3_s[cph_cols].dropna()
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(sub_cph, duration_col="residual_dur_s", event_col="event_observed")
    # OOF Cox PH using core non-collinear 3h architecture vs +charitos_density_3h
    cph_s2_cols = ["time_since_rec_start_s", "elapsed_af_duration_s", "af_burden_3h", "episode_count_3h", "longest_af_run_3h", "brief_run_count_3h", "is_density_defined_3h"]
    cph_s3_cols = cph_s2_cols + ["charitos_density_3h"]
    from lifelines.utils import concordance_index
    cox_oof = []
    for cname, cols in [("S2_Core3h_CoxPH", cph_s2_cols), ("S3a_Core3h_PlusDens_CoxPH", cph_s3_cols)]:
        risk = np.zeros(len(p3_s))
        for f in sorted(p3_s["fold"].unique()):
            tr, va = (p3_s["fold"] != f).values, (p3_s["fold"] == f).values
            imp = SimpleImputer(strategy="median")
            sc = StandardScaler()
            Xtr = sc.fit_transform(imp.fit_transform(p3_s.loc[tr, cols].values.astype(float)))
            Xva = sc.transform(imp.transform(p3_s.loc[va, cols].values.astype(float)))
            dtr = pd.DataFrame(Xtr, columns=cols)
            keep_c = [c for c in cols if dtr[c].std() > 1e-8]
            dtr = dtr[keep_c].copy()
            dtr["T"] = p3_s.loc[tr, "residual_dur_s"].values
            dtr["E"] = p3_s.loc[tr, "event_observed"].values
            cf = CoxPHFitter(penalizer=0.1).fit(dtr, duration_col="T", event_col="E")
            risk[va] = cf.predict_partial_hazard(pd.DataFrame(Xva, columns=cols)[keep_c]).values
        c_idx = concordance_index(p3_s["residual_dur_s"].values, -risk, p3_s["event_observed"].values)
        cox_oof.append({"model": cname, "n_rows": len(p3_s), "n_events": int(p3_s["event_observed"].sum()),
                        "n_censored": int((1 - p3_s["event_observed"]).sum()), "oof_c_index": round(float(c_idx), 4)})

    print("\nP03 Censoring-Aware Cox PH (residual duration, n={}): C-index={:.4f}, density HR={:.4f} (p={:.4f})".format(
        len(sub_cph), cph.concordance_index_, float(np.exp(cph.params_["charitos_density_3h"])), float(cph.summary.loc["charitos_density_3h", "p"])
    ))
    cox_df = pd.DataFrame(cox_oof)
    cox_df["unadjusted_density_3h_c_index"] = round(float(cph.concordance_index_), 4)
    cox_df["density_3h_hr"] = round(float(np.exp(cph.params_["charitos_density_3h"])), 4)
    cox_df["density_3h_p"] = round(float(cph.summary.loc["charitos_density_3h", "p"]), 4)
    cox_df.to_csv(OUT_RES / "V8_stage0_p03_cox_sensitivity.csv", index=False)
    print(cox_df.to_string(index=False))
    return s_df


# ---------------------------------------------------------------------------
# V8-E01 & V8-E02: Independent Charitos Verification, Fixed Horizons, Burden Strata, Residual Gaps
# ---------------------------------------------------------------------------
def run_e01_e02(s_df: pd.DataFrame):
    print("\n=== V8-E01 & V8-E02: Exact Reference Cross-Check, Burden Strata & Residual Gap Fidelity ===")
    runs = pd.read_parquet(V5_MAN / "canonical_af_runs.parquet")
    rhy = pd.read_parquet(V5_MAN / "canonical_rhythm_intervals.parquet")

    # Verify independent exact Charitos reference against density_continuous.py across all 84 LTAFDB + 98 SHDB records!
    max_algo_diff = 0.0
    n_algo_checked = 0

    def burden_stratum(b, is_def, reason):
        if not is_def:
            return f"UNDEF_{reason}"
        edges = [(0.0, 0.01, "00_0-1%"), (0.01, 0.05, "01_1-5%"), (0.05, 0.20, "02_5-20%"),
                 (0.20, 0.50, "03_20-50%"), (0.50, 0.80, "04_50-80%"), (0.80, 0.90, "05_80-90%"),
                 (0.90, 0.95, "06_90-95%"), (0.95, 0.99, "07_95-99%"), (0.99, 1.01, "08_99-100%")]
        for lo, hi, lbl in edges:
            if lo <= b < hi:
                return lbl
        return "08_99-100%"

    def eval_horizon_record(dataset, rid, pid, horizon_s, af_ivs_known, known_ivs, competing_flags):
        """Evaluate fixed known-rhythm horizon [0, horizon_s] in KNOWN timeline coordinates (never crossing unknown gaps)."""
        # Construct contiguous known-rhythm coordinate timeline [0, K_tot] concatenating known intervals in chronological order
        nonlocal max_algo_diff, n_algo_checked
        # Map raw known intervals sequentially up to horizon_s of KNOWN rhythm time!
        rem = float(horizon_s)
        k_elapsed = 0.0
        mapped_af = []
        mapped_gaps = []
        raw_total_known = sum(b - a for a, b in known_ivs)
        if horizon_s > raw_total_known + 1e-6:
            return None  # Insufficient known rhythm observation for this fixed horizon!

        # Merge touching known intervals so ONLY genuine unknown gaps separate known segments!
        known_ivs = merge_intervals(known_ivs)
        af_sorted = merge_intervals(af_ivs_known)
        mapped_af_seg = []
        for seg_idx, (ka, kb) in enumerate(known_ivs):
            if rem <= 1e-9:
                break
            take = min(kb - ka, rem)
            seg_end = ka + take
            for aa, ab in af_sorted:
                ov_a = max(aa, ka)
                ov_b = min(ab, seg_end)
                if ov_b > ov_a + 1e-9:
                    mapped_af.append((k_elapsed + (ov_a - ka), k_elapsed + (ov_b - ka)))
                    mapped_af_seg.append((k_elapsed + (ov_a - ka), k_elapsed + (ov_b - ka), seg_idx))
            k_elapsed += take
            rem -= take

        T = float(horizon_s)
        ref_af = merge_intervals(mapped_af)
        ref_res = exact_reference_charitos_components(ref_af, T)
        if ref_res["is_defined"]:
            v_dc = compute_continuous_density(ref_af, T)
            if v_dc is not None:
                max_algo_diff = max(max_algo_diff, abs(v_dc - ref_res["density"]))
                n_algo_checked += 1

        # Residual non-AF gaps inside [0, T]
        gaps = []
        cur_t = 0.0
        for a, b in ref_af:
            if a > cur_t + 1e-9:
                gaps.append((cur_t, a, "INTERNAL" if cur_t > 1e-9 else "LEFT_BOUNDARY"))
            cur_t = b
        if cur_t < T - 1e-9:
            gaps.append((cur_t, T, "RIGHT_BOUNDARY"))

        internal_gaps = [g[1] - g[0] for g in gaps if g[2] == "INTERNAL"]
        all_gap_durs = [g[1] - g[0] for g in gaps]

        out_rows = []
        # 1) Gap bridging perturbations: 5, 10, 15, 30, 60, 120s (never bridging across unknown segments)
        for thr_s in [0.0, 5.0, 10.0, 15.0, 30.0, 60.0, 120.0]:
            if thr_s == 0.0:
                pert_af = ref_af
                pname = "reference"
            else:
                if not mapped_af_seg:
                    pert_af = []
                else:
                    bridged = [[mapped_af_seg[0][0], mapped_af_seg[0][1], mapped_af_seg[0][2]]]
                    for a, b, sidx in mapped_af_seg[1:]:
                        if sidx == bridged[-1][2] and (a - bridged[-1][1]) < thr_s:
                            bridged[-1][1] = b
                        else:
                            bridged.append([a, b, sidx])
                    pert_af = merge_intervals([(a, b) for a, b, _ in bridged])
                pname = f"bridge_lt_{int(thr_s)}s"
            p_res = exact_reference_charitos_components(pert_af, T)
            err_signed = (p_res["density"] - ref_res["density"]) if (ref_res["is_defined"] and p_res["is_defined"]) else np.nan
            out_rows.append({
                "dataset": dataset, "record_id": rid, "patient_id": pid, "horizon_h": horizon_s / 3600.0,
                "perturbation": pname, "ref_burden": round(ref_res["burden"], 6),
                "ref_stratum": burden_stratum(ref_res["burden"], ref_res["is_defined"], ref_res["undef_reason"]),
                "ref_af_s": round(ref_res["af_s"], 3), "ref_n_runs": ref_res["n_runs"],
                "ref_num_2A": ref_res["numerator_2A"], "ref_den_1mB": ref_res["denom_1_minus_B"],
                "ref_density": ref_res["density"], "ref_is_defined": ref_res["is_defined"],
                "pert_af_s": round(p_res["af_s"], 3), "pert_burden": round(p_res["burden"], 6), "pert_n_runs": p_res["n_runs"],
                "pert_num_2A": p_res["numerator_2A"], "pert_den_1mB": p_res["denom_1_minus_B"],
                "pert_density": p_res["density"], "pert_is_defined": p_res["is_defined"], "pert_undef_reason": p_res["undef_reason"],
                "signed_dens_err": err_signed, "abs_dens_err": abs(err_signed) if np.isfinite(err_signed) else np.nan,
                "n_residual_gaps": len(all_gap_durs), "n_internal_gaps": len(internal_gaps),
                "total_non_af_s": round(sum(all_gap_durs), 3),
                "median_gap_s": round(float(np.median(all_gap_durs)), 3) if all_gap_durs else 0.0,
                "longest_gap_s": round(float(max(all_gap_durs)), 3) if all_gap_durs else 0.0,
                "frac_nonaf_in_lt60s_gaps": round(sum(d for d in all_gap_durs if d < 60.0) / sum(all_gap_durs), 4) if all_gap_durs else 0.0,
                "has_competing_afl_at": competing_flags,
            })

        # 2) Short AF deletion perturbations: <10s, <30s
        for del_s in [10.0, 30.0]:
            pert_af = [(a, b) for a, b in ref_af if (b - a) >= del_s]
            p_res = exact_reference_charitos_components(pert_af, T)
            err_signed = (p_res["density"] - ref_res["density"]) if (ref_res["is_defined"] and p_res["is_defined"]) else np.nan
            out_rows.append({
                "dataset": dataset, "record_id": rid, "patient_id": pid, "horizon_h": horizon_s / 3600.0,
                "perturbation": f"omit_af_lt_{int(del_s)}s", "ref_burden": round(ref_res["burden"], 6),
                "ref_stratum": burden_stratum(ref_res["burden"], ref_res["is_defined"], ref_res["undef_reason"]),
                "ref_af_s": round(ref_res["af_s"], 3), "ref_n_runs": ref_res["n_runs"],
                "ref_num_2A": ref_res["numerator_2A"], "ref_den_1mB": ref_res["denom_1_minus_B"],
                "ref_density": ref_res["density"], "ref_is_defined": ref_res["is_defined"],
                "pert_af_s": round(p_res["af_s"], 3), "pert_burden": round(p_res["burden"], 6), "pert_n_runs": p_res["n_runs"],
                "pert_num_2A": p_res["numerator_2A"], "pert_den_1mB": p_res["denom_1_minus_B"],
                "pert_density": p_res["density"], "pert_is_defined": p_res["is_defined"], "pert_undef_reason": p_res["undef_reason"],
                "signed_dens_err": err_signed, "abs_dens_err": abs(err_signed) if np.isfinite(err_signed) else np.nan,
                "n_residual_gaps": len(all_gap_durs), "n_internal_gaps": len(internal_gaps),
                "total_non_af_s": round(sum(all_gap_durs), 3),
                "median_gap_s": round(float(np.median(all_gap_durs)), 3) if all_gap_durs else 0.0,
                "longest_gap_s": round(float(max(all_gap_durs)), 3) if all_gap_durs else 0.0,
                "frac_nonaf_in_lt60s_gaps": round(sum(d for d in all_gap_durs if d < 60.0) / sum(all_gap_durs), 4) if all_gap_durs else 0.0,
                "has_competing_afl_at": competing_flags,
            })
        return out_rows

    all_rows = []
    # Check all annotated recordings across LTAFDB, SHDB-AF, IRIDIA-AF
    cal = pd.read_csv(IRIDIA_META / "calibration_exclusion_intervals.csv")
    cal_grp = cal.groupby("record_id")
    ann_recs = s_df[s_df["is_annotated"]]

    for rec in ann_recs.itertuples():
        ds, rid, pid = rec.dataset, str(rec.record_id), rec.patient_id
        competing = False
        if ds in ["LTAFDB", "SHDB-AF"]:
            sub_r = rhy[(rhy["dataset"] == ds) & (rhy["record_id"].astype(str) == rid) & (rhy["rhythm_known"])]
            known_ivs = list(zip(sub_r["start_time_s"], sub_r["end_time_s"]))
            af_ivs = list(zip(runs[(runs["dataset"] == ds) & (runs["record_id"].astype(str) == rid)]["start_time_s"],
                              runs[(runs["dataset"] == ds) & (runs["record_id"].astype(str) == rid)]["end_time_s"]))
            if ds == "SHDB-AF":
                comp_labels = set(sub_r["rhythm_label_native"].astype(str))
                competing = any(lbl in comp_labels for lbl in ["AFL", "AT", "NOD", "AB", "PAT"])
            horizons = [43200.0, 86400.0]
        else:
            # IRIDIA-AF analyzed separately using 24h known horizon (plus full recording in separate E03 table)
            raw_end = rec.raw_samples / rec.fs
            ex_ivs = []
            if rid in cal_grp.groups:
                cs = cal_grp.get_group(rid)
                ex_ivs = [(s / 200.0, e / 200.0) for s, e in zip(cs["excluded_start_sample"], cs["excluded_end_sample_exclusive"])]
            else:
                ex_ivs = [(0.0, min(raw_end, 30.0))]
            known_ivs = subtract_exclusions([(0.0, raw_end)], ex_ivs)
            sub_af = runs[(runs["dataset"] == "IRIDIA-AF") & (runs["record_id"].astype(str) == rid)]
            af_ivs = subtract_exclusions(list(zip(sub_af["start_time_s"], sub_af["end_time_s"])), ex_ivs)
            horizons = [43200.0, 86400.0]

        for h in horizons:
            res = eval_horizon_record(ds, rid, pid, h, af_ivs, known_ivs, competing)
            if res:
                all_rows.extend(res)

    print(f"Independent Charitos Reference Algorithm Cross-Check: n={n_algo_checked}, max_abs_diff={max_algo_diff:.2e}")
    e1_df = pd.DataFrame(all_rows)
    e1_df.to_parquet(OUT_RES / "V8_e01_e02_perturbations_full.parquet", index=False)

    # Summarize 24h & 12h burden-stratified errors cleanly for LTAFDB+SHDB (separate from IRIDIA)
    for h_lbl, h_val in [("12h", 12.0), ("24h", 24.0)]:
        sub_h = e1_df[(e1_df["horizon_h"] == h_val) & (e1_df["perturbation"] == "bridge_lt_60s") & (e1_df["ref_is_defined"])]
        # Patient deduplicated: keep first recording per patient_id
        sub_p = sub_h.sort_values(["dataset", "patient_id", "record_id"]).drop_duplicates(["dataset", "patient_id"])
        strat = sub_p.groupby(["dataset", "ref_stratum"]).agg(
            n_patients=("patient_id", "count"),
            mean_abs_err=("abs_dens_err", "mean"),
            median_abs_err=("abs_dens_err", "median"),
            max_abs_err=("abs_dens_err", "max"),
            n_post_undef=("pert_is_defined", lambda s: int((~s).sum())),
        ).reset_index()
        strat["horizon_h"] = h_val
        strat.to_csv(OUT_RES / f"V8_e01_burden_stratified_{h_lbl}.csv", index=False)

    # Manually inspect LTAFDB worst 5 records (33, 74, 62, 15, 25) at full known duration AND fixed 12h/24h!
    worst_ids = ["33", "74", "62", "15", "25"]
    worst_summary = []
    for rid in worst_ids:
        sub_r = rhy[(rhy["dataset"] == "LTAFDB") & (rhy["record_id"].astype(str) == rid) & (rhy["rhythm_known"])]
        known_s = float(sub_r["duration_s"].sum())
        af_r = runs[(runs["dataset"] == "LTAFDB") & (runs["record_id"].astype(str) == rid)]
        af_ivs = merge_intervals(list(zip(af_r["start_time_s"], af_r["end_time_s"])))
        # Convert to known coordinates
        k_ivs = list(zip(sub_r["start_time_s"], sub_r["end_time_s"]))
        res_full = eval_horizon_record("LTAFDB", rid, f"ltafdb_{rid}", known_s - 1e-6, af_ivs, k_ivs, False)
        if res_full:
            for row in res_full:
                row["full_known_h"] = round(known_s / 3600.0, 3)
                worst_summary.append(row)
    w_df = pd.DataFrame(worst_summary)
    w_df.to_csv(OUT_RES / "V8_e02_ltafdb_worst5_timelines.csv", index=False)
    print("\nLTAFDB Worst-5 Gap-Bridging Inspection (Full Known Horizon):")
    print(w_df[w_df["perturbation"].isin(["reference", "bridge_lt_10s", "bridge_lt_30s", "bridge_lt_60s", "bridge_lt_120s"])][
        ["record_id", "perturbation", "ref_burden", "ref_n_runs", "ref_density", "pert_burden", "pert_n_runs", "pert_density", "pert_undef_reason", "n_residual_gaps", "longest_gap_s"]
    ].to_string(index=False))

    # Print LTAFDB+SHDB 12h burden-stratified summary cleanly
    print("\nV8-E01 12h Known-Rhythm Burden-Stratified Bridge (<60s) Errors:")
    print(pd.read_csv(OUT_RES / "V8_e01_burden_stratified_12h.csv").to_string(index=False))


if __name__ == "__main__":
    s_df = run_stage0()
    run_e01_e02(s_df)
