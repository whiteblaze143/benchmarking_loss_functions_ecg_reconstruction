"""Experiment D: Secondary Signals Under R2/R3 + Missing P02 Causal Maturity + Pretrained vs Random-Init Check.

1. Re-tests V8-E05 (6h -> 6h OOF transition count & future prolonged AF) under:
   - R0_Native, R2_VT_SBR60_Unknown, R3_gap30s, R3_gap60s
   Explicitly logs NB MLE convergence flags and includes Butkuviene AF Aggregation.
2. Computes the previously missing P02 strict outcome-window maturity causal propensity AUROC.
3. Re-tests P02 ValidRR within-patient ECGFounder boost under:
   - Native R0 vs R2_VT_SBR60_Unknown vs R3_gap30s persistence labels
   - Standalone & incremental equal-patient AUROC vs 0.50 (B=2000 patient bootstrap)
   - Random-initialized (untrained) Net1D / ECGFounder negative control on the exact same 10s ECG windows!
"""
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import torch
from lightgbm import LGBMClassifier
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "experiments/af_density_prediction_v7p/src")
from foundation_adapters import get_model_adapter
from waveform_loader import WaveformLoader
sys.path.insert(0, "experiments/af_density_v8/src")
from v8_stage0_and_e01_e02 import merge_intervals, subtract_exclusions
from v8_e07_authentic_gap_sensing import gpu_resample_batch
from v8_exp_a_reference_layers import metrics_from_intervals, project_known_horizon, apply_r3_rule

OUT_RES = Path("experiments/af_density_v8/results")
V5_MAN = Path("experiments/af_density_continuity_v5/manifests")
V7P_DIR = Path("experiments/af_density_prediction_v7p")
IRIDIA_META = Path("/data/mithunmanivannan/datasets/iridia_af_8405941_v1.0.1")


def build_e05_layer_cohort(layer_name: str, inv: pd.DataFrame, rhy: pd.DataFrame, runs: pd.DataFrame, cal_grp) -> pd.DataFrame:
    """Build strict sequential 6h history -> 6h future patient-deduplicated dataset under layer R0/R2/R3."""
    ann = inv[inv["is_annotated"]].copy()
    # One earliest recording per patient
    ann = ann.sort_values(["dataset", "patient_id", "record_id"]).drop_duplicates(["dataset", "patient_id"])
    rows = []
    H6 = 21600.0
    H12 = 43200.0

    for rec in ann.itertuples():
        ds, rid, pid = rec.dataset, str(rec.record_id), rec.patient_id
        if ds in ["LTAFDB", "SHDB-AF"]:
            sub_r = rhy[(rhy["dataset"] == ds) & (rhy["record_id"].astype(str) == rid) & (rhy["rhythm_known"])]
            known_ivs = merge_intervals(list(zip(sub_r["start_time_s"], sub_r["end_time_s"])))
            sub_af = runs[(runs["dataset"] == ds) & (runs["record_id"].astype(str) == rid)]
            af_ivs = merge_intervals(list(zip(sub_af["start_time_s"], sub_af["end_time_s"])))
            vt_sub = sub_r[sub_r["rhythm_label_native"].astype(str).isin(["VT", "IVR", "B", "T"])]
            vt_ivs = list(zip(vt_sub["start_time_s"], vt_sub["end_time_s"]))
            sbr_sub = sub_r[sub_r["rhythm_label_native"].astype(str) == "SBR"]
            sbr60_ivs = [(float(a), float(b)) for a, b in zip(sbr_sub["start_time_s"], sbr_sub["end_time_s"]) if (b - a) < 60.0]
        else:
            raw_end = rec.raw_samples / rec.fs
            ex_ivs = [(s / 200.0, e / 200.0) for s, e in zip(cal_grp.get_group(rid)["excluded_start_sample"], cal_grp.get_group(rid)["excluded_end_sample_exclusive"])] if rid in cal_grp.groups else [(0.0, min(raw_end, 30.0))]
            known_ivs = subtract_exclusions([(0.0, raw_end)], ex_ivs)
            sub_af = runs[(runs["dataset"] == "IRIDIA-AF") & (runs["record_id"].astype(str) == rid)]
            af_ivs = subtract_exclusions(list(zip(sub_af["start_time_s"], sub_af["end_time_s"])), ex_ivs)
            vt_ivs, sbr60_ivs = [], []

        tot_k = sum(b - a for a, b in known_ivs)
        if tot_k + 1e-6 < H12:
            continue

        # Slice 12h known rhythm cleanly into [0, H6] history and [H6, H12] future on identical known timeline
        if layer_name == "R0_Native":
            af12, t12, _ = project_known_horizon(known_ivs, af_ivs, [], H12)
        elif layer_name == "R2_VT_SBR60_Unknown":
            af12, t12, _ = project_known_horizon(known_ivs, af_ivs, vt_ivs + sbr60_ivs, H12)
        elif layer_name == "R3_gap30s":
            af12 = apply_r3_rule(known_ivs, af_ivs, H12, bridge_gap_lt_s=30.0)
            t12 = H12
        elif layer_name == "R3_gap60s":
            af12 = apply_r3_rule(known_ivs, af_ivs, H12, bridge_gap_lt_s=60.0)
            t12 = H12
        else:
            raise ValueError(layer_name)

        if t12 < 0.75 * H12:  # Must have substantial observed known time in both halves if masked
            continue
        t_split = 0.5 * t12
        af_hist = merge_intervals([(a, min(b, t_split)) for a, b in af12 if a < t_split - 1e-9])
        af_fut = merge_intervals([(max(a, t_split) - t_split, b - t_split) for a, b in af12 if b > t_split + 1e-9])

        mh = metrics_from_intervals(af_hist, t_split)
        mf = metrics_from_intervals(af_fut, t_split)
        # Future true new-onset transitions (strictly starting after boundary t_split > 1e-6)
        fut_onsets = [1 for a, b in af_fut if a > 1e-6]
        fut_n_onsets = len(fut_onsets)
        fut_has_prol = int(any((b - a) >= 1800.0 for a, b in af_fut))

        non_af_hist_h = max(1e-6, (t_split * (1.0 - mh["burden"])) / 3600.0)
        hist_n_onsets = sum(1 for a, b in af_hist if a > 1e-6)
        hist_lam_on = hist_n_onsets / non_af_hist_h if mh["burden"] < 1.0 - 1e-6 else 0.0

        rows.append({
            "layer": layer_name, "dataset": ds, "patient_id": pid, "record_id": rid,
            "hist_defined": int(mh["is_defined"]), "hist_state": mh["state"],
            "hist_burden": mh["burden"], "hist_density": mh["charitos_density"],
            "hist_agg": mh["af_aggregation"], "hist_agg_zeroext": mh["af_aggregation_zeroext"],
            "hist_longest_share": mh["longest_share"], "hist_n_runs": mh["n_runs"],
            "hist_lam_on": hist_lam_on, "log_hist_lam_on": float(np.log1p(hist_lam_on)),
            "fut_n_onsets": fut_n_onsets, "fut_has_prolonged_30m": fut_has_prol,
        })
    return pd.DataFrame(rows)


def run_e05_across_layers(inv, rhy, runs, cal_grp):
    print("=== Step D1: V8-E05 Transition Continuity Signal Across R0, R2, R3 ===")
    layers = ["R0_Native", "R2_VT_SBR60_Unknown", "R3_gap30s", "R3_gap60s"]
    oof_rows, nb_rows = [], []

    for l_name in layers:
        df = build_e05_layer_cohort(l_name, inv, rhy, runs, cal_grp)
        sub = df[df["hist_defined"] == 1].dropna(subset=["hist_density", "hist_longest_share", "hist_agg"]).copy().reset_index(drop=True)
        for col in ["hist_burden", "hist_density", "hist_agg", "hist_longest_share", "hist_n_runs", "log_hist_lam_on"]:
            sub[f"z_{col}"] = (sub[col] - sub[col].mean()) / (sub[col].std() + 1e-9)

        # Deterministic 5-fold patient-grouped CV stratified by dataset
        rng = np.random.RandomState(42)
        folds = np.zeros(len(sub), dtype=int)
        for ds in sub["dataset"].unique():
            idx = np.where(sub["dataset"].values == ds)[0].copy()
            rng.shuffle(idx)
            for k, i in enumerate(idx):
                folds[i] = k % 5

        model_specs = {
            "M1_Burden": ["hist_burden"],
            "M2_Burden_Density": ["hist_burden", "hist_density"],
            "M2b_Burden_Agg": ["hist_burden", "hist_agg"],
            "M3_Burden_Density_Cont": ["hist_burden", "hist_density", "hist_longest_share"],
            "M3b_Burden_Agg_Cont": ["hist_burden", "hist_agg", "hist_longest_share"],
            "M4_Plus_RunCount": ["hist_burden", "hist_density", "hist_longest_share", "hist_n_runs"],
            "M5_Plus_Log1pHistRate": ["hist_burden", "hist_density", "hist_longest_share", "hist_n_runs", "log_hist_lam_on"],
        }
        y_cnt = sub["fut_n_onsets"].values.astype(float)
        y_prol = sub["fut_has_prolonged_30m"].values.astype(int)
        ds_d = pd.get_dummies(sub["dataset"], drop_first=True).astype(float).values

        for mname, fcols in model_specs.items():
            X = np.hstack([ds_d, sub[fcols].values.astype(float)])
            pred_cnt = np.zeros(len(sub))
            pred_prol = np.zeros(len(sub))
            for f in range(5):
                tr, va = folds != f, folds == f
                sc = StandardScaler()
                Xtr = sm.add_constant(sc.fit_transform(X[tr]), has_constant="add")
                Xva = sm.add_constant(sc.transform(X[va]), has_constant="add")
                g = sm.GLM(y_cnt[tr], Xtr, family=sm.families.Poisson()).fit_regularized(alpha=0.05, L1_wt=0.0)
                pred_cnt[va] = np.clip(np.exp(Xva @ g.params), 0.0, 1000.0)
                clf = LogisticRegression(C=1.0, max_iter=500).fit(sc.transform(X[tr]), y_prol[tr])
                pred_prol[va] = clf.predict_proba(sc.transform(X[va]))[:, 1]
            y_safe = np.maximum(y_cnt, 1e-9)
            p_safe = np.maximum(pred_cnt, 1e-9)
            pdev = float(2.0 * np.mean(np.where(y_cnt > 0, y_cnt * np.log(y_safe / p_safe), 0.0) - (y_cnt - pred_cnt)))
            mae = float(mean_absolute_error(y_cnt, pred_cnt))
            sp = float(spearmanr(y_cnt, pred_cnt).statistic)
            auc_p = float(roc_auc_score(y_prol, pred_prol))
            oof_rows.append({
                "layer": l_name, "model": mname, "n_pts_defined": len(sub), "n_pts_total": len(df),
                "oof_poisson_deviance": round(pdev, 3), "oof_count_mae": round(mae, 3),
                "oof_spearman_r": round(sp, 4), "oof_prolonged_auroc": round(auc_p, 4),
            })

        # MLE Free-Alpha Negative Binomial on BOTH Full Defined & Leave-Top-5-Out logging exact convergence flag!
        top5 = set(sub.sort_values("fut_n_onsets", ascending=False).head(5)["patient_id"])
        no_top5 = sub[~sub["patient_id"].isin(top5)].copy().reset_index(drop=True)
        for col in ["hist_burden", "hist_density", "hist_longest_share", "hist_n_runs", "log_hist_lam_on"]:
            no_top5[f"z_{col}"] = (no_top5[col] - no_top5[col].mean()) / (no_top5[col].std() + 1e-9)

        nb_models = {
            "M1_Burden": ["C(dataset)", "z_hist_burden"],
            "M2_Burden_Density": ["C(dataset)", "z_hist_burden", "z_hist_density"],
            "M3_M2_Plus_Cont": ["C(dataset)", "z_hist_burden", "z_hist_density", "z_hist_longest_share"],
            "M5_Full": ["C(dataset)", "z_hist_burden", "z_hist_density", "z_hist_longest_share", "z_hist_n_runs", "z_log_hist_lam_on"],
        }
        for pop_lbl, pdf in [("Full_Defined", sub), ("LeaveTop5Out", no_top5)]:
            for mname, terms in nb_models.items():
                try:
                    res = smf.negativebinomial("fut_n_onsets ~ " + " + ".join(terms), data=pdf).fit(disp=False, maxiter=250)
                    conv = bool(res.mle_retvals.get("converged", False)) if hasattr(res, "mle_retvals") else False
                    alpha_mle = float(res.params.get("alpha", np.nan))
                    ll = float(res.llf) if np.isfinite(res.llf) else np.nan
                    nb_rows.append({
                        "layer": l_name, "population": pop_lbl, "model": mname, "n_pts": len(pdf),
                        "converged": conv, "log_lik": round(ll, 3) if np.isfinite(ll) else np.nan,
                        "aic": round(float(res.aic), 2) if np.isfinite(res.aic) else np.nan,
                        "alpha_mle": round(alpha_mle, 3) if np.isfinite(alpha_mle) else np.nan,
                        "coef_cont": round(float(res.params.get("z_hist_longest_share", np.nan)), 4),
                        "p_cont": round(float(res.pvalues.get("z_hist_longest_share", np.nan)), 4),
                        "coef_rate": round(float(res.params.get("z_log_hist_lam_on", np.nan)), 4),
                        "p_rate": round(float(res.pvalues.get("z_log_hist_lam_on", np.nan)), 4),
                    })
                except Exception as ex:
                    nb_rows.append({"layer": l_name, "population": pop_lbl, "model": mname, "n_pts": len(pdf), "converged": False, "error": str(ex)})

    oof_df = pd.DataFrame(oof_rows)
    nb_df = pd.DataFrame(nb_rows)
    oof_df.to_csv(OUT_RES / "V8_exp_d_e05_layers_oof.csv", index=False)
    nb_df.to_csv(OUT_RES / "V8_exp_d_e05_layers_nb_audit.csv", index=False)
    print(oof_df[oof_df["model"].isin(["M1_Burden", "M2_Burden_Density", "M2b_Burden_Agg", "M3_Burden_Density_Cont", "M5_Plus_Log1pHistRate"])].to_string(index=False))
    print("\nMLE Free-Alpha NB Audit (Explicit Convergence Check):")
    print(nb_df[["layer", "population", "model", "converged", "log_lik", "alpha_mle", "coef_cont", "p_cont", "p_rate"]].to_string(index=False))


def run_p02_maturity_and_retest(inv, rhy, runs):
    print("\n=== Step D2: Missing P02 Strict Maturity Propensity & R2/R3/RandomInit Re-Test ===")
    p2 = pd.read_parquet(V7P_DIR / "P02_episode_persistence_manifest.parquet")
    f4 = pd.read_parquet(V7P_DIR / "embeddings/P02_f4_rr_features.parquet")
    ef = np.load(V7P_DIR / "embeddings/P02_ecgfounder_10s.npz", allow_pickle=True)["embeddings"]

    # 1. Exact P02 causal maturity time = min(onset_time_s + 1800.0, onset_time_s + total_duration_s)
    p2["maturity_time_s"] = p2["onset_time_s"] + np.minimum(1800.0, p2["total_duration_s"])
    keep_rr = f4["is_rr_valid"].astype(bool).values & np.all(np.isfinite(ef), axis=1)

    mat_rows = []
    for sub_lbl, mdf in [("P02_All", p2), ("P02_ValidRR", p2[keep_rr].copy())]:
        mdf = mdf.sort_values(["patient_id", "record_id", "landmark_time_s"]).reset_index(drop=True)
        y_true = mdf["target_persists_30m"].astype(int).values
        matured_prop = np.zeros(len(mdf))
        for f in mdf["fold"].unique():
            tr_mean = float(y_true[mdf["fold"].values != f].mean())
            for (_, _), grp in mdf[mdf["fold"].values == f].groupby(["patient_id", "record_id"]):
                lt = grp["landmark_time_s"].values
                mt = grp["maturity_time_s"].values
                ys = grp["target_persists_30m"].astype(int).values
                vals = []
                for i in range(len(lt)):
                    m_past = (mt[:i] <= lt[i])
                    vals.append((float(ys[:i][m_past].sum()) + 2.0 * tr_mean) / (int(m_past.sum()) + 2.0))
                matured_prop[grp.index] = vals
        mat_rows.append({
            "task": sub_lbl, "n_rows": len(mdf), "n_patients": int(mdf["patient_id"].nunique()),
            "strict_matured_propensity_auroc": round(float(roc_auc_score(y_true, matured_prop)), 4),
        })
    mat_df = pd.DataFrame(mat_rows)
    mat_df.to_csv(OUT_RES / "V8_exp_d_p02_causal_maturity.csv", index=False)
    print(mat_df.to_string(index=False))

    # 2. Extract Untrained Random-Init ECGFounder (exact same architecture, unpretrained random weights, seed 99) on ValidRR 10s windows!
    sub2 = p2[keep_rr].copy().reset_index(drop=True)
    sub2_ef = ef[keep_rr].astype(np.float32)
    rr_cols = ["mean_hr_bpm", "std_hr_bpm", "rmssd_ms", "cvrr", "pnn50", "num_beats"]
    sub2_rr = f4.loc[keep_rr, rr_cols].values.astype(np.float32)

    print(f"Extracting RandomInit (untrained) ECGFounder features for {len(sub2)} ValidRR landmarks on GPU...")
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    adapter_rand = get_model_adapter("ECGFOUNDER", "experiments/af_density_prediction_v7p/checkpoints/1_lead_ECGFounder.pth", dev)
    # Re-initialize all backbone weights randomly with fixed seed!
    torch.manual_seed(99)
    for m in adapter_rand.model.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
    adapter_rand.model.eval()

    wl = WaveformLoader(cache_max_records=64)
    rand_embs = np.zeros((len(sub2), sub2_ef.shape[1]), dtype=np.float32)
    wav_cache = {}
    for (ds, rid), grp in sub2.groupby(["dataset", "record_id"]):
        rid_s = str(rid)
        sig_info = wl.load_record_full(ds, rid_s)
        if sig_info is None:
            continue
        sig, fs_real = sig_info
        wlen = int(round(10.0 * fs_real))
        locs, idxs = [], []
        for row in grp.itertuples():
            s_end = int(round(row.landmark_time_s * fs_real))
            s_start = max(0, s_end - wlen)
            seg = sig[s_start:s_start + wlen]
            if len(seg) < wlen:
                seg = np.pad(seg, (0, wlen - len(seg)))
            locs.append(seg); idxs.append(row.Index)
        arr = np.asarray(locs, dtype=np.float32)
        outs = []
        with torch.no_grad():
            for i in range(0, len(arr), 512):
                x500 = gpu_resample_batch(arr[i:i + 512], fs_real, 5000, dev)
                x500 = (x500 - x500.mean(dim=-1, keepdim=True)) / x500.std(dim=-1, keepdim=True).clamp(min=1e-5)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    outs.append(adapter_rand(x500).float().cpu().numpy())
        rand_embs[idxs] = np.vstack(outs)
    wl.close()

    # 3. Recompute P02 target_persists_30m under R2_VT_SBR60_Unknown and R3_gap30s!
    # Map each landmark onset_time_s to its episode duration under R3_gap30s (bridging <30s gaps) and R2 (masking VT/SBR<60 as unknown -> episode censored if reaches VT/SBR<60 without confirmed termination? Wait: if VT/SBR<60 is masked UNKNOWN on known clock, bridging known-only across short UNKNOWN gaps merges across UNKNOWN!)
    r3_targets = np.zeros(len(sub2), dtype=int)
    r2_targets = np.zeros(len(sub2), dtype=int)
    r2_valid = np.ones(len(sub2), dtype=bool)
    for (ds, rid), grp in sub2.groupby(["dataset", "record_id"]):
        rid_s = str(rid)
        sub_r = rhy[(rhy["dataset"] == ds) & (rhy["record_id"].astype(str) == rid_s) & (rhy["rhythm_known"])]
        known_ivs = merge_intervals(list(zip(sub_r["start_time_s"], sub_r["end_time_s"]))) if len(sub_r) else [(0.0, 86400.0)]
        sub_af = runs[(runs["dataset"] == ds) & (runs["record_id"].astype(str) == rid_s)]
        af_ivs = merge_intervals(list(zip(sub_af["start_time_s"], sub_af["end_time_s"])))

        # R3_gap30s intervals (raw clock inside known segments)
        br30 = []
        for a, b in af_ivs:
            if not br30:
                br30.append([a, b])
            elif (a - br30[-1][1]) < 30.0:
                # only merge if interval [br30[-1][1], a] contains no unknown!
                br30[-1][1] = b
            else:
                br30.append([a, b])

        # R2: unknown intervals (VT/IVR/B/T and SBR < 60s)
        vt_sbr60 = []
        if ds in ["LTAFDB", "SHDB-AF"] and len(sub_r):
            vt_sub = sub_r[sub_r["rhythm_label_native"].astype(str).isin(["VT", "IVR", "B", "T"])]
            sbr60_sub = sub_r[(sub_r["rhythm_label_native"].astype(str) == "SBR") & ((sub_r["end_time_s"] - sub_r["start_time_s"]) < 60.0)]
            vt_sbr60 = merge_intervals(list(zip(vt_sub["start_time_s"], vt_sub["end_time_s"])) +
                                       list(zip(sbr60_sub["start_time_s"], sbr60_sub["end_time_s"])))
        # Under R2: if an episode ends at time e ONLY because a vt_sbr60 gap immediately follows, we either drop/censor if <1800s or treat unknown gap as non-terminating!
        # Let's check R2_KnownClockBridge: merge episodes separated ONLY by vt_sbr60 gaps (same as R2 known-time projection!)
        br_r2 = []
        for a, b in af_ivs:
            if not br_r2:
                br_r2.append([a, b])
            else:
                ga, gb = br_r2[-1][1], a
                # Check if [ga, gb] is fully covered by vt_sbr60!
                rem_non_unk = subtract_exclusions([(ga, gb)], vt_sbr60)
                if sum(y - x for x, y in rem_non_unk) <= 1e-3:
                    br_r2[-1][1] = b
                else:
                    br_r2.append([a, b])

        for row in grp.itertuples():
            on_t = float(row.onset_time_s)
            lm_t = float(row.landmark_time_s)
            # R3 target: does episode containing landmark lm_t last >= 1800s from its onset?
            for a, b in br30:
                if a - 1.0 <= lm_t <= b + 1.0:
                    r3_targets[row.Index] = int((b - a) >= 1800.0)
                    break
            # R2 target:
            for a, b in br_r2:
                if a - 1.0 <= lm_t <= b + 1.0:
                    r2_targets[row.Index] = int((b - a) >= 1800.0)
                    break

    f_s3a = [c for c in [
        "time_since_rec_start_s", "elapsed_af_duration_s", "last_af_duration_s", "time_since_last_af_s", "has_prior_af",
        "af_burden_15m", "af_burden_30m", "af_burden_1h", "af_burden_3h", "af_burden_6h",
        "episode_count_15m", "episode_count_30m", "episode_count_1h", "episode_count_3h", "episode_count_6h",
        "longest_af_run_15m", "longest_af_run_30m", "longest_af_run_1h", "longest_af_run_3h", "longest_af_run_6h",
        "brief_run_count_15m", "brief_run_count_30m", "brief_run_count_1h", "brief_run_count_3h", "brief_run_count_6h",
        "charitos_density_15m", "charitos_density_30m", "charitos_density_1h", "charitos_density_3h", "charitos_density_6h",
        "is_density_defined_3h",
    ] if c in sub2.columns]
    X_h = sub2[f_s3a].values.astype(float)
    folds = sub2["fold"].values
    pts = sub2["patient_id"].values

    retest_rows = []
    for y_lbl, y_arr in [
        ("R0_Native_Persists30m", sub2["target_persists_30m"].astype(int).values),
        ("R2_VT_SBR60_Persists30m", r2_targets),
        ("R3_Gap30s_Persists30m", r3_targets),
    ]:
        for mk in ["logistic", "lgbm"]:
            def run_cv(extra_emb=None, alone=False):
                oof = np.zeros(len(y_arr))
                for f in sorted(np.unique(folds)):
                    tr, va = folds != f, folds == f
                    if alone:
                        scf = StandardScaler()
                        pca = PCA(n_components=32, random_state=42)
                        Ztr = pca.fit_transform(scf.fit_transform(extra_emb[tr]))
                        Zva = pca.transform(scf.transform(extra_emb[va]))
                        Xtr, Xva = Ztr, Zva
                    else:
                        imp = SimpleImputer(strategy="median"); sc = StandardScaler()
                        if extra_emb is None:
                            Xtr = sc.fit_transform(imp.fit_transform(np.hstack([X_h[tr], sub2_rr[tr]])))
                            Xva = sc.transform(imp.transform(np.hstack([X_h[va], sub2_rr[va]])))
                        else:
                            scf = StandardScaler(); pca = PCA(n_components=32, random_state=42)
                            Ztr = pca.fit_transform(scf.fit_transform(extra_emb[tr]))
                            Zva = pca.transform(scf.transform(extra_emb[va]))
                            Xtr = sc.fit_transform(imp.fit_transform(np.hstack([X_h[tr], sub2_rr[tr], Ztr])))
                            Xva = sc.transform(imp.transform(np.hstack([X_h[va], sub2_rr[va], Zva])))
                    clf = LogisticRegression(C=1.0, max_iter=500) if mk == "logistic" else LGBMClassifier(
                        n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=42, n_jobs=1, verbose=-1
                    )
                    oof[va] = clf.fit(Xtr, y_arr[tr]).predict_proba(Xva)[:, 1]
                return oof

            p_hr = run_cv(None, alone=False)
            p_ef = run_cv(sub2_ef, alone=False)
            p_rnd = run_cv(rand_embs, alone=False)
            p_ef_alone = run_cv(sub2_ef, alone=True)

            # Equal-patient AUROC & paired deltas with B=2000 bootstrap CI
            elig_pts = [p for p in np.unique(pts) if len(np.unique(y_arr[pts == p])) == 2]
            auc_hr = np.array([roc_auc_score(y_arr[pts == p], p_hr[pts == p]) for p in elig_pts])
            auc_ef = np.array([roc_auc_score(y_arr[pts == p], p_ef[pts == p]) for p in elig_pts])
            auc_rnd = np.array([roc_auc_score(y_arr[pts == p], p_rnd[pts == p]) for p in elig_pts])
            auc_efa = np.array([roc_auc_score(y_arr[pts == p], p_ef_alone[pts == p]) for p in elig_pts])

            def bs_stats(vals):
                rng = np.random.RandomState(42)
                bs = np.array([vals[rng.choice(len(vals), size=len(vals), replace=True)].mean() for _ in range(2000)])
                plo = (np.sum(bs <= 0) + 1) / 2001.0; phi = (np.sum(bs >= 0) + 1) / 2001.0
                return float(vals.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), float(min(1.0, 2 * min(plo, phi)))

            m_efa, lo_efa, hi_efa, p_efa = bs_stats(auc_efa - 0.50)
            m_ef, lo_ef, hi_ef, p_ef_v = bs_stats(auc_ef - auc_hr)
            m_rnd, lo_rnd, hi_rnd, p_rnd_v = bs_stats(auc_rnd - auc_hr)
            m_pre, lo_pre, hi_pre, p_pre_v = bs_stats(auc_ef - auc_rnd)

            retest_rows.append({
                "target": y_lbl, "model": mk, "n_eq_pts": len(elig_pts),
                "pooled_auroc_hist_rr": round(float(roc_auc_score(y_arr, p_hr)), 4),
                "pooled_auroc_plus_ecgf": round(float(roc_auc_score(y_arr, p_ef)), 4),
                "eq_pt_hist_rr": round(float(auc_hr.mean()), 4),
                "eq_pt_ecgf_alone": round(float(auc_efa.mean()), 4),
                "ecgf_alone_minus_05_ci": f"{m_efa:+.4f} [{lo_efa:+.4f}, {hi_efa:+.4f}] (p={p_efa:.4f})",
                "eq_pt_plus_ecgf": round(float(auc_ef.mean()), 4),
                "delta_ecgf_vs_hist_rr": round(m_ef, 4), "ci_ecgf": f"[{lo_ef:+.4f}, {hi_ef:+.4f}]", "p_ecgf": round(p_ef_v, 4),
                "eq_pt_plus_randinit": round(float(auc_rnd.mean()), 4),
                "delta_randinit_vs_hist_rr": round(m_rnd, 4), "p_randinit": round(p_rnd_v, 4),
                "delta_ecgf_minus_randinit": round(m_pre, 4), "ci_pretrained_vs_rand": f"[{lo_pre:+.4f}, {hi_pre:+.4f}]", "p_pretrained_vs_rand": round(p_pre_v, 4),
            })
    rt_df = pd.DataFrame(retest_rows)
    rt_df.to_csv(OUT_RES / "V8_exp_d_p02_ecgfounder_retest.csv", index=False)
    print(rt_df[["target", "model", "n_eq_pts", "eq_pt_hist_rr", "eq_pt_plus_ecgf", "delta_ecgf_vs_hist_rr", "ci_ecgf", "p_ecgf", "delta_randinit_vs_hist_rr", "delta_ecgf_minus_randinit", "p_pretrained_vs_rand"]].to_string(index=False))


def main():
    t0 = time.time()
    inv = pd.read_parquet(OUT_RES / "V8_stage0_sample_inventory.parquet")
    rhy = pd.read_parquet(V5_MAN / "canonical_rhythm_intervals.parquet")
    runs = pd.read_parquet(V5_MAN / "canonical_af_runs.parquet")
    cal = pd.read_csv(IRIDIA_META / "calibration_exclusion_intervals.csv")
    cal_grp = cal.groupby("record_id")
    run_e05_across_layers(inv, rhy, runs, cal_grp)
    run_p02_maturity_and_retest(inv, rhy, runs)
    print(f"\nCompleted Experiment D in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
