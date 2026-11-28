"""V8 Revision Pass 2 (Experiment D Statistical & Negative-Control Refinements):

1. Exp D1 (V8-E05):
   - Applies operational R3_gap30s / R3_gap60s bridging STRICTLY SEPARATELY inside [0, 6h] history vs
     [6h, 12h] future (zero boundary gap peeking across the t=6h prediction landmark!).
   - Reports exact unrounded p-values (scipy/statsmodels scientific precision, never clipped to 0.0000).
   - Computes B=2,000 paired patient-level bootstrap 95% CIs on delta OOF Poisson deviance.
2. Exp D2 (P02 Persistence):
   - Evaluates 3 independent random seeds (99, 101, 202) of architecture- and PCA32-matched
     RandomInit_Net1D on all 2,116 P02_ValidRR 10s ECG windows on A100 GPU.
   - Tests R0, R2_VT_SBR60_Unknown, and R3_Gap30s across BOTH full eligible populations (59/52/48)
     AND the exact same-patient intersection (N_common) so cohort selection effects are isolated cleanly!
"""
import sys
import time
from pathlib import Path

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


def build_strict_causal_e05(layer_name: str, inv: pd.DataFrame, rhy: pd.DataFrame, runs: pd.DataFrame, cal_grp) -> pd.DataFrame:
    """Strictly separates [0, 6h] history from [6h, 12h] future BEFORE any R3 gap bridging!"""
    ann = inv[inv["is_annotated"]].sort_values(["dataset", "patient_id", "record_id"]).drop_duplicates(["dataset", "patient_id"])
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
            sbr60 = sub_r[(sub_r["rhythm_label_native"].astype(str) == "SBR") & ((sub_r["end_time_s"] - sub_r["start_time_s"]) < 60.0)]
            ex_r2 = list(zip(vt_sub["start_time_s"], vt_sub["end_time_s"])) + list(zip(sbr60["start_time_s"], sbr60["end_time_s"]))
        else:
            raw_end = rec.raw_samples / rec.fs
            ex_ivs = [(s / 200.0, e / 200.0) for s, e in zip(cal_grp.get_group(rid)["excluded_start_sample"], cal_grp.get_group(rid)["excluded_end_sample_exclusive"])] if rid in cal_grp.groups else [(0.0, min(raw_end, 30.0))]
            known_ivs = subtract_exclusions([(0.0, raw_end)], ex_ivs)
            sub_af = runs[(runs["dataset"] == "IRIDIA-AF") & (runs["record_id"].astype(str) == rid)]
            af_ivs = subtract_exclusions(list(zip(sub_af["start_time_s"], sub_af["end_time_s"])), ex_ivs)
            ex_r2 = []

        if sum(b - a for a, b in known_ivs) + 1e-6 < H12:
            continue

        # Base unbridged [0, 12h] timeline mapped sequentially -> split strictly at t_split BEFORE R3!
        ex_use = ex_r2 if layer_name == "R2_VT_SBR60_Unknown" else []
        af12_raw, t12, _ = project_known_horizon(known_ivs, af_ivs, ex_use, H12)
        if t12 < 0.75 * H12:
            continue
        t_split = 0.5 * t12
        af_hist_raw = merge_intervals([(a, min(b, t_split)) for a, b in af12_raw if a < t_split - 1e-9])
        af_fut_raw = merge_intervals([(max(a, t_split) - t_split, b - t_split) for a, b in af12_raw if b > t_split + 1e-9])

        def bridge_local(ivs, gap_thr):
            if gap_thr <= 0.0 or not ivs:
                return ivs
            br = [[ivs[0][0], ivs[0][1]]]
            for a, b in ivs[1:]:
                if a - br[-1][1] < gap_thr:
                    br[-1][1] = b
                else:
                    br.append([a, b])
            return [(a, b) for a, b in br]

        thr = 30.0 if layer_name == "R3_gap30s" else (60.0 if layer_name == "R3_gap60s" else 0.0)
        af_hist = bridge_local(af_hist_raw, thr)
        af_fut = bridge_local(af_fut_raw, thr)

        mh = metrics_from_intervals(af_hist, t_split)
        fut_n_onsets = sum(1 for a, b in af_fut if a > 1e-6)
        fut_has_prol = int(any((b - a) >= 1800.0 for a, b in af_fut))
        non_af_hist_h = max(1e-6, (t_split * (1.0 - mh["burden"])) / 3600.0)
        hist_n_onsets = sum(1 for a, b in af_hist if a > 1e-6)
        hist_lam_on = hist_n_onsets / non_af_hist_h if mh["burden"] < 1.0 - 1e-6 else 0.0

        rows.append({
            "layer": layer_name, "dataset": ds, "patient_id": pid, "record_id": rid,
            "hist_defined": int(mh["is_defined"]), "hist_burden": mh["burden"],
            "hist_density": mh["charitos_density"], "hist_agg": mh["af_aggregation"],
            "hist_longest_share": mh["longest_share"], "hist_n_runs": mh["n_runs"],
            "log_hist_lam_on": float(np.log1p(hist_lam_on)),
            "fut_n_onsets": fut_n_onsets, "fut_has_prolonged_30m": fut_has_prol,
        })
    return pd.DataFrame(rows)


def run_rev2_e05(inv, rhy, runs, cal_grp):
    print("=== Rev2-Step 1: Strict Non-Peeking E05 + Exact p-values + Paired OOF Deviance CIs ===")
    layers = ["R0_Native", "R2_VT_SBR60_Unknown", "R3_gap30s", "R3_gap60s"]
    oof_rows, delta_rows, nb_rows = [], [], []

    for l_name in layers:
        df = build_strict_causal_e05(l_name, inv, rhy, runs, cal_grp)
        sub = df[df["hist_defined"] == 1].dropna(subset=["hist_density", "hist_longest_share", "hist_agg"]).copy().reset_index(drop=True)
        for col in ["hist_burden", "hist_density", "hist_agg", "hist_longest_share", "hist_n_runs", "log_hist_lam_on"]:
            sub[f"z_{col}"] = (sub[col] - sub[col].mean()) / (sub[col].std() + 1e-9)

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
            "M4_Plus_RunCount": ["hist_burden", "hist_density", "hist_longest_share", "hist_n_runs"],
            "M5_Plus_Log1pHistRate": ["hist_burden", "hist_density", "hist_longest_share", "hist_n_runs", "log_hist_lam_on"],
        }
        y_cnt = sub["fut_n_onsets"].values.astype(float)
        y_prol = sub["fut_has_prolonged_30m"].values.astype(int)
        ds_d = pd.get_dummies(sub["dataset"], drop_first=True).astype(float).values

        row_dev = {}
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
            r_dev = 2.0 * (np.where(y_cnt > 0, y_cnt * np.log(y_safe / p_safe), 0.0) - (y_cnt - pred_cnt))
            row_dev[mname] = r_dev
            oof_rows.append({
                "layer": l_name, "model": mname, "n_pts_defined": len(sub),
                "oof_poisson_deviance": round(float(np.mean(r_dev)), 3),
                "oof_count_mae": round(float(mean_absolute_error(y_cnt, pred_cnt)), 3),
                "oof_spearman_r": round(float(spearmanr(y_cnt, pred_cnt).statistic), 4),
                "oof_prolonged_auroc": round(float(roc_auc_score(y_prol, pred_prol)), 4),
            })

        # Paired B=2000 patient-level bootstrap CI for deviance reduction M3 - M2 and M3 - M1!
        for ma, mb in [("M2_Burden_Density", "M3_Burden_Density_Cont"), ("M1_Burden", "M3_Burden_Density_Cont")]:
            diff_i = row_dev[mb] - row_dev[ma]
            bs = np.array([diff_i[rng.choice(len(diff_i), size=len(diff_i), replace=True)].mean() for _ in range(2000)])
            p_v = float(min(1.0, 2.0 * min((np.sum(bs <= 0) + 1) / 2001.0, (np.sum(bs >= 0) + 1) / 2001.0)))
            delta_rows.append({
                "layer": l_name, "contrast": f"{mb}_minus_{ma}", "n_pts": len(sub),
                "mean_delta_deviance": round(float(diff_i.mean()), 3),
                "ci95_lo": round(float(np.percentile(bs, 2.5)), 3),
                "ci95_hi": round(float(np.percentile(bs, 97.5)), 3),
                "boot_p": f"{p_v:.3e}" if p_v < 1e-3 else round(p_v, 4),
            })

        # Full-precision NB regression p-values (no rounding to 0.0000)
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
                res = smf.negativebinomial("fut_n_onsets ~ " + " + ".join(terms), data=pdf).fit(disp=False, maxiter=250)
                pc = float(res.pvalues.get("z_hist_longest_share", np.nan))
                pr = float(res.pvalues.get("z_log_hist_lam_on", np.nan))
                nb_rows.append({
                    "layer": l_name, "population": pop_lbl, "model": mname, "n_pts": len(pdf),
                    "converged": bool(res.mle_retvals.get("converged", False)),
                    "log_lik": round(float(res.llf), 3), "alpha_mle": round(float(res.params.get("alpha", np.nan)), 3),
                    "coef_cont": round(float(res.params.get("z_hist_longest_share", np.nan)), 4),
                    "p_cont": f"{pc:.3e}" if np.isfinite(pc) else np.nan,
                    "p_rate": f"{pr:.3e}" if np.isfinite(pr) else np.nan,
                })

    pd.DataFrame(oof_rows).to_csv(OUT_RES / "V8_rev2_e05_causal_oof.csv", index=False)
    pd.DataFrame(delta_rows).to_csv(OUT_RES / "V8_rev2_e05_causal_oof_ci.csv", index=False)
    pd.DataFrame(nb_rows).to_csv(OUT_RES / "V8_rev2_e05_causal_nb.csv", index=False)
    print("Non-Peeking E05 OOF Deviance Paired Delta CIs:")
    print(pd.DataFrame(delta_rows).to_string(index=False))
    print("\nExact Scientific p-values in Non-Peeking E05 M5_Full:")
    nd = pd.DataFrame(nb_rows)
    print(nd[nd["model"] == "M5_Full"][["layer", "population", "converged", "coef_cont", "p_cont", "p_rate"]].to_string(index=False))


def run_rev2_p02_same_patient_multiseed(rhy, runs):
    print("\n=== Rev2-Step 2: P02 Same-Patient Intersection + 3-Seed RandomInit_Net1D Control ===")
    p2 = pd.read_parquet(V7P_DIR / "P02_episode_persistence_manifest.parquet")
    f4 = pd.read_parquet(V7P_DIR / "embeddings/P02_f4_rr_features.parquet")
    ef = np.load(V7P_DIR / "embeddings/P02_ecgfounder_10s.npz", allow_pickle=True)["embeddings"]
    keep_rr = f4["is_rr_valid"].astype(bool).values & np.all(np.isfinite(ef), axis=1)
    sub2 = p2[keep_rr].copy().reset_index(drop=True)
    sub2_ef = ef[keep_rr].astype(np.float32)
    rr_cols = ["mean_hr_bpm", "std_hr_bpm", "rmssd_ms", "cvrr", "pnn50", "num_beats"]
    sub2_rr = f4.loc[keep_rr, rr_cols].values.astype(np.float32)

    # Extract 3 independent RandomInit_Net1D seeds on GPU
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    wl = WaveformLoader(cache_max_records=64)
    # Cache raw 10s ECG tensors once per record so 3 seeds run in <10 seconds total!
    rec_tensors = []
    for (ds, rid), grp in sub2.groupby(["dataset", "record_id"]):
        sig_info = wl.load_record_full(ds, str(rid))
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
        with torch.no_grad():
            x500 = gpu_resample_batch(arr, fs_real, 5000, dev)
            x500 = (x500 - x500.mean(dim=-1, keepdim=True)) / x500.std(dim=-1, keepdim=True).clamp(min=1e-5)
            rec_tensors.append((x500.half().cpu(), idxs))
    wl.close()

    rand_seeds = {}
    for seed in [99, 101, 202]:
        adapter_r = get_model_adapter("ECGFOUNDER", "experiments/af_density_prediction_v7p/checkpoints/1_lead_ECGFounder.pth", dev)
        torch.manual_seed(seed)
        for m in adapter_r.model.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        adapter_r.model.eval()
        emb_s = np.zeros((len(sub2), sub2_ef.shape[1]), dtype=np.float32)
        with torch.no_grad():
            for xt_cpu, idxs in rec_tensors:
                xt = xt_cpu.to(dev).float()
                outs = []
                for i in range(0, len(xt), 512):
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        outs.append(adapter_r(xt[i:i + 512]).float().cpu().numpy())
                emb_s[idxs] = np.vstack(outs)
        rand_seeds[seed] = emb_s
    print("Extracted all 3 RandomInit_Net1D seeds on GPU successfully.")

    # Compute R2 & R3 persistence targets
    r3_t = np.zeros(len(sub2), dtype=int)
    r2_t = np.zeros(len(sub2), dtype=int)
    for (ds, rid), grp in sub2.groupby(["dataset", "record_id"]):
        rid_s = str(rid)
        sub_r = rhy[(rhy["dataset"] == ds) & (rhy["record_id"].astype(str) == rid_s) & (rhy["rhythm_known"])]
        sub_af = runs[(runs["dataset"] == ds) & (runs["record_id"].astype(str) == rid_s)]
        af_ivs = merge_intervals(list(zip(sub_af["start_time_s"], sub_af["end_time_s"])))
        br30 = []
        for a, b in af_ivs:
            if not br30: br30.append([a, b])
            elif (a - br30[-1][1]) < 30.0: br30[-1][1] = b
            else: br30.append([a, b])
        vt_sbr60 = []
        if ds in ["LTAFDB", "SHDB-AF"] and len(sub_r):
            vt_sub = sub_r[sub_r["rhythm_label_native"].astype(str).isin(["VT", "IVR", "B", "T"])]
            sbr60_sub = sub_r[(sub_r["rhythm_label_native"].astype(str) == "SBR") & ((sub_r["end_time_s"] - sub_r["start_time_s"]) < 60.0)]
            vt_sbr60 = merge_intervals(list(zip(vt_sub["start_time_s"], vt_sub["end_time_s"])) + list(zip(sbr60_sub["start_time_s"], sbr60_sub["end_time_s"])))
        br_r2 = []
        for a, b in af_ivs:
            if not br_r2: br_r2.append([a, b])
            else:
                rem = subtract_exclusions([(br_r2[-1][1], a)], vt_sbr60)
                if sum(y - x for x, y in rem) <= 1e-3: br_r2[-1][1] = b
                else: br_r2.append([a, b])
        for row in grp.itertuples():
            lm_t = float(row.landmark_time_s)
            for a, b in br30:
                if a - 1.0 <= lm_t <= b + 1.0:
                    r3_t[row.Index] = int((b - a) >= 1800.0); break
            for a, b in br_r2:
                if a - 1.0 <= lm_t <= b + 1.0:
                    r2_t[row.Index] = int((b - a) >= 1800.0); break

    targets = {
        "R0_Native": sub2["target_persists_30m"].astype(int).values,
        "R2_VT_SBR60_Unknown": r2_t,
        "R3_Gap30s": r3_t,
    }
    pts = sub2["patient_id"].values
    # Common patient intersection having both classes under ALL 3 targets!
    common_pts = set(np.unique(pts))
    for yv in targets.values():
        common_pts &= {p for p in np.unique(pts) if len(np.unique(yv[pts == p])) == 2}
    common_pts = sorted(common_pts)
    print(f"Identified N_common = {len(common_pts)} patients eligible simultaneously under R0, R2, and R3!")

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

    out_rows = []
    for y_lbl, y_arr in targets.items():
        for mk in ["logistic", "lgbm"]:
            def cv_oof(emb=None, alone=False):
                oof = np.zeros(len(y_arr))
                for f in sorted(np.unique(folds)):
                    tr, va = folds != f, folds == f
                    if alone:
                        scf = StandardScaler(); pca = PCA(n_components=32, random_state=42)
                        Xtr = pca.fit_transform(scf.fit_transform(emb[tr])); Xva = pca.transform(scf.transform(emb[va]))
                    else:
                        imp = SimpleImputer(strategy="median"); sc = StandardScaler()
                        if emb is None:
                            Xtr = sc.fit_transform(imp.fit_transform(np.hstack([X_h[tr], sub2_rr[tr]])))
                            Xva = sc.transform(imp.transform(np.hstack([X_h[va], sub2_rr[va]])))
                        else:
                            scf = StandardScaler(); pca = PCA(n_components=32, random_state=42)
                            Ztr = pca.fit_transform(scf.fit_transform(emb[tr])); Zva = pca.transform(scf.transform(emb[va]))
                            Xtr = sc.fit_transform(imp.fit_transform(np.hstack([X_h[tr], sub2_rr[tr], Ztr])))
                            Xva = sc.transform(imp.transform(np.hstack([X_h[va], sub2_rr[va], Zva])))
                    clf = LogisticRegression(C=1.0, max_iter=500) if mk == "logistic" else LGBMClassifier(
                        n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=42, n_jobs=1, verbose=-1
                    )
                    oof[va] = clf.fit(Xtr, y_arr[tr]).predict_proba(Xva)[:, 1]
                return oof

            p_hr = cv_oof(None)
            p_ef = cv_oof(sub2_ef)
            p_efa = cv_oof(sub2_ef, alone=True)
            p_rnd_list = [cv_oof(rand_seeds[s]) for s in [99, 101, 202]]

            elig_all = [p for p in np.unique(pts) if len(np.unique(y_arr[pts == p])) == 2]
            for pop_name, pt_list in [("Full_Eligible", elig_all), ("Common_Intersection", common_pts)]:
                a_hr = np.array([roc_auc_score(y_arr[pts == p], p_hr[pts == p]) for p in pt_list])
                a_ef = np.array([roc_auc_score(y_arr[pts == p], p_ef[pts == p]) for p in pt_list])
                a_efa = np.array([roc_auc_score(y_arr[pts == p], p_efa[pts == p]) for p in pt_list])
                # Average per-patient AUC across the 3 independent RandomInit seeds
                a_rnd_mat = np.array([[roc_auc_score(y_arr[pts == p], pr[pts == p]) for p in pt_list] for pr in p_rnd_list])
                a_rnd = a_rnd_mat.mean(axis=0)

                def bci(vals):
                    rng = np.random.RandomState(42)
                    bs = np.array([vals[rng.choice(len(vals), size=len(vals), replace=True)].mean() for _ in range(2000)])
                    p_v = float(min(1.0, 2.0 * min((np.sum(bs <= 0) + 1) / 2001.0, (np.sum(bs >= 0) + 1) / 2001.0)))
                    return round(float(vals.mean()), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4), round(p_v, 4)

                m_pre, lo_pre, hi_pre, p_pre = bci(a_ef - a_rnd)
                m_ecfa, lo_ea, hi_ea, p_ea = bci(a_efa - 0.50)
                out_rows.append({
                    "population": pop_name, "target": y_lbl, "model": mk, "n_pts": len(pt_list),
                    "eq_hist_rr": round(float(a_hr.mean()), 4),
                    "eq_ecgf_alone": round(float(a_efa.mean()), 4), "p_ecgf_alone_gt05": p_ea,
                    "eq_plus_ecgf": round(float(a_ef.mean()), 4),
                    "eq_plus_rand3seed_mean": round(float(a_rnd.mean()), 4),
                    "eq_plus_rand_per_seed": "/".join(f"{v:.4f}" for v in a_rnd_mat.mean(axis=1)),
                    "ecgf_minus_rand3": m_pre, "ci95": f"[{lo_pre:+.4f}, {hi_pre:+.4f}]", "p_ecgf_vs_rand3": p_pre,
                })

    odf = pd.DataFrame(out_rows)
    odf.to_csv(OUT_RES / "V8_rev2_p02_same_patient_multiseed.csv", index=False)
    print(odf[odf["population"] == "Common_Intersection"].to_string(index=False))


def main():
    t0 = time.time()
    inv = pd.read_parquet(OUT_RES / "V8_stage0_sample_inventory.parquet")
    rhy = pd.read_parquet(V5_MAN / "canonical_rhythm_intervals.parquet")
    runs = pd.read_parquet(V5_MAN / "canonical_af_runs.parquet")
    cal = pd.read_csv(IRIDIA_META / "calibration_exclusion_intervals.csv")
    cal_grp = cal.groupby("record_id")
    run_rev2_e05(inv, rhy, runs, cal_grp)
    run_rev2_p02_same_patient_multiseed(rhy, runs)
    print(f"\nCompleted Rev2 in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
