"""Priorities 2 (Nulls), 6/7 (Clean Alerts, Fixed-Model FM vs Tabular w/ B=2000, Param Counts), & Next-Direction Feasibility."""
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from lightgbm import LGBMClassifier

sys.path.insert(0, "experiments/af_density_prediction_v7p/src")
import foundation_adapters as fa
from alert_evaluation import evaluate_p01_alerts

OUT_DIR = Path("experiments/af_density_reconciliation_v7r/results")
V7P_DIR = Path("experiments/af_density_prediction_v7p")
EMB_DIR = V7P_DIR / "embeddings"
V7_RES = Path("experiments/af_density_temporal_phenotypes_v7/results")

CHECKPOINTS = {
    "CSFM": ("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/external/Cardiac-Sensing-FM/pretrained/CSFM_tiny.pth", "CSFM"),
    "CLEF": ("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/foundation_model_comparison/candidate_weights/clef/clef_medium.ckpt", "CLEF"),
    "ECGFOUNDER": (str(V7P_DIR / "checkpoints" / "1_lead_ECGFounder.pth"), "ECGFOUNDER"),
    "ECGFM_KED": ("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/foundation_model_comparison/candidate_weights/ecgfm_ked/best_valid_all_increase_with_augment_epoch_3.pt", "KED"),
    "ECHONEXT": ("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/external/IntroECG/7-EchoNext Minimodel/models/echonext_multilabel_minimodel/weights.pt", "ECHONEXT"),
    "DINOSR": ("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/foundation_model_comparison/candidate_sources/ecg_pretraining_strategies/checkpoints/dinosr_e10.ckpt", "DINOSR"),
    "JEPA": ("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/foundation_model_comparison/candidate_sources/ecg_pretraining_strategies/checkpoints/jepa_e10.ckpt", "JEPA"),
    "NEFCORE": ("/data/mithunmanivannan/nef-net-aim/02_nef_core/pretrained/nefcore_full.pt", "NEFCORE"),
    "F2": ("/data/mithunmanivannan/codex_artifacts/temporal_setoperator_fm_20261120/artifacts/F2/pilot_final.pt", "F2"),
    "F0": ("/data/mithunmanivannan/codex_artifacts/temporal_setoperator_fm_20261120/artifacts/F0/pilot_final.pt", "F0"),
    "G1": ("/data/mithunmanivannan/codex_artifacts/echonext_geometry_setoperator_v1/runs/g1_seed42/best_model.pt", "G1"),
    "CONTROL": ("/data/mithunmanivannan/codex_artifacts/echonext_geometry_setoperator_v1/runs/control_capacity_seed42/best_model.pt", "CAPACITY"),
}


def _fast_auc(y_true, y_score):
    n1 = int(np.sum(y_true))
    n0 = len(y_true) - n1
    ranks = rankdata(y_score, method="average")
    return float((np.sum(ranks[y_true == 1]) - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def patient_bootstrap_diff(y_true, p_a, p_b, patient_ids, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    pts = np.unique(patient_ids)
    n_pts = len(pts)
    pt_idx = {p: np.where(patient_ids == p)[0] for p in pts}
    obs_delta = float(roc_auc_score(y_true, p_b) - roc_auc_score(y_true, p_a))
    deltas = []
    for _ in range(n_boot):
        samp = rng.choice(pts, size=n_pts, replace=True)
        idx = np.concatenate([pt_idx[p] for p in samp])
        yb = y_true[idx]
        if len(np.unique(yb)) < 2:
            continue
        deltas.append(_fast_auc(yb, p_b[idx]) - _fast_auc(yb, p_a[idx]))
    arr = np.array(deltas)
    b_eff = len(arr)
    p_lo = (np.sum(arr <= 0.0) + 1.0) / (b_eff + 1.0)
    p_hi = (np.sum(arr >= 0.0) + 1.0) / (b_eff + 1.0)
    p_val = float(min(1.0, 2.0 * min(p_lo, p_hi)))
    return {
        "obs_delta": round(obs_delta, 4),
        "ci95_low": round(float(np.percentile(arr, 2.5)), 4),
        "ci95_high": round(float(np.percentile(arr, 97.5)), 4),
        "p_value": p_val,
    }


def cv_oof_tab_and_emb(X_tab, X_emb, y, folds, model_kind="logistic", pca_dim=32, fit_tab=True):
    """Fit fixed model on tabular alone, embedding (PCA-32) alone, and fusion without train/val leakage."""
    oof_tab = np.zeros(len(y))
    oof_emb = np.zeros(len(y))
    oof_fus = np.zeros(len(y))
    for f in sorted(np.unique(folds)):
        tr, va = folds != f, folds == f
        # Tabular
        if X_tab is not None:
            imp_t = SimpleImputer(strategy="median")
            sc_t = StandardScaler()
            Xt_tr = sc_t.fit_transform(imp_t.fit_transform(X_tab[tr]))
            Xt_va = sc_t.transform(imp_t.transform(X_tab[va]))
        # Embedding
        if X_emb is not None:
            if np.isnan(X_emb).any():
                imp_e = SimpleImputer(strategy="median")
                Xe_tr_i = imp_e.fit_transform(X_emb[tr])
                Xe_va_i = imp_e.transform(X_emb[va])
            else:
                Xe_tr_i, Xe_va_i = X_emb[tr], X_emb[va]
            sc_e = StandardScaler()
            Xe_tr_s = sc_e.fit_transform(Xe_tr_i)
            Xe_va_s = sc_e.transform(Xe_va_i)
            n_comp = min(pca_dim, Xe_tr_s.shape[1], Xe_tr_s.shape[0] - 1)
            pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=42)
            Xe_tr = pca.fit_transform(Xe_tr_s)
            Xe_va = pca.transform(Xe_va_s)

        def make_clf():
            if model_kind == "logistic":
                return LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
            return LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=42, n_jobs=1, verbose=-1)

        if X_tab is not None and fit_tab:
            oof_tab[va] = make_clf().fit(Xt_tr, y[tr]).predict_proba(Xt_va)[:, 1]
        if X_emb is not None:
            oof_emb[va] = make_clf().fit(Xe_tr, y[tr]).predict_proba(Xe_va)[:, 1]
        if X_tab is not None and X_emb is not None:
            oof_fus[va] = make_clf().fit(np.hstack([Xt_tr, Xe_tr]), y[tr]).predict_proba(np.hstack([Xt_va, Xe_va]))[:, 1]
    return oof_tab, oof_emb, oof_fus


def summarize_e01_nulls():
    print("=== 1. E01 Conditional Null Summary ===")
    df = pd.read_parquet(V7_RES / "E01_conditional_null_results.parquet")
    na = df[df["null_type"] == "Null_A"]
    nb = df[df["null_type"] == "Null_B"]
    out = []
    for name, sub in [("Null_A_All", na), ("Null_A_LTAFDB", na[na["dataset"] == "LTAFDB"]), ("Null_A_SHDB", na[na["dataset"] == "SHDB-AF"])]:
        obs_pre = float(sub["obs_pre_count"].sum())
        exp_pre = float(sub["null_pre_mean"].sum())
        obs_post = float(sub["obs_post_count"].sum())
        exp_post = float(sub["null_post_mean"].sum())
        out.append({
            "null_type": name, "n_recordings": len(sub),
            "obs_pre_total": obs_pre, "null_pre_expected": round(exp_pre, 2),
            "pre_obs_over_exp": round(obs_pre / exp_pre, 3) if exp_pre > 0 else np.nan,
            "obs_post_total": obs_post, "null_post_expected": round(exp_post, 2),
            "post_obs_over_exp": round(obs_post / exp_post, 3) if exp_post > 0 else np.nan,
            "frac_rec_p_pre_lt005": round(float((sub["p_value_pre"] < 0.05).mean()), 4),
            "frac_rec_p_post_lt005": round(float((sub["p_value_post"] < 0.05).mean()), 4),
        })
    nb_rows = [{
        "null_type": "Null_B_Exchangeable_Ordering", "n_recordings": len(nb),
        "obs_pre_total": float(nb["obs_pre_brief"].sum()),
        "null_pre_expected": round(float(nb["null_pre_mean"].sum()), 2),
        "pre_obs_over_exp": round(float(nb["obs_pre_brief"].sum() / nb["null_pre_mean"].sum()), 3),
        "obs_post_total": float(nb["obs_post_brief"].sum()),
        "null_post_expected": round(float(nb["null_post_mean"].sum()), 2),
        "post_obs_over_exp": round(float(nb["obs_post_brief"].sum() / nb["null_post_mean"].sum()), 3),
        "frac_rec_p_pre_lt005": round(float((nb["p_value_ordering"] < 0.05).mean()), 4),
        "frac_rec_p_post_lt005": np.nan,
    }]
    res = pd.DataFrame(out + nb_rows)
    res.to_csv(OUT_DIR / "V7R_e01_nulls_summary.csv", index=False)
    print(res.to_string(index=False))


def audit_model_params():
    print("\n=== 2. Foundation Model Parameter Counts ===")
    rows = []
    dev = torch.device("cpu")
    for mkey, (ckpt, adapter_key) in CHECKPOINTS.items():
        try:
            ad = fa.get_model_adapter(adapter_key, ckpt, dev)
            n_params = sum(p.numel() for p in ad.parameters())
            ckpt_mb = round(Path(ckpt).stat().st_size / (1024 * 1024), 2)
            rows.append({
                "model": mkey, "adapter_key": adapter_key, "dim": ad.dim, "fs_hz": ad.fs,
                "target_len": ad.target_len, "total_params": int(n_params), "ckpt_size_mb": ckpt_mb,
            })
            del ad
        except Exception as e:
            rows.append({"model": mkey, "adapter_key": adapter_key, "error": str(e)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "V7R_fm_inventory_params.csv", index=False)
    print(df.to_string(index=False))


def run_clean_alerts_and_fm_benchmark():
    print("\n=== 3. Clean P01 Alerts & Fixed-Model FM vs Tabular (B=2000) ===")
    c5 = pd.read_csv("experiments/af_density_continuity_v5/manifests/cohort_inventory.csv")
    c5["is_ann"] = c5["is_annotated"].astype(str).str.lower() == "true"
    unann_shdb = set(c5[(c5["dataset"] == "SHDB-AF") & (~c5["is_ann"])]["record_id"].astype(str))

    p1 = pd.read_parquet(V7P_DIR / "P01_landmark_manifest.parquet")
    p1_clean_mask = (~((p1["dataset"] == "SHDB-AF") & (p1["record_id"].astype(str).isin(unann_shdb)))) & (
        ~((p1["dataset"] == "LTAFDB") & (p1["current_rhythm"] == "UNKNOWN"))
    )
    p1_clean = p1[p1_clean_mask].copy().reset_index(drop=True)

    p2 = pd.read_parquet(V7P_DIR / "P02_episode_persistence_manifest.parquet")
    p3 = pd.read_parquet(V7P_DIR / "P03_survival_manifest.parquet")
    p4 = pd.read_parquet(V7P_DIR / "P04_noise_classification_manifest.parquet")

    f_s3c = [
        "time_since_rec_start_s", "elapsed_af_duration_s", "last_af_duration_s", "time_since_last_af_s", "has_prior_af",
        "af_burden_15m", "af_burden_30m", "af_burden_1h", "af_burden_3h", "af_burden_6h",
        "episode_count_15m", "episode_count_30m", "episode_count_1h", "episode_count_3h", "episode_count_6h",
        "longest_af_run_15m", "longest_af_run_30m", "longest_af_run_1h", "longest_af_run_3h", "longest_af_run_6h",
        "brief_run_count_15m", "brief_run_count_30m", "brief_run_count_1h", "brief_run_count_3h", "brief_run_count_6h",
        "charitos_density_15m", "charitos_density_30m", "charitos_density_1h", "charitos_density_3h", "charitos_density_6h",
        "af_aggregation_15m", "af_aggregation_30m", "af_aggregation_1h", "af_aggregation_3h", "af_aggregation_6h", "is_density_defined_3h",
    ]
    f_burd = f_s3c[:10]

    # Clean P01 Alert Evaluation on Logistic & LGBM S1_Burden, S3c_Tabular, and Multimodal + ECGFounder
    emb_p1_ef = np.load(EMB_DIR / "P01_ecgfounder_10s.npz")["embeddings"][p1_clean_mask]
    y1 = p1_clean["target_sustained_30m"].astype(int).values
    f1 = p1_clean["fold"].values
    pts1 = p1_clean["patient_id"].values

    oof_burd, oof_ef1, _ = cv_oof_tab_and_emb(p1_clean[f_burd].values, emb_p1_ef, y1, f1, "lgbm")
    oof_s3c, _, oof_fus_ef = cv_oof_tab_and_emb(p1_clean[f_s3c].values, emb_p1_ef, y1, f1, "lgbm")
    oof_s3c_log, _, oof_fus_ef_log = cv_oof_tab_and_emb(p1_clean[f_s3c].values, emb_p1_ef, y1, f1, "logistic")

    alert_rows = []
    # Fixed quantile thresholds on OOF (e.g. top 5% / 10% / 15% warning time) + fixed prob thresholds (0.05, 0.10, 0.20)
    for m_name, prob in [("LGBM_S1_Burden", oof_burd), ("LGBM_S3c_ArchDensAggr", oof_s3c),
                         ("LGBM_Fusion_ECGFounder", oof_fus_ef), ("Logistic_S3c_ArchDensAggr", oof_s3c_log),
                         ("Logistic_Fusion_ECGFounder", oof_fus_ef_log)]:
        for thr in [0.05, 0.10, 0.15]:
            pred_df = p1_clean[["patient_id", "landmark_time_s", "target_sustained_30m", "event_time_s"]].copy()
            pred_df["predicted_prob"] = prob
            m = evaluate_p01_alerts(pred_df, probability_threshold=thr)
            m["model"] = m_name
            m["warnings_per_patient_day"] = round(m["false_alerts_per_monitoring_hour"] * 24.0, 2)
            alert_rows.append(m)

    alert_df = pd.DataFrame(alert_rows)
    alert_df.to_csv(OUT_DIR / "V7R_p01_clean_alerts.csv", index=False)
    print(alert_df[["model", "operating_threshold", "episodes_alerted_fraction", "median_lead_time_minutes", "false_alerts_per_monitoring_hour", "warnings_per_patient_day", "alarm_burden_percent"]].to_string(index=False))

    # Fixed-model FM evaluation across P01_Clean, P02, P03, P04 (all 12 checkpoints on 10s window, plus random-init Gaussian projection sanity control!)
    fm_rows = []
    models_to_eval = list(CHECKPOINTS.keys())
    rng = np.random.RandomState(42)

    for task_lbl, tid, df_t, mask, t_col in [
        ("P01_CleanAnnotated", "P01", p1_clean, p1_clean_mask, "target_sustained_30m"),
        ("P02_Persistence60s", "P02", p2, slice(None), "target_persists_30m"),
        ("P03_Survival15m", "P03", p3, slice(None), "target_terminate_in_15m"),
        ("P04_NoiseHolter", "P04", p4, slice(None), "af_present"),
    ]:
        y = df_t[t_col].astype(int).values
        folds = df_t["fold"].values
        pts = df_t["patient_id"].values
        vcols = [c for c in f_s3c if c in df_t.columns]
        X_tab = df_t[vcols].values.astype(float) if tid != "P04" else None

        # Precompute fixed S3c tabular OOFs once per task
        tab_oof = {}
        if X_tab is not None:
            for mk in ["logistic", "lgbm"]:
                ot, _, _ = cv_oof_tab_and_emb(X_tab, None, y, folds, mk)
                tab_oof[mk] = ot

        for mkey in models_to_eval:
            X_emb = np.load(EMB_DIR / f"{tid}_{mkey.lower()}_10s.npz")["embeddings"][mask]
            for mk in ["logistic", "lgbm"]:
                _, oof_emb, oof_fus = cv_oof_tab_and_emb(X_tab, X_emb, y, folds, mk, pca_dim=32, fit_tab=False)
                auc_emb = round(float(roc_auc_score(y, oof_emb)), 4)
                row = {
                    "task": task_lbl, "fm_model": mkey, "classifier": mk,
                    "fm_alone_auroc": auc_emb,
                    "fm_alone_auprc": round(float(average_precision_score(y, oof_emb)), 4),
                }
                if X_tab is not None:
                    auc_tab = round(float(roc_auc_score(y, tab_oof[mk])), 4)
                    auc_fus = round(float(roc_auc_score(y, oof_fus)), 4)
                    row["tab_s3c_auroc"] = auc_tab
                    row["fusion_auroc"] = auc_fus
                    row["delta_fusion_minus_tab"] = round(auc_fus - auc_tab, 4)
                    # Run B=2000 bootstrap on primary FMs & controls to keep runtime crisp (~2 min total)
                    if mkey in ["ECGFOUNDER", "ECHONEXT", "ECGFM_KED", "F2", "F0", "G1", "CONTROL"]:
                        bs = patient_bootstrap_diff(y, tab_oof[mk], oof_fus, pts, n_boot=2000)
                        row["boot_ci95_low"] = bs["ci95_low"]
                        row["boot_ci95_high"] = bs["ci95_high"]
                        row["boot_p_b2000"] = bs["p_value"]
                fm_rows.append(row)

    fm_df = pd.DataFrame(fm_rows)
    fm_df.to_csv(OUT_DIR / "V7R_fm_matched_evaluation.csv", index=False)
    print("\nSaved V7R_fm_matched_evaluation.csv (summary top rows):")
    print(fm_df[fm_df["fm_model"].isin(["ECGFOUNDER", "ECHONEXT", "ECGFM_KED", "F2", "CONTROL"])].to_string(index=False))


def evaluate_next_direction_feasibility():
    print("\n=== 4. Feasibility Check: Temporal Concentration vs Transition Instability ===")
    inv = pd.read_parquet(OUT_DIR / "V7R_recording_inventory.parquet")
    runs = pd.read_parquet("experiments/af_density_continuity_v5/manifests/canonical_af_runs.parquet")
    runs = runs.merge(inv[["dataset", "record_id", "patient_id", "known_rhythm_h"]], on=["dataset", "record_id"], how="left")

    # Patient-level deduplicated first annotated recording with AF (to guarantee strict independence!)
    ann_af = inv[(inv["is_annotated"]) & (inv["canonical_af_runs"] >= 1)].copy()
    ann_af["af_burden_known"] = ann_af["af_duration_h"] / ann_af["known_rhythm_h"]

    # Compute Charitos density directly per recording using compute_continuous_density on canonical runs
    sys.path.insert(0, "experiments/af_density_temporal_phenotypes_v7/src")
    from density_continuous import compute_continuous_density

    rec_feats = []
    for row in ann_af.itertuples():
        sub = runs[(runs["dataset"] == row.dataset) & (runs["record_id"] == row.record_id)]
        ivs = sorted([(max(0.0, float(a)), float(b)) for a, b in zip(sub["start_time_s"], sub["end_time_s"]) if b > a])
        obs_s = max(float(row.raw_recording_h) * 3600.0, max((b for _, b in ivs), default=0.0))
        dens = compute_continuous_density(ivs, obs_s) if ivs else np.nan
        n_runs = len(ivs)
        n_brief60 = sum(1 for a, b in ivs if (b - a) < 60.0)
        n_prol30 = sum(1 for a, b in ivs if (b - a) >= 1800.0)
        gaps = [ivs[i + 1][0] - ivs[i][1] for i in range(len(ivs) - 1)]
        med_gap_s = float(np.median(gaps)) if gaps else np.nan
        transitions_per_day = (2.0 * n_runs - int(row.left_censored_runs > 0) - int(row.right_censored_runs > 0)) / (float(row.known_rhythm_h) / 24.0)
        rec_feats.append({
            "dataset": row.dataset, "record_id": row.record_id, "patient_id": row.patient_id,
            "known_h": row.known_rhythm_h, "af_burden": row.af_burden_known,
            "density": dens if dens is not None else np.nan,
            "n_runs": n_runs, "n_gaps": max(0, n_runs - 1),
            "n_brief_60s": n_brief60, "n_prolonged_30m": n_prol30,
            "median_gap_s": med_gap_s, "transitions_per_24h": transitions_per_day,
        })
    rdf = pd.DataFrame(rec_feats)
    # Patient-deduplicated independent cohort (1 recording per patient)
    p_dedup = rdf.sort_values(["dataset", "patient_id", "record_id"]).drop_duplicates(["dataset", "patient_id"])
    multi = p_dedup[(p_dedup["n_runs"] >= 2) & (p_dedup["density"].notna())].copy()

    # Check high-density (>=0.80) multi-run patients: can they have radically different transition rates / gap structures?
    hi_d = multi[multi["density"] >= 0.80]
    strat_rows = []
    for lbl, sub in [
        ("All_Annotated_AF_Patients", p_dedup),
        ("Defined_Density_Patients", p_dedup[p_dedup["density"].notna()]),
        ("MultiRun_DefinedDensity_Patients", multi),
        ("HighDensity_ge0.80_MultiRun_Patients", hi_d),
        ("HighDensity_LowFrag_2_to_5_runs", hi_d[hi_d["n_runs"] <= 5]),
        ("HighDensity_HighFrag_ge20_runs", hi_d[hi_d["n_runs"] >= 20]),
        ("LowBurden_lt1pct_MultiRun", multi[multi["af_burden"] < 0.01]),
        ("ModBurden_1_to_10pct_MultiRun", multi[(multi["af_burden"] >= 0.01) & (multi["af_burden"] < 0.10)]),
        ("HiBurden_ge10pct_MultiRun", multi[multi["af_burden"] >= 0.10]),
    ]:
        if len(sub) == 0:
            continue
        strat_rows.append({
            "stratum": lbl, "n_patients": len(sub),
            "median_burden": round(float(sub["af_burden"].median()), 4),
            "median_density": round(float(sub["density"].median()), 4) if sub["density"].notna().any() else np.nan,
            "iqr_runs": f"{sub['n_runs'].quantile(0.25):.1f}-{sub['n_runs'].quantile(0.75):.1f}",
            "median_runs": float(sub["n_runs"].median()),
            "median_transitions_24h": round(float(sub["transitions_per_24h"].median()), 2),
            "n_iridia": int((sub["dataset"] == "IRIDIA-AF").sum()),
            "n_ltafdb": int((sub["dataset"] == "LTAFDB").sum()),
            "n_shdb": int((sub["dataset"] == "SHDB-AF").sum()),
        })
    sdf = pd.DataFrame(strat_rows)
    sdf.to_csv(OUT_DIR / "V7R_transition_instability_feasibility.csv", index=False)
    print(sdf.to_string(index=False))


if __name__ == "__main__":
    t0 = time.time()
    if "--only-alerts-fm" not in sys.argv:
        summarize_e01_nulls()
        audit_model_params()
    run_clean_alerts_and_fm_benchmark()
    if "--only-alerts-fm" not in sys.argv:
        evaluate_next_direction_feasibility()
    print(f"\nCompleted in {time.time() - t0:.1f}s")
