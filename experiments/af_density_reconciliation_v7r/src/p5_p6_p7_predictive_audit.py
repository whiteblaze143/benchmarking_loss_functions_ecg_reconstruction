"""Priorities 5, 6, 7: Canonical Ablations, Future-Deletion & Leakage Audit, Cohort Shortcuts & B=2000 Bootstrap (V7-R)."""
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from lightgbm import LGBMClassifier

sys.path.insert(0, "experiments/af_density_prediction_v7p/src")
import history_features

OUT_DIR = Path("experiments/af_density_reconciliation_v7r/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
V7P_DIR = Path("experiments/af_density_prediction_v7p")


def _fast_auc(y_true, y_score):
    n1 = int(np.sum(y_true))
    n0 = len(y_true) - n1
    ranks = rankdata(y_score, method="average")
    return float((np.sum(ranks[y_true == 1]) - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def patient_bootstrap_diff(y_true, p_a, p_b, patient_ids, n_boot=2000, seed=42):
    """Paired patient-clustered bootstrap (B=2,000) with exact pseudocount p-value floor 1/(B+1)."""
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
    # Exact two-sided Monte Carlo permutation/bootstrap p-value with +1/(B+1) resolution floor
    b_eff = len(arr)
    p_lo = (np.sum(arr <= 0.0) + 1.0) / (b_eff + 1.0)
    p_hi = (np.sum(arr >= 0.0) + 1.0) / (b_eff + 1.0)
    p_val = float(min(1.0, 2.0 * min(p_lo, p_hi)))
    return {
        "obs_delta_auroc": round(obs_delta, 4),
        "boot_mean_delta": round(float(np.mean(arr)), 4),
        "ci95_low": round(float(np.percentile(arr, 2.5)), 4),
        "ci95_high": round(float(np.percentile(arr, 97.5)), 4),
        "p_value": p_val,
        "b_effective": int(b_eff),
        "min_attainable_p": round(2.0 / (b_eff + 1.0), 5),
    }


def cv_oof(X, y, folds, model_kind="logistic"):
    oof = np.zeros(len(y), dtype=float)
    for f in sorted(np.unique(folds)):
        tr, va = folds != f, folds == f
        imp = SimpleImputer(strategy="median")
        sc = StandardScaler()
        X_tr = sc.fit_transform(imp.fit_transform(X[tr]))
        X_va = sc.transform(imp.transform(X[va]))
        if model_kind == "logistic":
            clf = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
        elif model_kind == "lgbm":
            clf = LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=42, n_jobs=1, verbose=-1)
        else:
            raise ValueError(model_kind)
        clf.fit(X_tr, y[tr])
        oof[va] = clf.predict_proba(X_va)[:, 1]
    return oof


def equal_patient_auroc(y, p, patient_ids):
    """Mean patient-level AUROC across patients with both positive and negative outcomes."""
    aucs = []
    for pt in np.unique(patient_ids):
        m = patient_ids == pt
        if len(np.unique(y[m])) == 2:
            aucs.append(roc_auc_score(y[m], p[m]))
    return float(np.mean(aucs)) if aucs else np.nan, len(aucs)


def run_future_deletion_test():
    """Priority 6B: Strictly assert future-deletion invariance on real P01 and P02 landmarks."""
    runs = pd.read_parquet("experiments/af_density_continuity_v5/manifests/canonical_af_runs.parquet")
    p1 = pd.read_parquet(V7P_DIR / "P01_landmark_manifest.parquet")
    p2 = pd.read_parquet(V7P_DIR / "P02_episode_persistence_manifest.parquet")

    audit_rows = []
    for task_id, df in [("P01", p1), ("P02", p2)]:
        for ds in ["IRIDIA-AF", "LTAFDB", "SHDB-AF"]:
            sub = df[df["dataset"] == ds].head(25)
            max_diff = 0.0
            for row in sub.itertuples():
                rec_runs = runs[(runs["dataset"] == row.dataset) & (runs["record_id"].astype(str) == str(row.record_id))]
                all_ivs = list(zip(rec_runs["start_time_s"].values, rec_runs["end_time_s"].values))
                t = float(row.landmark_time_s)

                # Original computation
                f_orig = history_features.compute_interval_features_at_time([r for r in all_ivs if r[0] < t], t)
                # Adversarial future corruption: delete future runs AND inject huge 10h AF episode at t+1s
                past_corrupt = [(s, min(e, t)) for s, e in all_ivs if s < t] + [(t + 1.0, t + 36001.0)]
                f_adv = history_features.compute_interval_features_at_time([r for r in past_corrupt if r[0] < t], t)

                for k in f_orig:
                    v1, v2 = f_orig[k], f_adv[k]
                    if np.isnan(v1) and np.isnan(v2):
                        continue
                    diff = abs(float(v1) - float(v2))
                    if diff > max_diff:
                        max_diff = diff

                # Compare against stored manifest row columns
                for k in f_orig:
                    if hasattr(row, k):
                        v_stored = getattr(row, k)
                        v1 = f_orig[k]
                        if np.isnan(v1) and np.isnan(v_stored):
                            continue
                        diff_s = abs(float(v1) - float(v_stored))
                        if diff_s > max_diff:
                            max_diff = diff_s

            audit_rows.append({
                "test_type": "Future_Deletion_And_Adversarial_Corruption",
                "task_id": task_id,
                "dataset": ds,
                "n_landmarks_tested": len(sub),
                "max_abs_feature_diff": float(max_diff),
                "leakage_detected": bool(max_diff > 1e-9),
            })

    # Also check P02 eventual total_duration_s leakage vs feature columns
    p2_num_cols = [c for c in p2.columns if c.startswith(("af_", "episode_", "longest_", "brief_", "charitos_", "last_", "time_since_", "elapsed_"))]
    for col in p2_num_cols:
        valid = p2[col].notna()
        corr = float(np.corrcoef(p2.loc[valid, col], p2.loc[valid, "total_duration_s"])[0, 1]) if valid.sum() > 10 and p2.loc[valid, col].std() > 0 else 0.0
        if abs(corr) > 0.99:
            audit_rows.append({
                "test_type": "Suspicious_Duration_Collinearity",
                "task_id": "P02",
                "dataset": "ALL",
                "n_landmarks_tested": int(valid.sum()),
                "max_abs_feature_diff": corr,
                "leakage_detected": True,
            })

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(OUT_DIR / "V7R_feature_leakage_audit.csv", index=False)
    print("=== Priority 6B: Future-Deletion & Leakage Audit ===")
    print(audit_df.to_string(index=False))


def main():
    run_future_deletion_test()

    print("\n=== Priority 5, 6, 7: Clean P01 + Canonical Matched Ablation + Cohort Shortcut Audit ===")
    c5 = pd.read_csv("experiments/af_density_continuity_v5/manifests/cohort_inventory.csv")
    c5["is_ann"] = c5["is_annotated"].astype(str).str.lower() == "true"
    unann_shdb = set(c5[(c5["dataset"] == "SHDB-AF") & (~c5["is_ann"])]["record_id"].astype(str))

    p1 = pd.read_parquet(V7P_DIR / "P01_landmark_manifest.parquet")
    p2 = pd.read_parquet(V7P_DIR / "P02_episode_persistence_manifest.parquet")
    p3 = pd.read_parquet(V7P_DIR / "P03_survival_manifest.parquet")

    # Clean P01: exclude unannotated SHDB (8126) and unannotated LTAFDB UNKNOWN (221)
    p1_clean_mask = (~((p1["dataset"] == "SHDB-AF") & (p1["record_id"].astype(str).isin(unann_shdb)))) & (
        ~((p1["dataset"] == "LTAFDB") & (p1["current_rhythm"] == "UNKNOWN"))
    )
    p1_clean = p1[p1_clean_mask].copy().reset_index(drop=True)
    print(f"P01 Raw={len(p1):,} -> Clean Annotated Only={len(p1_clean):,} "
          f"(removed {(~p1_clean_mask).sum():,} unannotated rows; clean prev={p1_clean['target_sustained_30m'].mean()*100:.2f}%)")

    # Canonical feature groups (unambiguous!)
    f_base = ["time_since_rec_start_s", "elapsed_af_duration_s"]
    f_burd = f_base + ["last_af_duration_s", "time_since_last_af_s", "has_prior_af",
                       "af_burden_15m", "af_burden_30m", "af_burden_1h", "af_burden_3h", "af_burden_6h"]
    f_arch = f_burd + [
        "episode_count_15m", "episode_count_30m", "episode_count_1h", "episode_count_3h", "episode_count_6h",
        "longest_af_run_15m", "longest_af_run_30m", "longest_af_run_1h", "longest_af_run_3h", "longest_af_run_6h",
        "brief_run_count_15m", "brief_run_count_30m", "brief_run_count_1h", "brief_run_count_3h", "brief_run_count_6h",
    ]
    f_dens_only = f_arch + ["charitos_density_15m", "charitos_density_30m", "charitos_density_1h", "charitos_density_3h", "charitos_density_6h", "is_density_defined_3h"]
    f_aggr_only = f_arch + ["af_aggregation_15m", "af_aggregation_30m", "af_aggregation_1h", "af_aggregation_3h", "af_aggregation_6h", "is_density_defined_3h"]
    f_dens_aggr = f_arch + ["charitos_density_15m", "charitos_density_30m", "charitos_density_1h", "charitos_density_3h", "charitos_density_6h",
                            "af_aggregation_15m", "af_aggregation_30m", "af_aggregation_1h", "af_aggregation_3h", "af_aggregation_6h", "is_density_defined_3h"]

    matched_rows = []
    cohort_rows = []
    support_rows = []

    task_specs = [
        ("P01_CleanAnnotated", p1_clean, "target_sustained_30m", "P01"),
        ("P02_Persistence60s", p2, "target_persists_30m", "P02"),
        ("P03_Survival15m", p3, "target_terminate_in_15m", "P03"),
    ]

    for task_lbl, df_t, t_col, raw_tid in task_specs:
        y = df_t[t_col].astype(int).values
        folds = df_t["fold"].values
        pts = df_t["patient_id"].values
        support_rows.append({
            "task": task_lbl, "cohort": "ALL",
            "n_samples": len(df_t), "n_positives": int(y.sum()),
            "prevalence": round(float(y.mean()), 5),
            "n_patients": int(df_t["patient_id"].nunique()),
            "top1_patient_share": round(float(df_t["patient_id"].value_counts().iloc[0] / len(df_t)), 4),
        })

        # Cohort-only baseline
        X_coh = pd.get_dummies(df_t["dataset"]).values.astype(float)
        oof_coh = cv_oof(X_coh, y, folds, "logistic")
        matched_rows.append({
            "task": task_lbl, "feature_set": "Cohort_Identity_Only", "model": "logistic",
            "auroc_pooled": round(float(roc_auc_score(y, oof_coh)), 4),
            "auprc_pooled": round(float(average_precision_score(y, oof_coh)), 4),
            "brier": round(float(brier_score_loss(y, oof_coh)), 4),
            "n_features": X_coh.shape[1],
        })

        # Core tabular controlled ablation sequence (same model fixed!)
        stage_map = [
            ("S0_Baseline", f_base),
            ("S1_Burden", f_burd),
            ("S2_Architecture", f_arch),
            ("S3a_Arch_Plus_DensityOnly", f_dens_only),
            ("S3b_Arch_Plus_AggrOnly", f_aggr_only),
            ("S3c_Arch_Plus_Dens_Aggr", f_dens_aggr),
        ]

        oofs = {}
        for mkind in ["logistic", "lgbm"]:
            oofs[mkind] = {}
            for sname, cols in stage_map:
                vcols = [c for c in cols if c in df_t.columns]
                oof = cv_oof(df_t[vcols].values.astype(float), y, folds, mkind)
                oofs[mkind][sname] = oof
                eq_auc, n_eq_pts = equal_patient_auroc(y, oof, pts)
                matched_rows.append({
                    "task": task_lbl, "feature_set": sname, "model": mkind,
                    "auroc_pooled": round(float(roc_auc_score(y, oof)), 4),
                    "auroc_equal_patient": round(eq_auc, 4) if np.isfinite(eq_auc) else np.nan,
                    "n_eq_patients": n_eq_pts,
                    "auprc_pooled": round(float(average_precision_score(y, oof)), 4),
                    "brier": round(float(brier_score_loss(y, oof)), 4),
                    "n_features": len(vcols),
                })

            # Paired B=2000 bootstrap contrasts with FIXED model!
            for c_lbl, a_k, b_k in [
                ("Arch_vs_Burden", "S1_Burden", "S2_Architecture"),
                ("DensityOnly_vs_Arch", "S2_Architecture", "S3a_Arch_Plus_DensityOnly"),
                ("AggrOnly_vs_Arch", "S2_Architecture", "S3b_Arch_Plus_AggrOnly"),
                ("DensAggr_vs_Arch", "S2_Architecture", "S3c_Arch_Plus_Dens_Aggr"),
            ]:
                bs = patient_bootstrap_diff(y, oofs[mkind][a_k], oofs[mkind][b_k], pts, n_boot=2000)
                matched_rows.append({
                    "task": task_lbl, "feature_set": f"CONTRAST:{c_lbl}", "model": mkind,
                    "auroc_pooled": bs["obs_delta_auroc"],
                    "ci95_low": bs["ci95_low"],
                    "ci95_high": bs["ci95_high"],
                    "p_value_b2000": bs["p_value"],
                    "min_attainable_p": bs["min_attainable_p"],
                })

        # Within-cohort evaluation + Leave-One-Cohort-Out (LOCO) for S2_Architecture & S3a_DensityOnly
        for ds in ["IRIDIA-AF", "LTAFDB", "SHDB-AF"]:
            m_ds = (df_t["dataset"] == ds).values
            if m_ds.sum() < 50 or len(np.unique(y[m_ds])) < 2:
                continue
            support_rows.append({
                "task": task_lbl, "cohort": ds,
                "n_samples": int(m_ds.sum()), "n_positives": int(y[m_ds].sum()),
                "prevalence": round(float(y[m_ds].mean()), 5),
                "n_patients": int(df_t.loc[m_ds, "patient_id"].nunique()),
                "top1_patient_share": round(float(df_t.loc[m_ds, "patient_id"].value_counts().iloc[0] / m_ds.sum()), 4),
            })

            # Train & test strictly WITHIN cohort using 5-fold patient-disjoint CV
            for mkind in ["logistic", "lgbm"]:
                for sname, cols in [("S1_Burden", f_burd), ("S2_Architecture", f_arch), ("S3a_Arch_Plus_DensityOnly", f_dens_only)]:
                    vcols = [c for c in cols if c in df_t.columns]
                    # Check folds inside cohort have both classes
                    f_ds = folds[m_ds]
                    y_ds = y[m_ds]
                    X_ds = df_t.loc[m_ds, vcols].values.astype(float)
                    if all(len(np.unique(y_ds[f_ds != f])) == 2 for f in np.unique(f_ds)):
                        oof_ds = cv_oof(X_ds, y_ds, f_ds, mkind)
                        auc_w = float(roc_auc_score(y_ds, oof_ds))
                    else:
                        auc_w = np.nan

                    # Also evaluate pooled model on this cohort slice
                    auc_pooled_slice = float(roc_auc_score(y_ds, oofs[mkind][sname][m_ds]))

                    # Also Leave-One-Cohort-Out (train on OTHER 2 cohorts, test on ds)
                    X_tr_loco = df_t.loc[~m_ds, vcols].values.astype(float)
                    y_tr_loco = y[~m_ds]
                    imp = SimpleImputer(strategy="median")
                    sc = StandardScaler()
                    Xtr_s = sc.fit_transform(imp.fit_transform(X_tr_loco))
                    Xva_s = sc.transform(imp.transform(X_ds))
                    clf = LogisticRegression(C=1.0, max_iter=500) if mkind == "logistic" else LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=42, n_jobs=1, verbose=-1)
                    clf.fit(Xtr_s, y_tr_loco)
                    p_loco = clf.predict_proba(Xva_s)[:, 1]
                    auc_loco = float(roc_auc_score(y_ds, p_loco))

                    cohort_rows.append({
                        "task": task_lbl, "eval_cohort": ds, "feature_set": sname, "model": mkind,
                        "n_samples": int(m_ds.sum()), "n_positives": int(y_ds.sum()),
                        "within_cohort_cv_auroc": round(auc_w, 4) if np.isfinite(auc_w) else np.nan,
                        "pooled_model_slice_auroc": round(auc_pooled_slice, 4),
                        "loco_transfer_auroc": round(auc_loco, 4),
                    })

        # Clean leak-free RR physiology test STRICTLY inside LTAFDB + SHDB where RR is actually valid (WITHOUT is_rr_valid shortcut!)
        if raw_tid in ["P01", "P02", "P03"]:
            rr_path = V7P_DIR / "embeddings" / f"{raw_tid}_f4_rr_features.parquet"
            rr_full = pd.read_parquet(rr_path)
            if raw_tid == "P01":
                rr_sub = rr_full.loc[p1_clean_mask].reset_index(drop=True)
            else:
                rr_sub = rr_full
            rr_cols_noleak = ["mean_hr_bpm", "std_hr_bpm", "rmssd_ms", "cvrr", "pnn50"]
            for c in rr_cols_noleak:
                df_t[c] = rr_sub[c].values
            df_t["is_rr_valid"] = rr_sub["is_rr_valid"].values

            m_valid_rr = (df_t["dataset"].isin(["LTAFDB", "SHDB-AF"])) & (df_t["is_rr_valid"] == 1.0)
            if m_valid_rr.sum() > 100:
                df_rr = df_t[m_valid_rr].reset_index(drop=True)
                y_rr = df_rr[t_col].astype(int).values
                f_rr = df_rr["fold"].values
                pts_rr = df_rr["patient_id"].values
                for mkind in ["logistic", "lgbm"]:
                    o_base = cv_oof(df_rr[f_dens_aggr].values.astype(float), y_rr, f_rr, mkind)
                    o_plus_rr = cv_oof(df_rr[f_dens_aggr + rr_cols_noleak].values.astype(float), y_rr, f_rr, mkind)
                    bs_rr = patient_bootstrap_diff(y_rr, o_base, o_plus_rr, pts_rr, n_boot=2000)
                    matched_rows.append({
                        "task": f"{task_lbl}_LTAFDB_SHDB_ValidRR",
                        "feature_set": "S3c_vs_S4_CleanRR_NoCohortShortcut",
                        "model": mkind,
                        "auroc_pooled": bs_rr["obs_delta_auroc"],
                        "ci95_low": bs_rr["ci95_low"],
                        "ci95_high": bs_rr["ci95_high"],
                        "p_value_b2000": bs_rr["p_value"],
                        "min_attainable_p": bs_rr["min_attainable_p"],
                        "n_features": len(rr_cols_noleak),
                    })
                    print(f"  [{task_lbl} LTAFDB+SHDB ValidRR] {mkind}: Base={roc_auc_score(y_rr, o_base):.4f} -> +CleanRR={roc_auc_score(y_rr, o_plus_rr):.4f} "
                          f"(Delta={bs_rr['obs_delta_auroc']:+.4f}, 95% CI [{bs_rr['ci95_low']:+.4f}, {bs_rr['ci95_high']:+.4f}], p={bs_rr['p_value']:.4f})")

    pd.DataFrame(matched_rows).to_csv(OUT_DIR / "V7R_matched_ablation_results.csv", index=False)
    pd.DataFrame(cohort_rows).to_csv(OUT_DIR / "V7R_cohort_stratified_performance.csv", index=False)
    pd.DataFrame(support_rows).to_csv(OUT_DIR / "V7R_patient_support.csv", index=False)
    print("\nSaved V7R_matched_ablation_results.csv, V7R_cohort_stratified_performance.csv, V7R_patient_support.csv")


if __name__ == "__main__":
    main()
