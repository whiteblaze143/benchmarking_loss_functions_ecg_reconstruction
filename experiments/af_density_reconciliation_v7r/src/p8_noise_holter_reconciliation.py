"""Priority 8: Noise Holter Reconciliation, Automated SQI vs Oracle & LTAFDB Dropout Simulation (V7-R)."""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt
from scipy.stats import kurtosis
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from lightgbm import LGBMClassifier

sys.path.insert(0, "experiments/af_density_temporal_phenotypes_v7/src")
from density_continuous import compute_continuous_density

OUT_DIR = Path("experiments/af_density_reconciliation_v7r/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
NH_DIR = Path("/data/mithunmanivannan/datasets/noise_holter_database")


def compute_signal_sqi_features(sig):
    """Causal signal-quality features computed strictly from raw ECG waveform (no oracle expert labels)."""
    x = np.asarray(sig[:, 0], dtype=float)
    x = x - np.median(x)
    std = float(np.std(x))
    kurt = float(kurtosis(x, fisher=True)) if std > 1e-9 else 0.0
    # High-frequency energy ratio (>30 Hz) vs QRS band (5-25 Hz)
    fft_vals = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), d=1.0 / 200.0)
    tot_p = float(np.sum(fft_vals)) + 1e-12
    hf_ratio = float(np.sum(fft_vals[freqs >= 35.0]) / tot_p)
    lf_drift = float(np.sum(fft_vals[freqs < 0.7]) / tot_p)
    max_amp = float(np.percentile(np.abs(x), 99.5))
    return {"sqi_std": std, "sqi_kurtosis": kurt, "sqi_hf_ratio": hf_ratio, "sqi_lf_drift": lf_drift, "sqi_p995": max_amp}


def merge_intervals(ivs):
    if not ivs:
        return []
    s = sorted(ivs)
    m = [list(s[0])]
    for a, b in s[1:]:
        if a <= m[-1][1] + 1e-6:
            m[-1][1] = max(m[-1][1], b)
        else:
            m.append([a, b])
    return [(a, b) for a, b in m]


def main():
    print("=== Priority 8: Noise Holter & LTAFDB Perturbation Reconciliation ===")
    # 1. Reconcile 6255 RECORDS vs 6130 files
    recs_list = (NH_DIR / "RECORDS").read_text().strip().splitlines()
    p4 = pd.read_parquet("experiments/af_density_prediction_v7p/P04_noise_classification_manifest.parquet")
    feat_m = pd.read_csv("experiments/af_density_temporal_phenotypes_v7/results/e04_noise/noise_holter_feature_matrix.csv")

    # Rename beat_coverage -> beat_detection_ratio (detected beats / expected beats at 60 bpm)
    feat_m = feat_m.rename(columns={"beat_coverage": "beat_detection_ratio"})
    print(f"RECORDS count: {len(recs_list)} | Disk .hea files / P04 rows: {len(p4)} | E04 feature rows: {len(feat_m)}")
    print("Mean beat_detection_ratio by quality_class (detected/expected at 60 bpm):")
    print(feat_m.groupby("quality_class")["beat_detection_ratio"].agg(["count", "mean", "median", "max"]).to_string())

    # 2. Sub-rhythm breakdown (AF vs Sinus vs AFL vs Ectopy) in P04 + real RR features
    merged_p4 = pd.merge(p4, feat_m[["record_id", "hr", "cvrr", "rmssd", "sdnn", "pnn50", "beat_detection_ratio"]],
                         left_on="strip_id", right_on="record_id", how="left")
    y_af = merged_p4["af_present"].astype(int).values
    folds = merged_p4["fold"].values
    rr_cols = ["hr", "cvrr", "rmssd", "sdnn", "pnn50", "beat_detection_ratio"]

    # Re-evaluate REAL RR Tabular model on P04 (fixing the all-NaN F4 bug!)
    X_rr = merged_p4[rr_cols].values.astype(float)
    oof_rr_lgb = np.zeros(len(y_af))
    oof_rr_log = np.zeros(len(y_af))
    for f in sorted(np.unique(folds)):
        tr, va = folds != f, folds == f
        imp = SimpleImputer(strategy="median")
        sc = StandardScaler()
        X_tr = sc.fit_transform(imp.fit_transform(X_rr[tr]))
        X_va = sc.transform(imp.transform(X_rr[va]))
        oof_rr_log[va] = LogisticRegression(C=1.0, max_iter=500).fit(X_tr, y_af[tr]).predict_proba(X_va)[:, 1]
        oof_rr_lgb[va] = LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1).fit(X_tr, y_af[tr]).predict_proba(X_va)[:, 1]

    print(f"\nP04 Real RR Tabular AUROC (previously reported as 0.4125 due to all-NaN bug!):")
    print(f"  Logistic(Real RR): {roc_auc_score(y_af, oof_rr_log):.4f} | LightGBM(Real RR): {roc_auc_score(y_af, oof_rr_lgb):.4f}")

    # Load ECGFounder OOF on P04 for direct apples-to-apples comparison & automated SQI!
    emb_ef = np.load("experiments/af_density_prediction_v7p/embeddings/P04_ecgfounder_10s.npz")["embeddings"]
    oof_ef = np.zeros(len(y_af))
    for f in sorted(np.unique(folds)):
        tr, va = folds != f, folds == f
        imp = SimpleImputer(strategy="median")
        sc = StandardScaler()
        X_tr = sc.fit_transform(imp.fit_transform(emb_ef[tr]))
        X_va = sc.transform(imp.transform(emb_ef[va]))
        oof_ef[va] = LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1).fit(X_tr, y_af[tr]).predict_proba(X_va)[:, 1]

    # 3. Extract waveform SQI features across all 6130 strips to compare Oracle vs Automated Deployable SQI!
    print("Extracting automated SQI waveform features across 6,130 strips...")
    sqi_rows = []
    wfdb_dir = NH_DIR / "wfdb_files"
    for sid in merged_p4["strip_id"]:
        try:
            rec = wfdb.rdrecord(str(wfdb_dir / sid))
            sqi_rows.append(compute_signal_sqi_features(rec.p_signal))
        except Exception:
            sqi_rows.append({"sqi_std": np.nan, "sqi_kurtosis": np.nan, "sqi_hf_ratio": np.nan, "sqi_lf_drift": np.nan, "sqi_p995": np.nan})
    sqi_df = pd.DataFrame(sqi_rows)
    X_sqi = np.column_stack([sqi_df.values, merged_p4[["beat_detection_ratio", "cvrr", "hr"]].values])
    y_n2 = (merged_p4["quality_stratum"] == "N2").astype(int).values
    oof_sqi_n2 = np.zeros(len(y_af))
    for f in sorted(np.unique(folds)):
        tr, va = folds != f, folds == f
        imp = SimpleImputer(strategy="median")
        sc = StandardScaler()
        oof_sqi_n2[va] = LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1).fit(
            sc.fit_transform(imp.fit_transform(X_sqi[tr])), y_n2[tr]
        ).predict_proba(sc.transform(imp.transform(X_sqi[va])))[:, 1]

    sqi_auroc = float(roc_auc_score(y_n2, oof_sqi_n2))
    # Abstain top ~20.6% predicted N2 probability (deployable automated SQI policy!)
    thr_auto = float(np.percentile(oof_sqi_n2, 79.4))
    keep_oracle = (merged_p4["quality_stratum"] != "N2").values
    keep_auto = oof_sqi_n2 < thr_auto

    nh_rows = [
        {"comparison": "All_6130_Real_RR_LGBM", "auroc": round(roc_auc_score(y_af, oof_rr_lgb), 4), "coverage_pct": 100.0, "n": len(y_af)},
        {"comparison": "All_6130_ECGFounder_LGBM", "auroc": round(roc_auc_score(y_af, oof_ef), 4), "coverage_pct": 100.0, "n": len(y_af)},
        {"comparison": "Oracle_Keep_Clean_N1_Real_RR", "auroc": round(roc_auc_score(y_af[keep_oracle], oof_rr_lgb[keep_oracle]), 4), "coverage_pct": round(100*keep_oracle.mean(), 2), "n": int(keep_oracle.sum())},
        {"comparison": "Oracle_Keep_Clean_N1_ECGFounder", "auroc": round(roc_auc_score(y_af[keep_oracle], oof_ef[keep_oracle]), 4), "coverage_pct": round(100*keep_oracle.mean(), 2), "n": int(keep_oracle.sum())},
        {"comparison": "Automated_Deployable_SQI_Real_RR", "auroc": round(roc_auc_score(y_af[keep_auto], oof_rr_lgb[keep_auto]), 4), "coverage_pct": round(100*keep_auto.mean(), 2), "n": int(keep_auto.sum())},
        {"comparison": "Automated_Deployable_SQI_ECGFounder", "auroc": round(roc_auc_score(y_af[keep_auto], oof_ef[keep_auto]), 4), "coverage_pct": round(100*keep_auto.mean(), 2), "n": int(keep_auto.sum())},
        {"comparison": "Automated_SQI_Detector_for_N2", "auroc": round(sqi_auroc, 4), "coverage_pct": 100.0, "n": len(y_n2)},
    ]
    # Competing arrhythmia subsets (AF vs AFL, AF vs Ectopy, AF vs Clean NSR)
    for sub_lbl, m_sub in [
        ("AF_vs_AFL_only_ECGFounder", (merged_p4["af_present"] | merged_p4["afl_present"]).values),
        ("AF_vs_Ectopy_only_ECGFounder", (merged_p4["af_present"] | merged_p4["ectopy_present"]).values),
        ("AF_vs_CleanNSR_ECGFounder", (merged_p4["af_present"] | merged_p4["clean_nsr"]).values),
    ]:
        if m_sub.sum() > 20 and len(np.unique(y_af[m_sub])) == 2:
            nh_rows.append({"comparison": sub_lbl, "auroc": round(roc_auc_score(y_af[m_sub], oof_ef[m_sub]), 4), "coverage_pct": 100.0, "n": int(m_sub.sum())})
            nh_rows.append({"comparison": sub_lbl.replace("ECGFounder", "RealRR"), "auroc": round(roc_auc_score(y_af[m_sub], oof_rr_lgb[m_sub]), 4), "coverage_pct": 100.0, "n": int(m_sub.sum())})

    pd.DataFrame(nh_rows).to_csv(OUT_DIR / "V7R_noise_holter_reconciled.csv", index=False)
    print("\nNoise Holter Reconciled Performance:")
    print(pd.DataFrame(nh_rows).to_string(index=False))

    # 4. Run MISSING Perturbation Simulation across ALL 84 LTAFDB recordings!
    runs = pd.read_parquet("experiments/af_density_continuity_v5/manifests/canonical_af_runs.parquet")
    cr = pd.read_csv("experiments/af_density_continuity_v5/manifests/canonical_cohort_inventory_reconciled.csv")
    ltaf_inv = cr[cr["dataset"] == "LTAFDB"]

    sim_rows = []
    for rec_row in ltaf_inv.itertuples():
        rid = str(rec_row.record_id)
        obs_s = float(rec_row.monitoring_duration_h) * 3600.0
        r_sub = runs[(runs["dataset"] == "LTAFDB") & (runs["record_id"].astype(str) == rid)]
        ivs = merge_intervals([(max(0.0, float(a)), min(obs_s, float(b))) for a, b in zip(r_sub["start_time_s"], r_sub["end_time_s"]) if b > a])
        if not ivs:
            continue
        d_true = compute_continuous_density(ivs, obs_s)
        b_true = sum(b - a for a, b in ivs) / obs_s
        if d_true is None:
            continue
        n_brief30 = sum(1 for a, b in ivs if (b - a) < 30.0)

        # S1a: Omit <30s
        ivs_30 = [(a, b) for a, b in ivs if (b - a) >= 30.0]
        d_30 = compute_continuous_density(ivs_30, obs_s) if ivs_30 else None
        b_30 = sum(b - a for a, b in ivs_30) / obs_s

        # S1b: Omit <10s
        ivs_10 = [(a, b) for a, b in ivs if (b - a) >= 10.0]
        d_10 = compute_continuous_density(ivs_10, obs_s) if ivs_10 else None

        # S2: Bridge gaps <60s
        bridged = [list(ivs[0])]
        for a, b in ivs[1:]:
            if (a - bridged[-1][1]) < 60.0:
                bridged[-1][1] = b
            else:
                bridged.append([a, b])
        d_br = compute_continuous_density([(a, b) for a, b in bridged], obs_s)

        sim_rows.append({
            "record_id": rid, "patient_id": rec_row.patient_id,
            "n_runs": len(ivs), "n_brief_30s": n_brief30,
            "true_burden": b_true, "true_density": d_true,
            "s1a_omit30s_dens_err": (d_30 - d_true) if d_30 is not None else np.nan,
            "s1a_omit30s_undef": d_30 is None,
            "s1a_omit30s_burd_err": b_30 - b_true,
            "s1b_omit10s_dens_err": (d_10 - d_true) if d_10 is not None else np.nan,
            "s2_bridge60s_dens_err": (d_br - d_true) if d_br is not None else np.nan,
        })

    sim_df = pd.DataFrame(sim_rows)
    sim_df.to_csv(OUT_DIR / "V7R_ltafdb_dropout_sensitivity.csv", index=False)
    print(f"\nLTAFDB Perturbation Simulation (n={len(sim_df)} density-defined recordings):")
    print(f"  Omit <30s: MAE Density = {sim_df['s1a_omit30s_dens_err'].abs().mean():.5f} "
          f"(max={sim_df['s1a_omit30s_dens_err'].abs().max():.4f}, undefined_after_omit={int(sim_df['s1a_omit30s_undef'].sum())}), "
          f"MAE Burden = {sim_df['s1a_omit30s_burd_err'].abs().mean():.5f}")
    print(f"  Omit <10s: MAE Density = {sim_df['s1b_omit10s_dens_err'].abs().mean():.5f}")
    print(f"  Bridge <60s gaps: MAE Density = {sim_df['s2_bridge60s_dens_err'].abs().mean():.5f} "
          f"(max={sim_df['s2_bridge60s_dens_err'].abs().max():.4f})")


if __name__ == "__main__":
    main()
