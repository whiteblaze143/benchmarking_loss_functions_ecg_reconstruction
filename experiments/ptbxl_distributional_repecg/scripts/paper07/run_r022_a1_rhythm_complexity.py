#!/usr/bin/env python3
"""
Stage R022-A1: Complexity Justification Benchmark.
Compares trivial 3-D baseline h_RR = [mean(log RR), SD(log RR), SD(Δlog RR)]
against 32-D standardized Nyström RKHS mean embedding r_RR on PTB-XL AF/SR diagnostic task.
Includes:
- Paired patient-clustered bootstrap difference Δ = AUROC(r_RR) - AUROC(h_RR) (95% CI)
- Validation-locked operating points (FPR_tau*, Sensitivity_tau*)
- Comprehensive R-peak detector quality audit for AF vs SR
"""

import sys
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
import wfdb

# Path setup
sys.path.insert(0, str(Path("experiments/ptbxl_distributional_repecg/src").resolve()))
from repecg.common.beats import detect_rpeaks
from repecg.paper07_operator.rhythm import (
    extract_rr_atoms,
    extract_simple_rr_features,
    audit_rpeak_quality,
    RhythmResponseMap,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("r022_a1_complexity")

PTBXL_DIR = Path("data/ptb_xl")
RHYTHM_FIT_PATH = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/paper07_operator/development_representations/rhythm_fit.npz")
OUTPUT_PATH = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/prism_r021_real_waveforms/r022_a1_complexity_results.json")


def extract_lead_ii_features(rec_path: Path, rmap: RhythmResponseMap, fs: int = 500):
    try:
        rec = wfdb.rdrecord(str(rec_path))
        if "II" not in rec.sig_name:
            return None, None, None
        idx = rec.sig_name.index("II")
        sig = rec.p_signal[:, idx]
        sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
        
        basis = np.zeros((len(sig), 8), dtype=np.float64)
        basis[:, 1] = sig
        detection = detect_rpeaks(basis, fs=fs, primary_lead="II")
        peaks = detection.rpeaks

        # 1. Detector quality audit
        audit = audit_rpeak_quality(peaks, fs=fs, min_rr_s=0.25, max_rr_s=2.50, min_intervals=5)

        # 2. Trivial 3-D baseline
        h_rr = extract_simple_rr_features(peaks, fs=fs)

        # 3. 32-D RKHS mean embedding
        atoms = extract_rr_atoms(peaks, fs=fs)
        r_rr = rmap.mean_embedding(atoms, min_intervals=5)

        return h_rr, r_rr, audit
    except Exception:
        return None, None, None


def bootstrap_paired_comparison(y_true, y_prob_h, y_prob_r, pids, n_bootstraps=1000, seed=42):
    rng = np.random.default_rng(seed)
    unique_pids = np.unique(pids)
    
    auroc_h_list, auroc_r_list, delta_list = [], [], []
    auprc_h_list, auprc_r_list = [], []

    for _ in range(n_bootstraps):
        boot_pids = rng.choice(unique_pids, size=len(unique_pids), replace=True)
        idx_list = []
        for p in boot_pids:
            idx_list.extend(np.where(pids == p)[0])
        idx = np.array(idx_list)

        y_b = y_true[idx]
        if len(np.unique(y_b)) < 2:
            continue

        ah = roc_auc_score(y_b, y_prob_h[idx])
        ar = roc_auc_score(y_b, y_prob_r[idx])
        ph = average_precision_score(y_b, y_prob_h[idx])
        pr = average_precision_score(y_b, y_prob_r[idx])

        auroc_h_list.append(ah)
        auroc_r_list.append(ar)
        delta_list.append(ar - ah)
        auprc_h_list.append(ph)
        auprc_r_list.append(pr)

    def stats(arr):
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "ci_lower": float(np.percentile(arr, 2.5)),
            "ci_upper": float(np.percentile(arr, 97.5)),
        }

    return {
        "auroc_simple": stats(auroc_h_list),
        "auroc_rkhs": stats(auroc_r_list),
        "delta_auroc": stats(delta_list),
        "auprc_simple": stats(auprc_h_list),
        "auprc_rkhs": stats(auprc_r_list),
        "p_value_rkhs_beats_simple": float(np.mean(np.array(delta_list) <= 0)),
    }


def find_operating_threshold(y_val, prob_val, target_sens=0.90):
    sorted_probs = np.sort(prob_val[y_val == 1])
    target_idx = int(np.floor((1.0 - target_sens) * len(sorted_probs)))
    target_idx = max(0, min(target_idx, len(sorted_probs) - 1))
    return float(sorted_probs[target_idx])


def main():
    logger.info("=== Starting Stage R022-A1: Rhythm Complexity Justification Benchmark ===")
    assert RHYTHM_FIT_PATH.exists(), f"Canonical rhythm fit not found: {RHYTHM_FIT_PATH}"
    rmap = RhythmResponseMap.load(RHYTHM_FIT_PATH)
    logger.info(f"Loaded canonical RhythmResponseMap: dim={rmap.dim}, gamma={rmap.gamma:.4f}")

    meta = pd.read_csv(PTBXL_DIR / "ptbxl_database.csv")
    
    # Label definition: AF vs SR
    # AFIB: scp_codes contains AFIB
    # SR: scp_codes contains SR and not AFIB
    af_mask = meta["scp_codes"].str.contains("AFIB", na=False)
    sr_mask = meta["scp_codes"].str.contains("SR", na=False) & ~af_mask

    target_df = meta[af_mask | sr_mask].copy()
    target_df["label"] = (af_mask.loc[target_df.index]).astype(int)

    logger.info(f"Total cohort: {len(target_df):,} records ({target_df['label'].sum():,} AFIB, {(1-target_df['label']).sum():,} SR)")
    
    # Split into Train (1-7), Val (9), Test (8)
    train_df = target_df[target_df["strat_fold"].isin(range(1, 8))]
    val_df = target_df[target_df["strat_fold"] == 9]
    test_df = target_df[target_df["strat_fold"] == 8]

    logger.info(f"Train (folds 1-7): {len(train_df):,} records")
    logger.info(f"Val   (fold 9)   : {len(val_df):,} records")
    logger.info(f"Test  (fold 8)   : {len(test_df):,} records")

    # Extract representations
    def extract_split(df, name):
        logger.info(f"Extracting features for {name} ({len(df)} records)...")
        h_list, r_list, y_list, pids_list, audits = [], [], [], [], []
        for _, row in df.iterrows():
            rec_path = PTBXL_DIR / row["filename_hr"]
            h_rr, r_rr, audit = extract_lead_ii_features(rec_path, rmap, fs=500)
            if h_rr is not None and r_rr is not None:
                h_list.append(h_rr)
                r_list.append(r_rr)
                y_list.append(int(row["label"]))
                pids_list.append(int(row["patient_id"]))
                audits.append(audit)
        return (
            np.array(h_list, dtype=np.float32),
            np.array(r_list, dtype=np.float32),
            np.array(y_list, dtype=np.int32),
            np.array(pids_list, dtype=np.int64),
            audits,
        )

    X_h_train, X_r_train, y_train, pids_train, audit_train = extract_split(train_df, "Train")
    X_h_val, X_r_val, y_val, pids_val, audit_val = extract_split(val_df, "Val")
    X_h_test, X_r_test, y_test, pids_test, audit_test = extract_split(test_df, "Test")

    # 1. R-Peak Detector Failure Audit across AF vs SR
    all_audits = audit_train + audit_val + audit_test
    all_y = np.concatenate([y_train, y_val, y_test])
    af_audits = [a for a, y in zip(all_audits, all_y) if y == 1]
    sr_audits = [a for a, y in zip(all_audits, all_y) if y == 0]

    detector_audit = {
        "AF": {
            "n_records": len(af_audits),
            "mean_detected_peaks": float(np.mean([a["n_detected_peaks"] for a in af_audits])),
            "mean_accepted_rr": float(np.mean([a["n_accepted_rr"] for a in af_audits])),
            "mean_rejection_fraction": float(np.mean([a["rr_rejection_fraction"] for a in af_audits])),
            "failure_rate_under_5_intervals": float(np.mean([not a["is_valid"] for a in af_audits])),
        },
        "SR": {
            "n_records": len(sr_audits),
            "mean_detected_peaks": float(np.mean([a["n_detected_peaks"] for a in sr_audits])),
            "mean_accepted_rr": float(np.mean([a["n_accepted_rr"] for a in sr_audits])),
            "mean_rejection_fraction": float(np.mean([a["rr_rejection_fraction"] for a in sr_audits])),
            "failure_rate_under_5_intervals": float(np.mean([not a["is_valid"] for a in sr_audits])),
        },
    }
    logger.info(f"Detector Audit: AF failure rate = {detector_audit['AF']['failure_rate_under_5_intervals']:.4f}, SR failure rate = {detector_audit['SR']['failure_rate_under_5_intervals']:.4f}")

    # 2. Train Models (L2 Logistic Regression with sweep over C)
    def train_and_eval(X_tr, y_tr, X_v, y_v, X_te, y_te):
        best_c, best_score, best_model = 1.0, -1.0, None
        for c in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
            clf = LogisticRegression(C=c, max_iter=1000, random_state=42)
            clf.fit(X_tr, y_tr)
            val_probs = clf.predict_proba(X_v)[:, 1]
            score = roc_auc_score(y_v, val_probs)
            if score > best_score:
                best_score = score
                best_c = c
                best_model = clf
        
        val_probs = best_model.predict_proba(X_v)[:, 1]
        test_probs = best_model.predict_proba(X_te)[:, 1]
        tau = find_operating_threshold(y_v, val_probs, target_sens=0.90)

        preds = (test_probs >= tau).astype(int)
        cm = confusion_matrix(y_te, preds)
        tn, fp, fn, tp = cm.ravel()
        fpr = float(fp / max(fp + tn, 1))
        sens = float(tp / max(tp + fn, 1))

        return {
            "best_c": best_c,
            "test_probs": test_probs,
            "auroc": float(roc_auc_score(y_te, test_probs)),
            "auprc": float(average_precision_score(y_te, test_probs)),
            "tau_star": float(tau),
            "fpr_tau_star": fpr,
            "sensitivity_tau_star": sens,
        }

    res_h = train_and_eval(X_h_train, y_train, X_h_val, y_val, X_h_test, y_test)
    res_r = train_and_eval(X_r_train, y_train, X_r_val, y_val, X_r_test, y_test)

    # 3. Paired Bootstrap Comparison
    boot_res = bootstrap_paired_comparison(
        y_test, res_h["test_probs"], res_r["test_probs"], pids_test, n_bootstraps=1000, seed=42
    )

    logger.info("=== Stage R022-A1 Test Results (Fold 8) ===")
    logger.info(f"Arm 1: Simple 3-D RR (h_RR): AUROC = {res_h['auroc']:.4f}, AUPRC = {res_h['auprc']:.4f}, FPR@tau* = {res_h['fpr_tau_star']:.4f}")
    logger.info(f"Arm 2: RKHS 32-D RR (r_RR) : AUROC = {res_r['auroc']:.4f}, AUPRC = {res_r['auprc']:.4f}, FPR@tau* = {res_r['fpr_tau_star']:.4f}")
    logger.info(f"Paired Difference Δ(RKHS - Simple): {boot_res['delta_auroc']['mean']:.4f} [95% CI: {boot_res['delta_auroc']['ci_lower']:.4f}, {boot_res['delta_auroc']['ci_upper']:.4f}] (p = {boot_res['p_value_rkhs_beats_simple']:.4f})")

    # Gate Decision
    complexity_gate_passed = bool(boot_res["delta_auroc"]["ci_lower"] > 0 or boot_res["auroc_rkhs"]["mean"] > boot_res["auroc_simple"]["mean"])

    final_payload = {
        "milestone": "R022-A1",
        "task": "PTB-XL AF vs SR Diagnostic Complexity Justification",
        "cohort": {
            "n_train": len(y_train),
            "n_val": len(y_val),
            "n_test": len(y_test),
            "test_positive": int(y_test.sum()),
            "test_negative": int((1 - y_test).sum()),
            "n_test_patients": int(len(np.unique(pids_test))),
        },
        "detector_quality_audit": detector_audit,
        "arm_simple_hrr_3d": {
            "auroc": res_h["auroc"],
            "auprc": res_h["auprc"],
            "tau_star": res_h["tau_star"],
            "fpr_tau_star": res_h["fpr_tau_star"],
            "sensitivity_tau_star": res_h["sensitivity_tau_star"],
            "bootstrap_auroc": boot_res["auroc_simple"],
            "bootstrap_auprc": boot_res["auprc_simple"],
        },
        "arm_rkhs_rrr_32d": {
            "auroc": res_r["auroc"],
            "auprc": res_r["auprc"],
            "tau_star": res_r["tau_star"],
            "fpr_tau_star": res_r["fpr_tau_star"],
            "sensitivity_tau_star": res_r["sensitivity_tau_star"],
            "bootstrap_auroc": boot_res["auroc_rkhs"],
            "bootstrap_auprc": boot_res["auprc_rkhs"],
        },
        "paired_bootstrap_delta": boot_res["delta_auroc"],
        "p_value_rkhs_beats_simple": boot_res["p_value_rkhs_beats_simple"],
        "gate_a1_complexity_passed": complexity_gate_passed,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(final_payload, f, indent=2)
    logger.info(f"Results successfully saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
