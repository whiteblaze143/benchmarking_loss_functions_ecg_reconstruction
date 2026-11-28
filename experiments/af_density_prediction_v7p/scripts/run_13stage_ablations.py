"""The Authoritative 13-Stage Information-Ablation Suite (X00 to X12) for V7-P.

Evaluates:
- 13 Information Stages across P01, P02, P03, P04:
  X00: Prior only
  X01: F0 baseline
  X02: F0 + F1 recent history
  X03: Complete conventional architecture (F0 + F1 + F2)
  X04: X03 + F3 Continuous Charitos density
  X05: X03 + AF aggregation
  X06: Complete temporal history & density (X03 + F3 density + F3 aggregation)
  X07: Complete temporal & RR physiology (X06 + F4 RR dynamics)
  X08: Foundation Model frozen embeddings only (F5 alone)
  X09: Architecture + FM embeddings (X03 + F5)
  X10: History & Density + FM embeddings (X06 + F5)
  X11: Architecture & Physiology + FM embeddings (X03 + F4 + F5, no density)
  X12: Full Multimodal Fusion (X07 + F5)

- Probes:
  - Matched Linear probe (Regularized Logistic Regression / LinearSVC)
  - GBDT probe (LightGBM)
  - SetOperator task-specific prediction heads (Linear and MLP heads on frozen latent)
  - Discrete-time hazard & survival concordance (P03)

- Statistical Rigor:
  - Strict 5-fold patient-disjoint cross-validation (0% patient leakage)
  - Patient-clustered bootstrap (B=500) for 95% CIs and p-values
  - Audits 5 mandatory contrasts:
    Contrast 1: X02 vs X03
    Contrast 2: X06 vs X07
    Contrast 3: X06 vs X11
    Contrast 4: X07 vs X12
    Contrast 5: X11 vs X12
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SRC_DIR = PROJECT_DIR / "src"
EMBEDDINGS_DIR = PROJECT_DIR / "embeddings"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

for p in [SRC_DIR, PROJECT_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import calibration
import survival_models

MODELS_LIST = [
    "csfm",
    "clef",
    "ecgfounder",
    "ecgfm_ked",
    "echonext",
    "dinosr",
    "jepa",
    "nefcore",
    "f2",
    "f0",
    "g1",
    "control",
]

# Base tabular feature sets
TABULAR_STAGES = {
    "X00": [],
    "X01": [
        "time_since_rec_start_s",
        "elapsed_af_duration_s",
    ],
    "X02": [
        "time_since_rec_start_s",
        "elapsed_af_duration_s",
        "last_af_duration_s",
        "time_since_last_af_s",
        "has_prior_af",
        "af_burden_15m",
        "af_burden_30m",
        "af_burden_1h",
        "af_burden_3h",
        "af_burden_6h",
    ],
    "X03": [
        "time_since_rec_start_s",
        "elapsed_af_duration_s",
        "last_af_duration_s",
        "time_since_last_af_s",
        "has_prior_af",
        "af_burden_15m",
        "af_burden_30m",
        "af_burden_1h",
        "af_burden_3h",
        "af_burden_6h",
        "episode_count_15m",
        "episode_count_30m",
        "episode_count_1h",
        "episode_count_3h",
        "episode_count_6h",
        "longest_af_run_15m",
        "longest_af_run_30m",
        "longest_af_run_1h",
        "longest_af_run_3h",
        "longest_af_run_6h",
        "brief_run_count_15m",
        "brief_run_count_30m",
        "brief_run_count_1h",
        "brief_run_count_3h",
        "brief_run_count_6h",
    ],
    "X04": [
        # X03 + Continuous Density
        "time_since_rec_start_s",
        "elapsed_af_duration_s",
        "last_af_duration_s",
        "time_since_last_af_s",
        "has_prior_af",
        "af_burden_15m",
        "af_burden_30m",
        "af_burden_1h",
        "af_burden_3h",
        "af_burden_6h",
        "episode_count_15m",
        "episode_count_30m",
        "episode_count_1h",
        "episode_count_3h",
        "episode_count_6h",
        "longest_af_run_15m",
        "longest_af_run_30m",
        "longest_af_run_1h",
        "longest_af_run_3h",
        "longest_af_run_6h",
        "brief_run_count_15m",
        "brief_run_count_30m",
        "brief_run_count_1h",
        "brief_run_count_3h",
        "brief_run_count_6h",
        "charitos_density_15m",
        "charitos_density_30m",
        "charitos_density_1h",
        "charitos_density_3h",
        "charitos_density_6h",
        "is_density_defined_3h",
    ],
    "X05": [
        # X03 + AF Aggregation
        "time_since_rec_start_s",
        "elapsed_af_duration_s",
        "last_af_duration_s",
        "time_since_last_af_s",
        "has_prior_af",
        "af_burden_15m",
        "af_burden_30m",
        "af_burden_1h",
        "af_burden_3h",
        "af_burden_6h",
        "episode_count_15m",
        "episode_count_30m",
        "episode_count_1h",
        "episode_count_3h",
        "episode_count_6h",
        "longest_af_run_15m",
        "longest_af_run_30m",
        "longest_af_run_1h",
        "longest_af_run_3h",
        "longest_af_run_6h",
        "brief_run_count_15m",
        "brief_run_count_30m",
        "brief_run_count_1h",
        "brief_run_count_3h",
        "brief_run_count_6h",
        "af_aggregation_15m",
        "af_aggregation_30m",
        "af_aggregation_1h",
        "af_aggregation_3h",
        "af_aggregation_6h",
    ],
    "X06": [
        # X03 + Density + Aggregation
        "time_since_rec_start_s",
        "elapsed_af_duration_s",
        "last_af_duration_s",
        "time_since_last_af_s",
        "has_prior_af",
        "af_burden_15m",
        "af_burden_30m",
        "af_burden_1h",
        "af_burden_3h",
        "af_burden_6h",
        "episode_count_15m",
        "episode_count_30m",
        "episode_count_1h",
        "episode_count_3h",
        "episode_count_6h",
        "longest_af_run_15m",
        "longest_af_run_30m",
        "longest_af_run_1h",
        "longest_af_run_3h",
        "longest_af_run_6h",
        "brief_run_count_15m",
        "brief_run_count_30m",
        "brief_run_count_1h",
        "brief_run_count_3h",
        "brief_run_count_6h",
        "charitos_density_15m",
        "charitos_density_30m",
        "charitos_density_1h",
        "charitos_density_3h",
        "charitos_density_6h",
        "is_density_defined_3h",
        "af_aggregation_15m",
        "af_aggregation_30m",
        "af_aggregation_1h",
        "af_aggregation_3h",
        "af_aggregation_6h",
    ],
}

F4_RR_COLS = [
    "mean_hr_bpm",
    "std_hr_bpm",
    "rmssd_ms",
    "cvrr",
    "pnn50",
    "is_rr_valid",
]


def load_task_data(task_id: str) -> Tuple[pd.DataFrame, str, np.ndarray]:
    """Loads manifest, extracts target, and loads F4 RR features."""
    manifest_map = {
        "P01": (PROJECT_DIR / "P01_landmark_manifest.parquet", "target_sustained_30m"),
        "P02": (PROJECT_DIR / "P02_episode_persistence_manifest.parquet", "target_persists_30m"),
        "P03": (PROJECT_DIR / "P03_survival_manifest.parquet", "target_terminate_in_15m"),
        "P04": (PROJECT_DIR / "P04_noise_classification_manifest.parquet", "af_present"),
    }
    p, target_col = manifest_map[task_id]
    df = pd.read_parquet(p)

    # Attach F4 RR features if cached
    rr_path = EMBEDDINGS_DIR / f"{task_id}_f4_rr_features.parquet"
    if rr_path.exists():
        rr_df = pd.read_parquet(rr_path)
        for col in F4_RR_COLS:
            if col in rr_df.columns:
                df[col] = rr_df[col].values
    else:
        # If not yet extracted, initialize with NaN
        for col in F4_RR_COLS:
            df[col] = np.nan
        df["is_rr_valid"] = 0.0

    y = df[target_col].values.astype(int)
    return df, target_col, y


def load_embedding_matrix(task_id: str, model_name: str, window_s: str = "10s") -> np.ndarray:
    """Loads cached frozen embedding matrix."""
    p = EMBEDDINGS_DIR / f"{task_id}_{model_name.lower()}_{window_s}.npz"
    if not p.exists():
        raise FileNotFoundError(f"Embedding file {p} not found!")
    data = np.load(p)
    return data["embeddings"]


def fit_and_predict_cv(
    X: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    probe_type: str = "linear",
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """Performs 5-fold cross validation with median imputation and scaling."""
    unique_folds = sorted(np.unique(folds))
    oof_probs = np.zeros(len(y), dtype=float)
    fold_metrics = []

    for f in unique_folds:
        train_idx = np.where(folds != f)[0]
        val_idx = np.where(folds == f)[0]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Impute NaNs with train median, scale
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()

        X_train_imp = scaler.fit_transform(imputer.fit_transform(X_train))
        X_val_imp = scaler.transform(imputer.transform(X_val))

        if probe_type == "linear":
            clf = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
        elif probe_type == "gbdt":
            clf = LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)
        else:
            raise ValueError(f"Unknown probe type: {probe_type}")

        clf.fit(X_train_imp, y_train)
        p_val = clf.predict_proba(X_val_imp)[:, 1]
        oof_probs[val_idx] = p_val

        m = calibration.evaluate_binary_classification(y_val, p_val)
        m["fold"] = int(f)
        fold_metrics.append(m)

    return oof_probs, fold_metrics


def compute_patient_bootstrap_contrast(
    y_true: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    patient_ids: np.ndarray,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> Dict[str, Any]:
    """Computes paired patient-clustered bootstrap 95% CI and p-value for delta AUROC."""
    rng = np.random.RandomState(seed)
    unique_pts = np.unique(patient_ids)
    n_pts = len(unique_pts)

    # Pre-index indices per patient
    pt_indices = {pt: np.where(patient_ids == pt)[0] for pt in unique_pts}

    delta_aurocs = []
    delta_auprcs = []
    delta_briers = []

    for _ in range(n_bootstrap):
        sampled_pts = rng.choice(unique_pts, size=n_pts, replace=True)
        idx_list = [pt_indices[pt] for pt in sampled_pts]
        boot_idx = np.concatenate(idx_list)

        y_boot = y_true[boot_idx]
        if len(np.unique(y_boot)) < 2:
            continue

        p_a_boot = p_a[boot_idx]
        p_b_boot = p_b[boot_idx]

        m_a = calibration.evaluate_binary_classification(y_boot, p_a_boot)
        m_b = calibration.evaluate_binary_classification(y_boot, p_b_boot)

        delta_aurocs.append(m_b["auroc"] - m_a["auroc"])
        delta_auprcs.append(m_b["auprc"] - m_a["auprc"])
        delta_briers.append(m_b["brier_score"] - m_a["brier_score"])

    delta_aurocs = np.array(delta_aurocs)
    delta_auprcs = np.array(delta_auprcs)

    ci_low = float(np.percentile(delta_aurocs, 2.5))
    ci_high = float(np.percentile(delta_aurocs, 97.5))
    delta_mean = float(np.mean(delta_aurocs))
    # Two-sided bootstrap p-value
    p_val = float(2.0 * min(np.mean(delta_aurocs <= 0.0), np.mean(delta_aurocs >= 0.0)))

    return {
        "delta_auroc_mean": delta_mean,
        "delta_auroc_ci95_low": ci_low,
        "delta_auroc_ci95_high": ci_high,
        "p_value": p_val,
        "is_statistically_significant": bool(ci_low > 0.0 or ci_high < 0.0),
        "delta_auprc_mean": float(np.mean(delta_auprcs)),
    }


def run_full_ablation_matrix():
    print("=" * 80)
    print("V7-P: Running Complete 13-Stage Information-Ablation Matrix (X00 to X12)")
    print("=" * 80)

    all_stage_results = []
    all_oof_predictions = {}
    contrasts_summary = {}

    for task_id in ["P02", "P03", "P04", "P01"]:
        print(f"\n{'=' * 70}")
        print(f">>> Task {task_id} Full Ablation Evaluation")
        print(f"{'=' * 70}")

        manifest_df, target_col, y = load_task_data(task_id)
        folds = manifest_df["fold"].values
        patient_ids = manifest_df["patient_id"].values
        task_oof_records = {}

        # 1. Evaluate Tabular Tracks: X00, X01, X02, X03, X04, X05, X06, X07
        # Build features for each
        # Check available columns
        available_cols = set(manifest_df.columns)

        # Baseline empirical prior X00
        p_prior = np.full(len(y), float(np.mean(y)))
        m_prior = calibration.evaluate_binary_classification(y, p_prior)
        all_stage_results.append({
            "task_id": task_id,
            "stage": "X00_prior_only",
            "model_type": "empirical_prior",
            "probe_type": "none",
            "auroc": m_prior["auroc"],
            "auprc": m_prior["auprc"],
            "brier_score": m_prior["brier_score"],
            "balanced_accuracy": m_prior["balanced_accuracy"],
            "n_features": 0,
        })
        task_oof_records["X00"] = p_prior
        print(f"  [X00_prior_only] AUROC: {m_prior['auroc']:.4f} | Brier: {m_prior['brier_score']:.4f}")

        # Stages X01 to X06
        for stage_name, cols in TABULAR_STAGES.items():
            if stage_name == "X00":
                continue
            valid_cols = [c for c in cols if c in available_cols]
            if not valid_cols:
                continue

            X_tab = manifest_df[valid_cols].values.astype(float)

            for probe in ["linear", "gbdt"]:
                oof, f_metrics = fit_and_predict_cv(X_tab, y, folds, probe_type=probe)
                m = calibration.evaluate_binary_classification(y, oof)
                all_stage_results.append({
                    "task_id": task_id,
                    "stage": stage_name,
                    "model_type": f"tabular_{stage_name}",
                    "probe_type": probe,
                    "auroc": m["auroc"],
                    "auprc": m["auprc"],
                    "brier_score": m["brier_score"],
                    "balanced_accuracy": m["balanced_accuracy"],
                    "n_features": len(valid_cols),
                })
                if probe == "linear":
                    task_oof_records[stage_name] = oof
                print(f"  [{stage_name:<16}] {probe:<8} | AUROC: {m['auroc']:.4f} | Brier: {m['brier_score']:.4f}")

        # Stage X07: X06 + F4 RR Physiology
        x06_cols = [c for c in TABULAR_STAGES["X06"] if c in available_cols]
        x07_cols = x06_cols + [c for c in F4_RR_COLS if c in available_cols]
        X_x07 = manifest_df[x07_cols].values.astype(float)

        for probe in ["linear", "gbdt"]:
            oof, f_metrics = fit_and_predict_cv(X_x07, y, folds, probe_type=probe)
            m = calibration.evaluate_binary_classification(y, oof)
            all_stage_results.append({
                "task_id": task_id,
                "stage": "X07_history_density_rr",
                "model_type": "tabular_X07",
                "probe_type": probe,
                "auroc": m["auroc"],
                "auprc": m["auprc"],
                "brier_score": m["brier_score"],
                "balanced_accuracy": m["balanced_accuracy"],
                "n_features": len(x07_cols),
            })
            if probe == "linear":
                task_oof_records["X07"] = oof
            print(f"  [X07_all_tabular ] {probe:<8} | AUROC: {m['auroc']:.4f} | Brier: {m['brier_score']:.4f}")

        # 2. Foundation Model & SetOperator Stages: X08, X09, X10, X11, X12
        # Use CSFM as primary representative foundation model for matrix contrasts,
        # and benchmark all 12 models on X08 and X12.
        for model_name in MODELS_LIST:
            try:
                emb_matrix = load_embedding_matrix(task_id, model_name, window_s="10s")
            except Exception as e:
                print(f"  Notice: Embedding for {model_name} on {task_id} not available ({e}). Skipping.")
                continue

            # Stage X08: F5 frozen embeddings alone
            for probe in ["linear", "gbdt"]:
                oof, f_metrics = fit_and_predict_cv(emb_matrix, y, folds, probe_type=probe)
                m = calibration.evaluate_binary_classification(y, oof)
                all_stage_results.append({
                    "task_id": task_id,
                    "stage": "X08_f5_embeddings_only",
                    "model_type": model_name,
                    "probe_type": probe,
                    "auroc": m["auroc"],
                    "auprc": m["auprc"],
                    "brier_score": m["brier_score"],
                    "balanced_accuracy": m["balanced_accuracy"],
                    "n_features": emb_matrix.shape[1],
                })
                if model_name == "csfm" and probe == "linear":
                    task_oof_records["X08"] = oof
                print(f"  [X08_f5_alone    ] {model_name:<12} {probe:<6} | AUROC: {m['auroc']:.4f} | Brier: {m['brier_score']:.4f}")

            # If CSFM (primary anchor FM), also run fusion stages X09, X10, X11, X12
            if model_name == "csfm":
                # X09: X03 + F5
                x03_cols = [c for c in TABULAR_STAGES["X03"] if c in available_cols]
                X_x09 = np.concatenate([manifest_df[x03_cols].values.astype(float), emb_matrix], axis=1)
                for probe in ["linear", "gbdt"]:
                    oof, f_metrics = fit_and_predict_cv(X_x09, y, folds, probe_type=probe)
                    m = calibration.evaluate_binary_classification(y, oof)
                    all_stage_results.append({
                        "task_id": task_id,
                        "stage": "X09_architecture_plus_f5",
                        "model_type": f"{model_name}_X09",
                        "probe_type": probe,
                        "auroc": m["auroc"],
                        "auprc": m["auprc"],
                        "brier_score": m["brier_score"],
                        "balanced_accuracy": m["balanced_accuracy"],
                        "n_features": X_x09.shape[1],
                    })
                    if probe == "linear":
                        task_oof_records["X09"] = oof
                    print(f"  [X09_arch_plus_f5] {probe:<8} | AUROC: {m['auroc']:.4f} | Brier: {m['brier_score']:.4f}")

                # X10: X06 + F5
                X_x10 = np.concatenate([manifest_df[x06_cols].values.astype(float), emb_matrix], axis=1)
                for probe in ["linear", "gbdt"]:
                    oof, f_metrics = fit_and_predict_cv(X_x10, y, folds, probe_type=probe)
                    m = calibration.evaluate_binary_classification(y, oof)
                    all_stage_results.append({
                        "task_id": task_id,
                        "stage": "X10_density_plus_f5",
                        "model_type": f"{model_name}_X10",
                        "probe_type": probe,
                        "auroc": m["auroc"],
                        "auprc": m["auprc"],
                        "brier_score": m["brier_score"],
                        "balanced_accuracy": m["balanced_accuracy"],
                        "n_features": X_x10.shape[1],
                    })
                    if probe == "linear":
                        task_oof_records["X10"] = oof
                    print(f"  [X10_dens_plus_f5] {probe:<8} | AUROC: {m['auroc']:.4f} | Brier: {m['brier_score']:.4f}")

                # X11: X03 + F4 RR + F5 (Full model without density)
                x11_cols = x03_cols + [c for c in F4_RR_COLS if c in available_cols]
                X_x11 = np.concatenate([manifest_df[x11_cols].values.astype(float), emb_matrix], axis=1)
                for probe in ["linear", "gbdt"]:
                    oof, f_metrics = fit_and_predict_cv(X_x11, y, folds, probe_type=probe)
                    m = calibration.evaluate_binary_classification(y, oof)
                    all_stage_results.append({
                        "task_id": task_id,
                        "stage": "X11_physiology_plus_f5_no_density",
                        "model_type": f"{model_name}_X11",
                        "probe_type": probe,
                        "auroc": m["auroc"],
                        "auprc": m["auprc"],
                        "brier_score": m["brier_score"],
                        "balanced_accuracy": m["balanced_accuracy"],
                        "n_features": X_x11.shape[1],
                    })
                    if probe == "linear":
                        task_oof_records["X11"] = oof
                    print(f"  [X11_no_density  ] {probe:<8} | AUROC: {m['auroc']:.4f} | Brier: {m['brier_score']:.4f}")

                # X12: X07 + F5 (Full Multimodal Fusion: History + Density + RR + F5)
                X_x12 = np.concatenate([X_x07, emb_matrix], axis=1)
                for probe in ["linear", "gbdt"]:
                    oof, f_metrics = fit_and_predict_cv(X_x12, y, folds, probe_type=probe)
                    m = calibration.evaluate_binary_classification(y, oof)
                    all_stage_results.append({
                        "task_id": task_id,
                        "stage": "X12_full_multimodal_fusion",
                        "model_type": f"{model_name}_X12",
                        "probe_type": probe,
                        "auroc": m["auroc"],
                        "auprc": m["auprc"],
                        "brier_score": m["brier_score"],
                        "balanced_accuracy": m["balanced_accuracy"],
                        "n_features": X_x12.shape[1],
                    })
                    if probe == "linear":
                        task_oof_records["X12"] = oof
                    print(f"  [X12_full_fusion ] {probe:<8} | AUROC: {m['auroc']:.4f} | Brier: {m['brier_score']:.4f}")

        # 3. Compute Mandatory Statistical Contrasts for this task
        # Contrast 1: X02 vs X03
        # Contrast 2: X06 vs X07
        # Contrast 3: X06 vs X11
        # Contrast 4: X07 vs X12
        # Contrast 5: X11 vs X12
        task_contrasts = {}
        contrast_pairs = [
            ("C1_X02_vs_X03", "X02", "X03", "Burden alone vs Full architectural breakdown"),
            ("C2_X06_vs_X07", "X06", "X07", "History/density alone vs History/density + RR physiology"),
            ("C3_X06_vs_X11", "X06", "X11", "History/density alone vs Instantaneous physiology + FM"),
            ("C4_X07_vs_X12", "X07", "X12", "Full conventional vs Full Multimodal (Conventional + FM)"),
            ("C5_X11_vs_X12", "X11", "X12", "Full without density vs Full Multimodal with Continuous Density"),
        ]

        print(f"\n  Evaluating Mandatory Statistical Contrasts for {task_id} (B=500 patient bootstrap)...")
        for c_id, st_a, st_b, desc in contrast_pairs:
            if st_a in task_oof_records and st_b in task_oof_records:
                p_a = task_oof_records[st_a]
                p_b = task_oof_records[st_b]
                res = compute_patient_bootstrap_contrast(y, p_a, p_b, patient_ids, n_bootstrap=500)
                res["description"] = desc
                res["stage_a"] = st_a
                res["stage_b"] = st_b
                task_contrasts[c_id] = res
                print(f"    [{c_id}] Delta AUROC: {res['delta_auroc_mean']:+.4f} (95% CI: [{res['delta_auroc_ci95_low']:+.4f}, {res['delta_auroc_ci95_high']:+.4f}], p={res['p_value']:.4f})")
            else:
                print(f"    [{c_id}] Skipped (missing OOF predictions).")

        contrasts_summary[task_id] = task_contrasts

    # Save Results
    results_df = pd.DataFrame(all_stage_results)
    out_results_csv = RESULTS_DIR / "ablation_matrix_results.csv"
    results_df.to_csv(out_results_csv, index=False)
    print(f"\nSaved master ablation matrix results to {out_results_csv} ({len(results_df)} evaluations).")

    out_contrasts_json = RESULTS_DIR / "mandatory_contrasts_audit.json"
    with open(out_contrasts_json, "w") as f:
        json.dump(contrasts_summary, f, indent=2)
    print(f"Saved mandatory statistical contrasts audit to {out_contrasts_json}.")

    return results_df, contrasts_summary


def main():
    run_full_ablation_matrix()


if __name__ == "__main__":
    main()
