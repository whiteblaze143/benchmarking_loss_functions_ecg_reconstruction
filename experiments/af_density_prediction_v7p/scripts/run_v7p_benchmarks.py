"""V7-P Benchmark Runner: Evaluates Tier A, B, and C models on P01-P04 prediction manifests."""
import sys
import time
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import tabular_models
import survival_models
import calibration
import alert_evaluation

RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Feature groups for tabular models
FEATURE_GROUPS = {
    "X00_prior_only": [],
    "X01_f0_baseline": [
        "time_since_rec_start_s",
        "elapsed_af_duration_s",
    ],
    "X02_f0_f1_burden": [
        "time_since_rec_start_s",
        "elapsed_af_duration_s",
        "last_af_duration_s",
        "time_since_last_af_s",
        "af_burden_15m",
        "af_burden_30m",
        "af_burden_1h",
        "af_burden_3h",
        "af_burden_6h",
    ],
    "X03_f0_f1_f2_architecture": [
        "time_since_rec_start_s",
        "elapsed_af_duration_s",
        "last_af_duration_s",
        "time_since_last_af_s",
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
    "X04_f0_f1_f2_f3_density": [
        "time_since_rec_start_s",
        "elapsed_af_duration_s",
        "last_af_duration_s",
        "time_since_last_af_s",
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
        "af_aggregation_15m",
        "af_aggregation_30m",
        "af_aggregation_1h",
        "af_aggregation_3h",
        "af_aggregation_6h",
        "is_density_defined_15m",
        "is_density_defined_30m",
        "is_density_defined_1h",
        "is_density_defined_3h",
        "is_density_defined_6h",
    ],
}


def prepare_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Imputes missing/non-finite values and applies standard scaling."""
    if not feature_cols:
        return np.zeros((len(train_df), 1)), np.zeros((len(val_df), 1))

    # Available columns
    cols = [c for c in feature_cols if c in train_df.columns]
    if not cols:
        return np.zeros((len(train_df), 1)), np.zeros((len(val_df), 1))

    X_train = train_df[cols].values.astype(float)
    X_val = val_df[cols].values.astype(float)

    # Impute NaNs with median, replace infs
    X_train = np.nan_to_num(X_train, nan=np.nan, posinf=np.nan, neginf=np.nan)
    X_val = np.nan_to_num(X_val, nan=np.nan, posinf=np.nan, neginf=np.nan)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train = imputer.fit_transform(X_train)
    X_train = scaler.fit_transform(X_train)

    X_val = imputer.transform(X_val)
    X_val = scaler.transform(X_val)

    return X_train, X_val


def evaluate_task_classification(
    manifest_df: pd.DataFrame,
    task_id: str,
    target_col: str,
    models: List[str],
    feature_tracks: List[str] = ["X00_prior_only", "X01_f0_baseline", "X02_f0_f1_burden", "X03_f0_f1_f2_architecture", "X04_f0_f1_f2_f3_density"],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Runs 5-fold patient-disjoint cross-validation for a classification task."""
    print(f"\n>>> Running 5-fold CV for {task_id} (Target: {target_col}, N={len(manifest_df):,}, Pts={manifest_df['patient_id'].nunique()})")
    
    folds = sorted(manifest_df["fold"].unique())
    n_splits = len(folds)

    all_fold_metrics = []
    oof_predictions = []

    for feat_track in feature_tracks:
        feat_cols = FEATURE_GROUPS[feat_track]
        
        for model_name in models:
            if feat_track == "X00_prior_only" and model_name != "empirical_baseline":
                continue
            if model_name == "empirical_baseline" and feat_track != "X00_prior_only":
                continue
            if model_name == "elapsed_duration_baseline" and feat_track not in ["X00_prior_only", "X01_f0_baseline"]:
                continue

            t0 = time.time()
            fold_oof_probs = np.zeros(len(manifest_df))
            fold_metrics_list = []

            for f in folds:
                train_mask = manifest_df["fold"] != f
                val_mask = manifest_df["fold"] == f

                train_df = manifest_df[train_mask]
                val_df = manifest_df[val_mask]

                y_train = train_df[target_col].values.astype(int)
                y_val = val_df[target_col].values.astype(int)

                X_train, X_val = prepare_features(train_df, val_df, feat_cols)

                model = tabular_models.build_tabular_model(model_name, random_state=42)
                try:
                    # Subsample if dataset is very large for quadratic/cubic models (ebm, rbf_svm, gam)
                    if len(X_train) > 15000 and model_name in ["ebm", "rbf_svm", "gam"]:
                        # Stratified subsampling of training fold
                        pos_idx = np.where(y_train == 1)[0]
                        neg_idx = np.where(y_train == 0)[0]
                        np.random.seed(42)
                        n_neg = min(len(neg_idx), 15000 - len(pos_idx))
                        sample_idx = np.concatenate([pos_idx, np.random.choice(neg_idx, n_neg, replace=False)])
                        np.random.shuffle(sample_idx)
                        model.fit(X_train[sample_idx], y_train[sample_idx])
                    else:
                        model.fit(X_train, y_train)
                    p_val = model.predict_proba(X_val)[:, 1]
                except Exception as e:
                    raise RuntimeError(f"Model {model_name} on fold {f} failed: {e}")

                fold_oof_probs[val_mask] = p_val
                metrics = calibration.evaluate_binary_classification(y_val, p_val)
                metrics.update({
                    "task_id": task_id,
                    "target_col": target_col,
                    "feature_track": feat_track,
                    "model_name": model_name,
                    "fold": int(f),
                    "n_features": len(feat_cols),
                })
                fold_metrics_list.append(metrics)

            # Store OOF predictions
            oof_df = pd.DataFrame({
                "task_id": task_id,
                "target_col": target_col,
                "feature_track": feat_track,
                "model_name": model_name,
                "fold": manifest_df["fold"].values,
                "patient_id": manifest_df["patient_id"].values,
                "record_id": manifest_df["record_id"].values if "record_id" in manifest_df.columns else manifest_df.get("strip_id", ""),
                "landmark_time_s": manifest_df["landmark_time_s"].values if "landmark_time_s" in manifest_df.columns else 0.0,
                "target_true": manifest_df[target_col].values.astype(int),
                "predicted_prob": fold_oof_probs,
            })
            oof_predictions.append(oof_df)

            # Aggregate across folds
            overall_metrics = calibration.evaluate_binary_classification(manifest_df[target_col].values.astype(int), fold_oof_probs)
            mean_auroc = np.mean([m["auroc"] for m in fold_metrics_list])
            std_auroc = np.std([m["auroc"] for m in fold_metrics_list])
            mean_brier = np.mean([m["brier_score"] for m in fold_metrics_list])

            print(f"  [{task_id}] {feat_track:<26} | {model_name:<16} | OOF AUROC: {overall_metrics['auroc']:.4f} (CV: {mean_auroc:.4f}±{std_auroc:.4f}) | Brier: {overall_metrics['brier_score']:.4f} | Time: {time.time()-t0:.1f}s")
            all_fold_metrics.extend(fold_metrics_list)

    fold_metrics_df = pd.DataFrame(all_fold_metrics)
    all_oof_df = pd.concat(oof_predictions, ignore_index=True)

    # Summary table
    summary_records = []
    for (trk, mdl), sub in fold_metrics_df.groupby(["feature_track", "model_name"]):
        summary_records.append({
            "task_id": task_id,
            "feature_track": trk,
            "model_name": mdl,
            "auroc_mean": float(sub["auroc"].mean()),
            "auroc_std": float(sub["auroc"].std()),
            "auprc_mean": float(sub["auprc"].mean()),
            "brier_mean": float(sub["brier_score"].mean()),
            "brier_skill_mean": float(sub["brier_skill_score"].mean()),
            "log_loss_mean": float(sub["log_loss"].mean()),
            "calib_intercept_mean": float(sub["calibration_intercept"].mean()),
            "calib_slope_mean": float(sub["calibration_slope"].mean()),
            "ece_mean": float(sub["ece_10_bins"].mean()),
            "bal_acc_mean": float(sub["balanced_accuracy"].mean()),
            "f1_mean": float(sub["f1_score"].mean()),
            "sensitivity_mean": float(sub["sensitivity"].mean()),
            "specificity_mean": float(sub["specificity"].mean()),
        })
    summary_df = pd.DataFrame(summary_records).sort_values(by="auroc_mean", ascending=False)
    return fold_metrics_df, summary_df, all_oof_df


def run_survival_benchmarks_p03(p03_df: pd.DataFrame) -> pd.DataFrame:
    """Runs Tier C Survival benchmarks (Cox PH and Weibull AFT) across 5 folds."""
    print("\n>>> Running Tier C Survival Models (Cox PH & Weibull AFT) on Task P03...")
    folds = sorted(p03_df["fold"].unique())
    records = []

    suite = survival_models.SurvivalBenchmarkSuite(penalizer=0.1)
    feature_cols = [
        "time_since_rec_start_s",
        "elapsed_af_duration_s",
        "af_burden_1h",
        "episode_count_1h",
        "longest_af_run_1h",
        "brief_run_count_1h",
        "charitos_density_1h",
        "af_aggregation_1h",
    ]

    for f in folds:
        train_df = p03_df[p03_df["fold"] != f].copy()
        val_df = p03_df[p03_df["fold"] == f].copy()

        res = suite.fit_and_evaluate(
            train_df=train_df,
            val_df=val_df,
            duration_col="time_to_event_s",
            event_col="event",
            feature_cols=feature_cols,
        )
        res["fold"] = int(f)
        records.append(res)
        print(f"  [P03-Survival] Fold {f}: Cox Val C-Index: {res['cox_c_index_val']:.4f} | Weibull Val C-Index: {res['weibull_c_index_val']:.4f}")

    res_df = pd.DataFrame(records)
    print(f"  [P03-Survival Summary] Cox C-Index: {res_df['cox_c_index_val'].mean():.4f}±{res_df['cox_c_index_val'].std():.4f} | Weibull C-Index: {res_df['weibull_c_index_val'].mean():.4f}±{res_df['weibull_c_index_val'].std():.4f}")
    return res_df


def run_noise_holter_benchmarks_p04(p04_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluates Noise Holter rhythm classification across noise strata (Clean, N1, N2)."""
    print("\n>>> Running Task P04 Noise Holter Authentic Rhythm Benchmarks...")
    
    # Feature columns for Noise Holter
    # Quality stratum one-hot
    df = p04_df.copy()
    for q in ["Clean", "N1", "N2"]:
        df[f"quality_{q}"] = (df["quality_stratum"] == q).astype(float)
    df["noise_is_emg"] = (df["noise_type"] == "EMG").astype(float)
    df["noise_is_bw"] = (df["noise_type"] == "BW").astype(float)
    df["noise_is_ma"] = (df["noise_type"] == "MA").astype(float)

    feat_cols = ["sampling_rate", "channels", "duration_seconds", "quality_Clean", "quality_N1", "quality_N2", "noise_is_emg", "noise_is_bw", "noise_is_ma"]

    models = ["empirical_baseline", "elastic_net", "decision_tree", "gbdt_lightgbm", "gbdt_xgboost"]
    folds = sorted(df["fold"].unique())
    records = []
    strata_records = []

    for model_name in models:
        oof_probs = np.zeros(len(df))

        for f in folds:
            train_mask = df["fold"] != f
            val_mask = df["fold"] == f

            train_df = df[train_mask]
            val_df = df[val_mask]

            y_train = train_df["af_present"].values.astype(int)
            y_val = val_df["af_present"].values.astype(int)

            if model_name == "empirical_baseline":
                p_val = np.full(len(y_val), float(np.mean(y_train)))
            else:
                X_train = train_df[feat_cols].values
                X_val = val_df[feat_cols].values
                imputer = SimpleImputer(strategy="median")
                scaler = StandardScaler()
                X_train = scaler.fit_transform(imputer.fit_transform(X_train))
                X_val = scaler.transform(imputer.transform(X_val))

                clf = tabular_models.build_tabular_model(model_name, random_state=42)
                clf.fit(X_train, y_train)
                p_val = clf.predict_proba(X_val)[:, 1]

            oof_probs[val_mask] = p_val

        # Overall metrics
        metrics = calibration.evaluate_binary_classification(df["af_present"].values.astype(int), oof_probs)
        metrics["model_name"] = model_name
        metrics["stratum"] = "ALL"
        records.append(metrics)
        print(f"  [P04-All] {model_name:<16} | AUROC: {metrics['auroc']:.4f} | Brier: {metrics['brier_score']:.4f}")

        # Stratified by noise quality
        for q in ["Clean", "N1", "N2"]:
            q_mask = df["quality_stratum"] == q
            q_metrics = calibration.evaluate_binary_classification(df.loc[q_mask, "af_present"].values.astype(int), oof_probs[q_mask])
            q_metrics["model_name"] = model_name
            q_metrics["stratum"] = q
            strata_records.append(q_metrics)

    return pd.DataFrame(records), pd.DataFrame(strata_records)


def main():
    print("=" * 80)
    print("V7-P Predictive Framework: Comprehensive Benchmark Execution Suite")
    print("=" * 80)

    # 1. Load Parquet Manifests
    print("\nLoading manifests...")
    p01_df = pd.read_parquet(PROJECT_DIR / "P01_landmark_manifest.parquet")
    p02_df = pd.read_parquet(PROJECT_DIR / "P02_episode_persistence_manifest.parquet")
    p03_df = pd.read_parquet(PROJECT_DIR / "P03_survival_manifest.parquet")
    p04_df = pd.read_parquet(PROJECT_DIR / "P04_noise_classification_manifest.parquet")

    # Priority model subset for fast, robust cross-validation
    priority_models = [
        "empirical_baseline",
        "elastic_net",
        "gam",
        "ebm",
        "linear_svm",
        "decision_tree",
        "gbdt_lightgbm",
        "gbdt_xgboost",
    ]

    all_summaries = []
    all_oof = []

    # Task P01: Forecasting Sustained AF Onset
    p01_folds, p01_sum, p01_oof = evaluate_task_classification(
        p01_df,
        task_id="P01",
        target_col="target_sustained_30m",
        models=priority_models,
    )
    all_summaries.append(p01_sum)
    all_oof.append(p01_oof)

    # Alert benchmark evaluation for P01
    print("\n>>> Computing Clinical Alert Benchmarks for Task P01...")
    alert_records = []
    # Evaluate best performing model (e.g. gbdt_lightgbm on X04)
    best_p01_oof = p01_oof[(p01_oof["model_name"] == "gbdt_lightgbm") & (p01_oof["feature_track"] == "X04_f0_f1_f2_f3_density")].copy()
    best_p01_oof["target_sustained_30m"] = p01_df["target_sustained_30m"].values
    best_p01_oof["landmark_time_s"] = p01_df["landmark_time_s"].values
    best_p01_oof["event_time_s"] = p01_df["event_time_s"].values

    for thresh in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        res = alert_evaluation.evaluate_p01_alerts(best_p01_oof, probability_threshold=thresh)
        res["model_name"] = "gbdt_lightgbm"
        res["feature_track"] = "X04_f0_f1_f2_f3_density"
        alert_records.append(res)
    alert_df = pd.DataFrame(alert_records)
    alert_path = RESULTS_DIR / "V7P_alert_benchmarks.csv"
    alert_df.to_csv(alert_path, index=False)
    print(f"Saved {alert_path}.")

    # Task P02: Episode Persistence at 60s Landmark
    p02_models = priority_models + ["elapsed_duration_baseline"]
    p02_folds, p02_sum, p02_oof = evaluate_task_classification(
        p02_df,
        task_id="P02",
        target_col="target_persists_30m",
        models=p02_models,
    )
    all_summaries.append(p02_sum)
    all_oof.append(p02_oof)

    # Task P03: Dynamic Survival / 15m Termination
    p03_folds, p03_sum, p03_oof = evaluate_task_classification(
        p03_df,
        task_id="P03",
        target_col="target_terminate_in_15m",
        models=priority_models,
    )
    all_summaries.append(p03_sum)
    all_oof.append(p03_oof)

    # Task P03 Survival Models (Cox PH, Weibull AFT)
    p03_survival_df = run_survival_benchmarks_p03(p03_df)
    p03_surv_path = RESULTS_DIR / "V7P_survival_benchmarks.csv"
    p03_survival_df.to_csv(p03_surv_path, index=False)
    print(f"Saved {p03_surv_path}.")

    # Task P04: Noise Holter Rhythm Classification
    p04_overall_df, p04_strata_df = run_noise_holter_benchmarks_p04(p04_df)
    p04_strata_path = RESULTS_DIR / "V7P_noise_strata_benchmarks.csv"
    p04_strata_df.to_csv(p04_strata_path, index=False)
    print(f"Saved {p04_strata_path}.")

    # Unified Deliverables
    print("\n>>> Authoring Master Unified Benchmark Deliverables...")
    master_summary = pd.concat(all_summaries, ignore_index=True)
    master_class_path = RESULTS_DIR / "V7P_classification_benchmarks.csv"
    master_summary.to_csv(master_class_path, index=False)
    print(f"Saved {master_class_path} ({len(master_summary)} configurations).")

    # Feature Ablation Results Table
    ablation_df = master_summary[["task_id", "feature_track", "model_name", "auroc_mean", "auroc_std", "brier_mean", "brier_skill_mean", "bal_acc_mean", "f1_mean"]].copy()
    ablation_path = RESULTS_DIR / "V7P_feature_ablation_results.csv"
    ablation_df.to_csv(ablation_path, index=False)
    print(f"Saved {ablation_path}.")

    # Out-of-fold predictions
    master_oof = pd.concat(all_oof, ignore_index=True)
    master_oof_path = RESULTS_DIR / "V7P_out_of_fold_predictions.parquet"
    master_oof.to_parquet(master_oof_path, index=False)
    print(f"Saved {master_oof_path} ({len(master_oof):,} prediction rows).")

    print("\n" + "=" * 80)
    print("V7-P Benchmark Suite Completed Successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
