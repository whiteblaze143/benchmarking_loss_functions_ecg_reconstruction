"""Statistical models, fractional logistic regression, and patient-clustered inference.
Implements nested models M0-M3 and head-to-head baselines B0-B6.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.families import Binomial
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def fit_fractional_logit(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "q_120",
    cluster_col: str = "patient_id"
) -> Tuple[Any, Dict[str, Any]]:
    """Fits quasi-binomial fractional logistic regression with patient-clustered covariance."""
    # Clip target to [0, 1]
    y = np.clip(df[target_col].values.astype(float), 1e-6, 1.0 - 1e-6)
    X = sm.add_constant(df[feature_cols].values.astype(float))
    
    # Model fit
    model = sm.GLM(y, X, family=Binomial())
    groups = pd.Categorical(df[cluster_col]).codes
    res = model.fit(cov_type="cluster", cov_kwds={"groups": groups})
    
    # Extract coefficients
    params = res.params
    pvalues = res.pvalues
    bse = res.bse
    
    coef_dict = {}
    for i, col in enumerate(["const"] + feature_cols):
        coef_dict[f"{col}_coef"] = float(params[i])
        coef_dict[f"{col}_se"] = float(bse[i])
        coef_dict[f"{col}_pval"] = float(pvalues[i])
        
    coef_dict["pseudo_r2_cs"] = float(res.pseudo_rsquared("cs")) if hasattr(res, "pseudo_rsquared") else np.nan
    coef_dict["aic"] = float(res.aic)
    coef_dict["bic"] = float(res.bic_llf)
    
    return res, coef_dict


def estimate_standardized_smax_contrast(
    model_res: Any,
    df_support: pd.DataFrame,
    feature_cols: List[str],
    smax_col: str = "s_max",
    contrast_delta: float = 0.25
) -> Dict[str, float]:
    """Calculates adjusted absolute percentage-point change in expected Q120 associated with +0.25 S_max."""
    # Baseline design matrix
    X_base = sm.add_constant(df_support[feature_cols].values.astype(float))
    smax_idx = feature_cols.index(smax_col) + 1  # accounting for const
    
    pred_base = model_res.predict(X_base)
    
    # Contrast design matrix (+0.25 on S_max, clipped to 1.0)
    X_contrast = X_base.copy()
    X_contrast[:, smax_idx] = np.clip(X_contrast[:, smax_idx] + contrast_delta, 0.0, 1.0)
    pred_contrast = model_res.predict(X_contrast)
    
    diff_arr = pred_contrast - pred_base
    return {
        "mean_adjusted_diff_pct_points": float(np.mean(diff_arr) * 100.0),
        "median_adjusted_diff_pct_points": float(np.median(diff_arr) * 100.0)
    }


def run_patient_bootstrap_smax_effect(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "q_120",
    cluster_col: str = "patient_id",
    n_bootstrap: int = 2000,
    seed: int = 42
) -> Tuple[float, float, float]:
    """Runs patient-clustered bootstrap (B=2000) for standardized S_max effect."""
    rng = np.random.RandomState(seed)
    unique_patients = df[cluster_col].unique()
    n_pats = len(unique_patients)
    
    boot_effects = []
    
    for _ in range(n_bootstrap):
        # Sample patients with replacement
        boot_pats = rng.choice(unique_patients, size=n_pats, replace=True)
        # Select all rows for chosen patients
        boot_df = pd.concat([df[df[cluster_col] == p] for p in boot_pats], ignore_index=True)
        
        try:
            res, _ = fit_fractional_logit(boot_df, feature_cols, target_col, cluster_col)
            contrast = estimate_standardized_smax_contrast(res, boot_df, feature_cols)
            boot_effects.append(contrast["mean_adjusted_diff_pct_points"])
        except Exception:
            continue
            
    if not boot_effects:
        return (np.nan, np.nan, np.nan)
        
    est = float(np.mean(boot_effects))
    ci_lower = float(np.percentile(boot_effects, 2.5))
    ci_upper = float(np.percentile(boot_effects, 97.5))
    return (est, ci_lower, ci_upper)


def run_patient_grouped_cv(
    df: pd.DataFrame,
    model_configs: Dict[str, List[str]],
    target_col: str = "q_120",
    group_col: str = "patient_id",
    n_splits: int = 5,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Runs 5-fold patient-grouped cross-validation across baseline models B0-B6."""
    gkf = GroupKFold(n_splits=n_splits)
    groups = pd.Categorical(df[group_col]).codes
    
    cv_records = []
    oof_predictions = df[["dataset", "patient_id", "record_id", target_col]].copy()
    
    for m_name, f_cols in model_configs.items():
        oof_preds = np.full(len(df), np.nan)
        fold_maes = []
        fold_rmses = []
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(df, groups=groups)):
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]
            
            y_train = np.clip(train_df[target_col].values.astype(float), 1e-6, 1.0 - 1e-6)
            X_train = sm.add_constant(train_df[f_cols].values.astype(float))
            
            y_val = val_df[target_col].values.astype(float)
            X_val = sm.add_constant(val_df[f_cols].values.astype(float))
            
            try:
                mod = sm.GLM(y_train, X_train, family=Binomial())
                res = mod.fit()
                val_pred = res.predict(X_val)
                oof_preds[val_idx] = val_pred
                
                f_mae = mean_absolute_error(y_val, val_pred)
                f_rmse = root_mean_squared_error(y_val, val_pred)
                fold_maes.append(f_mae)
                fold_rmses.append(f_rmse)
            except Exception:
                continue
                
        oof_predictions[f"pred_{m_name}"] = oof_preds
        valid_mask = np.isfinite(oof_preds)
        overall_mae = mean_absolute_error(df[target_col].values[valid_mask], oof_preds[valid_mask])
        overall_rmse = root_mean_squared_error(df[target_col].values[valid_mask], oof_preds[valid_mask])
        
        cv_records.append({
            "model": m_name,
            "features": "+".join(f_cols),
            "oof_mae": overall_mae,
            "oof_rmse": overall_rmse,
            "mean_fold_mae": float(np.mean(fold_maes)),
            "std_fold_mae": float(np.std(fold_maes))
        })
        
    return pd.DataFrame(cv_records), oof_predictions
