"""Analysis utilities for ECG reconstruction evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_categorical_dtype
from scipy import stats

from src.reconstruction.evaluate_functions.metrics import mse, mae


@dataclass
class BlandAltmanResult:
    bias: float
    loa_lower: float
    loa_upper: float
    slope: float
    intercept: float
    correlation: float
    figure_path: Optional[str] = None


def bland_altman_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lead_names: Sequence[str],
    output_dir: Optional[str] = None,
) -> Dict[str, BlandAltmanResult]:
    """Generate Bland–Altman statistics and optional plots per lead."""
    results: Dict[str, BlandAltmanResult] = {}
    for idx, lead in enumerate(lead_names):
        true_lead = y_true[:, idx].reshape(-1)
        pred_lead = y_pred[:, idx].reshape(-1)
        diff = pred_lead - true_lead
        mean = (pred_lead + true_lead) / 2.0
        bias = float(np.mean(diff))
        sd = float(np.std(diff))
        loa_low = bias - 1.96 * sd
        loa_high = bias + 1.96 * sd
        slope, intercept, r_value, _, _ = stats.linregress(mean, diff)
        fig_path = None
        if output_dir:
            output_dir_path = Path(output_dir)
            output_dir_path.mkdir(parents=True, exist_ok=True)
            fig_path = output_dir_path / f"bland_altman_{lead}.png"
            plt.figure(figsize=(6, 4))
            plt.scatter(mean, diff, s=3, alpha=0.35, color="tab:blue")
            plt.axhline(bias, color="tab:red", linestyle="--", linewidth=1.0, label=f"Bias {bias:.3f} mV")
            plt.axhline(loa_low, color="tab:gray", linestyle=":", linewidth=1.0, label=f"LoA − {loa_low:.3f}")
            plt.axhline(loa_high, color="tab:gray", linestyle=":", linewidth=1.0, label=f"LoA + {loa_high:.3f}")
            plt.plot(mean, slope * mean + intercept, color="tab:orange", linewidth=1.0, label=f"Slope {slope:.3f}")
            plt.xlabel("Mean of prediction and reference (mV)")
            plt.ylabel("Prediction − Reference (mV)")
            plt.title(f"Bland–Altman: {lead}")
            plt.legend(loc="upper right", fontsize=8)
            plt.tight_layout()
            plt.savefig(fig_path, dpi=200)
            plt.close()
        results[lead] = BlandAltmanResult(
            bias=bias,
            loa_lower=loa_low,
            loa_upper=loa_high,
            slope=float(slope),
            intercept=float(intercept),
            correlation=float(r_value),
            figure_path=str(fig_path) if fig_path else None,
        )
    return results


def subgroup_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metadata_df: pd.DataFrame,
    features: Sequence[str],
) -> pd.DataFrame:
    """Compute metrics across specified subgroups."""
    rows: List[Dict[str, object]] = []
    metrics = {
        "mse": mse,
        "mae": mae,
    }
    leads = y_true.shape[1]
    lead_axis = np.arange(leads)

    def _safe_float(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors="coerce")

    for feature in features:
        if feature not in metadata_df.columns:
            continue
        column = metadata_df[feature]
        if is_categorical_dtype(column):
            column = column.astype("object")
        if column.dtype == object:
            numeric = _safe_float(column)
            if numeric.notna().sum() > 0 and numeric.nunique() > 1:
                column = pd.cut(
                    numeric,
                    bins=[-np.inf, 50, 70, np.inf] if feature == "age_years" else [-np.inf, np.median(numeric.dropna()), np.inf],
                    labels=["low", "mid", "high"] if feature == "age_years" else ["below_median", "above_median"],
                )
                column = column.astype("object")
        groups = column.astype("object").fillna("missing").astype(str)
        for group, idx in groups.groupby(groups).groups.items():
            idx_list = list(idx)
            subset_true = y_true[idx_list]
            subset_pred = y_pred[idx_list]
            if subset_true.size == 0:
                continue
            entry = {
                "feature": feature,
                "group": group,
                "count": len(idx_list),
            }
            entry["mse"] = float(mse(subset_true, subset_pred))
            entry["mae"] = float(mae(subset_true, subset_pred))
            rows.append(entry)
    return pd.DataFrame(rows)

