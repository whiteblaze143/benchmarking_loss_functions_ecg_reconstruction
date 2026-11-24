"""Module for Rosman et al. (2026) methodological replication, phenotype categorization,
observation-horizon migration, factorial resolution analysis, and comparative metric stability.
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# Rosman 2026 Published Cutoffs
ROSMAN_TIERS = ["Low", "Medium", "Medium-High", "High"]
ROSMAN_EDGES = [0.0, 0.30, 0.60, 0.90, 1.0]

# Published AF-positive person-month marginal density distribution from Rosman et al. (2026)
ROSMAN_PUBLISHED_DENSITY_DIST = {
    "Low": 0.125,
    "Medium": 0.231,
    "Medium-High": 0.279,
    "High": 0.365
}


def assign_rosman_tier(value: float) -> str:
    """Assigns Rosman 4-tier category: Low (>0 to 0.3), Medium (>0.3 to 0.6),
    Medium-High (>0.6 to 0.9), High (>0.9 to 1.0).
    Returns 'Undefined' if value is NaN or <= 0.0 or > 1.0.
    """
    if pd.isna(value) or value <= 0.0 or value > 1.0:
        return "Undefined"
    if value <= 0.30:
        return "Low"
    elif value <= 0.60:
        return "Medium"
    elif value <= 0.90:
        return "Medium-High"
    else:
        return "High"


def compute_rosman_phenotype_matrix(
    df: pd.DataFrame,
    burden_col: str = "af_burden",
    density_col: str = "density_continuous",
    group_col: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Generates the exact 4 x 4 Rosman burden-density category matrix.
    If group_col is provided (e.g. 'patient_id'), aggregates to unique patients first.
    Returns: (contingency_table_df, summary_diagnostics)
    """
    data = df.copy()
    if group_col is not None:
        # Use median burden and density per patient for patient-level matrix
        data = data.groupby(group_col).agg({
            burden_col: "median",
            density_col: "median"
        }).reset_index()

    n_total = len(data)

    # Classify
    data["burden_tier"] = data[burden_col].apply(assign_rosman_tier)
    data["density_tier"] = data[density_col].apply(assign_rosman_tier)

    # Separate undefined
    undefined_mask = (data["burden_tier"] == "Undefined") | (data["density_tier"] == "Undefined")
    n_undefined = int(undefined_mask.sum())
    valid_data = data[~undefined_mask].copy()

    # Create 4x4 matrix
    matrix = pd.crosstab(
        valid_data["burden_tier"],
        valid_data["density_tier"],
        rownames=["Burden"],
        colnames=["Density"]
    )

    # Reindex to ensure all 4x4 rows and cols are present in order
    matrix = matrix.reindex(index=ROSMAN_TIERS, columns=ROSMAN_TIERS, fill_value=0)

    # Add margins
    matrix_with_totals = matrix.copy()
    matrix_with_totals["Total"] = matrix_with_totals.sum(axis=1)
    tot_row = matrix_with_totals.sum(axis=0)
    tot_row.name = "Total"
    matrix_with_totals = pd.concat([matrix_with_totals, pd.DataFrame([tot_row])])

    # Percentage matrix
    n_valid = len(valid_data)
    pct_matrix = (matrix / n_valid * 100.0) if n_valid > 0 else matrix.copy()

    # Compare marginal density distribution with published Rosman CIED distribution
    density_marginals = matrix.sum(axis=0) / n_valid if n_valid > 0 else pd.Series(0.0, index=ROSMAN_TIERS)
    density_comparison = []
    for tier in ROSMAN_TIERS:
        obs_p = float(density_marginals.get(tier, 0.0))
        pub_p = float(ROSMAN_PUBLISHED_DENSITY_DIST.get(tier, 0.0))
        density_comparison.append({
            "tier": tier,
            "observed_fraction": obs_p,
            "published_fraction": pub_p,
            "diff_percentage_points": (obs_p - pub_p) * 100.0
        })

    diagnostics = {
        "n_total_instances": n_total,
        "n_valid_joint_instances": n_valid,
        "n_undefined_instances": n_undefined,
        "undefined_fraction": n_undefined / n_total if n_total > 0 else 0.0,
        "density_comparison": density_comparison,
        "matrix_counts": matrix.to_dict(),
        "matrix_percentages": pct_matrix.to_dict()
    }

    return matrix_with_totals, diagnostics


def compute_weighted_cohen_kappa(
    y_true: List[str],
    y_pred: List[str],
    categories: List[str] = ROSMAN_TIERS,
    weight: str = "linear"
) -> float:
    """Computes weighted Cohen's kappa for ordered categorical tiers.
    weight: 'linear' or 'quadratic'.
    """
    n_cat = len(categories)
    cat_to_int = {c: i for i, c in enumerate(categories)}

    # Filter to pairs where both are valid categories
    valid_pairs = [(cat_to_int[t], cat_to_int[p]) for t, p in zip(y_true, y_pred) if t in cat_to_int and p in cat_to_int]
    if len(valid_pairs) == 0:
        return 0.0

    n = len(valid_pairs)
    # Observed matrix
    o_mat = np.zeros((n_cat, n_cat), dtype=float)
    for t_i, p_i in valid_pairs:
        o_mat[t_i, p_i] += 1.0
    o_mat /= n

    # Expected matrix
    row_sums = o_mat.sum(axis=1)
    col_sums = o_mat.sum(axis=0)
    e_mat = np.outer(row_sums, col_sums)

    # Weight matrix
    w_mat = np.zeros((n_cat, n_cat), dtype=float)
    for i in range(n_cat):
        for j in range(n_cat):
            if weight == "linear":
                w_mat[i, j] = 1.0 - abs(i - j) / (n_cat - 1)
            elif weight == "quadratic":
                w_mat[i, j] = 1.0 - ((i - j) ** 2) / ((n_cat - 1) ** 2)
            else:
                w_mat[i, j] = 1.0 if i == j else 0.0

    po = np.sum(w_mat * o_mat)
    pe = np.sum(w_mat * e_mat)

    if (1.0 - pe) == 0:
        return 1.0
    kappa = (po - pe) / (1.0 - pe)
    return float(kappa)


def compute_rank_stability(
    series_short: pd.Series,
    series_long: pd.Series
) -> Tuple[float, float]:
    """Computes Spearman rank correlation and p-value between two horizons for the same subjects.
    """
    mask = series_short.notna() & series_long.notna()
    if mask.sum() < 5:
        return np.nan, np.nan
    rho, pval = spearmanr(series_short[mask], series_long[mask])
    return float(rho), float(pval)


def compute_intraclass_correlation(df_measurements: pd.DataFrame) -> float:
    """Computes two-way mixed single-score ICC(3,1) consistency across horizons.
    df_measurements: shape (n_patients, n_horizons).
    """
    # Drop rows with any NaN
    clean = df_measurements.dropna()
    if len(clean) < 5 or clean.shape[1] < 2:
        return np.nan

    n, k = clean.shape
    data = clean.values

    # Mean squares
    row_means = data.mean(axis=1)
    col_means = data.mean(axis=0)
    grand_mean = data.mean()

    ss_total = np.sum((data - grand_mean) ** 2)
    ss_rows = k * np.sum((row_means - grand_mean) ** 2)
    ss_cols = n * np.sum((col_means - grand_mean) ** 2)
    ss_error = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1)
    ms_error = ss_error / ((n - 1) * (k - 1)) if (n - 1) * (k - 1) > 0 else 1e-6

    icc = (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error) if (ms_rows + (k - 1) * ms_error) > 0 else 0.0
    return float(icc)
