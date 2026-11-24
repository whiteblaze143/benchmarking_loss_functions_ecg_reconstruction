"""Deterministic patient matching and paired contrast analysis for AF-DENSITY V5.
Matches pairs within the same cohort and horizon with:
  |Delta Burden| <= 0.025
  |Delta Density| <= 0.025
  |Delta S_max| >= 0.30
  Different patients
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd


def find_matched_pairs(
    df_candidates: pd.DataFrame,
    burden_caliper: float = 0.025,
    density_caliper: float = 0.025,
    min_smax_contrast: float = 0.30,
    seed: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Performs deterministic 1:1 optimal/greedy matching between continuous and fragmented windows.
    Returns: (df_pairs, matching_diagnostics)
    """
    df = df_candidates.copy()
    # Require density and burden defined
    df = df[
        df["af_burden"].notna() &
        df["density_continuous"].notna() &
        df["s_max"].notna() &
        df["patient_id"].notna()
    ].reset_index(drop=True)

    used_patients = set()
    pairs = []

    # Sort deterministically by burden then density
    df = df.sort_values(["af_burden", "density_continuous", "s_max"]).reset_index(drop=True)

    # For each high-continuity candidate (S_max >= 0.50)
    high_candidates = df[df["s_max"] >= 0.50].index.tolist()

    for idx_hi in high_candidates:
        row_hi = df.loc[idx_hi]
        pat_hi = row_hi["patient_id"]
        if pat_hi in used_patients:
            continue

        b_hi = float(row_hi["af_burden"])
        d_hi = float(row_hi["density_continuous"])
        s_hi = float(row_hi["s_max"])

        # Find best candidate with s_lo <= s_hi - min_smax_contrast
        eligible = df[
            (df.index != idx_hi) &
            (~df["patient_id"].isin(used_patients)) &
            (df["patient_id"] != pat_hi) &
            (df["dataset"] == row_hi["dataset"]) &
            (df["horizon"] == row_hi["horizon"]) &
            ((df["af_burden"] - b_hi).abs() <= burden_caliper) &
            ((df["density_continuous"] - d_hi).abs() <= density_caliper) &
            (df["s_max"] <= s_hi - min_smax_contrast)
        ]

        if eligible.empty:
            continue

        # Choose nearest neighbor on Mahalanobis / Euclidean distance of (burden, density)
        dist = np.sqrt(
            ((eligible["af_burden"] - b_hi) / burden_caliper) ** 2 +
            ((eligible["density_continuous"] - d_hi) / density_caliper) ** 2
        )
        best_idx = dist.idxmin()
        row_lo = df.loc[best_idx]
        pat_lo = row_lo["patient_id"]

        used_patients.add(pat_hi)
        used_patients.add(pat_lo)

        delta_smax = s_hi - float(row_lo["s_max"])
        delta_q120 = float(row_hi.get("q_120", np.nan)) - float(row_lo.get("q_120", np.nan))

        pairs.append({
            "dataset": row_hi["dataset"],
            "horizon": row_hi["horizon"],
            "patient_hi": pat_hi,
            "record_hi": row_hi["record_id"],
            "patient_lo": pat_lo,
            "record_lo": row_lo["record_id"],
            "burden_hi": b_hi,
            "burden_lo": float(row_lo["af_burden"]),
            "density_hi": d_hi,
            "density_lo": float(row_lo["density_continuous"]),
            "s_max_hi": s_hi,
            "s_max_lo": float(row_lo["s_max"]),
            "q_120_hi": float(row_hi.get("q_120", np.nan)),
            "q_120_lo": float(row_lo.get("q_120", np.nan)),
            "delta_burden": abs(b_hi - float(row_lo["af_burden"])),
            "delta_density": abs(d_hi - float(row_lo["density_continuous"])),
            "delta_s_max": delta_smax,
            "delta_q_120": delta_q120,
            "diff_exceeds_10pct": abs(delta_q120) >= 0.10 if np.isfinite(delta_q120) else False,
            "diff_exceeds_20pct": abs(delta_q120) >= 0.20 if np.isfinite(delta_q120) else False
        })

    df_pairs = pd.DataFrame(pairs)
    diag = {
        "n_eligible_candidates": len(df),
        "n_matched_pairs": len(df_pairs),
        "n_unique_patients_matched": len(used_patients),
        "mean_delta_burden": float(df_pairs["delta_burden"].mean()) if not df_pairs.empty else 0.0,
        "mean_delta_density": float(df_pairs["delta_density"].mean()) if not df_pairs.empty else 0.0,
        "median_s_max_contrast": float(df_pairs["delta_s_max"].median()) if not df_pairs.empty else 0.0
    }
    return df_pairs, diag
