"""Beat-level Bland-Altman analysis for ECG fiducial features.

This module consumes beat-level fiducial measurements (real vs reconstructed)
for a single lead and produces Bland-Altman statistics and plots, with a focus
on regression-to-mean effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import RANSACRegressor


@dataclass
class FiducialBlandAltman:
    bias: float
    loa_lower: float
    loa_upper: float
    slope: float
    slope_ci: Tuple[float, float]
    robust_slope: float
    n: int


def bland_altman_for_fiducial(real: np.ndarray, recon: np.ndarray) -> FiducialBlandAltman:
    """Compute Bland-Altman stats and regression-to-mean slope for one fiducial."""

    mask = np.isfinite(real) & np.isfinite(recon)
    real = real[mask]
    recon = recon[mask]
    if real.size == 0:
        return FiducialBlandAltman(np.nan, np.nan, np.nan, np.nan, (np.nan, np.nan), np.nan, 0)

    mean_vals = 0.5 * (real + recon)
    errors = recon - real

    bias = float(errors.mean())
    sd = float(errors.std(ddof=1))
    loa_lower = bias - 1.96 * sd
    loa_upper = bias + 1.96 * sd

    # Simple OLS: error = a + b * mean
    X = np.vstack([np.ones_like(mean_vals), mean_vals]).T
    beta, *_ = np.linalg.lstsq(X, errors, rcond=None)
    slope = float(beta[1])

    # Approximate 95% CI for slope
    y_hat = X @ beta
    resid = errors - y_hat
    s2 = float((resid**2).sum() / (len(mean_vals) - 2))
    xtx_inv = np.linalg.inv(X.T @ X)
    se_slope = float(np.sqrt(s2 * xtx_inv[1, 1]))
    z = 1.96
    slope_ci = (slope - z * se_slope, slope + z * se_slope)

    # Robust Regression (RANSAC)
    try:
        ransac = RANSACRegressor(random_state=42)
        # RANSAC expects 2D X
        X_ransac = mean_vals.reshape(-1, 1)
        ransac.fit(X_ransac, errors)
        robust_slope = float(ransac.estimator_.coef_[0])
    except Exception:
        robust_slope = float("nan")

    return FiducialBlandAltman(
        bias=bias,
        loa_lower=loa_lower,
        loa_upper=loa_upper,
        slope=slope,
        slope_ci=slope_ci,
        robust_slope=robust_slope,
        n=len(mean_vals),
    )


def compute_icc(errors: np.ndarray, patient_ids: np.ndarray) -> float:
    """Compute a simple ICC for beat-level errors clustered by patient."""

    mask = np.isfinite(errors)
    errors = errors[mask]
    patient_ids = patient_ids[mask]
    if errors.size == 0:
        return float("nan")

    overall_mean = errors.mean()
    patients = np.unique(patient_ids)
    n_pat = len(patients)
    n_total = len(errors)
    if n_pat < 2:
        return float("nan")

    ss_between = 0.0
    ss_within = 0.0
    for pid in patients:
        idx = patient_ids == pid
        group = errors[idx]
        if group.size == 0:
            continue
        g_mean = group.mean()
        ss_between += float(((g_mean - overall_mean) ** 2) * group.size)
        ss_within += float(((group - g_mean) ** 2).sum())

    ms_between = ss_between / max(1, (n_pat - 1))
    ms_within = ss_within / max(1, (n_total - n_pat))
    if ms_between + ms_within == 0:
        return float("nan")
    icc = (ms_between - ms_within) / (ms_between + ms_within)
    return float(icc)


def plot_bland_altman(
    mean_vals: np.ndarray,
    errors: np.ndarray,
    stats: FiducialBlandAltman,
    title: str,
    save_path: Path,
) -> None:
    """Create a Bland-Altman scatter plot with regression line and LoA."""

    mask = np.isfinite(mean_vals) & np.isfinite(errors)
    mean_vals = mean_vals[mask]
    errors = errors[mask]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(mean_vals, errors, s=10, alpha=0.4, edgecolor="none")

    if stats.n > 0 and np.isfinite(stats.slope):
        x_line = np.linspace(mean_vals.min(), mean_vals.max(), 100)
        # Fit line: error = a + b * mean; recompute intercept from bias and slope
        # Note: This intercept calculation assumes the line passes through (mean(mean_vals), mean(errors)) which is true for OLS
        a = stats.bias - stats.slope * float(mean_vals.mean())
        y_line = a + stats.slope * x_line
        ax.plot(x_line, y_line, color="C1", label=f"slope={stats.slope:.3f}")

        # Robust line
        if np.isfinite(stats.robust_slope):
             # We need an intercept for the robust line. RANSACRegressor fits y = Xw + b.
             # We didn't save the intercept in the dataclass.
             # Let's just re-fit here for plotting or assume it passes close to the median?
             # Better to just plot it with a generic intercept or re-fit.
             # Re-fitting is cheap for plotting.
             try:
                ransac = RANSACRegressor(random_state=42)
                X_ransac = mean_vals.reshape(-1, 1)
                ransac.fit(X_ransac, errors)
                y_robust = ransac.predict(x_line.reshape(-1, 1))
                ax.plot(x_line, y_robust, color="C3", linestyle="-.", label=f"Robust slope={stats.robust_slope:.3f}")
             except:
                pass

    ax.axhline(stats.bias, color="C2", linestyle="--", label=f"bias={stats.bias:.3f}")
    ax.axhline(stats.loa_lower, color="grey", linestyle=":")
    ax.axhline(stats.loa_upper, color="grey", linestyle=":")

    ax.set_xlabel("Mean amplitude (mV)")
    ax.set_ylabel("Reconstructed − Real (mV)")
    # Annotate basic statistics on the plot for quick reading
    if stats.n > 0 and np.isfinite(stats.slope):
        text_lines = [
            f"n = {stats.n}",
            f"bias = {stats.bias:.3f} mV",
            f"LoA = [{stats.loa_lower:.3f}, {stats.loa_upper:.3f}]",
            f"slope = {stats.slope:.3f}",
            f"Robust slope = {stats.robust_slope:.3f}",
            f"95% CI = [{stats.slope_ci[0]:.3f}, {stats.slope_ci[1]:.3f}]",
        ]
        ax.text(
            0.02,
            0.98,
            "\n".join(text_lines),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def summarize_fiducials(df: pd.DataFrame, fiducials: Iterable[str]) -> pd.DataFrame:
    """Build a summary table of Bland-Altman stats for selected fiducials.

    Expects columns `real_<fid>`, `recon_<fid>` in df.
    """

    rows: List[Dict[str, object]] = []
    for fid in fiducials:
        real = df[f"real_{fid}"].to_numpy(dtype=float)
        recon = df[f"recon_{fid}"].to_numpy(dtype=float)
        ba = bland_altman_for_fiducial(real, recon)
        mean_vals = 0.5 * (real + recon)
        errors = recon - real
        rows.append(
            {
                "fiducial": fid,
                "mean_error_mv": ba.bias,
                "slope": ba.slope,
                "robust_slope": ba.robust_slope,
                "slope_ci_low": ba.slope_ci[0],
                "slope_ci_high": ba.slope_ci[1],
                "loa_lower_mv": ba.loa_lower,
                "loa_upper_mv": ba.loa_upper,
                "n_beats": ba.n,
                "mean_amplitude_mv": float(np.mean(np.abs(real))),
            }
        )
    return pd.DataFrame(rows)


def summarize_fiducials(df: pd.DataFrame, fiducials: Iterable[str]) -> pd.DataFrame:
    """Build a summary table of Bland-Altman stats for selected fiducials.

    Expects columns `real_<fid>`, `recon_<fid>` in df.
    """

    rows: List[Dict[str, object]] = []
    for fid in fiducials:
        real = df[f"real_{fid}"].to_numpy(dtype=float)
        recon = df[f"recon_{fid}"].to_numpy(dtype=float)
        ba = bland_altman_for_fiducial(real, recon)
        mean_vals = 0.5 * (real + recon)
        errors = recon - real
        rows.append(
            {
                "fiducial": fid,
                "mean_error_mv": ba.bias,
                "slope": ba.slope,
                "robust_slope": ba.robust_slope,
                "slope_ci_low": ba.slope_ci[0],
                "slope_ci_high": ba.slope_ci[1],
                "loa_lower_mv": ba.loa_lower,
                "loa_upper_mv": ba.loa_upper,
                "n_beats": ba.n,
                "mean_amplitude_mv": float(np.mean(np.abs(real))),
            }
        )
    return pd.DataFrame(rows)
