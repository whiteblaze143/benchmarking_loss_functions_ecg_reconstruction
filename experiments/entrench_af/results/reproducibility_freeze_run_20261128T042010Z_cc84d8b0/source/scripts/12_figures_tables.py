#!/usr/bin/env python3
"""
Stage 12 — Publication-Quality Figures (300 DPI PNG + PDF) and Consolidated Tables.

Uses standardized cardiac electrophysiology and survival-analysis terminology
across all axes, titles, and legends.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import common  # type: ignore

PALETTE = {
    "IRIDIA": "#0072B2",
    "SHDB": "#D55E00",
    "LTAFDB": "#009E73",
    "POOLED": "#2B2B2B",
    "SPITALER": "#AA3377",
}

COHORT_DISPLAY = {
    "IRIDIA": "IRIDIA-AF",
    "SHDB": "SHDB-AF",
    "LTAFDB": "LTAFDB",
    "POOLED": "Pooled Holter",
}

PREDICTOR_DISPLAY = {
    "log_dur": "Log AF Episode Duration",
    "fwave_SampEn_mean": "Terminal f-Wave Sample Entropy",
    "AFR_DF_mean": "Atrial Fibrillatory Rate (Mean)",
    "AFR_DF_slope": "Atrial Fibrillatory Rate (Slope)",
    "VCC": "Ventricular Cycle-Length Coherence",
    "RR_mean": "Mean Ventricular R–R Interval",
    "PMV_A": "Post-Termination PMV (0–120 s)",
    "PMV_D": "Post-Termination PMV (10–12 min)",
    "first_PAC_coupling_ms": "First PAC Coupling Interval",
    "cum_af_sec_before_t0": "Prior Cumulative AF Burden",
    "n_prior_terminations": "Prior Episode Terminations",
    "clock_sin": "Circadian Phase (sin 2πt/24)",
    "clock_cos": "Circadian Phase (cos 2πt/24)",
}

STRATUM_DISPLAY = {
    "AFR_DF_mean:T1_low": "AFR Mean — Tertile 1 (Low)",
    "AFR_DF_mean:T2_mid": "AFR Mean — Tertile 2 (Mid)",
    "AFR_DF_mean:T3_high": "AFR Mean — Tertile 3 (High)",
    "PMV_A:T1_low": "PMV (0–120 s) — Tertile 1 (Low)",
    "PMV_A:T2_mid": "PMV (0–120 s) — Tertile 2 (Mid)",
    "PMV_A:T3_high": "PMV (0–120 s) — Tertile 3 (High)",
}

SQI_MARKER_DISPLAY = {
    "AFR_DF": "Atrial Fibrillatory Rate (Hz)",
    "PAC_coupling": "First PAC Coupling Interval (ms)",
    "PMV": "P-Wave Morphological Variability",
    "RMSSD": "RMSSD of R–R Intervals (ms)",
    "RR_SampEn": "R–R Interval Sample Entropy",
    "VCC": "Ventricular Cycle Coherence (%)",
    "fwave_SampEn": "f-Wave Sample Entropy",
}

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def save_fig(fig: plt.Figure, fig_dir: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(fig_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_f1_aim1(run_dir: Path, fig_dir: Path) -> None:
    p = run_dir / "aim1" / "hazard_profile.csv"
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    if p.exists():
        df = pd.read_csv(p)
        for cohort in ["IRIDIA", "SHDB", "LTAFDB", "POOLED"]:
            sub = df[df["cohort"] == cohort]
            color = PALETTE[cohort]
            cname = COHORT_DISPLAY[cohort]
            na = sub[sub["estimator"] == "nelson_aalen_kernel"].sort_values("mid_sec")
            if not na.empty:
                na_ok = na[~na["suppressed_lt20"].astype(bool)] if "suppressed_lt20" in na.columns else na
                if not na_ok.empty:
                    ax.plot(
                        na_ok["mid_sec"], na_ok["hazard_per_hour"],
                        label=f"{cname} (Nelson–Aalen kernel ± 95% CI)",
                        color=color, linewidth=1.9,
                    )
                    if "hazard_ci_lo" in na_ok.columns and "hazard_ci_hi" in na_ok.columns:
                        lo_h = na_ok["hazard_ci_lo"].to_numpy() * 3600.0
                        hi_h = na_ok["hazard_ci_hi"].to_numpy() * 3600.0
                        ax.fill_between(
                            na_ok["mid_sec"], np.clip(lo_h, 1e-4, None), np.clip(hi_h, 1e-4, None),
                            alpha=0.12, color=color,
                        )
            mk = sub[sub["estimator"] == "piecewise_exp_frailty"].sort_values("mid_sec")
            if not mk.empty:
                mk_ok = (
                    mk[(~mk["suppressed_lt20"].astype(bool)) & (mk["hazard_per_hour"] > 0)]
                    if "suppressed_lt20" in mk.columns
                    else mk[mk["hazard_per_hour"] > 0]
                )
                if not mk_ok.empty:
                    ax.scatter(mk_ok["mid_sec"], mk_ok["hazard_per_hour"], s=22, marker="o", color=color, alpha=0.88)

        sp_ref = df[(df["cohort"] == "SPITALER_REF") & (df["mid_sec"] <= 172800.0)].sort_values("mid_sec")
        if not sp_ref.empty:
            sp_x = sp_ref["mid_sec"].to_numpy(dtype=float)
            sp_y = sp_ref["hazard_per_hour"].to_numpy(dtype=float)
            sp_lo = sp_ref["hazard_ci_lo"].to_numpy(dtype=float) * 3600.0
            sp_hi = sp_ref["hazard_ci_hi"].to_numpy(dtype=float) * 3600.0
            ax.plot(sp_x, sp_y, linestyle="--", linewidth=1.6, color=PALETTE["SPITALER"], zorder=5)
            ax.errorbar(
                sp_x, sp_y,
                yerr=[np.clip(sp_y - sp_lo, 0, None), np.clip(sp_hi - sp_y, 0, None)],
                fmt="D", markersize=6.0, color=PALETTE["SPITALER"], ecolor=PALETTE["SPITALER"],
                capsize=2.5, zorder=6,
                label="Spitaler et al. (2026) Pacemaker DDAF (1–48 h)",
            )

        handles, labels = ax.get_legend_handles_labels()
        handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor="#555555", markersize=5.5))
        labels.append("Piecewise-exponential shared gamma frailty (per bin)")
        ax.legend(handles, labels, frameon=False, fontsize=8.0, loc="upper right")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ticks = [30, 60, 120, 300, 600, 1800, 3600, 7200, 21600, 43200, 86400, 172800]
        labels_x = ["30 s", "1 m", "2 m", "5 m", "10 m", "30 m", "1 h", "2 h", "6 h", "12 h", "24 h", "48 h"]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels_x)
        ax.set_xlabel("Elapsed AF Episode Duration")
        ax.set_ylabel("Spontaneous AF Termination Hazard (h⁻¹)")
        ax.set_title("AF Termination Hazard Trajectory Across Holter Cohorts vs. Pacemaker Registry")
        ax.grid(True, which="both", alpha=0.2)
    save_fig(fig, fig_dir, "F1_aim1_hazard_spitaler")


def plot_f2_aim2(run_dir: Path, fig_dir: Path) -> None:
    p = run_dir / "aim2" / "table2_reconstruction_metrics.csv"
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4))
    if p.exists():
        df = pd.read_csv(p)
        if "gap_sec" in df.columns:
            df = df[df["gap_sec"] == 0].copy()
        order = ["IRIDIA", "SHDB", "LTAFDB", "POOLED"]
        df["cohort"] = pd.Categorical(df["cohort"].astype(str), categories=order, ordered=True)
        df = df.sort_values("cohort")
        labels = [COHORT_DISPLAY.get(str(c), str(c)) for c in df["cohort"]]
        x = np.arange(len(labels))

        true_col = next((c for c in ["n_true_runs_uncensored", "n_true_runs_all", "n_true_runs"] if c in df.columns), None)
        rec_col = next((c for c in ["n_recon_episodes", "n_reconstructed"] if c in df.columns), None)
        if true_col and rec_col:
            axes[0].bar(x - 0.19, df[true_col], width=0.36, label="Beat-level AF runs (g = 0 s)", color="#0072B2")
            axes[0].bar(x + 0.19, df[rec_col], width=0.36, label="Daily-burden reconstructed episodes", color="#D55E00")
            for idx_r, row in enumerate(df.itertuples()):
                axes[0].text(idx_r - 0.19, getattr(row, true_col) * 1.12, f"{int(getattr(row, true_col)):,}", ha="center", va="bottom", fontsize=7.5)
                axes[0].text(idx_r + 0.19, getattr(row, rec_col) * 1.12, f"{int(getattr(row, rec_col)):,}", ha="center", va="bottom", fontsize=7.5)
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(labels)
            axes[0].set_yscale("log")
            axes[0].set_ylim(bottom=1, top=float(df[true_col].max()) * 3.2)
            axes[0].set_ylabel("Episode Count (log scale)")
            axes[0].set_title("A. Beat-Level vs. Reconstructed Episode Counts")
            axes[0].legend(frameon=False, loc="upper left", fontsize=8.0)
            axes[0].grid(True, axis="y", alpha=0.2)

        int_col = "mean_internal_terminations" if "mean_internal_terminations" in df.columns else None
        if int_col:
            axes[1].bar(x, df[int_col], width=0.55, color="#009E73")
            for idx_r, val in enumerate(df[int_col]):
                axes[1].text(idx_r, float(val) + 1.0, f"{float(val):.1f}", ha="center", va="bottom", fontsize=8.0)
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(labels)
            axes[1].set_ylim(0, float(df[int_col].max()) * 1.18)
            axes[1].set_ylabel("Mean Concealed Terminations per Episode")
            axes[1].set_title("B. Internal AF Terminations Masked by Consolidation")
            axes[1].grid(True, axis="y", alpha=0.2)
    fig.suptitle("Daily-Burden Episode Reconstruction vs. Beat-Level Rhythm Transitions", y=1.02)
    save_fig(fig, fig_dir, "F2_aim2_reconstruction")


def plot_f3_aim3(run_dir: Path, fig_dir: Path) -> None:
    p_hr = run_dir / "aim3" / "table3_primary_hrs.csv"
    p_km = run_dir / "aim3" / "km_reinit_curves.parquet"
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0))
    if p_hr.exists():
        df = pd.read_csv(p_hr)
        sub = df[df["model"] == "primary_marker_eligible_complete_case"].copy()
        if sub.empty:
            sub = df[df["model"] == "marker_eligible_449_pac_count"].copy()
        if not sub.empty:
            sub = sub.sort_values("hr_per_sd")
            y = np.arange(len(sub))
            ylabels = [PREDICTOR_DISPLAY.get(str(p_name), str(p_name)) for p_name in sub["predictor"]]
            axes[0].errorbar(
                sub["hr_per_sd"], y,
                xerr=[sub["hr_per_sd"] - sub["hr_ci_lo"], sub["hr_ci_hi"] - sub["hr_per_sd"]],
                fmt="o", color="#0072B2", ecolor="#555555", capsize=3,
            )
            axes[0].axvline(1.0, linestyle="--", color="gray", linewidth=1)
            axes[0].set_yticks(y)
            axes[0].set_yticklabels(ylabels, fontsize=8.5)
            axes[0].set_xlabel("Adjusted Hazard Ratio per SD (95% CI)")
            axes[0].set_title(f"A. Complete-Case Stratified Cox Model (n = {int(sub['n_events'].iloc[0])})")
            axes[0].grid(True, axis="x", alpha=0.25)
    if p_km.exists():
        km = pd.read_parquet(p_km)
        sub_km = km[km["stratum"].astype(str).str.startswith("PMV_A:") | km["stratum"].astype(str).str.startswith("AFR_DF_mean:")]
        for label, g in sub_km.groupby("stratum"):
            g = g.sort_values("time_sec")
            disp = STRATUM_DISPLAY.get(str(label), str(label))
            axes[1].plot(g["time_sec"] / 60.0, g["reinit_prob"], label=disp, linewidth=1.6)
        axes[1].set_xlim(0, 60)
        axes[1].set_xlabel("Time Since AF Termination (min)")
        axes[1].set_ylabel("Cumulative AF Reinitiation Probability")
        axes[1].set_title("B. Reinitiation Incidence by Marker Tertile (AF ≥ 300 s)")
        axes[1].legend(frameon=False, fontsize=7.8, loc="lower right")
        axes[1].grid(True, alpha=0.25)
    fig.suptitle("Electrophysiological and Clinical Predictors of Post-Termination AF Reinitiation", y=1.02)
    save_fig(fig, fig_dir, "F3_aim3_forest_km")


def plot_f4_decisive_contrast(run_dir: Path, fig_dir: Path) -> None:
    p = run_dir / "aim3" / "decisive_contrast.csv"
    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    if p.exists():
        df = pd.read_csv(p)
        if not df.empty:
            order = ["IRIDIA", "SHDB", "LTAFDB", "POOLED"]
            df["cohort"] = pd.Categorical(df["cohort"].astype(str), categories=order, ordered=True)
            df = df.sort_values("cohort")
            labels = [COHORT_DISPLAY.get(str(c), str(c)) for c in df["cohort"]]
            x = np.arange(len(labels))
            chain = df["share_reinit_chain_360s"].to_numpy() * 100.0
            unint = 100.0 - chain
            ax.bar(x, chain, width=0.58, label="Early reinitiation chains (<360-s gaps)", color="#D55E00")
            ax.bar(x, unint, bottom=chain, width=0.58, label="Sustained single-episode AF", color="#0072B2")
            yerr_lo = np.clip(chain - df["ci_lo"].to_numpy() * 100.0, 0, None)
            yerr_hi = np.clip(df["ci_hi"].to_numpy() * 100.0 - chain, 0, None)
            ax.errorbar(x, chain, yerr=[yerr_lo, yerr_hi], fmt="none", ecolor="black", capsize=4)
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_ylim(0, 108)
            ax.set_ylabel("Share of ≥1-h Reconstructed AF Time (%)")
            ax.set_title("Consolidation vs. Early Reinitiation Chains in ≥1-h Reconstructed Episodes", pad=26)
            ax.legend(
                frameon=False,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.09),
                ncol=2,
                fontsize=8.2,
            )
            ax.grid(True, axis="y", alpha=0.2)
    save_fig(fig, fig_dir, "F4_decisive_contrast")


def plot_f5_aim4(run_dir: Path, fig_dir: Path) -> None:
    p = run_dir / "aim4" / "ccdf_data.parquet"
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6), sharex=True, sharey=True)
    if p.exists():
        df = pd.read_parquet(p)
        sub_beat = df[(df["dataset"] == "runs_g0") & (df["filter"] == "all")]
        sub_recon = df[(df["dataset"] == "spitaler_reconstructed") & (df["filter"] == "all")]
        for ax, sub, panel_title in [
            (axes[0], sub_beat, "A. Beat-Level AF Runs (g = 0 s)"),
            (axes[1], sub_recon, "B. Daily-Burden Reconstructed Episodes"),
        ]:
            for cohort in ["IRIDIA", "SHDB", "LTAFDB", "POOLED"]:
                g = sub[sub["cohort"] == cohort].sort_values("dur_sec")
                if not g.empty:
                    ax.loglog(
                        g["dur_sec"], g["ccdf"],
                        label=COHORT_DISPLAY[cohort],
                        color=PALETTE[cohort],
                        linewidth=1.8,
                        alpha=0.9,
                    )
            ax.set_title(panel_title)
            ax.set_xlabel("Episode Duration (s)")
            ax.grid(True, which="both", alpha=0.2)
        axes[0].set_ylabel("Complementary CDF, P(D ≥ d)")
        axes[0].legend(frameon=False, loc="lower left", fontsize=8.2)
        axes[1].legend(frameon=False, loc="lower left", fontsize=8.2)
    fig.suptitle("Empirical Episode-Duration Distributions: Beat-Level vs. Daily-Burden Reconstruction", y=1.02)
    save_fig(fig, fig_dir, "F5_aim4_ccdf_loglog")


def plot_s1_sqi(run_dir: Path, fig_dir: Path) -> None:
    p = run_dir / "stage3" / "sqi_calibration_detail.csv"
    if p.exists():
        df = pd.read_csv(p)
        markers = sorted(df["marker"].dropna().unique())
        ncols = 3
        nrows = int(np.ceil(len(markers) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(11.2, 3.2 * nrows))
        axes = np.atleast_1d(axes).ravel()
        for ax, m in zip(axes, markers):
            sub = df[df["marker"] == m].dropna(subset=["bsqi", "abs_error"]).copy()
            sub["bsqi_bin"] = pd.cut(sub["bsqi"], bins=np.linspace(0, 1, 11))
            med = sub.groupby("bsqi_bin", observed=True).agg(bsqi_mid=("bsqi", "mean"), med_err=("abs_error", "median")).dropna()
            ax.scatter(sub["bsqi"], sub["abs_error"], s=6, alpha=0.15, color="#0072B2")
            ax.plot(med["bsqi_mid"], med["med_err"], color="#D55E00", marker="o", linewidth=1.8, label="Bin median")
            ax.set_title(SQI_MARKER_DISPLAY.get(str(m), str(m)), fontsize=9.5)
            ax.set_xlabel("Beat-Detection SQI (bSQI)")
            ax.set_ylabel("Absolute Error")
            ax.grid(True, alpha=0.2)
        for ax in axes[len(markers):]:
            ax.axis("off")
        fig.suptitle("Signal-Quality Calibration: Electrophysiological Marker Error Across bSQI Strata", y=1.01)
    else:
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        ax.text(0.5, 0.5, "Stage 3 sqi_calibration_detail.csv not found", ha="center", va="center")
    save_fig(fig, fig_dir, "S1_sqi_calibration")


def plot_s2_gap_sensitivity(run_dir: Path, fig_dir: Path) -> None:
    p = run_dir / "aim2" / "table2_reconstruction_metrics.csv"
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    if p.exists():
        df = pd.read_csv(p)
        if not df.empty and "gap_sec" in df.columns:
            gaps = sorted(df["gap_sec"].unique())
            pos_map = {g: i for i, g in enumerate(gaps)}
            xt = np.arange(len(gaps))
            xt_labels = [f"{int(g):,}" for g in gaps]
            for cohort in ["IRIDIA", "SHDB", "LTAFDB", "POOLED"]:
                g = df[df["cohort"] == cohort].sort_values("gap_sec")
                if g.empty:
                    continue
                x = [pos_map[val] for val in g["gap_sec"]]
                axes[0].plot(
                    x, g["n_true_runs_uncensored"], marker="o",
                    label=COHORT_DISPLAY[cohort], color=PALETTE[cohort], linewidth=1.8,
                )
                axes[1].plot(
                    x, g["true_median_dur_sec"], marker="o",
                    label=COHORT_DISPLAY[cohort], color=PALETTE[cohort], linewidth=1.8,
                )
            axes[0].set_yscale("log")
            axes[0].set_xticks(xt)
            axes[0].set_xticklabels(xt_labels)
            axes[0].set_xlabel("Inter-Run Sinus-Gap Merge Threshold, g (s)")
            axes[0].set_ylabel("Uncensored AF Runs (log scale)")
            axes[0].set_title("A. Harmonized AF Run Count vs. Gap Threshold")
            axes[0].legend(frameon=False, fontsize=8.0)
            axes[0].grid(True, which="both", alpha=0.2)

            axes[1].set_yscale("log")
            axes[1].set_xticks(xt)
            axes[1].set_xticklabels(xt_labels)
            axes[1].set_xlabel("Inter-Run Sinus-Gap Merge Threshold, g (s)")
            axes[1].set_ylabel("Median AF Run Duration (s, log scale)")
            axes[1].set_title("B. Median AF Run Duration vs. Gap Threshold")
            axes[1].legend(frameon=False, fontsize=8.0)
            axes[1].grid(True, which="both", alpha=0.2)
    fig.suptitle("Sensitivity of Episode Counts and Duration to Sinus-Rhythm Gap Merging", y=1.02)
    save_fig(fig, fig_dir, "S2_gap_sensitivity")


def plot_s3_loco(run_dir: Path, fig_dir: Path) -> None:
    p = run_dir / "aim3" / "loco_calibration_bins.csv"
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    if p.exists():
        df = pd.read_csv(p)
        if not df.empty:
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ideal calibration")
            for cohort, g in df.groupby("held_out_cohort"):
                g = g.sort_values("pred_risk_360s")
                cname = COHORT_DISPLAY.get(str(cohort).upper(), str(cohort))
                ax.plot(
                    g["pred_risk_360s"], g["obs_risk_360s"], marker="o",
                    label=f"Held-out cohort: {cname}",
                    color=PALETTE.get(str(cohort).upper(), None),
                    linewidth=1.8,
                )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Predicted 360-s AF Reinitiation Probability")
            ax.set_ylabel("Observed Kaplan–Meier 360-s Reinitiation Probability")
            ax.set_title("Leave-One-Cohort-Out (LOCO) Calibration (Complete-Case Model)")
            ax.legend(frameon=False, loc="upper left")
            ax.grid(True, alpha=0.25)
    save_fig(fig, fig_dir, "S3_loco_calibration")


def plot_s4_weibull_min_run(run_dir: Path, fig_dir: Path) -> None:
    p = run_dir / "aim1" / "weibull_shape_vs_min_run.csv"
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    if p.exists():
        df = pd.read_csv(p)
        if not df.empty:
            xcol = next((c for c in ["min_run_sec", "min_dur_sec", "threshold_sec"] if c in df.columns), None)
            y_lt = next((c for c in ["shape_k_left_trunc", "shape_k"] if c in df.columns), None)
            lo_lt = next((c for c in ["shape_ci_lo_left_trunc", "shape_ci_low"] if c in df.columns), None)
            hi_lt = next((c for c in ["shape_ci_hi_left_trunc", "shape_ci_high"] if c in df.columns), None)
            y_sh = next((c for c in ["shape_k_shifted_origin", "untruncated_shape_k"] if c in df.columns), None)
            if xcol and y_lt:
                for cohort in ["IRIDIA", "SHDB", "LTAFDB", "POOLED"]:
                    g = df[df["cohort"] == cohort].sort_values(xcol)
                    if g.empty:
                        continue
                    x = g[xcol].to_numpy()
                    color = PALETTE[cohort]
                    cname = COHORT_DISPLAY[cohort]
                    ax.plot(x, g[y_lt].to_numpy(), marker="o", linestyle="-", label=f"{cname} (left-truncated)", color=color)
                    if lo_lt and hi_lt:
                        ax.fill_between(x, g[lo_lt].to_numpy(), g[hi_lt].to_numpy(), alpha=0.11, color=color)
                    if y_sh and g[y_sh].notna().any():
                        ax.plot(x, g[y_sh].to_numpy(), marker="x", linestyle="--", alpha=0.72, label=f"{cname} (shifted-origin)", color=color)
                ax.axhline(0.767, linestyle="--", color=PALETTE["SPITALER"], label="Spitaler et al. (2026) reference (k = 0.767)")
                ax.axhline(1.0, linestyle=":", color="gray", label="Memoryless exponential hazard (k = 1.0)")
                ax.set_xlabel("Minimum AF Run Inclusion Threshold (s)")
                ax.set_ylabel("Shared-Gamma-Frailty Weibull Shape, k (95% CI)")
                ax.set_title("Frailty-Adjusted Weibull Shape Across Minimum Run Duration Filters")
                ax.legend(frameon=False, fontsize=7.2, ncol=2, loc="upper left")
                ax.grid(True, alpha=0.25)
    save_fig(fig, fig_dir, "S4_weibull_shape_vs_min_run")


def plot_hrs_principal_figure(run_dir: Path, fig_dir: Path) -> None:
    """
    Principal 3-panel HRS 2027 figure aligned cleanly with the focused study thesis:
      A. AF Exposure vs. Uninterrupted Episode Duration (Schematic).
      B. Daily Aggregation Shifts Inferred Weibull Duration Dependence (model-implied
         normalized hazard curves & sensitivity forest annotation with 95% CIs).
      C. Clinical Duration-Threshold Disagreement (>=6 h) across All Eligible Windows
         and >=95% / >=99% Known-Rhythm Coverage Subsets.
    """
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.4))

    # Panel A: Schematic — Cumulative AF Exposure vs. Uninterrupted Run
    ax_a = axes[0]
    frag_blocks = [
        (1.0, 0.7), (2.0, 0.6), (3.1, 0.8), (4.6, 0.6), (6.0, 0.75),
        (7.5, 0.65), (9.2, 0.7), (11.0, 0.8), (13.2, 0.6), (15.5, 0.5),
        (18.0, 0.5), (20.2, 0.4),
    ]
    ax_a.add_patch(Rectangle((0.0, 1.15), 24.0, 0.44, facecolor="#F3F5F8", edgecolor="#C4CCD6", linewidth=0.9))
    for st, du in frag_blocks:
        ax_a.add_patch(Rectangle((st, 1.17), du, 0.40, facecolor="#D55E00", edgecolor="none", alpha=0.88))
    ax_a.text(0.1, 1.82, "Fragmented PAF (Rapid Reinitiation)", fontsize=9.1, fontweight="bold", color="#222222")
    ax_a.text(0.1, 1.65, "Daily A_W = 7.6 h (≥6 h)   |   Max L_W = 0.8 h (<6 h)", fontsize=8.0, color="#A83C00")

    ax_a.add_patch(Rectangle((0.0, 0.22), 24.0, 0.44, facecolor="#F3F5F8", edgecolor="#C4CCD6", linewidth=0.9))
    ax_a.add_patch(Rectangle((4.0, 0.24), 7.6, 0.40, facecolor="#0072B2", edgecolor="none", alpha=0.88))
    ax_a.text(0.1, 0.89, "Continuous Sustained Episode", fontsize=9.1, fontweight="bold", color="#222222")
    ax_a.text(0.1, 0.72, "Daily A_W = 7.6 h (≥6 h)   |   Max L_W = 7.6 h (≥6 h)", fontsize=8.0, color="#00558A")

    ax_a.annotate(
        "Same daily-burden episode under ≥6 min/day rule\n(conceals 11 spontaneous terminations in top track)",
        xy=(12.0, 1.04), xytext=(12.0, -0.06), ha="center", va="top", fontsize=7.9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFF8EB", edgecolor="#D8B365", linewidth=0.9),
    )
    ax_a.set_xlim(-0.3, 24.3)
    ax_a.set_ylim(-0.38, 2.02)
    ax_a.set_xticks([0, 6, 12, 18, 24])
    ax_a.set_xticklabels(["0 h", "6 h", "12 h", "18 h", "24 h"], fontsize=8.5)
    ax_a.set_yticks([])
    for sp in ("top", "right", "left"):
        ax_a.spines[sp].set_visible(False)
    ax_a.set_xlabel("24-h Observation Window")
    ax_a.set_title("A. Cumulative AF Exposure vs. Uninterrupted Run\n(Schematic Example; Not an Actual Recording)", fontsize=9.8)

    # Panel B: Model-Implied Weibull Hazard Shapes & Sensitivity Delta-k
    ax_b = axes[1]
    pb_path = run_dir / "aim2" / "paired_aggregation_bootstrap.csv"
    cov_agg_path = run_dir / "aim2" / "coverage_sensitivity_aggregation.csv"
    if pb_path.exists():
        pb = pd.read_csv(pb_path)
        pool_row = pb[(pb["rhythm_tier"] == "harmonized_annotation") & (pb["frame"] == "recording_24h_windows") & (pb["cohort"] == "POOLED")].iloc[0]
        kn = float(pool_row["native_frailty_k"])
        kr = float(pool_row["recon_frailty_k"])
        dk = float(pool_row["delta_frailty_k"])
        dk_lo = float(pool_row["delta_frailty_k_ci_lo"])
        dk_hi = float(pool_row["delta_frailty_k_ci_hi"])

        t_grid = np.geomspace(30.0, 86400.0, 200)
        ax_b.plot(t_grid, (t_grid / 3600.0) ** (kn - 1.0), color="#0072B2", linewidth=2.4, label=f"Harmonized Native (k = {kn:.3f})")
        ax_b.plot(t_grid, (t_grid / 3600.0) ** (kr - 1.0), color="#D55E00", linewidth=2.4, linestyle="--", label=f"Daily Aggregation (k = {kr:.3f})")
        ax_b.axhline(1.0, color="#666666", linestyle=":", linewidth=1.0, label="Constant Hazard Reference (k = 1)")

        lines = [
            f"Pooled Δk = {dk:+.3f} (95% CI {dk_lo:+.3f} to {dk_hi:+.3f})",
            f"Unfrailty Δk = {float(pool_row['delta_unfrailty_k']):+.3f} ({float(pool_row['delta_unfrailty_k_ci_lo']):+.3f} to {float(pool_row['delta_unfrailty_k_ci_hi']):+.3f})",
        ]
        if cov_agg_path.exists():
            ca = pd.read_csv(cov_agg_path)
            for cf, clbl in [("cov_ge_95pct", "≥95% coverage"), ("cov_ge_99pct", "≥99% coverage")]:
                sr = ca[(ca["coverage_filter"] == cf) & (ca["cohort"] == "POOLED")]
                if not sr.empty:
                    r0 = sr.iloc[0]
                    lines.append(f"{clbl}: Δk = {float(r0['delta_frailty_k']):+.3f} ({float(r0['delta_frailty_k_ci_lo']):+.3f} to {float(r0['delta_frailty_k_ci_hi']):+.3f})")
        for tkey, tlbl in [("continuity_nonsinus_gaps", "Non-sinus gaps"), ("harmonized_annotation_min360s_lt", "≥6-min min dur")]:
            sr = pb[(pb["rhythm_tier"] == tkey) & (pb["cohort"] == "POOLED")]
            if not sr.empty:
                r0 = sr.iloc[0]
                lines.append(f"{tlbl}: Δk = {float(r0['delta_frailty_k']):+.3f} ({float(r0['delta_frailty_k_ci_lo']):+.3f} to {float(r0['delta_frailty_k_ci_hi']):+.3f})")
        wc_r = pb[(pb["rhythm_tier"] == "wall_clock_spitaler") & (pb["cohort"] == "IRIDIA")]
        if not wc_r.empty:
            r0 = wc_r.iloc[0]
            lines.append(f"IRIDIA calendar-day: Δk = {float(r0['delta_frailty_k']):+.3f} ({float(r0['delta_frailty_k_ci_lo']):+.3f} to {float(r0['delta_frailty_k_ci_hi']):+.3f})")

        ax_b.text(
            0.03, 0.04, "\n".join(lines), transform=ax_b.transAxes,
            fontsize=7.1, va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8FAFC", edgecolor="#CBD5E1", alpha=0.95),
        )

    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xticks([30, 300, 1800, 3600, 21600, 86400])
    ax_b.set_xticklabels(["30 s", "5 m", "30 m", "1 h", "6 h", "24 h"], fontsize=8.4)
    ax_b.set_xlabel("Duration Scale (Log)")
    ax_b.set_ylabel("Model-Implied Normalized Hazard h(t) / h(1 h)")
    ax_b.set_title("B. Aggregation Changes Duration Dependence\n(Model-Implied Normalized Weibull Shapes)", fontsize=9.8)
    ax_b.legend(frameon=False, fontsize=7.8, loc="upper right")
    ax_b.grid(True, which="both", alpha=0.2)

    # Panel C: >=6-h Clinical Threshold Disagreement Across Coverage Tiers
    ax_c = axes[2]
    cov_td_path = run_dir / "aim2" / "coverage_sensitivity_threshold.csv"
    if cov_td_path.exists():
        ctd = pd.read_csv(cov_td_path)
        sub6 = ctd[(ctd["rhythm_tier"] == "harmonized_annotation") & (ctd["threshold_sec"] == 21600)].copy()
        s_order = ["IRIDIA", "SHDB_PAF", "LTAFDB", "POOLED"]
        s_labels = ["IRIDIA-AF\n(Primary PAF)", "SHDB-AF\n(PAF)", "LTAFDB\n(Supporting)", "Pooled"]
        cov_specs = [
            ("all", "All Eligible Windows", "#2B4C7E"),
            ("known_cov_ge_95pct", "≥95% Known Rhythm", "#009E73"),
            ("known_cov_ge_99pct", "≥99% Known Rhythm", "#E69F00"),
        ]
        x = np.arange(len(s_order))
        w = 0.25
        for c_i, (cf, clbl, col) in enumerate(cov_specs):
            cs = sub6[sub6["coverage_filter"] == cf].set_index("stratum").reindex(s_order)
            vals = cs["prop_daily_ge_T_disagree_win"].to_numpy(float) * 100.0
            lo = cs["prop_daily_ge_T_disagree_win_ci_lo"].to_numpy(float) * 100.0
            hi = cs["prop_daily_ge_T_disagree_win_ci_hi"].to_numpy(float) * 100.0
            xpos = x + (c_i - 1) * w
            ax_c.bar(xpos, np.nan_to_num(vals, nan=0.0), width=w, color=col, label=clbl, alpha=0.92)
            ax_c.errorbar(
                xpos, np.nan_to_num(vals, nan=0.0),
                yerr=[np.clip(np.nan_to_num(vals - lo, nan=0.0), 0, None), np.clip(np.nan_to_num(hi - vals, nan=0.0), 0, None)],
                fmt="none", ecolor="#222222", capsize=2.0, linewidth=0.85,
            )
            if cf == "all":
                for xi, rrow in enumerate(cs.itertuples()):
                    if pd.notna(rrow.prop_daily_ge_T_disagree_win):
                        v_pct = float(rrow.prop_daily_ge_T_disagree_win) * 100.0
                        hi_pct = float(rrow.prop_daily_ge_T_disagree_win_ci_hi) * 100.0 if pd.notna(rrow.prop_daily_ge_T_disagree_win_ci_hi) else v_pct
                        ax_c.text(xpos[xi], hi_pct + 1.5, f"{v_pct:.1f}%\n({int(rrow.n_windows_disagree)}/{int(rrow.n_windows_daily_ge_T)})", ha="center", va="bottom", fontsize=6.8)
        ax_c.set_xticks(x)
        ax_c.set_xticklabels(s_labels, fontsize=8.2)
        ax_c.set_ylim(0, 52)
    ax_c.set_xlabel("Cohort")
    ax_c.set_ylabel("Windows With Daily ≥6 h Lacking ≥6-h Run (%)")
    ax_c.set_title("C. Clinical Duration-Threshold Disagreement (≥6 h)\n(Verified on ≥95% and ≥99% Known-Rhythm Windows)", fontsize=9.8)
    ax_c.legend(frameon=False, fontsize=7.6, loc="upper left")
    ax_c.grid(True, axis="y", alpha=0.2)

    fig.suptitle(
        "Temporal Aggregation Alters Inferred Atrial Fibrillation Termination Dynamics Across Holter Cohorts",
        fontsize=11.8, fontweight="bold", y=1.03,
    )
    save_fig(fig, fig_dir, "HRS_Principal_Figure")


def consolidate_tables(run_dir: Path, tab_dir: Path) -> None:
    stale_s2 = tab_dir / "S2_changepoint_lrt.csv"
    if stale_s2.exists():
        stale_s2.unlink()
    mapping = {
        "T0_v5_reconciliation.csv": run_dir / "stage2" / "reconciliation_v5_to_g0.csv",
        "T1_conditional_probs.csv": run_dir / "aim1" / "table1_conditional_probs.csv",
        "T1_short_horizons.csv": run_dir / "aim1" / "table1_short_horizons.csv",
        "T1_onset_censoring_audit.csv": run_dir / "aim1" / "onset_censoring_audit.csv",
        "T1_model_comparison_frailty.csv": run_dir / "aim1" / "model_comparison_frailty.csv",
        "T1_cohort_hierarchy_summary.csv": run_dir / "aim1" / "cohort_hierarchy_summary.csv",
        "T1b_spitaler_grid.csv": run_dir / "aim1" / "table1_spitaler_grid.csv",
        "T1c_hazard_profile.csv": run_dir / "aim1" / "hazard_profile.csv",
        "T1d_delayed_entry_verification.csv": run_dir / "aim1" / "delayed_entry_likelihood_verification.csv",
        "T2_reconstruction_metrics.csv": run_dir / "aim2" / "table2_reconstruction_metrics.csv",
        "T2b_native_to_recon_reconciliation.csv": run_dir / "aim2" / "native_to_recon_reconciliation.csv",
        "T2c_matched_support_hazard_comparison.csv": run_dir / "aim2" / "matched_support_hazard_comparison.csv",
        "T2d_threshold_disagreement.csv": run_dir / "aim2" / "threshold_disagreement.csv",
        "T2e_threshold_disagreement_examples.csv": run_dir / "aim2" / "threshold_disagreement_examples.csv",
        "T2f_matched_eligibility_reconciliation.csv": run_dir / "aim2" / "matched_eligibility_reconciliation.csv",
        "T2g_paired_aggregation_bootstrap.csv": run_dir / "aim2" / "paired_aggregation_bootstrap.csv",
        "T2h_coverage_sensitivity_aggregation.csv": run_dir / "aim2" / "coverage_sensitivity_aggregation.csv",
        "T2i_coverage_sensitivity_threshold.csv": run_dir / "aim2" / "coverage_sensitivity_threshold.csv",
        "T2j_bootstrap_identifiability_audit.csv": run_dir / "aim2" / "bootstrap_identifiability_audit.csv",
        "T2k_matched_eligibility_run_ledger.csv": run_dir / "aim2" / "matched_eligibility_run_ledger.csv",
        "T3_primary_hrs.csv": run_dir / "aim3" / "table3_primary_hrs.csv",
        "T3b_eligibility_flow.csv": run_dir / "aim3" / "eligibility_flow.csv",
        "T3c_within_between_decomposition.csv": run_dir / "aim3" / "within_between_decomposition.csv",
        "T3d_pmva_specification_curve.csv": run_dir / "aim3" / "pmva_specification_curve.csv",
        "T3e_descriptive_reinit_km.csv": run_dir / "aim3" / "descriptive_reinit_km.csv",
        "T4_duration_distributions.csv": run_dir / "aim4" / "table4_duration_distributions.csv",
        "S1_weibull_frailty.csv": run_dir / "aim1" / "tableS1_weibull_frailty.csv",
        "S3_loco_validation.csv": run_dir / "aim3" / "tableS3_loco_validation.csv",
        "S4_decisive_contrast.csv": run_dir / "aim3" / "decisive_contrast.csv",
        "S5_two_state_summary.csv": run_dir / "aim3" / "two_state_summary.csv",
        "S6_shdb_subgroups.csv": run_dir / "secondary" / "shdb_subgroups.csv",
        "S7_iridia_repeat_icc.csv": run_dir / "secondary" / "iridia_repeat_icc.csv",
        "S7b_iridia_repeat_visits_audit.csv": run_dir / "secondary" / "iridia_repeat_visits_audit.csv",
        "S8_circadian_analysis.csv": run_dir / "secondary" / "circadian_analysis.csv",
        "S9_collinearity_vif.csv": run_dir / "aim3" / "collinearity_vif.csv",
        "S10_weibull_shape_vs_min_run.csv": run_dir / "aim1" / "weibull_shape_vs_min_run.csv",
        "S11_shdb_non_af_audit.csv": run_dir / "secondary" / "shdb_non_af_with_af.csv",
    }
    for dest_name, src in mapping.items():
        if src.exists():
            shutil.copy2(src, tab_dir / dest_name)


def main() -> None:
    run_dir = common.resolve_run_dir()
    fig_dir = PROJECT_ROOT / "reports" / "figures"
    tab_dir = PROJECT_ROOT / "reports" / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    plot_hrs_principal_figure(run_dir, fig_dir)
    plot_f1_aim1(run_dir, fig_dir)
    plot_f2_aim2(run_dir, fig_dir)
    plot_f3_aim3(run_dir, fig_dir)
    plot_f4_decisive_contrast(run_dir, fig_dir)
    plot_f5_aim4(run_dir, fig_dir)
    plot_s1_sqi(run_dir, fig_dir)
    plot_s2_gap_sensitivity(run_dir, fig_dir)
    plot_s3_loco(run_dir, fig_dir)
    plot_s4_weibull_min_run(run_dir, fig_dir)
    consolidate_tables(run_dir, tab_dir)
    print(f"[Figures & Tables] Revision complete. Figures in {fig_dir}, Tables in {tab_dir}")


if __name__ == "__main__":
    main()
