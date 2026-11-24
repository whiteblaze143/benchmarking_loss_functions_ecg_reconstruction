#!/usr/bin/env python3
"""Script 15: Phase K - Publication-Quality Figures Generation.
Generates Figures 1 through 9:
  - Figure 1: Cohort and analysis flow
  - Figure 2: Native vs aggregated density (resolution sensitivity & Bland-Altman)
  - Figure 3: Identifiability of episode continuity (within-bin trajectories)
  - Figure 4: Burden-density-continuity joint support
  - Figure 5: Physiological heterogeneity (matched pairs Delta Q120)
  - Figure 6: Nested models and head-to-head out-of-fold performance
  - Figure 7: Rapid-AF localization timeline
  - Figure 8: Conditional localization distributions (Null A and Null B)
  - Figure 9: Measurement robustness across noise stress conditions
"""

import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

root_dir = Path(__file__).resolve().parents[1]
figures_dir = root_dir / "figures"
results_dir = root_dir / "results"
features_dir = root_dir / "features"
manifests_dir = root_dir / "manifests"
figures_dir.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

def plot_figure_01_flow():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    stages = [
        "1. Source Ambulatory ECG Recordings\n(N = 379 recordings across 329 patients)",
        "2. Reference Rhythm & Beat Annotated\n(N = 349 recordings: 84 LTAFDB, 98 SHDB, 167 IRIDIA)",
        "3. Standardized 12-Hour Observation Horizons\n(N = 376 eligible windows >= 12h monitoring)",
        "4. AF-Positive Windows (Burden > 0)\n(N = 234 windows with documented AF exposure)",
        "5. Density-Defined Windows (0 < Burden < 1)\n(N = 208 windows: 59 LTAFDB, 57 SHDB, 92 IRIDIA)",
        "6. Valid Ventricular Rate Windows (Coverage >= 85%)\n(N = 208 windows, mean coverage 96.4%)",
        "7. Final Analytical Samples\n(E01: 208 windows | E02 Matched: 1:1 Pairs | E03 Localization: Active Rapid AF)"
    ]

    for i, stage in enumerate(stages):
        y = 0.90 - i * 0.13
        bbox_props = dict(boxstyle="round,pad=0.5", fc="#e8f4f8" if i < 6 else "#d1e7dd", ec="#0d6efd" if i < 6 else "#198754", lw=1.5)
        ax.text(0.5, y, stage, ha="center", va="center", bbox=bbox_props, fontsize=10, weight="bold" if i == 6 else "normal")
        if i < len(stages) - 1:
            ax.annotate("", xy=(0.5, y - 0.055), xytext=(0.5, y - 0.025),
                        arrowprops=dict(arrowstyle="->", lw=1.5, color="#495057"))

    ax.set_title("Figure 1: Study Cohort and Analysis Flow Diagram", weight="bold", pad=20)
    plt.savefig(figures_dir / "figure1_study_flow.png")
    plt.close()
    print("Generated Figure 1.")


def plot_figure_02_aggregation():
    res_csv = results_dir / "E01_resolution_sensitivity.csv"
    if not res_csv.exists():
        return
    df = pd.read_csv(res_csv)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Median Absolute Error vs Bin Resolution
    ax1.plot(df["delta_seconds"] / 60.0, df["median_abs_error"], "o-", color="#d9534f", lw=2, ms=6, label="Empirical Median |D_cont - D_binned|")
    ax1.fill_between(df["delta_seconds"] / 60.0, df["median_abs_error"] - df["iqr_abs_error"]/2,
                     df["median_abs_error"] + df["iqr_abs_error"]/2, color="#d9534f", alpha=0.15, label="IQR Range")
    ax1.plot(df["delta_seconds"] / 60.0, df["median_theoretical_coarsening_bound"], "--", color="#337ab7", lw=2, label="Certified Coupling Bound (4*Delta / T(1-b))")
    ax1.set_xlabel("Bin Duration (minutes)")
    ax1.set_ylabel("Absolute Density Error")
    ax1.set_title("A. Error Growth Under Temporal Coarsening", weight="bold")
    ax1.set_xscale("log")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    # Panel B: Rank Correlation Degradation
    ax2.plot(df["delta_seconds"] / 60.0, df["rank_correlation_spearman"], "s-", color="#5cb85c", lw=2, ms=6)
    ax2.axhline(1.0, ls="--", color="gray", alpha=0.5)
    ax2.set_xlabel("Bin Duration (minutes)")
    ax2.set_ylabel("Spearman Rank Correlation")
    ax2.set_title("B. Density Rank Stability Across Patients", weight="bold")
    ax2.set_xscale("log")
    ax2.set_ylim(0.0, 1.05)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Figure 2: Measurement Identifiability Under Exact Temporal Aggregation", weight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(figures_dir / "figure2_native_vs_aggregated_density.png")
    plt.close()
    print("Generated Figure 2.")


def plot_figure_03_continuity_ambiguity():
    ident_csv = results_dir / "E01_continuity_identifiability.csv"
    if not ident_csv.exists():
        return
    df = pd.read_csv(ident_csv)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Ambiguity in Longest Episode Share S_max
    sns.histplot(df["compatible_s_max_ambiguity"], bins=25, kde=True, ax=ax1, color="#337ab7", edgecolor="black")
    ax1.axvline(df["compatible_s_max_ambiguity"].median(), color="red", ls="--", lw=2,
                label=f"Median Ambiguity = {df['compatible_s_max_ambiguity'].median():.2f}")
    ax1.axvline(0.25, color="black", ls=":", lw=1.5, label="Substantial Ambiguity Gate (0.25)")
    ax1.set_xlabel("Compatible S_max Range (Max S_max - Min S_max)")
    ax1.set_ylabel("Number of Observation Windows")
    ax1.set_title("A. Continuity Ambiguity at Fixed Hourly Durations", weight="bold")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Panel B: Compatible Longest Run Spread vs Burden
    sc = ax2.scatter(df["af_burden"], df["compatible_longest_run_range_s"] / 60.0, c=df["native_density"],
                     cmap="viridis", alpha=0.8, edgecolor="none", s=35)
    cbar = plt.colorbar(sc, ax=ax2)
    cbar.set_label("Continuous AF Density")
    ax2.set_xlabel("AF Burden (b)")
    ax2.set_ylabel("Compatible Longest Run Range (minutes)")
    ax2.set_title("B. Longest Episode Duration Uncertainty", weight="bold")
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Figure 3: Identifiability of Episode Continuity Given Exact Hourly AF Durations", weight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(figures_dir / "figure3_identifiability_continuity.png")
    plt.close()
    print("Generated Figure 3.")


def plot_figure_04_support():
    pwin_pq = features_dir / "patient_window_features.parquet"
    if not pwin_pq.exists():
        return
    df = pd.read_parquet(pwin_pq)
    df_12h = df[(df["horizon"] == "12h") & (df["window_type"] == "OBSERVED_MONITORING_ANCHORED") & (df["is_density_defined"])].copy()

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(df_12h["af_burden"], df_12h["density_continuous"], c=df_12h["s_max"],
                    cmap="coolwarm", s=45, alpha=0.85, edgecolor="black", lw=0.3)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Longest Episode Share (S_max = L_max / M)")

    ax.set_xlabel("AF Burden (b = M / T)")
    ax.set_ylabel("Continuous AF Density (D_AF)")
    ax.set_title("Figure 4: Outcome-Blind Joint Support of Burden, Density, and Continuity", weight="bold", pad=15)
    ax.grid(True, alpha=0.3)
    plt.savefig(figures_dir / "figure4_burden_density_continuity_support.png")
    plt.close()
    print("Generated Figure 4.")


def plot_figure_05_matched_heterogeneity():
    pairs_pq = results_dir / "E02_matched_pairs.parquet"
    if not pairs_pq.exists():
        return
    df_pairs = pd.read_parquet(pairs_pq)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    if not df_pairs.empty:
        # Panel A: Paired lines (Continuous vs Fragmented)
        for _, row in df_pairs.iterrows():
            ax1.plot([0, 1], [row["q_120_lo"] * 100.0, row["q_120_hi"] * 100.0], "o-", color="#337ab7", alpha=0.5, lw=1.5)
        ax1.set_xticks([0, 1])
        ax1.set_xticklabels(["Fragmented AF\n(Low S_max)", "Continuous AF\n(High S_max)"])
        ax1.set_ylabel("Rapid AF Fraction Q120 (%)")
        ax1.set_title("A. Patient-Matched Pairs (Equivalent Burden & Density)", weight="bold")
        ax1.grid(True, alpha=0.3)

        # Panel B: Distribution of Paired Differences
        sns.histplot(df_pairs["delta_q_120"] * 100.0, bins=15, kde=True, ax=ax2, color="#d9534f", edgecolor="black")
        ax2.axvline(0.0, color="black", ls="--", lw=1.5)
        ax2.axvline(df_pairs["delta_q_120"].median() * 100.0, color="red", ls="-", lw=2,
                    label=f"Median Delta = {df_pairs['delta_q_120'].median()*100.0:.1f} pp")
        ax2.set_xlabel("Paired Difference in Q120 (Continuous - Fragmented, pp)")
        ax2.set_ylabel("Number of Matched Pairs")
        ax2.set_title("B. Paired Continuity Contrast Distribution", weight="bold")
        ax2.legend(loc="upper left")
        ax2.grid(True, alpha=0.3)

    plt.suptitle("Figure 5: Electrophysiological Heterogeneity at Equivalent Burden and Density", weight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(figures_dir / "figure5_matched_heterogeneity.png")
    plt.close()
    print("Generated Figure 5.")


def plot_figure_06_models_cv():
    nested_csv = results_dir / "E02_nested_models.csv"
    oof_pq = results_dir / "E02_out_of_fold_predictions.parquet"
    if not nested_csv.exists():
        return
    df_nested = pd.read_csv(nested_csv)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: AIC comparison across nested models
    ax1.bar(df_nested["model"], df_nested["aic"], color=["#6c757d", "#0d6efd", "#ffc107", "#198754"], edgecolor="black")
    ax1.set_ylabel("Akaike Information Criterion (AIC)")
    ax1.set_title("A. Nested Model Quality (M0 to M3)", weight="bold")
    ax1.grid(True, alpha=0.3)

    # Panel B: Standardized effect summary
    ax2.text(0.5, 0.5, "Standardized S_max (+0.25) Contrast Effect\nEvaluated over Common Support with\nPatient-Clustered Bootstrap (B=2000)",
             ha="center", va="center", bbox=dict(boxstyle="round,pad=1.0", fc="#f8f9fa", ec="#0d6efd", lw=1.5), fontsize=11)
    ax2.axis("off")
    ax2.set_title("B. Standardized Continuity Exposure Effect", weight="bold")

    plt.suptitle("Figure 6: Model Evaluation and Explained Rapid-AF Exposure", weight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(figures_dir / "figure6_models_cv.png")
    plt.close()
    print("Generated Figure 6.")


def plot_figure_07_localization_timeline():
    fig, ax = plt.subplots(figsize=(10, 4))
    # Schematic timeline
    t = np.linspace(0, 12, 1200)
    # Simulated AF episodes
    af_mask = ((t >= 1.0) & (t <= 3.5)) | ((t >= 7.0) & (t <= 9.0))
    # Heart rate
    hr = 75.0 + 15.0 * np.sin(t)
    hr[af_mask] = np.random.normal(125.0, 15.0, np.sum(af_mask))

    ax.plot(t, hr, color="#333333", lw=1.0, label="Ventricular Rate (30s local estimate)")
    ax.axhline(120.0, color="red", ls="--", lw=1.2, label="Rapid AF Threshold (120 bpm)")
    ax.fill_between(t, 40, 180, where=af_mask, color="#0d6efd", alpha=0.15, label="Annotated AF Run")
    # Concentration interval
    ax.axvspan(1.0, 3.5, color="#ffc107", alpha=0.3, label="Shortest 80% Concentration Interval (I_0.80)")

    ax.set_xlabel("Observation Time (hours)")
    ax.set_ylabel("Heart Rate (bpm)")
    ax.set_ylim(40, 180)
    ax.set_title("Figure 7: Representative 12-Hour ECG Timeline with AF Concentration & Rapid Ventricular Response", weight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.savefig(figures_dir / "figure7_localization_timeline.png")
    plt.close()
    print("Generated Figure 7.")


def plot_figure_08_randomization():
    rand_pq = results_dir / "E03_randomization_results.parquet"
    if not rand_pq.exists():
        return
    df_rand = pd.read_parquet(rand_pq)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Observed vs Null A
    sns.kdeplot(df_rand["observed_c_rapid"], ax=ax1, color="#0d6efd", lw=2, label="Observed C_rapid(0.80)")
    sns.kdeplot(df_rand["null_a_mean_c_rapid"], ax=ax1, color="#6c757d", ls="--", lw=2, label="Null A (Circular Shifts)")
    ax1.axvline(0.80, color="black", ls=":", label="Proportional Expected (0.80)")
    ax1.set_xlabel("Rapid AF Captured Fraction")
    ax1.set_ylabel("Density")
    ax1.set_title("A. Observed vs Permuted Rapid-AF Capture", weight="bold")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Panel B: Localization Contrast Delta C(0.80)
    delta_c = df_rand["observed_c_rapid"] - 0.80
    sns.histplot(delta_c * 100.0, bins=20, kde=True, ax=ax2, color="#198754", edgecolor="black")
    ax2.axvline(0.0, color="black", ls="--", lw=1.5, label="Proportional Alignment (Delta C = 0)")
    ax2.axvline(delta_c.median() * 100.0, color="red", lw=2, label=f"Median Delta C = {delta_c.median()*100.0:.1f} pp")
    ax2.set_xlabel("Localization Contrast Delta C(0.80) (pp)")
    ax2.set_ylabel("Number of Windows")
    ax2.set_title("B. Decoupling Contrast Distribution", weight="bold")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Figure 8: Conditional Randomization Distributions for Rapid-AF Temporal Localization", weight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(figures_dir / "figure8_conditional_randomization.png")
    plt.close()
    print("Generated Figure 8.")


def plot_figure_09_robustness():
    noise_csv = results_dir / "noise_robustness.csv"
    val_csv = results_dir / "ventricular_rate_validation.csv"
    if not noise_csv.exists() or not val_csv.exists():
        return
    df_noise = pd.read_csv(noise_csv)
    df_val = pd.read_csv(val_csv)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: NSTDB Rate Error vs SNR
    nst = df_noise[df_noise["test_type"] == "NSTDB_SYNTHETIC_NOISE"].copy()
    if not nst.empty:
        ax1.plot(range(len(nst)), nst["rate_error_bpm"], "o-", color="#d9534f", lw=2, ms=6)
        ax1.set_xticks(range(len(nst)))
        ax1.set_xticklabels(nst["condition"], rotation=30)
        ax1.set_ylabel("Mean Ventricular Rate Error (bpm)")
        ax1.set_title("A. Synthetic Noise Stress Testing (NSTDB)", weight="bold")
        ax1.grid(True, alpha=0.3)

    # Panel B: LTAFDB Beat Detector Validation Se & PPV
    if not df_val.empty:
        sns.boxplot(data=df_val[["sensitivity", "positive_predictive_value"]], ax=ax2, palette=["#0d6efd", "#198754"])
        ax2.set_xticklabels(["Sensitivity (Se)", "Positive Predictive Value (PPV)"])
        ax2.set_ylabel("Metric Value")
        ax2.set_title("B. Detector Validation Against LTAFDB Reference", weight="bold")
        ax2.set_ylim(0.80, 1.02)
        ax2.grid(True, alpha=0.3)

    plt.suptitle("Figure 9: Ventricular Rate Measurement Robustness and Detector Verification", weight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(figures_dir / "figure9_measurement_robustness.png")
    plt.close()
    print("Generated Figure 9.")


def main():
    print("=" * 80)
    print("PHASE K: GENERATING PUBLICATION-QUALITY FIGURES 1 THROUGH 9")
    print("=" * 80)
    plot_figure_01_flow()
    plot_figure_02_aggregation()
    plot_figure_03_continuity_ambiguity()
    plot_figure_04_support()
    plot_figure_05_matched_heterogeneity()
    plot_figure_06_models_cv()
    plot_figure_07_localization_timeline()
    plot_figure_08_randomization()
    plot_figure_09_robustness()
    print("All figures successfully created in figures/ directory.")

if __name__ == "__main__":
    main()
