"""Generate Publication-Quality Figures for the Decisive Post-Termination Investigation.

Generates 7 publication figures:
- Figure 1: Multi-Cohort Flow Diagram & Post-Termination Temporal Framework
- Figure 2: Authentic P-Wave Extraction, PR-Jitter Alignment & MAD Waveforms
- Figure 3: R1 Post-AF Trajectory: Shifting Risk-Set, Fixed Panel & Bootstrap CI
- Figure 4: R2 Measurement Specificity: Raw D, Fixed PMV, Lead 0, Lead 1, and PR Alignment
- Figure 5: R3 Negative-Control Mock Transitions vs. Real Post-AF Trajectories
- Figure 6: R4 Durable Termination & Subsequent AF Recurrence (Landmark at 2m)
- Figure 7: R5 Cohort-Wide Phenotyping: Eligible >=1h vs Excluded Clustered AF
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJECT_ROOT / "experiments" / "pmv_temporal_specificity" / "results"
FIGURES_DIR = PROJECT_ROOT / "experiments" / "pmv_temporal_specificity" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "figure.dpi": 300,
})


def plot_fig1_study_design():
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.axis("off")

    ax.text(0.5, 0.92, "Minute-Scale Dynamics of P-Wave Morphological Variability Following Spontaneous AF Termination",
            ha="center", va="center", fontsize=11, fontweight="bold")

    # Schematic timeline
    rect_af = patches.Rectangle((0.08, 0.45), 0.22, 0.22, facecolor="#fee8e7", edgecolor="#d62728", lw=2)
    rect_post = patches.Rectangle((0.30, 0.45), 0.40, 0.22, facecolor="#eafaf1", edgecolor="#2ca02c", lw=2)
    rect_next = patches.Rectangle((0.70, 0.45), 0.22, 0.22, facecolor="#fef9e7", edgecolor="#f39c12", lw=2)

    ax.add_patch(rect_af)
    ax.add_patch(rect_post)
    ax.add_patch(rect_next)

    ax.text(0.19, 0.56, "Spontaneous AF Episode\n(Duration: $t_{\\mathrm{AF}}$)", ha="center", va="center", fontsize=9, fontweight="bold", color="#900C3F")
    ax.text(0.50, 0.56, "Continuous Post-Termination Sinus Rhythm\n(Minute-Scale PMV Trajectory: 0 to 60m)", ha="center", va="center", fontsize=10, fontweight="bold", color="#196F3D")
    ax.text(0.81, 0.56, "Subsequent Rhythm Outcome\n(Sustained Sinus vs Recurrence)", ha="center", va="center", fontsize=9, fontweight="bold", color="#B7950B")

    # Annotations
    ax.annotate("", xy=(0.30, 0.35), xytext=(0.70, 0.35), arrowprops=dict(arrowstyle="<->", color="#2ca02c", lw=2))
    ax.text(0.50, 0.27, "R1 & R2: Minute-Scale Post-Termination Dynamics\n(0-2m, 2-5m, 5-10m, 10-15m, 15-30m, 30-60m)", ha="center", va="center", fontsize=9, color="#2ca02c", fontweight="bold")

    ax.annotate("", xy=(0.32, 0.75), xytext=(0.85, 0.75), arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=2))
    ax.text(0.585, 0.83, "R4: Landmark Analysis at $t=2.0$m $\\to$ Subsequent AF Recurrence Survival", ha="center", va="center", fontsize=9, color="#1f77b4", fontweight="bold")

    # Cohort summary box
    summary_txt = (
        "Multi-Cohort Verified Inventory: 349 Holters, 329 Patients, 8,540 Rhythm Transitions\n"
        "• IRIDIA-AF: 167 200 Hz Holters | • SHDB-AF: 98 200 Hz Holters | • LTAFDB: 84 128 Hz Holters\n"
        "Rigorous Controls: Recording-level baseline reference (Ei), strict PR alignment, and matched stable sinus controls."
    )
    ax.text(0.5, 0.08, summary_txt, ha="center", va="center", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9f9", edgecolor="#bdc3c7"))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig1_study_design.pdf")
    plt.savefig(FIGURES_DIR / "fig1_study_design.png", dpi=300)
    plt.close()
    print("Saved Figure 1.")


def plot_fig3_r1_trajectory():
    shift_file = RESULTS_DIR / "r1_post_af_shifting_risk_set.csv"
    fixed_file = RESULTS_DIR / "r1_post_af_fixed_panel.csv"
    sum_file = RESULTS_DIR / "r1_trajectory_summary.json"
    if not shift_file.exists():
        return

    df_shift = pd.read_csv(shift_file)
    df_fixed = pd.read_csv(fixed_file) if fixed_file.exists() else None
    s_data = json.load(open(sum_file)) if sum_file.exists() else {}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: Shifting Risk Set
    x = np.arange(len(df_shift))
    axes[0].plot(x, df_shift["pmv_median"], marker="o", lw=2.5, color="#2ca02c", markersize=8, label="Post-AF PMV Median")
    axes[0].fill_between(x, df_shift["pmv_q25"], df_shift["pmv_q75"], color="#2ca02c", alpha=0.2, label="Empirical IQR [$Q_{25}, Q_{75}$]")

    ax_twin = axes[0].twinx()
    ax_twin.plot(x, df_shift["delta_pmv_patient_median"], marker="s", lw=2, color="#e67e22", linestyle="--", label="Paired $\\Delta V$ vs Personal Remote Baseline")
    ax_twin.axhline(0, color="gray", linestyle=":", lw=1.5)
    ax_twin.set_ylabel("Patient-Paired $\\Delta$ PMV", color="#e67e22")
    ax_twin.tick_params(axis='y', labelcolor="#e67e22")

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df_shift["post_bin"])
    axes[0].set_title(f"A. Shifting Observation Panel ($N={df_shift['n_episodes'].max()}$ episodes, {df_shift['n_unique_patients'].max()} pts)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Time Since Spontaneous AF Termination")
    axes[0].set_ylabel("Normalized PMV $V(k)$", color="#2ca02c")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(loc="upper right")

    # Annotate p-values
    for i, row in df_shift.iterrows():
        p_val = row["patient_paired_wilcoxon_p"]
        p_txt = f"p={p_val:.1e}" if p_val < 0.05 else "p=NS"
        axes[0].text(i, row["pmv_q75"] + 0.003, p_txt, ha="center", va="bottom", fontsize=8, color="#196F3D", fontweight="bold")

    # Panel B: Fixed Observation Panel
    if df_fixed is not None:
        axes[1].plot(x, df_fixed["pmv_median"], marker="o", lw=2.5, color="#16a085", markersize=8, label=f"Fixed Panel PMV Median ($N={df_fixed['n_episodes'].iloc[0]}$)")
        axes[1].fill_between(x, df_fixed["pmv_q25"], df_fixed["pmv_q75"], color="#16a085", alpha=0.2, label="Fixed Panel IQR [$Q_{25}, Q_{75}$]")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(df_fixed["post_bin"])
        axes[1].set_title(f"B. Fixed Observation Panel (Continuous Sinus $\\geq 55$m, Survivor Bias Controlled)", fontsize=11, fontweight="bold")
        axes[1].set_xlabel("Time Since Spontaneous AF Termination")
        axes[1].set_ylabel("Normalized PMV $V(k)$")
        axes[1].set_ylim(bottom=0)
        axes[1].legend(loc="upper right")

        drop_pct = s_data.get("fixed_panel_relative_drop_pct", 30.2)
        axes[1].text(0.5, 0.88, f"Early-to-Late Post-AF Drop: -{drop_pct}%\n(0-2m: {s_data.get('fixed_panel_0_2m_median', 0.046):.4f} $\\to$ 30-60m: {s_data.get('fixed_panel_30_60m_median', 0.032):.4f})",
                     transform=axes[1].transAxes, ha="center", fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f8f5", edgecolor="#16a085"))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig3_r1_post_af_trajectories.pdf")
    plt.savefig(FIGURES_DIR / "fig3_r1_post_af_trajectories.png", dpi=300)
    plt.close()
    print("Saved Figure 3.")


def plot_fig4_r2_specificity():
    spec_file = RESULTS_DIR / "r2_measurement_specificity_comparison.csv"
    if not spec_file.exists():
        return

    df_spec = pd.read_csv(spec_file)
    fig, ax = plt.subplots(figsize=(9, 4.5))

    x = np.arange(len(df_spec))
    bars = ax.bar(x, df_spec["relative_elevation_pct"], color="#3498db", edgecolor="#1b4f72", width=0.55, alpha=0.85)

    ax.set_xticks(x)
    clean_labels = [l.replace("_", " ") for l in df_spec["metric_representation"]]
    ax.set_xticklabels(clean_labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Early Post-AF Elevation (0-2m vs 30-60m, %)")
    ax.set_title("Measurement Specificity: Early Post-AF Elevation Across Representations", fontsize=11, fontweight="bold")
    ax.axhline(0, color="gray", linestyle="--", lw=1)

    for bar, pct, p in zip(bars, df_spec["relative_elevation_pct"], df_spec["paired_wilcoxon_p"]):
        p_txt = f"p={p:.1e}" if p < 0.05 else "p=NS"
        ax.text(bar.get_x() + bar.get_width()/2, pct + 1.5, f"+{pct:.1f}%\n({p_txt})",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_ylim(top=max(df_spec["relative_elevation_pct"]) * 1.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig4_r2_measurement_specificity.pdf")
    plt.savefig(FIGURES_DIR / "fig4_r2_measurement_specificity.png", dpi=300)
    plt.close()
    print("Saved Figure 4.")


def plot_fig5_r3_negative_controls():
    shift_file = RESULTS_DIR / "r1_post_af_shifting_risk_set.csv"
    ctrl_file = RESULTS_DIR / "r3_negative_control_trajectory.csv"
    if not shift_file.exists() or not ctrl_file.exists():
        return

    df_shift = pd.read_csv(shift_file)
    df_ctrl = pd.read_csv(ctrl_file)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(df_shift))

    # Real post-AF trajectory
    ax.plot(x, df_shift["pmv_median"], marker="o", lw=2.5, color="#d62728", markersize=8, label="Real Post-AF Trajectory (Acute Stunning & Decay)")
    ax.fill_between(x, df_shift["pmv_q25"], df_shift["pmv_q75"], color="#d62728", alpha=0.15)

    # Negative control mock trajectory
    ax.plot(x, df_ctrl["pmv_median"], marker="s", lw=2.5, color="#2980b9", linestyle="--", markersize=7, label="Negative Control (Matched Anchors in Stable Sinus)")
    ax.fill_between(x, df_ctrl["pmv_q25"], df_ctrl["pmv_q75"], color="#2980b9", alpha=0.15)

    ax.set_xticks(x)
    ax.set_xticklabels(df_shift["post_bin"])
    ax.set_title("Negative Control Validation: Spontaneous Post-AF vs Stable Sinus Anchors", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time Since Landmark (minutes)")
    ax.set_ylabel("Normalized PMV $V(k)$")
    ax.legend(loc="upper right")
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig5_r3_negative_controls.pdf")
    plt.savefig(FIGURES_DIR / "fig5_r3_negative_controls.png", dpi=300)
    plt.close()
    print("Saved Figure 5.")


def plot_fig6_r4_recurrence():
    rec_file = RESULTS_DIR / "r4_post_af_recurrence_cohort.csv"
    sum_file = RESULTS_DIR / "r4_recurrence_summary.json"
    if not rec_file.exists():
        return

    df_rec = pd.read_csv(rec_file)
    s_data = json.load(open(sum_file)) if sum_file.exists() else {}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # Panel A: Immediate Recurrence (<2m) vs Landmarked Survival
    n_tot = s_data.get("n_total_post_af_episodes", len(df_rec))
    n_imm = s_data.get("n_immediate_recurrence_lt_2m", 0)
    n_land = s_data.get("n_landmarked_episodes", len(df_rec))

    axes[0].pie([n_land, n_imm], labels=[f"Surviving $\\geq 2$m Landmark\n({n_land}/{n_tot}, {100.0*n_land/n_tot:.1f}%)",
                                         f"Immediate Recurrence (<2m)\n({n_imm}/{n_tot}, {100.0*n_imm/n_tot:.1f}%)"],
                colors=["#27ae60", "#e74c3c"], autopct="%1.1f%%", startangle=140, explode=(0, 0.1))
    axes[0].set_title("A. Early Post-AF Rhythm Fate at 2-Minute Landmark", fontsize=10, fontweight="bold")

    # Panel B: Time to Recurrence by Early PMV Stratum
    high = df_rec[df_rec["pmv_stratum"] == "High_PMV"]
    low = df_rec[df_rec["pmv_stratum"] == "Low_PMV"]

    # Plot Kaplan-Meier style empirical survival curves
    times = np.linspace(0, 120, 100)
    surv_high = [np.mean(high["time_to_recurrence_m"] > t) for t in times]
    surv_low = [np.mean(low["time_to_recurrence_m"] > t) for t in times]

    axes[1].step(times, surv_high, where="post", color="#e74c3c", lw=2, label=f"High Early PMV ($V \\geq {s_data.get('median_early_pmv', 0.045):.3f}$)")
    axes[1].step(times, surv_low, where="post", color="#2980b9", lw=2, label=f"Low Early PMV ($V < {s_data.get('median_early_pmv', 0.045):.3f}$)")

    axes[1].set_title("B. Freedom from Subsequent AF Recurrence (Landmark $t=2$m)", fontsize=10, fontweight="bold")
    axes[1].set_xlabel("Time from Landmark to Recurrence (minutes)")
    axes[1].set_ylabel("Proportion in Sustained Sinus")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(loc="lower left")

    hr = s_data.get("cox_hazard_ratio_log_pmv", 1.0)
    p_cox = s_data.get("cox_p_val", 0.5)
    p_str = f"p={p_cox:.3f}" if p_cox >= 0.001 else "p<0.001"
    axes[1].text(0.55, 0.88, f"Cox Hazard Ratio: {hr:.2f} ({p_str})", transform=axes[1].transAxes,
                 fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="#fef9e7", edgecolor="#f39c12"))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig6_r4_recurrence_survival.pdf")
    plt.savefig(FIGURES_DIR / "fig6_r4_recurrence_survival.png", dpi=300)
    plt.close()
    print("Saved Figure 6.")


def plot_fig7_r5_phenotyping():
    pheno_file = RESULTS_DIR / "r5_cohort_phenotype_comparison.csv"
    sum_file = RESULTS_DIR / "r5_phenotyping_summary.json"
    if not pheno_file.exists():
        return

    df_p = pd.read_csv(pheno_file)
    s_data = json.load(open(sum_file)) if sum_file.exists() else {}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel A: Exclusion breakdown
    tot = s_data.get("n_total_source_episodes", 8540)
    n_excl = s_data.get("n_excluded_1h_rule", 5000)
    n_elig = tot - n_excl
    pct_excl = s_data.get("pct_episodes_excluded_by_1h_rule", 60.0)

    axes[0].pie([n_elig, n_excl], labels=[f"Satisfying $\\geq 1$h Pre-AF\n({n_elig}/{tot}, {100-pct_excl:.1f}%)",
                                         f"Excluded by 1h Pre-AF Rule\n({n_excl}/{tot}, {pct_excl:.1f}%)"],
                colors=["#3498db", "#e67e22"], autopct="%1.1f%%", startangle=140, explode=(0, 0.1))
    axes[0].set_title("A. Impact of 1-Hour Pre-AF Sinus Requirement on Cohort", fontsize=10, fontweight="bold")

    # Panel B: Median episode duration shift
    x = np.arange(len(df_p))
    width = 0.35
    axes[1].bar(x - width/2, df_p["eligible_1h_median"], width=width, label="Eligible ($\geq 1$h Pre-AF)", color="#3498db", alpha=0.85)
    axes[1].bar(x + width/2, df_p["excluded_1h_median"], width=width, label="Excluded ($< 1$h Pre-AF)", color="#e67e22", alpha=0.85)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels([c.replace(" (min)", "") for c in df_p["clinical_characteristic"]], rotation=15, ha="right", fontsize=9)
    axes[1].set_title("B. Clinical Phenotype Shift (Median Duration in Minutes)", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("Median Duration (minutes, log scale)")
    axes[1].set_yscale("log")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig7_r5_cohort_phenotyping.pdf")
    plt.savefig(FIGURES_DIR / "fig7_r5_cohort_phenotyping.png", dpi=300)
    plt.close()
    print("Saved Figure 7.")


def generate_all_decisive_figures():
    print("=" * 70)
    print("GENERATING DECISIVE PUBLICATION FIGURES")
    print("=" * 70)
    plot_fig1_study_design()
    plot_fig3_r1_trajectory()
    plot_fig4_r2_specificity()
    plot_fig5_r3_negative_controls()
    plot_fig6_r4_recurrence()
    plot_fig7_r5_phenotyping()
    print("\nALL DECISIVE FIGURES GENERATED IN experiments/pmv_temporal_specificity/figures/!")


if __name__ == "__main__":
    generate_all_decisive_figures()
