#!/home/mithunmanivannan/.venv/bin/python3
"""Analyze the temporal MMD evaluation results from the factorial grid.

Generates box plots and aggregate statistics to compare the K-Means Temporal MMD 
(Kernel 4) against Global MMD (Kernel 1) and Baseline (Kernel 0).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

def main():
    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / "results/factorial_v4/temporal_mmd_evaluation.csv"
    
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        print("CSV is empty.")
        return
        
    # Mapping for MMD Kernels
    kernel_names = {
        0: "None (MSE Baseline)",
        1: "Global RBF",
        2: "Anat. Laplacian",
        3: "Anat. IMQ-Multi",
        4: "Temporal K-Means IMQ"
    }
    df["kernel_name"] = df["mmd_kernel"].map(kernel_names)
    
    # We want to show how well Variance Ratio is preserved (closer to 1.0 is better)
    # Regression to the mean usually pushes variance ratio < 1.0
    
    out_dir = project_root / "results/factorial_v4/plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Variance Ratio for T-wave Amplitudes by MMD Kernel
    plt.figure(figsize=(10, 6))
    t_amp_df = df[df["clinical_feature"] == "T_Amp"]
    if len(t_amp_df) > 0:
        sns.boxplot(data=t_amp_df, x="kernel_name", y="variance_ratio", order=list(kernel_names.values()))
        plt.axhline(y=1.0, color='r', linestyle='--', label='Perfect Variance Preservation')
        plt.title("T-Wave Amplitude Variance Preservation by MMD Strategy")
        plt.ylabel("Variance Ratio (Reconstructed / Ground Truth)")
        plt.xlabel("MMD Strategy")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "t_wave_variance_ratio.png", dpi=300)
        plt.close()
        
    # Plot 2: QT Interval Variance Ratio
    plt.figure(figsize=(10, 6))
    qt_df = df[df["clinical_feature"] == "QT_Interval_ms"]
    if len(qt_df) > 0:
        sns.boxplot(data=qt_df, x="kernel_name", y="variance_ratio", order=list(kernel_names.values()))
        plt.axhline(y=1.0, color='r', linestyle='--', label='Perfect Variance Preservation')
        plt.title("QT Interval Variance Preservation by MMD Strategy")
        plt.ylabel("Variance Ratio (Reconstructed / Ground Truth)")
        plt.xlabel("MMD Strategy")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "qt_interval_variance_ratio.png", dpi=300)
        plt.close()

    # Generate Markdown Summary
    summary_md = project_root / "results/factorial_v4/mmd_analysis_summary.md"
    
    with open(summary_md, "w") as f:
        f.write("# Temporal MMD vs Global MMD: Regression to the Mean Analysis\n\n")
        
        f.write("## 1. Average Variance Ratio by Strategy\n")
        f.write("A ratio of 1.0 indicates perfect variance preservation. Lower ratios indicate regression to the mean (smoothing).\n\n")
        
        agg = df.groupby(["clinical_feature", "kernel_name"]).agg(
            mean_variance_ratio=("variance_ratio", "mean"),
            mean_ba_slope=("ba_robust_slope", "mean"),
            count=("model_mask", "nunique")
        ).reset_index()
        
        # Pivot table for display
        pivot = agg.pivot(index="clinical_feature", columns="kernel_name", values="mean_variance_ratio")
        f.write(pivot.to_markdown())
        f.write("\n\n")
        
        f.write("## 2. Bland-Altman Slopes by Strategy\n")
        f.write("A slope of 0 indicates unbiased reconstruction across all magnitudes. Negative slopes indicate proportional bias (underestimating high values, overestimating low values - typical of regression to the mean).\n\n")
        
        pivot_slope = agg.pivot(index="clinical_feature", columns="kernel_name", values="mean_ba_slope")
        f.write(pivot_slope.to_markdown())
        f.write("\n\n")
        
    print(f"Analysis complete. Generated {len(df)} records. Saved plots to {out_dir} and summary to {summary_md}")

if __name__ == "__main__":
    main()
