import pandas as pd
import numpy as np
from pathlib import Path
import json

def analyze_bland_altman():
    csv_path = Path("results/factorial_v4/bland_altman_plots/regression_to_mean_summary.csv")
    if not csv_path.exists():
        print(f"File {csv_path} not ready yet.")
        return
        
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path.name}")
    
    # We are interested in how different loss components (Correlation, MMD, Derivative)
    # affect the Bland-Altman robust slope (measure of regression to mean) 
    # and the variance ratio.
    # 
    # Ideal robust_slope should be close to 0 (no proportional bias).
    # Negative robust slope = regression to the mean (high values underestimated, low values overestimated).
    # Ideal variance_ratio should be close to 1.0 (variance is preserved).
    
    # Filter to T_Peaks and R_Peaks which are the most susceptible to amplitude smoothing
    df_peaks = df[df["fiducial"].isin(["T_Peaks", "R_Peaks"])]
    
    summary = df_peaks.groupby("model_id").agg({
        "variance_ratio": "mean",
        "ba_robust_slope": "mean",
        "ba_bias": "mean"
    }).reset_index()
    
    # Parse out the architecture and the loss mask
    # model_id format: e.g. unet__e0c0m0d0__s42
    summary["architecture"] = summary["model_id"].apply(lambda x: x.split("__")[0])
    summary["mask"] = summary["model_id"].apply(lambda x: x.split("__")[1] if len(x.split("__")) > 1 else "")
    
    for arch in summary["architecture"].unique():
        print(f"\n--- Architecture: {arch.upper()} ---")
        arch_df = summary[summary["architecture"] == arch].sort_values("ba_robust_slope", ascending=False)
        for _, row in arch_df.iterrows():
            print(f"Mask {row['mask']}: Var Ratio = {row['variance_ratio']:.3f} | BA Robust Slope = {row['ba_robust_slope']:.3f}")

if __name__ == "__main__":
    analyze_bland_altman()
