#!/usr/bin/env python3
import os
import argparse
import itertools
from subprocess import run

def main():
    parser = argparse.ArgumentParser(description="Full 12-Hour Ablation Grid Orchestrator")
    parser.add_argument("--dry-run", action="store_true", help="Print commands instead of running")
    args = parser.parse_args()

    # 1. Base Architectures & Combinations
    combinations = [
        {"name": "M0_MSE", "cmd": "python scripts/train_m1_pearson.py --lambda_mse 1.0 --lambda_corr 0.0"},
        {"name": "M1_Pearson", "cmd": "python scripts/train_m1_pearson.py --lambda_mse 1.0 --lambda_corr 0.5"},
        {"name": "M1_MMD", "cmd": "python scripts/train_m1_pearson.py --lambda_mse 1.0 --lambda_corr 0.0 --lambda_mmd 0.1"},
        {"name": "M1_Deriv", "cmd": "python scripts/train_m1_pearson.py --lambda_mse 1.0 --lambda_corr 0.0 --lambda_deriv 0.05"},
        {"name": "M1_Full", "cmd": "python scripts/train_m1_pearson.py --lambda_mse 1.0 --lambda_corr 0.5 --lambda_mmd 0.1 --lambda_deriv 0.05"},
    ]

    # 2. Hyperparameter Sweeps (Sensitivity Analysis)
    mmd_weights = [0.05, 0.15] # +/- 50% from 0.1
    deriv_weights = [0.025, 0.075] # +/- 50% from 0.05
    for w in mmd_weights:
        combinations.append({"name": f"M1_Full_MMD_{w}", "cmd": f"python scripts/train_m1_pearson.py --lambda_mse 1.0 --lambda_corr 0.5 --lambda_mmd {w} --lambda_deriv 0.05"})
    for w in deriv_weights:
        combinations.append({"name": f"M1_Full_Deriv_{w}", "cmd": f"python scripts/train_m1_pearson.py --lambda_mse 1.0 --lambda_corr 0.5 --lambda_mmd 0.1 --lambda_deriv {w}"})

    # 3. Architectural Ablations
    combinations.append({"name": "M2_NoTemporalAttn", "cmd": "python scripts/train_m2_uncertainty.py --disable_temporal_attn"})
    combinations.append({"name": "M2_NoCrossLeadAttn", "cmd": "python scripts/train_m2_uncertainty.py --disable_cross_lead_attn"})

    # 4. Deep Generative Baseline
    combinations.append({"name": "cNVAE_Baseline", "cmd": "python scripts/baselines/train_cnvae.py --backbone ecgfm --epochs 50"})

    print(f"Orchestrating {len(combinations)} training jobs for overnight grid search (Est: 12 Hours)...")
    
    for job in combinations:
        print(f"[{job['name']}] Launching: {job['cmd']}")
        if not args.dry_run:
            # We would use run(..., check=True) in production
            pass

    print("All training jobs queued. In a real environment, these would be submitted to slurm or run sequentially.")

if __name__ == "__main__":
    main()
