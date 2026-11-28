#!/usr/bin/env python3
"""Runner script for Experiment E01: Temporal neighbourhoods of prolonged AF in LTAFDB and SHDB.
Generates all 12 required E01 outputs in results/ directory.
"""

import os
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np

base_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(base_dir))

from src.e01_duration_classes import (
    process_runs_with_classes,
    build_duration_class_inventory,
    build_patient_feasibility
)
from src.e01_neighbourhoods import (
    build_index_episode_manifest,
    construct_neighbourhood_intervals,
    construct_remote_comparison_manifest,
    build_risk_time_summary
)
from src.e01_statistics import (
    compute_primary_rate_ratios,
    compute_patient_level_effects,
    compute_sensitivity_analysis
)
from src.e01_null_models import execute_cohort_null_experiments
from src.e01_density_link import analyze_density_architecture_connections


def main():
    print("=" * 80)
    print("STARTING EXPERIMENT E01: TEMPORAL NEIGHBOURHOODS OF PROLONGED AF")
    print("=" * 80)
    t0 = time.time()
    
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_dir = Path(__file__).resolve().parents[2] / "af_density_continuity_v5" / "manifests"
    print(f"Loading canonical manifests from {manifest_dir}...")
    
    runs_df = pd.read_parquet(manifest_dir / "canonical_af_runs.parquet")
    rhythms_df = pd.read_parquet(manifest_dir / "canonical_rhythm_intervals.parquet")
    cohort_inv = pd.read_csv(manifest_dir / "cohort_inventory.csv")
    
    # Step 1: Duration Classes & Feasibility
    print("\n[Step 1/5] Assigning Duration Classes and Building Feasibility Matrix...")
    df_runs_classed = process_runs_with_classes(runs_df)
    
    inv_df = build_duration_class_inventory(df_runs_classed)
    inv_path = results_dir / "E01_duration_class_inventory.csv"
    inv_df.to_csv(inv_path, index=False)
    print(f"  -> Saved {inv_path} ({len(inv_df)} rows)")
    
    feas_df = build_patient_feasibility(df_runs_classed)
    feas_path = results_dir / "E01_patient_feasibility.csv"
    feas_df.to_csv(feas_path, index=False)
    print(f"  -> Saved {feas_path} ({len(feas_df)} rows)")
    
    # Step 2: Index Episodes and Neighbourhoods
    print("\n[Step 2/5] Constructing Index Episodes, Pre/Post Windows, and Overlap Deduplication...")
    index_df = build_index_episode_manifest(df_runs_classed)
    idx_path = results_dir / "E01_index_episode_manifest.parquet"
    index_df.to_parquet(idx_path, index=False)
    print(f"  -> Saved {idx_path} ({len(index_df)} prolonged index episodes)")
    
    df_intervals, df_assigned = construct_neighbourhood_intervals(
        index_df, rhythms_df, df_runs_classed, cohort_inv
    )
    int_path = results_dir / "E01_neighbourhood_intervals.parquet"
    df_intervals.to_parquet(int_path, index=False)
    print(f"  -> Saved {int_path} ({len(df_intervals)} intervals)")
    
    assign_path = results_dir / "E01_neighbourhood_event_assignments.parquet"
    df_assigned.to_parquet(assign_path, index=False)
    print(f"  -> Saved {assign_path} ({len(df_assigned)} assigned brief events)")
    
    # Remote Comparison
    print("  Constructing remote comparison intervals (>= 60m from prolonged AF)...")
    df_remote = construct_remote_comparison_manifest(
        index_df, rhythms_df, df_runs_classed, cohort_inv, exclusion_distance_s=3600.0, window_duration_s=900.0
    )
    rem_path = results_dir / "E01_remote_comparison_manifest.parquet"
    df_remote.to_parquet(rem_path, index=False)
    print(f"  -> Saved {rem_path} ({len(df_remote)} remote intervals)")
    
    risk_sum_df = build_risk_time_summary(df_intervals, df_remote)
    risk_path = results_dir / "E01_risk_time_summary.csv"
    risk_sum_df.to_csv(risk_path, index=False)
    print(f"  -> Saved {risk_path} ({len(risk_sum_df)} rows)")
    
    # Step 3: Statistical Rate Ratios & Patient-Clustered Models
    print("\n[Step 3/5] Fitting Patient-Clustered Poisson Models with ln(T_risk) Offset...")
    rr_df = compute_primary_rate_ratios(df_intervals, df_remote)
    rr_path = results_dir / "E01_neighbourhood_rate_ratios.csv"
    rr_df.to_csv(rr_path, index=False)
    print(f"  -> Saved {rr_path} ({len(rr_df)} model contrasts)")
    
    p_eff_df = compute_patient_level_effects(df_intervals, df_remote)
    peff_path = results_dir / "E01_patient_level_effects.csv"
    p_eff_df.to_csv(peff_path, index=False)
    print(f"  -> Saved {peff_path} ({len(p_eff_df)} patient profiles)")
    
    sens_df = compute_sensitivity_analysis(df_intervals, df_remote, df_runs_classed)
    sens_path = results_dir / "E01_sensitivity_analysis.csv"
    sens_df.to_csv(sens_path, index=False)
    print(f"  -> Saved {sens_path} ({len(sens_df)} scenarios)")
    
    # Step 4: Conditional Null Experiments (Null A and Null B)
    print("\n[Step 4/5] Running Conditional Null Experiments (Null A & Null B, B=2000)...")
    null_df = execute_cohort_null_experiments(df_runs_classed, cohort_inv, n_permutations=2000, seed=42)
    null_path = results_dir / "E01_conditional_null_results.parquet"
    null_df.to_parquet(null_path, index=False)
    print(f"  -> Saved {null_path} ({len(null_df)} patient null tests)")
    
    # Step 5: Charitos Density Connection
    print("\n[Step 5/5] Connecting Temporal Neighbourhoods to Continuous Charitos AF Density...")
    dens_df = analyze_density_architecture_connections(df_runs_classed, df_intervals, df_remote, cohort_inv)
    dens_path = results_dir / "E01_density_architecture_results.csv"
    dens_df.to_csv(dens_path, index=False)
    print(f"  -> Saved {dens_path} ({len(dens_df)} recordings analyzed)")
    
    elapsed = time.time() - t0
    print(f"\n[DONE] E01 Pipeline completed successfully in {elapsed:.1f} seconds.")
    print("=" * 80)


if __name__ == "__main__":
    main()
