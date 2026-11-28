#!/usr/bin/env python3
"""Runner script for Experiment E02: Historical AF concentration and electrophysiology in IRIDIA.
Generates all 10 required E02 outputs in results/ directory.
"""

import os
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np

base_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(base_dir))

from src.e02_historical_exposure import (
    build_historical_exposure_manifest
)
from src.e02_ventricular_physiology import (
    extract_features_for_population_b
)
from src.e02_statistical_models import (
    build_trajectory_dataset,
    fit_nested_trajectory_models,
    fit_within_between_decomposition,
    fit_subsequent_episode_survival
)


def main():
    print("=" * 80)
    print("STARTING EXPERIMENT E02: HISTORICAL EXPOSURE AND ELECTROPHYSIOLOGY")
    print("=" * 80)
    t0 = time.time()
    
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_dir = Path(__file__).resolve().parents[2] / "af_density_continuity_v5" / "manifests"
    print(f"Loading canonical manifests from {manifest_dir}...")
    
    runs_df = pd.read_parquet(manifest_dir / "canonical_af_runs.parquet")
    
    # Step 1: Historical Exposure Manifest & Feasibility
    print("\n[Step 1/4] Constructing Population A & B and Historical Exposure Manifest (H1, H3, H6, H12)...")
    df_manifest, df_feas, df_cond = build_historical_exposure_manifest(runs_df, primary_horizon_h=6.0)
    
    man_path = results_dir / "E02_historical_exposure_manifest.parquet"
    df_manifest.to_parquet(man_path, index=False)
    print(f"  -> Saved {man_path} ({len(df_manifest)} episodes)")
    
    feas_path = results_dir / "E02_landmark_feasibility.csv"
    df_feas.to_csv(feas_path, index=False)
    print(f"  -> Saved {feas_path} ({len(df_feas)} feasibility rows)")
    
    cond_path = results_dir / "E02_conditional_support.csv"
    df_cond.to_csv(cond_path, index=False)
    print(f"  -> Saved {cond_path} ({len(df_cond)} support rows)")
    
    # Step 2: Fixed Elapsed-Time Physiology Extraction
    print("\n[Step 2/4] Extracting Fixed Elapsed-Time RR Features (30-180s, 5-7.5m, 15-17.5m, 30-32.5m)...")
    df_physio = extract_features_for_population_b(df_manifest)
    phys_path = results_dir / "E02_fixed_elapsed_rr_features.parquet"
    df_physio.to_parquet(phys_path, index=False)
    print(f"  -> Saved {phys_path} ({len(df_physio)} window feature rows)")
    
    # Step 3: Nested Trajectory Models (M0 to M4) and Within/Between Decomposition
    print("\n[Step 3/4] Fitting Nested Trajectory Models M0-M4 and Within/Between Decomposition...")
    df_traj = build_trajectory_dataset(df_physio)
    print(f"  Paired trajectory dataset contains {len(df_traj)} complete episodes across {df_traj['patient_id'].nunique()} patients.")
    
    df_models, df_nested_comp = fit_nested_trajectory_models(df_traj)
    mod_path = results_dir / "E02_ventricular_trajectory_models.csv"
    df_models.to_csv(mod_path, index=False)
    print(f"  -> Saved {mod_path}")
    
    comp_path = results_dir / "E02_nested_model_comparisons.csv"
    df_nested_comp.to_csv(comp_path, index=False)
    print(f"  -> Saved {comp_path}")
    
    df_within, df_between = fit_within_between_decomposition(df_traj)
    w_path = results_dir / "E02_within_patient_effects.csv"
    df_within.to_csv(w_path, index=False)
    print(f"  -> Saved {w_path}")
    
    b_path = results_dir / "E02_between_patient_effects.csv"
    df_between.to_csv(b_path, index=False)
    print(f"  -> Saved {b_path}")
    
    # Step 4: Subsequent Episode Duration Survival Models on Population A
    print("\n[Step 4/4] Fitting Cox Proportional Hazards Survival Models on Population A...")
    df_surv, df_risk = fit_subsequent_episode_survival(df_manifest)
    
    surv_path = results_dir / "E02_subsequent_episode_models.csv"
    df_surv.to_csv(surv_path, index=False)
    print(f"  -> Saved {surv_path}")
    
    risk_path = results_dir / "E02_survival_risk_sets.csv"
    df_risk.to_csv(risk_path, index=False)
    print(f"  -> Saved {risk_path}")
    
    elapsed = time.time() - t0
    print(f"\n[DONE] E02 Pipeline completed successfully in {elapsed:.1f} seconds.")
    print("=" * 80)


if __name__ == "__main__":
    main()
