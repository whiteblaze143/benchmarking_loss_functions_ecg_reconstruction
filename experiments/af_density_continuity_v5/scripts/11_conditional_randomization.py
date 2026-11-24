#!/usr/bin/env python3
"""Script 11: Experiment E03 - Conditional Randomization Tests (Null A and Null B).
Tests whether observed rapid-AF localization differs from conditional null distributions:
  - Null A: Random circular shifts along the cumulative AF clock (2000 perms)
  - Null B: Within-episode rate order shuffle (2000 perms)
Generates:
  - results/E03_randomization_results.parquet
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir))

from src.randomization import (
    generate_null_a_circular_shifts,
    generate_null_b_within_episode_permutations
)

def main():
    features_dir = root_dir / "features"
    results_dir = root_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE F04: EXPERIMENT E03 - CONDITIONAL RANDOMIZATION TESTS (NULL A & B)")
    print("=" * 80)

    ci_parquet = results_dir / "E03_concentration_intervals.parquet"
    rates_parquet = features_dir / "ventricular_rate_intervals.parquet"
    runs_parquet = root_dir / "manifests" / "canonical_af_runs.parquet"

    df_ci = pd.read_parquet(ci_parquet)
    df_rates = pd.read_parquet(rates_parquet)
    df_runs = pd.read_parquet(runs_parquet)

    # Deduplicate rate intervals across overlapping window definitions
    df_rates = df_rates.drop_duplicates(subset=["dataset", "record_id", "start_time_s", "end_time_s"]).copy()

    # Primary coverage fraction p = 0.80
    df_p80 = df_ci[df_ci["target_fraction_p"] == 0.80].copy()
    print(f"Running randomization tests for {len(df_p80)} windows with active rapid AF...")

    randomization_records = []
    n_perms = 2000

    for idx, row in df_p80.iterrows():
        dataset = row["dataset"]
        rec_id = row["record_id"]
        pat_id = row["patient_id"]
        target_s = float(row["start_time_s"])
        target_e = float(row["end_time_s"])
        obs_c = float(row["c_rapid"])
        total_rapid = float(row["total_rapid_af_s"])

        # Fetch rate intervals for this window
        w_rates = df_rates[
            (df_rates["dataset"] == dataset) &
            (df_rates["record_id"] == rec_id) &
            (df_rates["start_time_s"] >= target_s - 43200.0) &
            (df_rates["end_time_s"] <= target_e + 43200.0)
        ].to_dict(orient="records")

        # Fetch episode boundaries
        rec_runs = df_runs[(df_runs["dataset"] == dataset) & (df_runs["record_id"] == rec_id)]
        ep_boundaries = [(float(r["start_time_s"]), float(r["end_time_s"])) for _, r in rec_runs.iterrows()]

        # 1. Null A: Circular shifts
        null_a_fractions = generate_null_a_circular_shifts(
            w_rates, target_s, target_e, n_permutations=n_perms, seed=42 + idx
        )
        null_a_arr = np.array(null_a_fractions)
        mean_null_a = float(np.mean(null_a_arr))
        p_val_a_higher = float(np.mean(null_a_arr >= obs_c))
        p_val_a_lower = float(np.mean(null_a_arr <= obs_c))

        # 2. Null B: Within-episode rate order shuffle
        null_b_fractions = generate_null_b_within_episode_permutations(
            w_rates, ep_boundaries, target_s, target_e, n_permutations=n_perms, seed=142 + idx
        )
        null_b_arr = np.array(null_b_fractions)
        mean_null_b = float(np.mean(null_b_arr))
        p_val_b_higher = float(np.mean(null_b_arr >= obs_c))
        p_val_b_lower = float(np.mean(null_b_arr <= obs_c))

        randomization_records.append({
            "dataset": dataset,
            "record_id": rec_id,
            "patient_id": pat_id,
            "observed_c_rapid": obs_c,
            "null_a_mean_c_rapid": mean_null_a,
            "null_a_p_val_higher": p_val_a_higher,
            "null_a_p_val_lower": p_val_a_lower,
            "null_b_mean_c_rapid": mean_null_b,
            "null_b_p_val_higher": p_val_b_higher,
            "null_b_p_val_lower": p_val_b_lower,
            "significant_localization_null_a": (p_val_a_higher < 0.05 or p_val_a_lower < 0.05),
            "significant_localization_null_b": (p_val_b_higher < 0.05 or p_val_b_lower < 0.05)
        })

    df_rand = pd.DataFrame(randomization_records)
    out_rand = results_dir / "E03_randomization_results.parquet"
    df_rand.to_parquet(out_rand, index=False)
    print(f"\nSaved randomization results to {out_rand} (shape: {df_rand.shape})")

    print("\n" + "=" * 80)
    print("RANDOMIZATION TEST SUMMARY (p = 0.80)")
    print("=" * 80)
    print(f"Observed C_rapid median: {df_rand['observed_c_rapid'].median()*100:.1f}%")
    print(f"Null A Mean C_rapid: {df_rand['null_a_mean_c_rapid'].median()*100:.1f}%")
    print(f"Null B Mean C_rapid: {df_rand['null_b_mean_c_rapid'].median()*100:.1f}%")
    print(f"Windows with significant rate-concentration decoupling (Null A p < 0.05): {df_rand['significant_localization_null_a'].mean()*100:.1f}%")

if __name__ == "__main__":
    main()
