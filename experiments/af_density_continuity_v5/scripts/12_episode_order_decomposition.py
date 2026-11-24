#!/usr/bin/env python3
"""Script 12: Phase G - Episode-Order Decomposition of AF Density.
Permutes episode duration sequences and internal gap duration sequences
to determine whether observed density is driven by duration distributions
or their specific temporal order.
Generates:
  - results/episode_order_randomization.parquet
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir))

from src.density_continuous import compute_continuous_density
from src.randomization import permute_episode_order

def main():
    features_dir = root_dir / "features"
    results_dir = root_dir / "results"
    manifests_dir = root_dir / "manifests"

    print("=" * 80)
    print("PHASE G: EPISODE-ORDER DECOMPOSITION OF AF DENSITY")
    print("=" * 80)

    df_feats = pd.read_parquet(features_dir / "density_reference_features.parquet")
    df_runs = pd.read_parquet(manifests_dir / "canonical_af_runs.parquet")

    # Filter to multi-episode (>= 3 runs) 12h windows
    multi_ep = df_feats[
        (df_feats["horizon"] == "12h") &
        (df_feats["window_type"] == "OBSERVED_MONITORING_ANCHORED") &
        (df_feats["n_af_episodes"] >= 3) &
        (df_feats["is_density_defined"])
    ].copy()

    print(f"Decomposing episode order across {len(multi_ep)} multi-episode windows...")

    records = []
    n_perms = 200

    for idx, w in multi_ep.iterrows():
        dataset = w["dataset"]
        rec_id = w["record_id"]
        pat_id = w["patient_id"]
        w_start = float(w["window_start_s"])
        w_end = float(w["window_end_s"])
        w_dur = float(w["window_duration_s"])
        obs_d = float(w["density_continuous"])

        rec_runs = df_runs[(df_runs["dataset"] == dataset) & (df_runs["record_id"] == rec_id)]
        w_runs = []
        for _, r in rec_runs.iterrows():
            rs = max(w_start, float(r["start_time_s"]))
            re = min(w_end, float(r["end_time_s"]))
            if re > rs:
                w_runs.append((rs - w_start, re - w_start))
        w_runs.sort()

        if len(w_runs) < 3:
            continue

        perm_densities = []
        for rep in range(n_perms):
            p_eps = permute_episode_order(w_runs, w_dur, seed=42 + rep * 17)
            d_p = compute_continuous_density(p_eps, w_dur)
            if d_p is not None:
                perm_densities.append(d_p)

        if not perm_densities:
            continue

        perm_arr = np.array(perm_densities)
        mean_perm_d = float(np.mean(perm_arr))
        std_perm_d = float(np.std(perm_arr))
        d_iqr = float(np.percentile(perm_arr, 75) - np.percentile(perm_arr, 25))

        records.append({
            "dataset": dataset,
            "record_id": rec_id,
            "patient_id": pat_id,
            "n_episodes": len(w_runs),
            "observed_density": obs_d,
            "permuted_mean_density": mean_perm_d,
            "permuted_std_density": std_perm_d,
            "permuted_density_iqr": d_iqr,
            "density_change_by_reordering": abs(obs_d - mean_perm_d)
        })

    df_decomp = pd.DataFrame(records)
    out_pq = results_dir / "episode_order_randomization.parquet"
    df_decomp.to_parquet(out_pq, index=False)
    print(f"Saved episode-order decomposition to {out_pq}")
    print(f"Median density change caused by pure episode reordering: {df_decomp['density_change_by_reordering'].median():.4f}")
    print(f"Median permuted density spread (std): {df_decomp['permuted_std_density'].median():.4f}")

if __name__ == "__main__":
    main()
