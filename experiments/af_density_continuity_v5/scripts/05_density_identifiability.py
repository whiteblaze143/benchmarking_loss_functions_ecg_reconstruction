#!/usr/bin/env python3
"""Script 05: Experiment E01 - Identifiability of Episode Continuity and Certified Density Bounds.
For real 60-minute bin sequences:
  - Generates alternative feasible within-bin continuous trajectories preserving exact bin masses
  - Evaluates continuity ambiguity (range of compatible S_max, longest run, episode count)
  - Computes certified analytical density bounds [D_lower, D_upper]
Generates:
  - results/E01_continuity_identifiability.csv
  - results/E01_density_bounds.parquet
"""

import os
import sys
import math
import random
from pathlib import Path
import pandas as pd
import numpy as np

root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir))

from src.density_continuous import compute_continuous_density
from src.density_bounds import compute_certified_density_bounds, compute_coarsening_bound

def main():
    features_dir = root_dir / "features"
    results_dir = root_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE C03 & C04: EXPERIMENT E01 - CONTINUITY IDENTIFIABILITY & BOUNDS")
    print("=" * 80)

    feats_parquet = features_dir / "density_reference_features.parquet"
    runs_parquet = root_dir / "manifests" / "canonical_af_runs.parquet"

    df_feats = pd.read_parquet(feats_parquet)
    df_runs = pd.read_parquet(runs_parquet)

    # 12h horizon density-defined windows
    df_12h = df_feats[
        (df_feats["horizon"] == "12h") &
        (df_feats["window_type"] == "OBSERVED_MONITORING_ANCHORED") &
        (df_feats["is_density_defined"])
    ].copy()

    print(f"Analyzing continuity ambiguity for {len(df_12h)} windows (Delta = 60m = 3600s)...")

    delta = 3600.0  # 60m
    random.seed(42)
    np.random.seed(42)

    ident_rows = []
    bounds_rows = []

    for idx, w in df_12h.iterrows():
        dataset = w["dataset"]
        rec_id = w["record_id"]
        pat_id = w["patient_id"]
        w_start = float(w["window_start_s"])
        w_end = float(w["window_end_s"])
        w_dur = float(w["window_duration_s"])
        burden = float(w["af_burden"])
        d_native = float(w["density_continuous"])
        s_max_native = float(w["s_max"])
        l_max_native = float(w["longest_af_run_s"])
        n_ep_native = int(w["n_af_episodes"])

        rec_runs = df_runs[(df_runs["dataset"] == dataset) & (df_runs["record_id"] == rec_id)]
        w_runs = []
        for _, r in rec_runs.iterrows():
            rs = max(w_start, float(r["start_time_s"]))
            re = min(w_end, float(r["end_time_s"]))
            if re > rs:
                w_runs.append((rs - w_start, re - w_start))

        total_m = sum(e - s for s, e in w_runs)
        if total_m <= 0:
            continue

        n_bins = int(math.ceil(w_dur / delta))
        bin_masses = np.zeros(n_bins, dtype=np.float64)
        for s, e in w_runs:
            sb = int(s // delta)
            eb = int(min(n_bins - 1, e // delta))
            for b_i in range(sb, eb + 1):
                bs = b_i * delta
                be = (b_i + 1) * delta
                overlap = max(0.0, min(e, be) - max(s, bs))
                bin_masses[b_i] += overlap

        # 1. Certified analytical bounds
        d_lower, d_upper, bound_width = compute_certified_density_bounds(bin_masses, delta)
        coarsening_bound = compute_coarsening_bound(delta, w_dur, burden)

        bounds_rows.append({
            "dataset": dataset,
            "record_id": rec_id,
            "patient_id": pat_id,
            "total_af_s": total_m,
            "af_burden": burden,
            "density_native": d_native,
            "certified_lower_bound": d_lower,
            "certified_upper_bound": d_upper,
            "certified_bound_width": bound_width,
            "coarsening_bound": coarsening_bound,
            "native_within_bounds": (d_lower - 1e-6 <= d_native <= d_upper + 1e-6)
        })

        # 2. Alternative feasible continuous arrangements preserving exact bin masses
        feasible_trajectories = {}

        # Trajectory A: Left-concentrated in each bin
        traj_left = []
        for b_i, m in enumerate(bin_masses):
            if m > 0:
                bs = b_i * delta
                traj_left.append((bs, bs + m))
        feasible_trajectories["LEFT_CONCENTRATED"] = traj_left

        # Trajectory B: Right-concentrated in each bin
        traj_right = []
        for b_i, m in enumerate(bin_masses):
            if m > 0:
                be = (b_i + 1) * delta
                traj_right.append((be - m, be))
        feasible_trajectories["RIGHT_CONCENTRATED"] = traj_right

        # Trajectory C: Boundary-packed / Merged across bins
        traj_merged = []
        for b_i, m in enumerate(bin_masses):
            if m > 0:
                # Alternate: even bins place at end, odd bins place at start -> merges at odd boundaries
                if b_i % 2 == 0:
                    be = (b_i + 1) * delta
                    traj_merged.append((be - m, be))
                else:
                    bs = b_i * delta
                    traj_merged.append((bs, bs + m))
        # Merge overlapping/adjacent
        traj_merged_clean = []
        for a, b in sorted(traj_merged):
            if not traj_merged_clean:
                traj_merged_clean.append((a, b))
            else:
                pa, pb = traj_merged_clean[-1]
                if a <= pb:
                    traj_merged_clean[-1] = (pa, max(pb, b))
                else:
                    traj_merged_clean.append((a, b))
        feasible_trajectories["MERGED_BOUNDARY"] = traj_merged_clean

        # Trajectory D: Fragmented (4 micro-bursts per active bin)
        traj_frag = []
        for b_i, m in enumerate(bin_masses):
            if m > 0:
                sub_m = m / 4.0
                bs = b_i * delta
                gap = (delta - m) / 4.0
                for k in range(4):
                    seg_s = bs + k * (sub_m + gap)
                    traj_frag.append((seg_s, seg_s + sub_m))
        feasible_trajectories["FRAGMENTED_4"] = traj_frag

        # Trajectory E: 20 Randomized placements
        sim_smax = []
        sim_d = []
        sim_lmax = []
        for _ in range(20):
            traj_rand = []
            for b_i, m in enumerate(bin_masses):
                if m > 0:
                    max_offset = delta - m
                    offset = random.uniform(0.0, max(0.0, max_offset))
                    bs = b_i * delta + offset
                    traj_rand.append((bs, bs + m))
            # Merge touching
            traj_rand_clean = []
            for a, b in sorted(traj_rand):
                if not traj_rand_clean:
                    traj_rand_clean.append((a, b))
                else:
                    pa, pb = traj_rand_clean[-1]
                    if a <= pb:
                        traj_rand_clean[-1] = (pa, max(pb, b))
                    else:
                        traj_rand_clean.append((a, b))
            
            d_sim = compute_continuous_density(traj_rand_clean, w_dur)
            durations = [b - a for a, b in traj_rand_clean]
            lm = max(durations) if durations else 0.0
            sm = lm / total_m if total_m > 0 else 0.0
            if d_sim is not None:
                sim_d.append(d_sim)
                sim_smax.append(sm)
                sim_lmax.append(lm)

        # Compute trajectory properties for structured alternatives
        traj_results = {}
        for t_name, t_itvs in feasible_trajectories.items():
            d_val = compute_continuous_density(t_itvs, w_dur)
            durs = [b - a for a, b in t_itvs]
            lm = max(durs) if durs else 0.0
            sm = lm / total_m if total_m > 0 else 0.0
            traj_results[t_name] = {"density": d_val, "s_max": sm, "l_max": lm, "n_ep": len(t_itvs)}

        all_smax = [s_max_native] + [res["s_max"] for res in traj_results.values()] + sim_smax
        all_d = [d_native] + [res["density"] for res in traj_results.values() if res["density"] is not None] + sim_d
        all_lmax = [l_max_native] + [res["l_max"] for res in traj_results.values()] + sim_lmax

        min_smax = min(all_smax)
        max_smax = max(all_smax)
        smax_range = max_smax - min_smax
        lmax_range_s = max(all_lmax) - min(all_lmax)

        ident_rows.append({
            "dataset": dataset,
            "record_id": rec_id,
            "patient_id": pat_id,
            "af_burden": burden,
            "total_af_s": total_m,
            "native_density": d_native,
            "native_s_max": s_max_native,
            "native_longest_run_s": l_max_native,
            "native_n_episodes": n_ep_native,
            "compatible_min_s_max": min_smax,
            "compatible_max_s_max": max_smax,
            "compatible_s_max_ambiguity": smax_range,
            "compatible_longest_run_range_s": lmax_range_s,
            "compatible_min_density": min(all_d),
            "compatible_max_density": max(all_d),
            "compatible_density_spread": max(all_d) - min(all_d),
            "merged_s_max": traj_results["MERGED_BOUNDARY"]["s_max"],
            "fragmented_s_max": traj_results["FRAGMENTED_4"]["s_max"],
            "has_substantial_continuity_ambiguity": (smax_range >= 0.25)
        })

    df_ident = pd.DataFrame(ident_rows)
    df_bounds = pd.DataFrame(bounds_rows)

    out_ident = results_dir / "E01_continuity_identifiability.csv"
    out_bounds = results_dir / "E01_density_bounds.parquet"

    df_ident.to_csv(out_ident, index=False)
    df_bounds.to_parquet(out_bounds, index=False)

    print(f"\nSaved continuity identifiability results to {out_ident} (shape: {df_ident.shape})")
    print(f"Saved certified density bounds to {out_bounds} (shape: {df_bounds.shape})")

    # Summary
    print("\n" + "=" * 80)
    print("EXPERIMENT E01 IDENTIFIABILITY & CONTINUITY AMBIGUITY SUMMARY")
    print("=" * 80)
    print(f"Total windows analyzed: {len(df_ident)}")
    print(f"Median compatible S_max ambiguity (range): {df_ident['compatible_s_max_ambiguity'].median():.4f}")
    print(f"Windows with substantial ambiguity (Delta S_max >= 0.25): {df_ident['has_substantial_continuity_ambiguity'].sum()} / {len(df_ident)} ({df_ident['has_substantial_continuity_ambiguity'].mean()*100:.1f}%)")
    print(f"Median compatible longest run range: {df_ident['compatible_longest_run_range_s'].median() / 60.0:.2f} minutes")
    print(f"Certified bound coverage (native in [D_L, D_U]): {df_bounds['native_within_bounds'].mean()*100:.1f}%")
    print(f"Median certified bound width: {df_bounds['certified_bound_width'].median():.4f}")

if __name__ == "__main__":
    main()
