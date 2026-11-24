#!/usr/bin/env python3
"""Script 02: Build Standardized Observation Windows and Eligibility Flow.
Standardizes observation horizons:
  - Primary cross-cohort horizon: observed-monitoring-anchored first 12 hours [t0, t0 + 43200s]
  - Secondary horizon: observed-monitoring-anchored first 24 hours [t0, t0 + 86400s]
  - Sensitivity horizon: observed-monitoring-anchored first 6 hours [t0, t0 + 21600s]
  - Raw clock-zero sensitivity: [0, H]
  - Non-overlapping consecutive 12h windows for recordings >= 24h
  - Full recording window
Generates:
  - manifests/observation_windows.parquet
  - manifests/eligibility_flow.csv
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir))

def main():
    manifests_dir = root_dir / "manifests"
    intervals_parquet = manifests_dir / "canonical_rhythm_intervals.parquet"
    runs_parquet = manifests_dir / "canonical_af_runs.parquet"
    inv_csv = manifests_dir / "cohort_inventory.csv"

    print("=" * 80)
    print("PHASE A02: STANDARDIZE OBSERVATION HORIZONS AND ELIGIBILITY FLOW")
    print("=" * 80)

    df_intervals = pd.read_parquet(intervals_parquet)
    df_runs = pd.read_parquet(runs_parquet)
    df_inv = pd.read_csv(inv_csv)

    horizons = {
        "12h": 43200.0,
        "24h": 86400.0,
        "6h": 21600.0
    }

    windows = []
    eligibility_records = []

    # Map first known rhythm start time per (dataset, record_id)
    t0_map = df_intervals[df_intervals["rhythm_known"]].groupby(["dataset", "record_id"])["start_time_s"].min().to_dict()

    for _, r_meta in df_inv.iterrows():
        dataset = r_meta["dataset"]
        rec_id = str(r_meta["record_id"])
        pat_id = str(r_meta["patient_id"])
        rec_dur_s = float(r_meta["duration_s"])

        rec_itvs = df_intervals[
            (df_intervals["dataset"] == dataset) &
            (df_intervals["record_id"] == rec_id)
        ].sort_values("start_time_s")

        rec_runs = df_runs[
            (df_runs["dataset"] == dataset) &
            (df_runs["record_id"] == rec_id)
        ].sort_values("start_time_s")

        t0 = t0_map.get((dataset, rec_id), 0.0)

        # 1. Primary & Secondary Horizons anchored at t0 (start of valid rhythm monitoring)
        for h_name, h_dur in horizons.items():
            avail_dur_from_t0 = rec_dur_s - t0
            is_eligible_duration = (avail_dur_from_t0 >= h_dur)

            w_start = t0
            w_end = t0 + h_dur

            if not is_eligible_duration:
                eligibility_records.append({
                    "dataset": dataset,
                    "record_id": rec_id,
                    "patient_id": pat_id,
                    "horizon": h_name,
                    "anchor_type": "OBSERVED_MONITORING_ANCHORED",
                    "window_index": 0,
                    "duration_eligible": False,
                    "reason_excluded": "RECORD_TOO_SHORT",
                    "af_burden": np.nan,
                    "af_duration_s": np.nan,
                    "af_episodes": 0,
                    "is_density_defined": False,
                    "is_multi_episode": False
                })
                continue

            # Clip intervals to [w_start, w_end]
            itvs_clipped = []
            for _, itv in rec_itvs.iterrows():
                s = max(w_start, itv["start_time_s"])
                e = min(w_end, itv["end_time_s"])
                if e > s:
                    itvs_clipped.append({
                        "start_time_s": s,
                        "end_time_s": e,
                        "duration_s": e - s,
                        "harmonized": itv["rhythm_label_harmonized"],
                        "known": itv["rhythm_known"]
                    })
            df_clipped = pd.DataFrame(itvs_clipped)

            unknown_s = df_clipped[~df_clipped["known"]]["duration_s"].sum() if not df_clipped.empty else 0.0
            af_s = df_clipped[df_clipped["harmonized"] == "AF"]["duration_s"].sum() if not df_clipped.empty else 0.0
            burden = af_s / h_dur

            # Clip AF runs to [w_start, w_end]
            w_runs = []
            for _, r in rec_runs.iterrows():
                rs = max(w_start, r["start_time_s"])
                re = min(w_end, r["end_time_s"])
                if re > rs:
                    w_runs.append({
                        "start_time_s": rs,
                        "end_time_s": re,
                        "duration_s": re - rs
                    })

            n_episodes = len(w_runs)
            longest_run_s = max([r["duration_s"] for r in w_runs], default=0.0)
            s_max = (longest_run_s / af_s) if af_s > 0 else 0.0

            is_density_defined = (0.0 < burden < 1.0) and (unknown_s < 1.0)
            is_multi_ep = (n_episodes >= 2)

            win_entry = {
                "dataset": dataset,
                "record_id": rec_id,
                "patient_id": pat_id,
                "horizon": h_name,
                "window_type": "OBSERVED_MONITORING_ANCHORED",
                "window_index": 0,
                "window_start_s": w_start,
                "window_end_s": w_end,
                "window_duration_s": h_dur,
                "total_af_s": af_s,
                "af_burden": burden,
                "unknown_s": unknown_s,
                "n_af_episodes": n_episodes,
                "longest_af_run_s": longest_run_s,
                "s_max": s_max,
                "is_af_positive": (af_s > 0),
                "is_density_defined": is_density_defined,
                "is_multi_episode": is_multi_ep,
                "is_continuous_af": (burden >= 0.999),
                "is_zero_af": (burden == 0.0)
            }
            windows.append(win_entry)

            eligibility_records.append({
                "dataset": dataset,
                "record_id": rec_id,
                "patient_id": pat_id,
                "horizon": h_name,
                "anchor_type": "OBSERVED_MONITORING_ANCHORED",
                "window_index": 0,
                "duration_eligible": True,
                "reason_excluded": "NONE" if is_density_defined else ("ZERO_AF" if burden == 0 else ("CONTINUOUS_AF" if burden >= 0.999 else "UNKNOWN_RHYTHM")),
                "af_burden": burden,
                "af_duration_s": af_s,
                "af_episodes": n_episodes,
                "is_density_defined": is_density_defined,
                "is_multi_episode": is_multi_ep
            })

        # 2. Clock-zero acquisition-anchored sensitivity: [0, H]
        for h_name, h_dur in horizons.items():
            if rec_dur_s < h_dur:
                continue
            w_start = 0.0
            w_end = h_dur

            itvs_clipped = []
            for _, itv in rec_itvs.iterrows():
                s = max(w_start, itv["start_time_s"])
                e = min(w_end, itv["end_time_s"])
                if e > s:
                    itvs_clipped.append({
                        "duration_s": e - s,
                        "harmonized": itv["rhythm_label_harmonized"],
                        "known": itv["rhythm_known"]
                    })
            df_clipped = pd.DataFrame(itvs_clipped)

            unknown_s = df_clipped[~df_clipped["known"]]["duration_s"].sum() if not df_clipped.empty else 0.0
            af_s = df_clipped[df_clipped["harmonized"] == "AF"]["duration_s"].sum() if not df_clipped.empty else 0.0
            burden = af_s / h_dur

            w_runs = []
            for _, r in rec_runs.iterrows():
                rs = max(w_start, r["start_time_s"])
                re = min(w_end, r["end_time_s"])
                if re > rs:
                    w_runs.append(re - rs)

            n_episodes = len(w_runs)
            longest_run_s = max(w_runs, default=0.0)
            s_max = (longest_run_s / af_s) if af_s > 0 else 0.0

            windows.append({
                "dataset": dataset,
                "record_id": rec_id,
                "patient_id": pat_id,
                "horizon": f"{h_name}_clock_zero",
                "window_type": "CLOCK_ZERO_SENSITIVITY",
                "window_index": 0,
                "window_start_s": w_start,
                "window_end_s": w_end,
                "window_duration_s": h_dur,
                "total_af_s": af_s,
                "af_burden": burden,
                "unknown_s": unknown_s,
                "n_af_episodes": n_episodes,
                "longest_af_run_s": longest_run_s,
                "s_max": s_max,
                "is_af_positive": (af_s > 0),
                "is_density_defined": (0.0 < burden < 1.0) and (unknown_s < 1.0),
                "is_multi_episode": (n_episodes >= 2),
                "is_continuous_af": (burden >= 0.999),
                "is_zero_af": (burden == 0.0)
            })

        # 3. Non-overlapping 12h windows from t0
        avail_from_t0 = rec_dur_s - t0
        if avail_from_t0 >= 43200.0:
            n_consec = int(avail_from_t0 // 43200.0)
            for w_idx in range(n_consec):
                w_start = t0 + w_idx * 43200.0
                w_end = t0 + (w_idx + 1) * 43200.0
                h_dur = 43200.0

                itvs_clipped = []
                for _, itv in rec_itvs.iterrows():
                    s = max(w_start, itv["start_time_s"])
                    e = min(w_end, itv["end_time_s"])
                    if e > s:
                        itvs_clipped.append({
                            "duration_s": e - s,
                            "harmonized": itv["rhythm_label_harmonized"],
                            "known": itv["rhythm_known"]
                        })
                df_clipped = pd.DataFrame(itvs_clipped)
                unknown_s = df_clipped[~df_clipped["known"]]["duration_s"].sum() if not df_clipped.empty else 0.0
                af_s = df_clipped[df_clipped["harmonized"] == "AF"]["duration_s"].sum() if not df_clipped.empty else 0.0
                burden = af_s / h_dur

                w_runs = []
                for _, r in rec_runs.iterrows():
                    rs = max(w_start, r["start_time_s"])
                    re = min(w_end, r["end_time_s"])
                    if re > rs:
                        w_runs.append(re - rs)

                n_episodes = len(w_runs)
                longest_run_s = max(w_runs, default=0.0)
                s_max = (longest_run_s / af_s) if af_s > 0 else 0.0

                windows.append({
                    "dataset": dataset,
                    "record_id": rec_id,
                    "patient_id": pat_id,
                    "horizon": "12h_nonoverlapping",
                    "window_type": "CONSECUTIVE_WINDOW",
                    "window_index": w_idx,
                    "window_start_s": w_start,
                    "window_end_s": w_end,
                    "window_duration_s": h_dur,
                    "total_af_s": af_s,
                    "af_burden": burden,
                    "unknown_s": unknown_s,
                    "n_af_episodes": n_episodes,
                    "longest_af_run_s": longest_run_s,
                    "s_max": s_max,
                    "is_af_positive": (af_s > 0),
                    "is_density_defined": (0.0 < burden < 1.0) and (unknown_s < 1.0),
                    "is_multi_episode": (n_episodes >= 2),
                    "is_continuous_af": (burden >= 0.999),
                    "is_zero_af": (burden == 0.0)
                })

        # 4. Full recording window
        full_runs = []
        for _, r in rec_runs.iterrows():
            full_runs.append(r["duration_s"])
        full_af_s = sum(full_runs)
        full_burden = full_af_s / rec_dur_s
        full_longest = max(full_runs, default=0.0)
        full_s_max = (full_longest / full_af_s) if full_af_s > 0 else 0.0

        windows.append({
            "dataset": dataset,
            "record_id": rec_id,
            "patient_id": pat_id,
            "horizon": "full_recording",
            "window_type": "FULL_RECORDING",
            "window_index": 0,
            "window_start_s": 0.0,
            "window_end_s": rec_dur_s,
            "window_duration_s": rec_dur_s,
            "total_af_s": full_af_s,
            "af_burden": full_burden,
            "unknown_s": 0.0,
            "n_af_episodes": len(full_runs),
            "longest_af_run_s": full_longest,
            "s_max": full_s_max,
            "is_af_positive": (full_af_s > 0),
            "is_density_defined": (0.0 < full_burden < 1.0),
            "is_multi_episode": (len(full_runs) >= 2),
            "is_continuous_af": (full_burden >= 0.999),
            "is_zero_af": (full_burden == 0.0)
        })

    df_windows = pd.DataFrame(windows)
    df_elig = pd.DataFrame(eligibility_records)

    out_win = manifests_dir / "observation_windows.parquet"
    out_elig = manifests_dir / "eligibility_flow.csv"

    df_windows.to_parquet(out_win, index=False)
    df_elig.to_csv(out_elig, index=False)

    print(f"\nSaved observation windows to {out_win} (shape: {df_windows.shape})")
    print(f"Saved eligibility flow to {out_elig} (shape: {df_elig.shape})")

    # Print summary flow table for Primary 12h horizon
    print("\n" + "=" * 80)
    print("PRIMARY 12-HOUR HORIZON ELIGIBILITY FLOW (MONITORING ANCHORED)")
    print("=" * 80)
    flow_12h = df_windows[
        (df_windows["horizon"] == "12h") &
        (df_windows["window_type"] == "OBSERVED_MONITORING_ANCHORED")
    ]
    for ds in ["LTAFDB", "SHDB-AF", "IRIDIA-AF"]:
        sub = flow_12h[flow_12h["dataset"] == ds]
        print(f"\nCohort: {ds}")
        print(f"  Duration eligible (>= 12h): {len(sub)}")
        print(f"  AF positive (b > 0)       : {sub['is_af_positive'].sum()}")
        print(f"  Zero AF (b = 0)           : {sub['is_zero_af'].sum()}")
        print(f"  Continuous AF (b >= 0.999): {sub['is_continuous_af'].sum()}")
        print(f"  Density defined (0 < b < 1): {sub['is_density_defined'].sum()}")
        print(f"  Multi-episode (>= 2 runs) : {sub['is_multi_episode'].sum()}")
        print(f"  Single-run (1 run)        : {(sub['n_af_episodes'] == 1).sum()}")

if __name__ == "__main__":
    main()
