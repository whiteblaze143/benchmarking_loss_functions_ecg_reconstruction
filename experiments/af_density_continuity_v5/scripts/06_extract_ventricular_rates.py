#!/usr/bin/env python3
"""Script 06: Phase D - Ventricular Physiology Extraction and Validation.
Extracts ventricular beats, computes 30s local ventricular rates within AF runs,
calculates primary physiological endpoint Q_120 = R_120 / M, and validates
QRS detector against LTAFDB reference annotations.
Generates:
  - features/ventricular_rate_intervals.parquet
  - features/patient_window_features.parquet
  - results/ventricular_rate_validation.csv
"""

import os
import sys
import math
import time
from pathlib import Path
import pandas as pd
import numpy as np
import wfdb
import wfdb.processing

root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir))

from src.ventricular_rate import (
    extract_beats_ltafdb,
    extract_beats_shdb,
    extract_beats_iridia,
    compute_local_ventricular_rates,
    compute_physiological_endpoints
)

def validate_xqrs_against_ltafdb(n_test_segments: int = 50, seed: int = 42) -> pd.DataFrame:
    """Validates XQRS detector performance against LTAFDB reference beat annotations."""
    print("Validating XQRS detector against LTAFDB ground-truth beats...")
    ltaf_dir = "/data/mithunmanivannan/datasets/ltafdb_1.0.0_reference"
    runs_df = pd.read_parquet(root_dir / "manifests" / "canonical_af_runs.parquet")
    ltaf_runs = runs_df[(runs_df["dataset"] == "LTAFDB") & (runs_df["duration_s"] >= 60.0)].copy()

    np.random.seed(seed)
    sample_runs = ltaf_runs.sample(min(n_test_segments, len(ltaf_runs)), random_state=seed)

    records = []
    for _, run in sample_runs.iterrows():
        rec_id = str(run["record_id"])
        s_s = float(run["start_time_s"])
        # Take 60s slice
        e_s = s_s + 60.0
        fs = 128.0
        s_samp = int(round(s_s * fs))
        e_samp = int(round(e_s * fs))

        try:
            # Reference beats
            ref_beats, _, _ = extract_beats_ltafdb(rec_id, ltaf_dir, s_samp, e_samp)
            # Read signal for channel 0
            rec = wfdb.rdrecord(os.path.join(ltaf_dir, rec_id), sampfrom=s_samp, sampto=e_samp)
            sig = rec.p_signal[:, 0]
            # Detector beats
            det_beats = wfdb.processing.xqrs_detect(sig, fs=fs, verbose=False)
            det_global = det_beats + s_samp

            # Match within 50ms tolerance (0.050 * fs = 6.4 samples)
            tol_samp = int(round(0.050 * fs))
            matched_ref = 0
            for rb in ref_beats:
                if np.any(np.abs(det_global - rb) <= tol_samp):
                    matched_ref += 1

            matched_det = 0
            for db in det_global:
                if np.any(np.abs(ref_beats - db) <= tol_samp):
                    matched_det += 1

            n_ref = len(ref_beats)
            n_det = len(det_global)

            se = matched_ref / n_ref if n_ref > 0 else 1.0
            ppv = matched_det / n_det if n_det > 0 else 1.0
            ref_rate = (n_ref / 60.0) * 60.0
            det_rate = (n_det / 60.0) * 60.0
            rate_err = abs(det_rate - ref_rate)

            records.append({
                "record_id": rec_id,
                "n_ref_beats": n_ref,
                "n_det_beats": n_det,
                "sensitivity": se,
                "positive_predictive_value": ppv,
                "ref_rate_bpm": ref_rate,
                "det_rate_bpm": det_rate,
                "rate_error_bpm": rate_err
            })
        except Exception:
            continue

    df_val = pd.DataFrame(records)
    print(f"LTAFDB Beat Validation: Mean Se = {df_val['sensitivity'].mean():.4f}, Mean PPV = {df_val['positive_predictive_value'].mean():.4f}, Mean Rate Error = {df_val['rate_error_bpm'].mean():.2f} bpm")
    return df_val


def main():
    manifests_dir = root_dir / "manifests"
    features_dir = root_dir / "features"
    results_dir = root_dir / "results"

    print("=" * 80)
    print("PHASE D: VENTRICULAR PHYSIOLOGY EXTRACTION (30s LOCAL RATE)")
    print("=" * 80)

    # 1. Detector validation against LTAFDB
    df_val = validate_xqrs_against_ltafdb(n_test_segments=60)
    df_val.to_csv(results_dir / "ventricular_rate_validation.csv", index=False)

    # Load canonical runs, windows, and density features
    df_runs = pd.read_parquet(manifests_dir / "canonical_af_runs.parquet")
    df_windows = pd.read_parquet(manifests_dir / "observation_windows.parquet")
    df_dens = pd.read_parquet(features_dir / "density_reference_features.parquet")

    # Filter to density-defined windows
    df_windows = df_windows[df_windows["is_density_defined"]].copy()
    print(f"Extracting ventricular rates for {len(df_windows)} density-defined observation windows...")

    # Pre-cache beats by (dataset, record_id)
    ltaf_beats_cache = {}
    shdb_beats_cache = {}
    iridia_run_cache = {}

    print("Loading beats for LTAFDB...")
    ltaf_dir = "/data/mithunmanivannan/datasets/ltafdb_1.0.0_reference"
    for rec_id in df_windows[df_windows["dataset"] == "LTAFDB"]["record_id"].unique():
        b_samples, _, fs = extract_beats_ltafdb(rec_id, ltaf_dir)
        ltaf_beats_cache[rec_id] = (b_samples, fs)

    print("Loading beats for SHDB-AF...")
    shdb_dir = "/data/mithunmanivannan/datasets/shdb_af"
    for rec_id in df_windows[df_windows["dataset"] == "SHDB-AF"]["record_id"].unique():
        b_samples, _, fs = extract_beats_shdb(rec_id, shdb_dir)
        shdb_beats_cache[rec_id] = (b_samples, fs)

    # Compute rate intervals
    all_rate_intervals = []
    window_physiology = []

    print("Computing local 30s ventricular rates...")
    for idx, w in df_windows.iterrows():
        dataset = w["dataset"]
        rec_id = str(w["record_id"])
        pat_id = str(w["patient_id"])
        horizon = w["horizon"]
        w_type = w["window_type"]
        w_idx = int(w["window_index"])
        w_start = float(w["window_start_s"])
        w_end = float(w["window_end_s"])
        w_dur = float(w["window_duration_s"])
        total_af_s = float(w["total_af_s"])

        # Fetch runs in this window
        rec_runs = df_runs[(df_runs["dataset"] == dataset) & (df_runs["record_id"] == rec_id)]
        w_rate_intervals = []

        for _, r in rec_runs.iterrows():
            rs = max(w_start, float(r["start_time_s"]))
            re = min(w_end, float(r["end_time_s"]))
            dur = re - rs
            if dur < 30.0:
                continue

            if dataset == "LTAFDB":
                b_samples, fs = ltaf_beats_cache.get(rec_id, (np.array([]), 128.0))
            elif dataset == "SHDB-AF":
                b_samples, fs = shdb_beats_cache.get(rec_id, (np.array([]), 200.0))
            else:  # IRIDIA-AF
                cache_key = (rec_id, round(rs, 1), round(re, 1))
                if cache_key in iridia_run_cache:
                    b_samples, fs = iridia_run_cache[cache_key]
                else:
                    b_samples, fs = extract_beats_iridia(rec_id, rs, re)
                    iridia_run_cache[cache_key] = (b_samples, fs)

            if len(b_samples) > 0:
                r_itvs = compute_local_ventricular_rates(b_samples, fs, rs, re, window_s=30.0)
                for r_itv in r_itvs:
                    r_itv["dataset"] = dataset
                    r_itv["record_id"] = rec_id
                    r_itv["patient_id"] = pat_id
                    r_itv["window_horizon"] = horizon
                    w_rate_intervals.append(r_itv)
                    all_rate_intervals.append(r_itv)

        # Compute physiological endpoints for window
        phys = compute_physiological_endpoints(w_rate_intervals, total_af_s, w_dur)
        phys["dataset"] = dataset
        phys["record_id"] = rec_id
        phys["patient_id"] = pat_id
        phys["horizon"] = horizon
        phys["window_type"] = w_type
        phys["window_index"] = w_idx
        window_physiology.append(phys)

    df_rate_itvs = pd.DataFrame(all_rate_intervals)
    df_phys = pd.DataFrame(window_physiology)

    out_rates_parquet = features_dir / "ventricular_rate_intervals.parquet"
    df_rate_itvs.to_parquet(out_rates_parquet, index=False)
    print(f"Saved {len(df_rate_itvs)} 30s ventricular rate intervals to {out_rates_parquet}")

    # Merge physiology with density features to create patient_window_features.parquet
    df_patient_window = pd.merge(
        df_dens,
        df_phys,
        on=["dataset", "record_id", "patient_id", "horizon", "window_type", "window_index"],
        how="inner"
    )

    out_patient_win = features_dir / "patient_window_features.parquet"
    df_patient_window.to_parquet(out_patient_win, index=False)
    print(f"Saved patient window features to {out_patient_win} (shape: {df_patient_window.shape})")

    # Print summary
    p12 = df_patient_window[
        (df_patient_window["horizon"] == "12h") &
        (df_patient_window["window_type"] == "OBSERVED_MONITORING_ANCHORED")
    ]
    print("\n" + "=" * 80)
    print("PRIMARY 12-HOUR HORIZON VENTRICULAR PHYSIOLOGY SUMMARY")
    print("=" * 80)
    for ds in ["LTAFDB", "SHDB-AF", "IRIDIA-AF"]:
        sub = p12[p12["dataset"] == ds]
        print(f"Cohort {ds:10s}: N={len(sub)} | Rate coverage: {sub['rate_coverage_fraction'].mean()*100:.1f}% | Median Q120: {sub['q_120'].median():.4f} | Mean AF rate: {sub['mean_af_rate_bpm'].mean():.1f} bpm")

if __name__ == "__main__":
    main()
