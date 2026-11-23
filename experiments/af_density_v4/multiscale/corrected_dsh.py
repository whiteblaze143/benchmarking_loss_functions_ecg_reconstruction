"""Corrected Density Stabilization Horizon (DSH) & Multiscale AF Density (V4.1).

Fixes all V4 multiscale defects:
1. Strict numeric horizon sorting (seconds): 3600 < 10800 < 21600 < 43200 < 86400 < 172800 < 259200.
2. Independent Non-overlapping Windows (Scheme A) as primary unit for variance and DSH.
3. Explicit sample requirement: >= 3 non-overlapping windows required to compute IQR. If < 3, output INSUFFICIENT_WINDOWS.
4. Sustained stability condition: IQR <= tau at H* AND for all subsequent eligible horizons.
5. Multi-threshold sensitivity analysis (tau in {0.025, 0.05, 0.10}).
6. Explicit tracking of undefined density fractions (burden == 0 or continuous AF).
7. Strict quarantine on test split.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction")
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
V4_DIR = PROJECT_ROOT / "experiments" / "af_density_v4"
REPORTS_DIR = V4_DIR / "reports"
CODEX_DIR = Path("/data/mithunmanivannan/codex_artifacts/af_density_v4")
WAVEFORM_DIR = Path("/data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1")

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
CODEX_DIR.mkdir(parents=True, exist_ok=True)

from experiments.af_density_v4.metrics.af_pattern_metrics import measure_af_pattern, charitos_density


# Numeric horizons in seconds
NUMERIC_HORIZONS: List[Tuple[str, float]] = [
    ("1h", 3600.0),
    ("3h", 10800.0),
    ("6h", 21600.0),
    ("12h", 43200.0),
    ("24h", 86400.0),
    ("48h", 172800.0),
    ("72h", 259200.0),
]


def merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for cur in intervals[1:]:
        prev = merged[-1]
        if cur[0] <= prev[1] + 1e-6:
            merged[-1] = (prev[0], max(prev[1], cur[1]))
        else:
            merged.append(cur)
    return merged


def load_development_reference_intervals() -> Tuple[Dict[str, List[Tuple[float, float]]], Dict[str, float]]:
    manifest_csv = WAVEFORM_DIR / "file_manifest.csv"
    ref_csv = WAVEFORM_DIR / "reference_intervals.csv"

    manifest = list(csv.DictReader(manifest_csv.open()))
    fi_to_info = {
        int(r["file_index"]): {
            "patient_id": r["patient_id"],
            "partition": r["partition"],
            "windows": int(r["windows"]),
            "prefix_samples": float(r.get("excluded_prefix_samples", 0.0)),
            "offset_samples": float(r.get("record_file_offset_samples", 0.0))
        } for r in manifest
    }

    patient_durations: Dict[str, float] = {}
    for r in manifest:
        if r["partition"] != "test":
            pid = r["patient_id"]
            patient_durations[pid] = patient_durations.get(pid, 0.0) + int(r["windows"]) * 30.0

    patient_records: Dict[str, List[Tuple[float, float]]] = {}
    with open(ref_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fi = int(row["file_index"])
            info = fi_to_info.get(fi)
            if not info or info["partition"] == "test":  # Strict quarantine
                continue

            pid = info["patient_id"]
            if pid not in patient_records:
                patient_records[pid] = []

            offset = info["offset_samples"]
            prefix = info["prefix_samples"]
            start_s = max(0.0, (offset + float(row["start_sample"]) - prefix) / 200.0)
            end_s = max(start_s, (offset + float(row["end_sample_exclusive"]) - prefix) / 200.0)
            if end_s > start_s:
                patient_records[pid].append((start_s, end_s))

    clean_patients = {pid: merge_intervals(eps) for pid, eps in patient_records.items()}
    return clean_patients, patient_durations


def run_corrected_multiscale_analysis():
    print("=" * 80)
    print("T-CAIREM V4.1: Corrected Multiscale AF-Density Analysis & DSH Estimation")
    print("=" * 80)

    patient_eps, patient_durations = load_development_reference_intervals()
    print(f"Loaded {len(patient_eps)} development patients with AF reference episodes.")

    # Storage for all window measurements
    window_records: List[Dict[str, Any]] = []

    for pid, eps in sorted(patient_eps.items()):
        total_T = patient_durations.get(pid, 0.0)
        if total_T <= 0.0:
            continue

        for h_name, h_sec in NUMERIC_HORIZONS:
            if h_sec > total_T:
                continue

            # -------------------------------------------------------------
            # Scheme A: Strict Independent Non-Overlapping Windows
            # -------------------------------------------------------------
            n_blocks = int(np.floor(total_T / h_sec))
            for b_idx in range(n_blocks):
                w_start = b_idx * h_sec
                w_end = w_start + h_sec

                # Clip intervals to window
                w_eps = []
                for a, b in eps:
                    if b <= w_start or a >= w_end:
                        continue
                    cl_a = max(w_start, a) - w_start
                    cl_b = min(w_end, b) - w_start
                    if cl_b > cl_a:
                        w_eps.append((cl_a, cl_b))

                m = measure_af_pattern(w_eps, h_sec)
                window_records.append({
                    "subject_id": pid,
                    "scheme": "SchemeA_NonOverlapping",
                    "horizon_name": h_name,
                    "horizon_seconds": h_sec,
                    "window_index": b_idx,
                    "start_seconds": w_start,
                    "end_seconds": w_end,
                    "af_burden": m.af_burden,
                    "charitos_density": m.charitos_density,
                    "is_density_defined": bool(m.charitos_density is not None),
                    "undefined_reason": m.undefined_reason.value,
                    "episode_count": m.episode_count,
                    "longest_episode_s": m.longest_episode_s,
                    "transition_count": m.transition_count,
                })

            # -------------------------------------------------------------
            # Scheme B: Sliding Windows (stride = 1 hour / 3600s)
            # -------------------------------------------------------------
            stride_s = 3600.0
            if total_T >= h_sec + stride_s:
                max_start = total_T - h_sec
                n_slides = int(np.floor(max_start / stride_s)) + 1
                for s_idx in range(n_slides):
                    w_start = s_idx * stride_s
                    w_end = w_start + h_sec

                    w_eps = []
                    for a, b in eps:
                        if b <= w_start or a >= w_end:
                            continue
                        cl_a = max(w_start, a) - w_start
                        cl_b = min(w_end, b) - w_start
                        if cl_b > cl_a:
                            w_eps.append((cl_a, cl_b))

                    m = measure_af_pattern(w_eps, h_sec)
                    window_records.append({
                        "subject_id": pid,
                        "scheme": "SchemeB_Sliding1h",
                        "horizon_name": h_name,
                        "horizon_seconds": h_sec,
                        "window_index": s_idx,
                        "start_seconds": w_start,
                        "end_seconds": w_end,
                        "af_burden": m.af_burden,
                        "charitos_density": m.charitos_density,
                        "is_density_defined": bool(m.charitos_density is not None),
                        "undefined_reason": m.undefined_reason.value,
                        "episode_count": m.episode_count,
                        "longest_episode_s": m.longest_episode_s,
                        "transition_count": m.transition_count,
                    })

    df_windows = pd.DataFrame(window_records)
    out_parquet = REPORTS_DIR / "v41_corrected_multiscale_density.parquet"
    out_csv = REPORTS_DIR / "v41_corrected_multiscale_density.csv"
    df_windows.to_parquet(out_parquet, index=False)
    df_windows.to_csv(out_csv, index=False)
    df_windows.to_parquet(CODEX_DIR / "v41_corrected_multiscale_density.parquet", index=False)
    df_windows.to_csv(CODEX_DIR / "v41_corrected_multiscale_density.csv", index=False)
    print(f"Generated {len(df_windows)} window evaluations across schemes.")

    # -----------------------------------------------------------------
    # Compute Rigorous DSH on Independent Non-Overlapping Windows (Scheme A)
    # -----------------------------------------------------------------
    df_scheme_a = df_windows[df_windows["scheme"] == "SchemeA_NonOverlapping"]

    hcdd_summary: List[Dict[str, Any]] = []
    # Map horizon name to seconds for numerical sorting
    h_to_sec = {h_name: sec for h_name, sec in NUMERIC_HORIZONS}

    for (pid, h_name), group in df_scheme_a.groupby(["subject_id", "horizon_name"]):
        h_sec = h_to_sec[h_name]
        n_windows = len(group)
        defined_rows = group[group["is_density_defined"]]
        n_defined = len(defined_rows)
        defined_fraction = n_defined / n_windows if n_windows > 0 else 0.0

        if n_defined < 3:
            # Explicit gate: At least 3 independent defined windows required for IQR
            iqr = np.nan
            med = float(np.median(defined_rows["charitos_density"])) if n_defined > 0 else np.nan
            q25 = np.nan
            q75 = np.nan
            status = "INSUFFICIENT_WINDOWS"
        else:
            vals = defined_rows["charitos_density"].values
            med = float(np.median(vals))
            q25 = float(np.percentile(vals, 25))
            q75 = float(np.percentile(vals, 75))
            iqr = float(q75 - q25)
            status = "EVALUATED"

        hcdd_summary.append({
            "subject_id": pid,
            "horizon_name": h_name,
            "horizon_seconds": h_sec,
            "total_windows": n_windows,
            "defined_windows": n_defined,
            "defined_fraction": round(defined_fraction, 4),
            "density_median": round(med, 4) if np.isfinite(med) else np.nan,
            "density_q25": round(q25, 4) if np.isfinite(q25) else np.nan,
            "density_q75": round(q75, 4) if np.isfinite(q75) else np.nan,
            "density_iqr": round(iqr, 4) if np.isfinite(iqr) else np.nan,
            "evaluation_status": status,
            "stable_tau_025": bool(iqr <= 0.025) if np.isfinite(iqr) else False,
            "stable_tau_050": bool(iqr <= 0.050) if np.isfinite(iqr) else False,
            "stable_tau_100": bool(iqr <= 0.100) if np.isfinite(iqr) else False,
        })

    df_hcdd = pd.DataFrame(hcdd_summary)
    # Sort strictly by subject_id and horizon_seconds
    df_hcdd.sort_values(by=["subject_id", "horizon_seconds"], inplace=True)

    # Determine sustained DSH per patient
    dsh_results: List[Dict[str, Any]] = []

    for pid, group in df_hcdd.groupby("subject_id"):
        group_eval = group[group["evaluation_status"] == "EVALUATED"].copy()
        
        # Helper for sustained stability across subsequent evaluated horizons
        def find_sustained_dsh(tau_col: str) -> str:
            stable_horizons = []
            eval_list = group_eval.to_dict("records")
            for i, row in enumerate(eval_list):
                if row[tau_col]:
                    # Check sustained stability: all subsequent rows with >= 3 windows must also be stable
                    sustained = all(eval_list[j][tau_col] for j in range(i, len(eval_list)))
                    if sustained:
                        stable_horizons.append(row["horizon_name"])
            return stable_horizons[0] if stable_horizons else "NOT_ATTAINED"

        dsh_025 = find_sustained_dsh("stable_tau_025")
        dsh_050 = find_sustained_dsh("stable_tau_050")
        dsh_100 = find_sustained_dsh("stable_tau_100")

        total_hours = patient_durations.get(pid, 0.0) / 3600.0

        dsh_results.append({
            "subject_id": pid,
            "total_monitoring_hours": round(total_hours, 2),
            "evaluated_horizons": list(group_eval["horizon_name"]),
            "dsh_sustained_tau_025": dsh_025,
            "dsh_sustained_tau_050": dsh_050,
            "dsh_sustained_tau_100": dsh_100,
            "highest_evaluated_horizon": group_eval["horizon_name"].iloc[-1] if len(group_eval) > 0 else "NONE",
            "attained_at_all_tau_050": bool(dsh_050 != "NOT_ATTAINED"),
        })

    df_dsh = pd.DataFrame(dsh_results)
    out_dsh = REPORTS_DIR / "v41_corrected_dsh.csv"
    df_dsh.to_csv(out_dsh, index=False)
    df_dsh.to_csv(CODEX_DIR / "v41_corrected_dsh.csv", index=False)

    print(f"\nCorrected DSH summary saved to: {out_dsh}")
    print("DSH Attainment Distribution (tau = 0.05, Sustained, Independent Windows):")
    print(df_dsh["dsh_sustained_tau_050"].value_counts())

    return df_windows, df_hcdd, df_dsh


if __name__ == "__main__":
    run_corrected_multiscale_analysis()
