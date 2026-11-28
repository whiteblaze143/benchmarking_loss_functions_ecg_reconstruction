#!/usr/bin/env python3
"""Stage 6: Post-Termination Sinus Markers for ENTRENCH-AF.

For each event in events_g0.parquet:
  - Window A: [t0, t0 + 120s], truncated at t0 + time_to_reinit_sec if earlier
    (postA_truncated=1). Computes: PMV_A, HR_mean_A, RMSSD_A, PAC_count_A,
    first_PAC_coupling_ms, first_PAC_prematurity_index, repetitive_PACs_A
    (>=3 PACs within 24 cycles), postA_insufficient (<20 sinus beats),
    bsqi_postA, sqi_pass_postA.
  - Window D: [t0 + 600s, t0 + 720s], only when has_postD_window is True.
    Computes: PMV_D, HR_mean_D, RMSSD_D, bsqi_postD, sqi_pass_postD.
  - Applies Stage 3 SQI gates.
  - Saves: <run_dir>/stage6/post_markers.parquet and summary CSV.
"""

from __future__ import annotations

import datetime
import hashlib
import importlib.util
import os
import subprocess
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent

def _load_mod(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

S3 = _load_mod("stage3_mod", SCRIPTS_DIR / "03_sqi_calibration.py")
S5 = _load_mod("stage5_mod", SCRIPTS_DIR / "05_markers_terminal_af.py")


def compute_pac_metrics(qrs: np.ndarray, fs: float) -> Dict[str, Any]:
    """Compute PAC count, first PAC coupling (ms), prematurity index, and repetitive PACs (>=3 in 24 cycles)."""
    if len(qrs) < 7:
        return {
            "n_sinus_beats": len(qrs),
            "PAC_count_A": 0,
            "first_PAC_coupling_ms": np.nan,
            "first_PAC_prematurity_index": np.nan,
            "repetitive_PACs_A": 0,
        }
    rr_ms = np.diff(qrs) / fs * 1000.0
    valid_rr = rr_ms[(rr_ms >= 200.0) & (rr_ms <= 3000.0)]
    if len(valid_rr) < 6:
        return {
            "n_sinus_beats": len(valid_rr),
            "PAC_count_A": 0,
            "first_PAC_coupling_ms": np.nan,
            "first_PAC_prematurity_index": np.nan,
            "repetitive_PACs_A": 0,
        }

    pac_flags = np.zeros(len(valid_rr), dtype=int)
    first_coupling = np.nan
    first_prem = np.nan
    for idx in range(5, len(valid_rr)):
        local_med = float(np.median(valid_rr[idx - 5 : idx]))
        if local_med > 0 and 250.0 <= valid_rr[idx] < 0.80 * local_med:
            pac_flags[idx] = 1
            if np.isnan(first_coupling):
                first_coupling = float(valid_rr[idx])
                first_prem = float(valid_rr[idx] / local_med)

    pac_count = int(np.sum(pac_flags))
    n_sinus = int(len(valid_rr) - pac_count)
    # Repetitive PACs: >= 3 PACs within any 24-cycle window
    rep_pacs = 0
    if pac_count >= 3:
        csum = np.concatenate([[0], np.cumsum(pac_flags)])
        win = min(24, len(pac_flags))
        if np.any((csum[win:] - csum[:-win]) >= 3):
            rep_pacs = 1

    return {
        "n_sinus_beats": n_sinus,
        "PAC_count_A": pac_count,
        "first_PAC_coupling_ms": first_coupling,
        "first_PAC_prematurity_index": first_prem,
        "repetitive_PACs_A": rep_pacs,
    }


def _process_record_group(
    args: Tuple[str, str, List[Dict[str, Any]], float]
) -> List[Dict[str, Any]]:
    cohort, record_id, events, bsqi_gate = args
    reader = S5.RecordReader(cohort, record_id)
    fs = reader.fs
    results = []

    for ev in events:
        event_id = ev["event_id"]
        t0 = float(ev["t0_sec"])
        ttr = ev["time_to_reinit_sec"]
        has_postD = bool(ev["has_postD_window"])

        # Window A: [t0, t0 + 120s], truncated at reinit if earlier
        if np.isfinite(ttr) and ttr < 120.0:
            winA_len = max(0.0, float(ttr))
            postA_truncated = 1
        else:
            winA_len = 120.0
            postA_truncated = 0

        ecg_A = reader.read_segment(t0, t0 + winA_len) if winA_len >= 10.0 else None
        ref_energy_A = np.nan

        if ecg_A is not None and len(ecg_A) >= int(fs * 10):
            bsqi_A, qrs_A = S3.compute_bsqi(ecg_A, fs)
            pac_m = compute_pac_metrics(qrs_A, fs)
            postA_insufficient = int(pac_m["n_sinus_beats"] < 20)
            rr_ms_A = np.diff(qrs_A) / fs * 1000.0 if len(qrs_A) >= 2 else np.empty(0)
            hrv_A = S3.calc_hrv_and_vcc(rr_ms_A) if len(rr_ms_A) >= 4 else None
            hr_mean_A = float(hrv_A["mean_hr_bpm"]) if hrv_A else np.nan
            rmssd_A = float(hrv_A["rmssd_ms"]) if hrv_A else np.nan
            pmv_A, ref_energy_A = S3.compute_pmv_single_window(ecg_A, qrs_A, fs) if not postA_insufficient else (np.nan, np.nan)
            sqi_pass_A = int(bsqi_A >= bsqi_gate)
        else:
            bsqi_A = 0.0
            sqi_pass_A = 0
            postA_insufficient = 1
            pac_m = {
                "n_sinus_beats": 0,
                "PAC_count_A": np.nan,
                "first_PAC_coupling_ms": np.nan,
                "first_PAC_prematurity_index": np.nan,
                "repetitive_PACs_A": np.nan,
            }
            hr_mean_A, rmssd_A, pmv_A = np.nan, np.nan, np.nan

        # Window D: [t0 + 600s, t0 + 720s], only when has_postD_window
        pmv_D, hr_mean_D, rmssd_D, bsqi_D, sqi_pass_D = np.nan, np.nan, np.nan, np.nan, 0
        if has_postD:
            ecg_D = reader.read_segment(t0 + 600.0, t0 + 720.0)
            if ecg_D is not None and len(ecg_D) >= int(fs * 10):
                bsqi_D, qrs_D = S3.compute_bsqi(ecg_D, fs)
                rr_ms_D = np.diff(qrs_D) / fs * 1000.0 if len(qrs_D) >= 2 else np.empty(0)
                hrv_D = S3.calc_hrv_and_vcc(rr_ms_D) if len(rr_ms_D) >= 4 else None
                hr_mean_D = float(hrv_D["mean_hr_bpm"]) if hrv_D else np.nan
                rmssd_D = float(hrv_D["rmssd_ms"]) if hrv_D else np.nan
                pmv_D, _ = S3.compute_pmv_single_window(
                    ecg_D, qrs_D, fs, ref_energy=ref_energy_A if np.isfinite(ref_energy_A) else None
                )
                sqi_pass_D = int(bsqi_D >= bsqi_gate)

        results.append({
            "event_id": event_id,
            "cohort": cohort,
            "record_id": record_id,
            "postA_window_sec": winA_len,
            "postA_truncated": postA_truncated,
            "n_sinus_beats_A": pac_m["n_sinus_beats"],
            "postA_insufficient": postA_insufficient,
            "PMV_A": pmv_A,
            "HR_mean_A": hr_mean_A,
            "RMSSD_A": rmssd_A,
            "PAC_count_A": pac_m["PAC_count_A"],
            "first_PAC_coupling_ms": pac_m["first_PAC_coupling_ms"],
            "first_PAC_prematurity_index": pac_m["first_PAC_prematurity_index"],
            "repetitive_PACs_A": pac_m["repetitive_PACs_A"],
            "bsqi_postA": float(bsqi_A),
            "sqi_pass_postA": sqi_pass_A,
            "has_postD_window": int(has_postD),
            "PMV_D": pmv_D,
            "HR_mean_D": hr_mean_D,
            "RMSSD_D": rmssd_D,
            "bsqi_postD": float(bsqi_D) if np.isfinite(bsqi_D) else np.nan,
            "sqi_pass_postD": sqi_pass_D,
        })

    return results


def main():
    run_dir = S5.resolve_run_dir()
    events_path = run_dir / "stage4" / "events_g0.parquet"
    if not events_path.exists():
        raise FileNotFoundError(f"Missing {events_path}. Wait for Stage 4 (@foundation).")

    out_dir = run_dir / "stage6"
    out_dir.mkdir(parents=True, exist_ok=True)
    prov = S5.get_provenance()

    df_ev = pd.read_parquet(events_path)
    print(f"[Stage 6] Loaded {len(df_ev)} events from {events_path}")

    col_t0 = next(c for c in ["t0_sec", "end_sec", "end_time_s", "t0"] if c in df_ev.columns)
    col_ttr = next((c for c in ["time_to_reinit_sec", "ttr_sec"] if c in df_ev.columns), None)
    col_postD = next((c for c in ["has_postD_window", "has_postd_window"] if c in df_ev.columns), None)
    col_cohort = next(c for c in ["cohort", "dataset"] if c in df_ev.columns)
    col_rec = next(c for c in ["record_id", "record"] if c in df_ev.columns)
    if "event_id" not in df_ev.columns:
        df_ev["event_id"] = np.arange(len(df_ev))

    thresholds = S5.load_stage3_thresholds(run_dir)
    active_gates = [
        tau for m, (tau, dropped) in thresholds.items()
        if m in {"PMV", "PAC_coupling", "RMSSD"} and not dropped and np.isfinite(tau)
    ]
    if not active_gates:
        raise ValueError(f"No valid Stage 3 post-termination SQI thresholds found in {run_dir / 'stage3' / 'sqi_thresholds.csv'}")
    bsqi_gate = max(active_gates)
    print(f"[Stage 6] Post-termination bSQI gate = {bsqi_gate:.2f} (from Stage 3 thresholds)")

    groups = []
    for (cohort, rec_id), g in df_ev.groupby([col_cohort, col_rec], sort=False):
        evs = []
        for _, r in g.iterrows():
            ttr = float(r[col_ttr]) if (col_ttr and pd.notna(r[col_ttr])) else np.nan
            has_d = bool(r[col_postD]) if col_postD else (np.isfinite(ttr) and ttr >= 720.0)
            evs.append({
                "event_id": r["event_id"],
                "t0_sec": float(r[col_t0]),
                "time_to_reinit_sec": ttr,
                "has_postD_window": has_d,
            })
        groups.append((str(cohort), str(rec_id), evs, bsqi_gate))

    n_workers = min(8, max(1, (os.cpu_count() or 4) // 2))
    all_rows = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(_process_record_group, grp) for grp in groups]
        for f in as_completed(futs):
            all_rows.extend(f.result())

    df_out = pd.DataFrame(all_rows)
    df_out = df_ev[["event_id"]].merge(df_out, on="event_id", how="left")
    for k, v in prov.items():
        df_out[k] = v

    out_pq = out_dir / "post_markers.parquet"
    df_out.to_parquet(out_pq, index=False)

    pass_A = df_out[(df_out["sqi_pass_postA"] == 1) & (df_out["postA_insufficient"] == 0)]
    pass_D = df_out[(df_out["has_postD_window"] == 1) & (df_out["sqi_pass_postD"] == 1)]
    summary = pd.DataFrame([{
        "n_events": len(df_out),
        "n_postA_valid": int(len(pass_A)),
        "postA_valid_rate": float(len(pass_A) / max(1, len(df_out))),
        "n_postA_truncated": int(df_out["postA_truncated"].sum()),
        "n_postA_insufficient": int(df_out["postA_insufficient"].sum()),
        "n_postD_eligible": int(df_out["has_postD_window"].sum()),
        "n_postD_valid": int(len(pass_D)),
        "bsqi_gate": bsqi_gate,
        "PMV_A_median": float(pass_A["PMV_A"].median()) if len(pass_A) else np.nan,
        "PMV_D_median": float(pass_D["PMV_D"].median()) if len(pass_D) else np.nan,
        "HR_mean_A_median": float(pass_A["HR_mean_A"].median()) if len(pass_A) else np.nan,
        "RMSSD_A_median": float(pass_A["RMSSD_A"].median()) if len(pass_A) else np.nan,
        **prov,
    }])
    summary.to_csv(out_dir / "post_markers_summary.csv", index=False)
    print(f"[Stage 6] Saved {out_pq}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
