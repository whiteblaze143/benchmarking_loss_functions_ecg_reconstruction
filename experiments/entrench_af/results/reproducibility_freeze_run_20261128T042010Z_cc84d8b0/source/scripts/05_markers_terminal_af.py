#!/usr/bin/env python3
"""Stage 5: Terminal AF Markers for ENTRENCH-AF.

For each event in events_g0.parquet:
  - Window: last 300s before t0 (or whole run with terminal_partial=1 if dur < 300s).
  - Computes: AFR_DF_mean, AFR_DF_slope (Hz/min), fwave_SampEn_mean,
    fwave_SampEn_slope (per min), VCC, RR_mean, RR_min, RR_SampEn,
    bsqi_terminal, sqi_pass_terminal (using Stage 3 thresholds).
  - Acceptance check: AFR DF in [3, 12] Hz for >= 95% of passing windows.
  - Saves: <run_dir>/stage5/terminal_markers.parquet and summary CSV.
"""

from __future__ import annotations

import datetime
import hashlib
import os
import subprocess
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import wfdb

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(SCRIPTS_DIR))
import importlib.util

def _load_stage3():
    spec = importlib.util.spec_from_file_location("stage3_mod", SCRIPTS_DIR / "03_sqi_calibration.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

S3 = _load_stage3()

LTAF_DIR = Path("/data/mithunmanivannan/datasets/ltafdb_1.0.0_reference")
SHDB_DIR = Path("/data/mithunmanivannan/datasets/shdb_af")
IRIDIA_DIR = Path("/data/mithunmanivannan/datasets/iridia_af_8405941_v1.0.1")


def resolve_run_dir() -> Path:
    for candidate in [EXP_DIR / "LATEST_RUN.txt", EXP_DIR / "results" / "LATEST_RUN.txt"]:
        if candidate.exists():
            txt = candidate.read_text().strip()
            if txt:
                p = Path(txt)
                if not p.is_absolute():
                    p = EXP_DIR / p
                return p
    return EXP_DIR / "results"


def get_provenance() -> Dict[str, str]:
    try:
        git_hash = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        git_hash = "unknown"
    defs_path = EXP_DIR / "config" / "definitions.yaml"
    definitions_hash = hashlib.sha256(defs_path.read_bytes()).hexdigest()[:12] if defs_path.exists() else "missing"
    run_timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {"git_hash": git_hash, "definitions_hash": definitions_hash, "run_timestamp": run_timestamp}


def load_stage3_thresholds(run_dir: Path) -> Dict[str, Tuple[float, bool]]:
    p = run_dir / "stage3" / "sqi_thresholds.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing required Stage 3 SQI threshold file in run_dir: {p}")
    df = pd.read_csv(p)
    out = {}
    for _, r in df.iterrows():
        tau = float(r["bsqi_threshold"]) if pd.notna(r["bsqi_threshold"]) else np.nan
        out[str(r["marker"])] = (tau, bool(r["dropped"]))
    return out


def _normalize_cohort(c: str) -> str:
    cu = str(c).upper()
    if "IRIDIA" in cu:
        return "IRIDIA"
    if "SHDB" in cu:
        return "SHDB"
    if "LTAF" in cu:
        return "LTAFDB"
    return cu


class RecordReader:
    """Per-record ECG reader with IRIDIA HDF5 index caching."""

    def __init__(self, cohort: str, record_id: str):
        self.cohort = _normalize_cohort(cohort)
        self.record_id = str(record_id)
        self.fs = 128.0 if self.cohort == "LTAFDB" else 200.0
        self.iridia_meta: Optional[List[Tuple[Path, int, int]]] = None
        if self.cohort == "IRIDIA":
            rec_dir = IRIDIA_DIR / "ecg_hdf5" / self.record_id
            files = sorted(rec_dir.glob("*.h5"))
            meta = []
            cum = 0
            for f in files:
                with h5py.File(f, "r") as h5:
                    n = len(h5["ecg"])
                    meta.append((f, cum, cum + n))
                    cum += n
            self.iridia_meta = meta

    def read_segment(self, start_s: float, end_s: float) -> Optional[np.ndarray]:
        s0 = max(0, int(round(start_s * self.fs)))
        s1 = int(round(end_s * self.fs))
        if s1 <= s0 + int(self.fs * 5):
            return None
        try:
            if self.cohort == "LTAFDB":
                rec_str = f"{int(self.record_id):02d}" if self.record_id.isdigit() else self.record_id
                rec = wfdb.rdrecord(str(LTAF_DIR / rec_str), sampfrom=s0, sampto=s1)
                return rec.p_signal[:, 0].astype(np.float64)
            elif self.cohort == "SHDB":
                rec_str = f"{int(self.record_id):03d}" if self.record_id.isdigit() else self.record_id
                rec = wfdb.rdrecord(str(SHDB_DIR / rec_str), sampfrom=s0, sampto=s1)
                return rec.p_signal[:, 0].astype(np.float64)
            elif self.cohort == "IRIDIA" and self.iridia_meta:
                parts = []
                for fpath, c0, c1 in self.iridia_meta:
                    if s0 < c1 and s1 > c0:
                        a = max(0, s0 - c0)
                        b = min(c1 - c0, s1 - c0)
                        with h5py.File(fpath, "r") as h5:
                            parts.append(h5["ecg"][a:b, 0])
                if not parts:
                    return None
                return np.concatenate(parts, axis=0).astype(np.float64)
        except Exception:
            return None
        return None


def _process_record_group(
    args: Tuple[str, str, List[Dict[str, Any]], float]
) -> List[Dict[str, Any]]:
    cohort, record_id, events, bsqi_gate = args
    reader = RecordReader(cohort, record_id)
    fs = reader.fs
    results = []

    for ev in events:
        event_id = ev["event_id"]
        t0 = float(ev["t0_sec"])
        dur = float(ev["dur_sec"])
        win_len = min(300.0, dur)
        terminal_partial = int(dur < 300.0)
        w_start = max(0.0, t0 - win_len)
        w_end = t0

        ecg = reader.read_segment(w_start, w_end)
        if ecg is None or len(ecg) < int(fs * 10):
            results.append({
                "event_id": event_id,
                "cohort": cohort,
                "record_id": record_id,
                "terminal_partial": terminal_partial,
                "terminal_window_sec": win_len,
                "AFR_DF_mean": np.nan,
                "AFR_DF_slope": np.nan,
                "fwave_SampEn_mean": np.nan,
                "fwave_SampEn_slope": np.nan,
                "VCC": np.nan,
                "RR_mean": np.nan,
                "RR_min": np.nan,
                "RR_SampEn": np.nan,
                "bsqi_terminal": 0.0,
                "sqi_pass_terminal": 0,
            })
            continue

        bsqi, qrs = S3.compute_bsqi(ecg, fs)
        # RR markers over full terminal window
        rr_m = S3.compute_rr_markers(qrs, fs)
        if len(qrs) >= 2:
            rr_ms = np.diff(qrs) / fs * 1000.0
            valid_rr = rr_ms[(rr_ms >= 200.0) & (rr_ms <= 3000.0)]
            rr_mean = float(np.mean(valid_rr)) if len(valid_rr) > 0 else np.nan
            rr_min = float(np.min(valid_rr)) if len(valid_rr) > 0 else np.nan
        else:
            rr_mean, rr_min = np.nan, np.nan

        # Sub-windows of 60s (or single window if < 90s) for AFR_DF and fwave_SampEn mean & slope
        sub_len = 60.0 if (len(ecg) / fs) >= 90.0 else (len(ecg) / fs)
        sub_samp = int(round(sub_len * fs))
        n_sub = max(1, len(ecg) // sub_samp)
        dfs, fws, t_mins = [], [], []
        for k in range(n_sub):
            a = k * sub_samp
            b = min(len(ecg), a + sub_samp)
            if b - a < int(fs * 10):
                continue
            sub_qrs = qrs[(qrs >= a) & (qrs < b)] - a
            df_hz, fw_se = S3.compute_afr_and_fwave_sampen(ecg[a:b], sub_qrs, fs)
            t_mid_min = ((a + b) / 2.0) / fs / 60.0
            if np.isfinite(df_hz):
                dfs.append(df_hz)
                t_mins.append(t_mid_min)
            if np.isfinite(fw_se):
                fws.append((t_mid_min, fw_se))

        afr_mean = float(np.mean(dfs)) if dfs else np.nan
        afr_slope = float(np.polyfit(t_mins, dfs, 1)[0]) if len(dfs) >= 2 else np.nan
        fw_mean = float(np.mean([v for _, v in fws])) if fws else np.nan
        fw_slope = float(np.polyfit([t for t, _ in fws], [v for _, v in fws], 1)[0]) if len(fws) >= 2 else np.nan

        sqi_pass = int(bsqi >= bsqi_gate)
        results.append({
            "event_id": event_id,
            "cohort": cohort,
            "record_id": record_id,
            "terminal_partial": terminal_partial,
            "terminal_window_sec": float(len(ecg) / fs),
            "AFR_DF_mean": afr_mean,
            "AFR_DF_slope": afr_slope,
            "fwave_SampEn_mean": fw_mean,
            "fwave_SampEn_slope": fw_slope,
            "VCC": rr_m["VCC"],
            "RR_mean": rr_mean,
            "RR_min": rr_min,
            "RR_SampEn": rr_m["RR_SampEn"],
            "bsqi_terminal": float(bsqi),
            "sqi_pass_terminal": sqi_pass,
        })

    return results


def main():
    run_dir = resolve_run_dir()
    events_path = run_dir / "stage4" / "events_g0.parquet"
    if not events_path.exists():
        raise FileNotFoundError(f"Missing {events_path}. Wait for Stage 4 (@foundation).")

    out_dir = run_dir / "stage5"
    out_dir.mkdir(parents=True, exist_ok=True)
    prov = get_provenance()

    df_ev = pd.read_parquet(events_path)
    print(f"[Stage 5] Loaded {len(df_ev)} events from {events_path}")

    # Resolve column names flexibly
    col_t0 = next(c for c in ["t0_sec", "end_sec", "end_time_s", "t0"] if c in df_ev.columns)
    col_dur = next(c for c in ["dur_sec", "duration_sec", "duration_s", "af_dur_sec"] if c in df_ev.columns)
    col_cohort = next(c for c in ["cohort", "dataset"] if c in df_ev.columns)
    col_rec = next(c for c in ["record_id", "record"] if c in df_ev.columns)
    if "event_id" not in df_ev.columns:
        df_ev["event_id"] = np.arange(len(df_ev))

    thresholds = load_stage3_thresholds(run_dir)
    active_gates = [
        tau for m, (tau, dropped) in thresholds.items()
        if m in {"AFR_DF", "fwave_SampEn", "VCC", "RR_SampEn"} and not dropped and np.isfinite(tau)
    ]
    if not active_gates:
        raise ValueError(f"No valid Stage 3 terminal SQI thresholds found in {run_dir / 'stage3' / 'sqi_thresholds.csv'}")
    bsqi_gate = max(active_gates)
    print(f"[Stage 5] Terminal bSQI gate = {bsqi_gate:.2f} (from Stage 3 thresholds)")

    # Group by (cohort, record_id) for parallel execution
    groups = []
    for (cohort, rec_id), g in df_ev.groupby([col_cohort, col_rec], sort=False):
        evs = [
            {
                "event_id": r["event_id"],
                "t0_sec": float(r[col_t0]),
                "dur_sec": float(r[col_dur]),
            }
            for _, r in g.iterrows()
        ]
        groups.append((str(cohort), str(rec_id), evs, bsqi_gate))

    n_workers = min(8, max(1, (os.cpu_count() or 4) // 2))
    all_rows = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(_process_record_group, grp) for grp in groups]
        for f in as_completed(futs):
            all_rows.extend(f.result())

    df_out = pd.DataFrame(all_rows)
    # Preserve original event_id order
    df_out = df_ev[["event_id"]].merge(df_out, on="event_id", how="left")
    for k, v in prov.items():
        df_out[k] = v

    out_pq = out_dir / "terminal_markers.parquet"
    df_out.to_parquet(out_pq, index=False)

    # Acceptance check: AFR DF in 3-12 Hz for >= 95% of passing windows
    passing = df_out[df_out["sqi_pass_terminal"] == 1]
    afr_in_band = passing["AFR_DF_mean"].between(3.0, 12.0).mean() if len(passing) > 0 else 0.0
    summary = pd.DataFrame([{
        "n_events": len(df_out),
        "n_sqi_pass": int(len(passing)),
        "sqi_pass_rate": float(len(passing) / max(1, len(df_out))),
        "bsqi_gate": bsqi_gate,
        "afr_df_in_3_12hz_pct": float(afr_in_band * 100.0),
        "afr_df_median_hz": float(passing["AFR_DF_mean"].median()) if len(passing) else np.nan,
        "vcc_median_pct": float(passing["VCC"].median()) if len(passing) else np.nan,
        **prov,
    }])
    summary.to_csv(out_dir / "terminal_markers_summary.csv", index=False)
    print(f"[Stage 5] Saved {out_pq}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
