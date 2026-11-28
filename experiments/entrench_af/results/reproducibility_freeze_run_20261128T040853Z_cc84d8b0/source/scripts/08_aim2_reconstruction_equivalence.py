#!/usr/bin/env python3
"""
Aim 2: Reconstruction Equivalence (Spitaler Daily-Burden Emulation vs True Runs)
Script: experiments/entrench_af/scripts/08_aim2_reconstruction_equivalence.py

Provides reusable pure function `emulate_spitaler_episodes(runs_df, rules)`
importable by tests/ and downstream scripts.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import importlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, special

try:
    import yaml
except ImportError:
    yaml = None

try:
    from lifelines import KaplanMeierFitter, WeibullFitter
except ImportError:
    KaplanMeierFitter = None
    WeibullFitter = None

try:
    import wfdb
except ImportError:
    wfdb = None

# Ensure scripts/ is on sys.path for optional common.py import
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

SEED = 20260101


# ---------------------------------------------------------------------------
# Provenance & IO helpers (compatible with foundation's scripts/common.py)
# ---------------------------------------------------------------------------
def get_run_context() -> Dict[str, Any]:
    """Load run_dir, config, git_hash, definitions_hash, run_timestamp."""
    try:
        import common  # type: ignore

        if hasattr(common, "get_run_context"):
            return common.get_run_context()
    except Exception:
        pass

    latest_ptr = BASE_DIR / "results" / "LATEST_RUN.txt"
    if not latest_ptr.exists():
        raise FileNotFoundError(f"LATEST_RUN.txt not found at {latest_ptr}")
    run_dir_str = latest_ptr.read_text().strip()
    run_dir = Path(run_dir_str)
    if not run_dir.is_absolute():
        run_dir = BASE_DIR / run_dir_str
        if not run_dir.exists():
            run_dir = BASE_DIR / "results" / run_dir_str

    def_path = BASE_DIR / "config" / "definitions.yaml"
    cfg: Dict[str, Any] = {}
    definitions_hash = "unknown"
    if def_path.exists():
        raw = def_path.read_bytes()
        definitions_hash = hashlib.sha256(raw).hexdigest()
        if yaml is not None:
            cfg = yaml.safe_load(raw.decode("utf-8")) or {}

    try:
        git_hash = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(BASE_DIR),
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
    except Exception:
        git_hash = "nogit"

    run_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "run_dir": run_dir,
        "config": cfg,
        "git_hash": git_hash,
        "definitions_hash": definitions_hash,
        "run_timestamp": run_timestamp,
    }


def add_provenance(df: pd.DataFrame, ctx: Dict[str, Any]) -> pd.DataFrame:
    df = df.copy()
    df["git_hash"] = ctx["git_hash"]
    df["definitions_hash"] = ctx["definitions_hash"]
    df["run_timestamp"] = ctx["run_timestamp"]
    return df


# ---------------------------------------------------------------------------
# Pure Reusable Function: Spitaler Daily-Burden Episode Emulation
# ---------------------------------------------------------------------------
DEFAULT_SPITALER_RULES: Dict[str, Any] = {
    "min_daily_burden_min": 6.0,
    "full_day_min": 1400.0,
    "intra_day_tolerance_min": 40.0,
    "bridge_midnight_requires_full_day": True,
}


def emulate_spitaler_episodes(
    runs_df: pd.DataFrame,
    rules: Optional[Dict[str, Any]] = None,
    exclude_boundary: bool = True,
) -> pd.DataFrame:
    """
    Emulate Spitaler et al. daily-burden episode reconstruction from true AF runs.

    Algorithm:
    1. Aggregate AF seconds per calendar day (day d = [d*86400, (d+1)*86400)),
       splitting runs that cross midnight across the days they overlap.
    2. Filter out calendar days with daily AF burden < `min_daily_burden_min` (6 min).
    3. Mark each qualifying day as a "full day" if daily AF >= `full_day_min` (1400 min).
    4. Group consecutive qualifying calendar days into episodes:
       - Days d and d+1 bridge midnight ONLY IF `not bridge_midnight_requires_full_day`
         OR (day d is full OR day d+1 is full).
       - Back-to-back partial days (neither full) form SEPARATE episodes.
       - Any non-qualifying day (< 6 min) breaks an episode.
    5. Episode duration = summed daily AF burden (in seconds) across the episode's days.
    6. Exclude boundary episodes (episodes containing any run with
       `boundary_censored_start == True` or `boundary_censored_end == True`).

    For each reconstructed episode, also compute internal dynamics:
    - `n_true_runs`: number of true AF runs overlapping the episode's days
    - `n_internal_terminations`: max(0, n_true_runs - 1)
    - `n_internal_reinit_lt360`: count of internal inter-run gaps < 360 s
    - `af_sec_reinit_chain`: AF seconds from runs participating in a <360 s reinitiation chain
    - `af_sec_uninterrupted`: AF seconds from isolated runs (not linked by <360 s gaps)
    - `longest_true_run_sec`: duration of the longest single true run inside the episode

    Returns a DataFrame with one row per reconstructed episode.
    """
    r = dict(DEFAULT_SPITALER_RULES)
    if rules:
        r.update(rules)

    min_daily_sec = float(r.get("min_daily_burden_min", 6.0)) * 60.0
    full_day_sec = float(r.get("full_day_min", 1400.0)) * 60.0
    bridge_req_full = bool(r.get("bridge_midnight_requires_full_day", True))

    out_cols = [
        "record",
        "patient",
        "cohort",
        "ep_id",
        "start_day",
        "end_day",
        "n_days",
        "n_full_days",
        "dur_sec",
        "n_true_runs",
        "n_internal_terminations",
        "n_internal_reinit_lt360",
        "af_sec_reinit_chain",
        "af_sec_uninterrupted",
        "longest_true_run_sec",
        "boundary_censored",
    ]

    if runs_df is None or len(runs_df) == 0:
        return pd.DataFrame(columns=out_cols)

    df = runs_df.copy()
    if "patient" not in df.columns:
        df["patient"] = df["record"]
    if "cohort" not in df.columns:
        df["cohort"] = "UNKNOWN"
    if "boundary_censored_start" not in df.columns:
        df["boundary_censored_start"] = False
    if "boundary_censored_end" not in df.columns:
        df["boundary_censored_end"] = False

    episodes: List[Dict[str, Any]] = []

    for (cohort_id, record_id), grp in df.groupby(["cohort", "record"], sort=False):
        patient_id = grp["patient"].iloc[0]

        grp_sorted = grp.sort_values("onset_sec").reset_index(drop=True)

        # 1. Aggregate AF seconds per calendar day
        day_af_sec: Dict[int, float] = {}
        day_run_indices: Dict[int, List[int]] = {}
        day_boundary: Dict[int, bool] = {}

        onsets = grp_sorted["onset_sec"].to_numpy(dtype=float)
        offsets = grp_sorted["offset_sec"].to_numpy(dtype=float)
        b_start = grp_sorted["boundary_censored_start"].to_numpy(dtype=bool)
        b_end = grp_sorted["boundary_censored_end"].to_numpy(dtype=bool)

        for idx in range(len(grp_sorted)):
            s = onsets[idx]
            e = offsets[idx]
            if e <= s:
                continue
            d_start = int(np.floor(s / 86400.0))
            d_end = int(np.floor(max(s, e - 1e-9) / 86400.0))
            is_bc = bool(b_start[idx] or b_end[idx])
            for d in range(d_start, d_end + 1):
                seg_s = max(s, d * 86400.0)
                seg_e = min(e, (d + 1) * 86400.0)
                overlap = max(0.0, seg_e - seg_s)
                if overlap > 0.0:
                    day_af_sec[d] = day_af_sec.get(d, 0.0) + overlap
                    day_run_indices.setdefault(d, []).append(idx)
                    if is_bc:
                        day_boundary[d] = True

        # 2. Filter days >= min_daily_sec (6 min)
        qual_days = sorted(d for d, sec in day_af_sec.items() if sec >= min_daily_sec)
        if not qual_days:
            continue

        # 3 & 4. Group consecutive qualifying days into episodes
        ep_groups: List[List[int]] = []
        current_group: List[int] = [qual_days[0]]

        for d in qual_days[1:]:
            prev_d = current_group[-1]
            is_consecutive = d == prev_d + 1
            prev_full = day_af_sec[prev_d] >= full_day_sec
            curr_full = day_af_sec[d] >= full_day_sec
            can_bridge = is_consecutive and (
                (not bridge_req_full) or prev_full or curr_full
            )
            if can_bridge:
                current_group.append(d)
            else:
                ep_groups.append(current_group)
                current_group = [d]
        ep_groups.append(current_group)

        # 5 & 6. Summarize each episode
        ep_counter = 0
        for ep_days in ep_groups:
            is_bc_ep = any(day_boundary.get(d, False) for d in ep_days)
            if exclude_boundary and is_bc_ep:
                continue

            ep_counter += 1
            dur_sec = float(sum(day_af_sec[d] for d in ep_days))
            n_full = int(sum(1 for d in ep_days if day_af_sec[d] >= full_day_sec))

            # Collect unique true runs contributing to this episode's days
            run_idx_set = sorted(
                set(idx for d in ep_days for idx in day_run_indices.get(d, []))
            )
            n_true = len(run_idx_set)
            n_int_term = max(0, n_true - 1)

            if n_true == 0:
                n_int_reinit_360 = 0
                af_chain = 0.0
                af_unint = dur_sec
                longest_run = 0.0
            elif n_true == 1:
                n_int_reinit_360 = 0
                af_chain = 0.0
                af_unint = dur_sec
                i0 = run_idx_set[0]
                longest_run = float(offsets[i0] - onsets[i0])
            else:
                ep_on = onsets[run_idx_set]
                ep_off = offsets[run_idx_set]
                ep_run_durs = ep_off - ep_on
                # Clip each run to episode day boundaries for exact accounting
                ep_win_s = ep_days[0] * 86400.0
                ep_win_e = (ep_days[-1] + 1) * 86400.0
                clipped_durs = np.maximum(
                    0.0, np.minimum(ep_off, ep_win_e) - np.maximum(ep_on, ep_win_s)
                )
                gaps = ep_on[1:] - ep_off[:-1]
                lt360 = (gaps >= 0.0) & (gaps < 360.0)
                n_int_reinit_360 = int(np.sum(lt360))

                in_chain = np.zeros(n_true, dtype=bool)
                in_chain[:-1] |= lt360
                in_chain[1:] |= lt360

                af_chain = float(np.sum(clipped_durs[in_chain]))
                af_unint = float(np.sum(clipped_durs[~in_chain]))
                longest_run = float(np.max(ep_run_durs))

            episodes.append(
                {
                    "record": record_id,
                    "patient": patient_id,
                    "cohort": cohort_id,
                    "ep_id": ep_counter,
                    "start_day": ep_days[0],
                    "end_day": ep_days[-1],
                    "n_days": len(ep_days),
                    "n_full_days": n_full,
                    "dur_sec": dur_sec,
                    "n_true_runs": n_true,
                    "n_internal_terminations": n_int_term,
                    "n_internal_reinit_lt360": n_int_reinit_360,
                    "af_sec_reinit_chain": af_chain,
                    "af_sec_uninterrupted": af_unint,
                    "longest_true_run_sec": longest_run,
                    "boundary_censored": is_bc_ep,
                }
            )

    if not episodes:
        return pd.DataFrame(columns=out_cols)
    return pd.DataFrame(episodes, columns=out_cols)


# ---------------------------------------------------------------------------
# Helper stats for Aim 2 comparison
# ---------------------------------------------------------------------------
def fit_weibull_shape(durations: np.ndarray, events: np.ndarray, t_min: float = 0.0) -> Tuple[float, float]:
    """Fit simple Weibull MLE (shape k, scale lambda) using scipy, with optional left truncation t_min."""
    t = np.asarray(durations, dtype=float)
    e = np.asarray(events, dtype=float)
    tmin_val = max(0.0, float(t_min))
    mask = t > tmin_val
    t, e = t[mask], e[mask]
    if len(t) < 5 or np.sum(e) < 3:
        return float("nan"), float("nan")

    t_scale = float(np.median(t))
    log_t = np.log(t / t_scale)
    use_trunc = tmin_val > 0.0
    log_tmin = float(np.log(tmin_val / t_scale)) if use_trunc else 0.0
    sum_e = float(np.sum(e))
    sum_e_logt = float(np.sum(e * log_t))

    def nll(params: np.ndarray) -> float:
        log_k, log_lam = params
        if log_k < -5.0 or log_k > 3.0 or log_lam < -15.0 or log_lam > 15.0:
            return 1e15
        k = np.exp(log_k)
        z = np.clip(k * (log_t - log_lam), -40.0, 40.0)
        if use_trunc:
            zm = np.clip(k * (log_tmin - log_lam), -40.0, 40.0)
            hij = np.maximum(0.0, np.exp(z) - np.exp(zm))
        else:
            hij = np.exp(z)
        ll = sum_e * (log_k - k * log_lam) + (k - 1.0) * sum_e_logt - float(np.sum(hij))
        return -ll if np.isfinite(ll) else 1e15

    init = np.array([np.log(0.7), 0.0])
    res = optimize.minimize(nll, init, method="L-BFGS-B", bounds=[(-5.0, 3.0), (-15.0, 15.0)])
    if not res.success or not np.isfinite(res.fun):
        res = optimize.minimize(nll, init, method="Nelder-Mead")
    k_hat = float(np.exp(res.x[0]))
    lam_hat = float(np.exp(res.x[1]) * t_scale)
    return k_hat, lam_hat


def conditional_term_prob(durations: np.ndarray, t: float, w: float, events: Optional[np.ndarray] = None) -> float:
    """Right-censoring-aware Kaplan-Meier conditional termination probability P(T < t+w | T >= t)."""
    d = np.asarray(durations, dtype=float)
    if events is None:
        ev = np.ones(len(d), dtype=int)
    else:
        ev = np.asarray(events, dtype=int)
    at_risk_n = int(np.sum(d >= t))
    if at_risk_n == 0:
        return float("nan")
    if KaplanMeierFitter is not None:
        kmf = KaplanMeierFitter()
        kmf.fit(d, event_observed=ev)
        s_t = float(kmf.survival_function_at_times([t]).iloc[0])
        if s_t <= 1e-12:
            return float("nan")
        s_tw = float(kmf.survival_function_at_times([t + w]).iloc[0])
        return float(np.clip(1.0 - s_tw / s_t, 0.0, 1.0))
    uncens = d[ev == 1]
    ar_u = uncens >= t
    n_u = int(np.sum(ar_u))
    if n_u == 0:
        return float("nan")
    return float(np.sum(ar_u & (uncens < t + w)) / n_u)


def _load_record_metadata_and_clocks(run_dir: Path) -> pd.DataFrame:
    """Load exact record duration, recording start clock hour, and SHDB AF_Type per (cohort, record)."""
    bs = pd.read_parquet(run_dir / "stage1" / "beat_summary.parquet")[["cohort", "record", "patient", "record_dur_sec"]].copy()
    bs["record"] = bs["record"].astype(str)
    bs["start_hour"] = np.nan
    bs["af_subtype"] = np.where(bs["cohort"] == "IRIDIA", "PAF", "UNSPECIFIED")

    cohorts_cfg = yaml.safe_load((BASE_DIR / "config" / "cohorts.yaml").read_text()) if yaml else {}
    cdict = cohorts_cfg.get("cohorts", {})

    # IRIDIA clocks
    ir_csv = Path(cdict.get("IRIDIA", {}).get("metadata_csv", "/data/mithunmanivannan/datasets/iridia_af_8405941_v1.0.1/iridia-af-metadata-v1.0.1.csv"))
    if ir_csv.exists():
        ir_m = pd.read_csv(ir_csv)
        ir_hrs = {}
        for r in ir_m.itertuples(index=False):
            ts = getattr(r, "record_start_time", None)
            if pd.notna(ts):
                try:
                    dt = datetime.fromisoformat(str(ts))
                    ir_hrs[str(r.record_id)] = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
                except Exception:
                    pass
        m_ir = bs["cohort"] == "IRIDIA"
        bs.loc[m_ir, "start_hour"] = bs.loc[m_ir, "record"].map(ir_hrs)

    # SHDB clocks & clinical AF_Type
    sh_csv = Path(cdict.get("SHDB", {}).get("metadata_csv", "/data/mithunmanivannan/datasets/shdb_af/AdditionalData.csv"))
    if sh_csv.exists():
        sh_m = pd.read_csv(sh_csv, dtype={"Data_ID": str})
        sh_m["rec_key"] = sh_m["Data_ID"].str.zfill(3)
        sh_hrs, sh_types = {}, {}
        for r in sh_m.itertuples(index=False):
            sh_types[str(r.rec_key)] = str(getattr(r, "AF_Type", "UNSPECIFIED"))
            ts = getattr(r, "Holter_start_time", None)
            if pd.notna(ts):
                s = str(ts).strip()
                for fmt in ("%I:%M %p", "%H:%M:%S", "%H:%M"):
                    try:
                        dt = datetime.strptime(s, fmt)
                        sh_hrs[str(r.rec_key)] = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
                        break
                    except Exception:
                        continue
        m_sh = bs["cohort"] == "SHDB"
        bs.loc[m_sh, "start_hour"] = bs.loc[m_sh, "record"].map(sh_hrs)
        bs.loc[m_sh, "af_subtype"] = bs.loc[m_sh, "record"].map(sh_types).fillna("UNSPECIFIED")

    # LTAFDB clocks from WFDB headers
    ltaf_dir = Path(cdict.get("LTAFDB", {}).get("base_dir", "/data/mithunmanivannan/datasets/ltafdb_1.0.0_reference"))
    if wfdb is not None and ltaf_dir.exists():
        m_lt = bs["cohort"] == "LTAFDB"
        lt_hrs = {}
        for rec in bs.loc[m_lt, "record"].unique():
            try:
                hdr = wfdb.rdheader(str(ltaf_dir / rec))
                if hdr.base_time is not None:
                    bt = hdr.base_time
                    lt_hrs[rec] = bt.hour + bt.minute / 60.0 + bt.second / 3600.0
            except Exception:
                pass
        bs.loc[m_lt, "start_hour"] = bs.loc[m_lt, "record"].map(lt_hrs)

    return bs


def run_reconciliation_and_boundary_audit(
    runs_g0: pd.DataFrame,
    rec_meta: pd.DataFrame,
    rules: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reconcile 100% of runs_g0 into mutually exclusive outcomes under:
      1. `recording_24h_windows`: elapsed [k*86400, (k+1)*86400) blocks.
      2. `wall_clock_spitaler`: calendar midnight-to-midnight days using actual Holter
         start clock hour, excluding partial first/last calendar days per Spitaler 2026.
    """
    min_daily_sec = float(rules.get("min_daily_burden_min", 6.0)) * 60.0
    full_day_sec = float(rules.get("full_day_min", 1400.0)) * 60.0
    bridge_req_full = bool(rules.get("bridge_midnight_requires_full_day", True))

    rm_map = {
        (r.cohort, str(r.record)): (float(r.record_dur_sec), float(r.start_hour) if pd.notna(r.start_hour) else np.nan, str(r.af_subtype))
        for r in rec_meta.itertuples(index=False)
    }

    ledger_rows: List[Dict[str, Any]] = []
    wall_episodes: List[Dict[str, Any]] = []

    for mode in ("recording_24h_windows", "wall_clock_spitaler"):
        for (cohort, rec_id), grp in runs_g0.groupby(["cohort", "record"], sort=True):
            rec_dur, start_hr, subtype = rm_map[(cohort, str(rec_id))]
            grp = grp.sort_values("onset_sec").reset_index(drop=True)
            offset_shift = (start_hr * 3600.0) if (mode == "wall_clock_spitaler" and np.isfinite(start_hr)) else 0.0

            if mode == "wall_clock_spitaler" and not np.isfinite(start_hr):
                for r in grp.itertuples(index=False):
                    ledger_rows.append({
                        "mode": mode, "cohort": cohort, "af_subtype": subtype, "record": rec_id,
                        "patient": r.patient, "run_id": r.run_id, "outcome": "excluded_missing_clock",
                        "record_dur_sec": rec_dur, "is_multiday_ge36h": bool(rec_dur >= 36 * 3600),
                    })
                continue

            # Determine full calendar days vs partial first/last boundary days
            rec_end_shifted = offset_shift + rec_dur
            first_day = int(np.floor(offset_shift / 86400.0))
            last_day = int(np.floor(max(offset_shift, rec_end_shifted - 1e-9) / 86400.0))

            day_af_sec: Dict[int, float] = {}
            day_runs: Dict[int, List[int]] = {}
            day_bc: Dict[int, bool] = {}

            onsets = grp["onset_sec"].to_numpy(float) + offset_shift
            offsets = grp["offset_sec"].to_numpy(float) + offset_shift
            bc_start = grp["boundary_censored_start"].to_numpy(bool)
            bc_end = grp["boundary_censored_end"].to_numpy(bool)

            for idx in range(len(grp)):
                s, e = onsets[idx], offsets[idx]
                if e <= s:
                    continue
                ds = int(np.floor(s / 86400.0))
                de = int(np.floor(max(s, e - 1e-9) / 86400.0))
                for d in range(ds, de + 1):
                    ov = max(0.0, min(e, (d + 1) * 86400.0) - max(s, d * 86400.0))
                    if ov > 0:
                        day_af_sec[d] = day_af_sec.get(d, 0.0) + ov
                        day_runs.setdefault(d, []).append(idx)
                        if bc_start[idx] or bc_end[idx]:
                            day_bc[d] = True

            # Partial boundary days in wall_clock_spitaler: first_day if start_hr > 0.01, last_day if not ending near midnight
            partial_days = set()
            if mode == "wall_clock_spitaler":
                if (offset_shift - first_day * 86400.0) > 60.0:
                    partial_days.add(first_day)
                if ((last_day + 1) * 86400.0 - rec_end_shifted) > 60.0:
                    partial_days.add(last_day)

            qual_days = sorted(d for d, sec in day_af_sec.items() if sec >= min_daily_sec and d not in partial_days)

            # Group qualifying days per Spitaler rule
            ep_groups: List[List[int]] = []
            for d in qual_days:
                if not ep_groups:
                    ep_groups.append([d])
                    continue
                pd_prev = ep_groups[-1][-1]
                consec = d == pd_prev + 1
                can_bridge = consec and ((not bridge_req_full) or day_af_sec[pd_prev] >= full_day_sec or day_af_sec[d] >= full_day_sec)
                if can_bridge:
                    ep_groups[-1].append(d)
                else:
                    ep_groups.append([d])

            run_outcome: Dict[int, str] = {}
            for ep_days in ep_groups:
                is_bc_ep = any(day_bc.get(d, False) for d in ep_days)
                ridxs = sorted(set(i for d in ep_days for i in day_runs.get(d, [])))
                if is_bc_ep:
                    for i in ridxs:
                        run_outcome.setdefault(i, "excluded_boundary_censored")
                else:
                    lbl = "retained_single_run" if len(ridxs) == 1 else "merged_multi_run"
                    for i in ridxs:
                        run_outcome[i] = lbl
                    if mode == "wall_clock_spitaler":
                        wall_episodes.append({
                            "cohort": cohort, "af_subtype": subtype, "record": rec_id,
                            "patient": grp["patient"].iloc[0],
                            "is_multiday_ge36h": bool(rec_dur >= 36 * 3600),
                            "start_day": ep_days[0], "end_day": ep_days[-1],
                            "n_days": len(ep_days),
                            "n_full_days": int(sum(1 for d in ep_days if day_af_sec[d] >= full_day_sec)),
                            "dur_sec": float(sum(day_af_sec[d] for d in ep_days)),
                            "n_true_runs": len(ridxs),
                        })

            # Classify every remaining run mutually exclusively (priority: partial_day > boundary_censored > day_lt6min)
            for idx in range(len(grp)):
                if idx not in run_outcome:
                    ds = int(np.floor(onsets[idx] / 86400.0))
                    de = int(np.floor(max(onsets[idx], offsets[idx] - 1e-9) / 86400.0))
                    r_days = range(ds, de + 1)
                    if bc_start[idx] or bc_end[idx]:
                        run_outcome[idx] = "excluded_boundary_censored"
                    elif mode == "wall_clock_spitaler" and all(d in partial_days for d in r_days):
                        run_outcome[idx] = "excluded_partial_calendar_day"
                    else:
                        run_outcome[idx] = "excluded_day_lt6min"

                ledger_rows.append({
                    "mode": mode, "cohort": cohort, "af_subtype": subtype, "record": rec_id,
                    "patient": grp["patient"].iloc[idx], "run_id": grp["run_id"].iloc[idx],
                    "outcome": run_outcome[idx],
                    "record_dur_sec": rec_dur, "is_multiday_ge36h": bool(rec_dur >= 36 * 3600),
                })

    ldf = pd.DataFrame(ledger_rows)
    wdf = pd.DataFrame(wall_episodes)
    # Aggregate counts per mode x cohort (and pooled / multiday IRIDIA)
    out_rows: List[Dict[str, Any]] = []
    outcomes_order = [
        "retained_single_run",
        "merged_multi_run",
        "excluded_day_lt6min",
        "excluded_boundary_censored",
        "excluded_partial_calendar_day",
        "excluded_missing_clock",
    ]
    for mode, m_df in ldf.groupby("mode", sort=True):
        groups = [
            ("POOLED", m_df),
            ("IRIDIA", m_df[m_df["cohort"] == "IRIDIA"]),
            ("IRIDIA_1day_lt36h", m_df[(m_df["cohort"] == "IRIDIA") & (~m_df["is_multiday_ge36h"])]),
            ("IRIDIA_multiday_ge36h", m_df[(m_df["cohort"] == "IRIDIA") & (m_df["is_multiday_ge36h"])]),
            ("SHDB", m_df[m_df["cohort"] == "SHDB"]),
            ("LTAFDB", m_df[m_df["cohort"] == "LTAFDB"]),
        ]
        for g_name, g_sub in groups:
            if len(g_sub) == 0:
                continue
            vc = g_sub["outcome"].value_counts().to_dict()
            row: Dict[str, Any] = {
                "mode": mode,
                "stratum": g_name,
                "n_runs_total": int(len(g_sub)),
                "n_patients": int(g_sub["patient"].nunique()),
            }
            tot = 0
            for o in outcomes_order:
                c = int(vc.get(o, 0))
                row[o] = c
                tot += c
            row["balance_diff"] = int(row["n_runs_total"] - tot)
            assert row["balance_diff"] == 0, f"Run ledger imbalance in {mode} {g_name}"
            if mode == "wall_clock_spitaler" and not wdf.empty:
                if g_name == "POOLED":
                    ws = wdf
                elif g_name == "IRIDIA_1day_lt36h":
                    ws = wdf[(wdf["cohort"] == "IRIDIA") & (~wdf["is_multiday_ge36h"])]
                elif g_name == "IRIDIA_multiday_ge36h":
                    ws = wdf[(wdf["cohort"] == "IRIDIA") & (wdf["is_multiday_ge36h"])]
                else:
                    ws = wdf[wdf["cohort"] == g_name]
                row["n_wall_clock_episodes"] = int(len(ws))
                row["n_wall_clock_multiday_bridged"] = int((ws["n_days"] > 1).sum()) if len(ws) else 0
            else:
                row["n_wall_clock_episodes"] = np.nan
                row["n_wall_clock_multiday_bridged"] = np.nan
            out_rows.append(row)

    return pd.DataFrame(out_rows), wdf


def compute_matched_support_hazard_comparison(
    runs_g0: pd.DataFrame,
    recon_df: pd.DataFrame,
    rules: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compare unfrailty AND shared-gamma-frailty Weibull shapes on MATCHED eligible
    observation windows, AND emit exact run-level and cohort-level reconciliation
    verifying 100% agreement between retained/merged audit runs and matched survival runs.
    """
    aim1_mod = importlib.import_module("07_aim1_termination_hazard")
    fit_frailty = aim1_mod.fit_shared_gamma_frailty_weibull

    if recon_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    eligible_rec_days: set[Tuple[str, str, int]] = set()
    for r in recon_df.itertuples(index=False):
        for d in range(int(r.start_day), int(r.end_day) + 1):
            eligible_rec_days.add((str(r.cohort), str(r.record), d))

    in_elig_mask = []
    ledger_rows = []
    for _, r in runs_g0.iterrows():
        ds = int(np.floor(float(r["onset_sec"]) / 86400.0))
        de = int(np.floor(max(float(r["onset_sec"]), float(r["offset_sec"]) - 1e-9) / 86400.0))
        in_elig = any((str(r["cohort"]), str(r["record"]), d) in eligible_rec_days for d in range(ds, de + 1))
        in_elig_mask.append(in_elig)
        if in_elig:
            bcs = bool(r["boundary_censored_start"])
            bce = bool(r["boundary_censored_end"])
            if bcs:
                excl_reason = "excluded_unknown_onset"
            elif bce:
                excl_reason = "included_right_censored"
            else:
                excl_reason = "included_complete_termination"
            ledger_rows.append({
                "cohort": str(r["cohort"]),
                "record": str(r["record"]),
                "patient": str(r["patient"]),
                "run_id": str(r["run_id"]),
                "onset_sec": float(r["onset_sec"]),
                "offset_sec": float(r["offset_sec"]),
                "dur_sec": float(r["dur_sec"]),
                "boundary_censored_start": bcs,
                "boundary_censored_end": bce,
                "included_in_matched_survival": bool(not bcs),
                "reconciliation_status": excl_reason,
            })
    in_elig_series = pd.Series(in_elig_mask, index=runs_g0.index)
    run_ledger_df = pd.DataFrame(ledger_rows)

    in_elig_df = runs_g0[in_elig_series].copy()
    is_bc_start = in_elig_df["boundary_censored_start"].astype(bool)

    rec_rows = []
    for coh in ["IRIDIA", "SHDB", "LTAFDB", "POOLED"]:
        sub_e = in_elig_df if coh == "POOLED" else in_elig_df[in_elig_df["cohort"] == coh]
        s_bc_s = sub_e["boundary_censored_start"].astype(bool)
        s_bc_e = sub_e["boundary_censored_end"].astype(bool)
        known_e = sub_e[~s_bc_s]
        rec_rows.append({
            "cohort": coh,
            "n_runs_in_eligible_windows": int(len(sub_e)),
            "n_excluded_unknown_onset": int(s_bc_s.sum()),
            "n_matched_known_onset_runs": int(len(known_e)),
            "n_matched_complete_terminations": int((~s_bc_e[~s_bc_s]).sum()),
            "n_matched_right_censored": int(s_bc_e[~s_bc_s].sum()),
            "balance_diff": int(len(sub_e) - int(s_bc_s.sum()) - len(known_e)),
        })
    elig_recon_df = pd.DataFrame(rec_rows)

    matched_native = in_elig_df[~is_bc_start].copy()

    rows = []
    for coh in ["IRIDIA", "SHDB", "LTAFDB", "POOLED"]:
        rn = matched_native if coh == "POOLED" else matched_native[matched_native["cohort"] == coh]
        re = recon_df if coh == "POOLED" else recon_df[recon_df["cohort"] == coh]
        if len(rn) < 5 or len(re) < 5:
            continue

        rn_ev = (~rn["boundary_censored_end"].astype(bool)).astype(int).to_numpy()
        rn_dur = rn["dur_sec"].to_numpy(float)
        re_dur = re["dur_sec"].to_numpy(float)
        re_ev = np.ones(len(re_dur), dtype=int)

        unfr_native_k, _ = fit_weibull_shape(rn_dur, rn_ev)
        unfr_recon_k, _ = fit_weibull_shape(re_dur, re_ev)

        fr_native = fit_frailty(rn_dur, rn_ev.astype(float), rn["patient"].to_numpy())
        fr_recon = fit_frailty(re_dur, re_ev.astype(float), re["patient"].to_numpy())

        rows.append({
            "cohort": coh,
            "n_matched_native_runs": int(len(rn)),
            "n_matched_native_patients": int(rn["patient"].nunique()),
            "n_matched_recon_episodes": int(len(re)),
            "n_matched_recon_patients": int(re["patient"].nunique()),
            "native_median_dur_sec": float(np.median(rn_dur[rn_ev == 1])) if np.any(rn_ev == 1) else np.nan,
            "recon_median_dur_sec": float(np.median(re_dur)),
            "native_unfrailty_weibull_k": float(unfr_native_k),
            "recon_unfrailty_weibull_k": float(unfr_recon_k),
            "native_frailty_weibull_k": float(fr_native.get("shape_k", np.nan)),
            "native_frailty_theta": float(fr_native.get("frailty_theta", np.nan)),
            "recon_frailty_weibull_k": float(fr_recon.get("shape_k", np.nan)),
            "recon_frailty_theta": float(fr_recon.get("frailty_theta", np.nan)),
        })
    return pd.DataFrame(rows), elig_recon_df, run_ledger_df


def _fit_frailty_fast_point(
    durations: np.ndarray,
    events: np.ndarray,
    patient_ids: np.ndarray,
    t_min: float = 0.0,
    init_k: float = 0.7,
    init_theta: float = 0.5,
) -> Tuple[bool, float, float, bool, bool]:
    """Fast box-bounded L-BFGS-B (with Nelder-Mead fallback) for paired bootstrap & boundary audit."""
    t = np.asarray(durations, dtype=float)
    d = np.asarray(events, dtype=float)
    mask = t > max(0.0, float(t_min))
    t, d, pids = t[mask], d[mask], np.asarray(patient_ids)[mask]
    if len(t) < 10 or np.sum(d) < 5:
        return False, np.nan, np.nan, False, False

    t_scale = float(np.median(t))
    t_norm = t / t_scale
    log_t = np.log(t_norm)
    use_trunc = float(t_min) > 0.0
    log_tmin = float(np.log(float(t_min) / t_scale)) if use_trunc else 0.0

    _, pat_inv = np.unique(pids, return_inverse=True)
    n_pats = int(pat_inv.max() + 1)
    D_i = np.bincount(pat_inv, weights=d, minlength=n_pats)
    sum_d = float(np.sum(d))
    sum_d_logt = float(np.sum(d * log_t))

    def nll(params: np.ndarray) -> float:
        lk, llam, lth = params
        if lk < -5.0 or lk > 3.0 or llam < -15.0 or llam > 15.0 or lth < -12.0 or lth > 6.0:
            return 1e15
        k = np.exp(lk)
        th = np.exp(lth)
        inv_th = 1.0 / th
        z = np.clip(k * (log_t - llam), -40.0, 40.0)
        if use_trunc:
            zm = np.clip(k * (log_tmin - llam), -40.0, 40.0)
            hij = np.maximum(0.0, np.exp(z) - np.exp(zm))
        else:
            hij = np.exp(z)
        Hi = np.bincount(pat_inv, weights=hij, minlength=n_pats)
        val = -(
            sum_d * (lk - k * llam) + (k - 1.0) * sum_d_logt
            + float(np.sum(special.gammaln(inv_th + D_i) - special.gammaln(inv_th) + D_i * lth - (inv_th + D_i) * np.log1p(th * Hi)))
        )
        return val if np.isfinite(val) else 1e15

    x0 = np.array([np.log(max(0.1, min(3.0, init_k))), 0.0, np.log(max(1e-3, min(10.0, init_theta)))])
    bounds = [(-5.0, 3.0), (-15.0, 15.0), (-12.0, 6.0)]
    r = optimize.minimize(nll, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 250})
    if not r.success or not np.isfinite(r.fun) or r.fun >= 1e14:
        r = optimize.minimize(nll, x0, method="Nelder-Mead", options={"maxiter": 600})
    ok = bool(np.isfinite(r.fun) and r.fun < 1e14)
    if not ok:
        return False, np.nan, np.nan, False, False
    lk_h, _, lth_h = r.x
    hit_k = bool(lk_h <= -4.95 or lk_h >= 2.95)
    hit_th = bool(lth_h <= -11.95 or lth_h >= 5.95)
    return True, float(np.exp(lk_h)), float(np.exp(lth_h)), hit_k, hit_th


def compute_paired_aggregation_bootstrap(
    tiers_runs: Dict[str, pd.DataFrame],
    rec_meta: pd.DataFrame,
    rules: Dict[str, Any],
    n_boot: int = 300,
    rhythm_intervals_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Paired patient-clustered bootstrap CIs for delta Weibull shape, median episode
    duration ratio, and conditional 1-h termination shift (using eligible >=6-min
    landmark when restricted to >=6 min) across rhythm tiers, coverage filters
    (>=95%, >=99% known rhythm), and wall-clock multi-day IRIDIA.
    Also exports bootstrap replicates and identifiability / boundary-hit diagnostics.
    """
    aim1_mod = importlib.import_module("07_aim1_termination_hazard")
    fit_frailty_full = aim1_mod.fit_shared_gamma_frailty_weibull

    rm_hours = {
        (r.cohort, str(r.record)): (float(r.record_dur_sec), float(r.start_hour) if pd.notna(r.start_hour) else np.nan)
        for r in rec_meta.itertuples(index=False)
    }

    # Exact per-recording-window known-rhythm coverage fraction
    win_cov_map: Dict[Tuple[str, str, int], float] = {}
    itv_by_rec = (
        {k: g.sort_values("start_sec") for k, g in rhythm_intervals_df.groupby(["cohort", "record"], sort=True)}
        if (rhythm_intervals_df is not None and not rhythm_intervals_df.empty)
        else {}
    )
    for (coh, rec), (rec_dur, _) in rm_hours.items():
        n_win = max(1, int(np.floor((rec_dur + 400.0) / 86400.0))) if rec_dur >= 86000.0 else 1
        itv_rec = itv_by_rec.get((coh, str(rec)), pd.DataFrame())
        for d in range(n_win):
            ws, we = d * 86400.0, min(rec_dur, (d + 1) * 86400.0)
            wdur = max(1.0, we - ws)
            if not itv_rec.empty:
                iov = itv_rec[(itv_rec["end_sec"] > ws) & (itv_rec["start_sec"] < we)]
                ic_durs = np.maximum(0.0, np.minimum(iov["end_sec"], we) - np.maximum(iov["start_sec"], ws))
                unk_obs = float(np.sum(ic_durs[iov["rhythm"].to_numpy() == "unknown"]))
                win_cov_map[(coh, str(rec), d)] = float(max(0.0, wdur - unk_obs) / wdur)
            else:
                win_cov_map[(coh, str(rec), d)] = 1.0

    def _matched_pair(
        runs_df: pd.DataFrame,
        wall_clock_only: bool = False,
        min_dur: float = 0.0,
        min_cov: float = 0.0,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
        if not wall_clock_only:
            recon = emulate_spitaler_episodes(runs_df, rules, exclude_boundary=True)
            if recon.empty:
                return pd.DataFrame(), pd.DataFrame(), 0
            if min_cov > 0.0:
                keep_ep = [
                    all(win_cov_map.get((str(r.cohort), str(r.record), d), 1.0) >= min_cov for d in range(int(r.start_day), int(r.end_day) + 1))
                    for r in recon.itertuples(index=False)
                ]
                recon = recon.loc[keep_ep].copy()
                if recon.empty:
                    return pd.DataFrame(), pd.DataFrame(), 0
            elig_days = {(str(r.cohort), str(r.record), d) for r in recon.itertuples(index=False) for d in range(int(r.start_day), int(r.end_day) + 1)}
            n_elig_windows = len(elig_days)
            known = runs_df[~runs_df["boundary_censored_start"].astype(bool)].copy()
            keep = [
                any((str(r.cohort), str(r.record), d) in elig_days for d in range(int(np.floor(r.onset_sec / 86400.0)), int(np.floor(max(r.onset_sec, r.offset_sec - 1e-9) / 86400.0)) + 1))
                for r in known.itertuples(index=False)
            ]
            nat = known.loc[keep].copy()
        else:
            # Strict interior full calendar days across multi-day records
            min_daily_sec = float(rules.get("min_daily_burden_min", 6.0)) * 60.0
            full_day_sec = float(rules.get("full_day_min", 1400.0)) * 60.0
            ep_rows, nat_rows = [], []
            elig_cal_days = set()
            for (coh, rec_id), grp in runs_df.groupby(["cohort", "record"], sort=True):
                rec_dur, start_hr = rm_hours[(coh, str(rec_id))]
                if rec_dur < 36 * 3600 or not np.isfinite(start_hr):
                    continue
                shift = start_hr * 3600.0
                fd = int(np.floor(shift / 86400.0))
                ld = int(np.floor((shift + rec_dur - 1e-9) / 86400.0))
                partial = set()
                if (shift - fd * 86400.0) > 60.0:
                    partial.add(fd)
                if ((ld + 1) * 86400.0 - (shift + rec_dur)) > 60.0:
                    partial.add(ld)
                day_sec, day_bc = {}, {}
                for r in grp.itertuples(index=False):
                    s, e = r.onset_sec + shift, r.offset_sec + shift
                    for d in range(int(np.floor(s / 86400.0)), int(np.floor(max(s, e - 1e-9) / 86400.0)) + 1):
                        ov = max(0.0, min(e, (d + 1) * 86400.0) - max(s, d * 86400.0))
                        if ov > 0:
                            day_sec[d] = day_sec.get(d, 0.0) + ov
                            if r.boundary_censored_start or r.boundary_censored_end:
                                day_bc[d] = True
                qdays = sorted(d for d, sc in day_sec.items() if sc >= min_daily_sec and d not in partial)
                groups: List[List[int]] = []
                for d in qdays:
                    if not groups:
                        groups.append([d])
                    elif d == groups[-1][-1] + 1 and (day_sec[groups[-1][-1]] >= full_day_sec or day_sec[d] >= full_day_sec):
                        groups[-1].append(d)
                    else:
                        groups.append([d])
                elig_d = set()
                for g in groups:
                    if any(day_bc.get(d, False) for d in g):
                        continue
                    elig_d.update(g)
                    for gd in g:
                        elig_cal_days.add((coh, str(rec_id), gd))
                    ep_rows.append({"cohort": coh, "record": rec_id, "patient": grp["patient"].iloc[0], "dur_sec": float(sum(day_sec[d] for d in g))})
                for r in grp.itertuples(index=False):
                    if r.boundary_censored_start:
                        continue
                    s, e = r.onset_sec + shift, r.offset_sec + shift
                    if any(d in elig_d for d in range(int(np.floor(s / 86400.0)), int(np.floor(max(s, e - 1e-9) / 86400.0)) + 1)):
                        nat_rows.append({"cohort": coh, "record": rec_id, "patient": r.patient, "dur_sec": r.dur_sec, "boundary_censored_end": r.boundary_censored_end})
            nat = pd.DataFrame(nat_rows)
            recon = pd.DataFrame(ep_rows)
            n_elig_windows = len(elig_cal_days)

        if min_dur > 0.0 and not nat.empty and not recon.empty:
            nat = nat[nat["dur_sec"] > min_dur].copy()
            recon = recon[recon["dur_sec"] > min_dur].copy()
        return nat, recon, n_elig_windows

    rng = np.random.default_rng(SEED + 99)
    # Spec tuple: (tier_lbl, frame_lbl, cov_lbl, base_df, wc_flag, min_dur, min_cov, coh_list)
    specs: List[Tuple[str, str, str, pd.DataFrame, bool, float, float, List[str]]] = [
        ("harmonized_annotation", "recording_24h_windows", "all", tiers_runs["harmonized_annotation"], False, 0.0, 0.0, ["IRIDIA", "SHDB", "LTAFDB", "POOLED"]),
        ("harmonized_annotation_cov95", "recording_24h_windows", "cov_ge_95pct", tiers_runs["harmonized_annotation"], False, 0.0, 0.95, ["IRIDIA", "SHDB", "LTAFDB", "POOLED"]),
        ("harmonized_annotation_cov99", "recording_24h_windows", "cov_ge_99pct", tiers_runs["harmonized_annotation"], False, 0.0, 0.99, ["IRIDIA", "SHDB", "LTAFDB", "POOLED"]),
        ("harmonized_annotation_min360s_lt", "recording_24h_windows_lt360", "all", tiers_runs["harmonized_annotation"], False, 360.0, 0.0, ["IRIDIA", "POOLED"]),
        ("wall_clock_spitaler", "wall_clock_multiday_ge36h", "all", tiers_runs["harmonized_annotation"], True, 0.0, 0.0, ["IRIDIA"]),
    ]
    if "continuity_nonsinus_gaps" in tiers_runs:
        specs.append(("continuity_nonsinus_gaps", "recording_24h_windows", "all", tiers_runs["continuity_nonsinus_gaps"], False, 0.0, 0.0, ["SHDB", "LTAFDB", "POOLED"]))
    if "continuity_except_normal" in tiers_runs:
        specs.append(("continuity_except_normal", "recording_24h_windows", "all", tiers_runs["continuity_except_normal"], False, 0.0, 0.0, ["SHDB", "LTAFDB", "POOLED"]))

    out_rows = []
    cov_rows = []
    rep_rows = []
    ident_rows = []

    for tier_lbl, frame_lbl, cov_lbl, base_df, wc_flag, min_d, min_c, coh_list in specs:
        nat_all, rec_all, n_elig_w_all = _matched_pair(base_df, wall_clock_only=wc_flag, min_dur=min_d, min_cov=min_c)
        if nat_all.empty or rec_all.empty:
            continue
        t_min_val = float(min_d)
        t_lm_val = 360.0 if min_d >= 360.0 else 300.0
        for coh in coh_list:
            rn = nat_all if coh == "POOLED" else nat_all[nat_all["cohort"] == coh]
            re = rec_all if coh == "POOLED" else rec_all[rec_all["cohort"] == coh]
            if len(rn) < 15 or len(re) < 10:
                continue

            if not wc_flag and "start_day" in re.columns:
                n_w_coh = len({(str(r.cohort), str(r.record), d) for r in re.itertuples(index=False) for d in range(int(r.start_day), int(r.end_day) + 1)})
            else:
                n_w_coh = int(n_elig_w_all)

            rn_ev = (~rn["boundary_censored_end"].astype(bool)).astype(float).to_numpy()
            rn_dur = rn["dur_sec"].to_numpy(float)
            rn_pat = rn["patient"].to_numpy()
            re_dur = re["dur_sec"].to_numpy(float)
            re_ev = np.ones(len(re_dur), dtype=float)
            re_pat = re["patient"].to_numpy()

            ep_counts_r = pd.Series(re_pat).value_counts()
            mean_ep_r = float(ep_counts_r.mean())
            frac_single_ep_r = float((ep_counts_r == 1).mean())

            fr_n = fit_frailty_full(rn_dur, rn_ev, rn_pat, t_min=t_min_val)
            fr_r = fit_frailty_full(re_dur, re_ev, re_pat, t_min=t_min_val)
            kn_pt, kr_pt = float(fr_n.get("shape_k", np.nan)), float(fr_r.get("shape_k", np.nan))
            th_n, th_r = float(fr_n.get("frailty_theta", 0.5)), float(fr_r.get("frailty_theta", 0.3))
            ukn_pt, _ = fit_weibull_shape(rn_dur, rn_ev.astype(int), t_min=t_min_val)
            ukr_pt, _ = fit_weibull_shape(re_dur, re_ev.astype(int), t_min=t_min_val)
            med_n = float(np.median(rn_dur[rn_ev > 0.5])) if np.any(rn_ev > 0.5) else np.nan
            med_r = float(np.median(re_dur))
            p1h_n = conditional_term_prob(rn_dur, t_lm_val, 3600.0, events=rn_ev.astype(int))
            p1h_r = conditional_term_prob(re_dur, t_lm_val, 3600.0, events=re_ev.astype(int))

            shared_pats = np.array(sorted(set(rn_pat) | set(re_pat)))
            n_pts = len(shared_pats)
            rn_by_p = {p: np.where(rn_pat == p)[0] for p in shared_pats}
            re_by_p = {p: np.where(re_pat == p)[0] for p in shared_pats}

            d_k_fr, d_k_un, d_med_ratio, d_p1h = [], [], [], []
            kr_list, th_r_list = [], []
            hit_kn_cnt, hit_kr_cnt, hit_thr_cnt = 0, 0, 0
            n_fail = 0
            for b_i in range(n_boot):
                samp = rng.choice(shared_pats, size=n_pts, replace=True)
                idx_n = np.concatenate([rn_by_p[p] for p in samp if len(rn_by_p[p]) > 0])
                idx_r = np.concatenate([re_by_p[p] for p in samp if len(re_by_p[p]) > 0])
                # Assign distinct integer cluster IDs per bootstrap draw i so duplicate sampled patients never merge
                pids_n_b = np.concatenate([np.full(len(rn_by_p[p]), i) for i, p in enumerate(samp) if len(rn_by_p[p]) > 0])
                pids_r_b = np.concatenate([np.full(len(re_by_p[p]), i) for i, p in enumerate(samp) if len(re_by_p[p]) > 0])
                assert len(np.unique(pids_r_b)) == sum(1 for p in samp if len(re_by_p[p]) > 0)

                ok_n, kn_b, thn_b, hkn, hthn = _fit_frailty_fast_point(rn_dur[idx_n], rn_ev[idx_n], pids_n_b, t_min=t_min_val, init_k=kn_pt if np.isfinite(kn_pt) else 0.6, init_theta=th_n if np.isfinite(th_n) else 0.8)
                ok_r, kr_b, thr_b, hkr, hthr = _fit_frailty_fast_point(re_dur[idx_r], re_ev[idx_r], pids_r_b, t_min=t_min_val, init_k=kr_pt if np.isfinite(kr_pt) else 1.0, init_theta=th_r if np.isfinite(th_r) else 0.2)
                if not (ok_n and ok_r):
                    n_fail += 1
                    continue
                hit_kn_cnt += int(hkn)
                hit_kr_cnt += int(hkr)
                hit_thr_cnt += int(hthr)
                dk_fr_val = float(kr_b - kn_b)
                d_k_fr.append(dk_fr_val)
                kr_list.append(kr_b)
                th_r_list.append(thr_b)

                ukn_b, _ = fit_weibull_shape(rn_dur[idx_n], rn_ev[idx_n].astype(int), t_min=t_min_val)
                ukr_b, _ = fit_weibull_shape(re_dur[idx_r], re_ev[idx_r].astype(int), t_min=t_min_val)
                dk_un_val = float(ukr_b - ukn_b) if (np.isfinite(ukn_b) and np.isfinite(ukr_b)) else np.nan
                if np.isfinite(dk_un_val):
                    d_k_un.append(dk_un_val)

                ev_b = rn_ev[idx_n] > 0.5
                if np.any(ev_b):
                    mn_b = float(np.median(rn_dur[idx_n][ev_b]))
                    mr_b = float(np.median(re_dur[idx_r]))
                    if mn_b > 0:
                        d_med_ratio.append(mr_b / mn_b)
                pn_b = conditional_term_prob(rn_dur[idx_n], t_lm_val, 3600.0, events=rn_ev[idx_n].astype(int))
                pr_b = conditional_term_prob(re_dur[idx_r], t_lm_val, 3600.0, events=re_ev[idx_r].astype(int))
                if np.isfinite(pn_b) and np.isfinite(pr_b):
                    d_p1h.append(pr_b - pn_b)

                if cov_lbl == "all":
                    rep_rows.append({
                        "rhythm_tier": tier_lbl,
                        "frame": frame_lbl,
                        "cohort": coh,
                        "min_dur_sec": t_min_val,
                        "boot_idx": int(b_i),
                        "native_frailty_k": kn_b,
                        "native_frailty_theta": thn_b,
                        "recon_frailty_k": kr_b,
                        "recon_frailty_theta": thr_b,
                        "delta_frailty_k": dk_fr_val,
                        "native_unfrailty_k": ukn_b,
                        "recon_unfrailty_k": ukr_b,
                        "delta_unfrailty_k": dk_un_val,
                        "hit_bound_k_nat": bool(hkn),
                        "hit_bound_k_rec": bool(hkr),
                        "hit_bound_theta_rec": bool(hthr),
                    })

            def _ci(arr: list[float]) -> Tuple[float, float]:
                return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))) if len(arr) >= 20 else (np.nan, np.nan)

            dk_lo, dk_hi = _ci(d_k_fr)
            duk_lo, duk_hi = _ci(d_k_un)
            rm_lo, rm_hi = _ci(d_med_ratio)
            dp_lo, dp_hi = _ci(d_p1h)

            row_summary = {
                "rhythm_tier": tier_lbl,
                "frame": frame_lbl,
                "coverage_filter": cov_lbl,
                "cohort": coh,
                "min_dur_sec": t_min_val,
                "cond_landmark_sec": t_lm_val,
                "n_windows_eligible": int(n_w_coh),
                "n_native_runs": int(len(rn)),
                "n_recon_episodes": int(len(re)),
                "n_patients": int(n_pts),
                "n_boot_requested": int(n_boot),
                "n_boot_converged_both": int(len(d_k_fr)),
                "n_boot_failed": int(n_fail),
                "native_frailty_k": kn_pt,
                "recon_frailty_k": kr_pt,
                "delta_frailty_k": float(kr_pt - kn_pt),
                "delta_frailty_k_ci_lo": dk_lo,
                "delta_frailty_k_ci_hi": dk_hi,
                "native_unfrailty_k": float(ukn_pt),
                "recon_unfrailty_k": float(ukr_pt),
                "delta_unfrailty_k": float(ukr_pt - ukn_pt),
                "delta_unfrailty_k_ci_lo": duk_lo,
                "delta_unfrailty_k_ci_hi": duk_hi,
                "native_median_dur_sec": med_n,
                "recon_median_dur_sec": med_r,
                "ratio_median_dur": float(med_r / med_n) if med_n > 0 else np.nan,
                "ratio_median_dur_ci_lo": rm_lo,
                "ratio_median_dur_ci_hi": rm_hi,
                "native_p_term_lm_1h": p1h_n,
                "recon_p_term_lm_1h": p1h_r,
                "delta_p_term_lm_1h": float(p1h_r - p1h_n) if (np.isfinite(p1h_n) and np.isfinite(p1h_r)) else np.nan,
                "delta_p_term_lm_1h_ci_lo": dp_lo,
                "delta_p_term_lm_1h_ci_hi": dp_hi,
            }
            if cov_lbl == "all":
                out_rows.append(row_summary)
            if tier_lbl.startswith("harmonized_annotation") and frame_lbl == "recording_24h_windows" and min_d == 0.0:
                cov_rows.append(row_summary)

            if kr_list and cov_lbl == "all":
                ident_rows.append({
                    "rhythm_tier": tier_lbl,
                    "frame": frame_lbl,
                    "cohort": coh,
                    "min_dur_sec": t_min_val,
                    "n_patients": int(n_pts),
                    "n_recon_episodes": int(len(re)),
                    "mean_eps_per_patient_recon": mean_ep_r,
                    "frac_single_ep_patients_recon": frac_single_ep_r,
                    "frailty_identifiable": bool(mean_ep_r > 1.05 and hit_kr_cnt == 0),
                    "n_boot_converged": int(len(kr_list)),
                    "n_boundary_hit_k_nat": int(hit_kn_cnt),
                    "n_boundary_hit_k_rec": int(hit_kr_cnt),
                    "n_boundary_hit_theta_rec": int(hit_thr_cnt),
                    "recon_frailty_k_p50": float(np.percentile(kr_list, 50)),
                    "recon_frailty_k_p95": float(np.percentile(kr_list, 95)),
                    "recon_frailty_k_max": float(np.max(kr_list)),
                    "recon_frailty_theta_p50": float(np.percentile(th_r_list, 50)),
                    "recon_frailty_theta_p95": float(np.percentile(th_r_list, 95)),
                    "recon_frailty_theta_max": float(np.max(th_r_list)),
                })

    return pd.DataFrame(out_rows), pd.DataFrame(cov_rows), pd.DataFrame(rep_rows), pd.DataFrame(ident_rows)


def compute_threshold_disagreement(
    tiers_runs: Dict[str, pd.DataFrame],
    rec_meta: pd.DataFrame,
    thresholds_sec: List[int],
    rhythm_intervals_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Identical-window duration-threshold consistency & known-rhythm coverage sensitivity:
    For every physical observation window W = [w_s, w_e], clip EVERY AF interval to W
    before computing cumulative AF exposure A_W and longest uninterrupted run L_W
    (enforcing L_W <= A_W identically). Repeat across all windows, >=95% known-rhythm
    coverage windows, and >=99% known-rhythm coverage windows.
    """
    rm_map = {
        (r.cohort, str(r.record)): (float(r.record_dur_sec), str(r.af_subtype), str(r.patient))
        for r in rec_meta.itertuples(index=False)
    }
    itv_by_rec = (
        {k: g.sort_values("start_sec") for k, g in rhythm_intervals_df.groupby(["cohort", "record"], sort=True)}
        if (rhythm_intervals_df is not None and not rhythm_intervals_df.empty)
        else {}
    )

    rng = np.random.default_rng(SEED)
    summary_rows: List[Dict[str, Any]] = []
    cov_summary_rows: List[Dict[str, Any]] = []
    example_rows: List[Dict[str, Any]] = []

    for tier_name, runs_df in tiers_runs.items():
        if runs_df is None or len(runs_df) == 0:
            continue

        win_records = []
        runs_by_rec = {k: g.sort_values("onset_sec") for k, g in runs_df.groupby(["cohort", "record"], sort=True)}

        for (coh, rec), (rec_dur, subtype, pat) in sorted(rm_map.items()):
            n_win = max(1, int(np.floor((rec_dur + 400.0) / 86400.0))) if rec_dur >= 86000.0 else 0
            grp = runs_by_rec.get((coh, str(rec)), pd.DataFrame())
            itv_rec = itv_by_rec.get((coh, str(rec)), pd.DataFrame())
            for k in range(n_win):
                ws, we = k * 86400.0, min(rec_dur, (k + 1) * 86400.0)
                win_dur = float(we - ws)
                if not itv_rec.empty:
                    iov = itv_rec[(itv_rec["end_sec"] > ws) & (itv_rec["start_sec"] < we)]
                    ic_durs = np.maximum(0.0, np.minimum(iov["end_sec"], we) - np.maximum(iov["start_sec"], ws))
                    unk_obs = float(np.sum(ic_durs[iov["rhythm"].to_numpy() == "unknown"]))
                    known_obs = float(max(0.0, win_dur - unk_obs))
                else:
                    known_obs, unk_obs = win_dur, 0.0
                known_frac = float(known_obs / max(1.0, win_dur))

                if not grp.empty:
                    ov_mask = (grp["offset_sec"] > ws) & (grp["onset_sec"] < we)
                    sub = grp[ov_mask]
                    clipped_durs = np.maximum(0.0, np.minimum(sub["offset_sec"].to_numpy(float), we) - np.maximum(sub["onset_sec"].to_numpy(float), ws))
                    pos = clipped_durs > 0.0
                    clipped_durs = clipped_durs[pos]
                    sub_pos = sub.iloc[pos] if len(sub) else sub
                    sum_af = float(np.sum(clipped_durs)) if len(clipped_durs) else 0.0
                    max_run = float(np.max(clipped_durs)) if len(clipped_durs) else 0.0
                    assert max_run <= sum_af + 1e-6, f"Clipped L_W={max_run} > A_W={sum_af} in {coh} {rec} w={k}"
                    n_runs = int(len(clipped_durs))
                    c_on = np.maximum(sub_pos["onset_sec"].to_numpy(float), ws)
                    c_off = np.minimum(sub_pos["offset_sec"].to_numpy(float), we)
                    med_gap = float(np.median(c_on[1:] - c_off[:-1])) if n_runs >= 2 else np.nan
                    run_list_str = ";".join(f"{d:.0f}s" for d in clipped_durs[:8])
                else:
                    sum_af, max_run, n_runs, med_gap, run_list_str = 0.0, 0.0, 0, np.nan, ""

                win_records.append({
                    "cohort": coh,
                    "af_subtype": subtype,
                    "record": rec,
                    "patient": pat,
                    "window_idx": k,
                    "window_dur_sec": win_dur,
                    "known_obs_sec": known_obs,
                    "unknown_obs_sec": unk_obs,
                    "known_obs_frac": known_frac,
                    "is_multiday_rec": bool(rec_dur >= 36 * 3600),
                    "sum_af_sec": sum_af,
                    "max_run_sec": max_run,
                    "n_runs": n_runs,
                    "median_gap_sec": med_gap,
                    "run_durs_preview": run_list_str,
                })

        wdf_all = pd.DataFrame(win_records)
        if wdf_all.empty:
            continue

        cov_specs = [("all", 0.0), ("known_cov_ge_95pct", 0.95), ("known_cov_ge_99pct", 0.99)]
        for cov_name, cov_min in cov_specs:
            wdf = wdf_all[wdf_all["known_obs_frac"] >= cov_min].copy()
            if wdf.empty:
                continue

            strata = [
                ("POOLED", wdf),
                ("IRIDIA", wdf[wdf["cohort"] == "IRIDIA"]),
                ("IRIDIA_multiday_ge36h", wdf[(wdf["cohort"] == "IRIDIA") & (wdf["is_multiday_rec"])]),
                ("SHDB", wdf[wdf["cohort"] == "SHDB"]),
                ("SHDB_PAF", wdf[(wdf["cohort"] == "SHDB") & (wdf["af_subtype"] == "PAF")]),
                ("SHDB_PerAF", wdf[(wdf["cohort"] == "SHDB") & (wdf["af_subtype"].str.upper().str.contains("PER"))]),
                ("LTAFDB", wdf[wdf["cohort"] == "LTAFDB"]),
            ]

            for s_name, sw in strata:
                if len(sw) == 0:
                    continue
                if cov_name != "all" and s_name not in ("POOLED", "IRIDIA", "SHDB_PAF", "LTAFDB"):
                    continue
                pts, pt_inv = np.unique(sw["patient"].to_numpy(), return_inverse=True)
                n_pts = len(pts)
                n_win_tot = len(sw)

                a_arr = sw["sum_af_sec"].to_numpy(float)
                l_arr = sw["max_run_sec"].to_numpy(float)
                pt_agg = sw.groupby("patient", as_index=False).agg(
                    max_daily_af=("sum_af_sec", "max"),
                    max_unint_run=("max_run_sec", "max"),
                )

                n_boot = 500
                boot_pat_w = np.zeros((n_boot, n_pts), dtype=np.float64)
                for b in range(n_boot):
                    boot_pat_w[b] = np.bincount(rng.integers(0, n_pts, size=n_pts), minlength=n_pts)

                w_dur_min = float(sw["window_dur_sec"].min())
                w_dur_med = float(sw["window_dur_sec"].median())
                w_dur_max = float(sw["window_dur_sec"].max())
                n_lt_24h = int((sw["window_dur_sec"] < 86399.0).sum())
                known_med = float(sw["known_obs_sec"].median())
                unk_max = float(sw["unknown_obs_sec"].max())
                n_with_unk = int((sw["unknown_obs_sec"] > 1e-6).sum())

                for thr in thresholds_sec:
                    daily_ge = a_arr >= thr
                    native_ge = l_arr >= thr
                    both_w = int(np.sum(daily_ge & native_ge))
                    up_dis_w = int(np.sum(daily_ge & (~native_ge)))
                    rev_dis_w = int(np.sum((~daily_ge) & native_ge))
                    neither_w = int(np.sum((~daily_ge) & (~native_ge)))
                    assert rev_dis_w == 0, f"Threshold consistency violated: {tier_name} {s_name} T={thr} has L_W>=T & A_W<T count={rev_dis_w}"

                    p_dis_win = float(up_dis_w / n_win_tot)
                    n_daily_w = both_w + up_dis_w
                    p_dis_given_daily_w = float(up_dis_w / n_daily_w) if n_daily_w > 0 else np.nan
                    n_dis_unk_free = int(np.sum(daily_ge & (~native_ge) & (sw["unknown_obs_sec"].to_numpy(float) <= 1e-6)))

                    pat_tot = np.bincount(pt_inv, minlength=n_pts).astype(float)
                    pat_up = np.bincount(pt_inv, weights=(daily_ge & (~native_ge)).astype(float), minlength=n_pts)
                    pat_d = np.bincount(pt_inv, weights=daily_ge.astype(float), minlength=n_pts)

                    b_tot = boot_pat_w @ pat_tot
                    b_up = boot_pat_w @ pat_up
                    b_d = boot_pat_w @ pat_d
                    b_win = b_up / np.maximum(1.0, b_tot)
                    valid_bd = b_d > 0
                    w_ci_lo, w_ci_hi = np.percentile(b_win, [2.5, 97.5])
                    cw_lo, cw_hi = np.percentile(b_up[valid_bd] / b_d[valid_bd], [2.5, 97.5]) if np.any(valid_bd) else (np.nan, np.nan)

                    p_daily_pt = pt_agg["max_daily_af"].to_numpy(float) >= thr
                    p_native_pt = pt_agg["max_unint_run"].to_numpy(float) >= thr
                    both_pt = int(np.sum(p_daily_pt & p_native_pt))
                    up_dis_pt = int(np.sum(p_daily_pt & (~p_native_pt)))
                    rev_dis_pt = int(np.sum((~p_daily_pt) & p_native_pt))
                    neither_pt = int(np.sum((~p_daily_pt) & (~p_native_pt)))
                    assert rev_dis_pt == 0
                    p_dis_pt = float(up_dis_pt / n_pts)
                    n_daily_pt = both_pt + up_dis_pt
                    p_dis_given_daily_pt = float(up_dis_pt / n_daily_pt) if n_daily_pt > 0 else np.nan
                    b_pt = (boot_pat_w @ (p_daily_pt & (~p_native_pt)).astype(float)) / n_pts
                    p_ci_lo, p_ci_hi = np.percentile(b_pt, [2.5, 97.5])

                    row_td = {
                        "rhythm_tier": tier_name,
                        "coverage_filter": cov_name,
                        "stratum": s_name,
                        "threshold_sec": int(thr),
                        "n_patients_eligible": int(n_pts),
                        "n_windows_eligible": int(n_win_tot),
                        "n_runs_clipped_sum": int(sw["n_runs"].sum()),
                        "window_dur_min_sec": w_dur_min,
                        "window_dur_median_sec": w_dur_med,
                        "window_dur_max_sec": w_dur_max,
                        "n_windows_below_86400s": n_lt_24h,
                        "known_obs_median_sec": known_med,
                        "unknown_obs_max_sec": unk_max,
                        "n_windows_with_any_unknown": n_with_unk,
                        "n_windows_disagree_unknown_free": n_dis_unk_free,
                        "n_windows_both_ge_T": both_w,
                        "n_windows_daily_ge_T": int(n_daily_w),
                        "n_windows_native_ge_T": int(both_w),
                        "n_windows_disagree": up_dis_w,
                        "n_windows_native_only_ge_T": rev_dis_w,
                        "n_windows_neither_ge_T": neither_w,
                        "prop_windows_disagree": p_dis_win,
                        "prop_windows_disagree_ci_lo": float(w_ci_lo),
                        "prop_windows_disagree_ci_hi": float(w_ci_hi),
                        "prop_daily_ge_T_disagree_win": p_dis_given_daily_w,
                        "prop_daily_ge_T_disagree_win_ci_lo": float(cw_lo),
                        "prop_daily_ge_T_disagree_win_ci_hi": float(cw_hi),
                        "n_patients_both_ge_T": both_pt,
                        "n_patients_daily_ge_T": int(n_daily_pt),
                        "n_patients_native_ge_T": int(both_pt),
                        "n_patients_disagree": up_dis_pt,
                        "n_patients_native_only_ge_T": rev_dis_pt,
                        "n_patients_neither_ge_T": neither_pt,
                        "prop_patients_disagree": p_dis_pt,
                        "prop_patients_disagree_ci_lo": float(p_ci_lo),
                        "prop_patients_disagree_ci_hi": float(p_ci_hi),
                        "prop_daily_ge_T_disagree_pt": p_dis_given_daily_pt,
                        "unknown_interval_assumption": "native_breaks_on_unknown_daily_sums_within_window",
                    }
                    if cov_name == "all":
                        summary_rows.append(row_td)
                    if thr in (3600, 21600):
                        cov_summary_rows.append(row_td)

        if tier_name == "harmonized_annotation":
            for thr_ex in [3600, 21600]:
                ex = wdf_all[(wdf_all["sum_af_sec"] >= thr_ex) & (wdf_all["max_run_sec"] < thr_ex)].copy()
                if not ex.empty:
                    ex["threshold_crossed_sec"] = thr_ex
                    ex["inflation_ratio_sum_to_max"] = ex["sum_af_sec"] / np.maximum(ex["max_run_sec"], 1.0)
                    ex = ex.sort_values(["cohort", "n_runs", "sum_af_sec"], ascending=[True, False, False]).groupby("cohort").head(5)
                    example_rows.append(ex)

    ex_df = pd.concat(example_rows, ignore_index=True) if example_rows else pd.DataFrame()
    return pd.DataFrame(summary_rows), pd.DataFrame(cov_summary_rows), ex_df


def main() -> None:
    ctx = get_run_context()
    run_dir: Path = ctx["run_dir"]
    cfg: Dict[str, Any] = ctx["config"]

    stage2_dir = run_dir / "stage2"
    out_dir = run_dir / "aim2"
    out_dir.mkdir(parents=True, exist_ok=True)

    spitaler_rules = cfg.get("spitaler_rules", DEFAULT_SPITALER_RULES)
    gap_family = cfg.get("gap_merge_family_sec", [0, 30, 120, 360, 2400])
    cohorts = cfg.get("cohorts", ["IRIDIA", "SHDB", "LTAFDB"])

    metrics_rows: List[Dict[str, Any]] = []
    all_recon_g0: Optional[pd.DataFrame] = None

    for g in gap_family:
        runs_path = stage2_dir / f"runs_g{g}.parquet"
        if not runs_path.exists():
            print(f"[Aim 2] WARNING: {runs_path} not found, skipping g={g}")
            continue

        runs_df = pd.read_parquet(runs_path)
        recon_df = emulate_spitaler_episodes(runs_df, spitaler_rules, exclude_boundary=True)
        recon_df["gap_sec"] = g

        if g == 0:
            all_recon_g0 = recon_df.copy()

        # Evaluate per cohort and POOLED (known-onset runs only for elapsed durations!)
        runs_known = runs_df[~runs_df["boundary_censored_start"].astype(bool)].copy()
        for coh in list(cohorts) + ["POOLED"]:
            if coh == "POOLED":
                r_sub = runs_known
                e_sub = recon_df
            else:
                r_sub = runs_known[runs_known["cohort"] == coh]
                e_sub = recon_df[recon_df["cohort"] == coh]

            r_events = (~r_sub["boundary_censored_end"].astype(bool)).astype(int).to_numpy()
            r_durs = r_sub["dur_sec"].to_numpy(dtype=float)
            r_uncens_durs = r_durs[r_events == 1]

            e_durs = e_sub["dur_sec"].to_numpy(dtype=float)
            e_events = np.ones(len(e_durs), dtype=int)

            true_k, true_lam = fit_weibull_shape(r_durs, r_events)
            recon_k, recon_lam = fit_weibull_shape(e_durs, e_events)

            total_recon_af = float(np.sum(e_durs)) if len(e_durs) > 0 else 0.0
            total_chain_af = float(e_sub["af_sec_reinit_chain"].sum()) if len(e_sub) > 0 else 0.0
            total_unint_af = float(e_sub["af_sec_uninterrupted"].sum()) if len(e_sub) > 0 else 0.0
            frac_chain = total_chain_af / total_recon_af if total_recon_af > 0 else float("nan")
            frac_unint = total_unint_af / total_recon_af if total_recon_af > 0 else float("nan")

            true_p_1h_1h = conditional_term_prob(r_durs, 3600.0, 3600.0, events=r_events)
            recon_p_1h_1h = conditional_term_prob(e_durs, 3600.0, 3600.0, events=e_events)
            true_p_1h_24h = conditional_term_prob(r_durs, 3600.0, 86400.0, events=r_events)
            recon_p_1h_24h = conditional_term_prob(e_durs, 3600.0, 86400.0, events=e_events)
            true_p_24h_24h = conditional_term_prob(r_durs, 86400.0, 86400.0, events=r_events)
            recon_p_24h_24h = conditional_term_prob(e_durs, 86400.0, 86400.0, events=e_events)

            metrics_rows.append(
                {
                    "gap_sec": g,
                    "cohort": coh,
                    "n_true_runs_all": int(len(r_sub)),
                    "n_true_runs_uncensored": int(len(r_uncens_durs)),
                    "n_recon_episodes": int(len(e_sub)),
                    "compression_ratio": (
                        float(len(r_uncens_durs) / len(e_sub)) if len(e_sub) > 0 else float("nan")
                    ),
                    "true_median_dur_sec": float(np.median(r_uncens_durs)) if len(r_uncens_durs) > 0 else float("nan"),
                    "recon_median_dur_sec": float(np.median(e_durs)) if len(e_durs) > 0 else float("nan"),
                    "true_weibull_shape": true_k,
                    "recon_weibull_shape": recon_k,
                    "true_p_term_1h_w1h": true_p_1h_1h,
                    "recon_p_term_1h_w1h": recon_p_1h_1h,
                    "true_p_term_1h_w24h": true_p_1h_24h,
                    "recon_p_term_1h_w24h": recon_p_1h_24h,
                    "true_p_term_24h_w24h": true_p_24h_24h,
                    "recon_p_term_24h_w24h": recon_p_24h_24h,
                    "total_internal_terminations": int(e_sub["n_internal_terminations"].sum()) if len(e_sub) > 0 else 0,
                    "mean_internal_terminations": float(e_sub["n_internal_terminations"].mean()) if len(e_sub) > 0 else float("nan"),
                    "median_internal_terminations": float(e_sub["n_internal_terminations"].median()) if len(e_sub) > 0 else float("nan"),
                    "total_internal_reinit_lt360": int(e_sub["n_internal_reinit_lt360"].sum()) if len(e_sub) > 0 else 0,
                    "frac_recon_af_reinit_chain": frac_chain,
                    "frac_recon_af_uninterrupted": frac_unint,
                }
            )

    metrics_df = add_provenance(pd.DataFrame(metrics_rows), ctx)
    metrics_path = out_dir / "table2_reconstruction_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    # Priority 4 & 5: Reconciliation audit, matched-support Weibull comparison, & threshold disagreement
    runs_g0_full = pd.read_parquet(stage2_dir / "runs_g0.parquet")
    rec_meta = _load_record_metadata_and_clocks(run_dir)

    recon_audit_df, wall_ep_df = run_reconciliation_and_boundary_audit(runs_g0_full, rec_meta, spitaler_rules)
    add_provenance(recon_audit_df, ctx).to_csv(out_dir / "native_to_recon_reconciliation.csv", index=False)
    if not wall_ep_df.empty:
        add_provenance(wall_ep_df, ctx).to_parquet(out_dir / "wall_clock_reconstructed_episodes.parquet", index=False)

    if all_recon_g0 is not None:
        recon_out = add_provenance(all_recon_g0, ctx)
        recon_path = out_dir / "reconstructed_episodes.parquet"
        recon_out.to_parquet(recon_path, index=False)

        matched_df, elig_recon_df, run_ledger_df = compute_matched_support_hazard_comparison(runs_g0_full, all_recon_g0, spitaler_rules)
        add_provenance(matched_df, ctx).to_csv(out_dir / "matched_support_hazard_comparison.csv", index=False)
        add_provenance(elig_recon_df, ctx).to_csv(out_dir / "matched_eligibility_reconciliation.csv", index=False)
        add_provenance(run_ledger_df, ctx).to_csv(out_dir / "matched_eligibility_run_ledger.csv", index=False)

    tiers_runs: Dict[str, pd.DataFrame] = {"harmonized_annotation": runs_g0_full}
    cs_path = stage2_dir / "runs_g0_continuity_nonsinus_gaps.parquet"
    if cs_path.exists():
        tiers_runs["continuity_nonsinus_gaps"] = pd.read_parquet(cs_path)
    sn_path = stage2_dir / "runs_g0_continuity_except_normal.parquet"
    if sn_path.exists():
        tiers_runs["continuity_except_normal"] = pd.read_parquet(sn_path)

    itv_path = run_dir / "stage1" / "rhythm_intervals.parquet"
    rhythm_itv = pd.read_parquet(itv_path) if itv_path.exists() else None

    paired_boot_df, cov_agg_df, boot_rep_df, ident_df = compute_paired_aggregation_bootstrap(
        tiers_runs, rec_meta, spitaler_rules, n_boot=300, rhythm_intervals_df=rhythm_itv
    )
    add_provenance(paired_boot_df, ctx).to_csv(out_dir / "paired_aggregation_bootstrap.csv", index=False)
    add_provenance(cov_agg_df, ctx).to_csv(out_dir / "coverage_sensitivity_aggregation.csv", index=False)
    add_provenance(boot_rep_df, ctx).to_csv(out_dir / "paired_bootstrap_replicates.csv", index=False)
    add_provenance(ident_df, ctx).to_csv(out_dir / "bootstrap_identifiability_audit.csv", index=False)

    disagree_df, cov_td_df, ex_df = compute_threshold_disagreement(tiers_runs, rec_meta, [30, 360, 3600, 21600, 86400], rhythm_intervals_df=rhythm_itv)
    add_provenance(disagree_df, ctx).to_csv(out_dir / "threshold_disagreement.csv", index=False)
    add_provenance(cov_td_df, ctx).to_csv(out_dir / "coverage_sensitivity_threshold.csv", index=False)
    add_provenance(ex_df, ctx).to_csv(out_dir / "threshold_disagreement_examples.csv", index=False)

    print(f"[Aim 2] Saved {metrics_path} ({len(metrics_df)} rows), reconciliation, matched-support, paired bootstrap ({len(paired_boot_df)} rows), coverage sensitivity, replicates & threshold disagreement in {out_dir}")


if __name__ == "__main__":
    main()
