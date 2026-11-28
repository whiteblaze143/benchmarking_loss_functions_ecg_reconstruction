#!/usr/bin/env python3
"""Stage 4: Build termination event tables from Stage 2 harmonised runs."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from common import (
    compute_definitions_hash,
    get_git_hash,
    load_cohorts,
    load_definitions,
    resolve_run_dir,
    stamp_dataframe,
    utc_timestamp_iso,
    write_json,
)


def _parse_clock_hour_start(cohorts_cfg: dict, beat_summary: pd.DataFrame) -> dict[tuple[str, str], float]:
    """Return recording start clock hour in [0, 24) (or NaN if unavailable) per (cohort, record)."""
    start_hours: dict[tuple[str, str], float] = {}

    # 1. IRIDIA
    ir_meta = pd.read_csv(cohorts_cfg["cohorts"]["IRIDIA"]["metadata_csv"])
    for r in ir_meta.itertuples(index=False):
        ts = getattr(r, "record_start_time", None)
        hr = np.nan
        if pd.notna(ts):
            try:
                dt = datetime.fromisoformat(str(ts))
                hr = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
            except Exception:
                hr = np.nan
        start_hours[("IRIDIA", str(r.record_id))] = float(hr)

    # 2. SHDB
    shdb_meta = pd.read_csv(cohorts_cfg["cohorts"]["SHDB"]["metadata_csv"])
    for r in shdb_meta.itertuples(index=False):
        rec_id = str(r.Data_ID).zfill(3)
        ts = getattr(r, "Holter_start_time", None)
        hr = np.nan
        if pd.notna(ts):
            s = str(ts).strip()
            for fmt in ("%I:%M %p", "%H:%M:%S", "%H:%M"):
                try:
                    dt = datetime.strptime(s, fmt)
                    hr = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
                    break
                except Exception:
                    continue
        start_hours[("SHDB", rec_id)] = float(hr)

    # 3. LTAFDB
    ltaf_base = Path(cohorts_cfg["cohorts"]["LTAFDB"]["base_dir"])
    for r in beat_summary[beat_summary["cohort"] == "LTAFDB"].itertuples(index=False):
        rec_id = str(r.record)
        hr = np.nan
        try:
            hdr = wfdb.rdheader(str(ltaf_base / rec_id))
            if hdr.base_time is not None:
                bt = hdr.base_time
                hr = bt.hour + bt.minute / 60.0 + bt.second / 3600.0
        except Exception:
            hr = np.nan
        start_hours[("LTAFDB", rec_id)] = float(hr)

    return start_hours


def _load_shdb_covariates(cohorts_cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(cohorts_cfg["cohorts"]["SHDB"]["metadata_csv"])
    df["record"] = df["Data_ID"].astype(str).str.zfill(3)
    df["cohort"] = "SHDB"
    keep = {
        "record": "record",
        "cohort": "cohort",
        "Age_at_Holter": "shdb_age",
        "Sex": "shdb_sex",
        "BMI": "shdb_bmi",
        "AF_Type": "shdb_af_type",
        "Previous_AF_Ablation": "shdb_prev_ablation",
        "Antiarrhythmic_Drug_nonBB": "shdb_aad_nonbb",
        "Antiarrhythmic_Drug_BB": "shdb_aad_bb",
        "Anticoagulation": "shdb_anticoag",
        "Echo_LAD": "shdb_echo_lad",
        "Echo_LVEF": "shdb_echo_lvef",
        "CHF": "shdb_chf",
        "HTN": "shdb_htn",
        "DM": "shdb_dm",
        "Stroke": "shdb_stroke",
        "Vascular_Diseases": "shdb_vascular",
    }
    cols_present = [c for c in keep if c in df.columns]
    out = df[cols_present].rename(columns={c: keep[c] for c in cols_present})
    return out.drop_duplicates(subset=["cohort", "record"])


def _build_event_table(
    runs_df: pd.DataFrame,
    rec_dur_map: dict[tuple[str, str], float],
    start_hour_map: dict[tuple[str, str], float],
    shdb_cov: pd.DataFrame,
    min_run_sec: float,
    reinit_thresholds: list[int],
    followup_min_sec: float,
    postD_end_sec: float,
) -> pd.DataFrame:
    rows = []
    for (cohort, record), grp in runs_df.groupby(["cohort", "record"], sort=True):
        grp = grp.sort_values("onset_sec").reset_index(drop=True)
        rec_dur = rec_dur_map[(cohort, str(record))]
        rec_start_hr = start_hour_map.get((cohort, str(record)), np.nan)

        cum_af = 0.0
        n_prior_term = 0

        for idx in range(len(grp)):
            r = grp.iloc[idx]
            dur = float(r["dur_sec"])
            af_only = float(r["af_only_dur_sec"]) if "af_only_dur_sec" in r else dur
            t0 = float(r["offset_sec"])
            bc_end = bool(r["boundary_censored_end"])

            cum_af_before_t0 = cum_af + af_only
            cum_af = cum_af_before_t0

            if dur < min_run_sec or bc_end:
                continue

            # Next run in same record (any AF run after t0)
            if idx + 1 < len(grp):
                next_onset = float(grp.iloc[idx + 1]["onset_sec"])
                time_to_reinit = max(0.0, next_onset - t0)
                event = 1
            else:
                time_to_reinit = max(0.0, rec_dur - t0)
                event = 0

            rem_followup = max(0.0, rec_dur - t0)
            clock_hr = float((rec_start_hr + t0 / 3600.0) % 24.0) if np.isfinite(rec_start_hr) else np.nan
            cosinor_sin = float(np.sin(2.0 * np.pi * clock_hr / 24.0)) if np.isfinite(clock_hr) else np.nan
            cosinor_cos = float(np.cos(2.0 * np.pi * clock_hr / 24.0)) if np.isfinite(clock_hr) else np.nan

            row = {
                "event_id": f"{r['run_id']}_term",
                "run_id": str(r["run_id"]),
                "record": str(record),
                "patient": str(r["patient"]),
                "cohort": str(cohort),
                "onset_sec": float(r["onset_sec"]),
                "t0": t0,
                "dur_sec": dur,
                "log_dur": float(np.log(dur)),
                "n_beats": int(r["n_beats"]) if pd.notna(r["n_beats"]) else pd.NA,
                "beat_source": str(r["beat_source"]) if "beat_source" in r.index and pd.notna(r["beat_source"]) else "unknown",
                "boundary_censored_start": bool(r["boundary_censored_start"]),
                "prev_rhythm_class": str(r["prev_rhythm_class"]) if "prev_rhythm_class" in r.index and pd.notna(r["prev_rhythm_class"]) else "UNKNOWN",
                "next_rhythm_class": str(r["next_rhythm_class"]) if "next_rhythm_class" in r.index and pd.notna(r["next_rhythm_class"]) else "UNKNOWN",
                "next_rhythm_native": str(r["next_rhythm_native"]) if "next_rhythm_native" in r.index and pd.notna(r["next_rhythm_native"]) else "UNKNOWN",
                "terminates_to_confirmed_sinus": bool(r["terminates_to_confirmed_sinus"]) if "terminates_to_confirmed_sinus" in r.index and pd.notna(r["terminates_to_confirmed_sinus"]) else False,
                "terminates_to_strict_n": bool(r["terminates_to_strict_n"]) if "terminates_to_strict_n" in r.index and pd.notna(r["terminates_to_strict_n"]) else False,
                "time_to_reinit_sec": float(time_to_reinit),
                "event": int(event),
                "followup_rem_sec": float(rem_followup),
                "has_followup_3600s": bool(rem_followup >= followup_min_sec),
                "clock_hour_t0": clock_hr,
                "cosinor_sin": cosinor_sin,
                "cosinor_cos": cosinor_cos,
                "cum_af_sec_before_t0": float(cum_af_before_t0),
                "n_prior_terminations": int(n_prior_term),
                "position_in_record_frac": float(t0 / rec_dur) if rec_dur > 0 else np.nan,
                "has_postD_window": bool(rem_followup >= postD_end_sec and (event == 0 or time_to_reinit >= postD_end_sec)),
                "sqi_bsqi_postA": np.nan,
                "sqi_bsqi_postD": np.nan,
                "sqi_stratum_postA": "uncomputed",
                "sqi_stratum_postD": "uncomputed",
            }

            for thr in reinit_thresholds:
                col = f"reinit_{thr}s"
                if event == 1 and time_to_reinit <= thr:
                    row[col] = 1.0
                elif time_to_reinit > thr or (event == 0 and rem_followup >= thr):
                    row[col] = 0.0
                else:
                    row[col] = np.nan

            rows.append(row)
            n_prior_term += 1

    ev_df = pd.DataFrame(rows)
    if not ev_df.empty:
        if "n_beats" in ev_df.columns:
            ev_df["n_beats"] = pd.array(ev_df["n_beats"].tolist(), dtype="Int64")
        ev_df = ev_df.merge(shdb_cov, on=["cohort", "record"], how="left")
        assert (ev_df["time_to_reinit_sec"] >= 0).all(), "Negative time_to_reinit_sec detected!"
        assert (ev_df["dur_sec"] >= min_run_sec).all(), "Run shorter than min_run_sec in event table!"
    return ev_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4: Build termination event tables")
    parser.add_argument("--run-dir", type=str, default=None, help="Run directory (defaults to LATEST_RUN.txt)")
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.run_dir)
    stage1_dir = run_dir / "stage1"
    stage2_dir = run_dir / "stage2"
    stage4_dir = run_dir / "stage4"
    stage4_dir.mkdir(parents=True, exist_ok=True)

    defs = load_definitions()
    cohorts_cfg = load_cohorts()
    defs_hash = compute_definitions_hash()
    git_hash = get_git_hash()
    run_ts = utc_timestamp_iso()

    beat_summary = pd.read_parquet(stage1_dir / "beat_summary.parquet")
    rec_dur_map = {
        (r.cohort, str(r.record)): float(r.record_dur_sec)
        for r in beat_summary.itertuples(index=False)
    }
    start_hour_map = _parse_clock_hour_start(cohorts_cfg, beat_summary)
    shdb_cov = _load_shdb_covariates(cohorts_cfg)

    min_run_primary = float(defs["min_run_sec_primary"])
    min_run_sens = [float(x) for x in defs["min_run_sec_sensitivity"]]
    reinit_thresholds = [int(x) for x in defs["reinit_thresholds_sec"]]
    followup_min_sec = float(defs["post_termination_followup_min_sec"])
    postD_end_sec = float(defs["post_window_D_sec"][1])
    gap_family = [int(g) for g in defs["gap_merge_family_sec"]]

    summary = {"variants": {}, "git_hash": git_hash, "definitions_hash": defs_hash, "run_timestamp": run_ts}

    for g in gap_family:
        runs_df = pd.read_parquet(stage2_dir / f"runs_g{g}.parquet")
        ev_df = _build_event_table(
            runs_df,
            rec_dur_map,
            start_hour_map,
            shdb_cov,
            min_run_sec=min_run_primary,
            reinit_thresholds=reinit_thresholds,
            followup_min_sec=followup_min_sec,
            postD_end_sec=postD_end_sec,
        )
        ev_df = stamp_dataframe(ev_df, git_hash=git_hash, definitions_hash=defs_hash, run_timestamp=run_ts)
        out_path = stage4_dir / f"events_g{g}.parquet"
        ev_df.to_parquet(out_path, index=False)

        cohort_stats = {}
        for c, cgrp in ev_df.groupby("cohort"):
            r360 = cgrp["reinit_360s"].dropna()
            cohort_stats[c] = {
                "n_events": int(len(cgrp)),
                "n_patients": int(cgrp["patient"].nunique()),
                "reinit_360s_rate": float(r360.mean()) if len(r360) > 0 else None,
                "reinit_360s_n": int(len(r360)),
            }
        summary["variants"][f"g{g}"] = {
            "n_events": int(len(ev_df)),
            "min_run_sec": min_run_primary,
            "cohorts": cohort_stats,
            "path": str(out_path),
        }

    # Duration sensitivity variants on g0 (120s, 300s)
    runs_g0 = pd.read_parquet(stage2_dir / "runs_g0.parquet")
    for min_s in min_run_sens:
        ev_s = _build_event_table(
            runs_g0,
            rec_dur_map,
            start_hour_map,
            shdb_cov,
            min_run_sec=min_s,
            reinit_thresholds=reinit_thresholds,
            followup_min_sec=followup_min_sec,
            postD_end_sec=postD_end_sec,
        )
        ev_s = stamp_dataframe(ev_s, git_hash=git_hash, definitions_hash=defs_hash, run_timestamp=run_ts)
        out_path = stage4_dir / f"events_g0_min{int(min_s)}s.parquet"
        ev_s.to_parquet(out_path, index=False)
        summary["variants"][f"g0_min{int(min_s)}s"] = {
            "n_events": int(len(ev_s)),
            "min_run_sec": min_s,
            "path": str(out_path),
        }

    # Rhythm-definition tier variants on g0
    for tier_name in ["continuity_nonsinus_gaps", "continuity_except_normal"]:
        tp = stage2_dir / f"runs_g0_{tier_name}.parquet"
        if tp.exists():
            runs_t = pd.read_parquet(tp)
            ev_t = _build_event_table(
                runs_t,
                rec_dur_map,
                start_hour_map,
                shdb_cov,
                min_run_sec=min_run_primary,
                reinit_thresholds=reinit_thresholds,
                followup_min_sec=followup_min_sec,
                postD_end_sec=postD_end_sec,
            )
            ev_t = stamp_dataframe(ev_t, git_hash=git_hash, definitions_hash=defs_hash, run_timestamp=run_ts)
            out_t = stage4_dir / f"events_g0_{tier_name}.parquet"
            ev_t.to_parquet(out_t, index=False)
            summary["variants"][f"g0_{tier_name}"] = {
                "n_events": int(len(ev_t)),
                "min_run_sec": min_run_primary,
                "path": str(out_t),
            }

    write_json(stage4_dir / "stage4_summary.json", summary)

    print("[Stage 4] OK | Primary events_g0:")
    g0_info = summary["variants"]["g0"]
    print(f"  total events={g0_info['n_events']}")
    for c, st in g0_info["cohorts"].items():
        rate_str = f"{st['reinit_360s_rate']*100:.1f}%" if st["reinit_360s_rate"] is not None else "NA"
        print(f"  {c:6s}: events={st['n_events']:4d}, patients={st['n_patients']:3d}, reinit<360s={rate_str} (n={st['reinit_360s_n']})")


if __name__ == "__main__":
    main()
