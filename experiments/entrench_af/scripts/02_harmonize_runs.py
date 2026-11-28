#!/usr/bin/env python3
"""Stage 2: Harmonise AF runs (annotation_merge_beats=5, min_run_beats=5) and build gap-merge family."""

from __future__ import annotations

import argparse
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

NON_BEAT_SYMBOLS = {"+", "~", "|", "s", "T", "*", "D", "=", '"', "@", "[", "]", "!", "x", "(", ")", "p", "t", "u", "`", "'", "^"}


def _load_beat_times_sec(cohort: str, record_id: str, fs: float, cohorts_cfg: dict) -> np.ndarray | None:
    base = Path(cohorts_cfg["cohorts"][cohort]["base_dir"])
    if cohort == "LTAFDB":
        ann = wfdb.rdann(str(base / record_id), "atr")
        samples = np.array([s for s, sym in zip(ann.sample, ann.symbol) if sym not in NON_BEAT_SYMBOLS], dtype=np.float64)
        return np.sort(samples / fs)
    if cohort == "SHDB":
        qrs_path = base / f"{record_id}.qrs"
        ext = "qrs" if qrs_path.exists() else "atr"
        ann = wfdb.rdann(str(base / record_id), ext)
        samples = np.array([s for s, sym in zip(ann.sample, ann.symbol) if sym not in NON_BEAT_SYMBOLS], dtype=np.float64)
        return np.sort(samples / fs)
    return None


def _count_beats_in_interval(beat_times: np.ndarray | None, start_sec: float, end_sec: float) -> int | None:
    if beat_times is None or len(beat_times) == 0:
        return None
    lo = int(np.searchsorted(beat_times, start_sec, side="left"))
    hi = int(np.searchsorted(beat_times, end_sec, side="right"))
    return max(0, hi - lo)


def _has_unknown_between(rec_intervals: pd.DataFrame, t_start: float, t_end: float) -> bool:
    sub = rec_intervals[(rec_intervals["end_sec"] > t_start + 1e-6) & (rec_intervals["start_sec"] < t_end - 1e-6)]
    return bool((sub["rhythm"] == "unknown").any())


def _between_all_disallowed_nonaf(
    rec_intervals: pd.DataFrame, t_start: float, t_end: float, allowed_break_classes: set[str] | None
) -> bool:
    """Return True if all intervening segments between two AF runs are known non-AF and NONE are allowed breaks."""
    if allowed_break_classes is None:
        return False
    sub = rec_intervals[(rec_intervals["end_sec"] > t_start + 1e-6) & (rec_intervals["start_sec"] < t_end - 1e-6)]
    if sub.empty or (sub["rhythm"] == "unknown").any():
        return False
    nonaf = sub[sub["rhythm"] == "nonAF"]
    if nonaf.empty:
        return False
    return not bool(nonaf["rhythm_class"].isin(allowed_break_classes).any())


def _harmonize_record_g0(
    rec_intervals: pd.DataFrame,
    beat_times: np.ndarray | None,
    record_dur_sec: float,
    fs: float,
    annotation_merge_beats: int,
    min_run_beats: int,
    allowed_break_classes: set[str] | None = None,
) -> tuple[list[dict], dict[str, int]]:
    af_rows = rec_intervals[rec_intervals["rhythm"] == "AF"].sort_values("start_sec")
    stats = {
        "n_canon_af_intervals": int(len(af_rows)),
        "n_merged_lt5_gap_beats": 0,
        "n_merged_non_sinus_gap": 0,
        "n_dropped_lt5_af_beats": 0,
        "n_retained_g0": 0,
    }
    if af_rows.empty:
        return [], stats

    raw_runs = [
        {
            "onset_sec": float(r.start_sec),
            "offset_sec": float(r.end_sec),
            "af_only_dur_sec": float(r.end_sec - r.start_sec),
        }
        for r in af_rows.itertuples(index=False)
    ]

    # Step 1: Merge adjacent AF runs separated by known nonAF gap with < annotation_merge_beats
    # (only when exact beat annotations exist; never fabricate synthetic beats),
    # or when allowed_break_classes is specified and no confirmed-sinus interval lies between them.
    merged = [raw_runs[0].copy()]
    for nxt in raw_runs[1:]:
        prev = merged[-1]
        gap_start, gap_end = prev["offset_sec"], nxt["onset_sec"]
        if not _has_unknown_between(rec_intervals, gap_start, gap_end):
            gap_beats = _count_beats_in_interval(beat_times, gap_start, gap_end)
            if gap_beats is not None and gap_beats < annotation_merge_beats:
                prev["offset_sec"] = nxt["offset_sec"]
                prev["af_only_dur_sec"] += nxt["af_only_dur_sec"]
                stats["n_merged_lt5_gap_beats"] += 1
                continue
            if _between_all_disallowed_nonaf(rec_intervals, gap_start, gap_end, allowed_break_classes):
                prev["offset_sec"] = nxt["offset_sec"]
                prev["af_only_dur_sec"] += nxt["af_only_dur_sec"]
                stats["n_merged_non_sinus_gap"] += 1
                continue
        merged.append(nxt.copy())

    # Step 2: Filter out runs with < min_run_beats when exact beat annotations exist.
    # For IRIDIA (no beat annotations), preserve exact interval with n_beats = None.
    kept = []
    for r in merged:
        n_b = _count_beats_in_interval(beat_times, r["onset_sec"], r["offset_sec"])
        if n_b is not None and n_b < min_run_beats:
            stats["n_dropped_lt5_af_beats"] += 1
            continue
        r["n_beats"] = int(n_b) if n_b is not None else None
        r["beat_source"] = "wfdb_annotation" if n_b is not None else "unavailable_iridia"
        r["dur_sec"] = float(r["offset_sec"] - r["onset_sec"])
        kept.append(r)

    stats["n_retained_g0"] = int(len(kept))
    return kept, stats


def _finalize_runs_table(
    runs_by_rec: dict[tuple[str, str], list[dict]],
    rec_meta: dict[tuple[str, str], dict],
    intervals_by_rec: dict[tuple[str, str], pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for (cohort, record), runs in sorted(runs_by_rec.items()):
        meta = rec_meta[(cohort, record)]
        rec_dur = meta["record_dur_sec"]
        fs = meta["fs"]
        patient = meta["patient"]
        rec_itv = intervals_by_rec[(cohort, record)]
        has_detail = "rhythm_class" in rec_itv.columns and "rhythm_native" in rec_itv.columns
        eps = 1.0 / fs + 1e-6

        for idx, r in enumerate(runs):
            onset = r["onset_sec"]
            offset = r["offset_sec"]
            prev_gap = float(onset - runs[idx - 1]["offset_sec"]) if idx > 0 else np.nan
            next_gap = float(runs[idx + 1]["onset_sec"] - offset) if idx + 1 < len(runs) else np.nan

            # Boundary censored start: touches start of record or only unknown before onset
            bc_start = bool(onset <= eps)
            before = rec_itv[rec_itv["end_sec"] <= onset + 1e-6]
            if not bc_start and not before.empty and (before["rhythm"] == "unknown").all():
                bc_start = True

            # Boundary censored end: touches end of record or only unknown after offset
            bc_end = bool((rec_dur - offset) <= eps)
            after = rec_itv[rec_itv["start_sec"] >= offset - 1e-6]
            if not bc_end and not after.empty and (after["rhythm"] == "unknown").all():
                bc_end = True

            # Adjacent non-AF / unknown interval labels for rhythm-definition stratification
            prev_class = "RECORD_START"
            prev_native = "RECORD_START"
            if not bc_start and not before.empty:
                prev_non_af = before[before["rhythm"] != "AF"]
                if not prev_non_af.empty:
                    p_row = prev_non_af.iloc[-1]
                    prev_class = str(p_row["rhythm_class"]) if has_detail else str(p_row["rhythm"])
                    prev_native = str(p_row["rhythm_native"]) if has_detail else str(p_row["rhythm"])

            next_class = "BOUNDARY_END"
            next_native = "BOUNDARY_END"
            if not bc_end and not after.empty:
                next_non_af = after[after["rhythm"] != "AF"]
                if not next_non_af.empty:
                    n_row = next_non_af.iloc[0]
                    next_class = str(n_row["rhythm_class"]) if has_detail else str(n_row["rhythm"])
                    next_native = str(n_row["rhythm_native"]) if has_detail else str(n_row["rhythm"])

            term_sinus = next_class in {"CONFIRMED_SINUS_N", "SINUS_BRADYCARDIA"}
            term_strict_n = next_class == "CONFIRMED_SINUS_N"

            rows.append(
                {
                    "record": record,
                    "patient": patient,
                    "cohort": cohort,
                    "run_id": f"{cohort.lower()}_{record}_run_{idx:04d}",
                    "onset_sec": onset,
                    "offset_sec": offset,
                    "dur_sec": float(offset - onset),
                    "af_only_dur_sec": float(r["af_only_dur_sec"]),
                    "n_beats": r["n_beats"],
                    "beat_source": r.get("beat_source", "wfdb_annotation"),
                    "prev_gap_sec": prev_gap,
                    "next_gap_sec": next_gap,
                    "boundary_censored_start": bc_start,
                    "boundary_censored_end": bc_end,
                    "prev_rhythm_class": prev_class,
                    "prev_rhythm_native": prev_native,
                    "next_rhythm_class": next_class,
                    "next_rhythm_native": next_native,
                    "terminates_to_confirmed_sinus": bool(term_sinus),
                    "terminates_to_strict_n": bool(term_strict_n),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty and "n_beats" in out.columns:
        out["n_beats"] = pd.array(out["n_beats"].tolist(), dtype="Int64")
    return out


def _merge_gap_threshold(
    runs_g0_by_rec: dict[tuple[str, str], list[dict]],
    intervals_by_rec: dict[tuple[str, str], pd.DataFrame],
    gap_sec: float,
) -> dict[tuple[str, str], list[dict]]:
    if gap_sec <= 0:
        return runs_g0_by_rec
    out: dict[tuple[str, str], list[dict]] = {}
    for key, runs in runs_g0_by_rec.items():
        if not runs:
            out[key] = []
            continue
        rec_itv = intervals_by_rec[key]
        merged = [runs[0].copy()]
        for nxt in runs[1:]:
            prev = merged[-1]
            gap = nxt["onset_sec"] - prev["offset_sec"]
            if gap <= gap_sec and not _has_unknown_between(rec_itv, prev["offset_sec"], nxt["onset_sec"]):
                prev["offset_sec"] = nxt["offset_sec"]
                prev["dur_sec"] = prev["offset_sec"] - prev["onset_sec"]
                prev["af_only_dur_sec"] += nxt["af_only_dur_sec"]
                if prev["n_beats"] is not None and nxt["n_beats"] is not None:
                    prev["n_beats"] = int(prev["n_beats"]) + int(nxt["n_beats"])
                else:
                    prev["n_beats"] = None
            else:
                merged.append(nxt.copy())
        out[key] = merged
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2: Harmonise AF runs")
    parser.add_argument("--run-dir", type=str, default=None, help="Run directory (defaults to LATEST_RUN.txt)")
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.run_dir)
    stage1_dir = run_dir / "stage1"
    stage2_dir = run_dir / "stage2"
    stage2_dir.mkdir(parents=True, exist_ok=True)

    defs = load_definitions()
    cohorts_cfg = load_cohorts()
    defs_hash = compute_definitions_hash()
    git_hash = get_git_hash()
    run_ts = utc_timestamp_iso()

    annotation_merge_beats = int(defs["annotation_merge_beats"])
    min_run_beats = int(defs["min_run_beats"])
    gap_family = [int(g) for g in defs["gap_merge_family_sec"]]

    intervals_df = pd.read_parquet(stage1_dir / "rhythm_intervals.parquet")
    beat_summary = pd.read_parquet(stage1_dir / "beat_summary.parquet")

    rec_meta = {}
    for r in beat_summary.itertuples(index=False):
        rec_meta[(r.cohort, str(r.record))] = {
            "patient": str(r.patient),
            "fs": float(r.fs),
            "record_dur_sec": float(r.record_dur_sec),
        }

    intervals_by_rec = {
        (c, str(r)): grp.sort_values("start_sec").reset_index(drop=True)
        for (c, r), grp in intervals_df.groupby(["cohort", "record"])
    }

    runs_g0_by_rec: dict[tuple[str, str], list[dict]] = {}
    runs_cs_by_rec: dict[tuple[str, str], list[dict]] = {}
    runs_sn_by_rec: dict[tuple[str, str], list[dict]] = {}
    recon_rows: list[dict] = []

    beat_cache: dict[tuple[str, str], np.ndarray | None] = {}
    for (cohort, record), rec_itv in sorted(intervals_by_rec.items()):
        meta = rec_meta[(cohort, record)]
        beat_times = _load_beat_times_sec(cohort, record, meta["fs"], cohorts_cfg)
        beat_cache[(cohort, record)] = beat_times

        kept_g0, st_g0 = _harmonize_record_g0(
            rec_itv,
            beat_times,
            record_dur_sec=meta["record_dur_sec"],
            fs=meta["fs"],
            annotation_merge_beats=annotation_merge_beats,
            min_run_beats=min_run_beats,
            allowed_break_classes=None,
        )
        runs_g0_by_rec[(cohort, record)] = kept_g0
        recon_rows.append({"tier": "harmonized_annotation", "cohort": cohort, "record": record, **st_g0})

        # Rhythm-continuity sensitivity tiers (LTAFDB & SHDB only; IRIDIA remains binary AF vs NON_AF):
        # Tier 2: continuity assumed across non-sinus-labelled interruptions (VT/B/IVR/T/AB/SVTA/AFL/AT/PAT)
        cs_allowed = {"CONFIRMED_SINUS_N", "SINUS_BRADYCARDIA"} if cohort in {"LTAFDB", "SHDB"} else None
        kept_cs, st_cs = _harmonize_record_g0(
            rec_itv,
            beat_times,
            record_dur_sec=meta["record_dur_sec"],
            fs=meta["fs"],
            annotation_merge_beats=annotation_merge_beats,
            min_run_beats=min_run_beats,
            allowed_break_classes=cs_allowed,
        )
        runs_cs_by_rec[(cohort, record)] = kept_cs
        recon_rows.append({"tier": "continuity_nonsinus_gaps", "cohort": cohort, "record": record, **st_cs})

        # Tier 3: continuity assumed except across normal-rhythm-labelled interruptions (N only)
        sn_allowed = {"CONFIRMED_SINUS_N"} if cohort in {"LTAFDB", "SHDB"} else None
        kept_sn, st_sn = _harmonize_record_g0(
            rec_itv,
            beat_times,
            record_dur_sec=meta["record_dur_sec"],
            fs=meta["fs"],
            annotation_merge_beats=annotation_merge_beats,
            min_run_beats=min_run_beats,
            allowed_break_classes=sn_allowed,
        )
        runs_sn_by_rec[(cohort, record)] = kept_sn
        recon_rows.append({"tier": "continuity_except_normal", "cohort": cohort, "record": record, **st_sn})

    recon_detail_df = pd.DataFrame(recon_rows)
    recon_summary = (
        recon_detail_df.groupby(["tier", "cohort"], as_index=False)[
            ["n_canon_af_intervals", "n_merged_lt5_gap_beats", "n_merged_non_sinus_gap", "n_dropped_lt5_af_beats", "n_retained_g0"]
        ]
        .sum()
    )
    recon_summary["check_balance_diff"] = (
        recon_summary["n_canon_af_intervals"]
        - recon_summary["n_merged_lt5_gap_beats"]
        - recon_summary["n_merged_non_sinus_gap"]
        - recon_summary["n_dropped_lt5_af_beats"]
        - recon_summary["n_retained_g0"]
    )
    assert (recon_summary["check_balance_diff"] == 0).all(), f"Reconciliation mismatch:\n{recon_summary}"
    recon_summary = stamp_dataframe(recon_summary, git_hash=git_hash, definitions_hash=defs_hash, run_timestamp=run_ts)
    recon_summary.to_csv(stage2_dir / "reconciliation_v5_to_g0.csv", index=False)

    # Save rhythm-tier g=0 tables
    for tier_name, tier_runs in [
        ("continuity_nonsinus_gaps", runs_cs_by_rec),
        ("continuity_except_normal", runs_sn_by_rec),
    ]:
        df_t = _finalize_runs_table(tier_runs, rec_meta, intervals_by_rec)
        df_t = stamp_dataframe(df_t, git_hash=git_hash, definitions_hash=defs_hash, run_timestamp=run_ts)
        df_t.to_parquet(stage2_dir / f"runs_g0_{tier_name}.parquet", index=False)

    summary_rows = []
    prev_count = None
    g0_af_hours = None

    for g in gap_family:
        runs_g_by_rec = _merge_gap_threshold(runs_g0_by_rec, intervals_by_rec, float(g))
        df_g = _finalize_runs_table(runs_g_by_rec, rec_meta, intervals_by_rec)
        df_g = stamp_dataframe(df_g, git_hash=git_hash, definitions_hash=defs_hash, run_timestamp=run_ts)
        out_path = stage2_dir / f"runs_g{g}.parquet"
        df_g.to_parquet(out_path, index=False)

        n_runs = int(len(df_g))
        af_only_h = float(df_g["af_only_dur_sec"].sum() / 3600.0)
        span_h = float(df_g["dur_sec"].sum() / 3600.0)
        if prev_count is not None:
            assert n_runs <= prev_count, f"Non-monotone run count: g={g} has {n_runs} > prev {prev_count}"
        prev_count = n_runs
        if g0_af_hours is None:
            g0_af_hours = af_only_h
        rel_af_diff_pct = abs(af_only_h - g0_af_hours) / g0_af_hours * 100.0
        assert rel_af_diff_pct <= 0.5, f"AF burden drift {rel_af_diff_pct:.4f}% > 0.5% at g={g}"

        summary_rows.append(
            {
                "gap_merge_sec": g,
                "n_runs": n_runs,
                "n_iridia": int((df_g["cohort"] == "IRIDIA").sum()),
                "n_shdb": int((df_g["cohort"] == "SHDB").sum()),
                "n_ltafdb": int((df_g["cohort"] == "LTAFDB").sum()),
                "af_only_hours": af_only_h,
                "merged_span_hours": span_h,
                "af_burden_rel_diff_pct": rel_af_diff_pct,
                "path": str(out_path),
            }
        )

    write_json(
        stage2_dir / "stage2_summary.json",
        {
            "annotation_merge_beats": annotation_merge_beats,
            "min_run_beats": min_run_beats,
            "gap_family": summary_rows,
            "git_hash": git_hash,
            "definitions_hash": defs_hash,
            "run_timestamp": run_ts,
        },
    )

    print("[Stage 2] OK | Gap-merge family:")
    for row in summary_rows:
        print(
            f"  g={row['gap_merge_sec']:4d}s: runs={row['n_runs']:4d} "
            f"(IRIDIA={row['n_iridia']}, SHDB={row['n_shdb']}, LTAFDB={row['n_ltafdb']}) | "
            f"af_only={row['af_only_hours']:.2f}h, span={row['merged_span_hours']:.2f}h"
        )


if __name__ == "__main__":
    main()
