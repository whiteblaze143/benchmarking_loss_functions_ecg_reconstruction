#!/usr/bin/env python3
"""Stage 1: Build canonical rhythm-interval timelines and beat summary per annotated record."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from common import (
    REPO_ROOT,
    canonical_cohort,
    compute_definitions_hash,
    get_git_hash,
    load_cohorts,
    resolve_run_dir,
    stamp_dataframe,
    utc_timestamp_iso,
    write_json,
)

NON_BEAT_SYMBOLS = {"+", "~", "|", "s", "T", "*", "D", "=", '"', "@", "[", "]", "!", "x", "(", ")", "p", "t", "u", "`", "'", "^"}


def _map_rhythm(rhythm_known: bool, label_harm: str, label_native: str) -> tuple[str, str, str]:
    """Map v5 interval labels into binary rhythm, clinical rhythm_class, and native label."""
    nat = str(label_native).strip() if label_native is not None else "UNKNOWN"
    harm = str(label_harm).strip() if label_harm is not None else "UNKNOWN"
    if not rhythm_known:
        return "unknown", "UNKNOWN", nat
    if harm == "AF" or nat in {"AF", "AFIB"}:
        return "AF", "AF", nat
    if nat == "NON_AF":
        return "nonAF", "IRIDIA_NON_AF_UNSPECIFIED", nat
    if nat == "N" or harm == "CONFIRMED_SINUS":
        return "nonAF", "CONFIRMED_SINUS_N", nat
    if nat == "SBR":
        return "nonAF", "SINUS_BRADYCARDIA", nat
    if nat in {"AFL", "AT", "SVTA", "PAT"} or harm in {"ATRIAL_FLUTTER", "OTHER_ATRIAL_ARRHYTHMIA"}:
        return "nonAF", "ORGANIZED_ATRIAL", nat
    if nat in {"VT", "IVR", "B", "T", "AB", "NOD"}:
        return "nonAF", "VENTRICULAR_OR_ECTOPIC", nat
    if harm == "UNKNOWN":
        return "unknown", "UNKNOWN", nat
    return "nonAF", "OTHER_NON_AF_UNCLASSIFIED", nat


def _count_beats_wfdb(cohort: str, record_id: str, cohorts_cfg: dict) -> tuple[int | None, str]:
    base = Path(cohorts_cfg["cohorts"][cohort]["base_dir"])
    if cohort == "LTAFDB":
        ann = wfdb.rdann(str(base / record_id), "atr")
        n_beats = sum(1 for s in ann.symbol if s not in NON_BEAT_SYMBOLS)
        return int(n_beats), "ltafdb_atr"
    if cohort == "SHDB":
        qrs_path = base / f"{record_id}.qrs"
        ext = "qrs" if qrs_path.exists() else "atr"
        ann = wfdb.rdann(str(base / record_id), ext)
        n_beats = sum(1 for s in ann.symbol if s not in NON_BEAT_SYMBOLS)
        return int(n_beats), f"shdb_{ext}"
    return None, "none"


def _partition_record_intervals(
    rec_df: pd.DataFrame,
    raw_samples: int,
    fs: float,
    cohort: str,
    record: str,
    patient: str,
) -> tuple[list[dict], int]:
    """Resolve any overlapping intervals (e.g. IRIDIA calibration exclusions) into a strictly disjoint
    partition of [0, raw_samples] with precedence: unknown > AF > nonAF, labeling uncovered gaps unknown."""
    starts = np.clip(rec_df["start_sample"].to_numpy(dtype=np.int64), 0, raw_samples)
    ends = np.clip(rec_df["end_sample"].to_numpy(dtype=np.int64), 0, raw_samples)
    rhythms = rec_df["rhythm"].to_numpy()
    rclasses = rec_df["rhythm_class"].to_numpy()
    rnatives = rec_df["rhythm_native"].to_numpy()

    # Fast path: already disjoint and contiguous
    if len(starts) > 0 and starts[0] == 0 and ends[-1] == raw_samples and np.all(starts[1:] == ends[:-1]):
        rows = [
            {
                "record": record,
                "patient": patient,
                "cohort": cohort,
                "start_sec": float(s / fs),
                "end_sec": float(e / fs),
                "dur_sec": float((e - s) / fs),
                "start_sample": int(s),
                "end_sample": int(e),
                "fs": fs,
                "rhythm": str(r),
                "rhythm_class": str(rc),
                "rhythm_native": str(rn),
            }
            for s, e, r, rc, rn in zip(starts, ends, rhythms, rclasses, rnatives)
            if e > s
        ]
        return rows, 0

    # General path: sweep-line partition on breakpoints
    pts = np.unique(np.concatenate(([0, raw_samples], starts, ends)))
    prio = {"nonAF": 1, "AF": 2, "unknown": 3}
    inv_prio = {1: "nonAF", 2: "AF", 3: "unknown"}
    n_segs = len(pts) - 1
    seg_prio = np.zeros(n_segs, dtype=np.int8)
    seg_class = np.full(n_segs, "UNKNOWN", dtype=object)
    seg_native = np.full(n_segs, "UNCOVERED", dtype=object)

    # Apply intervals in ascending priority order so higher-priority labels overwrite atomically
    order = np.argsort([prio[str(r)] for r in rhythms], kind="mergesort")
    for k in order:
        s, e = int(starts[k]), int(ends[k])
        if e <= s:
            continue
        i0 = int(np.searchsorted(pts, s))
        i1 = int(np.searchsorted(pts, e))
        p = prio[str(rhythms[k])]
        mask_update = seg_prio[i0:i1] <= p
        sl_p = seg_prio[i0:i1]
        sl_c = seg_class[i0:i1]
        sl_n = seg_native[i0:i1]
        sl_p[mask_update] = p
        sl_c[mask_update] = str(rclasses[k])
        sl_n[mask_update] = str(rnatives[k])

    # Uncovered segments MUST become unknown (3), NEVER AF (2)
    uncov_mask = seg_prio == 0
    n_uncovered = int(np.sum(uncov_mask))
    seg_prio[uncov_mask] = 3
    seg_class[uncov_mask] = "UNKNOWN"
    seg_native[uncov_mask] = "UNCOVERED"

    # Merge adjacent segments with identical (rhythm, rhythm_class, rhythm_native)
    rows = []
    cur_s = int(pts[0])
    cur_key = (int(seg_prio[0]), str(seg_class[0]), str(seg_native[0]))
    for i in range(1, len(seg_prio)):
        nxt_key = (int(seg_prio[i]), str(seg_class[i]), str(seg_native[i]))
        if nxt_key != cur_key:
            cur_e = int(pts[i])
            if cur_e > cur_s:
                rows.append(
                    {
                        "record": record,
                        "patient": patient,
                        "cohort": cohort,
                        "start_sec": float(cur_s / fs),
                        "end_sec": float(cur_e / fs),
                        "dur_sec": float((cur_e - cur_s) / fs),
                        "start_sample": cur_s,
                        "end_sample": cur_e,
                        "fs": fs,
                        "rhythm": inv_prio[cur_key[0]],
                        "rhythm_class": cur_key[1],
                        "rhythm_native": cur_key[2],
                    }
                )
            cur_s = int(pts[i])
            cur_key = nxt_key
    cur_e = int(pts[-1])
    if cur_e > cur_s:
        rows.append(
            {
                "record": record,
                "patient": patient,
                "cohort": cohort,
                "start_sec": float(cur_s / fs),
                "end_sec": float(cur_e / fs),
                "dur_sec": float((cur_e - cur_s) / fs),
                "start_sample": cur_s,
                "end_sample": cur_e,
                "fs": fs,
                "rhythm": inv_prio[cur_key[0]],
                "rhythm_class": cur_key[1],
                "rhythm_native": cur_key[2],
            }
        )
    return rows, n_uncovered


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: Build canonical timelines")
    parser.add_argument("--run-dir", type=str, default=None, help="Run directory (defaults to LATEST_RUN.txt)")
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.run_dir)
    stage0_dir = run_dir / "stage0"
    stage1_dir = run_dir / "stage1"
    stage1_dir.mkdir(parents=True, exist_ok=True)

    cohorts_cfg = load_cohorts()
    defs_hash = compute_definitions_hash()
    git_hash = get_git_hash()
    run_ts = utc_timestamp_iso()

    inv = pd.read_parquet(stage0_dir / "inventory_frozen.parquet")
    ann_inv = inv[inv["is_annotated"]].copy()

    v5_intervals_path = REPO_ROOT / cohorts_cfg["upstream_manifests"]["v5_intervals_parquet"]
    v5_i = pd.read_parquet(v5_intervals_path)
    v5_i["cohort"] = v5_i["dataset"].map(canonical_cohort)
    v5_i["record"] = v5_i["record_id"].astype(str)
    v5_i["patient"] = v5_i["patient_id"].astype(str)

    # Filter to annotated records only
    ann_keys = set(zip(ann_inv["cohort"], ann_inv["record_id"].astype(str)))
    mask = [((c, r) in ann_keys) for c, r in zip(v5_i["cohort"], v5_i["record"])]
    v5_i = v5_i.loc[mask].copy()

    mapped = [
        _map_rhythm(bool(k), str(h), str(n))
        for k, h, n in zip(v5_i["rhythm_known"], v5_i["rhythm_label_harmonized"], v5_i["rhythm_label_native"])
    ]
    v5_i["rhythm"] = [m[0] for m in mapped]
    v5_i["rhythm_class"] = [m[1] for m in mapped]
    v5_i["rhythm_native"] = [m[2] for m in mapped]
    v5_i = v5_i.sort_values(["cohort", "record", "start_sample", "end_sample"])

    # Resolve overlaps / gaps per annotated record into strictly disjoint partition of [0, raw_samples]
    grouped = {k: g for k, g in v5_i.groupby(["cohort", "record"], sort=False)}
    all_rows = []
    total_uncovered = 0
    for row in ann_inv.itertuples(index=False):
        key = (row.cohort, str(row.record_id))
        rec_df = grouped[key]
        r_rows, n_uncov = _partition_record_intervals(
            rec_df,
            raw_samples=int(row.raw_samples),
            fs=float(row.fs),
            cohort=row.cohort,
            record=str(row.record_id),
            patient=str(row.patient_id),
        )
        all_rows.extend(r_rows)
        total_uncovered += n_uncov
    intervals_df = pd.DataFrame(all_rows).sort_values(["cohort", "record", "start_sec"]).reset_index(drop=True)

    # Audit: sum of intervals == record length +/- 1 sample for every annotated record
    rec_sums = intervals_df.groupby(["cohort", "record"], as_index=False).agg(
        sum_samples=("end_sample", lambda s: int(s.sum())),
        sum_start_samples=("start_sample", lambda s: int(s.sum())),
        sum_dur_sec=("dur_sec", "sum"),
        af_dur_sec=("dur_sec", lambda s: float(s[intervals_df.loc[s.index, "rhythm"] == "AF"].sum())),
    )
    rec_sums["interval_samples"] = rec_sums["sum_samples"] - rec_sums["sum_start_samples"]
    merged = ann_inv.merge(
        rec_sums,
        left_on=["cohort", "record_id"],
        right_on=["cohort", "record"],
        how="left",
    )
    merged["sample_diff"] = (merged["interval_samples"] - merged["raw_samples"]).abs()
    max_sample_diff = int(merged["sample_diff"].max())
    assert max_sample_diff <= 1, f"Interval sum != record length: max_sample_diff={max_sample_diff}"

    total_af_h = float(intervals_df.loc[intervals_df["rhythm"] == "AF", "dur_sec"].sum() / 3600.0)
    target_af_h = 3095.2250
    rel_err_pct = abs(total_af_h - target_af_h) / target_af_h * 100.0
    assert rel_err_pct <= 0.01, f"AF hours {total_af_h:.4f}h differs from {target_af_h}h by {rel_err_pct:.4f}%"

    # Build beat summary per annotated record
    beat_rows = []
    for row in ann_inv.itertuples(index=False):
        n_beats, src = _count_beats_wfdb(row.cohort, str(row.record_id), cohorts_cfg)
        beat_rows.append(
            {
                "record": str(row.record_id),
                "patient": str(row.patient_id),
                "cohort": row.cohort,
                "fs": float(row.fs),
                "raw_samples": int(row.raw_samples),
                "record_dur_sec": float(row.raw_samples / row.fs),
                "n_beats": n_beats,
                "beat_source": src,
            }
        )
    beat_df = pd.DataFrame(beat_rows)

    intervals_df = stamp_dataframe(intervals_df, git_hash=git_hash, definitions_hash=defs_hash, run_timestamp=run_ts)
    beat_df = stamp_dataframe(beat_df, git_hash=git_hash, definitions_hash=defs_hash, run_timestamp=run_ts)

    intervals_df.to_parquet(stage1_dir / "rhythm_intervals.parquet", index=False)
    beat_df.to_parquet(stage1_dir / "beat_summary.parquet", index=False)

    audit = {
        "n_annotated_records": int(len(ann_inv)),
        "n_intervals": int(len(intervals_df)),
        "n_uncovered_segments_labeled_unknown": int(total_uncovered),
        "max_sample_diff": max_sample_diff,
        "total_af_hours": total_af_h,
        "target_af_hours": target_af_h,
        "af_hours_rel_err_pct": rel_err_pct,
        "rhythm_counts": {k: int(v) for k, v in intervals_df["rhythm"].value_counts().items()},
        "rhythm_class_counts": {k: int(v) for k, v in intervals_df["rhythm_class"].value_counts().items()},
        "git_hash": git_hash,
        "definitions_hash": defs_hash,
        "run_timestamp": run_ts,
    }
    write_json(stage1_dir / "stage1_audit.json", audit)

    print(f"[Stage 1] OK | intervals={len(intervals_df)}, max_sample_diff={max_sample_diff}")
    print(f"  total_af_hours={total_af_h:.4f}h (target={target_af_h}h, rel_err={rel_err_pct:.6f}%)")


if __name__ == "__main__":
    main()
