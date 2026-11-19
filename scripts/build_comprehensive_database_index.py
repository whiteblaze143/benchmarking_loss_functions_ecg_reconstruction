#!/usr/bin/env python3
"""Build Comprehensive Master Indices for SHDB-AF and Noise Holter Databases.

Generates:
1. /data/mithunmanivannan/codex_artifacts/hrs2027_af_recurrence/shdb_af_master_index.csv (.json)
   - All 128 recordings (3,050 hours)
   - Clinical metadata, prior ablation status (41 True / 87 False)
   - Exact rhythm-adjudicated AF episode counts, AF duration, and AF burden (%)
   - 33 non-AF negative control Holters (792 hours of pure sinus rhythm)
   - Post-blanking recurrence eligibility (>90 days post-ablation)

2. /data/mithunmanivannan/codex_artifacts/hrs2027_af_recurrence/noise_holter_master_index.csv (.json)
   - All 6,130 physical segments on disk across 310 subjects
   - Exact Clean / N1 / N2 quality classifications
   - Multi-label arrhythmia events (379 A. Fib., 122 A. Flutter, VPB, SVPB, AVB, etc.)
   - Full signal metadata (fs=200Hz, duration, lead configuration)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import wfdb


def parse_date_safe(date_str: Any) -> Optional[datetime]:
    """Safely parse various date formats."""
    if pd.isna(date_str):
        return None
    s = str(date_str).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y", "%Y%m%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None


def build_shdb_af_index(
    data_dir: Path,
    metadata_csv: Path,
    output_dir: Path,
) -> pd.DataFrame:
    """Build master index for SHDB-AF database."""
    print(f"\n[build_shdb_af_index] Loading metadata from {metadata_csv}...")
    df = pd.read_csv(metadata_csv)
    df["rec_id"] = df["Data_ID"].apply(lambda x: f"{int(x):03d}")

    records_list = []

    for idx, row in df.iterrows():
        rec_id = row["rec_id"]
        rec_stem = str(data_dir / rec_id)
        hea_file = data_dir / f"{rec_id}.hea"
        dat_file = data_dir / f"{rec_id}.dat"
        atr_file = data_dir / f"{rec_id}.atr"
        qrs_file = data_dir / f"{rec_id}.qrs"

        header = wfdb.rdheader(rec_stem)
        fs = header.fs
        sig_len = header.sig_len
        dur_sec = sig_len / fs if fs > 0 else 0.0
        dur_hours = dur_sec / 3600.0

        # Prior ablation flags
        prev_ablation = bool(row.get("Previous_AF_Ablation", False))
        pvi_done = bool(row.get("Ablation1_PVI", 0) == 1.0)
        cti_done = bool(row.get("Ablation1_CTI", 0) == 1.0)
        has_redo = pd.notna(row.get("Date_Redo_AF_Ablation"))
        af_type = str(row.get("AF_Type", "Unknown"))
        is_annotated = bool(row.get("Annotated", False)) and atr_file.is_file()
        is_non_af_control = (af_type == "non-AF")

        # Blanking period calculation
        date_holter = parse_date_safe(row.get("Date_Holter"))
        date_ablation = parse_date_safe(row.get("Date_1st_AF_Ablation"))
        date_redo = parse_date_safe(row.get("Date_Redo_AF_Ablation"))

        days_post_1st_ablation = (date_holter - date_ablation).days if (date_holter and date_ablation) else None
        days_post_redo = (date_holter - date_redo).days if (date_holter and date_redo) else None
        
        # Most recent ablation before holter
        days_post_most_recent = days_post_redo if days_post_redo is not None else days_post_1st_ablation
        is_post_blanking = (days_post_most_recent > 90) if days_post_most_recent is not None else False

        # Rhythm analysis from .atr
        af_samples = 0
        af_episodes = 0
        current_rhythm = "(N"
        current_start = 0
        rhythm_transitions = 0
        rhythm_set = set()

        if is_annotated:
            try:
                ann = wfdb.rdann(rec_stem, "atr")
                rhythm_changes = [
                    (int(s), note.strip())
                    for s, note in zip(ann.sample, ann.aux_note)
                    if note.strip().startswith("(")
                ]
                rhythm_transitions = len(rhythm_changes)

                for s, note in rhythm_changes:
                    clean_note = note.strip()
                    rhythm_set.add(clean_note)
                    if current_rhythm in ("(AFIB", "(AFL"):
                        af_samples += (s - current_start)
                    if clean_note in ("(AFIB", "(AFL") and current_rhythm not in ("(AFIB", "(AFL"):
                        af_episodes += 1
                    current_rhythm = clean_note
                    current_start = s

                # Trailing segment
                if current_rhythm in ("(AFIB", "(AFL"):
                    af_samples += (sig_len - current_start)

            except Exception as e:
                print(f"  Warning reading atr for {rec_id}: {e}")

        af_dur_sec = af_samples / fs if fs > 0 else 0.0
        af_dur_hours = af_dur_sec / 3600.0
        af_burden_pct = (af_samples / sig_len * 100.0) if sig_len > 0 else 0.0
        has_af_episode = (af_samples > 0)

        # Recurrence definition: post-ablation Holter with adjudicated AF episode
        recurrent_af_detected = (prev_ablation and has_af_episode)

        records_list.append({
            "rec_id": rec_id,
            "subject_id": str(row.get("Subject_ID", "")),
            "data_id": int(row.get("Data_ID", 0)),
            "fs": fs,
            "sig_len": sig_len,
            "duration_seconds": round(dur_sec, 2),
            "duration_hours": round(dur_hours, 3),
            "n_leads": header.n_sig,
            "lead_names": "+".join(header.sig_name),
            "af_type": af_type,
            "is_non_af_control": is_non_af_control,
            "annotated": is_annotated,
            "previous_af_ablation": prev_ablation,
            "pvi_ablation": pvi_done,
            "cti_ablation": cti_done,
            "has_redo_ablation": has_redo,
            "redo_detail": str(row.get("Redo_Detail", "")) if pd.notna(row.get("Redo_Detail")) else "",
            "date_holter": str(row.get("Date_Holter", "")),
            "date_1st_ablation": str(row.get("Date_1st_AF_Ablation", "")),
            "date_redo_ablation": str(row.get("Date_Redo_AF_Ablation", "")),
            "days_post_ablation": days_post_most_recent,
            "is_post_blanking": is_post_blanking,
            "af_episodes_count": af_episodes,
            "af_duration_seconds": round(af_dur_sec, 2),
            "af_duration_hours": round(af_dur_hours, 3),
            "af_burden_percentage": round(af_burden_pct, 4),
            "has_adjudicated_af": has_af_episode,
            "recurrent_af_post_ablation": recurrent_af_detected,
            "rhythm_transitions_count": rhythm_transitions,
            "rhythms_observed": "+".join(sorted(list(rhythm_set))),
            "sex": str(row.get("Sex", "")),
            "age": float(row.get("Age_at_Holter", np.nan)),
            "bmi": float(row.get("BMI", np.nan)),
            "htn": bool(row.get("HTN", False)),
            "dm": bool(row.get("DM", False)),
            "chf": bool(row.get("CHF", False)),
            "stroke": bool(row.get("Stroke", False)),
            "vascular": bool(row.get("Vascular_Diseases", False)),
            "ppm_on_holter": bool(row.get("PPM_on_Holter", False)),
            "antiarrhythmic_nonbb": str(row.get("Antiarrhythmic_Drug_nonBB", "")),
            "antiarrhythmic_bb": str(row.get("Antiarrhythmic_Drug_BB", "")),
            "anticoagulation": str(row.get("Anticoagulation", "")),
            "hea_path": str(hea_file),
            "dat_path": str(dat_file),
            "atr_path": str(atr_file) if atr_file.is_file() else "",
            "qrs_path": str(qrs_file) if qrs_file.is_file() else "",
        })

    master_df = pd.DataFrame(records_list)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_out = output_dir / "shdb_af_master_index.csv"
    json_out = output_dir / "shdb_af_master_index.json"

    master_df.to_csv(csv_out, index=False)
    
    # Save JSON summary with nested metadata
    summary = {
        "dataset_name": "SHDB-AF Master Index",
        "total_recordings": len(master_df),
        "total_duration_hours": round(float(master_df["duration_hours"].sum()), 2),
        "cohort_breakdown": {
            "non_af_negative_controls": int(master_df["is_non_af_control"].sum()),
            "non_af_total_hours": round(float(master_df[master_df["is_non_af_control"]]["duration_hours"].sum()), 2),
            "paf_recordings": int((master_df["af_type"] == "PAF").sum()),
            "peraf_recordings": int((master_df["af_type"] == "PerAF").sum()),
            "prior_ablation_recordings": int(master_df["previous_af_ablation"].sum()),
            "post_blanking_prior_ablation": int((master_df["previous_af_ablation"] & master_df["is_post_blanking"]).sum()),
            "annotated_recordings": int(master_df["annotated"].sum()),
        },
        "burden_by_cohort": {
            "non_af_controls": {
                "mean_burden_pct": round(float(master_df[master_df["is_non_af_control"]]["af_burden_percentage"].mean()), 3),
                "max_burden_pct": round(float(master_df[master_df["is_non_af_control"]]["af_burden_percentage"].max()), 3),
            },
            "paf_annotated": {
                "mean_burden_pct": round(float(master_df[(master_df["af_type"] == "PAF") & master_df["annotated"]]["af_burden_percentage"].mean()), 3),
                "median_burden_pct": round(float(master_df[(master_df["af_type"] == "PAF") & master_df["annotated"]]["af_burden_percentage"].median()), 3),
            },
            "prior_ablation_annotated": {
                "mean_burden_pct": round(float(master_df[master_df["previous_af_ablation"] & master_df["annotated"]]["af_burden_percentage"].mean()), 3),
                "median_burden_pct": round(float(master_df[master_df["previous_af_ablation"] & master_df["annotated"]]["af_burden_percentage"].median()), 3),
            },
        },
        "records": records_list,
    }

    with open(json_out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[build_shdb_af_index] Indexed {len(master_df)} records ({summary['total_duration_hours']} hours).")
    print(f"  - 33 non-AF controls: {summary['cohort_breakdown']['non_af_total_hours']} hours (Mean AF Burden = {summary['burden_by_cohort']['non_af_controls']['mean_burden_pct']}%)")
    print(f"  - 41 prior-ablation Holters ({summary['cohort_breakdown']['post_blanking_prior_ablation']} post-blanking >90d)")
    print(f"  Saved master index to {csv_out} and {json_out}")
    return master_df


def build_noise_holter_index(
    data_dir: Path,
    output_dir: Path,
    records_file: Optional[Path] = None,
) -> pd.DataFrame:
    """Build master index for Noise Holter database."""
    wfdb_dir = (data_dir / "wfdb_files") if (data_dir / "wfdb_files").is_dir() else data_dir
    print(f"\n[build_noise_holter_index] Indexing files in {wfdb_dir}...")

    hea_files = sorted(list(wfdb_dir.glob("*.hea")))
    print(f"[build_noise_holter_index] Found {len(hea_files)} .hea files on disk")

    records_list = []

    for hf in hea_files:
        rec_id = hf.stem
        subject_id = rec_id.split("-")[0] if "-" in rec_id else rec_id
        dat_file = wfdb_dir / f"{rec_id}.dat"
        atr_file = wfdb_dir / f"{rec_id}.atr"
        qrs_file = wfdb_dir / f"{rec_id}.qrs"

        header = wfdb.rdheader(str(wfdb_dir / rec_id))
        fs = header.fs
        sig_len = header.sig_len
        dur_sec = sig_len / fs if fs > 0 else 0.0

        # Parse comments
        raw_comment = " ".join(header.comments)
        cls_val = "Unknown"
        noise_val = "None"
        events_list = []

        line = raw_comment.lstrip("#").strip()
        parts = [p.strip() for p in line.split("|")]
        cur_k = None
        for p in parts:
            if ":" in p:
                k, v = p.split(":", 1)
                k = k.strip().upper()
                v = v.strip()
                if k in ("CLASS", "QUALITY"):
                    cls_val = v
                    cur_k = "CLASS"
                elif k == "NOISE":
                    noise_val = v
                    cur_k = "NOISE"
                elif k in ("EVENTS", "EVENT"):
                    cur_k = "EVENTS"
                    if v and v.lower() != "none":
                        events_list.append(v)
            else:
                if cur_k == "EVENTS" and p and p.lower() != "none":
                    events_list.append(p)

        is_af = 1 if any(e in ("A. Fib.", "A. Flutter") for e in events_list) else 0

        records_list.append({
            "rec_id": rec_id,
            "subject_id": subject_id,
            "fs": fs,
            "sig_len": sig_len,
            "duration_seconds": round(dur_sec, 2),
            "n_leads": header.n_sig,
            "lead_names": "+".join(header.sig_name),
            "quality_class": cls_val,
            "is_clean": int(cls_val == "Clean"),
            "is_n1": int(cls_val == "N1"),
            "is_n2": int(cls_val == "N2"),
            "noise_type": noise_val,
            "events": "+".join(events_list) if events_list else "None",
            "is_af": is_af,
            "hea_path": str(hf),
            "dat_path": str(dat_file) if dat_file.is_file() else "",
            "atr_path": str(atr_file) if atr_file.is_file() else "",
            "qrs_path": str(qrs_file) if qrs_file.is_file() else "",
        })

    master_df = pd.DataFrame(records_list)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_out = output_dir / "noise_holter_master_index.csv"
    json_out = output_dir / "noise_holter_master_index.json"

    master_df.to_csv(csv_out, index=False)

    summary = {
        "dataset_name": "Noise Holter Database Master Index",
        "total_segments": len(master_df),
        "unique_subjects": int(master_df["subject_id"].nunique()),
        "total_duration_hours": round(float(master_df["duration_seconds"].sum() / 3600.0), 3),
        "quality_distribution": {
            "Clean": int((master_df["quality_class"] == "Clean").sum()),
            "N1": int((master_df["quality_class"] == "N1").sum()),
            "N2": int((master_df["quality_class"] == "N2").sum()),
        },
        "af_segments_count": int(master_df["is_af"].sum()),
        "noise_types": dict(Counter(master_df["noise_type"]).most_common()),
        "records": records_list,
    }

    with open(json_out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[build_noise_holter_index] Indexed {len(master_df)} segments across {summary['unique_subjects']} subjects.")
    print(f"  - Quality: Clean={summary['quality_distribution']['Clean']}, N1={summary['quality_distribution']['N1']}, N2={summary['quality_distribution']['N2']}")
    print(f"  - Arrhythmia: {summary['af_segments_count']} AF/AFL segments")
    print(f"  Saved master index to {csv_out} and {json_out}")
    return master_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Master Ingestion Indices for SHDB-AF and Noise Holter")
    parser.add_argument(
        "--shdb-dir",
        type=str,
        default="/data/mithunmanivannan/datasets/shdb_af",
        help="Path to SHDB-AF dataset",
    )
    parser.add_argument(
        "--shdb-meta",
        type=str,
        default="/data/mithunmanivannan/datasets/shdb_af/AdditionalData.csv",
        help="Path to SHDB-AF AdditionalData.csv",
    )
    parser.add_argument(
        "--noise-dir",
        type=str,
        default="/data/mithunmanivannan/datasets/noise_holter_database",
        help="Path to Noise Holter dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/data/mithunmanivannan/codex_artifacts/hrs2027_af_recurrence",
        help="Output directory for master indices",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    build_shdb_af_index(Path(args.shdb_dir), Path(args.shdb_meta), out_dir)
    build_noise_holter_index(Path(args.noise_dir), out_dir)


if __name__ == "__main__":
    main()
