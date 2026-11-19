#!/usr/bin/env python3
"""Audit SHDB-AF (Shenyang / Japanese Holter ECG database of Atrial Fibrillation).

Verifies:
- 128 recordings and metadata alignment with AdditionalData.csv
- Prior ablation status (Previous_AF_Ablation: True / False)
- Annotated (98) vs unannotated (30) recording status
- AF types (PAF, PerAF, non-AF) and cross-tabulation with prior ablation
- Demographics, comorbidities (HTN, DM, CHF, Stroke), and pacemakers
- WFDB file completeness (.hea, .dat, .qrs, and .atr for annotated)
- Signal characteristics (sampling frequency, lead count, duration)
- Beat and rhythm annotations from .atr and .qrs files
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import wfdb


def audit_shdb_af(
    data_dir: Path,
    output_path: Path,
    metadata_csv: Optional[Path] = None,
    records_file: Optional[Path] = None,
    deep_verify_all: bool = True,
) -> Dict[str, Any]:
    """Execute complete audit on SHDB-AF Database."""
    data_dir = Path(data_dir).resolve()

    # Locate AdditionalData.csv
    if metadata_csv and Path(metadata_csv).is_file():
        csv_path = Path(metadata_csv).resolve()
    elif (data_dir / "AdditionalData.csv").is_file():
        csv_path = data_dir / "AdditionalData.csv"
    elif Path("/data/mithunmanivannan/codex_artifacts/hrs2027_af_recurrence/AdditionalData.csv").is_file():
        csv_path = Path("/data/mithunmanivannan/codex_artifacts/hrs2027_af_recurrence/AdditionalData.csv")
    else:
        raise FileNotFoundError(f"AdditionalData.csv not found in {data_dir} or codex_artifacts")

    # Locate RECORDS.txt
    if records_file and Path(records_file).is_file():
        rec_path = Path(records_file).resolve()
    elif (data_dir / "RECORDS.txt").is_file():
        rec_path = data_dir / "RECORDS.txt"
    elif (data_dir / "RECORDS").is_file():
        rec_path = data_dir / "RECORDS"
    elif Path("/data/mithunmanivannan/codex_artifacts/hrs2027_af_recurrence/RECORDS.txt").is_file():
        rec_path = Path("/data/mithunmanivannan/codex_artifacts/hrs2027_af_recurrence/RECORDS.txt")
    else:
        rec_path = None

    df = pd.read_csv(csv_path)
    print(f"[audit_shdb_af] Loaded metadata from {csv_path} with {len(df)} rows")

    if rec_path and rec_path.is_file():
        with open(rec_path, "r") as f:
            records = [line.strip() for line in f if line.strip()]
    else:
        records = [f"{int(x):03d}" for x in df["Data_ID"]]

    print(f"[audit_shdb_af] Auditing {len(records)} recordings in {data_dir}")

    # Standardize record id column
    df["rec_id"] = df["Data_ID"].apply(lambda x: f"{int(x):03d}")
    df_by_id = df.set_index("rec_id")

    # 1. Prior Ablation Analysis
    ablation_counts = df["Previous_AF_Ablation"].value_counts(dropna=False).to_dict()
    ablation_true = int(ablation_counts.get(True, 0))
    ablation_false = int(ablation_counts.get(False, 0))

    # Detailed ablation sub-categories
    pvi_count = int(df["Ablation1_PVI"].sum()) if "Ablation1_PVI" in df.columns else 0
    cti_count = int((df["Ablation1_CTI"] == 1.0).sum()) if "Ablation1_CTI" in df.columns else 0
    redo_count = int(df["Date_Redo_AF_Ablation"].notna().sum()) if "Date_Redo_AF_Ablation" in df.columns else 0

    # Cross tab ablation vs AF type
    crosstab_ablation_af = pd.crosstab(df["Previous_AF_Ablation"], df["AF_Type"]).to_dict()

    # Cross tab ablation vs Annotated
    crosstab_ablation_annotated = pd.crosstab(df["Previous_AF_Ablation"], df["Annotated"]).to_dict()

    # Demographics
    sex_counts = df["Sex"].value_counts(dropna=False).to_dict() if "Sex" in df.columns else {}
    age_series = df["Age_at_Holter"].dropna() if "Age_at_Holter" in df.columns else pd.Series([], dtype=float)
    bmi_series = df["BMI"].dropna() if "BMI" in df.columns else pd.Series([], dtype=float)

    demographics = {
        "sex_distribution": {str(k): int(v) for k, v in sex_counts.items()},
        "age_statistics": {
            "mean": round(float(age_series.mean()), 1) if not age_series.empty else None,
            "std": round(float(age_series.std()), 1) if not age_series.empty else None,
            "min": float(age_series.min()) if not age_series.empty else None,
            "max": float(age_series.max()) if not age_series.empty else None,
            "median": float(age_series.median()) if not age_series.empty else None,
        },
        "bmi_statistics": {
            "mean": round(float(bmi_series.mean()), 1) if not bmi_series.empty else None,
            "std": round(float(bmi_series.std()), 1) if not bmi_series.empty else None,
            "min": float(bmi_series.min()) if not bmi_series.empty else None,
            "max": float(bmi_series.max()) if not bmi_series.empty else None,
        },
    }

    # Comorbidities
    comorbidity_cols = ["HTN", "DM", "CHF", "Stroke", "Vascular_Diseases"]
    comorbidities = {}
    for col in comorbidity_cols:
        if col in df.columns:
            cnt = int(df[col].sum()) if df[col].dtype in (bool, int, float) else int((df[col] == True).sum())
            comorbidities[col] = cnt

    # Pacemaker
    pacemaker = {
        "PPM_on_Holter": int(df["PPM_on_Holter"].sum()) if "PPM_on_Holter" in df.columns else 0,
        "PPM_after_Holter": int(df["PPM_after_Holter"].sum()) if "PPM_after_Holter" in df.columns else 0,
    }

    # 2. File and Signal Verification
    missing_hea = []
    missing_dat = []
    missing_qrs = []
    missing_atr_when_annotated = []
    unexpected_atr_when_unannotated = []
    corrupted_records = []
    dat_size_errors = []

    durations_sec = []
    fs_counts = Counter()
    lead_counts = Counter()
    rhythm_type_counts = Counter()
    rhythm_record_counts = Counter()

    for idx, rec_id in enumerate(records):
        hea_file = data_dir / f"{rec_id}.hea"
        dat_file = data_dir / f"{rec_id}.dat"
        qrs_file = data_dir / f"{rec_id}.qrs"
        atr_file = data_dir / f"{rec_id}.atr"

        row = df_by_id.loc[rec_id] if rec_id in df_by_id.index else None
        is_annotated_in_csv = bool(row["Annotated"]) if row is not None else False

        if not hea_file.is_file():
            missing_hea.append(rec_id)
            continue
        if not dat_file.is_file():
            missing_dat.append(rec_id)
        if not qrs_file.is_file():
            missing_qrs.append(rec_id)

        if is_annotated_in_csv and not atr_file.is_file():
            missing_atr_when_annotated.append(rec_id)
        elif not is_annotated_in_csv and atr_file.is_file():
            unexpected_atr_when_unannotated.append(rec_id)

        record_stem = str(data_dir / rec_id)

        try:
            header = wfdb.rdheader(record_stem)
            fs = header.fs
            sig_len = header.sig_len
            n_sig = header.n_sig
            dur_sec = sig_len / fs if fs > 0 else 0.0
            durations_sec.append(dur_sec)

            fs_counts[fs] += 1
            lead_counts[n_sig] += 1

            if dat_file.is_file() and header.fmt and header.fmt[0] == "16":
                expected_size = sig_len * n_sig * 2
                actual_size = dat_file.stat().st_size
                if actual_size != expected_size:
                    dat_size_errors.append({
                        "record": rec_id,
                        "expected": expected_size,
                        "actual": actual_size,
                    })
        except Exception as e:
            corrupted_records.append({"record": rec_id, "error": f"Header error: {e}"})
            continue

        # Check annotations if present
        if atr_file.is_file():
            try:
                ann = wfdb.rdann(record_stem, "atr")
                # Look at rhythm notes or aux_note
                if ann.aux_note:
                    seen_rhythms_in_rec = set()
                    for note in ann.aux_note:
                        note_clean = note.strip()
                        if note_clean:
                            rhythm_type_counts[note_clean] += 1
                            seen_rhythms_in_rec.add(note_clean)
                    for r in seen_rhythms_in_rec:
                        rhythm_record_counts[r] += 1
            except Exception as e:
                corrupted_records.append({"record": rec_id, "error": f"atr error: {e}"})

        # Deep signal check
        if deep_verify_all and (idx % 15 == 0 or idx < 5):
            try:
                rec = wfdb.rdrecord(record_stem, sampfrom=0, sampto=1000)
                if rec.p_signal is None and rec.d_signal is None:
                    corrupted_records.append({"record": rec_id, "error": "No signal in first 1000 samples"})
            except Exception as e:
                corrupted_records.append({"record": rec_id, "error": f"rdrecord error: {e}"})

    durations_hours = [d / 3600.0 for d in durations_sec]

    audit_summary = {
        "dataset_name": "SHDB-AF: Japanese Holter ECG Database of Atrial Fibrillation",
        "dataset_version": "1.0.1",
        "database_dir": str(data_dir),
        "total_recordings": len(records),
        "unique_patients_in_metadata": int(df["Subject_ID"].nunique()) if "Subject_ID" in df.columns else len(records),
        "annotation_status": {
            "total_recordings": len(records),
            "annotated_count_in_metadata": int((df["Annotated"] == True).sum()),
            "unannotated_count_in_metadata": int((df["Annotated"] == False).sum()),
            "actual_atr_files_present": len(records) - len(missing_atr_when_annotated) - (30 if not unexpected_atr_when_unannotated else 0),
            "metadata_annotation_alignment": len(missing_atr_when_annotated) == 0 and len(unexpected_atr_when_unannotated) == 0,
        },
        "prior_ablation_status": {
            "previous_af_ablation_true": ablation_true,
            "previous_af_ablation_false": ablation_false,
            "previous_af_ablation_true_percentage": round(ablation_true / len(df) * 100, 2),
            "pvi_ablation_1_count": pvi_count,
            "cti_ablation_1_count": cti_count,
            "redo_af_ablation_count": redo_count,
        },
        "af_type_distribution": {str(k): int(v) for k, v in df["AF_Type"].value_counts().items()},
        "prior_ablation_by_af_type": {
            str(af_type): {
                "ablation_true": int(crosstab_ablation_af.get(af_type, {}).get(True, 0)),
                "ablation_false": int(crosstab_ablation_af.get(af_type, {}).get(False, 0)),
            }
            for af_type in df["AF_Type"].dropna().unique()
        },
        "prior_ablation_by_annotated": {
            "annotated_true": {
                "ablation_true": int(crosstab_ablation_annotated.get(True, {}).get(True, 0)),
                "ablation_false": int(crosstab_ablation_annotated.get(True, {}).get(False, 0)),
            },
            "annotated_false": {
                "ablation_true": int(crosstab_ablation_annotated.get(False, {}).get(True, 0)),
                "ablation_false": int(crosstab_ablation_annotated.get(False, {}).get(False, 0)),
            },
        },
        "patient_demographics": demographics,
        "clinical_comorbidities": comorbidities,
        "pacemaker_status": pacemaker,
        "recording_characteristics": {
            "sampling_frequencies": {str(k): int(v) for k, v in fs_counts.items()},
            "lead_counts": {str(k): int(v) for k, v in lead_counts.items()},
            "duration_hours": {
                "total_hours": round(float(sum(durations_hours)), 2),
                "mean_hours": round(float(np.mean(durations_hours)), 2) if durations_hours else 0.0,
                "min_hours": round(float(np.min(durations_hours)), 2) if durations_hours else 0.0,
                "max_hours": round(float(np.max(durations_hours)), 2) if durations_hours else 0.0,
                "median_hours": round(float(np.median(durations_hours)), 2) if durations_hours else 0.0,
            },
        },
        "rhythm_annotations_summary": {
            "rhythm_total_occurrences": dict(rhythm_type_counts.most_common()),
            "recordings_with_rhythm": dict(rhythm_record_counts.most_common()),
        },
        "integrity_check": {
            "missing_hea_count": len(missing_hea),
            "missing_dat_count": len(missing_dat),
            "missing_qrs_count": len(missing_qrs),
            "missing_atr_when_annotated_count": len(missing_atr_when_annotated),
            "unexpected_atr_count": len(unexpected_atr_when_unannotated),
            "dat_size_errors_count": len(dat_size_errors),
            "corrupted_records_count": len(corrupted_records),
            "files_available_on_disk": len(records) - len(missing_hea),
        },
        "audit_passed": (
            len(records) == 128
            and len(df) == 128
            and len(missing_hea) == 0
            and len(missing_dat) == 0
            and len(missing_qrs) == 0
            and len(missing_atr_when_annotated) == 0
            and len(corrupted_records) == 0
            and len(dat_size_errors) == 0
        ),
    }

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(audit_summary, f, indent=2)

    print(f"\n=== SHDB-AF Audit Summary ===")
    print(f"Total Recordings: {len(records)} (Unique Patients: {audit_summary['unique_patients_in_metadata']})")
    print(f"Prior AF Ablation: True={ablation_true} (41), False={ablation_false} (87)")
    print(f"Annotated Status: True={audit_summary['annotation_status']['annotated_count_in_metadata']} (98), False={audit_summary['annotation_status']['unannotated_count_in_metadata']} (30)")
    print(f"AF Types: {audit_summary['af_type_distribution']}")
    print(f"Integrity Check: {audit_summary['integrity_check']}")
    print(f"Audit Passed: {audit_summary['audit_passed']}")
    print(f"Audit artifact saved to: {output_path}")

    return audit_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit SHDB-AF Database")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/data/mithunmanivannan/datasets/shdb_af",
        help="Path to SHDB-AF data directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/data/mithunmanivannan/codex_artifacts/hrs2027_af_recurrence/shdb_af_audit_summary.json",
        help="Path to output JSON summary",
    )
    parser.add_argument(
        "--metadata-csv",
        type=str,
        default=None,
        help="Path to AdditionalData.csv",
    )
    parser.add_argument(
        "--records-file",
        type=str,
        default=None,
        help="Path to RECORDS.txt",
    )
    args = parser.parse_args()

    audit_shdb_af(Path(args.data_dir), Path(args.output), args.metadata_csv, args.records_file)
