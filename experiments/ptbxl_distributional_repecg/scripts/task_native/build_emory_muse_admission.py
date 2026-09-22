#!/usr/bin/env python3
"""Create the frozen, patient-disjoint Emory-MUSE task-native admission table.

This is an admission artifact, not a training script.  It deliberately fails
when an admitted diagnosis row lacks patient linkage or any diagnosis-eligible
row lacks its exact WFDB header and signal pair. Rows with blank source patient
IDs are frozen as explicit exclusions. The diagnosis vocabulary is learned
from training patients only.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import tempfile

import pandas as pd

from repecg.evaluation.native_labels import load_emory_diagnoses


SPLIT_SALT = "benchmark2_emory_muse_patient_split_v1"
SPLIT_THRESHOLDS = ((80, "train"), (90, "validation"), (100, "test"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def patient_split(patient_id: str) -> str:
    """Stable 80/10/10 patient split with an explicitly versioned salt."""
    bucket = int.from_bytes(
        hashlib.sha256(f"{SPLIT_SALT}\0{patient_id}".encode("utf-8")).digest()[:8], "big"
    ) % 100
    for threshold, name in SPLIT_THRESHOLDS:
        if bucket < threshold:
            return name
    raise AssertionError("split thresholds must cover all hash buckets")


def _verify_waveform_pairs(root: Path, filenames: pd.Series) -> None:
    by_directory: dict[Path, set[str]] = defaultdict(set)
    for relative in filenames:
        path = Path(str(relative))
        if len(path.parts) != 3 or path.parts[0] != "WFDB":
            raise ValueError(f"invalid Emory WFDB-relative path: {relative}")
        by_directory[path.parent].add(path.name)

    missing: list[str] = []
    for relative_directory, expected_stems in sorted(by_directory.items(), key=lambda item: str(item[0])):
        directory = root / relative_directory
        if not directory.is_dir():
            raise FileNotFoundError(f"missing Emory WFDB directory: {directory}")
        with os.scandir(directory) as entries:
            present = {entry.name for entry in entries}
        for stem in sorted(expected_stems):
            if f"{stem}.hea" not in present or f"{stem}.dat" not in present:
                missing.append(str(relative_directory / stem))
                if len(missing) == 20:
                    break
        print(json.dumps({
            "event": "waveform_directory_verified",
            "directory": str(relative_directory),
            "expected_records": len(expected_stems),
        }), flush=True)
        if missing:
            break
    if missing:
        raise FileNotFoundError(
            "eligible Emory diagnosis rows are missing exact WFDB header/signal pairs; "
            f"first_missing={missing}"
        )


def build_admission(root: Path, verify_waveforms: bool) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    diagnosis_path = root / "12SL_diagnoses/diagnoses_v24.csv"
    metadata_path = root / "metadata/metadata.csv"
    diagnoses = load_emory_diagnoses(diagnosis_path)
    metadata = pd.read_csv(metadata_path, usecols=["FileName", "BDSPPatientID"], dtype="string")
    if metadata["FileName"].duplicated().any():
        raise ValueError("Emory metadata has duplicate FileName rows")
    table = diagnoses.merge(metadata, on="FileName", how="left", validate="one_to_one", indicator=True)
    if not table["_merge"].eq("both").all():
        raise ValueError(f"{int((table['_merge'] != 'both').sum())} diagnosis rows lack patient linkage")
    table = table.drop(columns="_merge")
    table["exclusion_reason"] = ""
    table.loc[~table["eligible"], "exclusion_reason"] = "no_12sl_diagnosis_codes"
    missing_patient = table["BDSPPatientID"].isna() | table["BDSPPatientID"].fillna("").str.strip().eq("")
    table.loc[table["eligible"] & missing_patient, "exclusion_reason"] = "missing_patient_id"
    diagnosis_eligible = table.loc[table["eligible"]].copy()
    if diagnosis_eligible.empty:
        raise ValueError("no eligible Emory diagnosis rows")
    if not diagnosis_eligible["FileName"].str.startswith("WFDB/").all():
        raise ValueError("Emory FileName values must be WFDB-relative paths")
    if verify_waveforms:
        _verify_waveform_pairs(root, diagnosis_eligible["FileName"])
    eligible = table.loc[table["exclusion_reason"].eq("")].copy()
    eligible["patient_id"] = eligible["BDSPPatientID"].astype(str)
    eligible["split"] = eligible["patient_id"].map(patient_split)
    if eligible.groupby("patient_id")["split"].nunique().gt(1).any():
        raise RuntimeError("patient-disjoint split construction failed")
    train = eligible.loc[eligible["split"] == "train"]
    vocabulary = Counter(code for codes in train["codes"] for code in codes)
    if not vocabulary:
        raise ValueError("training split has no diagnosis vocabulary")
    eligible["codes"] = eligible["codes"].map(lambda values: ",".join(map(str, values)))
    admitted = eligible.loc[:, ["FileName", "patient_id", "split", "codes"]].rename(columns={"FileName": "wfdb_path"})
    excluded = table.loc[table["exclusion_reason"].ne(""), ["FileName", "exclusion_reason"]].rename(columns={"FileName": "wfdb_path"})
    summary = {
        "schema_version": "emory_muse_admission_v1",
        "source": {
            "root": str(root),
            "diagnoses_sha256": sha256_file(diagnosis_path),
            "metadata_sha256": sha256_file(metadata_path),
        },
        "split": {"salt": SPLIT_SALT, "thresholds": list(SPLIT_THRESHOLDS)},
        "waveform_pairs_verified": verify_waveforms,
        "diagnosis_rows": int(len(diagnoses)),
        "diagnosis_eligible_records": int(len(diagnosis_eligible)),
        "excluded_records": int(len(excluded)),
        "exclusion_counts": excluded["exclusion_reason"].value_counts().sort_index().astype(int).to_dict(),
        "admitted_records": int(len(admitted)),
        "admitted_patients": int(admitted["patient_id"].nunique()),
        "split_records": admitted["split"].value_counts().sort_index().astype(int).to_dict(),
        "split_patients": admitted.groupby("split")["patient_id"].nunique().sort_index().astype(int).to_dict(),
        "training_vocabulary": {str(code): int(count) for code, count in sorted(vocabulary.items())},
    }
    return admitted, excluded, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-waveforms", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite populated admission directory: {args.output_dir}")
    admitted, excluded, summary = build_admission(args.source_root, args.verify_waveforms)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with tempfile.NamedTemporaryFile("w", dir=args.output_dir, suffix=".csv", delete=False) as handle:
        temporary = Path(handle.name)
        admitted.to_csv(handle, index=False)
    temporary.replace(args.output_dir / "admitted_records.csv")
    with tempfile.NamedTemporaryFile("w", dir=args.output_dir, suffix=".csv", delete=False) as handle:
        temporary = Path(handle.name)
        excluded.to_csv(handle, index=False)
    temporary.replace(args.output_dir / "excluded_records.csv")
    (args.output_dir / "manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": "complete",
        "admitted_records": summary["admitted_records"],
        "admitted_patients": summary["admitted_patients"],
        "waveform_pairs_verified": summary["waveform_pairs_verified"],
    }))


if __name__ == "__main__":
    main()
