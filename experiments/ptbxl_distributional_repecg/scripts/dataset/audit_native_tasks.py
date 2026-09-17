#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from repecg.common.config import load_config
from repecg.common.labels import encode_superdiagnostic, load_superdiagnostic_map
from repecg.evaluation.native_labels import (
    ECHONEXT_LABELS,
    LUDB_DIAGNOSTIC_COLUMNS,
    load_echonext_labels,
    load_emory_diagnoses,
    load_isp_intervals,
    load_ludb_labels,
    load_sunnybrook_statement_codes,
    load_zhejiang_labels,
)
from repecg.evaluation.tasks import load_dataset_tasks


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ptbxl(root: Path, config_path: Path) -> dict[str, object]:
    config = load_config(config_path)
    metadata_path = root / "ptb_xl/ptbxl_database.csv"
    statements_path = root / "ptb_xl/scp_statements.csv"
    metadata = pd.read_csv(metadata_path)
    mapping = load_superdiagnostic_map(statements_path)
    labels = np.stack([
        encode_superdiagnostic(value, mapping, config.classes) for value in metadata["scp_codes"]
    ])
    split_folds = config.raw["splits"]
    splits = {}
    for split, folds in split_folds.items():
        selected = metadata["strat_fold"].isin(folds).to_numpy()
        splits[split] = {
            "records": int(selected.sum()),
            "patients": int(metadata.loc[selected, "patient_id"].nunique()),
            "positive_records": dict(zip(config.classes, labels[selected].sum(axis=0).astype(int).tolist(), strict=True)),
        }
    return {
        "task_ids": ["ptbxl_superdiagnostic"],
        "records": len(metadata),
        "patients": int(metadata["patient_id"].nunique()),
        "splits": splits,
        "metadata_sha256": _sha256(metadata_path),
        "statements_sha256": _sha256(statements_path),
    }


def _echonext(root: Path) -> dict[str, object]:
    metadata_path = root / "echonext/echonext_metadata_100k.csv"
    table = pd.read_csv(metadata_path, usecols=["split"])
    splits = {}
    for split in table["split"].drop_duplicates():
        labels = load_echonext_labels(metadata_path, str(split))
        splits[str(split)] = {
            "records": len(labels),
            "patients": int(labels["patient_key"].nunique()),
            "positive_records": {label: int(labels[label].sum()) for label in ECHONEXT_LABELS},
            "waveforms_available": split == "test",
        }
    return {
        "task_ids": ["echonext_shd_12"],
        "records": int(sum(item["records"] for item in splits.values())),
        "splits": splits,
        "metadata_sha256": _sha256(metadata_path),
    }


def _kingston(root: Path) -> dict[str, object]:
    path = root / "kingston-icu-af-dataset-1.0.0/metadata.csv"
    table = pd.read_csv(path)
    splits = {}
    for split, frame in table.groupby("TrainOrTest"):
        counts = frame["Rhythm"].replace({"AFIB/AFLT": "AFIB_AFLT"}).value_counts()
        splits[str(split)] = {"records": len(frame), "label_counts": counts.astype(int).to_dict()}
    return {
        "task_ids": ["kingston_rhythm"],
        "records": len(table),
        "splits": splits,
        "patient_id_available": False,
        "metadata_sha256": _sha256(path),
    }


def _ludb(root: Path) -> dict[str, object]:
    path = root / "ludb/ludb.csv"
    labels = load_ludb_labels(path)
    return {
        "task_ids": ["ludb_rhythm", "ludb_diagnostic_categories", "ludb_wave_delineation"],
        "records": len(labels),
        "rhythm_counts": labels["rhythm"].value_counts().astype(int).to_dict(),
        "diagnostic_nonempty_records": {
            column: int(labels[column].map(len).gt(0).sum()) for column in LUDB_DIAGNOSTIC_COLUMNS
        },
        "delineation_source": "WFDB physician annotations per lead",
        "metadata_sha256": _sha256(path),
    }


def _rdb(root: Path) -> dict[str, object]:
    path = root / "rdb_wavelet_delineation_cache/manifest.json"
    manifest = json.loads(path.read_text())
    return {
        "task_ids": ["rdb_canonical_rhythm", "rdb_wave_delineation"],
        "records": int(manifest["conversion"]["records"]),
        "splits": manifest["split"]["counts"],
        "rhythm_counts": manifest["split"]["rhythm_counts"],
        "segmentation_class_pixels": manifest["class_counts"],
        "valid_boundaries": int(manifest["conversion"]["valid_boundaries"]),
        "manifest_sha256": _sha256(path),
    }


def _isp(root: Path) -> dict[str, object]:
    result = {"task_ids": ["isp_wave_delineation", "isp_sex", "isp_age"], "splits": {}}
    for split in ("train", "test"):
        path = root / f"isp_delineation_dataset/{split}_isp_delineation_data.csv"
        table = load_isp_intervals(path)
        counts = Counter(interval[0] for intervals in table["intervals"] for interval in intervals)
        result["splits"][split] = {
            "records": len(table),
            "interval_counts": {"P": counts[0], "QRS": counts[1], "T": counts[2]},
            "sex_counts": table["sex"].value_counts().sort_index().astype(int).to_dict(),
            "age_years": {
                "minimum": int(table["age"].min()),
                "maximum": int(table["age"].max()),
                "median": float(table["age"].median()),
            },
            "source_sha256": _sha256(path),
        }
    result["records"] = sum(item["records"] for item in result["splits"].values())
    return result


def _zhejiang(root: Path) -> dict[str, object]:
    directory = root / "zhejiang/label"
    counts = np.zeros(4, dtype=np.int64)
    files = sorted(directory.glob("*.pkl"))
    record_ids = {path.stem for path in files}
    diagnosis_path = root / "zhejiang/Diagnosis.xlsx"
    diagnoses = load_zhejiang_labels(diagnosis_path, expected_record_ids=record_ids)
    for path in files:
        with path.open("rb") as handle:
            values = np.asarray(pickle.load(handle), dtype=np.int64)
        if not set(np.unique(values)).issubset({0, 1, 2, 3}):
            raise ValueError(f"invalid Zhejiang mask labels: {path}")
        counts += np.bincount(values, minlength=4)
    return {
        "task_ids": ["zhejiang_otva_origin", "zhejiang_arrhythmia_type", "zhejiang_wave_delineation"],
        "records": len(files),
        "origin_counts": diagnoses["otva_origin"].value_counts().astype(int).to_dict(),
        "arrhythmia_type_counts": diagnoses["arrhythmia_type"].value_counts().astype(int).to_dict(),
        "sublocation_nonempty_records": int(diagnoses["sublocation"].notna().sum()),
        "sex_counts": diagnoses["sex"].value_counts().astype(int).to_dict(),
        "pixel_counts": dict(zip(("ISO", "P", "QRS", "T"), counts.tolist(), strict=True)),
        "diagnosis_sha256": _sha256(diagnosis_path),
    }


def _emory() -> dict[str, object]:
    diagnoses_path = Path("/data/mithunmanivannan/heedb_emory/12SL_diagnoses/diagnoses_v24.csv")
    metadata_path = Path("/data/mithunmanivannan/heedb_emory/metadata/metadata.csv")
    diagnoses = load_emory_diagnoses(diagnoses_path)
    metadata = pd.read_csv(metadata_path, usecols=["FileName", "BDSPPatientID"])
    if metadata["FileName"].duplicated().any():
        raise ValueError("Emory metadata has duplicate FileName values")
    matched = diagnoses["FileName"].isin(set(metadata["FileName"]))
    if not matched.all():
        raise ValueError(f"{int((~matched).sum())} Emory diagnosis rows do not match metadata")
    code_counts = Counter(code for codes in diagnoses.loc[diagnoses["eligible"], "codes"] for code in codes)
    return {
        "task_ids": ["emory_12sl_diagnoses"],
        "diagnosis_rows": len(diagnoses),
        "eligible_records": int(diagnoses["eligible"].sum()),
        "ineligible_records": int((~diagnoses["eligible"]).sum()),
        "metadata_rows": len(metadata),
        "patients": int(metadata["BDSPPatientID"].nunique()),
        "observed_codes": len(code_counts),
        "code_positive_records": {str(code): count for code, count in sorted(code_counts.items())},
        "diagnoses_sha256": _sha256(diagnoses_path),
        "metadata_sha256": _sha256(metadata_path),
    }


def _sunnybrook(root: Path) -> dict[str, object]:
    files = sorted((root / "sunnybrook_12_lead_ecg_samples").glob("*.xml"))
    counts = Counter(code for path in files for code in set(load_sunnybrook_statement_codes(path)))
    return {
        "task_ids": ["sunnybrook_statement_codes"],
        "records": len(files),
        "observed_codes": len(counts),
        "positive_records": dict(sorted(counts.items())),
        "evaluation_role": "external_only_no_fitting",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registry = load_dataset_tasks(args.task_registry)
    expected_tasks = {
        task["task_id"] for specification in registry.values() for task in specification["tasks"]
    }
    datasets = {
        "ptbxl": _ptbxl(args.data_root, args.config),
        "echonext": _echonext(args.data_root),
        "kingston_icu": _kingston(args.data_root),
        "ludb": _ludb(args.data_root),
        "rdb": _rdb(args.data_root),
        "isp": _isp(args.data_root),
        "zhejiang": _zhejiang(args.data_root),
        "emory_muse": _emory(),
        "sunnybrook": _sunnybrook(args.data_root),
    }
    observed_tasks = {task for dataset in datasets.values() for task in dataset["task_ids"]}
    if observed_tasks != expected_tasks:
        raise ValueError(
            f"native task reconciliation mismatch; missing={sorted(expected_tasks - observed_tasks)}, "
            f"extra={sorted(observed_tasks - expected_tasks)}"
        )
    payload = {
        "kind": "native_task_reconciliation",
        "status": "complete",
        "task_registry_sha256": _sha256(args.task_registry),
        "task_ids": sorted(observed_tasks),
        "datasets": datasets,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "datasets": len(datasets), "tasks": len(observed_tasks)}))


if __name__ == "__main__":
    main()
