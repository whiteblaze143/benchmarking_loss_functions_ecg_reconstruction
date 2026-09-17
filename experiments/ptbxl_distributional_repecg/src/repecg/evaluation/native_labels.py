from __future__ import annotations

import ast
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd


ECHONEXT_LABELS = (
    "lvef_lte_45",
    "lvwt_gte_13",
    "aortic_stenosis_moderate_or_greater",
    "aortic_regurgitation_moderate_or_greater",
    "mitral_regurgitation_moderate_or_greater",
    "tricuspid_regurgitation_moderate_or_greater",
    "pulmonary_regurgitation_moderate_or_greater",
    "rv_systolic_dysfunction_moderate_or_greater",
    "pericardial_effusion_moderate_large",
    "pasp_gte_45",
    "tr_max_gte_32",
    "shd_moderate_or_greater",
)

LUDB_DIAGNOSTIC_COLUMNS = (
    "Electric axis of the heart",
    "Conduction abnormalities",
    "Extrasystolies",
    "Hypertrophies",
    "Cardiac pacing",
    "Ischemia",
    "Non-specific repolarization abnormalities",
    "Other states",
)


def load_echonext_labels(
    metadata_path: Path,
    split: str,
    expected_records: int | None = None,
) -> pd.DataFrame:
    table = pd.read_csv(metadata_path)
    source_columns = [f"{label}_flag" for label in ECHONEXT_LABELS]
    required = {"ecg_key", "patient_key", "split", *source_columns}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"EchoNext metadata is missing columns: {sorted(missing)}")
    selected = table.loc[table["split"] == split, ["ecg_key", "patient_key", *source_columns]].copy()
    selected = selected.rename(columns=dict(zip(source_columns, ECHONEXT_LABELS, strict=True)))
    if expected_records is not None and len(selected) != expected_records:
        raise ValueError(
            f"EchoNext {split} label/waveform count mismatch: "
            f"{len(selected)} labels versus {expected_records} waveforms"
        )
    labels = selected.loc[:, ECHONEXT_LABELS]
    if labels.isna().any().any() or not labels.isin([0, 1]).all().all():
        raise ValueError(f"EchoNext {split} labels are not complete binary targets")
    selected.insert(0, "record_index", range(len(selected)))
    return selected.reset_index(drop=True)


def _terms(value: object) -> tuple[str, ...]:
    if pd.isna(value):
        return ()
    return tuple(term.strip() for term in str(value).splitlines() if term.strip())


def load_ludb_labels(metadata_path: Path) -> pd.DataFrame:
    table = pd.read_csv(metadata_path)
    required = {"ID", "Rhythms", *LUDB_DIAGNOSTIC_COLUMNS}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"LUDB metadata is missing columns: {sorted(missing)}")
    if table["ID"].duplicated().any() or table["Rhythms"].isna().any():
        raise ValueError("LUDB requires unique IDs and a rhythm label for every record")
    result = pd.DataFrame({"record_id": table["ID"].astype(str), "rhythm": table["Rhythms"].str.strip()})
    for source in LUDB_DIAGNOSTIC_COLUMNS:
        result[source] = table[source].map(_terms)
    return result


def load_isp_intervals(csv_path: Path) -> pd.DataFrame:
    table = pd.read_csv(csv_path)
    required = {"file_name", "age", "sex", "target"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"ISP table is missing columns: {sorted(missing)}")
    result = table.loc[:, ["file_name", "age", "sex", "target"]].copy()
    if result[["file_name", "age", "sex"]].isna().any().any():
        raise ValueError("ISP requires complete record ID, age, and sex labels")
    if not result["sex"].isin([0, 1]).all():
        raise ValueError("ISP sex labels must use the released binary 0/1 coding")
    if not result["age"].between(0, 120).all():
        raise ValueError("ISP age labels must be plausible years in [0, 120]")
    result["intervals"] = result["target"].map(ast.literal_eval)
    valid_codes = {0, 1, 2}
    for record_id, intervals in zip(result["file_name"], result["intervals"], strict=True):
        if any(len(interval) != 3 or interval[0] not in valid_codes for interval in intervals):
            raise ValueError(f"ISP record {record_id} has an invalid delineation target")
    return result.drop(columns="target")


def load_zhejiang_labels(diagnosis_path: Path, expected_record_ids: set[str] | None = None) -> pd.DataFrame:
    table = pd.read_excel(diagnosis_path, dtype={"HospitalID": str})
    required = {"HospitalID", "Type", "LeftRight", "Sublocation", "Gender"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Zhejiang diagnosis table is missing columns: {sorted(missing)}")
    if table["HospitalID"].isna().any() or table["HospitalID"].duplicated().any():
        raise ValueError("Zhejiang diagnoses require one unique HospitalID per record")
    if not table["Type"].isin(["PVC", "VT"]).all():
        raise ValueError("Zhejiang Type must contain only PVC or VT")
    if not table["LeftRight"].isin(["Right", "Left"]).all():
        raise ValueError("Zhejiang LeftRight must contain only Right or Left")
    if not table["Gender"].isin(["female", "male"]).all():
        raise ValueError("Zhejiang Gender must contain only female or male")
    table = table.rename(
        columns={
            "HospitalID": "record_id",
            "Type": "arrhythmia_type",
            "LeftRight": "otva_origin",
            "Sublocation": "sublocation",
            "Gender": "sex",
        }
    )
    table["otva_origin"] = table["otva_origin"].map({"Right": "RVOT", "Left": "LVOT"})
    if expected_record_ids is not None:
        observed = set(table["record_id"])
        if observed != expected_record_ids:
            raise ValueError(
                "Zhejiang diagnosis/mask ID mismatch: "
                f"missing={sorted(expected_record_ids - observed)}, "
                f"extra={sorted(observed - expected_record_ids)}"
            )
    return table.loc[:, ["record_id", "otva_origin", "arrhythmia_type", "sublocation", "sex"]]


def load_emory_diagnoses(diagnoses_path: Path) -> pd.DataFrame:
    table = pd.read_csv(diagnoses_path, dtype={"FileName": str, "codes": str})
    required = {"FileName", "codes"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Emory diagnosis table is missing columns: {sorted(missing)}")
    if table["FileName"].duplicated().any() or table["FileName"].isna().any():
        raise ValueError("Emory diagnoses require one unique nonempty FileName per row")
    table["codes"] = table["codes"].fillna("").map(
        lambda value: tuple(int(code.strip()) for code in value.split(",") if code.strip())
    )
    table["eligible"] = table["codes"].map(len).gt(0)
    table["exclusion_reason"] = table["eligible"].map(
        {True: "", False: "no_12sl_diagnosis_codes"}
    )
    return table


def load_sunnybrook_statement_codes(xml_path: Path) -> tuple[str, ...]:
    root = ET.parse(xml_path).getroot()
    codes = tuple(
        element.text.strip()
        for element in root.iter()
        if element.tag.rsplit("}", maxsplit=1)[-1].lower() == "statementcode"
        and element.text
        and element.text.strip()
    )
    if not codes:
        raise ValueError(f"Sunnybrook record has no statement codes: {xml_path}")
    return codes
