from pathlib import Path

import numpy as np
import pytest

from repecg.evaluation.native_labels import (
    ECHONEXT_LABELS,
    load_echonext_labels,
    load_emory_diagnoses,
    load_isp_intervals,
    load_ludb_labels,
    load_sunnybrook_statement_codes,
    load_zhejiang_labels,
)


REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "data"


def test_echonext_test_labels_align_with_waveform_count() -> None:
    waveform_count = np.load(DATA / "echonext/EchoNext_test_waveforms.npy", mmap_mode="r").shape[0]
    labels = load_echonext_labels(
        DATA / "echonext/echonext_metadata_100k.csv", "test", expected_records=waveform_count
    )
    assert len(labels) == 5_442
    assert tuple(labels.columns[-12:]) == ECHONEXT_LABELS
    assert int(labels["shd_moderate_or_greater"].sum()) == 2_318


def test_echonext_count_mismatch_fails() -> None:
    with pytest.raises(ValueError, match="label/waveform count mismatch"):
        load_echonext_labels(DATA / "echonext/echonext_metadata_100k.csv", "test", expected_records=1)


def test_ludb_native_labels_are_complete() -> None:
    labels = load_ludb_labels(DATA / "ludb/ludb.csv")
    assert len(labels) == 200
    assert labels["rhythm"].ne("").all()


def test_isp_interval_labels_parse() -> None:
    labels = load_isp_intervals(DATA / "isp_delineation_dataset/test_isp_delineation_data.csv")
    assert len(labels) == 72
    assert labels["intervals"].map(len).gt(0).all()
    assert labels["age"].notna().all()
    assert set(labels["sex"]) == {0, 1}


def test_zhejiang_record_labels_align_with_delineation_masks() -> None:
    record_ids = {path.stem for path in (DATA / "zhejiang/label").glob("*.pkl")}
    labels = load_zhejiang_labels(DATA / "zhejiang/Diagnosis.xlsx", record_ids)
    assert len(labels) == 334
    assert labels["otva_origin"].value_counts().to_dict() == {"RVOT": 257, "LVOT": 77}
    assert labels["arrhythmia_type"].value_counts().to_dict() == {"PVC": 329, "VT": 5}


def test_emory_diagnosis_codes_parse() -> None:
    labels = load_emory_diagnoses(Path("/data/mithunmanivannan/heedb_emory/12SL_diagnoses/diagnoses_v24.csv"))
    assert len(labels) == 974_172
    assert int(labels["eligible"].sum()) == 968_680
    assert int((labels["exclusion_reason"] == "no_12sl_diagnosis_codes").sum()) == 5_492


def test_sunnybrook_statement_codes_parse() -> None:
    files = sorted((DATA / "sunnybrook_12_lead_ecg_samples").glob("*.xml"))
    code_sets = [load_sunnybrook_statement_codes(path) for path in files]
    assert len(code_sets) == 20
    assert len({code for codes in code_sets for code in codes}) == 29
