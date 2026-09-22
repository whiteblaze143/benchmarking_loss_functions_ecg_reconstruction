import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/task_native/build_emory_muse_admission.py"
SPEC = importlib.util.spec_from_file_location("build_emory_muse_admission", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_patient_split_is_stable_and_patient_disjoint() -> None:
    first = MODULE.patient_split("12345")
    assert first in {"train", "validation", "test"}
    assert MODULE.patient_split("12345") == first
    assert sum(MODULE.patient_split(str(index)) == "train" for index in range(10_000)) > 7_500


def test_bulk_waveform_verification_requires_exact_header_signal_pairs(tmp_path: Path) -> None:
    directory = tmp_path / "WFDB/2018"
    directory.mkdir(parents=True)
    (directory / "record.hea").write_text("header")
    (directory / "record.dat").write_bytes(b"signal")
    records = pd.Series(["WFDB/2018/record"])
    MODULE._verify_waveform_pairs(tmp_path, records)
    (directory / "record.dat").unlink()
    with pytest.raises(FileNotFoundError, match="first_missing"):
        MODULE._verify_waveform_pairs(tmp_path, records)
