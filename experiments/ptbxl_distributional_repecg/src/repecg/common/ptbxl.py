from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from .config import ProgramConfig
from .labels import encode_superdiagnostic, load_superdiagnostic_map


CANONICAL_LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
INDEPENDENT_LEADS = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")
WFDB_LEAD_ALIASES = {"AVR": "aVR", "AVL": "aVL", "AVF": "aVF"}


@dataclass(frozen=True)
class SignalRecord:
    ecg_id: int
    patient_id: int
    fold: int
    signal_mv: np.ndarray
    labels: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class PTBXLStore:
    def __init__(self, config: ProgramConfig):
        self.config = config
        data = config.raw["data"]
        self.metadata_path = config.data_root / data["metadata"]
        self.statements_path = config.data_root / data["statements"]
        self.metadata = pd.read_csv(self.metadata_path, index_col="ecg_id")
        self.label_map = load_superdiagnostic_map(self.statements_path)
        self._validate_metadata()

    def _validate_metadata(self) -> None:
        required = {"patient_id", "strat_fold", "filename_hr", "scp_codes"}
        missing = required - set(self.metadata.columns)
        if missing:
            raise ValueError(f"missing PTB-XL metadata columns: {sorted(missing)}")
        if self.metadata.index.has_duplicates:
            raise ValueError("duplicate ecg_id values")
        if self.metadata["patient_id"].isna().any():
            raise ValueError("missing patient_id values")
        observed = set(self.metadata["strat_fold"].astype(int).unique())
        if observed != set(range(1, 11)):
            raise ValueError(f"unexpected strat_fold values: {sorted(observed)}")
        groups = {
            "development_train": set(range(1, 8)),
            "development_select": {8},
            "final_select": {9},
            "locked_test": {10},
        }
        patients = {
            name: set(self.metadata.loc[self.metadata.strat_fold.isin(folds), "patient_id"])
            for name, folds in groups.items()
        }
        names = tuple(patients)
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                overlap = patients[left] & patients[right]
                if overlap:
                    raise ValueError(f"patient leakage between {left} and {right}: {len(overlap)}")

    def split_frame(self, name: str) -> pd.DataFrame:
        folds = self.config.raw["splits"][name]
        return self.metadata[self.metadata.strat_fold.isin(folds)].copy()

    def cohort_manifest(self) -> dict[str, object]:
        splits = {}
        for name, folds in self.config.raw["splits"].items():
            frame = self.metadata[self.metadata.strat_fold.isin(folds)]
            splits[name] = {
                "folds": list(folds),
                "records": int(len(frame)),
                "patients": int(frame.patient_id.nunique()),
            }
        return {
            "metadata_sha256": _sha256(self.metadata_path),
            "statements_sha256": _sha256(self.statements_path),
            "records": int(len(self.metadata)),
            "patients": int(self.metadata.patient_id.nunique()),
            "splits": splits,
        }

    def _test_unlocked(self, freeze_manifest: str | Path | None) -> bool:
        if freeze_manifest is None:
            return False
        path = Path(freeze_manifest)
        if not path.is_file():
            return False
        payload = json.loads(path.read_text())
        return payload.get("fold10_unlocked") is True and payload.get("config_sha256")

    def read_record(
        self,
        ecg_id: int,
        *,
        freeze_manifest: str | Path | None = None,
    ) -> SignalRecord:
        row = self.metadata.loc[ecg_id]
        fold = int(row.strat_fold)
        if fold == 10 and not self._test_unlocked(freeze_manifest):
            raise PermissionError("fold 10 signal access requires a valid freeze manifest")
        base = self.config.data_root / str(row.filename_hr)
        signal, fields = wfdb.rdsamp(str(base))
        names = tuple(WFDB_LEAD_ALIASES.get(name, name) for name in fields["sig_name"])
        if set(names) != set(CANONICAL_LEADS):
            raise ValueError(f"unexpected lead names for ecg_id={ecg_id}: {names}")
        order = [names.index(name) for name in CANONICAL_LEADS]
        signal = np.asarray(signal[:, order], dtype=np.float64)
        if int(fields["fs"]) != int(self.config.raw["data"]["sampling_hz"]):
            raise ValueError(f"unexpected sampling frequency for ecg_id={ecg_id}")
        if not np.isfinite(signal).all():
            raise ValueError(f"non-finite samples for ecg_id={ecg_id}")
        labels = encode_superdiagnostic(
            row.scp_codes,
            self.label_map,
            self.config.classes,
        )
        return SignalRecord(
            ecg_id=int(ecg_id),
            patient_id=int(row.patient_id),
            fold=fold,
            signal_mv=signal,
            labels=labels,
        )
