"""Dataset-Specific Adapters for Empirical repSpat CAHC Benchmarking.

Provides specialized adapters for four distinct real-world ECG/VCG datasets:
1. PTBXLAdapter:
   - Real 12-lead clinical ECG at 500 Hz (21,837 records).
   - Stratified diagnostic classes: NORM, MI, STTC, CD, HYP.
   - 10-fold CV partitions.
2. EchoNextAdapter:
   - Real 12-lead ECG paired with echocardiography at 250 Hz.
   - Standardized physical voltage conversion (uV -> mV).
   - Characterizes cross-acquisition sampling and voltage scaling.
3. LUDBAdapter:
   - Lobachevsky University Database (200 real 12-lead patient ECGs at 500 Hz).
   - Gold-standard physician delineations for P, QRS, T wave boundaries.
   - Provides true empirical post-CAHC selection null calibration (recurring waves across beats).
4. RDBAdapter:
   - Clinical RDB cache (2,398 patient recordings at 500 Hz).
   - Wavelet-derived wave segmentations (P=1, QRS=2, T=3, Iso=0) across clinical rhythms.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import glob
import json
import os
from typing import Any
import numpy as np
import pandas as pd
import torch
import wfdb

from .config import RepSpatConfig
from .vcg_transform import ecg_to_vcg_kors


class BaseDatasetAdapter(ABC):
    """Abstract base class for dataset-specific CAHC adaptation."""

    def __init__(self, name: str, sampling_rate: float) -> None:
        self.name = name
        self.sampling_rate = float(sampling_rate)

    @abstractmethod
    def load_record(self, record_id: Any, n_samples: int = 1500) -> dict[str, Any]:
        """Loads physical 12-lead ECG, converts to 3D VCG, and returns metadata."""
        pass

    @abstractmethod
    def list_records(self, max_records: int = 16) -> list[Any]:
        """Returns a list of valid record identifiers."""
        pass

    def get_cahc_config(
        self,
        m_ms_grid: list[float] | None = None,
        G_grid: list[int] | None = None,
        gamma: float = 1.0,
        block_mode: str = "attribute_kmeans",
        n_permutations: int = 200,
    ) -> RepSpatConfig:
        """Constructs a dataset-tailored RepSpat configuration."""
        if m_ms_grid is None:
            m_ms_grid = [32.0, 64.0, 128.0]
        if G_grid is None:
            G_grid = [4, 6, 8]

        # Convert millisecond neighborhood scale to integer sample count
        m_samples = [
            max(2, int(round(ms * self.sampling_rate / 1000.0)))
            for ms in m_ms_grid
        ]
        m_samples = sorted(list(set(m_samples)))

        cfg = RepSpatConfig.fast_test_config(
            m_grid=m_samples,
            G_grid=G_grid,
            block_permutation_mode=block_mode,
            kernel_scale_rule="median_heuristic",
            kernel_param=gamma,
            n_permutations=n_permutations,
            kmeans_n_init=10,
        )
        cfg.m_ms_grid = m_ms_grid
        return cfg


class PTBXLAdapter(BaseDatasetAdapter):
    """Adapter for PTB-XL multi-lead clinical ECG database."""

    def __init__(self, data_dir: str = "data/ptb_xl") -> None:
        super().__init__(name="PTB-XL", sampling_rate=500.0)
        self.data_dir = data_dir
        self.meta_path = os.path.join(data_dir, "ptbxl_database.csv")
        self._meta_df = None

    @property
    def meta_df(self) -> pd.DataFrame:
        if self._meta_df is None:
            self._meta_df = pd.read_csv(self.meta_path, index_col="ecg_id")
        return self._meta_df

    def list_records(
        self,
        max_records: int = 20,
        fold: int | None = 10,
        diagnostic_class: str | None = None,
    ) -> list[int]:
        df = self.meta_df
        if fold is not None:
            df = df[df["strat_fold"] == fold]
        if diagnostic_class is not None:
            # Match superclass in scp_codes
            df = df[df["scp_codes"].str.contains(diagnostic_class, na=False)]
        return list(df.index.values[:max_records])

    def load_record(self, record_id: int, n_samples: int = 1500) -> dict[str, Any]:
        row = self.meta_df.loc[record_id]
        rel_path = row["filename_hr"]
        full_path = os.path.join(self.data_dir, rel_path)
        signals, fields = wfdb.rdsamp(full_path)

        ecg = signals[:n_samples]
        vcg = ecg_to_vcg_kors(ecg)
        fs = float(fields["fs"])
        time_vec = np.arange(len(ecg)) / fs

        return {
            "record_id": str(record_id),
            "dataset": self.name,
            "ecg": ecg,
            "vcg": vcg,
            "fs": fs,
            "time": time_vec,
            "strat_fold": int(row.get("strat_fold", 0)),
            "age": float(row.get("age", 0.0)) if pd.notna(row.get("age")) else np.nan,
            "sex": str(row.get("sex", "unknown")),
            "scp_codes": str(row.get("scp_codes", "")),
            "segmentation": None,  # PTB-XL does not have beat wave segmentations
        }


class EchoNextAdapter(BaseDatasetAdapter):
    """Adapter for EchoNext external test cohort (250 Hz)."""

    def __init__(self, data_dir: str = "data/echonext") -> None:
        super().__init__(name="EchoNext", sampling_rate=250.0)
        self.data_dir = data_dir
        self.waveforms_path = os.path.join(data_dir, "EchoNext_test_waveforms.npy")
        self.prov_path = os.path.join(data_dir, "PROVENANCE.json")
        self._cache = {}

    def _load_cache(self):
        if "waveforms" not in self._cache:
            with open(self.prov_path) as f:
                prov = json.load(f)
            self._cache["mean"] = np.array(prov["normalization"]["mean"], dtype=float)
            self._cache["std"] = np.array(prov["normalization"]["std"], dtype=float)
            self._cache["waveforms"] = np.load(self.waveforms_path, mmap_mode="r")

    def list_records(self, max_records: int = 16) -> list[int]:
        self._load_cache()
        # Spread indices across the test set
        n_total = len(self._cache["waveforms"])
        step = max(1, n_total // max_records)
        return list(range(0, n_total, step))[:max_records]

    def load_record(self, record_id: int, n_samples: int = 1000) -> dict[str, Any]:
        self._load_cache()
        raw = self._cache["waveforms"][record_id, 0]  # [2500, 12]
        mean_val = self._cache["mean"]
        std_val = self._cache["std"]

        # Invert z-score to uV, then to physical mV
        wave_uv = np.asarray(raw[:n_samples], dtype=float) * std_val + mean_val
        wave_mv = wave_uv / 1000.0
        vcg = ecg_to_vcg_kors(wave_mv)
        time_vec = np.arange(len(wave_mv)) / self.sampling_rate

        return {
            "record_id": str(record_id),
            "dataset": self.name,
            "ecg": wave_mv,
            "vcg": vcg,
            "fs": self.sampling_rate,
            "time": time_vec,
            "segmentation": None,
        }


class LUDBAdapter(BaseDatasetAdapter):
    """Adapter for Lobachevsky University Database with expert wave delineations."""

    def __init__(self, data_dir: str = "data/ludb") -> None:
        super().__init__(name="LUDB", sampling_rate=500.0)
        self.data_dir = data_dir

    def list_records(self, max_records: int = 20) -> list[int]:
        hea_files = sorted(glob.glob(os.path.join(self.data_dir, "*.hea")))
        records = []
        for h in hea_files:
            bname = os.path.splitext(os.path.basename(h))[0]
            if bname.isdigit():
                records.append(int(bname))
        return records[:max_records]

    def load_record(
        self,
        record_id: int,
        n_samples: int = 2000,
        annot_lead: str = "ii",
    ) -> dict[str, Any]:
        rec_path = os.path.join(self.data_dir, str(record_id))
        rec = wfdb.rdrecord(rec_path)
        ecg = rec.p_signal[:n_samples]
        fs = float(rec.fs)
        vcg = ecg_to_vcg_kors(ecg)
        time_vec = np.arange(len(ecg)) / fs

        # Parse wave delineations and beat boundaries
        seg = np.zeros(len(ecg), dtype=int)
        beat_ids = np.zeros(len(ecg), dtype=int)

        try:
            ann = wfdb.rdann(rec_path, annot_lead)
            samples = ann.sample
            symbols = ann.symbol

            current_beat = 0
            i = 0
            while i < len(symbols):
                if symbols[i] == "(":
                    onset = samples[i]
                    if i + 1 < len(symbols):
                        peak_type = symbols[i + 1]
                        if i + 2 < len(symbols) and symbols[i + 2] == ")":
                            offset = samples[i + 2]
                            if onset < n_samples:
                                off_clamped = min(offset, n_samples - 1)
                                if peak_type.lower() == "p":
                                    code = 1
                                elif peak_type.upper() in ["N", "V", "A"]:
                                    code = 2
                                    current_beat += 1
                                elif peak_type.lower() == "t":
                                    code = 3
                                else:
                                    code = 0
                                seg[onset : off_clamped + 1] = code
                                beat_ids[onset : off_clamped + 1] = current_beat
                            i += 3
                            continue
                i += 1
        except Exception:
            pass

        return {
            "record_id": str(record_id),
            "dataset": self.name,
            "ecg": ecg,
            "vcg": vcg,
            "fs": fs,
            "time": time_vec,
            "segmentation": seg,
            "beat_ids": beat_ids,
        }


class RDBAdapter(BaseDatasetAdapter):
    """Adapter for Clinical RDB Wavelet Delineation Cache."""

    def __init__(self, cache_root: str = "data/rdb_wavelet_delineation_cache") -> None:
        super().__init__(name="RDB", sampling_rate=500.0)
        self.cache_root = cache_root

    def list_records(self, max_records: int = 20) -> list[str]:
        files = sorted(glob.glob(os.path.join(self.cache_root, "**", "*.pt"), recursive=True))
        return files[:max_records]

    def load_record(self, record_path: str, n_samples: int = 2000) -> dict[str, Any]:
        data = torch.load(record_path, map_location="cpu", weights_only=False)
        waveform = data["waveform"].numpy()
        segmentation = data["segmentation"].numpy()

        if waveform.shape[0] == 12:
            ecg = waveform.T[:n_samples]
        else:
            ecg = waveform[:n_samples]

        # Use Lead II segmentation as primary reference
        if segmentation.ndim > 1:
            seg = segmentation[1][:n_samples]
        else:
            seg = segmentation[:n_samples]

        vcg = ecg_to_vcg_kors(ecg)
        time_vec = np.arange(len(ecg)) / self.sampling_rate

        clean_id = os.path.basename(record_path).replace(".pt", "")

        return {
            "record_id": clean_id,
            "dataset": self.name,
            "ecg": ecg,
            "vcg": vcg,
            "fs": self.sampling_rate,
            "time": time_vec,
            "segmentation": seg.astype(int),
            "canonical_rhythm": str(data.get("canonical_rhythm", "unknown")),
        }


class ISPAdapter(BaseDatasetAdapter):
    """Adapter for ISP Delineation Dataset (1000 Hz)."""

    def __init__(self, data_dir: str = "data/isp_delineation_dataset") -> None:
        super().__init__(name="ISP", sampling_rate=1000.0)
        self.data_dir = data_dir
        self.csv_path = os.path.join(data_dir, "test_isp_delineation_data.csv")
        self._df = None

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            self._df = pd.read_csv(self.csv_path)
        return self._df

    def list_records(self, max_records: int = 16) -> list[int]:
        return list(self.df["file_name"].values[:max_records])

    def load_record(self, record_id: int, n_samples: int = 3000) -> dict[str, Any]:
        rec_path = os.path.join(self.data_dir, "test_data", str(record_id))
        rec = wfdb.rdrecord(rec_path)
        ecg = rec.p_signal[:n_samples]
        fs = float(rec.fs)
        vcg = ecg_to_vcg_kors(ecg)
        time_vec = np.arange(len(ecg)) / fs

        # Parse target intervals
        seg = np.zeros(len(ecg), dtype=int)
        beat_ids = np.zeros(len(ecg), dtype=int)

        row = self.df[self.df["file_name"] == record_id].iloc[0]
        import ast

        try:
            intervals = ast.literal_eval(row["target"])
            current_beat = 0
            for w_type, start, end in intervals:
                if start < n_samples:
                    end_clamped = min(end, n_samples - 1)
                    # 0=P -> 1, 1=QRS -> 2, 2=T -> 3
                    if w_type == 0:
                        code = 1
                    elif w_type == 1:
                        code = 2
                        current_beat += 1
                    elif w_type == 2:
                        code = 3
                    else:
                        code = 0
                    seg[start : end_clamped + 1] = code
                    beat_ids[start : end_clamped + 1] = current_beat
        except Exception:
            pass

        return {
            "record_id": str(record_id),
            "dataset": self.name,
            "ecg": ecg,
            "vcg": vcg,
            "fs": fs,
            "time": time_vec,
            "segmentation": seg,
            "beat_ids": beat_ids,
        }


class KingstonICUAdapter(BaseDatasetAdapter):
    """Adapter for Kingston ICU AF Bedside Monitoring Dataset (240 Hz, 4 leads)."""

    def __init__(self, data_dir: str = "data/kingston-icu-af-dataset-1.0.0") -> None:
        super().__init__(name="Kingston-ICU", sampling_rate=240.0)
        self.data_dir = data_dir
        self.meta_path = os.path.join(data_dir, "metadata.csv")
        self._meta = None

    @property
    def meta(self) -> pd.DataFrame:
        if self._meta is None:
            self._meta = pd.read_csv(self.meta_path)
        return self._meta

    def list_records(self, max_records: int = 16) -> list[str]:
        return list(self.meta["ECG"].values[:max_records])

    def load_record(self, record_rel: str, n_samples: int = 1200) -> dict[str, Any]:
        rec_path = os.path.join(self.data_dir, record_rel)
        rec = wfdb.rdrecord(rec_path)
        ecg = rec.p_signal[:n_samples]
        fs = float(rec.fs)

        # 4 leads: I, II, III, V
        # Geometric 3D dipole approximation:
        vx = ecg[:, 0]
        vy = (ecg[:, 1] + ecg[:, 2]) / 2.0
        vz = -ecg[:, 3]
        vcg = np.column_stack([vx, vy, vz])
        time_vec = np.arange(len(ecg)) / fs

        clean_id = os.path.basename(record_rel)

        return {
            "record_id": clean_id,
            "dataset": self.name,
            "ecg": ecg,
            "vcg": vcg,
            "fs": fs,
            "time": time_vec,
            "segmentation": None,
        }


class EmoryAdapter(BaseDatasetAdapter):
    """Adapter for Emory Healthcare Clinical GE MUSE WFDB Database (500 Hz)."""

    def __init__(self, data_dir: str = "/data/mithunmanivannan/heedb_emory/WFDB") -> None:
        super().__init__(name="Emory-MUSE", sampling_rate=500.0)
        self.data_dir = data_dir

    def list_records(self, max_records: int = 16) -> list[str]:
        if not os.path.isdir(self.data_dir):
            return []
        subdirs = sorted([d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))])
        records = []
        for s in subdirs:
            s_path = os.path.join(self.data_dir, s)
            for f in os.listdir(s_path):
                if f.endswith(".hea"):
                    records.append(os.path.join(s_path, os.path.splitext(f)[0]))
                    if len(records) >= max_records:
                        return records
        return records[:max_records]

    def load_record(self, record_prefix: str, n_samples: int = 1500) -> dict[str, Any]:
        signals, fields = wfdb.rdsamp(record_prefix)
        ecg = signals[:n_samples]
        fs = float(fields["fs"])
        vcg = ecg_to_vcg_kors(ecg)
        time_vec = np.arange(len(ecg)) / fs
        clean_id = os.path.basename(record_prefix)

        return {
            "record_id": clean_id,
            "dataset": self.name,
            "ecg": ecg,
            "vcg": vcg,
            "fs": fs,
            "time": time_vec,
            "segmentation": None,
        }


class SunnybrookAdapter(BaseDatasetAdapter):
    """Adapter for Sunnybrook Health Sciences Centre 12-lead ECGs (Philips Sierra XML, 500 Hz)."""

    def __init__(self, data_dir: str = "data/sunnybrook_12_lead_ecg_samples") -> None:
        super().__init__(name="Sunnybrook", sampling_rate=500.0)
        self.data_dir = data_dir

    def list_records(self, max_records: int = 20) -> list[str]:
        xmls = sorted(glob.glob(os.path.join(self.data_dir, "*.xml")))
        return xmls[:max_records]

    def load_record(self, xml_path: str, n_samples: int = 1500) -> dict[str, Any]:
        import sierraecg

        lead_order = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        f = sierraecg.read_file(xml_path)
        s_map = {l.label: l.samples for l in f.leads}
        sig = np.stack([s_map[l] for l in lead_order]).astype(np.float32)  # [12, N]
        ecg = sig.T[:n_samples] / 1000.0  # uV to mV
        vcg = ecg_to_vcg_kors(ecg)
        fs = self.sampling_rate
        time_vec = np.arange(len(ecg)) / fs
        clean_id = os.path.splitext(os.path.basename(xml_path))[0]

        return {
            "record_id": clean_id,
            "dataset": self.name,
            "ecg": ecg,
            "vcg": vcg,
            "fs": fs,
            "time": time_vec,
            "segmentation": None,
        }


class ZhejiangAdapter(BaseDatasetAdapter):
    """Adapter for Zhejiang Hospital Multi-Lead ECG cohort with wave delineations (500 Hz)."""

    def __init__(self, data_dir: str = "data/zhejiang") -> None:
        super().__init__(name="Zhejiang", sampling_rate=500.0)
        self.data_dir = data_dir
        self.ecg_dir = os.path.join(data_dir, "ecg")
        self.label_dir = os.path.join(data_dir, "label")

    def list_records(self, max_records: int = 20) -> list[str]:
        lbl_files = sorted(glob.glob(os.path.join(self.label_dir, "*.pkl")))
        rec_ids = [os.path.splitext(os.path.basename(f))[0] for f in lbl_files]
        return rec_ids[:max_records]

    def load_record(self, record_id: str, n_samples: int = 2000) -> dict[str, Any]:
        import pickle

        lead_order = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        signals = []
        for l in lead_order:
            fpath = os.path.join(self.ecg_dir, f"{record_id}_{l}.pkl")
            with open(fpath, "rb") as f:
                sig = pickle.load(f)
            signals.append(np.asarray(sig, dtype=np.float32))
        raw_ecg = np.column_stack(signals)[:n_samples]
        ecg = raw_ecg / 1000.0  # standardize uV/ADC scale to physical mV
        vcg = ecg_to_vcg_kors(ecg)
        fs = self.sampling_rate
        time_vec = np.arange(len(ecg)) / fs

        # Load wave label: 0=iso, 1=P, 2=QRS, 3=T
        lbl_path = os.path.join(self.label_dir, f"{record_id}.pkl")
        seg = np.zeros(len(ecg), dtype=int)
        if os.path.exists(lbl_path):
            with open(lbl_path, "rb") as f:
                raw_lbl = pickle.load(f)
            seg = np.asarray(raw_lbl[:n_samples], dtype=int)

        return {
            "record_id": str(record_id),
            "dataset": self.name,
            "ecg": ecg,
            "vcg": vcg,
            "fs": fs,
            "time": time_vec,
            "segmentation": seg,
        }

