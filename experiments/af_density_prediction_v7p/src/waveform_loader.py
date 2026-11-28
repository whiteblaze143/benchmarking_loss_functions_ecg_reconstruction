"""Zero-leakage, high-performance waveform loader for continuous ECG streams.

Supports:
- LTAFDB (128 Hz, PhysioNet WFDB)
- SHDB-AF (200 Hz, PhysioNet WFDB)
- IRIDIA-AF (200 Hz, 24h chunked HDF5)
- Noise-Holter (PhysioNet WFDB v1.0.0)
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import numpy as np
import wfdb
import h5py

LTAFDB_DIR = Path("/data/mithunmanivannan/datasets/ltafdb_1.0.0_reference")
SHDB_DIR = Path("/data/mithunmanivannan/datasets/shdb_af")
IRIDIA_DIR = Path("/data/mithunmanivannan/datasets/iridia_af_8405941_v1.0.1/ecg_hdf5")
NOISE_HOLTER_DIR = Path("/data/mithunmanivannan/datasets/noise-holter-database-a-multi-device-multi-arrhythmia-real-world-clinical-ecg-signal-quality-benchmark-1.0.0/wfdb_files")


class WaveformLoader:
    def __init__(self, cache_max_records: int = 64):
        self._record_cache: Dict[str, Tuple[np.ndarray, float]] = {}
        self._h5_handles: Dict[str, h5py.File] = {}
        self.cache_max_records = cache_max_records

    def load_record_full(self, dataset: str, record_id: str) -> Optional[Tuple[np.ndarray, float]]:
        """Loads and caches full continuous record in memory for ultra-fast slicing."""
        cache_key = f"{dataset}_{record_id}"
        if cache_key in self._record_cache:
            return self._record_cache[cache_key]

        sig = None
        fs = 200.0

        try:
            if dataset == "LTAFDB":
                rec_path = str(LTAFDB_DIR / str(record_id))
                rec = wfdb.rdrecord(rec_path, channels=[0])
                sig = rec.p_signal[:, 0].astype(np.float32)
                fs = float(rec.fs)
            elif dataset == "SHDB-AF":
                rec_path = str(SHDB_DIR / str(record_id))
                rec = wfdb.rdrecord(rec_path, channels=[0])
                sig = rec.p_signal[:, 0].astype(np.float32)
                fs = float(rec.fs)
            elif dataset == "Noise-Holter":
                rec_path = str(NOISE_HOLTER_DIR / str(record_id))
                rec = wfdb.rdrecord(rec_path, channels=[0])
                sig = rec.p_signal[:, 0].astype(np.float32)
                fs = float(rec.fs)
        except Exception:
            return None

        if sig is not None:
            if len(self._record_cache) >= self.cache_max_records:
                # Evict oldest entry
                oldest_k = next(iter(self._record_cache))
                del self._record_cache[oldest_k]
            self._record_cache[cache_key] = (sig, fs)
            return (sig, fs)
        return None

    def get_causal_segment(
        self,
        dataset: str,
        record_id: str,
        landmark_time_s: float,
        duration_s: float = 10.0,
    ) -> Tuple[np.ndarray, float]:
        """Extracts strictly causal ECG segment ending at landmark_time_s.
        
        Returns:
            (signal_1d, fs)
        """
        start_s = landmark_time_s - duration_s
        end_s = landmark_time_s

        if dataset in ["LTAFDB", "SHDB-AF"]:
            rec_info = self.load_record_full(dataset, record_id)
            if rec_info is None:
                raise FileNotFoundError(f"Could not load waveform for record {record_id} in {dataset}")
            sig, fs = rec_info
            start_samp = int(round(start_s * fs))
            end_samp = int(round(end_s * fs))

            target_len = int(round(duration_s * fs))
            if start_samp < 0:
                pad_len = -start_samp
                sliced = sig[0:max(0, end_samp)]
                pad = np.zeros(pad_len, dtype=np.float32)
                out = np.concatenate([pad, sliced])
            else:
                out = sig[start_samp:end_samp]

            if len(out) < target_len:
                pad = np.zeros(target_len - len(out), dtype=np.float32)
                out = np.concatenate([pad, out])
            elif len(out) > target_len:
                out = out[-target_len:]
            return out.astype(np.float32), fs

        elif dataset == "IRIDIA-AF":
            fs = 200.0
            day = int(start_s // 86400.0)
            local_start_s = max(0.0, start_s - day * 86400.0)
            local_end_s = end_s - day * 86400.0

            h5_key = f"{record_id}_day{day}"
            h5_file = IRIDIA_DIR / record_id / f"{record_id}_ecg_{day:02d}.h5"

            if h5_key not in self._h5_handles:
                if h5_file.exists():
                    self._h5_handles[h5_key] = h5py.File(h5_file, "r")
                else:
                    raise FileNotFoundError(f"Missing IRIDIA HDF5 file: {h5_file}")

            h5_ds = self._h5_handles[h5_key]["ecg"]
            start_samp = int(round(local_start_s * fs))
            end_samp = int(round(local_end_s * fs))
            target_len = int(round(duration_s * fs))

            sliced = h5_ds[start_samp:end_samp, 0].astype(np.float32)

            if len(sliced) < target_len:
                pad = np.zeros(target_len - len(sliced), dtype=np.float32)
                sliced = np.concatenate([pad, sliced])
            elif len(sliced) > target_len:
                sliced = sliced[-target_len:]
            return sliced, fs

        elif dataset == "Noise-Holter":
            rec_info = self.load_record_full("Noise-Holter", record_id)
            if rec_info is None:
                raise FileNotFoundError(f"Could not load Noise-Holter excerpt {record_id}")
            sig, fs = rec_info
            target_len = int(round(duration_s * fs))
            if len(sig) >= target_len:
                out = sig[-target_len:]
            else:
                pad = np.zeros(target_len - len(sig), dtype=np.float32)
                out = np.concatenate([pad, sig])
            return out.astype(np.float32), fs

        else:
            fs = 200.0
            return np.zeros(int(round(duration_s * fs)), dtype=np.float32), fs

    def close(self):
        for h in self._h5_handles.values():
            try:
                h.close()
            except Exception:
                pass
        self._h5_handles.clear()
        self._record_cache.clear()
