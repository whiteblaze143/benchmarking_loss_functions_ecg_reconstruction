"""Data loader for real clinical multi-lead ECG patient recordings.

Loads records from `data/rdb_wavelet_delineation_cache/`:
- Multi-lead physical ECG signals (12 leads, 500 Hz)
- Kors 3D VCG trajectory v(t) = [Vx, Vy, Vz]
- Expert ground-truth wave delineations (P-wave, QRS-complex, T-wave)
- Clinical rhythm classifications (SR, AF, SB, ST, etc.)
- Graceful synthetic signal generation fallback for offline unit tests.
"""
from __future__ import annotations

import glob
import os
import numpy as np
import torch

from .vcg_transform import ecg_to_vcg_kors


def load_clinical_ecg_record(
    record_path_or_id: str,
    cache_root: str = "data/rdb_wavelet_delineation_cache",
) -> dict:
    """Loads a clinical ECG recording and projects to 3D VCG.

    Parameters
    ----------
    record_path_or_id : str
        File path or ID (e.g. 'rdb_SR0001.pt' or 'SR0001').
    cache_root : str
        Directory holding RDB delineation cache files.

    Returns
    -------
    record : dict
        {
            'ecg': np.ndarray [n_samples, 12],
            'vcg': np.ndarray [n_samples, 3],
            'segmentation': np.ndarray [n_samples], # 0=iso/bg, 1=P, 2=QRS, 3=T
            'canonical_rhythm': str,
            'record_id': str,
            'fs': float,
            'time': np.ndarray [n_samples],
        }
    """
    clean_id = os.path.basename(record_path_or_id).replace(".pt", "")
    if not clean_id.startswith("rdb_") and not os.path.isfile(record_path_or_id):
        clean_id = f"rdb_{clean_id}"

    if os.path.isfile(record_path_or_id):
        pt_path = record_path_or_id
    else:
        matches = glob.glob(os.path.join(cache_root, "**", f"{clean_id}.pt"), recursive=True)
        if matches:
            pt_path = matches[0]
        else:
            raise FileNotFoundError(f"Record '{clean_id}' not found in '{cache_root}'")

    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    waveform = data["waveform"].numpy()
    segmentation = data["segmentation"].numpy()

    # Shape: [12, n_samples] -> [n_samples, 12]
    if waveform.shape[0] == 12:
        ecg_12 = waveform.T
    else:
        ecg_12 = waveform

    n_samples = len(ecg_12)
    fs = 500.0
    time = np.arange(n_samples) / fs

    # Kors 3D VCG dipole
    vcg_3d = ecg_to_vcg_kors(ecg_12)

    # Lead II segmentation as primary reference
    lead_idx = 1
    seg_1d = segmentation[lead_idx] if segmentation.ndim > 1 else segmentation

    return {
        "ecg": ecg_12,
        "vcg": vcg_3d,
        "segmentation": seg_1d,
        "canonical_rhythm": data.get("canonical_rhythm", "unknown"),
        "record_id": data.get("record_id", clean_id),
        "fs": fs,
        "time": time,
    }


def list_clinical_ecg_records(
    cache_root: str = "data/rdb_wavelet_delineation_cache",
    split: str = "test",
    max_records: int = 12,
    balance_rhythms: bool = True,
) -> list[str]:
    """Discovers clinical ECG records from RDB cache.

    Parameters
    ----------
    cache_root : str
        Cache root directory.
    split : str
        'test', 'train', or 'val'.
    max_records : int
        Maximum number of records to return.
    balance_rhythms : bool
        If True, samples across rhythm categories (SR, AF, SB, ST, etc.).

    Returns
    -------
    paths : list of str
    """
    pattern = os.path.join(cache_root, split, "*.pt")
    paths = sorted(glob.glob(pattern))
    if not paths:
        paths = sorted(glob.glob(os.path.join(cache_root, "**", "*.pt"), recursive=True))

    if not balance_rhythms or not paths:
        return paths[:max_records]

    rhythm_prefixes = ["SR", "AF", "SB", "SI", "ST", "VT", "AT"]
    grouped: dict[str, list[str]] = {p: [] for p in rhythm_prefixes}
    others: list[str] = []

    for p in paths:
        bname = os.path.basename(p).replace("rdb_", "")
        matched = False
        for prefix in rhythm_prefixes:
            if bname.startswith(prefix):
                grouped[prefix].append(p)
                matched = True
                break
        if not matched:
            others.append(p)

    selected: list[str] = []
    while len(selected) < max_records:
        added_any = False
        for prefix in rhythm_prefixes:
            if grouped[prefix]:
                selected.append(grouped[prefix].pop(0))
                added_any = True
                if len(selected) >= max_records:
                    break
        if not added_any:
            while others and len(selected) < max_records:
                selected.append(others.pop(0))
            break

    return selected


def generate_synthetic_ecg_vcg(
    n_samples: int = 1000,
    fs: float = 500.0,
    heart_rate: float = 60.0,
    random_state: int = 42,
) -> dict:
    """Generates synthetic ECG and VCG with known periodic wave morphology for tests."""
    rng = np.random.RandomState(random_state)
    time = np.arange(n_samples) / fs
    period = 60.0 / heart_rate  # seconds per beat
    phase = (time % period) / period  # 0 to 1

    # Approximate P, QRS, T waves
    seg = np.zeros(n_samples, dtype=int)
    # P wave: phase 0.1 to 0.2
    seg[(phase >= 0.1) & (phase < 0.2)] = 1
    # QRS: phase 0.25 to 0.35
    seg[(phase >= 0.25) & (phase < 0.35)] = 2
    # T wave: phase 0.45 to 0.65
    seg[(phase >= 0.45) & (phase < 0.65)] = 3

    # VCG Vx, Vy, Vz signals
    vx = np.sin(2 * np.pi * phase) + 0.1 * rng.randn(n_samples)
    vy = np.cos(2 * np.pi * phase) * 2.0 + 0.1 * rng.randn(n_samples)
    vz = np.sin(4 * np.pi * phase) * 0.5 + 0.1 * rng.randn(n_samples)

    # QRS sharp deflection
    qrs_mask = seg == 2
    vy[qrs_mask] += 3.0 * np.sin(np.pi * (phase[qrs_mask] - 0.25) / 0.1)
    vx[qrs_mask] += 1.5 * np.sin(np.pi * (phase[qrs_mask] - 0.25) / 0.1)

    vcg = np.column_stack([vx, vy, vz])
    # Dummy 12 leads
    ecg = np.tile(vy[:, None], (1, 12)) + 0.1 * rng.randn(n_samples, 12)

    return {
        "ecg": ecg,
        "vcg": vcg,
        "segmentation": seg,
        "canonical_rhythm": "Synthetic-SR",
        "record_id": "synth_001",
        "fs": fs,
        "time": time,
    }
