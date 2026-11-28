#!/usr/bin/env python3
"""Stage 3: Empirical SQI Calibration for ENTRENCH-AF Markers.

Calibrates bSQI acceptance thresholds for each marker M in:
  {AFR_DF, fwave_SampEn, VCC, RR_SampEn, PMV, PAC_coupling, RMSSD}
using:
  1. NSTDB noise injection (em, ma, bw) at calibrated SNRs onto clean reference
     ECG segments from LTAFDB and SHDB-AF.
  2. Real-world Noise-Holter Database AF excerpts stratified by expert quality
     class (Clean, N1, N2).

Rule:
  Lowest bSQI threshold where median_error < 0.5 * between_event_SD.
  If a marker never meets the rule, it is logged as dropped.

Outputs (under <run_dir>/stage3/):
  - sqi_thresholds.csv (carries git_hash, definitions_hash, run_timestamp)
  - sqi_calibration_detail.csv
  - noise_holter_af_by_stratum.csv
  - fig_S1_sqi_calibration.png
"""

from __future__ import annotations

import datetime
import hashlib
import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
import wfdb
import wfdb.processing

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = Path(__file__).resolve().parents[1]

# Reuse verified modules by import
sys.path.insert(0, str(ROOT / "experiments" / "af_density_continuity_v5"))
sys.path.insert(0, str(ROOT / "experiments" / "af_density_continuity_v5" / "scripts"))
sys.path.insert(0, str(ROOT / "experiments" / "af_entrenchment"))
sys.path.insert(0, str(ROOT / "experiments" / "pmv_temporal_specificity" / "scripts"))

from src.afr_pilot import (
    adaptive_median_qrst_cancellation,
    bandpass_filter_ecg,
    estimate_dominant_frequency,
)
from run_e04_vcc_modeling import calc_hrv_and_vcc
from pmv_core import PMVExtractor

# Dataset paths
NOISE_HOLTER_DIR = Path(
    "/data/mithunmanivannan/datasets/noise-holter-database-a-multi-device-multi-arrhythmia-real-world-clinical-ecg-signal-quality-benchmark-1.0.0"
)
NSTDB_DIR = Path(
    "/data/mithunmanivannan/datasets/nstdb/mit-bih-noise-stress-test-database-1.0.0"
)
LTAF_DIR = Path("/data/mithunmanivannan/datasets/ltafdb_1.0.0_reference")
SHDB_DIR = Path("/data/mithunmanivannan/datasets/shdb_af")

MARKERS = [
    "AFR_DF",
    "fwave_SampEn",
    "VCC",
    "RR_SampEn",
    "PMV",
    "PAC_coupling",
    "RMSSD",
]


def get_provenance() -> Dict[str, str]:
    try:
        git_hash = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        git_hash = "unknown"

    defs_path = EXP_DIR / "config" / "definitions.yaml"
    if defs_path.exists():
        definitions_hash = hashlib.sha256(defs_path.read_bytes()).hexdigest()[:12]
    else:
        definitions_hash = "pending_config"

    run_timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "git_hash": git_hash,
        "definitions_hash": definitions_hash,
        "run_timestamp": run_timestamp,
    }


def resolve_output_dir() -> Path:
    for candidate in [EXP_DIR / "LATEST_RUN.txt", EXP_DIR / "results" / "LATEST_RUN.txt"]:
        if candidate.exists():
            txt = candidate.read_text().strip()
            if txt:
                p = Path(txt)
                if not p.is_absolute():
                    p = EXP_DIR / p
                return p / "stage3"
    return EXP_DIR / "results" / "stage3"


def detect_qrs_energy(ecg: np.ndarray, fs: float) -> np.ndarray:
    """Secondary energy-based QRS detector (from 27_afr_rigorous_validation.py)."""
    nyq = 0.5 * fs
    b, a = signal.butter(3, [5.0 / nyq, min(25.0 / nyq, 0.99)], btype="band")
    filt = signal.filtfilt(b, a, ecg)
    sq = np.diff(filt) ** 2
    int_win = max(1, int(round(0.12 * fs)))
    integrated = np.convolve(sq, np.ones(int_win) / int_win, mode="same")
    thresh = np.mean(integrated) + 1.0 * np.std(integrated)
    min_dist = max(1, int(round(0.28 * fs)))
    peaks, _ = signal.find_peaks(integrated, height=thresh, distance=min_dist)
    return peaks.astype(np.int64)


def detect_qrs_xqrs(ecg: np.ndarray, fs: float) -> np.ndarray:
    try:
        peaks = wfdb.processing.xqrs_detect(
            np.asarray(ecg, dtype=np.float64), fs=fs, verbose=False
        )
        return np.asarray(peaks, dtype=np.int64)
    except Exception:
        return np.empty(0, dtype=np.int64)


def compute_bsqi(
    ecg: np.ndarray,
    fs: float,
    qrs1: Optional[np.ndarray] = None,
    tol_ms: float = 50.0,
) -> Tuple[float, np.ndarray]:
    """Beat-agreement Signal Quality Index (F1 between xqrs_detect and energy detector)."""
    if qrs1 is None:
        qrs1 = detect_qrs_xqrs(ecg, fs)
    qrs2 = detect_qrs_energy(ecg, fs)
    if len(qrs1) == 0 or len(qrs2) == 0:
        return 0.0, qrs1
    tol = int(round((tol_ms / 1000.0) * fs))
    # Two-pointer match
    i = j = matched = 0
    while i < len(qrs1) and j < len(qrs2):
        diff = qrs1[i] - qrs2[j]
        if abs(diff) <= tol:
            matched += 1
            i += 1
            j += 1
        elif diff < 0:
            i += 1
        else:
            j += 1
    f1 = (2.0 * matched) / (len(qrs1) + len(qrs2))
    return float(f1), qrs1


def sample_entropy(x: np.ndarray, m: int = 2, r_factor: float = 0.2, max_len: int = 1200) -> float:
    """Fast vectorized Sample Entropy (m=2, r=0.2*SD)."""
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) < m + 5:
        return np.nan
    if len(arr) > max_len:
        # Uniformly subsample/slice central window for speed
        arr = arr[:max_len]
    sd = float(np.std(arr))
    if sd < 1e-12:
        return np.nan
    r = r_factor * sd
    N = len(arr)
    # Build templates of length m and m+1
    xm = np.lib.stride_tricks.sliding_window_view(arr, m)[:-1]  # (N-m, m)
    xm1 = np.lib.stride_tricks.sliding_window_view(arr, m + 1)  # (N-m, m+1)
    n_t = len(xm)
    if n_t < 4:
        return np.nan
    # Pairwise Chebyshev distance in chunks to bound memory
    B = 0
    A = 0
    chunk = 256
    for i in range(0, n_t, chunk):
        i_end = min(n_t, i + chunk)
        # Compare [i:i_end] against [i+1:n_t] upper triangle
        d_m = np.max(np.abs(xm[i:i_end, None, :] - xm[None, :, :]), axis=-1)
        # Mask out j <= idx
        rows = np.arange(i, i_end)[:, None]
        cols = np.arange(n_t)[None, :]
        upper = cols > rows
        m_match = (d_m <= r) & upper
        B += int(np.sum(m_match))
        if np.any(m_match):
            d_last = np.abs(xm1[i:i_end, None, m] - xm1[None, :, m])
            A += int(np.sum(m_match & (d_last <= r)))
    if B == 0 or A == 0:
        return np.nan
    return float(-np.log(A / B))


def compute_afr_and_fwave_sampen(
    ecg: np.ndarray, qrs: np.ndarray, fs: float
) -> Tuple[float, float]:
    """Compute AFR dominant frequency (3-12 Hz) and f-wave SampEn."""
    if len(qrs) < 8 or len(ecg) < int(fs * 5):
        return np.nan, np.nan
    filt = bandpass_filter_ecg(ecg, fs)
    res, _ = adaptive_median_qrst_cancellation(filt, qrs, fs)
    df_res = estimate_dominant_frequency(res, fs, f_min=3.0, f_max=12.0)
    df_hz = float(df_res["dominant_frequency_hz"]) if df_res.get("dominant_frequency_hz") is not None else np.nan
    # Bandpass residual 2.5-12 Hz and downsample to ~50 Hz for f-wave SampEn
    nyq = 0.5 * fs
    b, a = signal.butter(3, [2.5 / nyq, min(12.0 / nyq, 0.99)], btype="band")
    aa = signal.filtfilt(b, a, res)
    decim = max(1, int(round(fs / 50.0)))
    aa_ds = aa[::decim]
    fw_se = sample_entropy(aa_ds, m=2, r_factor=0.2, max_len=1000)
    return df_hz, fw_se


def compute_rr_markers(qrs: np.ndarray, fs: float) -> Dict[str, float]:
    """Compute VCC, RR_SampEn, RMSSD, and first PAC coupling interval (ms)."""
    if len(qrs) < 6:
        return {"VCC": np.nan, "RR_SampEn": np.nan, "RMSSD": np.nan, "PAC_coupling": np.nan}
    rr_ms = np.diff(qrs) / fs * 1000.0
    hrv = calc_hrv_and_vcc(rr_ms)
    if hrv is None:
        return {"VCC": np.nan, "RR_SampEn": np.nan, "RMSSD": np.nan, "PAC_coupling": np.nan}
    valid_rr = rr_ms[(rr_ms >= 200.0) & (rr_ms <= 3000.0)]
    rr_se = sample_entropy(valid_rr, m=2, r_factor=0.2, max_len=500)
    # First PAC coupling: RR_i < 0.80 * median(preceding 5 RR) and >= 250 ms
    pac_coupling = np.nan
    for idx in range(5, len(valid_rr)):
        local_med = float(np.median(valid_rr[idx - 5 : idx]))
        if 250.0 <= valid_rr[idx] < 0.80 * local_med:
            pac_coupling = float(valid_rr[idx])
            break
    return {
        "VCC": float(hrv["vcc_pct"]),
        "RR_SampEn": rr_se,
        "RMSSD": float(hrv["rmssd_ms"]),
        "PAC_coupling": pac_coupling,
    }


def compute_pmv_single_window(
    ecg: np.ndarray, qrs: np.ndarray, fs: float, ref_energy: Optional[float] = None
) -> Tuple[float, float]:
    """Compute PMV_V on a single window using PMVExtractor. Returns (pmv_V, Ei)."""
    if len(qrs) < 20:
        return np.nan, np.nan
    extractor = PMVExtractor(fs=fs, min_beats_per_subsegment=15)
    filt = extractor.filter_signal(ecg)
    p_waves, _ = extractor.extract_p_wave_ensemble(filt, qrs)
    if len(p_waves) < 15:
        return np.nan, np.nan
    Ei = ref_energy if (ref_energy is not None and np.isfinite(ref_energy) and ref_energy > 1e-6) else extractor.compute_reference_energy(p_waves)
    res = extractor.compute_subsegment_pmv(
        filt, qrs, time_center_s=len(ecg) / (2.0 * fs), fixed_reference_energy=Ei
    )
    return (float(res.pmv_V) if res.is_valid else np.nan), float(Ei)


def load_nstdb_noises() -> Dict[str, Tuple[np.ndarray, float]]:
    noises = {}
    for name in ["em", "ma", "bw"]:
        rec = wfdb.rdrecord(str(NSTDB_DIR / name))
        noises[name] = (rec.p_signal[:, 0].astype(np.float64), float(rec.fs))
    return noises


def inject_noise(
    clean_sig: np.ndarray,
    fs: float,
    noise_sig: np.ndarray,
    noise_fs: float,
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Inject NSTDB noise resampled to fs at exact SNR (dB)."""
    n_clean = len(clean_sig)
    n_src = int(np.ceil(n_clean * noise_fs / fs)) + 10
    max_start = max(1, len(noise_sig) - n_src - 1)
    start = int(rng.integers(0, max_start))
    chunk = noise_sig[start : start + n_src]
    if abs(noise_fs - fs) > 1e-3:
        chunk = signal.resample_poly(chunk, int(round(fs)), int(round(noise_fs)))
    chunk = chunk[:n_clean]
    if len(chunk) < n_clean:
        chunk = np.pad(chunk, (0, n_clean - len(chunk)), mode="wrap")
    p_sig = float(np.mean((clean_sig - np.mean(clean_sig)) ** 2))
    p_noise = float(np.mean((chunk - np.mean(chunk)) ** 2))
    if p_sig <= 1e-12 or p_noise <= 1e-12:
        return clean_sig.copy()
    scale = np.sqrt(p_sig / (p_noise * (10.0 ** (snr_db / 10.0))))
    return clean_sig + scale * chunk


def extract_clean_reference_segments(
    n_af: int = 24, n_sinus: int = 24
) -> Tuple[List[dict], List[dict]]:
    """Extract clean 60s AF segments and 60s Sinus segments from LTAFDB & SHDB-AF."""
    af_segs: List[dict] = []
    sinus_segs: List[dict] = []

    # LTAFDB records
    ltaf_recs = sorted([p.stem for p in LTAF_DIR.glob("*.hea")])
    for rec_id in ltaf_recs:
        if len(af_segs) >= n_af // 2 and len(sinus_segs) >= n_sinus // 2:
            break
        rec_path = str(LTAF_DIR / rec_id)
        try:
            hdr = wfdb.rdheader(rec_path)
            fs = float(hdr.fs)
            ann = wfdb.rdann(rec_path, "atr")
        except Exception:
            continue

        # Parse rhythm changes from aux_note
        rhythm_events = [
            (int(s), str(note).strip().strip("\x00"))
            for s, note in zip(ann.sample, ann.aux_note)
            if note and str(note).strip().strip("\x00").startswith("(")
        ]
        win_samp = int(round(60.0 * fs))
        got_af = len(af_segs) >= n_af // 2
        got_sr = len(sinus_segs) >= n_sinus // 2

        for idx in range(len(rhythm_events)):
            if got_af and got_sr:
                break
            s0, r_label = rhythm_events[idx]
            s1 = rhythm_events[idx + 1][0] if idx + 1 < len(rhythm_events) else hdr.sig_len
            if s1 - s0 < win_samp + int(10 * fs):
                continue
            seg_s = s0 + int(5 * fs)
            seg_e = seg_s + win_samp
            if not got_af and "(AFIB" in r_label:
                rec = wfdb.rdrecord(rec_path, sampfrom=seg_s, sampto=seg_e)
                sig = rec.p_signal[:, 0].astype(np.float64)
                bsqi, qrs = compute_bsqi(sig, fs)
                if bsqi >= 0.85 and len(qrs) >= 30:
                    af_segs.append({"cohort": "LTAFDB", "record_id": rec_id, "fs": fs, "ecg": sig, "qrs": qrs, "bsqi": bsqi})
                    got_af = True
            elif not got_sr and "(N" in r_label:
                rec = wfdb.rdrecord(rec_path, sampfrom=seg_s, sampto=seg_e)
                sig = rec.p_signal[:, 0].astype(np.float64)
                bsqi, qrs = compute_bsqi(sig, fs)
                if bsqi >= 0.85 and len(qrs) >= 30:
                    sinus_segs.append({"cohort": "LTAFDB", "record_id": rec_id, "fs": fs, "ecg": sig, "qrs": qrs, "bsqi": bsqi})
                    got_sr = True

    # SHDB-AF records
    shdb_recs = sorted([p.stem for p in SHDB_DIR.glob("*.hea")])
    for rec_id in shdb_recs:
        if len(af_segs) >= n_af and len(sinus_segs) >= n_sinus:
            break
        rec_path = str(SHDB_DIR / rec_id)
        if not (SHDB_DIR / f"{rec_id}.atr").exists():
            continue
        try:
            hdr = wfdb.rdheader(rec_path)
            fs = float(hdr.fs)
            ann = wfdb.rdann(rec_path, "atr")
        except Exception:
            continue

        rhythm_events = [
            (int(s), str(note).strip().strip("\x00"))
            for s, note in zip(ann.sample, ann.aux_note)
            if note and str(note).strip().strip("\x00").startswith("(")
        ]
        win_samp = int(round(60.0 * fs))
        got_af = len(af_segs) >= n_af
        got_sr = len(sinus_segs) >= n_sinus

        for idx in range(len(rhythm_events)):
            if got_af and got_sr:
                break
            s0, r_label = rhythm_events[idx]
            s1 = rhythm_events[idx + 1][0] if idx + 1 < len(rhythm_events) else hdr.sig_len
            if s1 - s0 < win_samp + int(10 * fs):
                continue
            seg_s = s0 + int(5 * fs)
            seg_e = seg_s + win_samp
            if not got_af and "(AFIB" in r_label:
                rec = wfdb.rdrecord(rec_path, sampfrom=seg_s, sampto=seg_e)
                sig = rec.p_signal[:, 0].astype(np.float64)
                bsqi, qrs = compute_bsqi(sig, fs)
                if bsqi >= 0.85 and len(qrs) >= 30:
                    af_segs.append({"cohort": "SHDB-AF", "record_id": rec_id, "fs": fs, "ecg": sig, "qrs": qrs, "bsqi": bsqi})
                    got_af = True
            elif not got_sr and "(N" in r_label:
                rec = wfdb.rdrecord(rec_path, sampfrom=seg_s, sampto=seg_e)
                sig = rec.p_signal[:, 0].astype(np.float64)
                bsqi, qrs = compute_bsqi(sig, fs)
                if bsqi >= 0.85 and len(qrs) >= 30:
                    sinus_segs.append({"cohort": "SHDB-AF", "record_id": rec_id, "fs": fs, "ecg": sig, "qrs": qrs, "bsqi": bsqi})
                    got_sr = True

    return af_segs, sinus_segs


def evaluate_noise_holter_af() -> pd.DataFrame:
    """Evaluate AF markers on real Noise-Holter AF excerpts stratified by Clean/N1/N2."""
    records_file = NOISE_HOLTER_DIR / "RECORDS"
    if not records_file.exists():
        return pd.DataFrame()
    rec_names = [line.strip() for line in records_file.read_text().splitlines() if line.strip()]
    rows = []
    for rname in rec_names:
        rec_path = NOISE_HOLTER_DIR / rname
        hea_path = Path(f"{rec_path}.hea")
        if not hea_path.exists():
            continue
        # Fast header comment scan for AFIB / AF
        text = hea_path.read_text(errors="ignore")
        class_str = "Unknown"
        events_str = ""
        for line in text.splitlines():
            if line.startswith("#") and "CLASS:" in line:
                parts = [p.strip() for p in line.lstrip("#").split("|")]
                for p in parts:
                    if p.startswith("CLASS:"):
                        class_str = p.split(":", 1)[1].strip()
                    elif p.startswith("EVENTS:"):
                        events_str = p.split(":", 1)[1].strip()
        ev_upper = events_str.upper()
        if "A. FIB" not in ev_upper and "AFIB" not in ev_upper and "ATRIAL FIBRILLATION" not in ev_upper:
            continue

        try:
            rec = wfdb.rdrecord(str(rec_path))
            fs = float(rec.fs)
            sig = rec.p_signal[:, 0].astype(np.float64)
            bsqi, det_qrs = compute_bsqi(sig, fs)
            # Reference beats from .atr
            ref_qrs = None
            if Path(f"{rec_path}.atr").exists():
                ann = wfdb.rdann(str(rec_path), "atr")
                ref_qrs = np.asarray(ann.sample, dtype=np.int64)

            df_det, fw_det = compute_afr_and_fwave_sampen(sig, det_qrs, fs)
            rr_det = compute_rr_markers(det_qrs, fs)

            if ref_qrs is not None and len(ref_qrs) >= 8:
                df_ref, fw_ref = compute_afr_and_fwave_sampen(sig, ref_qrs, fs)
                rr_ref = compute_rr_markers(ref_qrs, fs)
            else:
                df_ref, fw_ref = np.nan, np.nan
                rr_ref = {"VCC": np.nan, "RR_SampEn": np.nan, "RMSSD": np.nan, "PAC_coupling": np.nan}

            rows.append({
                "record": rname,
                "stratum": class_str,
                "bsqi": bsqi,
                "AFR_DF_err": abs(df_det - df_ref) if np.isfinite(df_det) and np.isfinite(df_ref) else np.nan,
                "fwave_SampEn_err": abs(fw_det - fw_ref) if np.isfinite(fw_det) and np.isfinite(fw_ref) else np.nan,
                "VCC_err": abs(rr_det["VCC"] - rr_ref["VCC"]) if np.isfinite(rr_det["VCC"]) and np.isfinite(rr_ref["VCC"]) else np.nan,
                "RR_SampEn_err": abs(rr_det["RR_SampEn"] - rr_ref["RR_SampEn"]) if np.isfinite(rr_det["RR_SampEn"]) and np.isfinite(rr_ref["RR_SampEn"]) else np.nan,
                "AFR_DF_det": df_det,
                "VCC_det": rr_det["VCC"],
            })
        except Exception:
            continue

    return pd.DataFrame(rows)


def main():
    out_dir = resolve_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    prov = get_provenance()
    print(f"[Stage 3] Output dir: {out_dir}")

    # 1. Load clean reference segments
    af_segs, sinus_segs = extract_clean_reference_segments(n_af=20, n_sinus=20)
    print(f"[Stage 3] Loaded {len(af_segs)} clean AF segments and {len(sinus_segs)} clean Sinus segments.")

    # Compute clean baseline values & between-event SD
    clean_vals: Dict[str, List[float]] = {m: [] for m in MARKERS}
    for seg in af_segs:
        df_hz, fw_se = compute_afr_and_fwave_sampen(seg["ecg"], seg["qrs"], seg["fs"])
        rr_m = compute_rr_markers(seg["qrs"], seg["fs"])
        seg["clean"] = {
            "AFR_DF": df_hz,
            "fwave_SampEn": fw_se,
            "VCC": rr_m["VCC"],
            "RR_SampEn": rr_m["RR_SampEn"],
        }
        for k, v in seg["clean"].items():
            if np.isfinite(v):
                clean_vals[k].append(v)

    for seg in sinus_segs:
        pmv_v, ei = compute_pmv_single_window(seg["ecg"], seg["qrs"], seg["fs"])
        rr_m = compute_rr_markers(seg["qrs"], seg["fs"])
        seg["ref_energy"] = ei
        seg["clean"] = {
            "PMV": pmv_v,
            "PAC_coupling": rr_m["PAC_coupling"],
            "RMSSD": rr_m["RMSSD"],
        }
        for k, v in seg["clean"].items():
            if np.isfinite(v):
                clean_vals[k].append(v)

    between_sd = {
        m: float(np.std(clean_vals[m], ddof=1)) if len(clean_vals[m]) >= 3 else np.nan
        for m in MARKERS
    }
    print("[Stage 3] Clean between-event SDs:", {k: round(v, 4) if np.isfinite(v) else None for k, v in between_sd.items()})

    # 2. NSTDB Noise Injection across SNRs
    noises = load_nstdb_noises()
    snr_grid = [24.0, 18.0, 12.0, 6.0, 3.0, 0.0, -3.0, -6.0]
    rng = np.random.default_rng(42)

    detail_rows = []
    # AF markers
    for seg in af_segs:
        for ntype, (nsig, nfs) in noises.items():
            for snr in snr_grid:
                noisy = inject_noise(seg["ecg"], seg["fs"], nsig, nfs, snr, rng)
                bsqi, qrs = compute_bsqi(noisy, seg["fs"])
                df_hz, fw_se = compute_afr_and_fwave_sampen(noisy, qrs, seg["fs"])
                rr_m = compute_rr_markers(qrs, seg["fs"])
                obs = {
                    "AFR_DF": df_hz,
                    "fwave_SampEn": fw_se,
                    "VCC": rr_m["VCC"],
                    "RR_SampEn": rr_m["RR_SampEn"],
                }
                for m, val in obs.items():
                    cval = seg["clean"][m]
                    if np.isfinite(cval):
                        err = abs(val - cval) if np.isfinite(val) else np.nan
                        detail_rows.append({
                            "marker": m,
                            "cohort": seg["cohort"],
                            "record_id": seg["record_id"],
                            "noise_type": ntype,
                            "snr_db": snr,
                            "bsqi": bsqi,
                            "clean_val": cval,
                            "noisy_val": val,
                            "abs_error": err,
                        })

    # Sinus markers
    for seg in sinus_segs:
        for ntype, (nsig, nfs) in noises.items():
            for snr in snr_grid:
                noisy = inject_noise(seg["ecg"], seg["fs"], nsig, nfs, snr, rng)
                bsqi, qrs = compute_bsqi(noisy, seg["fs"])
                pmv_v, _ = compute_pmv_single_window(noisy, qrs, seg["fs"], ref_energy=seg["ref_energy"])
                rr_m = compute_rr_markers(qrs, seg["fs"])
                obs = {
                    "PMV": pmv_v,
                    "PAC_coupling": rr_m["PAC_coupling"],
                    "RMSSD": rr_m["RMSSD"],
                }
                for m, val in obs.items():
                    cval = seg["clean"][m]
                    if np.isfinite(cval):
                        err = abs(val - cval) if np.isfinite(val) else np.nan
                        detail_rows.append({
                            "marker": m,
                            "cohort": seg["cohort"],
                            "record_id": seg["record_id"],
                            "noise_type": ntype,
                            "snr_db": snr,
                            "bsqi": bsqi,
                            "clean_val": cval,
                            "noisy_val": val,
                            "abs_error": err,
                        })

    df_detail = pd.DataFrame(detail_rows)
    for k, v in prov.items():
        df_detail[k] = v
    df_detail.to_csv(out_dir / "sqi_calibration_detail.csv", index=False)

    # 3. Determine lowest bSQI threshold where local bin [tau, tau+0.10) median_error < 0.5 * between_event_SD
    # Scan downward from 0.90 to 0.10; stop at first bin that violates the rule.
    thresh_grid = np.round(np.arange(0.90, 0.09, -0.05), 2)
    thresh_rows = []
    for m in MARKERS:
        sub = df_detail[(df_detail["marker"] == m) & df_detail["abs_error"].notna()]
        sd = between_sd[m]
        target = 0.5 * sd if np.isfinite(sd) else np.nan

        chosen_thresh = np.nan
        chosen_med_err = np.nan
        dropped = True

        if np.isfinite(target) and len(sub) > 0:
            # First check clean stratum [0.85, 1.0]
            clean_sub = sub[sub["bsqi"] >= 0.85]
            if len(clean_sub) >= 3 and float(clean_sub["abs_error"].median()) < target:
                chosen_thresh = 0.85
                chosen_med_err = float(clean_sub["abs_error"].median())
                dropped = False
                for tau in thresh_grid:
                    if tau >= 0.85:
                        continue
                    in_bin = sub[(sub["bsqi"] >= tau) & (sub["bsqi"] < tau + 0.10)]
                    if len(in_bin) < 5:
                        continue
                    med_err = float(in_bin["abs_error"].median())
                    if med_err < target:
                        chosen_thresh = float(tau)
                        chosen_med_err = med_err
                    else:
                        break

        # Stratum medians
        med_clean = float(sub[sub["bsqi"] >= 0.85]["abs_error"].median()) if len(sub[sub["bsqi"] >= 0.85]) else np.nan
        med_n1 = float(sub[(sub["bsqi"] >= 0.50) & (sub["bsqi"] < 0.85)]["abs_error"].median()) if len(sub[(sub["bsqi"] >= 0.50) & (sub["bsqi"] < 0.85)]) else np.nan
        med_n2 = float(sub[sub["bsqi"] < 0.50]["abs_error"].median()) if len(sub[sub["bsqi"] < 0.50]) else np.nan

        thresh_rows.append({
            "marker": m,
            "n_clean_ref": len(clean_vals[m]),
            "between_event_sd": sd,
            "target_half_sd": target,
            "bsqi_threshold": chosen_thresh,
            "median_err_at_threshold": chosen_med_err,
            "median_err_clean": med_clean,
            "median_err_N1": med_n1,
            "median_err_N2": med_n2,
            "dropped": dropped,
            **prov,
        })

    df_thresh = pd.DataFrame(thresh_rows)
    df_thresh.to_csv(out_dir / "sqi_thresholds.csv", index=False)
    print("\n[Stage 3] Calibrated SQI Thresholds:")
    print(df_thresh[["marker", "between_event_sd", "target_half_sd", "bsqi_threshold", "median_err_at_threshold", "dropped"]].to_string(index=False))

    # 4. Real Noise-Holter AF excerpts by stratum
    df_nh = evaluate_noise_holter_af()
    print(f"\n[Stage 3] Evaluated {len(df_nh)} real Noise-Holter AF excerpts.")
    if len(df_nh) > 0:
        nh_summary = (
            df_nh.groupby("stratum", dropna=False)
            .agg(
                n_excerpts=("record", "count"),
                bsqi_median=("bsqi", "median"),
                AFR_DF_med_err=("AFR_DF_err", "median"),
                fwave_SampEn_med_err=("fwave_SampEn_err", "median"),
                VCC_med_err=("VCC_err", "median"),
                RR_SampEn_med_err=("RR_SampEn_err", "median"),
            )
            .reset_index()
        )
        for k, v in prov.items():
            nh_summary[k] = v
        nh_summary.to_csv(out_dir / "noise_holter_af_by_stratum.csv", index=False)
        print(nh_summary[["stratum", "n_excerpts", "bsqi_median", "AFR_DF_med_err", "VCC_med_err"]].to_string(index=False))

    # 5. Plot Figure S1: one panel per marker
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), dpi=200)
    axes_flat = axes.flatten()
    bins = np.linspace(0.0, 1.0, 11)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    for idx, m in enumerate(MARKERS):
        ax = axes_flat[idx]
        sub = df_detail[(df_detail["marker"] == m) & df_detail["abs_error"].notna()]
        row = df_thresh[df_thresh["marker"] == m].iloc[0]
        meds, q25s, q75s, xs = [], [], [], []
        for b0, b1, xc in zip(bins[:-1], bins[1:], bin_centers):
            in_b = sub[(sub["bsqi"] >= b0) & (sub["bsqi"] < (b1 + 1e-6 if b1 == 1.0 else b1))]["abs_error"]
            if len(in_b) >= 3:
                xs.append(xc)
                meds.append(float(in_b.median()))
                q25s.append(float(in_b.quantile(0.25)))
                q75s.append(float(in_b.quantile(0.75)))
        if xs:
            ax.plot(xs, meds, "-o", color="#1f77b4", lw=2, ms=4, label="Median |err|")
            ax.fill_between(xs, q25s, q75s, color="#1f77b4", alpha=0.2, label="IQR")
        if np.isfinite(row["target_half_sd"]):
            ax.axhline(row["target_half_sd"], color="#d62728", ls="--", lw=1.5, label="0.5 × SD")
        if np.isfinite(row["bsqi_threshold"]):
            ax.axvline(row["bsqi_threshold"], color="#2ca02c", ls=":", lw=1.8, label=f"τ={row['bsqi_threshold']:.2f}")
        ax.axvspan(0.85, 1.0, color="#2ca02c", alpha=0.07)
        ax.axvspan(0.50, 0.85, color="#ff7f0e", alpha=0.07)
        ax.axvspan(0.0, 0.50, color="#d62728", alpha=0.07)
        title_sfx = " (DROPPED)" if row["dropped"] else f" (τ={row['bsqi_threshold']:.2f})"
        ax.set_title(f"{m}{title_sfx}", fontsize=10, fontweight="bold")
        ax.set_xlabel("bSQI")
        ax.set_ylabel("Absolute Error")
        ax.set_xlim(0.0, 1.0)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7.5, loc="upper right")

    axes_flat[-1].axis("off")
    fig.suptitle("Figure S1: Empirical bSQI Calibration (NSTDB em/ma/bw Injection)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "fig_S1_sqi_calibration.png", dpi=200)
    plt.close(fig)
    print(f"[Stage 3] Saved {out_dir / 'fig_S1_sqi_calibration.png'}")


if __name__ == "__main__":
    main()
