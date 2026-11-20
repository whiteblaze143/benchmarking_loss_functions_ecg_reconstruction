"""Shared, deterministic data and endpoint definitions for the HRS rerun."""
from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
import re

import numpy as np
import pandas as pd
from scipy import signal, stats

AF_RHYTHMS = {"AFIB", "AFL", "AT"}
_BANDPASS = signal.butter(3,[5,25],fs=200,btype="bandpass",output="sos")
_PHASE_GRID = np.linspace(0,1,256)


@lru_cache(maxsize=512)
def _beat_grid(length):
    return np.linspace(0,1,length)


def _welch_power(ecg):
    """Vectorized Hann/50%-overlap Welch estimator, equivalent to scipy.signal.welch defaults."""
    length=min(len(ecg),400)
    window=signal.get_window('hann',length).astype(np.float32)
    segments=np.lib.stride_tricks.sliding_window_view(ecg,length)[::length-length//2]
    spectrum=np.fft.rfft((segments-segments.mean(-1,keepdims=True))*window)
    psd=(spectrum*spectrum.conj()).real.mean(0)/(200*(window*window).sum())
    psd[1:-1 if length%2==0 else None]*=2
    return np.fft.rfftfreq(length,1/200),psd


def parse_atr_gold_standard_episodes(atr, sig_len, fs):
    """Merge contiguous AF/AFL/AT states into one atrial tachyarrhythmia episode."""
    transitions = [(int(s), n.strip().strip("(\x00").strip())
                   for s, n in zip(atr.sample, atr.aux_note)
                   if n.strip().startswith("(")]
    episodes = []
    for i, (start, rhythm) in enumerate(transitions):
        end = transitions[i + 1][0] if i + 1 < len(transitions) else sig_len
        if end <= start:
            continue
        is_af = rhythm in AF_RHYTHMS
        if episodes and episodes[-1]["is_af"] == is_af and episodes[-1]["end_sample"] == start:
            ep = episodes[-1]
            ep["end_sample"] = end
            ep["end_sec"] = end / fs
            ep["duration_sec"] = (end - ep["start_sample"]) / fs
        else:
            episodes.append(dict(rhythm=rhythm, start_sample=start, end_sample=end,
                                 start_sec=start / fs, end_sec=end / fs,
                                 duration_sec=(end - start) / fs, is_af=is_af))
    return episodes


def add_ablation_dates(meta):
    meta = meta.copy()
    holter = pd.to_datetime(meta["Date_Holter"], errors="raise")
    first = pd.to_datetime(meta["Date_1st_AF_Ablation"], errors="raise")
    redo = pd.to_datetime(meta["Date_Redo_AF_Ablation"], errors="raise")
    # Most recent procedure preceding the recording; future procedures do not count.
    prior = pd.concat([first.where(first <= holter), redo.where(redo <= holter)], axis=1).max(axis=1)
    meta["days_from_ablation"] = (holter - prior).dt.days
    meta["post_blanking"] = meta["Previous_AF_Ablation"] & (meta["days_from_ablation"] > 90)
    return meta


def parse_noise_labels(comments):
    fields = {}
    for part in " | ".join(comments).split("|"):
        if ":" in part:
            key, value = part.strip().lstrip("#").strip().split(":", 1)
            fields[key.strip()] = value.strip()
    quality = fields["CLASS"]
    if quality not in {"Clean", "N1", "N2"}:
        raise ValueError(f"Unknown quality label: {quality}")
    events = fields.get("EVENTS", "")
    if not events:
        raise ValueError("Missing rhythm/event label")
    af = bool(re.search(r"A\.\s*(?:Fib\.|Flutter)", events))
    return quality, int(af), events


def extract_features(ecg, fs, anchors, return_cells=False):
    """Use actual signal and autonomous peaks; never synthesize beats or RR variability."""
    ecg = np.asarray(ecg, dtype=np.float32)
    if not np.isfinite(ecg).all() or len(ecg) < fs:
        raise ValueError("ECG must contain finite samples and at least one second")
    if fs != 200:
        ratio = Fraction(200 / fs).limit_denominator(10000)
        ecg = signal.resample_poly(ecg, ratio.numerator, ratio.denominator).astype(np.float32)
    fs = 200.0
    filtered = signal.sosfiltfilt(_BANDPASS, ecg)
    energy = signal.convolve(filtered ** 2, np.ones(16) / 16, mode="same")
    prominence = max(float(np.std(energy)) * 0.5, 1e-12)
    peaks, _ = signal.find_peaks(energy, distance=50, prominence=prominence)
    rrs = np.diff(peaks) / fs
    rr = np.zeros(60, dtype=np.float32)
    if len(rrs):
        rr[:] = np.mean(rrs)
        rr[:min(60, len(rrs))] = rrs[-60:]
    hrv = np.zeros(8, dtype=np.float32)
    if len(rrs) >= 3:
        diff = np.diff(rrs)
        mean, sd = np.mean(rrs), np.std(rrs, ddof=1)
        sd1 = np.sqrt(max(0, 0.5 * np.var(diff)))
        sd2 = np.sqrt(max(0, 2 * np.var(rrs) - 0.5 * np.var(diff)))
        hrv[:] = [mean, sd, np.sqrt(np.mean(diff ** 2)), np.mean(np.abs(diff) > .05),
                  sd / (mean + 1e-6), sd1, sd2, sd1 / (sd2 + 1e-6)]
    beats = [np.interp(_PHASE_GRID, _beat_grid(int(b-a)), ecg[a:b])
             for a, b in zip(peaks[:-1], peaks[1:]) if 50 <= b - a <= 500]
    kme = np.zeros((16, 128), dtype=np.float32)
    cells = np.asarray(beats, dtype=np.float32).reshape(-1,16,16)
    if beats and not return_cells:
        for g in range(16):
            distances = ((cells[:, g, None, :] - anchors[None, :, :]) ** 2).sum(-1)
            kme[g] = np.exp(-distances / 16).mean(0)
        kme /= np.linalg.norm(kme, axis=1, keepdims=True) + 1e-8
    f, psd = _welch_power(ecg)
    power = psd[(f >= 5) & (f <= 15)].sum() / (psd[(f >= 1) & (f <= 40)].sum() + 1e-10)
    centered=ecg-ecg.mean()
    second_moment=np.mean(centered**2)
    kurt = float(np.mean(centered**4)/second_moment**2) if second_moment>0 else 0
    bpm = len(peaks) / (len(ecg) / fs) * 60
    bsqi = 1 if 40 <= bpm <= 180 else (0.5 if 30 <= bpm <= 220 else 0)
    sqi = .4 * bsqi + .35 * np.clip((power - .15) / .5, 0, 1) + .25 * np.clip((kurt - 2.5) / 5, 0, 1)
    result = (kme, rr, hrv, float(sqi), len(peaks))
    return (*result,cells) if return_cells else result


def quality_mask(probs, sqi, quality_threshold=.55, uncertainty_threshold=.35):
    return (sqi < quality_threshold) | (np.minimum(probs, 1 - probs) > uncertainty_threshold)
