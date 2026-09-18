"""Streaming detector and fixed-support morphology tokens for Paper 11."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
from scipy.signal import butter, lfilter, sosfilt


@dataclass(frozen=True)
class CausalTokenConfig:
    sampling_hz: int = 500
    detector_low_hz: float = 5.0
    detector_high_hz: float = 20.0
    detector_order: int = 2
    integration_ms: int = 120
    burn_in_ms: int = 2000
    refractory_ms: int = 250
    backward_search_ms: int = 200
    max_latency_ms: int = 250
    token_before_ms: int = 200
    token_after_ms: int = 300

    def sha256(self) -> str:
        encoded = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CausalAnchor:
    anchor_sample: int
    confirmation_sample: int

    @property
    def latency_samples(self) -> int:
        return self.confirmation_sample - self.anchor_sample


@dataclass(frozen=True)
class MorphologyToken:
    token_id: int
    anchor_sample: int
    confirmation_sample: int
    token_start: int
    token_stop: int
    availability_sample: int
    values: np.ndarray


def causal_qrs_anchors(signal_mv: np.ndarray, config: CausalTokenConfig = CausalTokenConfig()) -> tuple[CausalAnchor, ...]:
    if signal_mv.ndim != 2 or signal_mv.shape[1] != 12:
        raise ValueError("signal must have shape (time,12)")
    if not np.isfinite(signal_mv).all():
        raise ValueError("signal contains non-finite samples")
    fs = config.sampling_hz
    lead_ii = np.asarray(signal_mv[:, 1], dtype=np.float64)
    sos = butter(config.detector_order, [config.detector_low_hz, config.detector_high_hz], btype="bandpass", fs=fs, output="sos")
    filtered = sosfilt(sos, lead_ii)
    derivative = np.diff(filtered, prepend=filtered[0])
    width = max(1, round(config.integration_ms * fs / 1000))
    integrated = lfilter(np.full(width, 1.0 / width), [1.0], np.square(derivative))
    burn_in = round(config.burn_in_ms * fs / 1000)
    refractory = round(config.refractory_ms * fs / 1000)
    backward = round(config.backward_search_ms * fs / 1000)
    if len(integrated) <= burn_in:
        return ()
    initial = integrated[:burn_in]
    noise_level = float(np.median(initial))
    signal_level = float(np.quantile(initial, 0.99))
    threshold = noise_level + 0.25 * max(signal_level - noise_level, np.finfo(float).eps)
    anchors: list[CausalAnchor] = []
    above = False
    last_confirmation = -refractory
    last_anchor = -refractory
    for sample in range(burn_in, len(integrated)):
        value = float(integrated[sample])
        crossing = value >= threshold and not above
        above = value >= threshold
        if crossing and sample - last_confirmation >= refractory:
            start = max(0, sample - backward)
            anchor = start + int(np.argmax(np.abs(filtered[start : sample + 1])))
            if anchor - last_anchor >= refractory:
                anchors.append(CausalAnchor(anchor, sample))
                last_anchor = anchor
                last_confirmation = sample
                signal_level = 0.125 * value + 0.875 * signal_level
                threshold = noise_level + 0.25 * max(signal_level - noise_level, np.finfo(float).eps)
                continue
        noise_level = 0.01 * value + 0.99 * noise_level
        threshold = noise_level + 0.25 * max(signal_level - noise_level, np.finfo(float).eps)
    return tuple(anchors)


def build_morphology_tokens(
    signal_mv: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    config: CausalTokenConfig = CausalTokenConfig(),
) -> tuple[MorphologyToken, ...]:
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    if mean.shape != (12,) or std.shape != (12,) or np.any(std <= 0):
        raise ValueError("mean/std must have shape (12,) and positive std")
    before = round(config.token_before_ms * config.sampling_hz / 1000)
    after = round(config.token_after_ms * config.sampling_hz / 1000)
    maximum_latency = round(config.max_latency_ms * config.sampling_hz / 1000)
    standardized = (np.asarray(signal_mv, dtype=np.float64) - mean) / std
    tokens: list[MorphologyToken] = []
    for anchor in causal_qrs_anchors(signal_mv, config):
        start = anchor.anchor_sample - before
        stop = anchor.anchor_sample + after
        if start < 0 or stop > len(signal_mv) or not (0 <= anchor.latency_samples <= maximum_latency):
            continue
        values = np.ascontiguousarray(standardized[start:stop], dtype=np.float32)
        tokens.append(MorphologyToken(len(tokens), anchor.anchor_sample, anchor.confirmation_sample, start, stop, max(anchor.confirmation_sample, stop), values))
    return tuple(tokens)


def eligible_next_token_examples(tokens: tuple[MorphologyToken, ...], prefix_length: int) -> tuple[dict[str, object], ...]:
    if prefix_length < 1:
        raise ValueError("prefix_length must be positive")
    rows: list[dict[str, object]] = []
    for target_index in range(prefix_length, len(tokens)):
        prefix = tokens[target_index - prefix_length : target_index]
        target = tokens[target_index]
        prefix_stop = max(token.token_stop for token in prefix)
        prefix_available = max(token.availability_sample for token in prefix)
        eligible = target.token_start > prefix_stop and target.token_start > prefix_available
        rows.append({
            "prefix_token_ids": [token.token_id for token in prefix],
            "target_token_id": target.token_id,
            "prefix_raw_max": prefix_stop,
            "prefix_availability_max": prefix_available,
            "target_raw_min": target.token_start,
            "target_raw_stop": target.token_stop,
            "eligible": eligible,
            "eligibility_reason": "" if eligible else "target_not_strictly_after_prefix_availability",
        })
    return tuple(rows)
