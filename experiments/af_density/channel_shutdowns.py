"""Prespecified controlled availability changes, without signal corruption."""
import hashlib
import json
import numpy as np


def shutdown_schedule(window_count, *, dataset, record, segment, fraction, block_seconds, seed=20261120):
    """Select disjoint 30s-window slots using no labels or predictions.

    Randomly permute candidate slots and take the prefix closest to the rounded
    requested window count, at least one slot. A final short slot is retained.
    Realized elapsed missing time must be computed from original assignments.
    """
    if not isinstance(window_count, int) or window_count < 1 or fraction not in [.01, .05, .10] or block_seconds not in [60, 300]:
        raise ValueError('Unsupported declared shutdown configuration')
    identity = [dataset, record, segment, fraction, block_seconds, seed]
    encoded = json.dumps(identity, separators=(',', ':'), ensure_ascii=True).encode()
    rng_seed = int.from_bytes(hashlib.sha256(encoded).digest()[:8], 'little')
    width = block_seconds//30
    slots = [(a, min(a+width, window_count)) for a in range(0, window_count, width)]
    order = np.random.default_rng(rng_seed).permutation(len(slots))
    cumulative = np.cumsum([slots[i][1]-slots[i][0] for i in order])
    target = int(round(window_count*fraction))
    count = int(np.argmin(abs(cumulative-target)))+1
    chosen = sorted(slots[i] for i in order[:count])
    missing = np.zeros(window_count, bool)
    for a, b in chosen: missing[a:b] = True
    return missing, {'seed_identity': identity, 'rng_seed': rng_seed, 'selected_slots': chosen,
                     'requested_fraction': fraction, 'realized_window_fraction': float(missing.mean()),
                     'missing_windows': int(missing.sum()), 'clock_policy': 'Mask original windows; no clock compression'}


def select_available_scores(scores, available_channels):
    """Select each model's genuine cached subset score, never zero-fill inputs.

    Scores columns correspond to channel0, channel1, both. A model failure in
    the actually available subset stays NaN; no fallback to another subset.
    """
    p = np.asarray(scores, dtype=np.float64);available = np.asarray(available_channels)
    if p.ndim != 2 or p.shape[1] != 3 or available.shape != (len(p), 2) or available.dtype != bool:
        raise ValueError('Expected three subset scores and boolean channel availability')
    if np.isinf(p).any() or ((p[np.isfinite(p)] < 0) | (p[np.isfinite(p)] > 1)).any():
        raise ValueError('Expected probabilities or explicit missing scores')
    code = available[:, 0].astype(np.int8)+2*available[:, 1]
    result = np.full(len(p), np.nan)
    for key, column in [(1, 0), (2, 1), (3, 2)]:
        take = code == key;result[take] = p[take, column]
    return result
