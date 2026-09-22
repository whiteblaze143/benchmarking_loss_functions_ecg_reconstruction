"""Strict adapters for released P/QRS/T delineation supervision.

All functions preserve native coordinates. Resampling belongs to a model's
declared input contract, never to label ingestion.
"""

from __future__ import annotations

from pathlib import Path
import pickle
from typing import Iterable

import numpy as np
import torch
import wfdb


DELINEATION_CLASSES = ("background", "P", "QRS", "T")
ISP_CODE_TO_CLASS = {0: 1, 1: 2, 2: 3}
LUDB_SYMBOL_TO_CLASS = {"p": 1, "N": 2, "t": 3}


def intervals_to_mask(
    intervals: Iterable[tuple[int, int, int]],
    length: int,
    *,
    code_to_class: dict[int, int],
) -> np.ndarray:
    """Convert inclusive released intervals to an unambiguous class mask."""
    if length < 1:
        raise ValueError("delineation mask length must be positive")
    mask = np.zeros(length, dtype=np.int64)
    for code, start, end in intervals:
        if code not in code_to_class:
            raise ValueError(f"unknown delineation code: {code}")
        if not (0 <= start <= end < length):
            raise ValueError(f"invalid inclusive delineation interval: {(code, start, end)}")
        target = code_to_class[code]
        occupied = mask[start : end + 1]
        if np.any(occupied != 0):
            raise ValueError(f"overlapping delineation intervals at {(code, start, end)}")
        mask[start : end + 1] = target
    return mask


def ludb_lead_intervals(record_path: Path, lead_extension: str) -> list[tuple[int, int, int]]:
    """Read LUDB's parenthesized onset/peak/offset annotation triplets."""
    annotation = wfdb.rdann(str(record_path), lead_extension)
    active: tuple[int, int] | None = None
    intervals: list[tuple[int, int, int]] = []
    for sample, symbol in zip(annotation.sample, annotation.symbol, strict=True):
        sample = int(sample)
        if symbol == "(":
            if active is not None:
                raise ValueError(f"nested LUDB onset in {record_path}.{lead_extension}")
            active = (sample, 0)
            continue
        if symbol in LUDB_SYMBOL_TO_CLASS:
            if active is None:
                raise ValueError(f"LUDB wave symbol without onset in {record_path}.{lead_extension}")
            start, existing_class = active
            if existing_class != 0:
                raise ValueError(f"multiple LUDB wave symbols in one interval: {record_path}.{lead_extension}")
            active = (start, LUDB_SYMBOL_TO_CLASS[symbol])
            continue
        if symbol == ")":
            if active is None or active[1] == 0:
                raise ValueError(f"LUDB offset without typed onset in {record_path}.{lead_extension}")
            intervals.append((active[1], active[0], sample))
            active = None
            continue
        raise ValueError(f"unknown LUDB annotation symbol {symbol!r} in {record_path}.{lead_extension}")
    if active is not None:
        raise ValueError(f"unterminated LUDB annotation interval in {record_path}.{lead_extension}")
    return intervals


def ludb_lead_mask(record_path: Path, lead_extension: str, length: int) -> np.ndarray:
    """Return the exact native-coordinate LUDB P/QRS/T mask for one lead."""
    return intervals_to_mask(
        ludb_lead_intervals(record_path, lead_extension),
        length,
        code_to_class={1: 1, 2: 2, 3: 3},
    )


def load_zhejiang_mask(path: Path) -> np.ndarray:
    """Load one released Zhejiang four-class native mask."""
    with path.open("rb") as handle:
        values = np.asarray(pickle.load(handle), dtype=np.int64)
    if values.ndim != 1 or not np.isin(values, (0, 1, 2, 3)).all():
        raise ValueError(f"invalid Zhejiang delineation mask: {path}")
    return values


def rdb_payload_masks(payload: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    """Validate and return native RDB segmentation and validity masks."""
    segmentation = torch.as_tensor(payload["segmentation"]).cpu().numpy().astype(np.int64, copy=False)
    valid = torch.as_tensor(payload["seg_valid"]).cpu().numpy().astype(bool, copy=False)
    if segmentation.ndim != 2 or segmentation.shape[0] != 12 or valid.shape != segmentation.shape:
        raise ValueError("RDB segmentation and validity masks must be matching [12,time] arrays")
    if not np.isin(segmentation, (-1, 0, 1, 2, 3)).all():
        raise ValueError("RDB segmentation contains unknown class IDs")
    if np.any(valid & (segmentation < 0)):
        raise ValueError("RDB valid mask includes invalid segmentation samples")
    return segmentation, valid
