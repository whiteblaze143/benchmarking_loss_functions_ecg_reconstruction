"""Deterministic adapter for the released 48-channel PanoBench files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io
from scipy.signal import resample
import torch
from torch.utils.data import Dataset


PANOBENCH_ANGLES_RAD = np.deg2rad(
    np.asarray(
        [(90, 90), (150, 90)]
        + [
            (106, -102), (121, -101), (132, -99), (52, -83), (68, -78),
            (90, -74), (109, -75), (125, -77), (137, -81), (43, -74),
            (63, -61), (90, -54), (113, -55), (131, -62), (144, -70),
            (30, -73), (54, -51), (90, -33), (118, -40), (137, -54),
            (149, -64), (20, 70), (48, 42), (90, 11), (122, 32),
            (141, 51), (153, 63), (30, 69), (54, 48), (90, 32),
            (119, 41), (139, 55), (152, 67), (40, 80), (60, 71),
            (90, 65), (117, 66), (135, 69), (147, 77), (112, 105),
            (129, 103), (140, 100),
        ],
        dtype=np.float32,
    )
).astype(np.float32)


class ReleasedPanoBench(Dataset):
    """Any-pairs samples using I and II plus one recorded torso view."""

    def __init__(self, root: Path, seed: int):
        self.files = sorted(Path(root).glob("*.mat"), key=lambda path: int(path.stem))
        if not self.files:
            raise FileNotFoundError(f"no PanoBench .mat files under {root}")
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int):
        values = scipy.io.loadmat(self.files[index])["Panobench"][:44].astype(np.float32)
        if values.shape != (44, 2500) or not np.isfinite(values).all():
            raise ValueError(f"invalid PanoBench record: {self.files[index]}")
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        values = resample(values, 5000, axis=-1).astype(np.float32)
        crop_start = int(rng.integers(0, 5000 - 4608 + 1))
        values = values[:, crop_start : crop_start + 4608]
        lo, hi = float(values.min()), float(values.max())
        if not hi > lo:
            raise ValueError(f"constant PanoBench record: {self.files[index]}")
        values = (values - lo) / (hi - lo)

        third_input = int(rng.integers(2, 44))
        candidates = np.delete(np.arange(2, 44), third_input - 2)
        target = int(rng.choice(candidates))
        input_indices = np.asarray([0, 1, third_input])
        return {
            "input": torch.from_numpy(values[input_indices]),
            "input_angles": torch.from_numpy(PANOBENCH_ANGLES_RAD[input_indices]),
            "target": torch.from_numpy(values[target : target + 1]),
            "target_angle": torch.from_numpy(PANOBENCH_ANGLES_RAD[target]),
        }
