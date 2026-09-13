"""Clinical Any-Pairs dataset adapter for Stage I N003 pretraining.

Adheres strictly to the user-specified deployment-consistent normalization contract:
    m_obs = min_{j in O, t} x_j(t)
    M_obs = max_{j in O, t} x_j(t)
    x_tilde = (x - m_obs) / (M_obs - m_obs)

Target query waveforms are normalized strictly using observed-view extrema,
preventing target statistics leakage into the model.

Datasets:
    PTB-XL, Zhejiang/ChinaDB, CPSC2018 (strictly separated from LUDB, ISP, RDB, Emory).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


# Standard 12-lead angles in spherical coordinates (theta, phi) in radians:
# Lead ordering matching standard WFDB 12-lead:
# 0: I, 1: II, 2: III, 3: aVR, 4: aVL, 5: aVF, 6: V1, 7: V2, 8: V3, 9: V4, 10: V5, 11: V6
STANDARD_12LEAD_ANGLES_RAD = np.asarray([
    [np.pi / 2, np.pi / 2],          # 0: I
    [np.pi * 5 / 6, np.pi / 2],      # 1: II
    [np.pi * 5 / 6, -np.pi / 2],     # 2: III
    [np.pi / 3, -np.pi / 2],         # 3: aVR
    [np.pi / 3, np.pi / 2],          # 4: aVL
    [np.pi, np.pi / 2],              # 5: aVF
    [np.pi / 2, -np.pi / 18],        # 6: V1
    [np.pi / 2, np.pi / 18],         # 7: V2
    [np.pi * 19 / 36, np.pi / 12],   # 8: V3
    [np.pi * 11 / 20, np.pi / 6],    # 9: V4
    [np.pi * 16 / 30, np.pi / 3],    # 10: V5
    [np.pi * 16 / 30, np.pi / 2],    # 11: V6
], dtype=np.float32)


class PTBXL12LeadDataset(Dataset):
    """PTB-XL dataset returning all 12 leads for batch-level dynamic Any-Pairs sampling."""

    def __init__(
        self,
        tensors_dir: Path | str,
        split: str = "train",
        seed: int = 123,
        target_len: int = 4608,
    ):
        self.tensors_dir = Path(tensors_dir) / split
        if not self.tensors_dir.is_dir():
            raise FileNotFoundError(f"Tensors directory not found: {self.tensors_dir}")
        self.files = sorted(list(self.tensors_dir.glob("*.pt")), key=lambda p: int(p.stem))
        if not self.files:
            raise FileNotFoundError(f"No .pt files found in {self.tensors_dir}")
        self.split = split
        self.seed = int(seed)
        self.target_len = int(target_len)
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> torch.Tensor:
        filepath = self.files[index]
        data = torch.load(filepath, map_location="cpu", weights_only=True)
        if isinstance(data, dict):
            data = data["ecg"]
        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(np.asarray(data))
        data = data.float()

        num_leads, total_len = data.shape
        if num_leads != 12:
            raise ValueError(f"Expected 12 leads, got {num_leads} in {filepath}")

        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        if total_len > self.target_len:
            crop_start = int(rng.integers(0, total_len - self.target_len + 1))
            crop = data[:, crop_start : crop_start + self.target_len]
        elif total_len < self.target_len:
            pad_len = self.target_len - total_len
            crop = torch.nn.functional.pad(data, (0, pad_len), mode="constant", value=0.0)
        else:
            crop = data
        return crop  # [12, target_len]


class PTBXLClinicalAnyPairs(Dataset):
    """PTB-XL Any-Pairs dataset for Stage I GeoVT clinical pretraining."""

    def __init__(
        self,
        tensors_dir: Path | str,
        split: str = "train",
        lead_cardinality: int | None = 3,
        seed: int = 123,
        target_len: int = 4608,
    ):
        self.tensors_dir = Path(tensors_dir) / split
        if not self.tensors_dir.is_dir():
            raise FileNotFoundError(f"Tensors directory not found: {self.tensors_dir}")
        self.files = sorted(list(self.tensors_dir.glob("*.pt")), key=lambda p: int(p.stem))
        if not self.files:
            raise FileNotFoundError(f"No .pt files found in {self.tensors_dir}")
        self.split = split
        self.lead_cardinality = int(lead_cardinality) if lead_cardinality is not None else None
        self.seed = int(seed)
        self.target_len = int(target_len)
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        filepath = self.files[index]
        # Raw shape [12, 5000]
        data = torch.load(filepath, map_location="cpu", weights_only=True)
        if isinstance(data, dict):
            data = data["ecg"]
        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(np.asarray(data))
        data = data.float()

        num_leads, total_len = data.shape
        if num_leads != 12:
            raise ValueError(f"Expected 12 leads, got {num_leads} in {filepath}")

        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))

        # Random temporal crop to target_len (4608)
        if total_len > self.target_len:
            crop_start = int(rng.integers(0, total_len - self.target_len + 1))
            crop = data[:, crop_start : crop_start + self.target_len]
        elif total_len < self.target_len:
            pad_len = self.target_len - total_len
            crop = torch.nn.functional.pad(data, (0, pad_len), mode="constant", value=0.0)
        else:
            crop = data

        # Select input slots: anchor leads I (0) and II (1), plus additional leads
        if self.lead_cardinality is not None:
            num_additional = max(1, self.lead_cardinality - 2)
        else:
            # Dynamic cardinality: k in {1, 2, 3} -> L_input in {3, 4, 5}
            num_additional = int(rng.choice([1, 2, 3]))

        available_additional = list(range(2, 12))
        selected_additional = rng.choice(available_additional, size=num_additional, replace=False).tolist()
        input_indices = [0, 1] + sorted(selected_additional)

        # Select 1 target lead from remaining leads
        unused_leads = [i for i in range(12) if i not in input_indices]
        target_index = int(rng.choice(unused_leads))

        input_waveforms = crop[input_indices]  # [C_in, L]
        target_waveform = crop[target_index : target_index + 1]  # [1, L]

        # Prospective deployment-consistent normalization:
        # Compute min/max strictly from observed input views!
        m_obs = float(input_waveforms.min())
        M_obs = float(input_waveforms.max())
        diff = M_obs - m_obs
        if diff < 1e-6:
            diff = 1.0

        input_norm = (input_waveforms - m_obs) / diff
        target_norm = (target_waveform - m_obs) / diff

        input_angles = torch.from_numpy(STANDARD_12LEAD_ANGLES_RAD[input_indices])
        target_angle = torch.from_numpy(STANDARD_12LEAD_ANGLES_RAD[target_index])

        return {
            "input": input_norm,                     # [lead_cardinality, target_len]
            "input_angles": input_angles,           # [lead_cardinality, 2]
            "target": target_norm,                   # [1, target_len]
            "target_angle": target_angle,           # [2]
            "input_indices": torch.tensor(input_indices, dtype=torch.long),
            "target_index": torch.tensor(target_index, dtype=torch.long),
            "m_obs": torch.tensor(m_obs, dtype=torch.float32),
            "M_obs": torch.tensor(M_obs, dtype=torch.float32),
        }
