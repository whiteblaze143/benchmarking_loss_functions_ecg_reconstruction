"""Clinical Any-Pairs dataset adapter for Stage I N003 pretraining.

PTB-XL tensors are reordered from WFDB order to the Nef-Net canonical order.
Both inputs and targets use one frozen, train-derived amplitude transform.

Datasets:
    PTB-XL, Zhejiang/ChinaDB, CPSC2018 (strictly separated from LUDB, ISP, RDB, Emory).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


# On-disk WFDB order -> Nef-Net canonical order
# [I, II, V1, V2, V3, V4, V5, V6, III, aVR, aVL, aVF].
DISK_TO_CANONICAL = [0, 1, 6, 7, 8, 9, 10, 11, 2, 3, 4, 5]
INDEPENDENT_LEAD_INDICES = tuple(range(8))
PRECORDIAL_LEAD_INDICES = tuple(range(2, 8))
TRAIN_LOWER_MV = -4.0
TRAIN_UPPER_MV = 4.0
NORMALIZATION_EPS = 1e-4


# Legacy WFDB-order angle table used by existing evaluation scripts.
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

CANONICAL_12LEAD_ANGLES_RAD = STANDARD_12LEAD_ANGLES_RAD[DISK_TO_CANONICAL].copy()


def normalize_fixed_train_bounds(waveforms: torch.Tensor) -> torch.Tensor:
    """Map physical mV to the decoder support using frozen PTB-XL train bounds."""
    normalized = (waveforms - TRAIN_LOWER_MV) / (TRAIN_UPPER_MV - TRAIN_LOWER_MV)
    return normalized.clamp(NORMALIZATION_EPS, 1.0 - NORMALIZATION_EPS)


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
        return crop[DISK_TO_CANONICAL]  # [12, target_len], canonical order


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

        crop = crop[DISK_TO_CANONICAL]

        # Select input slots: anchors I/II plus independent precordial leads only.
        if self.lead_cardinality is not None:
            num_additional = max(1, self.lead_cardinality - 2)
        else:
            # Dynamic cardinality: k in {1, 2, 3} -> L_input in {3, 4, 5}
            num_additional = int(rng.choice([1, 2, 3]))

        available_additional = list(PRECORDIAL_LEAD_INDICES)
        selected_additional = rng.choice(available_additional, size=num_additional, replace=False).tolist()
        input_indices = [0, 1] + sorted(selected_additional)

        # Select 1 target lead from remaining leads
        unused_leads = [i for i in PRECORDIAL_LEAD_INDICES if i not in input_indices]
        target_index = int(rng.choice(unused_leads))

        input_waveforms = crop[input_indices]  # [C_in, L]
        target_waveform = crop[target_index : target_index + 1]  # [1, L]

        input_norm = normalize_fixed_train_bounds(input_waveforms)
        target_norm = normalize_fixed_train_bounds(target_waveform)

        input_angles = torch.from_numpy(CANONICAL_12LEAD_ANGLES_RAD[input_indices])
        target_angle = torch.from_numpy(CANONICAL_12LEAD_ANGLES_RAD[target_index])

        return {
            "input": input_norm,                     # [lead_cardinality, target_len]
            "input_angles": input_angles,           # [lead_cardinality, 2]
            "target": target_norm,                   # [1, target_len]
            "target_angle": target_angle,           # [2]
            "input_indices": torch.tensor(input_indices, dtype=torch.long),
            "target_index": torch.tensor(target_index, dtype=torch.long),
        }
