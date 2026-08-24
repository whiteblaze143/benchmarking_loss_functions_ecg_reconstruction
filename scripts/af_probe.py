"""AFProbe-v1 model, PTB-XL dataset, and frozen-checkpoint helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
import wfdb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.af_protocol import LEADS, parse_scp_codes, ptbxl_af_label, sha256_file

PTB_ROOT = ROOT / "data/ptb_xl"
PTB_META = PTB_ROOT / "ptbxl_database.csv"
TRAIN_CONDITIONS = ("real12", "source_I", "source_II", "real11_I", "real11_II")


def availability(condition: str) -> np.ndarray:
    mask = np.ones(12, dtype=np.float32)
    if condition == "source_I": mask[1:] = 0
    elif condition == "source_II": mask[:] = 0; mask[1] = 1
    elif condition == "real11_I": mask[0] = 0
    elif condition == "real11_II": mask[1] = 0
    elif condition != "real12": raise ValueError(condition)
    return mask


def preprocess(signal: np.ndarray, mask: np.ndarray) -> torch.Tensor:
    if signal.shape[1] != 12:
        raise ValueError(f"expected 12 leads, got {signal.shape}")
    x = signal.T.astype(np.float32, copy=True)
    for lead in range(12):
        if mask[lead]:
            x[lead] = (x[lead] - x[lead].mean()) / max(float(x[lead].std()), 1e-6)
        else:
            x[lead] = 0
    mask_channels = np.repeat(mask[:, None], x.shape[1], axis=1)
    return torch.from_numpy(np.concatenate([x, mask_channels], axis=0))


def load_rows(folds: set[int]) -> pd.DataFrame:
    frame = pd.read_csv(PTB_META, index_col="ecg_id")
    frame = frame[frame.strat_fold.isin(folds)].copy()
    labels = frame.scp_codes.map(lambda value: ptbxl_af_label(parse_scp_codes(value))[0])
    frame["af_label"] = labels
    return frame[frame.af_label.notna()]


class PTBAFDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, conditions: tuple[str, ...], expand: bool = True):
        self.rows, self.conditions, self.expand = rows, conditions, expand
    def __len__(self): return len(self.rows) * (len(self.conditions) if self.expand else 1)
    def __getitem__(self, index):
        condition = self.conditions[index % len(self.conditions)] if self.expand else self.conditions[0]
        row = self.rows.iloc[index // len(self.conditions) if self.expand else index]
        signal, fields = wfdb.rdsamp(str(PTB_ROOT / row.filename_hr))
        if int(fields["fs"]) != 500: raise RuntimeError("AFProbe-v1 requires 500 Hz")
        return preprocess(signal, availability(condition)), torch.tensor(float(row.af_label)), int(row.name), condition


class AFProbeV1(nn.Module):
    def __init__(self, width: int = 64):
        super().__init__()
        layers = []
        channels = 24
        for out, kernel, stride in [(width,15,2),(width*2,9,2),(width*2,7,2),(width*4,5,2),(width*4,5,2)]:
            layers += [nn.Conv1d(channels,out,kernel,stride,padding=kernel//2,bias=False), nn.BatchNorm1d(out), nn.GELU()]
            channels = out
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(channels, 1)
    def forward(self, x): return self.head(self.encoder(x).mean(-1)).squeeze(-1)


def checkpoint_contract() -> dict:
    return {"name":"AFProbe-v1", "sample_rate":500, "leads":list(LEADS),
            "input_channels":24, "train_folds":list(range(1,9)),
            "train_conditions":list(TRAIN_CONDITIONS), "real_ecg_only":True,
            "ptb_metadata_sha256":sha256_file(PTB_META)}


def load_checkpoint(path: Path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("contract") != checkpoint_contract(): raise RuntimeError("AFProbe checkpoint contract mismatch")
    model = AFProbeV1(payload["width"]); model.load_state_dict(payload["model"]); model.to(device).eval()
    return model, payload
