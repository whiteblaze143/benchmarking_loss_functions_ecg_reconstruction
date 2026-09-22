#!/usr/bin/env python3
"""Extract Fold-8 latents from the four frozen PTB-XL encoders.

This script never updates encoder parameters.  It uses the original GraphECG
per-lead z-scoring and the same Q8 independent-lead ordering used by the
existing configuration-shift runner.  The output is an input to the figure
scripts in this directory, not a new performance evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wfdb
from torch_geometric.data import Batch

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1]
REPO = HERE.parents[3]
sys.path[:0] = [str(EXP), str(EXP / "src")]

from graphECG_author_code.graph import ECGGraphBuilder
from graphECG_author_code.model import GraphECG
from repecg.paper07_operator import OperatorSetModel


CHECKPOINTS = {
    "GraphECG": EXP / "outputs/graphecg/graphecg_ptbxl_best.pt",
    "SetOp robust": EXP / "outputs/paper07_operator_reconstruction/continuous_primary_best.pt",
    "SetOp robust + aux": EXP / "outputs/paper07_operator_reconstruction/continuous_auxiliary_best.pt",
    "SetOp full-lead only": EXP / "outputs/paper07_fulllead_only/continuous_primary_best.pt",
}
Q8_INDICES = [0, 1, 6, 7, 8, 9, 10, 11]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class RKHSFeatureExtractor:
    def __init__(self, path: Path, device: torch.device):
        if not path.is_file():
            raise FileNotFoundError(path)
        item = np.load(path)
        self.mean = torch.as_tensor(item["whitening_mean"], device=device, dtype=torch.float32)
        self.components = torch.as_tensor(item["whitening_components"], device=device, dtype=torch.float32)
        self.scales = torch.as_tensor(item["whitening_scales"], device=device, dtype=torch.float32)
        self.landmarks = torch.as_tensor(item["landmarks"], device=device, dtype=torch.float32)
        self.inverse_root = torch.as_tensor(item["inverse_root"], device=device, dtype=torch.float32)
        self.c2 = float(item["c2"])
        self.voltage_scale = float(item["voltage_scale"])

    def compute(self, waveforms: torch.Tensor) -> torch.Tensor:
        # Matches the frozen task-native SetOperator extraction contract.
        y = waveforms / self.voltage_scale
        if y.shape[-1] != 256:
            y = F.interpolate(y, size=256, mode="linear", align_corners=False)
        dy = torch.gradient(y, dim=-1)[0]
        atoms = torch.stack([y, dy], dim=-1).reshape(*y.shape[:2], 16, 16, 2)
        white = (atoms - self.mean) @ self.components * self.scales
        distances = torch.cdist(white.reshape(-1, 2), self.landmarks)
        rbf = torch.rsqrt(distances.square() + self.c2) @ self.inverse_root
        return rbf.reshape(*y.shape[:2], 16, 16, 128).mean(dim=3)


def load_fold8(ptb_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    table = pd.read_csv(ptb_root / "ptbxl_database.csv", index_col="ecg_id")
    table = table.loc[table["strat_fold"] == 8]
    signals = []
    for _, row in table.iterrows():
        waveform, _ = wfdb.rdsamp(str(ptb_root / row["filename_lr"]))
        waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        waveform = (waveform - waveform.mean(0, keepdims=True)) / (waveform.std(0, keepdims=True) + 1e-6)
        signals.append(waveform.T)
    reference = np.load(EXP / "outputs/ptbxl_masking_control/representation_ptbxl_full_fold8.npz")
    ecg_ids = table.index.to_numpy(dtype=np.int64)
    if not np.array_equal(ecg_ids, reference["ecg_ids"]):
        raise ValueError("Fold-8 waveform order does not match the frozen reference labels")
    return np.stack(signals), reference["labels"], ecg_ids, reference["patient_ids"]


@torch.no_grad()
def graph_latents(model: GraphECG, signals: np.ndarray, device: torch.device) -> np.ndarray:
    builder = ECGGraphBuilder()
    chunks = []
    for start in range(0, len(signals), 32):
        graphs = [builder.build_from_array(x, lead_indices=list(range(12)), bidirectional=True)
                  for x in signals[start:start + 32]]
        chunks.append(model(Batch.from_data_list(graphs).to(device))["embedding"].cpu().numpy())
    return np.concatenate(chunks)


@torch.no_grad()
def set_latents(model: OperatorSetModel, extractor: RKHSFeatureExtractor,
                signals: np.ndarray, device: torch.device) -> np.ndarray:
    canonical_ops = torch.eye(8, device=device).unsqueeze(0)
    chunks = []
    for start in range(0, len(signals), 64):
        waveforms = torch.from_numpy(signals[start:start + 64, Q8_INDICES]).to(device)
        operators = canonical_ops.expand(len(waveforms), -1, -1)
        chunks.append(model.encode_context(operators, extractor.compute(waveforms)).cpu().numpy())
    return np.concatenate(chunks)


def fixed_tensor_latents(signals: np.ndarray) -> np.ndarray:
    """Frozen FixedTensor contract: 12 lead-wise mean/std/min/max features."""
    return np.concatenate(
        [signals.mean(axis=-1), signals.std(axis=-1), signals.min(axis=-1), signals.max(axis=-1)],
        axis=-1,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--ptb-root", type=Path, default=REPO / "data/ptbxl")
    parser.add_argument("--output", type=Path, default=HERE / "frozen_fold8_latents.npz")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    signals, labels, ecg_ids, patient_ids = load_fold8(args.ptb_root)
    graph_state = torch.load(CHECKPOINTS["GraphECG"], map_location=device, weights_only=False)
    graph = GraphECG(node_dim=128, edge_dim=192, hidden_dim=192, num_layers=3, tabular_dim=0, num_classes=5).to(device).eval()
    graph.load_state_dict(graph_state["model_state_dict"])
    rkhs = RKHSFeatureExtractor(Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/paper07_operator/development_representations/response_fit.npz"), device)
    latents = {
        "graphecg": graph_latents(graph, signals, device),
        "fixed_tensor": fixed_tensor_latents(signals),
    }
    del graph
    for key, stem in (("SetOp robust", "setop_robust"), ("SetOp robust + aux", "setop_robust_aux"), ("SetOp full-lead only", "setop_fulllead")):
        state = torch.load(CHECKPOINTS[key], map_location=device, weights_only=False)
        model = OperatorSetModel(response_dim=128, classes=5, operator_mode="continuous").to(device).eval()
        model.load_state_dict(state["state_dict"])
        latents[stem] = set_latents(model, rkhs, signals, device)
        del model

    np.savez_compressed(args.output, labels=labels, ecg_ids=ecg_ids, patient_ids=patient_ids, **latents)
    manifest = {
        "protocol": "Frozen encoders; Fold 8 only; Q8/full-lead input; no parameter updates.",
        "n_records": int(len(ecg_ids)),
        "checkpoint_sha256": {name: digest(path) for name, path in CHECKPOINTS.items()},
        "latent_dimensions": {name: int(value.shape[1]) for name, value in latents.items()},
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
