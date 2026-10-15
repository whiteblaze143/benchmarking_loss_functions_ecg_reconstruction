#!/usr/bin/env python3
"""Fast GPU wiring test for GraphECG on PTB-XL."""

import sys
from pathlib import Path
import torch
import numpy as np

REPO_DIR = Path(__file__).resolve().parents[4]
EXPERIMENT_DIR = REPO_DIR / "experiments/ptbxl_distributional_repecg"
sys.path.insert(0, str(EXPERIMENT_DIR / "src"))

from graphECG_author_code.graph import ECGGraphBuilder
from graphECG_author_code.model import GraphECG
from torch_geometric.data import Batch

def main():
    print("Testing GraphECG instantiation and forward/backward on GPU...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "CUDA required"

    builder = ECGGraphBuilder()
    model = GraphECG(
        node_dim=128,
        edge_dim=192,
        hidden_dim=192,
        num_layers=3,
        tabular_dim=0,
        num_classes=5,
        dropout=0.3,
    ).to(device)

    # 4 synthetic ECGs of shape (12, 1000)
    batch_size = 4
    signals = [np.random.randn(12, 1000).astype(np.float32) for _ in range(batch_size)]
    graphs = [builder.build_from_array(s) for s in signals]
    batch_graph = Batch.from_data_list(graphs).to(device)
    targets = (torch.rand(batch_size, 5, device=device) > 0.7).float()

    model.train()
    logits = model(batch_graph)
    assert logits.shape == (batch_size, 5), f"Shape mismatch: {logits.shape}"

    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    assert torch.isfinite(grad_norm), "Non-finite gradient"
    print(f"  Passed GraphECG: loss={loss.item():.4f}, grad_norm={grad_norm.item():.4f}")
    print("ALL GRAPHECG WIRING TESTS PASSED!")

if __name__ == "__main__":
    main()
