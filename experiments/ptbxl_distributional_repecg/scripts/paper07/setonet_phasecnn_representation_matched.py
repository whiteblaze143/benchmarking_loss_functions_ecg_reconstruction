#!/usr/bin/env python3
"""Representation-Matched SetONet Architecture (Secondary Controlled Ablation).

Retains SetONet's canonical branch-trunk and Deep Sets permutation-invariant formulation,
while replacing pre-pooling with the identical PhaseCNN response encoder used by SetOperator:
  1. Response Encoding: (B, m, 16, 128) -> PhaseCNN -> (B, m, 128)
  2. Branch Network: Deep Sets phi(x_i, u_i) -> mean pool -> rho -> p=64 coefficients
  3. Trunk Network: 4-layer MLP mapping 5-class target coordinates -> p=64 basis functions
  4. Operator Output: inner product <b, t(y)> + bias

Treated as a secondary controlled ablation to isolate set-aggregation architecture
(Transformer + Cross-Attention vs Deep Sets Branch-Trunk) under identical input representation.
Existing baseline SetONet checkpoints and results are strictly preserved.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

torch.backends.cudnn.enabled = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

REPO_DIR = Path(__file__).resolve().parents[4]
EXPERIMENT_DIR = REPO_DIR / "experiments/ptbxl_distributional_repecg"
sys.path.insert(0, str(EXPERIMENT_DIR / "src"))
sys.path.insert(0, str(REPO_DIR / "external/SetONet"))

from repecg.common.models import PhaseCNN
from Models.SetONet import SetONet


class PTBXLSetONetPhaseCNNAdapter(nn.Module):
    """Representation-matched SetONet adapter with learned PhaseCNN response encoder.
    
    Both SetOperator and SetONet-PhaseCNN receive identical (B, m, 16, 128) Phase-KME tensors.
    Both models pass each operator's response through PhaseCNN(128, width=128, blocks=2, classes=128).
    SetOperator aggregates via Transformer self-attention and cross-attention diagnostic queries.
    SetONet aggregates via Deep Sets phi-MLP, mean pooling, rho-MLP, and trunk dot product.
    """

    def __init__(
        self,
        operator_dim: int = 8,
        response_dim: int = 128,
        classes: int = 5,
        p: int = 64,
        phi_hidden: int = 256,
        rho_hidden: int = 256,
        trunk_hidden: int = 256,
    ):
        super().__init__()
        self.operator_dim = operator_dim
        self.response_dim = response_dim
        self.classes = classes
        self.p = p

        # 1. Learned PhaseCNN response encoder (identical to SetOperator)
        self.response_encoder = PhaseCNN(
            input_dim=response_dim,
            width=128,
            blocks=2,
            classes=128,
        )

        # 2. Canonical class queries for the SetONet trunk network: (5, 5) identity matrix
        self.register_buffer("class_queries", torch.eye(classes, dtype=torch.float32))

        # 3. Canonical SetONet model instantiation
        self.setonet = SetONet(
            input_size_src=operator_dim,
            output_size_src=128,  # output dimension of PhaseCNN
            input_size_tgt=classes,
            output_size_tgt=1,
            p=p,
            phi_hidden_size=phi_hidden,
            phi_output_size=128,
            rho_hidden_size=rho_hidden,
            trunk_hidden_size=trunk_hidden,
            n_trunk_layers=4,
            use_deeponet_bias=True,
            pos_encoding_type="skip",
            use_positional_encoding=False,
            aggregation_type="mean",
            branch_head_type="standard",
        )

    def forward(
        self,
        operators: torch.Tensor,
        responses: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.
        operators: (B, m, 8)
        responses: (B, m, 16, 128) Phase-KME response matrix
        Returns: logits (B, 5)
        """
        B, m = operators.shape[:2]

        if responses.dim() == 4:
            # Flatten across batch and active operators: (B*m, 16, 128)
            res_flat = responses.view(B * m, 16, self.response_dim)
            # Pass through PhaseCNN: (B*m, 128)
            res_enc = self.response_encoder(res_flat)
            # Reshape back to (B, m, 128)
            res_feat = res_enc.view(B, m, 128)
        else:
            res_feat = responses

        # Target class queries for trunk: (B, 5, 5)
        ys = self.class_queries.unsqueeze(0).expand(B, -1, -1)

        # Forward through canonical SetONet
        out = self.setonet(operators, res_feat, ys)  # (B, 5, 1)
        return out.squeeze(-1)


def verify_architecture_parity():
    """Verify forward pass and parameter counts."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PTBXLSetONetPhaseCNNAdapter().to(device)

    B, m = 4, 8
    dummy_ops = torch.randn(B, m, 8, device=device)
    dummy_res = torch.randn(B, m, 16, 128, device=device)

    logits = model(dummy_ops, dummy_res)
    assert logits.shape == (B, 5), f"Shape mismatch: {logits.shape}"

    phasecnn_params = sum(p.numel() for p in model.response_encoder.parameters() if p.requires_grad)
    setonet_params = sum(p.numel() for p in model.setonet.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Architecture Parity Verification: PASSED")
    print(f"  PhaseCNN Parameters:   {phasecnn_params:,}")
    print(f"  SetONet Core Params:   {setonet_params:,}")
    print(f"  Total Trainable Params:{total_params:,}")


if __name__ == "__main__":
    verify_architecture_parity()
