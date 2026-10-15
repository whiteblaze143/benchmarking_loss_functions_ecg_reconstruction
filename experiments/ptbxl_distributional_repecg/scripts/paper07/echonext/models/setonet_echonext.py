"""SetONet model adaptation for EchoNext Binary SHD Detection.

Preserves the branch-trunk operator factorization from upstream SetONet:
- Branch: processes variable operator-response sets {(q_i, u_i)} using Deep Sets / Attention pooling
- Trunk: evaluates a binary diagnostic query y in R^d (optionally conditioned on 7 clinical tabular features)
- Synthesis: scalar product <branch(u), trunk(y)> + b -> binary classification logit
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, Any, Optional, Literal

import torch
import torch.nn as nn

REPO_ROOT = Path("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction")
SETONET_DIR = REPO_ROOT / "external/SetONet"
if str(SETONET_DIR) not in sys.path:
    sys.path.insert(0, str(SETONET_DIR))

from Models.SetONet import SetONet
from repecg.common.models import PhaseCNN


class SetONetEchoNext(nn.Module):
    """SetONet adapted for Binary Structural Heart Disease classification."""

    def __init__(
        self,
        operator_dim: int = 8,
        response_dim: int = 128,
        tabular_dim: int = 7,
        p: int = 128,
        phi_hidden_size: int = 256,
        rho_hidden_size: int = 256,
        trunk_hidden_size: int = 256,
        n_trunk_layers: int = 3,
        use_phase_cnn: bool = True,
        tabular_mode: Literal["none", "trunk_query", "late_concat"] = "trunk_query",
    ):
        super().__init__()
        self.operator_dim = operator_dim
        self.response_dim = response_dim
        self.tabular_dim = tabular_dim
        self.tabular_mode = tabular_mode
        self.use_phase_cnn = use_phase_cnn

        # Response encoder: converts (16, 128) Phase-KME to 128-D response vector
        if use_phase_cnn:
            self.response_encoder = PhaseCNN(response_dim, width=128, blocks=2, classes=128)
            u_dim = 128
        else:
            self.response_encoder = None
            u_dim = response_dim

        # Trunk query dimensionality
        query_token_dim = 16
        self.query_token = nn.Parameter(torch.randn(1, 1, query_token_dim))

        if tabular_mode == "trunk_query":
            trunk_input_dim = query_token_dim + tabular_dim
        else:
            trunk_input_dim = query_token_dim

        # Upstream SetONet
        self.setonet = SetONet(
            input_size_src=operator_dim,
            output_size_src=u_dim,
            input_size_tgt=trunk_input_dim,
            output_size_tgt=1,
            p=p,
            phi_hidden_size=phi_hidden_size,
            rho_hidden_size=rho_hidden_size,
            trunk_hidden_size=trunk_hidden_size,
            n_trunk_layers=n_trunk_layers,
            activation_fn=nn.GELU,
            aggregation_type="attention",
            use_positional_encoding=False,
            pos_encoding_type="skip",
            branch_head_type="standard",
        )

        if tabular_mode == "late_concat":
            # Additional adapter for late concatenation comparison
            self.fusion_head = nn.Linear(1 + tabular_dim, 1)
        else:
            self.fusion_head = None

    def forward(
        self,
        operators: torch.Tensor,
        responses: torch.Tensor,
        tabular: Optional[torch.Tensor] = None,
        sensor_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            operators: (batch, n_sensors, 8)
            responses: (batch, n_sensors, 16, 128) if using PhaseCNN or (batch, n_sensors, u_dim)
            tabular: (batch, 7) optional clinical features
            sensor_mask: (batch, n_sensors) bool mask, True for valid sensors

        Returns:
            Dict with 'logits' (batch, 1)
        """
        batch_size, n_sensors = operators.shape[:2]

        if self.use_phase_cnn:
            # responses shape: (batch, n_sensors, 16, 128)
            u_flat = responses.reshape(batch_size * n_sensors, responses.shape[2], responses.shape[3])
            us = self.response_encoder(u_flat).reshape(batch_size, n_sensors, -1)
        else:
            us = responses

        # Construct trunk query
        query = self.query_token.expand(batch_size, -1, -1)  # (batch, 1, query_token_dim)
        if self.tabular_mode == "trunk_query":
            assert tabular is not None, "tabular features required for trunk_query mode"
            tab_query = tabular.unsqueeze(1)  # (batch, 1, 7)
            ys = torch.cat([query, tab_query], dim=-1)  # (batch, 1, query_token_dim + 7)
        else:
            ys = query

        # SetONet forward: returns (batch, 1, 1)
        out = self.setonet(operators, us, ys, sensor_mask=sensor_mask)
        logits = out.squeeze(1)  # (batch, 1)

        if self.tabular_mode == "late_concat":
            assert tabular is not None, "tabular features required for late_concat mode"
            fused = torch.cat([logits, tabular], dim=-1)
            logits = self.fusion_head(fused)

        return {"logits": logits}


def build_setonet_echonext(
    tabular_condition: Literal["SO-ECG", "SO-ECG+Tab-Trunk", "SO-ECG+Tab-Concat"] = "SO-ECG+Tab-Trunk",
) -> SetONetEchoNext:
    """Factory function for SetONet EchoNext configurations."""
    if tabular_condition == "SO-ECG":
        return SetONetEchoNext(tabular_dim=0, tabular_mode="none")
    elif tabular_condition == "SO-ECG+Tab-Trunk":
        return SetONetEchoNext(tabular_dim=7, tabular_mode="trunk_query")
    elif tabular_condition == "SO-ECG+Tab-Concat":
        return SetONetEchoNext(tabular_dim=7, tabular_mode="late_concat")
    else:
        raise ValueError(f"Unknown tabular condition: {tabular_condition}")
