"""SetOperator model wrapper and adaptations for EchoNext Binary SHD Detection.

Preserves the frozen SetOperator core architecture:
- OperatorSetModel(response_dim=128, classes=1, operator_mode="continuous")
- PhaseCNN response encoder (16x128 -> 128-D)
- Continuous operator encoder (8-D cardiac basis -> 64-D)
- Transformer Set Encoder (d_model=192, 2 layers) + Cross-Attention Set Pooling (192-D) -> Latent (256-D)
- Head: Linear(256, 1) binary logit -> Sigmoid

Two explicit tabular conditions:
1. SO-ECG: ECG waveforms and measurement operators only (no tabular features).
2. SO-ECG+Tab: Same ECG pathway plus all 7 clinical features:
   - Native Additive Latent Context: latent + MLP(tabular) -> head
   - Late Concatenation Adapter: cat(latent, tabular) -> Linear(256 + 7, 1)
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, Any, Optional, Tuple, Literal

import torch
import torch.nn as nn
from torch.nn import functional as F

REPO_ROOT = Path("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction")
SRC_DIR = REPO_ROOT / "experiments/ptbxl_distributional_repecg/src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from repecg.paper07_operator.model import OperatorSetModel


class SetOperatorEchoNext(nn.Module):
    """EchoNext Binary Classifier based on the frozen SetOperator architecture."""

    def __init__(
        self,
        response_dim: int = 128,
        operator_mode: str = "continuous",
        operator_dim: int = 8,
        tabular_dim: int = 7,
        tabular_mode: Literal["none", "native_additive", "late_concat"] = "native_additive",
    ):
        super().__init__()
        self.tabular_mode = tabular_mode
        self.tabular_dim = tabular_dim

        # Native patient context in OperatorSetModel is used if tabular_mode == "native_additive"
        patient_context_dim = tabular_dim if tabular_mode == "native_additive" else 0

        self.core = OperatorSetModel(
            response_dim=response_dim,
            classes=1,
            operator_mode=operator_mode,
            operator_dim=operator_dim,
            patient_context_dim=patient_context_dim,
        )

        if tabular_mode == "late_concat":
            # Late concatenation fusion like GraphECG
            self.fusion_head = nn.Linear(256 + tabular_dim, 1)
        else:
            self.fusion_head = None

    def forward(
        self,
        operators: torch.Tensor,
        responses: torch.Tensor,
        tabular: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            operators: (batch, set_size, operator_dim)
            responses: (batch, set_size, 16, response_dim)
            tabular: (batch, 7) optional clinical features
            mask: (batch, set_size) padding mask (True for padded/missing tokens)

        Returns:
            Dict containing 'logits' (batch, 1) and 'latent' (batch, 256)
        """
        if self.tabular_mode == "native_additive":
            assert tabular is not None, "tabular features required for native_additive mode"
            latent = self.core.encode_context(operators, responses, mask=mask, patient_context=tabular)
            logits = self.core.head(latent)
        elif self.tabular_mode == "late_concat":
            assert tabular is not None, "tabular features required for late_concat mode"
            latent = self.core.encode_context(operators, responses, mask=mask, patient_context=None)
            fused = torch.cat([latent, tabular], dim=-1)
            logits = self.fusion_head(fused)
        else:
            # SO-ECG only
            latent = self.core.encode_context(operators, responses, mask=mask, patient_context=None)
            logits = self.core.head(latent)

        return {"logits": logits, "latent": latent}


def build_setoperator_echonext(
    tabular_condition: Literal["SO-ECG", "SO-ECG+Tab-Native", "SO-ECG+Tab-Concat"] = "SO-ECG+Tab-Native",
) -> SetOperatorEchoNext:
    """Factory function for SetOperator EchoNext configurations."""
    if tabular_condition == "SO-ECG":
        return SetOperatorEchoNext(tabular_dim=0, tabular_mode="none")
    elif tabular_condition == "SO-ECG+Tab-Native":
        return SetOperatorEchoNext(tabular_dim=7, tabular_mode="native_additive")
    elif tabular_condition == "SO-ECG+Tab-Concat":
        return SetOperatorEchoNext(tabular_dim=7, tabular_mode="late_concat")
    else:
        raise ValueError(f"Unknown tabular condition: {tabular_condition}")
