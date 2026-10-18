"""Geometry-Aware SetOperator Architecture (NEF-Net Inspired Extension).

Preserves the complete original SetOperator pipeline:
- Authentic Phase-KME input (16x128)
- Authentic PhaseCNN response encoder (16x128 -> 128-D)
- Continuous operator representation (either raw 8-D, capacity control, or G1 gamma(q) in R^44)
- Transformer set encoder (either standard nn.TransformerEncoder or G2 GeometryTransformerEncoder)
- Learned-query cross-attention set pooling (192-D)
- Latent projection (192 -> 256) + Classification head (256 -> classes)
- Optional patient context / tabular fusion

Provides 5 strictly controlled configurations:
1. 'baseline': Original SetOperator (raw q in R^8 -> MLP(8->64->64), standard attention)
2. 'control_capacity': Capacity control (raw q in R^8 -> MLP(8->96->64), standard attention)
3. 'G1': Geometry-aware operator encoder (gamma(q) in R^44 -> MLP(44->64->64), standard attention)
4. 'G2': Geometry-aware relational attention (raw q in R^8 -> MLP(8->64->64), G2 attention with beta)
5. 'G1_G2': Combined geometry (gamma(q) in R^44 -> MLP(44->64->64), G2 attention with beta)
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, Literal

import torch
from torch import nn
from torch.nn import functional as F

from repecg.common.models import PhaseCNN
from repecg.geometry_setoperator.geometry_features import gamma, compute_operator_gram
from repecg.geometry_setoperator.geometry_attention import (
    GeometryTransformerEncoder,
    GeometryTransformerEncoderLayer,
)


class GeometryOperatorSetModel(nn.Module):
    """Geometry-Aware SetOperator Model."""

    def __init__(
        self,
        response_dim: int = 128,
        classes: int = 5,
        *,
        geometry_mode: Literal["baseline", "control_capacity", "G1", "G2", "G1_G2"] = "baseline",
        patient_context_dim: int = 0,
        operator_dim: int = 8,
    ):
        super().__init__()
        valid_modes = {"baseline", "control_capacity", "G1", "G2", "G1_G2"}
        if geometry_mode not in valid_modes:
            raise ValueError(f"geometry_mode must be one of {valid_modes}, got {geometry_mode}")

        self.geometry_mode = geometry_mode
        self.operator_dim = operator_dim
        self.response_dim = response_dim
        self.classes = classes
        self.patient_context_dim = patient_context_dim

        # 1. Authentic PhaseCNN response encoder (16x128 -> 128)
        self.response = PhaseCNN(response_dim, width=128, blocks=2, classes=128)

        # 2. Continuous Operator MLP based on geometry_mode
        if geometry_mode in {"baseline", "G2"}:
            # Original 8 -> 64 -> 64 (4,736 params)
            self.operator = nn.Sequential(nn.Linear(operator_dim, 64), nn.GELU(), nn.Linear(64, 64))
        elif geometry_mode == "control_capacity":
            # Capacity-matched control: 8 -> 96 -> 64 (7,072 params, matching G1 capacity with raw q)
            self.operator = nn.Sequential(nn.Linear(operator_dim, 96), nn.GELU(), nn.Linear(96, 64))
        elif geometry_mode in {"G1", "G1_G2"}:
            # G1 geometry-aware encoder: 44 -> 64 -> 64 (7,040 params)
            self.operator = nn.Sequential(nn.Linear(44, 64), nn.GELU(), nn.Linear(64, 64))

        # 3. Transformer Set Encoder (2 layers, 4 heads, d_model=192, d_ff=384)
        if geometry_mode in {"G2", "G1_G2"}:
            encoder_layer = GeometryTransformerEncoderLayer(
                d_model=192, nhead=4, dim_feedforward=384, activation="gelu"
            )
            self.set_encoder = GeometryTransformerEncoder(encoder_layer, num_layers=2)
        else:
            std_layer = nn.TransformerEncoderLayer(
                d_model=192, nhead=4, batch_first=True, dim_feedforward=384, activation="gelu"
            )
            self.set_encoder = nn.TransformerEncoder(std_layer, num_layers=2)

        # 4. Learned-query cross-attention pooling (d_model=192, 4 heads)
        self.pool_query = nn.Parameter(torch.zeros(1, 1, 192))
        self.pool = nn.MultiheadAttention(192, 4, batch_first=True)

        # 5. Diagnostic Latent & Head
        self.latent = nn.Sequential(nn.Linear(192, 256), nn.GELU())
        self.head = nn.Linear(256, classes)

        # 6. Optional Reconstruction Decoder (for multi-task compatibility with OperatorSetModel)
        self.decoder = nn.Sequential(
            nn.Linear(320, 256), nn.GELU(), nn.Linear(256, 16 * response_dim)
        )

        # 7. Optional Patient / Tabular Context
        self.patient_context = (
            nn.Sequential(nn.Linear(patient_context_dim, 64), nn.GELU(), nn.Linear(64, 256))
            if patient_context_dim > 0
            else None
        )


    def encode_operator(self, operators: torch.Tensor) -> torch.Tensor:
        """Encode operators into 64-D vectors.
        
        Args:
            operators: (batch, set_size, 8)
            
        Returns:
            (batch, set_size, 64)
        """
        if self.geometry_mode in {"G1", "G1_G2"}:
            # Feature expansion to 44 dimensions
            q_geom = gamma(operators)
            return self.operator(q_geom)
        else:
            # Raw operator directly into MLP
            return self.operator(operators)

    def encode_context(
        self,
        operators: torch.Tensor,
        responses: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        patient_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode operator-response pairs into patient latent vector."""
        if responses.ndim != 4:
            raise ValueError("responses must have shape (batch, set, phase, feature)")
        batch, count = responses.shape[:2]

        # 1. Encode operators: (batch, count, 64)
        q = self.encode_operator(operators)

        # 2. Encode responses: (batch, count, 128)
        resp_encoded = self.response(
            responses.reshape(batch * count, responses.shape[2], responses.shape[3])
        ).reshape(batch, count, 128)

        # 3. Concatenate into measurement token: (batch, count, 192)
        tokens = torch.cat((q, resp_encoded), dim=-1)

        # 4. Transformer Set Encoder
        if self.geometry_mode in {"G2", "G1_G2"}:
            # Compute operator Gram matrix and relational alignment R
            _, R = compute_operator_gram(operators, mask=mask)
            encoded = self.set_encoder(tokens, src_key_padding_mask=mask, R=R)
        else:
            encoded = self.set_encoder(tokens, src_key_padding_mask=mask)

        # 5. Learned-query cross-attention pooling
        query = self.pool_query.expand(batch, -1, -1)
        pooled, _ = self.pool(query, encoded, encoded, key_padding_mask=mask)

        # 6. Latent projection: (batch, 256)
        latent = self.latent(pooled[:, 0])

        if self.patient_context is not None:
            if patient_context is None:
                raise ValueError("patient_context is required for this model")
            latent = latent + self.patient_context(patient_context)

        return latent

    def forward(
        self,
        operators: torch.Tensor,
        responses: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        patient_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        latent = self.encode_context(operators, responses, mask=mask, patient_context=patient_context)
        return self.head(latent)

    def load_from_original_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = False) -> None:
        """Load weights from a standard OperatorSetModel checkpoint.
        
        Leaves G2 beta parameters at their 0.0 initialization for exact numerical parity.
        """
        current_state = self.state_dict()
        compatible_state = {}
        for k, v in state_dict.items():
            if k in current_state:
                if current_state[k].shape == v.shape:
                    compatible_state[k] = v
                else:
                    # Parameter shape mismatch (e.g. operator.0.weight between baseline and G1)
                    if strict:
                        raise RuntimeError(f"Shape mismatch for key {k}: {current_state[k].shape} vs {v.shape}")
        self.load_state_dict(compatible_state, strict=False)


class EchoNextGeometrySetOperator(nn.Module):
    """EchoNext Classifier Wrapper for Geometry-Aware SetOperator."""

    def __init__(
        self,
        geometry_mode: Literal["baseline", "control_capacity", "G1", "G2", "G1_G2"] = "baseline",
        response_dim: int = 128,
        operator_dim: int = 8,
        tabular_dim: int = 7,
    ):
        super().__init__()
        self.core = GeometryOperatorSetModel(
            response_dim=response_dim,
            classes=1,
            geometry_mode=geometry_mode,
            operator_dim=operator_dim,
            patient_context_dim=tabular_dim,
        )

    def forward(
        self,
        operators: torch.Tensor,
        responses: torch.Tensor,
        tabular: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        latent = self.core.encode_context(operators, responses, mask=mask, patient_context=tabular)
        logits = self.core.head(latent)
        return {"logits": logits, "latent": latent}
