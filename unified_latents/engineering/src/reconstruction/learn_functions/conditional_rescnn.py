"""Conditional Residual CNN with FiLM modulation for ECG reconstruction."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch import nn

from src.reconstruction.learn_functions.conditioning import ConditioningEmbedding
from src.reconstruction.learn_functions.film_layer import FiLMModulation


DEFAULT_CONTINUOUS_KEYS = [
    "age_years",
    "sampling_rate_hz",
    "lowpass_hz",
    "hipass_hz",
    "qrs_duration_ms",
    "qt_interval_ms",
    "pr_interval_ms",
    "qrs_axis_deg",
]


class ResidualBlock(nn.Module):
    """Residual block with FiLM modulation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        conditioning_dim: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.film = FiLMModulation(out_channels, conditioning_dim)

        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.skip = nn.Identity()

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")
        if self.conv1.bias is not None:
            nn.init.zeros_(self.conv1.bias)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.film(out, conditioning)
        out = out + identity
        out = self.relu(out)
        return out


class ConditionalResCNN(nn.Module):
    """Residual CNN conditioned on metadata embeddings."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 9,
        num_layers: int = 4,
        conditioning_dim: int = 64,
        channels: Optional[List[int]] = None,
        kernel_size: int = 7,
        scaling_params_path: Optional[str] = None,
        predict_uncertainty: bool = False,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conditioning_dim = conditioning_dim
        self.predict_uncertainty = predict_uncertainty

        if channels is None:
            channels = [64, 128, 128, 128][:num_layers]
        if channels[0] < conditioning_dim // 2:
            raise ValueError("Initial channel width too small for conditioning dimension.")

        scaling_path = Path(scaling_params_path) if scaling_params_path else None
        self.conditioning = ConditioningEmbedding(
            embedding_dim=conditioning_dim,
            scaling_params_path=scaling_path,
            continuous_keys=DEFAULT_CONTINUOUS_KEYS,
        )

        self.input_conv = nn.Sequential(
            nn.Conv1d(in_channels, channels[0], kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
        )

        blocks: List[nn.Module] = []
        in_ch = channels[0]
        for out_ch in channels:
            blocks.append(ResidualBlock(in_ch, out_ch, kernel_size, conditioning_dim, stride=1))
            in_ch = out_ch

        self.blocks = nn.ModuleList(blocks)
        self.output_conv = nn.Conv1d(in_ch, out_channels, kernel_size=1)
        
        if self.predict_uncertainty:
            # Log-variance head: same structure as output head
            self.logvar_conv = nn.Conv1d(in_ch, out_channels, kernel_size=1)
            
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_normal_(self.output_conv.weight, nonlinearity="linear")
        if self.output_conv.bias is not None:
            nn.init.zeros_(self.output_conv.bias)
            
        if self.predict_uncertainty:
            nn.init.kaiming_normal_(self.logvar_conv.weight, nonlinearity="linear")
            if self.logvar_conv.bias is not None:
                nn.init.zeros_(self.logvar_conv.bias)

    def forward(self, inputs: torch.Tensor, metadata: Dict[str, torch.Tensor]) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        conditioning = self.conditioning(metadata)
        out = self.input_conv(inputs)
        for block in self.blocks:
            out = block(out, conditioning)
            
        mu = self.output_conv(out)
        
        if self.predict_uncertainty:
            logvar = self.logvar_conv(out)
            return mu, logvar
            
        return mu

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":  # pragma: no cover
    import json

    batch_size = 4
    seq_len = 5000
    model = ConditionalResCNN()
    x = torch.randn(batch_size, 3, seq_len)
    metadata = {
        "age_years": torch.zeros(batch_size),
        "is_missing_age_years": torch.ones(batch_size),
        "sex_code": torch.zeros(batch_size, dtype=torch.long),
        "pacemaker_status": torch.zeros(batch_size, dtype=torch.long),
    }
    y = model(x, metadata)
    print("Output shape:", y.shape)
    print("Parameters:", model.parameter_count())

