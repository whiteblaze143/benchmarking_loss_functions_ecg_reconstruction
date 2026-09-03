import math
import torch
import torch.nn as nn


class SinusoidalPosition1D(nn.Module):
    """
    Standard 1D Sinusoidal Positional Encoding for temporal latent tokens.
    """

    def __init__(
        self,
        channels: int,
        max_len: int = 2048,
    ):
        super().__init__()
        position = torch.arange(max_len).float()[:, None]
        div = torch.exp(
            torch.arange(0, channels, 2).float() * (-math.log(10000.0) / channels)
        )
        pe = torch.zeros(max_len, channels)
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe[None])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [B, T, C]
        return x + self.pe[:, : x.shape[1]]


class QTTemporalHead(nn.Module):
    """
    Transformer-based QT regression branch (Ansari et al., 2026).
    
    Status: QT_TEMPORAL_TRANSFORMER_APPROX
    Captures temporal relationships across cardiac cycles from the shared latent Z
    without target-angle conditioning.
    """

    def __init__(
        self,
        channels: int = 256,
        depth: int = 4,
        heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.position = SinusoidalPosition1D(channels)

        layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=heads,
            dim_feedforward=4 * channels,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=depth,
            norm=nn.LayerNorm(channels),
        )

        self.regressor = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels // 2),
            nn.Mish(),
            nn.Linear(channels // 2, 1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        # [B, C, T] -> [B, T, C]
        tokens = latent.transpose(1, 2)
        tokens = self.position(tokens)
        tokens = self.transformer(tokens)
        # Global temporal average pooling
        pooled = tokens.mean(dim=1)
        # [B] scalar QT prediction in ms
        return self.regressor(pooled).squeeze(-1)
