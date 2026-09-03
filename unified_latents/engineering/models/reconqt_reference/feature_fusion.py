import torch
import torch.nn as nn


class ConvFeatureBlock(nn.Module):
    """
    Standard 1D Convolutional Feature Block: Conv1D -> BatchNorm1D -> ReLU.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalAttention1D(nn.Module):
    """
    Multi-Head Temporal Self-Attention followed by LayerNorm as visually confirmed
    in 3DRECON-QT Fig. 2 on the Z2 feature stream.
    """

    def __init__(
        self,
        channels: int,
        heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert channels % heads == 0, f"channels ({channels}) must be divisible by heads ({heads})"
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, T] -> [B, T, C]
        tokens = x.transpose(1, 2)
        attended, _ = self.attn(
            tokens,
            tokens,
            tokens,
            need_weights=False,
        )
        # Fig. 2 explicitly shows attention output passing directly into LayerNorm
        tokens = self.norm(attended)
        # [B, T, C] -> [B, C, T]
        return tokens.transpose(1, 2)


class FeatureExtractionFusion(nn.Module):
    r"""
    Feature Extraction and Z1/Z2 Fusion block (Ansari et al., 2026 Fig. 2).
    
    Path:
      SE-ResNeXt latent (2048 ch)
             |
          Conv1D (256 ch)
             |
           Chunk
         /       \
     Z1 Conv   Z2 Conv
        |         |
        |    Temporal Attention
        |         |
        |     LayerNorm
         \       /
        Concatenate
             |
      Shared Latent Z (256 ch)
    """

    def __init__(
        self,
        encoder_channels: int = 2048,
        latent_channels: int = 256,
        attention_heads: int = 8,
    ):
        super().__init__()
        assert latent_channels % 2 == 0, "latent_channels must be even"
        half = latent_channels // 2

        self.extract = ConvFeatureBlock(
            encoder_channels,
            latent_channels,
        )
        self.z1_conv = ConvFeatureBlock(
            half,
            half,
        )
        self.z2_conv = ConvFeatureBlock(
            half,
            half,
        )
        self.z2_attention = TemporalAttention1D(
            half,
            heads=attention_heads,
        )

    def forward(self, x: torch.Tensor) -> dict:
        w = self.extract(x)

        z1, z2 = torch.chunk(w, 2, dim=1)

        z1 = self.z1_conv(z1)

        z2 = self.z2_conv(z2)
        z2 = self.z2_attention(z2)

        latent = torch.cat([z1, z2], dim=1)

        return {
            "latent": latent,
            "z1": z1,
            "z2": z2,
        }
