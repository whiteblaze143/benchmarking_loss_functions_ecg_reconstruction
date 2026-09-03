import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv1D(nn.Module):
    """
    1D Double Convolution block: (Conv1D => BatchNorm1D => ReLU) * 2
    Matches the Electrocardio-Panorama NefNet decoder block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpsampleDoubleConv(nn.Module):
    """
    1D Linear Interpolation upsampling (x2) followed by a DoubleConv1D block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()
        self.conv = DoubleConv1D(
            in_channels,
            out_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x,
            scale_factor=2,
            mode="linear",
            align_corners=False,
        )
        return self.conv(x)


class ECGDecoder(nn.Module):
    """
    1D Convolutional Upsampling ECG Decoder.
    Reconstructs 5000-sample single-lead waveform from query-conditioned latent Z_l.
    
    IMPORTANT: Output is UNBOUNDED linear (no sigmoid or tanh), properly supporting
    zero-mean standardized ECG signals with both positive and negative values.
    """

    def __init__(
        self,
        latent_channels: int = 256,
        target_length: int = 5000,
    ):
        super().__init__()
        self.target_length = target_length

        self.blocks = nn.Sequential(
            UpsampleDoubleConv(latent_channels, 128),
            UpsampleDoubleConv(128, 64),
            UpsampleDoubleConv(64, 32),
            UpsampleDoubleConv(32, 16),
            UpsampleDoubleConv(16, 8),
        )

        self.output = nn.Conv1d(
            8,
            1,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        x = self.output(x)

        # Due to encoder downsampling strides (e.g. 5000 -> 157 -> 5024),
        # guard to guarantee exact target temporal dimension.
        if x.shape[-1] != self.target_length:
            x = F.interpolate(
                x,
                size=self.target_length,
                mode="linear",
                align_corners=False,
            )

        return x
