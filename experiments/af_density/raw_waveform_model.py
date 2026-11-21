"""Existing eight-block ResNet with explicit observed-channel masks."""
import sys
from pathlib import Path
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'zzu_avblock'))
from baseline import ResNet


class RawMaskResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = ResNet()
        # Two observed voltages plus two constant availability channels.
        self.network.stem[0] = nn.Conv1d(4, 32, 15, 2, padding=7, bias=False)

    def forward(self, waveform, available):
        if waveform.ndim != 3 or waveform.shape[1:] != (2, 6000):
            raise ValueError('Expected (batch,2,6000) normalized waveform')
        if available.shape != waveform.shape[:2] or available.dtype != torch.bool:
            raise ValueError('Expected boolean observed-channel mask')
        if not available.any(dim=1).all():
            raise ValueError('No observed channel: prediction must remain indeterminate')
        mask = available[:, :, None]
        observed = torch.where(mask, waveform, 0.)
        return self.network(torch.cat((observed, mask.expand_as(waveform).to(waveform.dtype)), dim=1))


def availability_ids(global_window_ids, epoch):
    """Every window cycles stored channel0, channel1, both over three epochs."""
    return (global_window_ids+epoch) % 3
