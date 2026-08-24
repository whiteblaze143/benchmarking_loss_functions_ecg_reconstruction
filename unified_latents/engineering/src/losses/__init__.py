"""Loss functions init."""
from .signal_losses import STFTLoss, BandPowerLoss, CompositeLoss

__all__ = ['STFTLoss', 'BandPowerLoss', 'CompositeLoss']
