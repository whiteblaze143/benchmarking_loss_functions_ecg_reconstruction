"""
Single source of truth for Mason et al. (2024) 12-lead ECG protocol.

All constants and helpers for limited 12-lead ECG reconstruction follow
third_party/ecg_reconstruction. We only add: FM encoder + spatial fusion.
Everything else (lead order, value range, R² loss, decoder architecture) is Mason's.

References:
- third_party/ecg_reconstruction/util_functions/general.py (get_twelve_keys, get_lead_keys, get_value_range)
- third_party/ecg_reconstruction/training_functions/reconstruction_functions.py (process_batch, batch_r2_function)
- third_party/ecg_reconstruction/learn_functions/reconstructor.py, generate_model.py
"""

from src.reconstruction.util_functions.general import get_twelve_keys, get_lead_keys, get_value_range

# Re-export scaling and R² from Mason reconstruction_functions (same formulas as third_party)
from src.reconstruction.learn_functions.reconstruction_functions import (
    normalize_mason,
    denormalize_mason,
    mason_batch_r2_loss,
    mason_element_r2,
    mason_element_mse,
)
import torch.nn as nn

class MasonR2Loss(nn.Module):
    """
    Standard R² loss for Mason et al. (2024) protocol.
    Wrapper around mason_batch_r2_loss.
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, model_output, model_target):
        return mason_batch_r2_loss(model_output, model_target)

# --- Value range (Mason get_value_range): raw mV -> [0,1] via (x - min_value) / amplitude ---
MASON_MIN = -2.5
MASON_AMP = 5.0
MASON_WAVE_SAMPLE = 2500  # Mason default sample length (we use 5000 for FM encoder)

# --- 12-lead tensor indices (Mason order from get_twelve_keys) ---
# Order: I=0, II=1, III=2, aVL=3, aVR=4, aVF=5, V1=6, V2=7, V3=8, V4=9, V5=10, V6=11
PRECORDIAL_INDICES = list(range(6, 12))  # V1–V6 (reconstruction target in Mason)
# Mason 'limb+v3' input: I, III, V3 -> indices 0, 2, 8 (Absolute Parity)
MASON_INPUT_LIMB_V3 = [0, 2, 8]


def get_precordial_keys():
    """Precordial lead names in Mason order (same as get_lead_keys('precordial'))."""
    return get_lead_keys('precordial')


def get_limb_v3_keys():
    """Input lead names for Mason limb+v3 configuration."""
    return get_lead_keys('limb+v3')


__all__ = [
    'get_twelve_keys',
    'get_lead_keys',
    'get_value_range',
    'get_precordial_keys',
    'get_limb_v3_keys',
    'MASON_MIN',
    'MASON_AMP',
    'MASON_WAVE_SAMPLE',
    'PRECORDIAL_INDICES',
    'LIMB_INDICES',
    'MASON_INPUT_LIMB_V3',
    'normalize_mason',
    'denormalize_mason',
    'mason_batch_r2_loss',
    'mason_element_r2',
    'mason_element_mse',
    'MasonR2Loss',
]
