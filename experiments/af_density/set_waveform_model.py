"""AF head adapter around the authentic measurement-set encoder."""
import sys
from pathlib import Path
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'ptbxl_distributional_repecg'/'src'))
from repecg.paper07_operator.model import OperatorSetModel


class AFSetOperator(nn.Module):
    """Original architecture; stored-channel identity or q-ablated mode only.

    This class does not impute an all-channel extraction failure. The caller
    must keep that row indeterminate and preserve it in evaluation denominators.
    Operator IDs0/1 are training-dataset channel identities, not canonical q.
    An external unseen channel must explicitly receive ID-1.
    """
    def __init__(self, operator_mode):
        super().__init__()
        if operator_mode not in ['categorical', 'q_ablated']:
            raise ValueError('Unverified Holter geometry: only categorical/q_ablated admitted')
        self.original = OperatorSetModel(response_dim=128, classes=1,
                                         operator_mode=operator_mode,
                                         vocabulary_size=2 if operator_mode == 'categorical' else 0)

    def encode(self, responses, available, phase_valid, operator_ids):
        if responses.ndim != 4 or responses.shape[1:] != (2, 16, 128):
            raise ValueError('Expected (batch,2,16,128) authentic responses')
        if (available.shape != responses.shape[:2] or available.dtype != torch.bool or
                phase_valid.shape != available.shape or phase_valid.dtype != torch.bool or
                operator_ids.shape != available.shape or operator_ids.dtype != torch.long):
            raise ValueError('Invalid availability, phase-valid mask or operator IDs')
        if ((operator_ids < -1) | (operator_ids > 1)).any():
            raise ValueError('Use explicit ID-1 for unseen external channels')
        active = available & phase_valid
        if not active.any(dim=1).all():
            raise ValueError('No usable observed phase representation: retain indeterminate')
        clean = torch.where(active[:, :, None, None], responses, 0.)
        if not torch.isfinite(clean).all():
            raise ValueError('Nonfinite active phase response')
        return self.original.encode_context(None, clean, mask=~active, operator_ids=operator_ids)

    def forward(self, responses, available, phase_valid, operator_ids):
        latent = self.encode(responses, available, phase_valid, operator_ids)
        return self.original.head(latent).squeeze(-1)


class ContinuousAFSetOperator(nn.Module):
    """Continuous SetOperator with 8-lead clinical basis geometry [I, II, V1, V2, V3, V4, V5, V6].

    Embeds continuous operator coordinates q in R^8.
    Nominal Lead I = [1, 0, 0, 0, 0, 0, 0, 0]
    Nominal Lead II = [0, 1, 0, 0, 0, 0, 0, 0]
    """
    def __init__(self, operator_dim=8, operator_mode='continuous'):
        super().__init__()
        if operator_mode not in ['continuous', 'projective']:
            raise ValueError(f'ContinuousAFSetOperator requires continuous or projective mode, got {operator_mode}')
        self.operator_mode = operator_mode
        self.operator_dim = operator_dim
        self.original = OperatorSetModel(response_dim=128, classes=1,
                                         operator_mode=operator_mode,
                                         operator_dim=operator_dim)
        # Register nominal Lead I and Lead II coordinates in the 8-lead clinical basis
        nominal = torch.zeros(2, operator_dim, dtype=torch.float32)
        nominal[0, 0] = 1.0  # Lead I
        nominal[1, 1] = 1.0  # Lead II
        self.register_buffer('nominal_operators', nominal, persistent=True)

    def default_operators(self, batch_size, device=None):
        return self.nominal_operators.unsqueeze(0).expand(batch_size, -1, -1).to(device=device)

    def encode(self, responses, available, phase_valid, operators=None, operator_ids=None):
        if responses.ndim != 4 or responses.shape[1:] != (2, 16, 128):
            raise ValueError('Expected (batch,2,16,128) authentic responses')
        batch = responses.shape[0]
        if (available.shape != responses.shape[:2] or available.dtype != torch.bool or
                phase_valid.shape != available.shape or phase_valid.dtype != torch.bool):
            raise ValueError('Invalid availability or phase-valid mask')
        if operators is None:
            operators = self.default_operators(batch, device=responses.device)
        elif operators.shape != (batch, 2, self.operator_dim) or not torch.isfinite(operators).all():
            raise ValueError(f'Expected finite operators of shape ({batch}, 2, {self.operator_dim})')
        active = available & phase_valid
        if not active.any(dim=1).all():
            raise ValueError('No usable observed phase representation: retain indeterminate')
        clean = torch.where(active[:, :, None, None], responses, 0.)
        if not torch.isfinite(clean).all():
            raise ValueError('Nonfinite active phase response')
        return self.original.encode_context(operators, clean, mask=~active, operator_ids=operator_ids)

    def forward(self, responses, available, phase_valid, operators=None, operator_ids=None):
        latent = self.encode(responses, available, phase_valid, operators=operators, operator_ids=operator_ids)
        return self.original.head(latent).squeeze(-1)

