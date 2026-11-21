"""Declared morphology/RR fusion with explicit modality-availability flags."""
import numpy as np
import torch
from torch import nn
from set_waveform_model import AFSetOperator


def fit_rr_scaling(rr_features, rr_valid):
    rr = np.asarray(rr_features);valid = np.asarray(rr_valid)
    if rr.ndim != 3 or rr.shape[1:] != (2, 16) or rr.dtype != np.float32 or valid.shape != rr.shape[:2] or valid.dtype != bool:
        raise ValueError('Expected float32 observed RR training features and boolean validity')
    if np.isinf(rr[valid]).any(): raise ValueError('Infinite active training RR feature')
    mean = np.zeros((2, 16));sd = np.ones((2, 16));counts = np.zeros((2, 16), int)
    for channel in [0, 1]:
        for feature in range(16):
            x = rr[:, channel, feature][valid[:, channel] & np.isfinite(rr[:, channel, feature])].astype(np.float64)
            if not len(x): raise ValueError('Training RR feature has no finite support')
            counts[channel, feature] = len(x);mean[channel, feature] = x.mean();sd[channel, feature] = x.std()
    constant = sd == 0
    sd[constant] = 1.  # Declared unit scale for an exactly constant training feature.
    return {'mean': mean, 'sd': sd, 'finite_counts': counts, 'constant_feature': constant}


def rr_side_input(features, rr_valid, available, mean, sd):
    if (features.ndim != 3 or features.shape[1:] != (2, 16) or rr_valid.shape != features.shape[:2] or available.shape != rr_valid.shape or
            features.dtype != torch.float32 or mean.shape != (2, 16) or sd.shape != (2, 16) or
            mean.dtype != torch.float32 or sd.dtype != torch.float32):
        raise ValueError('Invalid local RR side-input shape')
    if rr_valid.dtype != torch.bool or available.dtype != torch.bool:
        raise ValueError('RR validity and availability must be boolean')
    active = rr_valid & available
    if not torch.isfinite(features[:, :, 2][active]).all(): raise ValueError('Missing active mean-RR statistic')
    standardized = (features-mean)/sd
    if torch.isinf(standardized[active]).any(): raise ValueError('Infinite active RR statistic')
    # Missing individual statistics use the fitted mean; masks remain explicit.
    clean = torch.where(active[:, :, None] & torch.isfinite(standardized), standardized, 0.)
    return torch.cat((clean.flatten(1), available.float(), active.float()), dim=1), active.any(dim=1)


class AFSetRR(nn.Module):
    """Original categorical morphology encoder plus one linear fusion head.

    No morphology representation is manufactured: zero latent is an explicitly
    flagged missing-modality sentinel. Actual RR may support a hybrid prediction
    in that case. If both modalities are unavailable, the caller must retain an
    indeterminate result. Such RR-only recovery is not evidence for operator
    geometry or morphology superiority.
    """
    def __init__(self):
        super().__init__()
        self.phase = AFSetOperator('categorical')
        self.head = nn.Linear(256+36+2+2, 1)

    def forward(self, responses, available, phase_valid, operator_ids, rr_input, rr_present, represented_fraction):
        phase_present = (available & phase_valid).any(dim=1)
        if (rr_input.shape != (len(responses), 36) or rr_present.shape != (len(responses),) or represented_fraction.shape != available.shape or
                rr_input.dtype != torch.float32 or rr_present.dtype != torch.bool):
            raise ValueError('Invalid declared hybrid side inputs')
        if not (phase_present | rr_present).all():
            raise ValueError('Both representations unavailable: retain indeterminate')
        latent = torch.zeros((len(responses), 256), dtype=responses.dtype, device=responses.device)
        if phase_present.any():
            encoded = self.phase.encode(responses[phase_present], available[phase_present],
                                        phase_valid[phase_present], operator_ids[phase_present])
            latent[phase_present] = encoded.to(latent.dtype)
        coverage = torch.where(available & phase_valid, represented_fraction, 0.)
        if not torch.isfinite(rr_input).all() or not torch.isfinite(coverage).all():
            raise ValueError('Nonfinite observed hybrid side input')
        modalities = torch.stack((phase_present, rr_present), dim=1).to(latent.dtype)
        return self.head(torch.cat((latent, rr_input, modalities, coverage), dim=1)).squeeze(-1)
