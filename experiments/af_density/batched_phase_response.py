"""Bounded GPU batches of the same variable-beat Phase-KME feature mean."""
import numpy as np
import torch


@torch.inference_mode()
def batched_phase_response(transform, atoms_list):
    """At most16 channel windows; padding is excluded from every feature mean.

    Padding exists only in this compute buffer. It never adds detected peaks,
    accepted beats, represented time or denominator weight to a recording.
    """
    if not 1 <= len(atoms_list) <= 16:
        raise ValueError('Expected1--16 actual channel representations')
    values = [np.asarray(a) for a in atoms_list]
    for a in values:
        if (a.ndim != 3 or a.shape[0] != 16 or a.shape[2] != 2 or
                a.shape[1] < 32 or a.shape[1] % 16 or not np.isfinite(a).all()):
            raise ValueError('Invalid actual phase atoms')
    lengths = np.array([a.shape[1] for a in values]);maximum = int(lengths.max())
    padded = np.zeros((len(values), 16, maximum, 2), np.float64)
    for i, a in enumerate(values): padded[i, :, :len(a[0])] = a
    x = torch.as_tensor(padded, dtype=torch.float64, device=transform.device)
    count = torch.as_tensor(lengths, device=transform.device)
    valid = torch.arange(maximum, device=transform.device)[None, :] < count[:, None]
    white = ((x-transform.mean)@transform.components)*transform.scales
    distance = ((white[:, :, :, None, :]-transform.landmarks[None, None, None, :, :])**2).sum(dim=-1)
    kernel = torch.rsqrt(distance+transform.c2)
    kernel.masked_fill_(~valid[:, None, :, None], 0.)
    result = (kernel.sum(dim=2)/count[:, None, None])@transform.inverse_root
    if not torch.isfinite(result).all():
        raise ValueError('Nonfinite batched Phase-KME response')
    return result.to(dtype=torch.float32).cpu().numpy()
