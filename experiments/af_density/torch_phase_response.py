"""Fixed float64 torch evaluation of the authentic Phase-KME feature mean."""
import numpy as np
import torch


class TorchPhaseResponse:
    def __init__(self, whitening, mapping, *, device):
        self.device = torch.device(device)
        if mapping.landmarks.shape != (128, 2) or mapping.inverse_root.shape != (128, 128):
            raise ValueError('Expected authentic128-landmark two-coordinate map')
        self.mean = torch.as_tensor(whitening.mean, dtype=torch.float64, device=self.device)
        self.components = torch.as_tensor(whitening.components, dtype=torch.float64, device=self.device)
        self.scales = torch.as_tensor(whitening.scales, dtype=torch.float64, device=self.device)
        self.landmarks = torch.as_tensor(mapping.landmarks, dtype=torch.float64, device=self.device)
        self.inverse_root = torch.as_tensor(mapping.inverse_root, dtype=torch.float64, device=self.device)
        self.c2 = mapping.c2

    @torch.inference_mode()
    def __call__(self, atoms):
        values = np.asarray(atoms)
        if (values.ndim != 3 or values.shape[0] != 16 or values.shape[2] != 2 or
                values.shape[1] < 32 or values.shape[1] % 16 or not np.isfinite(values).all()):
            raise ValueError('Expected16 phase cells from at least two accepted beats')
        x = torch.as_tensor(values, dtype=torch.float64, device=self.device)
        white = ((x-self.mean)@self.components)*self.scales
        distance = ((white[:, :, None, :]-self.landmarks[None, None, :, :])**2).sum(dim=-1)
        result = torch.rsqrt(distance+self.c2).mean(dim=1)@self.inverse_root
        if not torch.isfinite(result).all():
            raise ValueError('Nonfinite Phase-KME response')
        return result.to(dtype=torch.float32).cpu().numpy()
