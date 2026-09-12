"""GRAIL-ECG encoder wrapper for LVCG linear probing pipeline."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

# Disable cuDNN to avoid ptrDesc->finalize() error on PyTorch 2.6 + A100
torch.backends.cudnn.enabled = False

# encoders/ -> probing/ -> LVCG/ -> external/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grail_ecg.src.models.baselines import FactorialGRAILEncoder
from grail_ecg.src.geometry.lead_geometry import independent_from_standard, get_angles_tensor, INDEPENDENT_8_LEADS
from .base import BaseEncoder


class GRAILEncoder(BaseEncoder):
    """Frozen GRAIL-ECG backbone for LVCG multi-dataset linear probing.
    
    Extracts the 96-dimensional clinical latent state Z in R^{96}
    from raw 12-lead ECGs (500 Hz canonical).
    """

    TARGET_FS = 500
    TARGET_LEN = 5000

    def __init__(
        self,
        checkpoint: str,
        use_geometry: bool = True,
        use_slots: bool = True,
        use_view_aux: bool = True,
        hidden_dim: int = 128,
        num_slots: int = 6,
        slot_dim: int = 16,
        concept_config: str = "configs/ptbxl_concept_tiers.yaml",
    ):
        super().__init__()
        self.out_features = num_slots * slot_dim  # 96

        anchor_counts_per_domain = None
        if use_slots:
            concept_path = Path(concept_config)
            if not concept_path.is_absolute():
                concept_path = PROJECT_ROOT / concept_path
            with open(concept_path) as f:
                anchor_counts_per_domain = yaml.safe_load(f)["anchor_counts_per_domain"]

        self.backbone = FactorialGRAILEncoder(
            use_geometry=use_geometry,
            use_slots=use_slots,
            use_view_aux=use_view_aux,
            num_leads=8,
            num_tokens_per_lead=32,
            hidden_dim=hidden_dim,
            num_slots=num_slots,
            slot_dim=slot_dim,
            anchor_counts_per_domain=anchor_counts_per_domain,
            total_anchor_classes=25,
        )

        ckpt_path = Path(checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = PROJECT_ROOT / ckpt_path

        if not ckpt_path.exists():
            raise FileNotFoundError(f"GRAIL checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
        # A probing run must never continue with a partly initialized encoder.
        # The architecture flags in the evaluation config must exactly match the
        # frozen checkpoint being evaluated.
        self.backbone.load_state_dict(state_dict, strict=True)

        self.backbone.eval()
        self.angles = get_angles_tensor(INDEPENDENT_8_LEADS)

    def _resample(self, ecg: torch.Tensor, source_fs: int = 500) -> torch.Tensor:
        if source_fs != self.TARGET_FS:
            new_len = int(ecg.shape[-1] * self.TARGET_FS / source_fs)
            ecg = F.interpolate(ecg, size=new_len, mode="linear", align_corners=False)
        T = ecg.shape[-1]
        if T > self.TARGET_LEN:
            ecg = ecg[..., : self.TARGET_LEN]
        elif T < self.TARGET_LEN:
            ecg = F.pad(ecg, (0, self.TARGET_LEN - T))
        return ecg

    @torch.no_grad()
    def ext_ecg_emb(self, ecg: torch.Tensor, source_fs: int = 500) -> torch.Tensor:
        """Extracts the 96D clinical latent representation Z from 12-lead ECG.
        
        Args:
            ecg: [B, 12, T] or [B, 8, T] raw ECG tensor in mV.
            source_fs: sampling frequency (Hz).
            
        Returns:
            [B, 96] latent representation.
        """
        ecg = self._resample(ecg, source_fs)

        if ecg.shape[1] == 12:
            ecg_8l = independent_from_standard(ecg)
        elif ecg.shape[1] == 8:
            ecg_8l = ecg
        else:
            raise ValueError(f"Expected 12 or 8 leads, got {ecg.shape[1]}")

        device = next(self.backbone.parameters()).device
        angles = self.angles.to(device)

        slots, z_flat, _ = self.backbone(ecg_8l.to(device), custom_angles=angles)
        return z_flat
