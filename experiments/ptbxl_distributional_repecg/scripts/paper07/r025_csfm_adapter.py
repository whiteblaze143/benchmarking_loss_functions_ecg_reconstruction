from __future__ import annotations

import hashlib
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.signal import resample

logger = logging.getLogger("r025_csfm_adapter")

# Pinned official SHA256 for CSFM_base.pth
ADMITTED_CSFM_BASE_SHA256 = "497f32ec79b5e135fddcf1073774db2466c11260de102424f8d9afc108c11f1b"

# Standard 12-lead names recognized by CSFM
VALID_CSFM_CHANNELS = {"I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"}


class CSFMAdapter:
    """Official adapter for Cardiac-Sensing Foundation Model (CSFM Base).

    Enforces:
    - Checkpoint admission check (SHA256 match against official release).
    - Native channel identification (standard leads only; rejects arbitrary linear q with N/A).
    - Native 250 Hz resampled preprocessing.
    - Deterministic frozen feature extraction.
    """

    def __init__(self, csfm_repo_path: Path, checkpoint_path: Path, device: torch.device):
        self.device = device
        self.repo_path = csfm_repo_path
        self.ckpt_path = checkpoint_path
        self.model = None
        self.admission_status = "NOT_ADMITTED"

        # Check admission
        if not self.ckpt_path.exists():
            logger.warning(f"CSFM checkpoint not found at {self.ckpt_path}. Admission: NOT_ADMITTED.")
            return

        sha = self._compute_sha256(self.ckpt_path)
        if sha != ADMITTED_CSFM_BASE_SHA256:
            logger.warning(f"CSFM checkpoint SHA256 {sha} does not match admitted {ADMITTED_CSFM_BASE_SHA256}. NOT_ADMITTED.")
            return

        self.admission_status = "ADMITTED"
        self._load_model()

    @staticmethod
    def _compute_sha256(filepath: Path) -> str:
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(1024 * 1024):
                h.update(chunk)
        return h.hexdigest()

    def _load_model(self):
        if str(self.repo_path) not in sys.path:
            sys.path.insert(0, str(self.repo_path))

        try:
            from network.model import CSFM_model
            model = CSFM_model("Base")
            ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
            encoder_state_dict = {
                k.replace("encoder.", ""): v
                for k, v in ckpt.items()
                if k.startswith("encoder.") and "mlp_head" not in k
            }
            model.load_state_dict(encoder_state_dict, strict=False)
            model.mlp_head = torch.nn.Identity()
            model.to(self.device)
            model.eval()
            self.model = model
            logger.info("CSFM Base model successfully loaded and admitted.")
        except Exception as e:
            logger.error(f"Failed to load CSFM model: {e}")
            self.admission_status = "NOT_ADMITTED"
            self.model = None

    def preprocess_signal(
        self,
        signal_1d: np.ndarray,
        orig_fs: int = 100,
        target_fs: int = 250,
        target_len: int = 2500,
    ) -> np.ndarray:
        """Native CSFM resampling to 250 Hz and length 2500 with z-score standardization."""
        n_target = int(len(signal_1d) * target_fs / orig_fs)
        resampled = resample(signal_1d, n_target)
        if len(resampled) >= target_len:
            out = resampled[:target_len]
        else:
            out = np.zeros(target_len, dtype=resampled.dtype)
            out[:len(resampled)] = resampled
        std = np.std(out)
        if std > 1e-6:
            out = (out - np.mean(out)) / std
        return out.astype(np.float32)

    def extract_features(
        self,
        leads_dict: Dict[str, np.ndarray],
        orig_fs: int = 100,
    ) -> Optional[torch.Tensor]:
        """Extract frozen representations for supported standard leads.

        Args:
            leads_dict: Dict mapping lead names to raw 1D waveforms.

        Returns:
            Latent representation tensor, or None if invalid leads / arbitrary combinations.
        """
        if self.admission_status != "ADMITTED" or self.model is None:
            return None

        # Verify all requested leads are standard valid channels
        for name in leads_dict:
            if name not in VALID_CSFM_CHANNELS:
                logger.info(f"CSFM rejecting non-standard channel {name} -> N/A")
                return None

        # Prepare batch input for CSFM
        # Native CSFM forward expects multi-lead tensor
        lead_order = sorted(leads_dict.keys())
        proc_leads = [
            self.preprocess_signal(leads_dict[k], orig_fs=orig_fs)
            for k in lead_order
        ]
        stacked = np.stack(proc_leads, axis=0)  # [n_leads, 2500]
        tensor_in = torch.as_tensor(stacked, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            try:
                # Extract representation
                feats = self.model.forward_features(tensor_in, lead_order)
                return feats
            except Exception as e:
                logger.warning(f"CSFM feature extraction failed: {e}")
                return None
