#!/usr/bin/env python3
"""Authentic Phase-KME Representation Module for EchoNext.

Reuses the exact Nystrom RKHS Phase-conditioned Kernel Mean Embedding (Phase-KME)
algorithm established on PTB-XL, using parameters fitted exclusively on the
EchoNext training split (N=72,475).

Pipeline per record:
1. Slices the 8 canonical leads [0, 1, 6, 7, 8, 9, 10, 11] (I, II, V1..V6).
2. Detects R-peaks with detect_rpeaks(basis, fs=250).
3. Phase-normalizes valid RR intervals into 256 samples: (n_beats, 256, 8).
4. For any set of operators Q (k x 8):
   - Projects beats: y = Q @ beats -> (k, n_beats, 256)
   - Scales by training voltage_scale (0.9641)
   - Computes derivatives: dy = grad(y)
   - Bins into 16 cardiac phase cells (16 samples per cell) -> 2D atoms (y, dy)
   - Whitens 2D atoms with training whitening transform
   - Evaluates Nystrom inverse-multiquadric kernel features: M=128 landmarks
   - Averages features across atoms in each phase cell -> (k, 16, 128)
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = REPO_ROOT / "experiments/ptbxl_distributional_repecg/src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from repecg.common.beats import detect_rpeaks
from repecg.common.phase import phase_normalize

DEFAULT_FIT_PATH = Path("/data/mithunmanivannan/codex_artifacts/echonext_three_way_v1/representations/echonext_phase_kme_fit.npz")
CANONICAL_8_LEADS = [0, 1, 6, 7, 8, 9, 10, 11]


class EchoNextPhaseKMEExtractor:
    """Authentic Phase-KME Extractor for EchoNext."""

    def __init__(self, fit_path: Union[str, Path] = DEFAULT_FIT_PATH, device: Union[str, torch.device] = "cpu"):
        self.device = torch.device(device)
        fit = np.load(str(fit_path))
        self.voltage_scale = float(fit["voltage_scale"])
        self.sampling_rate = int(fit.get("sampling_rate", 250))
        self.c2 = float(fit.get("c2", 1.0))

        self.whitening_mean = torch.as_tensor(fit["whitening_mean"], dtype=torch.float32, device=self.device)
        self.whitening_comp = torch.as_tensor(fit["whitening_components"], dtype=torch.float32, device=self.device)
        self.whitening_scales = torch.as_tensor(fit["whitening_scales"], dtype=torch.float32, device=self.device)
        self.landmarks = torch.as_tensor(fit["landmarks"], dtype=torch.float32, device=self.device)
        self.inverse_root = torch.as_tensor(fit["inverse_root"], dtype=torch.float32, device=self.device)

    def extract_beats(self, wf_12: np.ndarray) -> np.ndarray:
        """Extract phase-normalized beats from a (2500, 12) or (2500, 8) waveform.
        
        Returns:
            np.ndarray of shape (n_beats, 256, 8)
        """
        if wf_12.shape[-1] == 12:
            sig_8 = wf_12[:, CANONICAL_8_LEADS].astype(np.float64)
        else:
            sig_8 = wf_12.astype(np.float64)

        det = detect_rpeaks(sig_8, fs=self.sampling_rate)
        if len(det.valid_intervals) < 2:
            # Fallback for arrhythmic/flat records: synthesize regular 1-second interval grid
            intervals = np.array([[i * self.sampling_rate, (i + 1) * self.sampling_rate] for i in range(len(sig_8) // self.sampling_rate - 1)])
            beats = phase_normalize(sig_8, intervals, samples=256)
        else:
            beats = phase_normalize(sig_8, det.valid_intervals, samples=256)

        if len(beats) == 0:
            # Emergency fallback: single center window
            beats = np.zeros((1, 256, 8), dtype=np.float64)

        return beats.astype(np.float32)

    def compute_phase_kme(self, beats: torch.Tensor, ops: torch.Tensor) -> torch.Tensor:
        """Compute Phase-KME response tensor from beats and operators.
        
        Args:
            beats: (n_beats, 256, 8) torch.Tensor
            ops: (k, 8) torch.Tensor
            
        Returns:
            responses: (k, 16, 128) torch.Tensor
        """
        beats = torch.as_tensor(beats, dtype=torch.float32, device=self.device)
        ops = torch.as_tensor(ops, dtype=torch.float32, device=self.device)
        k = ops.shape[0]
        n_beats = beats.shape[0]

        # 1. Project beats: (k, n_beats, 256)
        y = torch.einsum("bij,kj->kbi", beats, ops) / self.voltage_scale
        # 2. Gradient along phase: (k, n_beats, 256)
        dy = torch.gradient(y, dim=-1)[0]
        # 3. Stack into 2D atoms: (k, n_beats, 256, 2)
        atoms = torch.stack([y, dy], dim=-1)

        # 4. Partition into 16 phase cells (16 samples per cell): (k, 16, n_beats * 16, 2)
        cells = atoms.reshape(k, n_beats, 16, 16, 2).permute(0, 2, 1, 3, 4).reshape(k, 16, -1, 2)

        # 5. Whiten 2D atoms: (cells - mean) @ comp * scales
        white = (cells - self.whitening_mean) @ self.whitening_comp * self.whitening_scales

        # 6. Nystrom Kernel Map: 1 / sqrt(||white - landmarks||^2 + c2) @ inverse_root
        # Shape: (k, 16, total_atoms_per_cell, 128)
        dists_sq = torch.cdist(white, self.landmarks).square()
        features = torch.rsqrt(dists_sq + self.c2) @ self.inverse_root

        # 7. Mean embedding per phase cell: (k, 16, 128)
        responses = features.mean(dim=2)
        return responses
