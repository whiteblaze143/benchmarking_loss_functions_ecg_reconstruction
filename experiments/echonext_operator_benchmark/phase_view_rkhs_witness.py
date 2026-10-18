#!/usr/bin/env python3
r"""Phase–View RKHS Witness Network (NEF-Net Paradigm Meets RKHS Phase-KME).

A continuous, intrinsically interpretable diagnostic evidence field over ECG measurement
viewpoints (q in S^7) and cardiac phase cells (p in {1, ..., 16}).

Key Mathematical Principles:
1. Waveform-First Projection:
   y_q(t) = q^T x(t) is computed in physical coordinates before nonlinear kernel embedding.
2. Authentic RKHS Phase-KME:
   mu_{q, p} = (1 / |C_p|) sum_{t in C_p} phi(y_q(t), dy_q/dt(t)) in R^128,
   representing phase-conditioned distributions of physical voltage and time-scaled slope.
3. Queryable Diagnostic Witness:
   E(q, p) = <w(q, p), mu_{q, p}>_H = E_{Z_{q,p}}[w_{q,p}(y_q, dy_q/dt)]
   scoring observed electrical states directly via the reproducing kernel property.
4. Compact Linear Witness Parameterization:
   w_{q, p} = w_{p, 0} + sum_{a=1}^8 q_a w_{p, a}
   requiring exactly 16 x (1 + 8) x 128 = 18,432 trainable coefficients!
5. Exact Polarity Invariance:
   Even/odd decomposition mu_{q, p}^+ = (mu_q + mu_{-q}) / 2, mu_{q, p}^- = (mu_q - mu_{-q}) / 2
   E(q, p) = <w_{p, 0}, mu_{q, p}^+> + sum_{a=1}^8 q_a <w_{p, a}, mu_{q, p}^->
   strictly invariant under (q, y_q) -> (-q, -y_q).
6. Observable Diagnostic Panorama & Subspace Integration:
   Evaluates queried directions q_* ONLY when q_* in row(Q).
   Aggregates evidence over the observed subspace sphere S(row(Q)):
   ell(Q, Y) = b + (1/16) sum_{p=1}^16 \int_{S(row(Q))} E(q, p) d\nu_Q(q) + w_tab^T x_tab.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.echonext_operator_benchmark.measurement_information import (
    compute_noiseless_observable_torch,
    compute_noiseless_observable_np,
)


class PhaseViewRKHSWitness(nn.Module):
    """Phase–View RKHS Witness Classifier.

    Parameters:
        w_iso: (num_classes, 16, 128) Isotropic witness component w_{p, 0}
        w_dir: (num_classes, 16, 8, 128) Directional witness component w_{p, a} (if not isotropic_only)
        bias:  (num_classes,) Base prevalence log-odds
        w_tab: (num_classes, 7) Optional tabular covariate weights

    Total witness coefficients for binary (num_classes=1):
        - Directional: 16 * 9 * 128 = 18,432
        - Isotropic only: 16 * 128 = 2,048
    """

    def __init__(
        self,
        num_classes: int = 1,
        num_phases: int = 16,
        canonical_dim: int = 8,
        nystrom_dim: int = 128,
        num_tabular: int = 0,
        isotropic_only: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_phases = num_phases
        self.canonical_dim = canonical_dim
        self.nystrom_dim = nystrom_dim
        self.num_tabular = num_tabular
        self.isotropic_only = isotropic_only

        # Isotropic witness: w_{p, 0} in R^(num_classes, 16, 128)
        self.w_iso = nn.Parameter(torch.zeros(num_classes, num_phases, nystrom_dim))
        nn.init.normal_(self.w_iso, std=0.01)

        # Directional witness: w_{p, a} in R^(num_classes, 16, 8, 128)
        if not isotropic_only:
            self.w_dir = nn.Parameter(torch.zeros(num_classes, num_phases, canonical_dim, nystrom_dim))
            nn.init.normal_(self.w_dir, std=0.01)
        else:
            self.register_parameter("w_dir", None)

        self.bias = nn.Parameter(torch.zeros(num_classes))
        if num_tabular > 0:
            self.w_tab = nn.Linear(num_tabular, num_classes, bias=False)
        else:
            self.w_tab = None

    @property
    def num_witness_parameters(self) -> int:
        """Returns the exact number of trainable witness coefficients."""
        count = self.w_iso.numel()
        if self.w_dir is not None:
            count += self.w_dir.numel()
        return count

    def compute_evidence(
        self,
        mu_plus: torch.Tensor,
        mu_minus: torch.Tensor,
        q: torch.Tensor,
    ) -> torch.Tensor:
        """Compute signed diagnostic evidence E(q, p) for each viewpoint and phase.

        E(q, p) = <w_{p, 0}, mu_{q, p}^+> + sum_{a=1}^8 q_a <w_{p, a}, mu_{q, p}^->

        Args:
            mu_plus:  (B, m, 16, 128) Even Phase-KME embedding
            mu_minus: (B, m, 16, 128) Odd Phase-KME embedding
            q:        (B, m, 8) Unit measurement operators

        Returns:
            E_qp:     (B, m, num_classes, 16) Signed evidence per phase cell
        """
        # mu_plus: (B, m, 16, 128) -> unsqueeze to (B, m, 1, 16, 128)
        # w_iso:   (C, 16, 128)    -> unsqueeze to (1, 1, C, 16, 128)
        mu_p = mu_plus.unsqueeze(2)
        w_i = self.w_iso.unsqueeze(0).unsqueeze(0)
        e_iso = torch.sum(mu_p * w_i, dim=-1)  # (B, m, C, 16)

        if self.w_dir is not None:
            # mu_minus: (B, m, 16, 128) -> (B, m, 1, 16, 1, 128)
            # w_dir:    (C, 16, 8, 128)  -> (1, 1, C, 16, 8, 128)
            mu_m = mu_minus.unsqueeze(2).unsqueeze(-2)
            w_d = self.w_dir.unsqueeze(0).unsqueeze(0)
            inner_a = torch.sum(mu_m * w_d, dim=-1)  # (B, m, C, 16, 8)

            # q: (B, m, 8) -> (B, m, 1, 1, 8)
            q_exp = q.unsqueeze(2).unsqueeze(2)
            e_dir = torch.sum(q_exp * inner_a, dim=-1)  # (B, m, C, 16)
            return e_iso + e_dir
        else:
            return e_iso

    def forward_from_embeddings(
        self,
        mu_pos: torch.Tensor,
        mu_neg: torch.Tensor,
        Q: torch.Tensor,
        tabular: Optional[torch.Tensor] = None,
        aggregation: Literal["mean_tokens", "orthonormal_basis"] = "mean_tokens",
    ) -> Dict[str, Any]:
        """Forward pass from precomputed Phase-KME embeddings."""
        B, m, _, _ = mu_pos.shape

        # Even and odd decomposition
        mu_plus = 0.5 * (mu_pos + mu_neg)   # (B, m, 16, 128)
        mu_minus = 0.5 * (mu_pos - mu_neg)  # (B, m, 16, 128)

        # Compute signed evidence E(q_i, p): shape (B, m, C, 16)
        E_qp = self.compute_evidence(mu_plus, mu_minus, Q)

        # Aggregate across operators: (B, C, 16)
        E_p = torch.mean(E_qp, dim=1)

        # Aggregate across 16 cardiac phases: (B, C)
        ell_ecg = torch.mean(E_p, dim=-1)

        # Add bias: (B, C)
        logits = self.bias.unsqueeze(0) + ell_ecg

        # Tabular contribution
        if tabular is not None and self.w_tab is not None:
            tab_contrib = self.w_tab(tabular)  # (B, C)
            logits = logits + tab_contrib
        else:
            tab_contrib = torch.zeros_like(logits)

        probs = torch.sigmoid(logits)

        # If num_classes == 1, keep last dimension as 1
        return {
            "logits": logits,
            "probs": probs,
            "evidence_field": E_qp.squeeze(2) if self.num_classes == 1 else E_qp,
            "phase_evidence": E_p.squeeze(1) if self.num_classes == 1 else E_p,
            "tab_contribution": tab_contrib,
            "bias": self.bias,
        }

    def evaluate_witness_grid(
        self,
        q: np.ndarray,
        p: int,
        v_grid: np.ndarray,
        s_grid: np.ndarray,
        fit_data: Dict[str, Any],
        class_idx: int = 0,
    ) -> np.ndarray:
        """Evaluate continuous witness w_{q, p}(v, s) on a 2D physical grid.

        By the reproducing kernel property:
            w_{q, p}(v, s) = sum_{k=1}^{128} (w_{q, p})_k phi_k(v, s)
        where (v, s) are physical voltage (mV) and physical slope (mV/s).

        Args:
            q: Direction vector in R^8.
            p: Phase bin index (0 to 15).
            v_grid: 1D array of physical voltages in mV (e.g., np.linspace(-2.0, 2.0, 100)).
            s_grid: 1D array of physical slopes in mV/s (e.g., np.linspace(-50.0, 50.0, 100)).
            fit_data: Dictionary containing Nystrom fit landmarks, inverse root, whitening parameters.
            class_idx: Diagnostic disease class index (default 0).

        Returns:
            grid_score: 2D array of shape (len(v_grid), len(s_grid)) giving diagnostic score.
        """
        q_norm = q / np.maximum(np.linalg.norm(q), 1e-12)
        
        # 1. Compute witness vector w_{q, p} in R^128
        w_iso_np = self.w_iso[class_idx, p].detach().cpu().numpy()  # (128,)
        if self.w_dir is not None:
            w_dir_np = self.w_dir[class_idx, p].detach().cpu().numpy()  # (8, 128)
            w_qp = w_iso_np + q_norm @ w_dir_np                         # (128,)
        else:
            w_qp = w_iso_np

        # 2. Evaluate Nystrom feature map phi(v, s) on grid
        V, S = np.meshgrid(v_grid, s_grid, indexing="ij")
        v_scale = float(fit_data.get("voltage_scale", 1.0))
        # Standard beat duration ~0.8s, 256 samples -> dt ~ 0.003125s
        dt = float(fit_data.get("dt", 0.8 / 256.0))
        V_atom = V.ravel() / v_scale
        S_atom = (S.ravel() * dt) / v_scale
        atoms_grid = np.stack([V_atom, S_atom], axis=-1)  # (N_grid, 2)

        # Whitening
        w_mean = fit_data["whitening_mean"].astype(np.float64)
        w_comp = fit_data["whitening_components"].astype(np.float64)
        w_scale = fit_data["whitening_scales"].astype(np.float64)
        white_grid = (atoms_grid - w_mean) @ w_comp * w_scale

        # IMQ Nystrom features: phi(x) = k(x, L) @ inv_root
        landmarks = fit_data["landmarks"].astype(np.float64)        # (128, 2)
        inv_root = fit_data["inverse_root"].astype(np.float64)      # (128, 128)
        c2 = float(fit_data["c2"])

        diff = white_grid[:, None, :] - landmarks[None, :, :]       # (N_grid, 128, 2)
        d2 = np.sum(diff ** 2, axis=-1) + c2                        # (N_grid, 128)
        K_val = 1.0 / np.sqrt(d2)                                   # (N_grid, 128)
        phi_grid = K_val @ inv_root                                 # (N_grid, 128)

        # 3. Inner product <w_{q, p}, phi(v, s)>
        scores = phi_grid @ w_qp                                    # (N_grid,)
        return scores.reshape(len(v_grid), len(s_grid))


class PhysicalPhaseKMEComputer:
    """Computes exact Phase-KME embeddings from physical beats with physical units."""

    def __init__(self, fit_path_or_dict: Union[str, Path, Dict[str, Any]], device: torch.device):
        if isinstance(fit_path_or_dict, (str, Path)):
            data = np.load(fit_path_or_dict)
            self.fit_data = {k: data[k] for k in data.files}
        else:
            self.fit_data = fit_path_or_dict

        self.device = device
        self.v_scale = float(self.fit_data["voltage_scale"])
        self.landmarks = torch.as_tensor(self.fit_data["landmarks"], device=device, dtype=torch.float32)
        self.inv_root = torch.as_tensor(self.fit_data["inverse_root"], device=device, dtype=torch.float32)
        self.c2 = float(self.fit_data["c2"])
        self.w_mean = torch.as_tensor(self.fit_data["whitening_mean"], device=device, dtype=torch.float32)
        self.w_comp = torch.as_tensor(self.fit_data["whitening_components"], device=device, dtype=torch.float32)
        self.w_scale = torch.as_tensor(self.fit_data["whitening_scales"], device=device, dtype=torch.float32)

    def compute_kme_for_operators(
        self,
        beats_mv: np.ndarray,
        q_bank: np.ndarray,
        sampling_rate: float = 250.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Phase-KME for both +q and -q for exact polarity decomposition.

        Args:
            beats_mv: shape (n_beats, 256, 8) in physical millivolts
            q_bank: shape (m, 8)
            sampling_rate: ECG sampling frequency in Hz (default 250Hz)

        Returns:
            mu_pos: (m, 16, 128) float32 tensor on device
            mu_neg: (m, 16, 128) float32 tensor on device
        """
        n_beats = len(beats_mv)
        m = len(q_bank)
        
        # Normalize operator vectors
        norms = np.linalg.norm(q_bank, axis=1, keepdims=True)
        q_norm = q_bank / np.maximum(norms, 1e-12)

        # 1. Project physical waveform: (n_beats, 256, m)
        # beats_mv: (n_beats, 256, 8), q_norm: (m, 8)
        wf_pos = np.matmul(beats_mv, q_norm.T) / self.v_scale  # (n_beats, 256, m)
        
        # Physical time derivative (accounting for physical sampling interval)
        dt_sample = 1.0 / sampling_rate
        deriv_pos = np.gradient(wf_pos, axis=1) / dt_sample

        # Permute to (m, n_beats, 256, 2)
        atoms_pos = np.stack([wf_pos.transpose(2, 0, 1), deriv_pos.transpose(2, 0, 1)], axis=-1)
        # Under negation: y_{-q} = -y_q, dy_{-q}/dt = -dy_q/dt
        atoms_neg = -atoms_pos

        # Reshape into 16 phase cells: (m, 16, n_beats * 16, 2)
        cells_pos = atoms_pos.reshape(m, n_beats, 16, 16, 2).transpose(0, 2, 1, 3, 4).reshape(m, 16, -1, 2)
        cells_neg = atoms_neg.reshape(m, n_beats, 16, 16, 2).transpose(0, 2, 1, 3, 4).reshape(m, 16, -1, 2)

        t_cells_pos = torch.as_tensor(cells_pos, device=self.device, dtype=torch.float32)
        t_cells_neg = torch.as_tensor(cells_neg, device=self.device, dtype=torch.float32)

        # Whitening
        white_pos = (t_cells_pos - self.w_mean) @ self.w_comp * self.w_scale
        white_neg = (t_cells_neg - self.w_mean) @ self.w_comp * self.w_scale

        # IMQ Nystrom kernel embedding
        # landmarks: (128, 2) -> expand to (m, 16, 1, 128, 2)
        d2_pos = torch.cdist(white_pos, self.landmarks.expand(m, 16, -1, -1)).square() + self.c2
        feat_pos = torch.rsqrt(d2_pos) @ self.inv_root
        mu_pos = feat_pos.mean(dim=2)  # (m, 16, 128)

        d2_neg = torch.cdist(white_neg, self.landmarks.expand(m, 16, -1, -1)).square() + self.c2
        feat_neg = torch.rsqrt(d2_neg) @ self.inv_root
        mu_neg = feat_neg.mean(dim=2)  # (m, 16, 128)

        return mu_pos, mu_neg


class ObservableSubspaceEvaluator:
    """Evaluates queried measurement directions q_* under linear acquisition model Y = Q X."""

    def __init__(self, Q: np.ndarray, tol_observable: float = 1e-4):
        self.Q = np.asarray(Q, dtype=np.float64)
        self.m, self.d = self.Q.shape
        self.tol_observable = tol_observable
        
        # Compute projection matrix P_Q = Q^pinv Q
        self.Q_pinv = np.linalg.pinv(self.Q)
        self.P_Q = self.Q_pinv @ self.Q
        self.P_Q = 0.5 * (self.P_Q + self.P_Q.T)
        self.P_perp = np.eye(self.d) - self.P_Q
        self.rank = int(np.round(np.trace(self.P_Q)))

    def is_observable(self, q_query: np.ndarray) -> bool:
        """Returns True iff q_query lies in row(Q) within tolerance."""
        q = np.asarray(q_query, dtype=np.float64).ravel()
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-12:
            return True
        q_u = q / q_norm
        null_res = np.linalg.norm(self.P_perp @ q_u) ** 2
        return bool(null_res < self.tol_observable)

    def recover_waveform(self, Y: np.ndarray, q_query: np.ndarray) -> np.ndarray:
        """Synthesize queried waveform y_{q_*}(t) = q_*^T Q^pinv Y(t).

        Raises ValueError if q_query is not observable from Q.
        """
        if not self.is_observable(q_query):
            raise ValueError(
                f"Query operator is unobservable: nullspace residue exceeds {self.tol_observable}. "
                "Refusing to synthesize unidentifiable waveform."
            )
        q = np.asarray(q_query, dtype=np.float64).ravel()
        # X_obs = Q_pinv @ Y: shape (8, T)
        X_obs = self.Q_pinv @ Y
        return q @ X_obs  # (T,)

    def canonical_orthonormal_basis(self) -> np.ndarray:
        """Returns orthonormal basis U in R^(k, 8) spanning row(Q)."""
        U, s, Vt = np.linalg.svd(self.Q, full_matrices=False)
        keep = s > 1e-5
        basis = Vt[keep]  # (k, 8)
        return basis
