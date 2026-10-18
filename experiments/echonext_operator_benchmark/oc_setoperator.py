#!/usr/bin/env python3
"""Observable-Concept SetOperator (OC-SetOperator).

An intrinsically interpretable architecture for continuous ECG measurement operators.
Separates measurement information, clinical evidence, and transparent diagnostic inference
into distinct, auditable modules:

Module A: Measurement Information Layer (measurement_information.py)
    Computes exact linear canonicalization X_obs = Q^pinv Y, P_Q = Q^pinv Q, P_perp = I - P_Q.
    Mathematically invariant to coordinate rotation, invertible basis change, row permutation,
    polarity flipping, and algebraically redundant measurements.

Module B: ECG Evidence Encoder
    1D Temporal Convolutional Encoder operating directly on canonical X_obs in R^(8, T).
    Combines waveform morphology with observed subspace geometry P_Q.

Module C: Observable Clinical Concept Layer (clinical_concepts.py)
    Estimates 6 core clinically defined ECG findings with heteroscedastic uncertainty:
    C_HR, C_QRS, C_QTc, C_AXIS, C_ST_ANT, C_R_PROG.
    Categorizes evidence strictly via nullspace projection residue:
    - MEASURED (Directly Observable, rho < 0.05)
    - MODEL_ESTIMATED (Partially Observable, 0.05 <= rho < 0.90)
    - UNSUPPORTED / DECLINED (Nullspace, rho >= 0.90)

Module D: Transparent Diagnostic Inference
    Additive, auditable log-odds:
        ell_SHD = b + sum_{j=1}^6 f_j(C_j) + f_tab(x_tab)
    No hidden predictive bypass.
    Uncertainty integration:
        p_hat = E_{C ~ N(mu, sigma^2)} [ sigma(ell(C)) ]

Model Modes:
    1. 'strict': 100% transparent concept-only decision bottleneck.
    2. 'hybrid': Adds an explicit residual branch to quantify latent information outside concepts.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple, Literal
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.echonext_operator_benchmark.measurement_information import (
    compute_noiseless_observable_torch,
)
from experiments.echonext_operator_benchmark.clinical_concepts import (
    CLINICAL_CONCEPTS,
    CONCEPT_KEYS,
    NUM_CONCEPTS,
    audit_concepts_observability,
)


class TemporalResBlock1D(nn.Module):
    """Residual 1D convolutional block for multichannel ECG waveforms."""

    def __init__(self, channels: int, kernel_size: int = 7, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + res)


class TemporalWaveformEncoder(nn.Module):
    """Module B: Temporal ECG Evidence Encoder.

    Maps canonical observable signal X_obs in R^(B, 8, T) and geometry P_Q into
    representation embedding z_obs in R^(B, hidden_dim).
    """

    def __init__(
        self,
        in_leads: int = 8,
        base_channels: int = 32,
        out_dim: int = 128,
    ):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv1d(in_leads, base_channels, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        # Downsampling temporal stages
        self.stage1 = TemporalResBlock1D(base_channels, kernel_size=7, dilation=1)
        self.down1 = nn.Sequential(
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(inplace=True),
        )
        self.stage2 = TemporalResBlock1D(base_channels * 2, kernel_size=7, dilation=2)
        self.down2 = nn.Sequential(
            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(base_channels * 4),
            nn.ReLU(inplace=True),
        )
        self.stage3 = TemporalResBlock1D(base_channels * 4, kernel_size=7, dilation=4)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_wave = nn.Linear(base_channels * 4, out_dim)

        # Geometry encoder for projection matrix P_Q (8x8 = 64 inputs -> 32-D)
        self.geom_mlp = nn.Sequential(
            nn.Linear(in_leads * in_leads, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
        )

        self.out_dim = out_dim + 32

    def forward(self, X_obs: torch.Tensor, P_Q: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            X_obs: (B, 8, T)
            P_Q: (B, 8, 8)

        Returns:
            z_obs: (B, out_dim)
        """
        B = X_obs.shape[0]
        # Temporal path
        h = self.in_conv(X_obs)
        h = self.stage1(h)
        h = self.down1(h)
        h = self.stage2(h)
        h = self.down2(h)
        h = self.stage3(h)
        h = self.global_pool(h).squeeze(-1)
        z_wave = self.fc_wave(h)  # (B, out_dim)

        # Geometry path
        P_flat = P_Q.reshape(B, -1)
        z_geom = self.geom_mlp(P_flat)  # (B, 32)

        return torch.cat([z_wave, z_geom], dim=-1)  # (B, out_dim + 32)


class ConceptHead(nn.Module):
    """Predicts a single concept mean and log-variance from evidence embedding."""

    def __init__(self, in_features: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),  # [mu, log_var]
        )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.net(z)
        mu = out[:, 0]
        log_var = torch.clamp(out[:, 1], min=-6.0, max=4.0)
        sigma = torch.exp(0.5 * log_var)
        return mu, sigma


class UnivariateConceptFunction(nn.Module):
    """Transparent nonlinear component f_j(c_j) with zero-mean centering constraint."""

    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        # Running mean for empirical zero-centering E_train[f_j(C_j)] = 0
        self.register_buffer("running_mean", torch.zeros(1))
        self.register_buffer("count", torch.zeros(1))

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """Compute f_j(c). c: (B,) or (B, N_samples). Returns same shape."""
        orig_shape = c.shape
        c_in = c.reshape(-1, 1)
        val = self.mlp(c_in).reshape(orig_shape)
        
        if self.training:
            batch_mean = torch.mean(val.detach())
            self.running_mean = 0.95 * self.running_mean + 0.05 * batch_mean
            
        return val - self.running_mean


class OCSetOperator(nn.Module):
    """Observable-Concept SetOperator (OC-SetOperator).

    Architecture:
        1. Module A: Linear Canonicalization (Q^pinv Y -> X_obs, P_Q)
        2. Module B: Temporal Evidence Encoder
        3. Module C: Observable Clinical Concept Heads (6 concepts, mu & sigma)
        4. Module D: Transparent Additive Diagnostic Inference (b + sum f_j(c_j) + f_tab(x_tab))
    """

    def __init__(
        self,
        mode: Literal["strict", "hybrid"] = "strict",
        in_leads: int = 8,
        num_tabular: int = 7,
        num_samples_mc: int = 32,
    ):
        super().__init__()
        self.mode = mode
        self.num_samples_mc = num_samples_mc
        self.integration_mode = "probit_approx"
        self.concept_keys = CONCEPT_KEYS
        self.num_concepts = NUM_CONCEPTS

        # Module B: Temporal evidence encoder
        self.encoder = TemporalWaveformEncoder(in_leads=in_leads, base_channels=32, out_dim=128)
        enc_dim = self.encoder.out_dim  # 160

        # Module C: 6 Clinical concept heads
        self.concept_heads = nn.ModuleDict({
            key: ConceptHead(in_features=enc_dim, hidden_dim=64)
            for key in self.concept_keys
        })

        # Module D: Transparent decision functions f_j(c_j)
        self.f_concepts = nn.ModuleDict({
            key: UnivariateConceptFunction(hidden_dim=16)
            for key in self.concept_keys
        })

        # Tabular linear predictor
        self.f_tab = nn.Linear(num_tabular, 1, bias=False) if num_tabular > 0 else None

        # Base bias b
        self.bias = nn.Parameter(torch.zeros(1))

        # Explicit residual branch for 'hybrid' mode only (quantifies uncaptured disease info)
        if self.mode == "hybrid":
            self.residual_head = nn.Sequential(
                nn.Linear(enc_dim, 16),
                nn.ReLU(inplace=True),
                nn.Linear(16, 1, bias=False),
            )
        else:
            self.residual_head = None

    def forward(
        self,
        Q: torch.Tensor,
        Y: torch.Tensor,
        tabular: Optional[torch.Tensor] = None,
        return_evidence_report: bool = False,
    ) -> Dict[str, Any]:
        """Forward pass.

        Args:
            Q: (B, m, 8) Measurement operator matrix
            Y: (B, m, T) Measured lead signals
            tabular: Optional (B, 7) tabular features
            return_evidence_report: If True, computes detailed patient evidence breakdown.

        Returns:
            Dict containing:
                - 'logits': (B, 1) predicted log-odds
                - 'probs': (B, 1) predicted probability
                - 'concept_means': Dict[str, Tensor (B,)]
                - 'concept_sigmas': Dict[str, Tensor (B,)]
                - 'concept_contributions': Dict[str, Tensor (B,)]
                - 'observability': Dict[str, Any] (if return_evidence_report)
        """
        B = Q.shape[0]

        # 1. Module A: Exact linear canonicalization
        X_obs, P_Q, P_perp = compute_noiseless_observable_torch(Q, Y, rcond=1e-5, normalize_rows=True)

        # 2. Module B: Temporal evidence encoding
        z_obs = self.encoder(X_obs, P_Q)  # (B, enc_dim)

        # 3. Module C: Concept estimation with uncertainty
        concept_means: Dict[str, torch.Tensor] = {}
        concept_sigmas: Dict[str, torch.Tensor] = {}
        concept_contribs: Dict[str, torch.Tensor] = {}

        for key in self.concept_keys:
            mu_j, sigma_j = self.concept_heads[key](z_obs)
            concept_means[key] = mu_j
            concept_sigmas[key] = sigma_j
            # Transparent univariate contribution at mean estimate
            concept_contribs[key] = self.f_concepts[key](mu_j)

        # 4. Module D: Transparent additive log-odds
        # Deterministic log-odds at expected concept values:
        ell_deterministic = self.bias.expand(B)
        for key in self.concept_keys:
            ell_deterministic = ell_deterministic + concept_contribs[key]

        if tabular is not None and self.f_tab is not None:
            tab_contrib = self.f_tab(tabular).squeeze(-1)
            ell_deterministic = ell_deterministic + tab_contrib
        else:
            tab_contrib = torch.zeros(B, device=Q.device)

        # Residual term for hybrid mode
        if self.mode == "hybrid" and self.residual_head is not None:
            res_contrib = self.residual_head(z_obs).squeeze(-1)
            ell_deterministic = ell_deterministic + res_contrib
        else:
            res_contrib = torch.zeros(B, device=Q.device)

        # Uncertainty Integration over conditional distribution:
        # p_hat = E_{C ~ N(mu, sigma^2)} [ sigma(ell(C)) ]
        if self.integration_mode == "deterministic" or self.training:
            probs = torch.sigmoid(ell_deterministic).unsqueeze(-1)
            logits = ell_deterministic.unsqueeze(-1)
        elif self.integration_mode == "probit_approx":
            # Deterministic closed-form MacKay Gaussian-logistic convolution:
            # int sigma(x) N(x | mu, sigma^2) dx approx sigma( mu / sqrt(1 + pi * sigma^2 / 8) )
            total_var = torch.zeros_like(ell_deterministic)
            for key in self.concept_keys:
                mu_val = concept_means[key]
                h = 1e-3
                f_plus = self.f_concepts[key](mu_val + h)
                f_minus = self.f_concepts[key](mu_val - h)
                df_dc = (f_plus - f_minus) / (2.0 * h)
                total_var = total_var + (df_dc ** 2) * (concept_sigmas[key] ** 2)

            kappa = torch.sqrt(1.0 + (math.pi / 8.0) * total_var)
            attenuated_logits = ell_deterministic / kappa
            probs = torch.sigmoid(attenuated_logits).unsqueeze(-1)
            logits = ell_deterministic.unsqueeze(-1)
        else:
            # Monte Carlo integration across num_samples_mc samples
            eps = torch.randn(B, self.num_samples_mc, device=Q.device)
            ell_samples = self.bias.expand(B, self.num_samples_mc) + tab_contrib.unsqueeze(-1) + res_contrib.unsqueeze(-1)
            for key in self.concept_keys:
                c_samples = concept_means[key].unsqueeze(-1) + concept_sigmas[key].unsqueeze(-1) * eps
                f_sample = self.f_concepts[key](c_samples)
                ell_samples = ell_samples + f_sample
            probs = torch.mean(torch.sigmoid(ell_samples), dim=-1, keepdim=True)
            logits = ell_deterministic.unsqueeze(-1)

        result: Dict[str, Any] = {
            "logits": logits,
            "probs": probs,
            "concept_means": concept_means,
            "concept_sigmas": concept_sigmas,
            "concept_contributions": concept_contribs,
            "tab_contribution": tab_contrib,
            "residual_contribution": res_contrib,
            "P_Q": P_Q,
            "X_obs": X_obs,
        }

        if return_evidence_report and B == 1:
            # Generate patient evidence breakdown
            obs_audit = audit_concepts_observability(P_Q[0])
            report_lines = []
            report_lines.append(f"Predicted SHD Risk: {probs.item() * 100:.1f}% (Log-odds: {logits.item():+.3f})")
            report_lines.append(f"Base prevalence bias: {self.bias.item():+.3f}")
            if self.f_tab is not None and tabular is not None:
                report_lines.append(f"Tabular contribution: {tab_contrib.item():+.3f}")
            if self.mode == "hybrid":
                report_lines.append(f"Unmodeled residual latent contribution: {res_contrib.item():+.3f}")
            
            report_lines.append("\nClinical Concept Evidence Table:")
            for key in self.concept_keys:
                meta = CLINICAL_CONCEPTS[key]
                obs_status = obs_audit[key]["status"]
                val = concept_means[key].item()
                unc = concept_sigmas[key].item()
                contrib = concept_contribs[key].item()
                residue = obs_audit[key]["nullspace_residue"]
                report_lines.append(
                    f" - {meta.name:32s} [{meta.symbol:8s}]: {val:7.2f} +/- {1.96*unc:5.2f} {meta.unit:4s} | "
                    f"Status: {obs_status:19s} (Nullspace: {residue*100:4.1f}%) | Contribution: {contrib:+.3f} log-odds"
                )
            result["patient_report"] = "\n".join(report_lines)
            result["observability_audit"] = obs_audit

        return result
