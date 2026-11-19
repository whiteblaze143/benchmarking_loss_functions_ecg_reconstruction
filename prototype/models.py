"""
Model Architecture Benchmark Suite for Wearable AF Recurrence Detection.
Implements:
1) RR-based detector (ArNet2)
2) Raw ECG CNN (RawECGNet / ArNetECG)
3) SetOperator Phase-KME
4) SetOperator + RR (Multimodal Morphology-Rhythm Fusion)
5) Quality-aware SetOperator with Selective Abstention
"""
from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from sqi import compute_composite_sqi


# ============================================================================
# 1. Baseline: RR-based Detector (ArNet2 Architecture)
# ============================================================================

class ArNet2ResBlock(nn.Module):
    """Residual block for 1D RR interval sequence (He et al. style)."""
    def __init__(self, channels: int, kernel_size: int = 5):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + res)


class ArNet2Detector(nn.Module):
    """
    ArNet2: Deep learning AF detector from beat-to-beat (RR interval) data.
    Part 1: 1D ResNet embedding over 60-beat RR window.
    Part 2: Bidirectional GRU sequence modeling across temporal context.
    """
    def __init__(self, window_size: int = 60, embedding_dim: int = 64, num_classes: int = 1):
        super().__init__()
        self.window_size = window_size
        self.input_proj = nn.Sequential(
            nn.Conv1d(1, embedding_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
        )
        self.res_blocks = nn.Sequential(
            ArNet2ResBlock(embedding_dim, kernel_size=5),
            ArNet2ResBlock(embedding_dim, kernel_size=5),
            nn.MaxPool1d(2),
            ArNet2ResBlock(embedding_dim, kernel_size=3),
            nn.MaxPool1d(2),
        )
        # Sequence modeling over temporal beat embeddings
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, rr_intervals: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rr_intervals: (batch, 60) or (batch, 1, 60) in seconds.
        Returns:
            logits: (batch, 1) AF detection logit.
        """
        if rr_intervals.ndim == 2:
            x = rr_intervals.unsqueeze(1)
        else:
            x = rr_intervals

        # Scale normalization (expected RR ~ 0.3s - 2.0s)
        x = (x - 0.8) / 0.3
        feat = self.input_proj(x)
        feat = self.res_blocks(feat)  # (batch, channels, time_downsampled)
        feat = feat.transpose(1, 2)   # (batch, time, channels)
        gru_out, _ = self.gru(feat)
        pooled = torch.mean(gru_out, dim=1)
        logits = self.head(pooled)
        return logits


# ============================================================================
# 2. Baseline: Raw ECG CNN (RawECGNet / ArNetECG Architecture)
# ============================================================================

class Conv1DBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.shortcut = (
            nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, stride=stride), nn.BatchNorm1d(out_ch))
            if in_ch != out_ch or stride != 1
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + res)


class RawECGNet(nn.Module):
    """
    RawECGNet: Deep CNN operating directly on raw 30-second single-lead ECG (6000 samples at 200 Hz).
    Extracts both morphology (P/f waves) and rhythm directly from continuous waveform.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        # Downsampling stages: 1500 -> 750 -> 375 -> 187 -> 93
        self.stage1 = Conv1DBlock(32, 64, stride=2, kernel_size=7)
        self.stage2 = Conv1DBlock(64, 128, stride=2, kernel_size=7)
        self.stage3 = Conv1DBlock(128, 128, stride=2, kernel_size=5)
        self.stage4 = Conv1DBlock(128, 256, stride=2, kernel_size=5)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, raw_ecg: torch.Tensor) -> torch.Tensor:
        """
        Args:
            raw_ecg: (batch, 6000) or (batch, 1, 6000)
        Returns:
            logits: (batch, 1) AF logit
        """
        if raw_ecg.ndim == 2:
            x = raw_ecg.unsqueeze(1)
        else:
            x = raw_ecg
        out = self.stem(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        pooled = self.global_pool(out).flatten(1)
        return self.classifier(pooled)


# ============================================================================
# 3. Model: SetOperator Phase-KME Encoder
# ============================================================================

class CircularPhaseBlock(nn.Module):
    """Circular 1D residual block respecting the cyclic topology S^1 of the heartbeat."""
    def __init__(self, width: int):
        super().__init__()
        self.conv1 = nn.Conv1d(width, width, 3, padding=0)
        self.conv2 = nn.Conv1d(width, width, 3, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_pad = F.pad(x, (1, 1), mode="circular")
        x = F.gelu(self.conv1(x_pad))
        x_pad = F.pad(x, (1, 1), mode="circular")
        return F.gelu(self.conv2(x_pad) + residual)


class SetOperatorPhaseKME(nn.Module):
    """
    SetOperator Phase-KME:
    1) Takes cardiac phase distributions (16 phase cells per beat).
    2) Maps each phase cell to Reproducing Kernel Hilbert Space (RKHS) embedding.
    3) Applies SetOperator Transformer with circular residual phase blocks and query attention pooling.
    Invariant to heart rate, robust to high-frequency micro-voltage noise.
    """
    def __init__(
        self,
        phase_cells: int = 16,
        feature_dim: int = 128,
        hidden_dim: int = 128,
        num_classes: int = 1,
        num_layers: int = 2,
    ):
        super().__init__()
        self.phase_cells = phase_cells
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # Phase feature projection
        self.input_proj = nn.Conv1d(feature_dim, hidden_dim, 1)
        self.phase_blocks = nn.Sequential(
            CircularPhaseBlock(hidden_dim),
            CircularPhaseBlock(hidden_dim),
        )

        # SetOperator Transformer Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Learnable Query Attention Pooling
        self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.pool_mha = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, phase_kme: torch.Tensor) -> torch.Tensor:
        """
        Args:
            phase_kme: (batch, 16_phases, feature_dim)
        Returns:
            logits: (batch, 1) AF logit
        """
        batch_size = phase_kme.shape[0]
        # Transpose to (batch, feature_dim, 16_phases) for 1D circular conv
        x = phase_kme.transpose(1, 2)
        feat = self.phase_blocks(self.input_proj(x))
        # Transpose back to (batch, 16_phases, hidden_dim) for Transformer
        seq = feat.transpose(1, 2)
        trans_out = self.transformer(seq)

        # Query pooling
        query = self.pool_query.expand(batch_size, -1, -1)
        pooled, _ = self.pool_mha(query, trans_out, trans_out)
        z_morph = pooled.squeeze(1)

        logits = self.head(z_morph)
        return logits

    def extract_morph_latent(self, phase_kme: torch.Tensor) -> torch.Tensor:
        """Extract morphological representation vector z_morph in R^128."""
        batch_size = phase_kme.shape[0]
        x = phase_kme.transpose(1, 2)
        feat = self.phase_blocks(self.input_proj(x))
        seq = feat.transpose(1, 2)
        trans_out = self.transformer(seq)
        query = self.pool_query.expand(batch_size, -1, -1)
        pooled, _ = self.pool_mha(query, trans_out, trans_out)
        return pooled.squeeze(1)


# ============================================================================
# 4. Model: SetOperator + RR (Multimodal Morphology-Rhythm Fusion)
# ============================================================================

class RRRhythmBranch(nn.Module):
    """
    Rhythm branch processing beat-to-beat RR intervals and HRV descriptors.
    Fuses 1D temporal GRU sequence embedding with statistical rhythm features.
    """
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.gru = nn.GRU(32, 64, num_layers=2, batch_first=True, bidirectional=True)
        # HRV stats: mean_rr, sdnn, rmssd, pnn50, cv, sd1, sd2, sampen (8 features)
        self.hrv_proj = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )
        self.proj = nn.Sequential(
            nn.Linear(128 + 32, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, rr_seq: torch.Tensor, hrv_stats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rr_seq: (batch, 60)
            hrv_stats: (batch, 8)
        Returns:
            z_rhythm: (batch, latent_dim)
        """
        if rr_seq.ndim == 2:
            x = rr_seq.unsqueeze(1)
        else:
            x = rr_seq
        # Normalize RR sequence
        x = (x - 0.8) / 0.3
        h = self.conv_in(x).transpose(1, 2)
        gru_out, _ = self.gru(h)
        seq_emb = gru_out.mean(dim=1)  # (batch, 128)

        hrv_emb = self.hrv_proj(hrv_stats)
        concat = torch.cat([seq_emb, hrv_emb], dim=-1)
        return self.proj(concat)


class SetOperatorPlusRR(nn.Module):
    """
    Multimodal Morphology-Rhythm Fusion:
    - Branch A: SetOperator Phase-KME encoder (morphology, f-waves, P-wave absence).
    - Branch B: RR Rhythm Branch (beat-to-beat irregularity, Poincaré dispersion).
    - Gated Bilinear Cross-Attention Fusion.
    """
    def __init__(self, latent_dim: int = 128, num_classes: int = 1):
        super().__init__()
        self.morph_encoder = SetOperatorPhaseKME(
            phase_cells=16, feature_dim=128, hidden_dim=latent_dim, num_classes=num_classes
        )
        self.rhythm_encoder = RRRhythmBranch(latent_dim=latent_dim)

        # Cross-gating projection
        self.gate_morph = nn.Linear(latent_dim, latent_dim)
        self.gate_rhythm = nn.Linear(latent_dim, latent_dim)
        self.norm = nn.LayerNorm(latent_dim * 2)

        self.head = nn.Sequential(
            nn.Linear(latent_dim * 2, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        phase_kme: torch.Tensor,
        rr_seq: torch.Tensor,
        hrv_stats: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            phase_kme: (batch, 16, 128)
            rr_seq: (batch, 60)
            hrv_stats: (batch, 8)
        Returns:
            logits: (batch, 1)
        """
        z_morph = self.morph_encoder.extract_morph_latent(phase_kme)
        z_rhythm = self.rhythm_encoder(rr_seq, hrv_stats)

        # Multimodal Gating
        g_m = torch.sigmoid(self.gate_rhythm(z_rhythm))
        g_r = torch.sigmoid(self.gate_morph(z_morph))

        z_m_gated = z_morph * g_m
        z_r_gated = z_rhythm * g_r

        fused = self.norm(torch.cat([z_m_gated, z_r_gated], dim=-1))
        logits = self.head(fused)
        return logits

    def predict_proba(
        self,
        phase_kme: torch.Tensor,
        rr_seq: torch.Tensor,
        hrv_stats: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.forward(phase_kme, rr_seq, hrv_stats)
        return torch.sigmoid(logits)


# ============================================================================
# 5. Model: Quality-Aware SetOperator with Abstention Policy
# ============================================================================

class QualityAwareSetOperator(nn.Module):
    """
    Quality-Aware SetOperator with Cost-Sensitive Selective Abstention.
    Couples the multimodal SetOperator+RR architecture with:
      1) Signal Quality Index (SQI) evaluator (bSQI, pSQI, kSQI, bwSQI).
      2) Epistemic/Predictive Uncertainty estimation.
      3) Chow's Rule selective classifier:
         If Q(x) < tau_qual OR Uncertainty(x) > tau_unc:
             Output: ABSTAIN (Window omitted from recurrence clock)
         Else:
             Output: AF if P(AF) >= tau_p else NSR.
    """
    def __init__(
        self,
        latent_dim: int = 128,
        tau_qual: float = 0.55,
        tau_unc: float = 0.35,
        tau_p: float = 0.50,
    ):
        super().__init__()
        self.backbone = SetOperatorPlusRR(latent_dim=latent_dim, num_classes=1)
        self.tau_qual = tau_qual
        self.tau_unc = tau_unc
        self.tau_p = tau_p

    def forward(
        self,
        phase_kme: torch.Tensor,
        rr_seq: torch.Tensor,
        hrv_stats: torch.Tensor,
    ) -> torch.Tensor:
        return self.backbone(phase_kme, rr_seq, hrv_stats)

    def predict_with_abstention(
        self,
        phase_kme: torch.Tensor,
        rr_seq: torch.Tensor,
        hrv_stats: torch.Tensor,
        raw_signals: np.ndarray,
        fs: float = 200.0,
    ) -> dict[str, np.ndarray]:
        """
        Execute prediction with quality-aware selective abstention.
        Args:
            phase_kme: (batch, 16, 128)
            rr_seq: (batch, 60)
            hrv_stats: (batch, 8)
            raw_signals: (batch, N_samples) numpy array for SQI calculation
            fs: sampling frequency
        Returns:
            dict containing:
              - proba: predicted AF probability
              - pred: binary classification (1=AF, 0=NSR) for accepted windows
              - abstain: boolean mask (True if abstained)
              - sqi_scores: composite Q values
              - uncertainty: predictive entropy/uncertainty
        """
        self.eval()
        with torch.no_grad():
            logits = self.backbone(phase_kme, rr_seq, hrv_stats)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

        batch_size = len(probs)
        sqi_scores = np.zeros(batch_size, dtype=np.float32)
        for i in range(batch_size):
            sqi_dict = compute_composite_sqi(raw_signals[i], fs=fs)
            sqi_scores[i] = sqi_dict["Q"]

        # Predictive uncertainty: 1.0 - max(p, 1-p) in [0, 0.5]
        # Max uncertainty is 0.5 (p=0.5), min is 0.0 (p=0 or 1)
        uncertainty = 1.0 - np.maximum(probs, 1.0 - probs)

        # Abstention Condition
        abstain_mask = (sqi_scores < self.tau_qual) | (uncertainty > self.tau_unc)

        preds = (probs >= self.tau_p).astype(int)
        # Marked as -1 for abstained windows
        masked_preds = np.where(abstain_mask, -1, preds)

        return {
            "proba": probs,
            "pred": preds,
            "final_decision": masked_preds,  # 1=AF, 0=NSR, -1=ABSTAIN
            "abstain": abstain_mask,
            "sqi_q": sqi_scores,
            "uncertainty": uncertainty,
        }
