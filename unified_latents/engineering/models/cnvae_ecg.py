#!/usr/bin/env python3
"""
cNVAE-ECG Hierarchical Regularized Autoencoder (Phase 45).

Implements the Phase 3 unified-latents design:
  - at inference, the FM is an evidence-pure encoder
  - at training, the same frozen FM acts as a full-12-lead teacher
  - the cNVAE latent hierarchy, not the FM token tensor, is the shared
    physiological state across observation regimes
  - the decoder emits a universal 8-head bank and regime logic determines
    which heads are supervised and used in reconstruction

Coarse latent scales are intended to capture global rhythm and broad
cross-lead morphology; fine scales capture local QRS/ST-T detail.

References:
  - Vahdat & Kautz (2020) NVAE (DOI: 10.48550/arxiv.2007.03898)
  - McKeen et al. (2025) ECG-FM (DOI: 10.1093/jamiaopen/ooaf122)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

def _setup_fairseq_path():
    """Robustly add fairseq-signals to path using project root relative to this file."""
    benchmark_root = Path(__file__).resolve().parents[3]
    engineering_root = benchmark_root / "unified_latents" / "engineering"
    candidates = [
        benchmark_root / "ecg_fm_integration" / "fairseq-signals",
        engineering_root / "third_party" / "WearECG-reconstruction" / "ecg_fm_integration" / "fairseq-signals",
    ]
    for fs_path in candidates:
        if fs_path.exists():
            fs_path_str = str(fs_path)
            if fs_path_str not in sys.path:
                sys.path.insert(0, fs_path_str)
            break

@torch.jit.script
def soft_clamp5(x: torch.Tensor):
    """Stability fix: Differentially clamp values to [-5, 5] to prevent NaNs in deep KL hierarchies."""
    return x.div(5.0).tanh().mul(5.0)

_setup_fairseq_path()

# =============================================================================
# ECG CONVOLUTIONAL CORE ARCHITECTURAL BLOCKS
# =============================================================================

@torch.jit.script
def normalize_weight_jit(log_weight_norm, weight):
    """JIT optimized weight normalization (Reference Parity)."""
    n = torch.exp(log_weight_norm)
    wn = torch.sqrt(torch.sum(weight * weight, dim=[1, 2]))
    weight = n * weight / (wn.view(-1, 1, 1) + 1e-5)
    return weight

class Conv1D(nn.Conv1d):
    """Conv1d with integrated Weight Normalization and JIT normalization logic."""
    def __init__(self, C_in, C_out, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, 
                 weight_norm=True):
        super().__init__(C_in, C_out, kernel_size, stride, padding, dilation, groups, bias)
        self.log_weight_norm = None
        if weight_norm:
            # norm(w) initialization
            w_norm = torch.sqrt(torch.sum(self.weight ** 2, dim=[1, 2])).view(-1, 1, 1)
            self.log_weight_norm = nn.Parameter(torch.log(w_norm + 1e-2), requires_grad=True)

    def forward(self, x):
        weight = self.weight
        if self.log_weight_norm is not None:
            weight = normalize_weight_jit(self.log_weight_norm, self.weight)
        return F.conv1d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class NVAECell(nn.Module):
    """
    Absolute NVAE Parity Cell.
    Topology: Skip Path + 0.1 * (BN -> Act -> Conv -> BN -> Act -> Conv -> SE)
    Matches 'Cell' in reference model_conditional_1d.py and BNELUConv/BNSwishConv.
    """
    def __init__(self, Cin, Cout, stride=1, kernel_size=3, use_se=True, activation='swish'):
        super().__init__()
        self.stride = stride
        self.use_se = use_se
        
        # OPS in reference: lambda Cin, Cout, stride: BNSwishConv(Cin, Cout, 3, stride, 1)
        # We wrap the two-conv residual block logic into one NVAE-compliant cell.
        self.bn1 = nn.BatchNorm1d(Cin, eps=1e-5, momentum=0.05)
        self.act = nn.SiLU() if activation == 'swish' else nn.ELU()
        
        padding = kernel_size // 2
        self.conv1 = Conv1D(Cin, Cout, kernel_size, stride=stride, padding=padding)
        
        self.bn2 = nn.BatchNorm1d(Cout, eps=1e-5, momentum=0.05)
        self.conv2 = Conv1D(Cout, Cout, kernel_size, stride=1, padding=padding)
        
        if use_se:
            self.se = SqueezeExcitation1D(Cout)
            
        # Skip connection: Identity or FactorizedReduce
        if stride == 1 and Cin == Cout:
            self.skip = nn.Identity()
        elif stride == 2:
            self.skip = FactorizedReduce(Cin, Cout)
        else:
            self.skip = Conv1D(Cin, Cout, 1, stride=stride)

    def forward(self, x):
        skip = self.skip(x)
        
        # Pre-activation residual branch
        out = self.bn1(x)
        out = self.act(out)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.act(out)
        out = self.conv2(out)
        
        if self.use_se:
            out = self.se(out)
            
        # 0.1 scaling: absolute reference parity
        return skip + 0.1 * out

class DepthwiseECGConvBlock1D(nn.Module):
    """Computionally efficient ECGConv block using depthwise-separable convolutions."""
    def __init__(self, input_channel, output_channel, kernel_size=17, stride_size=1, 
                 inner_activation='relu', output_activation='relu', use_residual_block=True, average_pool=1):
        super().__init__()
        # Depthwise-separable refinement
        self.inner_conv = nn.Sequential(
            nn.Conv1d(input_channel, input_channel, kernel_size, groups=input_channel, padding='same'),
            nn.Conv1d(input_channel, output_channel, 1),
            nn.ELU() if inner_activation == 'elu' else nn.ReLU()
        )
        
        padding = int(kernel_size / 2) if stride_size > 1 and kernel_size % 2 == 1 else 'same'
        self.out_conv = nn.Sequential(
            nn.Conv1d(output_channel, output_channel, kernel_size, groups=output_channel, stride=stride_size, padding=padding),
            nn.Conv1d(output_channel, output_channel, 1)
        )
        
        self.res_conv = nn.Conv1d(input_channel, output_channel, 1, stride=stride_size) if use_residual_block else None
        self.output_act = nn.ReLU() if output_activation == 'relu' else nn.Identity()
        self.average_pool = nn.AvgPool1d(average_pool) if average_pool > 1 else None

    def forward(self, x):
        residual = x
        x = self.inner_conv(x)
        x = self.out_conv(x)
        if self.res_conv is not None:
            x = self.res_conv(residual) + 0.1 * x
        x = self.output_act(x)
        if self.average_pool is not None:
            x = self.average_pool(x)
        return x

class DepthwiseECGConvNetwork1D(nn.Module):
    """Sequence of Depthwise-Separable ECGConv blocks."""
    def __init__(self, input_channel, output_channel, block_num, kernel_size=17, stride_size=1, average_pool=1):
        super().__init__()
        self.blocks = nn.ModuleList()
        channels = np.linspace(input_channel, output_channel, block_num + 1).astype(int)
        for i in range(block_num):
            self.blocks.append(DepthwiseECGConvBlock1D(
                channels[i], channels[i+1], kernel_size, 
                stride_size if i == block_num - 1 else 1,
                average_pool=average_pool if i == block_num - 1 else 1
            ))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class FactorizedReduce(nn.Module):
    """
    Principled downsampling for Phase 28 (Reference Parity).
    Avoids aliasing by concatenating two strided convolutions with different spatial indices.
    """
    def __init__(self, C_in, C_out):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        # Reference uses act() before convs
        self.act = nn.SiLU()
        self.conv_1 = Conv1D(C_in, C_out // 2, 1, stride=2, padding=0, bias=True)
        self.conv_2 = Conv1D(C_in, C_out - (C_out // 2), 1, stride=2, padding=0, bias=True)

    def forward(self, x):
        x = self.act(x)
        conv1 = self.conv_1(x)
        # Fix to match reference size behavior for odd dimensions
        conv2 = self.conv_2(x[:, :, 1:])
        if conv2.size(2) < conv1.size(2):
            conv2 = F.pad(conv2, (0, 1), mode='replicate')
        
        return torch.cat([conv1, conv2], dim=1)

# =========================================================================
# CLINICAL 12-LEAD PROTOCOL CONSTANTS
# =========================================================================
# Lead naming is centralized here so this cNVAE branch can move from the
# historical fixed-basis setup toward a sparse, regime-masked reconstruction
# setting. The decoder always emits the canonical independent head bank:
#   I, II, V1, V2, V3, V4, V5, V6
# while regime logic decides which heads are supervised and used in assembly.
# Dependent limb leads are completed analytically.
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_NAME_TO_INDEX = {name: idx for idx, name in enumerate(LEAD_NAMES)}
INDEPENDENT_LIMB_INDICES = [0, 1, 2]  # I, II, III
CANONICAL_INDEPENDENT_INDICES = [0, 1, 6, 7, 8, 9, 10, 11]  # I, II, V1-V6
HEAD_INDEX_BY_LEAD = {lead_idx: head_idx for head_idx, lead_idx in enumerate(CANONICAL_INDEPENDENT_INDICES)}

# This branch is not a fully general sparse wearable model. It supports:
# - exact I-only input
# - analytical multi-limb regimes with at least two observed limb leads
# The decoder always emits a fixed universal 8-head bank (I, II, V1-V6); regime
# logic decides which heads are supervised and used in assembly.
ECG_MIN = -2.5
ECG_AMP = 5.0


def obs_indices_to_names(obs_indices):
    return [LEAD_NAMES[idx] for idx in obs_indices]


def validate_obs_indices(obs_indices):
    """
    Supported regimes for this cNVAE branch:
    - smartwatch one-limb regime: exactly {I}
    - analytical multi-limb regimes: at least two observed leads from {I, II, III}

    This intentionally excludes II-only, III-only, or one-limb-plus-chest
    regimes because the learned-target convention is only defined for I-only.
    """
    unique = sorted(set(int(i) for i in obs_indices))
    limb_count = sum(idx in INDEPENDENT_LIMB_INDICES for idx in unique)
    if unique == [LEAD_NAME_TO_INDEX["I"]]:
        return unique
    if limb_count >= 2:
        return unique

    names = ", ".join(obs_indices_to_names(unique))
    raise ValueError(
        "cNVAE sparse reconstruction currently supports either exact I-only "
        "input or regimes with at least two observed independent limb leads "
        f"from {{I, II, III}}. Got: [{names}]"
    )

def denormalize_ecg(lead, min_value=ECG_MIN, amplitude=ECG_AMP):
    """Maps normalized [0,1] output back to mV scale."""
    return lead * amplitude + min_value

# =============================================================================
# HIERARCHICAL VAE COMPONENTS
# =============================================================================

class CNVAE_DecCombinerCell(nn.Module):
    """
    Combines top-down prior features (s) with sampled latent z.
    A 1×1 convolution fuses the concatenated channels.
    """
    def __init__(self, Cin_s, Cin_z, Cout):
        super().__init__()
        self.conv = Conv1D(Cin_s + Cin_z, Cout, kernel_size=1, bias=True)

    def forward(self, s, z):
        # Align temporal dimensions if needed (R1/C6: Interpolation over Padding)
        if s.size(2) != z.size(2):
            z = F.interpolate(z, size=s.size(2), mode='linear', align_corners=False)
        return self.conv(torch.cat([s, z], dim=1))

class CNVAE_EncCombinerCell(nn.Module):
    """
    Combines top-down features (s) with bottom-up FM features (bu)
    to parameterize the posterior q(z | s, bu).
    Uses ADDITIVE combination for semantic parity with cNVAE-ECG.
    """
    def __init__(self, Cin_s, Cin_bu, Cout):
        super().__init__()
        # R4: Semantic Parity - project TD to match BU, then add.
        self.conv_s = Conv1D(Cin_s, Cin_bu, kernel_size=1)
        # R5: Weight normalization on the final sampling head
        # Weight parity: using Conv1D with weight_norm=True
        self.sampler = Conv1D(Cin_bu, Cout, kernel_size=1, bias=True, weight_norm=True)
        nn.init.normal_(self.sampler.weight, std=0.001)
        if self.sampler.bias is not None: nn.init.zeros_(self.sampler.bias)

    def forward(self, s, bu):
        # Align dimensions if needed (R1/C6: Interpolation over Padding)
        if s.size(2) != bu.size(2):
            bu = F.interpolate(bu, size=s.size(2), mode='linear', align_corners=False)
        
        # R4: Additive combination
        feat = bu + self.conv_s(s)
        return self.sampler(feat)

class CNVAE_PriorNet(nn.Module):
    """Predicts prior distribution params p(z | s) from top-down features."""
    def __init__(self, Cin, Cout):
        super().__init__()
        # R5: Weight normalization for distribution head stability (Reference Parity)
        self.conv = Conv1D(Cin, Cout, kernel_size=1, weight_norm=True)

    def forward(self, s):
        return self.conv(s)

class CNVAE_PosteriorNet(nn.Module):
    """Predicts posterior distribution params q(z | s, bu) as residual to prior."""
    def __init__(self, Cin_s, Cin_bu, Cout):
        super().__init__()
        self.combiner = CNVAE_EncCombinerCell(Cin_s, Cin_bu, Cout)

    def forward(self, s, bu):
        return self.combiner(s, bu)

class Normal:
    """Soft-clamped diagonal Gaussian with effective sigma for stable recovery (Reference Parity)."""
    def __init__(self, mu, log_sigma):
        # Full reference parity: clamp BOTH mu and log_sigma
        self.mu = soft_clamp5(mu)
        self.log_sigma = soft_clamp5(log_sigma)
        
        # sigma = exp(log_sigma) + floor
        self.sigma_eff = torch.exp(self.log_sigma) + 1e-2
        self.log_sigma_eff = torch.log(self.sigma_eff)

    def sample(self):
        return self.mu + self.sigma_eff * torch.randn_like(self.sigma_eff)

    def log_p(self, z):
        return -0.5 * (np.log(2 * np.pi) + 2 * self.log_sigma_eff + ((z - self.mu) / self.sigma_eff) ** 2)

    @staticmethod
    def kl(q, p):
        """Analytical KL Divergence between two diagonal Gaussians using effective sigma."""
        # Standard KL formula using sigma_eff to ensure ELBO matches sampled distribution.
        return -0.5 + p.log_sigma_eff - q.log_sigma_eff + 0.5 * (q.sigma_eff**2 + (q.mu - p.mu) ** 2) / (p.sigma_eff**2)

# =============================================================================
# ATTENTION & SCALE BLOCKS
# =============================================================================

class SqueezeExcitation1D(nn.Module):
    """R3: Squeeze-and-Excitation block for cross-channel importance (Reference Parity)."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        # Bottleneck sync: max(Cout // 16, 4)
        reduced = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, T = x.shape
        y = torch.mean(x, dim=2) # Global Average Pooling
        y = self.fc(y).view(B, C, 1)
        return x * y

class CrossLeadAttention(nn.Module):
    """A1: Enforces consistent spatial mixing across the decoder output leads."""
    def __init__(self, channels, n_leads=6, num_heads=2):
        super().__init__()
        self.n_leads = n_leads
        self.c_per_lead = channels // n_leads
        self.attn = nn.MultiheadAttention(embed_dim=self.c_per_lead, num_heads=num_heads, batch_first=True)
        
        # C4: Principled lead-specific projections
        # Each lead is formed by an independent projection from the aggregate global features.
        self.lead_projs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(channels, self.c_per_lead, kernel_size=1), nn.ELU())
            for _ in range(n_leads)
        ])
        
        # Zero-initialized residual to start identical to baseline
        self.res_weight = nn.Parameter(torch.zeros(1))

    def forward(self, s):
        """s: (B, channels, T)"""
        B, C, T = s.shape
        # C4: Form lead-specific features (B, n_leads, c_per_lead, T)
        leads = [proj(s) for proj in self.lead_projs]
        x = torch.stack(leads, dim=1) # (B, 6, 16, T)
        
        # N7: Safe Batch-wise loop for CUDA stability.
        # While B*T flattening is theoretically possible, it often hits grid/dispatcher limits
        # in full model contexts (e.g. B=32, T=5000 -> 160k batch elements).
        x_perm = x.permute(0, 3, 1, 2) # (B, T, 6, 16)
        
        attn_results = []
        for b in range(B):
            # (T, 6, 16) - MHA treats T as a batch dimension internally.
            sample_attn, _ = self.attn(x_perm[b], x_perm[b], x_perm[b], need_weights=False)
            attn_results.append(sample_attn)
        
        attn_out = torch.stack(attn_results) # (B, T, 6, 16)
        
        # Reshape back to (B, C, T)
        attn_out = attn_out.permute(0, 2, 3, 1).reshape(B, C, T)
        return s + self.res_weight * attn_out

class TemporalSelfAttention(nn.Module):
    """A2: Captures long-range temporal dependencies at the finest scale."""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.res_weight = nn.Parameter(torch.zeros(1))

    def forward(self, s):
        """s: (B, channels, T)"""
        B, C, T = s.shape
        x = s.transpose(1, 2)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        return s + self.res_weight * attn_out.transpose(1, 2)

class FMTokenAttention(nn.Module):
    """A3: Re-contextualizes frozen FM masked autoencoder features."""
    def __init__(self, embed_dim=768, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(embed_dim)
        # C8 & N4: Sequential Training Gating
        # LayerNorm is placed on the residual branch to ensure identity at init.
        self.res_weight = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """x: (B, T_enc, embed_dim)"""
        # N4: Identity at initialization (res_weight=0)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        return x + self.res_weight * self.norm(attn_out)


# =============================================================================
# TOP-DOWN HIERARCHICAL DECODER
# =============================================================================

class CNVAE_CardiacStateDecoder(nn.Module):
    """
    Hierarchical VAE Decoder using ECG-FM as the bottom-up inference path.
    Reconstructs the canonical independent lead set from a hierarchy of latent
    scales:
      I, II, V1, V2, V3, V4, V5, V6

    Theory:
    - FM tokens provide frozen evidence features, not the final unified latent.
    - A hierarchical posterior q(z | x) combines top-down state with those FM
      features at each scale.
    - The cNVAE latent hierarchy is the shared cardiac-state representation.
    - The decoder emits a universal 8-head bank, while regime logic decides
      which heads are supervised or ignored.
    - Limb-lead completion is handled outside this class via analytical
      reconstruction from the observed limb basis.
    """
    def __init__(self, fm_embed_dim=768, target_len=5000, n_output_leads=len(CANONICAL_INDEPENDENT_INDICES), 
                 num_latent_scales=4, channels=96, latent_dim=32, kernel_size=17, use_residual=True):
        super().__init__()
        self.target_len = target_len
        self.num_latent_scales = num_latent_scales
        self.channels = channels
        self.latent_dim = latent_dim

        # 1. Bottom-Up (BU) Pathway: Downsample FM tokens to hierarchical scales.
        # We use depthwise-separable convolutions for massive parameter savings.
        self.bu_downsamplers = nn.ModuleList()
        self.bu_projs = nn.ModuleList()
        for s in range(num_latent_scales):
            # resolution progression: 39 -> 78 -> 156 -> 312
            stride = 2 ** (num_latent_scales - 1 - s)
            
            # Principled Downsampling: FactorizedReduce for stride=2, else Conv1D
            if stride == 2:
                self.bu_downsamplers.append(FactorizedReduce(fm_embed_dim, channels))
            elif stride > 2:
                # For large strides, chain FactorizedReduce if possible, or use strided Conv1D
                self.bu_downsamplers.append(nn.Sequential(
                    Conv1D(fm_embed_dim, channels, kernel_size=stride*2+1, 
                           stride=stride, padding=stride),
                    nn.ELU()
                ))
            else:
                self.bu_downsamplers.append(Conv1D(fm_embed_dim, channels, kernel_size=1))

        # 2. Latent Variable primitives per scale
        self.prior_nets = nn.ModuleList()
        self.post_nets = nn.ModuleList()
        self.dec_combiners = nn.ModuleList()
        
        # Scale 0 uses a fixed standard Normal prior; PriorNets only for scale 1+
        for s in range(num_latent_scales):
            if s > 0:
                self.prior_nets.append(CNVAE_PriorNet(channels, 2 * latent_dim))
            self.post_nets.append(CNVAE_PosteriorNet(channels, channels, 2 * latent_dim))
            self.dec_combiners.append(CNVAE_DecCombinerCell(channels, latent_dim, channels))

        # R3: Squeeze-and-Excitation blocks per scale for lead-wise weighting
        self.se_blocks = nn.ModuleList([
            SqueezeExcitation1D(channels) for _ in range(num_latent_scales)
        ])

        # Initial seed project
        self.seed_proj = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.ELU(),
        )

        # Pre-compute and register KL balancer coefficients as a buffer
        # This avoids frequent cat/to calls in forward which can trigger CUDA instabilities.
        self.register_buffer("kl_coeffs_square", self._get_kl_balancer_coeff(num_latent_scales))

        # A2: Temporal self-attention at finest scale
        self.temporal_attention = TemporalSelfAttention(channels)

        # 3. Top-Down (TD) Pathway: Upsampling and ResNet (Mason) processing blocks.
        self.upsamplers = nn.ModuleList()
        self.processors = nn.ModuleList()
        for s in range(num_latent_scales):
            # All scales use learned ConvTranspose1d 2× upsampling for consistency.
            self.upsamplers.append(nn.Sequential(
                nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1),
                nn.ELU(),
            ))

            # Reference Parity: Replace Mason processors with NVAE Pre-activation cells
            self.processors.append(nn.Sequential(
                NVAECell(channels, channels, stride=1, kernel_size=kernel_size),
                NVAECell(channels, channels, stride=1, kernel_size=kernel_size)
            ))

        # 4. Final Output: Independent lead networks
        # Each lead head is now a sequence of NVAE cells for consistency.
        self.output_nets = nn.ModuleList([
            nn.Sequential(
                NVAECell(channels, channels, stride=1, kernel_size=kernel_size),
                nn.Conv1d(channels, 1, kernel_size=1)
            )
            for _ in range(n_output_leads)
        ])

        # A1: Cross-Lead Attention applied before final output
        self.cross_lead_attention = CrossLeadAttention(channels, n_output_leads)

        # C2 & N2 & P4 & Q1: Post-Upsampling Tower (Learned refinement to 5000Hz)
        # 624 -> 1248 -> 2496 -> 4992
        # N2: Added Squeeze-Excitation refinement at each resolution stage.
        # P4 & Q1: Added Depthwise-Separable ECGConv blocks for efficient temporal processing.
        self.post_upsamplers = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose1d(channels, channels, 4, 2, 1), 
                nn.ELU(),
                SqueezeExcitation1D(channels),
                DepthwiseECGConvNetwork1D(channels, channels, block_num=1, kernel_size=kernel_size)
            ), 
            nn.Sequential(
                nn.ConvTranspose1d(channels, channels, 4, 2, 1), 
                nn.ELU(),
                SqueezeExcitation1D(channels),
                DepthwiseECGConvNetwork1D(channels, channels, block_num=1, kernel_size=kernel_size)
            ), 
            nn.Sequential(
                nn.ConvTranspose1d(channels, channels, 4, 2, 1), 
                nn.ELU(),
                SqueezeExcitation1D(channels),
                DepthwiseECGConvNetwork1D(channels, channels, block_num=1, kernel_size=kernel_size)
            ), 
        ])
        
        # N6: Auxiliary Multiscale Heads
        # Provides gradient signal to coarse scales via intermediate reconstructions.
        self.aux_heads = nn.ModuleList([
            nn.Conv1d(channels, n_output_leads, kernel_size=1)
            for _ in range(num_latent_scales)
        ])

        # S1: Spectral Norm Tracking (C11)
        self.sr_u = {}
        self.sr_v = {}
        self.num_power_iter = 4
        
        # O2: Layer Caching for performance
        self.cached_convs = [
            (n, l) for n, l in self.named_modules() if isinstance(l, nn.Conv1d)
        ]
        self.cached_norms = [
            l for _, l in self.named_modules() 
            if isinstance(l, (nn.BatchNorm1d, nn.LayerNorm))
        ]
        
        # C9 & N1 & P3: Per-Lead Consistency Layer
        # Initialize as exactly zero to ensure true identity residual at start.
        self.final_refinement = nn.Conv1d(n_output_leads, n_output_leads, kernel_size=1)
        with torch.no_grad():
            self.final_refinement.weight.zero_()
            self.final_refinement.bias.zero_()
        self.refine_weight = nn.Parameter(torch.zeros(1))

    def _get_bu_features(self, fm_tokens):
        """
        Downsample FM token sequence to each hierarchical scale using learned strided convolutions.
        Strided Conv1d preserves more discriminative features than average pooling.
        """
        fm_tokens = fm_tokens.transpose(1, 2) # (B, 768, 312)
        
        # Absolute Parity: Normalize model stem input
        # Reference: x = 2.0 * x - 1.0
        fm_tokens = 2.0 * fm_tokens - 1.0
        bu_features = []
        for down in self.bu_downsamplers:
            bu_features.append(down(fm_tokens))
        return bu_features # Coarse-to-fine sequence

    def _complete_limb_leads(self, x_12: torch.Tensor, mask: torch.Tensor):
        """
        Complete the limb-lead system from any valid pair among I, II, III.

        Theory:
        - Any two independent limb leads in {I, II, III} determine the third.
        - Once I and II are available, the augmented leads are fixed by
          Goldberger relations.

        Inputs:
        - x_12:  sparse 12-lead canvas, shape (B, 12, T)
        - mask:  observed/available lead mask, shape (B, 12, 1)

        Returns:
        - completed signal canvas
        - completed availability mask
        """
        x_full = x_12.clone()
        full_mask = mask.clone()

        has_I = full_mask[:, 0, 0] > 0
        has_II = full_mask[:, 1, 0] > 0
        has_III = full_mask[:, 2, 0] > 0

        # Recover the missing independent limb lead when any valid pair exists.
        need_I = (~has_I) & has_II & has_III
        if need_I.any():
            x_full[need_I, 0, :] = x_full[need_I, 1, :] - x_full[need_I, 2, :]
            full_mask[need_I, 0, :] = 1.0

        need_II = has_I & (~has_II) & has_III
        if need_II.any():
            x_full[need_II, 1, :] = x_full[need_II, 0, :] + x_full[need_II, 2, :]
            full_mask[need_II, 1, :] = 1.0

        need_III = has_I & has_II & (~has_III)
        if need_III.any():
            x_full[need_III, 2, :] = x_full[need_III, 1, :] - x_full[need_III, 0, :]
            full_mask[need_III, 2, :] = 1.0

        # After completing the independent limb system, derive the augmented leads.
        has_I = full_mask[:, 0, 0] > 0
        has_II = full_mask[:, 1, 0] > 0
        can_derive_aug = has_I & has_II
        if can_derive_aug.any():
            lead_I = x_full[:, 0, :]
            lead_II = x_full[:, 1, :]
            x_full[can_derive_aug, 3, :] = -(lead_I[can_derive_aug] + lead_II[can_derive_aug]) / 2.0
            x_full[can_derive_aug, 4, :] = lead_I[can_derive_aug] - lead_II[can_derive_aug] / 2.0
            x_full[can_derive_aug, 5, :] = lead_II[can_derive_aug] - lead_I[can_derive_aug] / 2.0
            full_mask[can_derive_aug, 3:6, :] = 1.0

        return x_full, full_mask

    def _get_kl_balancer_coeff(self, num_scales):
        """Pre-calculates 'square' weighting coefficients."""
        coeffs = []
        for i in range(num_scales):
            # (2^i)^i / groups=1 formula
            c = np.square(2 ** i) / 1.0
            coeffs.append(torch.ones(1) * c)
        coeff = torch.cat(coeffs, dim=0)
        coeff /= torch.min(coeff)
        return coeff

    def kl_balancer(self, kl_all, kl_coeff=1.0, kl_balance=False, alpha_i=None):
        """
        Balances KL divergence across hierarchical scales.
        """
        if kl_balance and alpha_i is not None:
            # kl_all: (B, S)
            kl_vals = torch.mean(kl_all, dim=0) # (S,)
            kl_coeff_i = torch.abs(kl_all).mean(dim=0, keepdim=True) + 0.01 # (1, S)
            total_kl = torch.sum(kl_coeff_i)

            alpha_i = alpha_i.unsqueeze(0) # (1, S)
            kl_coeff_i = (kl_coeff_i / alpha_i) * total_kl
            kl_coeff_i = kl_coeff_i / torch.mean(kl_coeff_i, dim=1, keepdim=True)
            
            # Weighted sum over scales
            kl = torch.sum(kl_all * kl_coeff_i.detach(), dim=1) # (B,)
            kl_coeffs = kl_coeff_i.squeeze(0)
        else:
            kl = torch.sum(kl_all, dim=1) # (B,)
            kl_coeffs = torch.ones(kl_all.shape[1])
            kl_vals = torch.mean(kl_all, dim=0)

        return kl_coeff * kl, kl_coeffs, kl_vals

    def spectral_norm_parallel(self):
        """
        Computes the sum of largest singular values for all tracked Conv1d layers.
        Used for spectral regularization (Pass 12).
        O4: Uses cached layers for performance.
        """
        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for name, l in self.cached_convs:
            
            weight = l.weight
            weight_mat = weight.view(weight.size(0), -1)
            
            if name not in self.sr_u:
                row, col = weight_mat.shape
                # Initialize u, v with correct precision
                self.sr_u[name] = F.normalize(torch.randn(row, device=weight.device, dtype=weight.dtype), dim=0, eps=1e-3).detach()
                self.sr_v[name] = F.normalize(torch.randn(col, device=weight.device, dtype=weight.dtype), dim=0, eps=1e-3).detach()

            u = self.sr_u[name].to(weight.dtype)
            v = self.sr_v[name].to(weight.dtype)

            # Power Iteration: Update u, v without tracking gradients through them
            with torch.no_grad():
                for _ in range(self.num_power_iter):
                    v = F.normalize(torch.matmul(u.unsqueeze(0), weight_mat).squeeze(0), dim=0, eps=1e-3)
                    u = F.normalize(torch.matmul(weight_mat, v.unsqueeze(1)).squeeze(1), dim=0, eps=1e-3)

            # Store detached versions for the next batch
            self.sr_u[name] = u.detach()
            self.sr_v[name] = v.detach()

            # Compute largest singular value (sigma) for the current batch's spectral loss
            # sigma = u^T * W * v
            # Detach vectors to only provide gradients to the weights (standard spectral reg)
            sigma = torch.matmul(u.detach().unsqueeze(0), torch.matmul(weight_mat, v.detach().unsqueeze(1)))
            loss += torch.sum(sigma)
            
        return loss

    def batchnorm_loss(self):
        """
        Regularizes BatchNorm/LayerNorm weights (L1 sum of absolute values).
        O4: Uses cached layers for performance.
        """
        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for l in self.cached_norms:
            if l.elementwise_affine if hasattr(l, 'elementwise_affine') else l.affine:
                # R1: L1 Regularization (sum of absolute values) for gamma stability.
                loss += torch.sum(torch.abs(l.weight))
        return loss

    def forward(self, fm_tokens, use_posterior=None, kl_balance=False, return_internal=False):
        if use_posterior is None:
            use_posterior = self.training

        # 1. Bottom-up extraction from frozen FM tokens.
        #    Each scale sees a progressively coarser view of the token sequence.
        bu_feats = self._get_bu_features(fm_tokens)
        
        # 2. Seed the top-down state from the coarsest FM-derived feature map.
        s = self.seed_proj(bu_feats[0])
        
        all_q, all_p, all_z = [], [], []
        B = fm_tokens.shape[0]

        # 3. Hierarchical top-down generation.
        #    At each scale:
        #    - build a prior p(z_s | z_<s)
        #    - build a posterior q(z_s | z_<s, x)
        #    - sample z_s
        #    - fuse z_s back into the decoder state
        aux_signals = []
        posterior_scale_summaries = []
        posterior_scale_means = []
        decoder_scale_states = []
        for s_idx in range(self.num_latent_scales):
            if s_idx == 0:
                p_mu = torch.zeros(B, self.latent_dim, s.shape[-1], device=s.device)
                p_log_sigma = torch.zeros_like(p_mu)
            else:
                p_params = self.prior_nets[s_idx - 1](s)
                p_mu, p_log_sigma = p_params.chunk(2, dim=1)
            
            p_dist = Normal(p_mu, p_log_sigma)
            all_p.append(p_dist)

            q_params = self.post_nets[s_idx](s, bu_feats[s_idx])
            q_mu_delta, q_log_sigma_delta = q_params.chunk(2, dim=1)
            
            q_dist = Normal(p_mu + q_mu_delta, p_log_sigma + q_log_sigma_delta)
            all_q.append(q_dist)
            posterior_scale_means.append(q_dist.mu)
            posterior_scale_summaries.append(q_dist.mu.mean(dim=2))

            z = q_dist.sample() if use_posterior else p_dist.sample()
            all_z.append(z)

            s = self.dec_combiners[s_idx](s, z)
            s = self.processors[s_idx](s)
            s = self.se_blocks[s_idx](s)
            decoder_scale_states.append(s)
            aux_signals.append(self.aux_heads[s_idx](s))
            
            if s_idx == self.num_latent_scales - 1:
                s = self.temporal_attention(s)
                
            s = self.upsamplers[s_idx](s)

        # 4. Final temporal refinement up to 5000 samples.
        for up in self.post_upsamplers:
            s = up(s)

        if s.shape[-1] != self.target_len:
            s = F.interpolate(s, size=self.target_len, mode='linear', align_corners=False)
            
        # 5. Cross-lead coupling before the final six independent lead heads.
        s = self.cross_lead_attention(s)
        
        outputs = [net(s) for net in self.output_nets]
        independent = torch.cat(outputs, dim=1)
        
        independent = independent + self.refine_weight * self.final_refinement(independent)
        
        # 6. KL bookkeeping for each latent scale.
        kl_scales = []
        for q, p in zip(all_q, all_p):
            kl_val = Normal.kl(q, p)
            kl_scales.append(kl_val.sum(dim=(1, 2)))
        kl_all = torch.stack(kl_scales, dim=1)

        balanced_kl, kl_coeffs, kl_vals = self.kl_balancer(kl_all, kl_balance=kl_balance, alpha_i=self.kl_coeffs_square)

        output = {
            "independent": independent,
            "kl": kl_all,
            "kl_balanced": balanced_kl,
            "kl_coeffs": kl_coeffs,
            "kl_vals": kl_vals
        }
        if self.training:
            output["aux"] = aux_signals
        if return_internal:
            output["sparse_bu_pyramid"] = bu_feats
            output["posterior_scale_means"] = posterior_scale_means
            output["posterior_scale_summaries"] = posterior_scale_summaries
            output["decoder_scale_states"] = decoder_scale_states
            
        return output

    def _masked_zscore_from_mask(self, x_12: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Per-lead Z-score normalization using an explicit availability mask."""
        B, C, T = x_12.shape
        # Compute statistics only over available channels; masked channels remain zero.
        mean = (x_12 * mask).sum(dim=2, keepdim=True) / T
        diff = (x_12 - mean) * mask
        var = (diff * diff).sum(dim=2, keepdim=True) / T
        std = var.sqrt().clamp(min=1e-6)
        
        x_norm = (x_12 - mean) / std
        
        return x_norm * mask

    def summarize_latents_for_distill(
        self,
        posterior_scale_summaries: List[torch.Tensor],
        teacher_pyramid: List[torch.Tensor],
        teacher_projectors: nn.ModuleList,
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Summarize sparse posterior states and align them with projected teacher
        pyramid summaries. This is the default latent-level distillation path;
        the raw posterior means per scale are also exposed through
        `return_internal=True` for downstream analysis.
        """
        teacher_summaries = []
        for proj, teacher_feat in zip(teacher_projectors, teacher_pyramid):
            pooled = teacher_feat.detach().mean(dim=2)
            teacher_summaries.append(proj(pooled))
        return {
            "posterior": posterior_scale_summaries,
            "teacher": teacher_summaries,
        }

    def summarize_decoder_states_for_distill(
        self,
        decoder_scale_states: List[torch.Tensor],
        teacher_pyramid: List[torch.Tensor],
        projector_heads: nn.ModuleList,
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Optional hook for future decoder-state distillation.
        It is intentionally unused by default in the Phase 3 trainer.
        """
        decoder_summaries = [state.mean(dim=2) for state in decoder_scale_states]
        teacher_summaries = []
        for proj, teacher_feat in zip(projector_heads, teacher_pyramid):
            teacher_summaries.append(proj(teacher_feat.detach().mean(dim=2)))
        return {
            "decoder": decoder_summaries,
            "teacher": teacher_summaries,
        }

    def _build_12lead(self, x_input: torch.Tensor, independent_pred: torch.Tensor,
                      lead_indices: torch.Tensor, learned_indices) -> torch.Tensor:
        """Assemble full 12-lead ECG from sparse input + decoded independent leads.

        Einthoven derivation is applied per-sample.
        The decoder predicts the canonical independent set:
          I, II, V1, V2, V3, V4, V5, V6
        Observed leads are copied through exactly; only missing independent leads
        are filled from the decoder outputs. Dependent limb leads are then
        completed analytically.
        """
        B, _, T = independent_pred.shape
        device = independent_pred.device
        li_2d = lead_indices if lead_indices.dim() > 1 else lead_indices.unsqueeze(0).expand(B, -1)
        sparse_canvas = torch.zeros(B, 12, T, device=device, dtype=independent_pred.dtype)
        sparse_mask = torch.zeros(B, 12, 1, device=device, dtype=independent_pred.dtype)
        learned_set = {int(idx) for idx in learned_indices}

        for b in range(B):
            sparse_canvas[b, li_2d[b], :] = x_input[b]
            sparse_mask[b, li_2d[b], :] = 1.0

        # Fill only regime-learned independent leads from decoder outputs.
        out = sparse_canvas.clone()
        full_mask = sparse_mask.clone()
        for b in range(B):
            for head_idx, lead_idx in enumerate(CANONICAL_INDEPENDENT_INDICES):
                if lead_idx not in learned_set:
                    continue
                out[b, lead_idx, :] = independent_pred[b, head_idx, :]
                full_mask[b, lead_idx, :] = 1.0

        # Complete the limb system analytically from the observed basis plus
        # any regime-learned independent degrees of freedom.
        out, _ = self._complete_limb_leads(out, full_mask)

        # Re-copy the observed leads exactly after analytical completion so the
        # final assembled ECG never alters provided evidence.
        for b in range(B):
            out[b, li_2d[b], :] = x_input[b]
        return out

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract FM features with an explicit expected backbone output contract."""
        with torch.no_grad():
            res = self.backbone.extract_features(x, None)
            if isinstance(res, dict):
                if "x" not in res:
                    raise RuntimeError(
                        "ECG-FM backbone output contract changed: expected key 'x' "
                        f"but got keys {sorted(res.keys())}"
                    )
                feat = res["x"]
            else:
                feat = res
        return feat.to(x.dtype)

# =============================================================================
# TOP-LEVEL BRIDGE (Interface for training script)
# =============================================================================

class ECGConvECGFMBridge(nn.Module):
    """
    Principled ECG-FM + cNVAE Bridge.
    Wraps the hierarchical decoder and handles FM backbone lifecycle.

    Model logic:
    - build a sparse 12-lead evidence canvas from the observed basis
    - form an evidence-pure FM input state
    - run the frozen FM as an encoder at inference and as a frozen teacher at training
    - infer a hierarchical latent cardiac state with cNVAE
    - decode the universal 8-head bank with regime-masked supervision
    - analytically rebuild the remaining limb leads from the observed basis
    """
    def __init__(self, checkpoint_path, embed_dim=768, freeze_encoder=True, 
                 target_len=5000, num_latent_scales=4, channels=96, latent_dim=32,
                 n_output_leads=len(CANONICAL_INDEPENDENT_INDICES)):
        super().__init__()
        self.target_len = target_len
        self._allowed_backbone_unexpected_keys = {
            "quantizer.vars",
            "quantizer.weight_proj.weight",
            "quantizer.weight_proj.bias",
            "project_q.weight",
            "project_q.bias",
            "final_proj.weight",
            "final_proj.bias",
        }
        
        # Load backbone
        from fairseq_signals.models.ecg_transformer import ECGTransformerModel
        from fairseq_signals.utils.checkpoint_utils import load_checkpoint_to_cpu

        state = load_checkpoint_to_cpu(checkpoint_path)
        cfg = state['cfg']['model']
        from omegaconf import OmegaConf
        OmegaConf.set_struct(cfg, False)
        if getattr(cfg, 'saliency', None) is None:
            cfg.saliency = False
        self.backbone = ECGTransformerModel.build_model(cfg)
        incompatible = self.backbone.load_state_dict(state['model'], strict=False)
        if incompatible.missing_keys:
            raise RuntimeError(
                "FM backbone checkpoint is missing required keys: "
                f"{incompatible.missing_keys}"
            )
        unexpected = set(incompatible.unexpected_keys)
        disallowed_unexpected = sorted(unexpected - self._allowed_backbone_unexpected_keys)
        if disallowed_unexpected:
            raise RuntimeError(
                "FM backbone checkpoint contains unsupported unexpected keys: "
                f"{disallowed_unexpected}"
            )
        
        if freeze_encoder:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
            print(f"ECG-FM Backbone frozen: {checkpoint_path}")

        # LayerNorm on FM tokens before decoding provides a learnable normalization.
        self.feature_norm = nn.LayerNorm(embed_dim)
        
        # A3: Lightweight Self-Attention on FM Tokens
        self.fm_token_attention = FMTokenAttention(embed_dim)
        self.token_projector = nn.Linear(embed_dim, embed_dim)

        self.decoder = CNVAE_CardiacStateDecoder(
            fm_embed_dim=embed_dim,
            target_len=target_len,
            n_output_leads=n_output_leads,
            num_latent_scales=num_latent_scales,
            channels=channels,
            latent_dim=latent_dim
        )
        self.teacher_summary_projectors = nn.ModuleList(
            [nn.Linear(channels, latent_dim) for _ in range(num_latent_scales)]
        )
        # Shared backbone access
        self.decoder.backbone = self.backbone

    def _validate_lead_indices(self, lead_indices):
        li = lead_indices if lead_indices.dim() > 1 else lead_indices.unsqueeze(0)
        for row in li:
            validate_obs_indices(row.tolist())

    def _build_sparse_canvas(self, x_input: torch.Tensor, lead_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, _, T = x_input.shape
        device = x_input.device
        li = lead_indices if lead_indices.dim() > 1 else lead_indices.unsqueeze(0).expand(B, -1)
        x_12 = torch.zeros(B, 12, T, device=device, dtype=x_input.dtype)
        obs_mask = torch.zeros(B, 12, 1, device=device, dtype=x_input.dtype)
        for b in range(B):
            x_12[b, li[b], :] = x_input[b]
            obs_mask[b, li[b], :] = 1.0
        return x_12, obs_mask

    def _build_fm_input_state(
        self,
        x_12: torch.Tensor,
        obs_mask: torch.Tensor,
        regime_name: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if regime_name == "multi_limb":
            return self.decoder._complete_limb_leads(x_12, obs_mask)
        if regime_name == "one_limb_I":
            return x_12.clone(), obs_mask.clone()
        raise ValueError(f"Unsupported regime_name for FM input state: {regime_name}")

    def _prepare_fm_tokens(self, x_norm: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        fm_tokens = self.decoder.extract_features(x_norm.to(torch.float32))
        assert fm_tokens.shape[1] == 312, f"Unexpected FM token length: {fm_tokens.shape[1]}"
        fm_tokens = fm_tokens.to(dtype)
        fm_tokens = self.feature_norm(fm_tokens)
        return self.fm_token_attention(fm_tokens)

    def extract_teacher_features(self, full_x12: torch.Tensor) -> torch.Tensor:
        full_mask = torch.ones(full_x12.size(0), full_x12.size(1), 1, device=full_x12.device, dtype=full_x12.dtype)
        full_norm = self.decoder._masked_zscore_from_mask(full_x12, full_mask)
        with torch.no_grad():
            return self._prepare_fm_tokens(full_norm, full_x12.dtype).detach()

    def build_teacher_pyramid(self, teacher_tokens: torch.Tensor) -> List[torch.Tensor]:
        with torch.no_grad():
            return [feat.detach() for feat in self.decoder._get_bu_features(teacher_tokens)]

    def forward(
        self,
        x,
        lead_indices,
        learned_indices,
        regime_name,
        kl_balance=False,
        use_posterior=None,
        return_internal=False,
    ):
        """Sparse-evidence FM inference with optional internal returns for distillation."""
        if learned_indices is None:
            raise ValueError("Explicit learned_indices are required for regime-consistent assembly.")
        if lead_indices is None:
            raise ValueError("Explicit lead_indices are required for this cNVAE branch.")

        li = lead_indices if lead_indices.dim() > 1 else lead_indices.unsqueeze(0).expand(x.size(0), -1)
        self._validate_lead_indices(li)

        sparse_canvas, obs_mask = self._build_sparse_canvas(x, li)
        fm_x12, fm_mask = self._build_fm_input_state(sparse_canvas, obs_mask, regime_name)
        x_norm = self.decoder._masked_zscore_from_mask(fm_x12, fm_mask)
        sparse_fm_tokens = self._prepare_fm_tokens(x_norm, x.dtype)

        dec_out = self.decoder(
            sparse_fm_tokens,
            use_posterior=use_posterior,
            kl_balance=kl_balance,
            return_internal=return_internal,
        )
        independent = dec_out["independent"]
        recon_12 = self.decoder._build_12lead(x, independent, li, learned_indices)

        output = {
            "recon": recon_12,
            "kl": dec_out["kl"],
            "kl_balanced": dec_out["kl_balanced"],
            "kl_coeffs": dec_out["kl_coeffs"],
            "kl_vals": dec_out["kl_vals"],
        }
        if self.training:
            output["aux"] = dec_out["aux"]
        if return_internal:
            output["sparse_fm_tokens"] = sparse_fm_tokens
            output["sparse_bu_pyramid"] = dec_out["sparse_bu_pyramid"]
            output["posterior_scale_means"] = dec_out["posterior_scale_means"]
            output["posterior_scale_summaries"] = dec_out["posterior_scale_summaries"]
            output["decoder_scale_states"] = dec_out["decoder_scale_states"]
            output["fm_input_mask"] = fm_mask
        return output
