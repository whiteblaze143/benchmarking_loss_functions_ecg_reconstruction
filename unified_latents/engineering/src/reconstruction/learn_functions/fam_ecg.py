#!/usr/bin/env python3
"""
Strict Mason VAE Components (Phase 44 Standardization).

Implements strictly Mason-compliant VAE components:
  - Parallel pathways for each lead (no shared lead embeddings).
  - Standard Mason convolutional blocks and residuals.
  - Per-lead Z-score normalization with robust vectorized masking.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
from pathlib import Path

def _setup_fairseq_path():
    """Robustly add fairseq-signals to path using project root relative to this file."""
    PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
    fs_path = str(PROJECT_ROOT / 'ecg_fm_integration' / 'fairseq-signals')
    if fs_path not in sys.path:
        sys.path.insert(0, fs_path)

_setup_fairseq_path()

# =============================================================================
# MASON ET AL. (2024) CORE ARCHITECTURAL BLOCKS
# =============================================================================

class MasonConvolutionalBlock(nn.Module):
    """Standard PyTorch nn.Module port of Mason's ConvolutionalBlock."""
    def __init__(self, input_channel, output_channel, kernel_size=17, stride_size=1,
                 inner_activation='relu', output_activation='relu', use_residual_block=True, average_pool=1):
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.use_residual_block = use_residual_block

        self.inner_conv = nn.Conv1d(input_channel, output_channel, kernel_size, stride=1, padding='same')
        self.inner_activation_function = torch.relu
        
        padding = int(kernel_size / 2) if stride_size > 1 and kernel_size % 2 == 1 else 'same'
        self.out_conv = nn.Conv1d(output_channel, output_channel, kernel_size, stride=stride_size, padding=padding)
        
        self.res_conv = nn.Conv1d(input_channel, output_channel, 1, stride=stride_size, padding=0) if use_residual_block else None
        
        self.output_activation_function = None if output_activation is None else torch.relu
        self.average_pool = nn.AvgPool1d(average_pool) if average_pool > 1 else None
        
        self._mason_reset(inner_activation, output_activation)

    def _mason_reset(self, inner_activation, output_activation):
        """Kaiming initialization with Mason-specific gains."""
        nn.init.kaiming_normal_(self.inner_conv.weight, mode='fan_in', 
                               nonlinearity='leaky_relu' if inner_activation == 'elu' else inner_activation)
        nn.init.kaiming_normal_(self.out_conv.weight, mode='fan_in', 
                               nonlinearity='linear' if output_activation is None else output_activation)
        if self.res_conv:
            nn.init.kaiming_normal_(self.res_conv.weight, mode='fan_in', 
                                   nonlinearity='linear' if output_activation is None else output_activation)
        nn.init.zeros_(self.inner_conv.bias)
        nn.init.zeros_(self.out_conv.bias)
        if self.res_conv: nn.init.zeros_(self.res_conv.bias)

    def forward(self, x):
        residual = x
        x = self.inner_conv(x)
        x = self.inner_activation_function(x)
        x = self.out_conv(x)
        if self.use_residual_block:
            x += self.res_conv(residual)
        if self.output_activation_function is not None:
            x = self.output_activation_function(x)
        if self.average_pool is not None:
            x = self.average_pool(x)
        return x

class MasonConvolutionalNetwork(nn.Module):
    """Sequence of Mason blocks with channel interpolation."""
    def __init__(self, input_channel, output_channel, block_num, kernel_size=17, stride_size=1, average_pool=1,
                 inner_activation='relu', output_activation='relu', use_residual_block=True):
        super().__init__()
        self.blocks = nn.ModuleList()
        channels = np.linspace(input_channel, output_channel, block_num + 1).astype(int)
        for i in range(block_num):
            inp, out = channels[i], channels[i+1]
            s = stride_size if i == block_num - 1 else 1
            ap = average_pool if i == block_num - 1 else 1
            act_o = output_activation if i == block_num - 1 else inner_activation
            self.blocks.append(MasonConvolutionalBlock(inp, out, kernel_size, s, inner_activation, act_o, use_residual_block, ap))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

# =========================================================================
# PHASE 44: GENERATIVE WORLD MODELS (VAE) COMPONENTS
# =========================================================================

class MasonStyleDecoder(nn.Module):
    """
    Standard adaptation: Deterministic 12-lead reconstruction.
    Uses interpolation + MasonConvolutionalNetwork for direct feature mapping.
    """
    def __init__(self, embed_dim=768, output_len=5000, n_leads=6, 
                 channels=96, kernel_size=17, depth=3):
        super().__init__()
        self.output_len = output_len
        self.channels = channels
        
        # 1. Projection from Transformer space
        self.proj = nn.Linear(embed_dim, channels)
        
        # 2. Mason Processing Blocks (ResNet-style)
        self.middle = MasonConvolutionalNetwork(
            channels, 128, depth, kernel_size, 1, 1, 'relu', 'relu', True
        )
        
        # 3. Predict leads (Strict Mason Parallelism: no shared lead embeddings)
        # output_activation=None -> unbounded output (raw mV scale).
        self.output_nets = nn.ModuleList([
            MasonConvolutionalNetwork(128, 1, depth, kernel_size, 1, 1, 'relu', None, True)
            for _ in range(n_leads)
        ])

    def forward(self, x, skip_input=None, target_len=None):
        T = target_len or self.output_len
        B = x.shape[0]
        
        # S1: Project and Interpolate
        h = self.proj(x).transpose(1, 2)
        h = F.interpolate(h, size=T, mode='linear', align_corners=False)
        
        # S2: Process
        h = self.middle(h)
        
        # S3: Output (Parallel Stage)
        outputs = [net(h) for net in self.output_nets]
        return torch.cat(outputs, dim=1)

class VAEMasonPosterior(nn.Module):
    """Encodes clinical state from FM feature tokens into latent space."""
    def __init__(self, embed_dim=768, latent_dim=256, channels=128):
        super().__init__()
        # 1. Input reduction (temporal average pooling on tokens)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(channels, latent_dim)
        self.logvar_head = nn.Linear(channels, latent_dim)

    def forward(self, x):
        # x: (B, T_enc, embed_dim)
        x = x.mean(dim=1) # (B, embed_dim)
        x = self.net(x)
        mu = self.mu_head(x)
        logvar = self.logvar_head(x)
        
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + std * eps
        else:
            z = mu
        return mu, logvar, z

class StrictMasonDecoder(nn.Module):
    """
    Strict Mason adaptation: Generates precordial leads using 
    strictly Mason blocks and standard linear interpolation.
    """
    def __init__(self, global_latent_dim=256, local_latent_dim=64, target_len=5000, n_output_leads=6,
                 channels=96, kernel_size=17, middle_depth=2, output_depth=3):
        super().__init__()
        self.target_len = target_len
        self.channels = channels
        
        self.global_proj = nn.Linear(global_latent_dim, channels)
        self.local_proj = nn.Conv1d(local_latent_dim, channels, kernel_size=1)
        
        self.middle = MasonConvolutionalNetwork(
            channels, 128, middle_depth, kernel_size, 1, 1, 'relu', 'relu', True
        )
        
        # output_activation=None -> unbounded output (raw mV scale).
        self.output_nets = nn.ModuleList([
            MasonConvolutionalNetwork(128, 1, output_depth, kernel_size, 1, 1, 'relu', None, True)
            for _ in range(n_output_leads)
        ])

    def forward(self, z_global, z_local, target_len=None):
        T = target_len or self.target_len
        B = z_global.shape[0]
        
        # S1: Project global state via broadcast
        h_g = self.global_proj(z_global).view(B, self.channels, 1).expand(-1, -1, T)
        
        # S2: Project local state and interpolate
        z_l_up = F.interpolate(z_local, size=T, mode='linear', align_corners=False)
        h_l = self.local_proj(z_l_up)
        
        h = h_g + h_l
        h = self.middle(h)
        
        # S3: Output (Parallel Stage)
        outputs = [net(h) for net in self.output_nets]
        return torch.cat(outputs, dim=1)

# =========================================================================
# MASON ET AL. (2024) 12-LEAD PROTOCOL CONSTANTS
# =========================================================================
MASON_INPUT_LIMB_V3 = [0, 1, 8]
PRECORDIAL_INDICES = list(range(6, 12))
MASON_MIN = -2.5
MASON_AMP = 5.0

def denormalize_mason(lead, min_value=MASON_MIN, amplitude=MASON_AMP):
    """Maps Mason-normalized [0,1] output back to mV scale."""
    return lead * amplitude + min_value

def batch_r2_function(model_output, model_target, lead_num: int, batch_size: int, compute_loss_per_element: bool):
    """Vectorized Negative R2 Loss for Mason compliance."""
    out = torch.stack(list(model_output), dim=1)
    tgt = torch.stack(list(model_target), dim=1)
    ssr = ((out - tgt) ** 2).sum(dim=2)
    sst = ((tgt - tgt.mean(dim=2, keepdim=True)) ** 2).sum(dim=2)
    r2  = 1.0 - ssr / sst.clamp(min=1e-7)
    batch_r2 = r2.mean()
    r2_per_element = -r2.mean(dim=1).detach().cpu().numpy() if compute_loss_per_element else np.zeros(batch_size)
    return -batch_r2, r2_per_element

# =============================================================================
# BASE BRIDGE (Common Logic)
# =============================================================================

class ECGFMBridgeBase(nn.Module):
    def __init__(self, checkpoint_path, embed_dim=768, target_len=5000):
        super().__init__()
        self.target_len = target_len
        self.embed_dim = embed_dim
        
        from fairseq_signals.models.ecg_transformer import ECGTransformerModel
        from fairseq_signals.utils.checkpoint_utils import load_checkpoint_to_cpu

        state = load_checkpoint_to_cpu(checkpoint_path)
        cfg = state['cfg']['model']
        from omegaconf import OmegaConf
        OmegaConf.set_struct(cfg, False)
        if getattr(cfg, 'saliency', None) is None:
            cfg.saliency = False
        self.backbone = ECGTransformerModel.build_model(cfg)
        self.backbone.load_state_dict(state['model'], strict=False)
        
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        print(f"ECG-FM Backbone frozen: {checkpoint_path}")

    def _prepare_input(self, x, lead_indices=None):
        """Standardizes input leads to padded 12-lead tensor for backbone."""
        B, Cin, T = x.shape
        out = torch.zeros(B, 12, T, device=x.device, dtype=x.dtype)
        if lead_indices is not None:
            li = lead_indices if lead_indices.dim() > 1 else lead_indices.unsqueeze(0).expand(B, -1)
            for b in range(B):
                out[b, li[b], :] = x[b]
        else:
            out[:, MASON_INPUT_LIMB_V3, :] = x
        return out

    def extract_features(self, x):
        """Robust extraction supporting different backbone versions (C10)."""
        with torch.no_grad():
            res = self.backbone.extract_features(x, None)
            if isinstance(res, dict):
                feat = res.get('x', res.get('encoder_out'))
            else:
                feat = res
        return feat.to(x.dtype)

    def _masked_zscore(self, x_12: torch.Tensor, lead_indices: torch.Tensor = None) -> torch.Tensor:
        """Per-lead Z-score normalization (McKeen et al. 2025 parity)."""
        B, C, T = x_12.shape
        if lead_indices is not None:
            if torch.is_grad_enabled() and lead_indices.max() >= C:
                raise ValueError(f"lead_indices out of range: max={lead_indices.max().item()} >= C={C}")
            mask = torch.zeros(B, C, 1, device=x_12.device, dtype=x_12.dtype)
            if lead_indices.dim() == 1:
                known = lead_indices.clamp(0, C - 1).tolist()
                if 0 in known and 1 in known: known = list(set(known + [2, 3, 4, 5]))
                mask[:, known, :] = 1.0
            elif (lead_indices == lead_indices[0]).all():
                known = lead_indices[0].clamp(0, C - 1).tolist()
                if 0 in known and 1 in known: known = list(set(known + [2, 3, 4, 5]))
                mask[:, known, :] = 1.0
            else:
                idx = lead_indices.clamp(0, C - 1).unsqueeze(-1)
                mask.scatter_(1, idx, 1.0)
                has_einthoven = ((lead_indices == 0).any(dim=1) & (lead_indices == 1).any(dim=1))
                einth = has_einthoven.float().view(B, 1, 1)
                mask[:, 2:6, :] = torch.max(mask[:, 2:6, :], einth)
            x_masked = x_12 * mask
        else:
            x_masked = x_12

        mean = x_masked.mean(dim=2, keepdim=True)
        std  = x_masked.std(dim=2, keepdim=True).clamp(min=1e-6)
        x_norm = (x_masked - mean) / std
        
        # C5: Einthoven-Normalization Coherence (Synchronized)
        has_einthoven = (mask[:, 0, :] > 0) & (mask[:, 1, :] > 0)
        if has_einthoven.any():
            lI_norm = x_norm[:, 0, :]
            lII_norm = x_norm[:, 1, :]
            mask_e = has_einthoven.float()
            x_norm[:, 2, :] = (lII_norm - lI_norm) * mask_e
            x_norm[:, 3, :] = (-(lI_norm + lII_norm) / 2.0) * mask_e
            x_norm[:, 4, :] = (lI_norm - (lII_norm / 2.0)) * mask_e
            x_norm[:, 5, :] = (lII_norm - (lI_norm / 2.0)) * mask_e

        return x_norm * mask if lead_indices is not None else x_norm

    def _build_12lead(self, x_input: torch.Tensor, precordial: torch.Tensor,
                      lead_indices: torch.Tensor) -> torch.Tensor:
        """Assemble full 12-lead ECG from input limb leads + decoded precordials."""
        B, _, T = precordial.shape
        device = precordial.device
        out = torch.zeros(B, 12, T, device=device, dtype=precordial.dtype)
        out[:, 6:12, :] = precordial
        li_2d = lead_indices if lead_indices.dim() > 1 else lead_indices.unsqueeze(0).expand(B, -1)

        has_I   = (li_2d == 0).any(dim=1)
        has_II  = (li_2d == 1).any(dim=1)
        has_both = has_I & has_II

        if has_both.any():
            batch_idx = torch.arange(B, device=device)
            lI = torch.zeros(B, T, device=device, dtype=precordial.dtype)
            lII = torch.zeros(B, T, device=device, dtype=precordial.dtype)
            valid_I_indices = batch_idx[has_I]
            if valid_I_indices.numel() > 0:
                lI_pos = (li_2d[valid_I_indices] == 0).long().argmax(dim=1)
                lI[valid_I_indices] = x_input[valid_I_indices, lI_pos, :]
            valid_II_indices = batch_idx[has_II]
            if valid_II_indices.numel() > 0:
                lII_pos = (li_2d[valid_II_indices] == 1).long().argmax(dim=1)
                lII[valid_II_indices] = x_input[valid_II_indices, lII_pos, :]

            e = has_both.float().view(B, 1)
            out[:, 0, :] = lI * e
            out[:, 1, :] = lII * e
            out[:, 2, :] = (lII - lI) * e
            out[:, 3, :] = (-(lI + lII) / 2.0) * e
            out[:, 4, :] = (lI - (lII / 2.0)) * e
            out[:, 5, :] = (lII - (lI / 2.0)) * e
        return out

    def get_embedding(self, x, lead_indices=None, pool=True):
        if x.dim() == 2: x = x.unsqueeze(0)
        x_12 = self._prepare_input(x.float(), lead_indices=lead_indices)
        features = self.extract_features(self._masked_zscore(x_12, lead_indices))
        return features.mean(dim=1) if pool else features

# =============================================================================
# CONCRETE IMPLEMENTATIONS
# =============================================================================

class VAEMasonECGFMBridge(ECGFMBridgeBase):
    """Hierarchical VAE Bridge (Phase 43/44 Baseline)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.posterior = VAEMasonPosterior(self.embed_dim)
        self.decoder = StrictMasonDecoder(target_len=self.target_len)

    def forward(self, x, lead_indices=None, **kwargs):
        with torch.amp.autocast('cuda', enabled=False):
            x = x.float()
            x_12 = self._prepare_input(x, lead_indices=lead_indices)
            tokens = self.extract_features(self._masked_zscore(x_12, lead_indices))
            mu, logvar, z_global = self.posterior(tokens)
            
            # Local state for refinement
            z_local = tokens.transpose(1, 2)
            precordial = self.decoder(z_global, z_local)
            recon_12 = self._build_12lead(x, precordial, lead_indices)
            
        if self.training:
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            return recon_12, kl
        return recon_12

class ECGFMBridge(ECGFMBridgeBase):
    """Deterministic Phase 43 Bridge Baseline."""
    def __init__(self, *args, **kwargs):
        n_output_leads = kwargs.pop('n_output_leads', 6)
        super().__init__(*args, **kwargs)
        self.decoder = MasonStyleDecoder(self.embed_dim, self.target_len, n_output_leads)

    def forward(self, x, lead_indices=None, **kwargs):
        with torch.amp.autocast('cuda', enabled=False):
            x = x.float()
            x_12 = self._prepare_input(x, lead_indices=lead_indices)
            tokens = self.extract_features(self._masked_zscore(x_12, lead_indices))
            precordial = self.decoder(tokens)
            recon_12 = self._build_12lead(x, precordial, lead_indices)
        return recon_12
