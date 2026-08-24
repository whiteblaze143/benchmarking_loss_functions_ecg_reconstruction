#!/usr/bin/env python3
"""
WearECG + ECG-FM integration.

Design goal:
- Preserve the baseline WearECG VAE geometry and reconstruction path.
- Use ECG-FM as a frozen semantic teacher first.
- Optionally use pooled ECG-FM context to condition the decoder.
- Keep the external API compatible with the existing evaluator.

Input contract:
- public wrappers expect x shaped [B, 12, T], where x is a sparse 12-lead canvas
- y_full, when provided, is the full 12-lead target shaped [B, 12, T]
- internal sequence conversions stay local to the model

Main classes:
- WearECGVAE: exact-ish baseline wrapper
- WearECGFMVAE: baseline + frozen ECG-FM perceptual supervision
"""

from __future__ import annotations

import math
import sys
sys.path.insert(0, '/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/unified_latents/engineering')
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import kl_divergence
from torch.distributions.normal import Normal

from src.reconstruction.unified_latents.engineering.common import mask_unobserved_leads

LATENT_SCALE = 0.18215


# =============================================================================
# FAIRSEQ / ECG-FM LOADING
# =============================================================================

def _setup_fairseq_path() -> None:
    """Best-effort addition of fairseq-signals to sys.path."""
    try:
        if '__file__' in globals():
            project_root = Path(__file__).resolve().parent
            candidates = [
                project_root / "ecg_fm_integration" / "fairseq-signals",
                project_root.parent / "ecg_fm_integration" / "fairseq-signals",
                project_root.parent.parent / "ecg_fm_integration" / "fairseq-signals",
                project_root.parent.parent.parent / "ecg_fm_integration" / "fairseq-signals",
                project_root.parent.parent.parent.parent / "ecg_fm_integration" / "fairseq-signals",
                project_root.parent.parent.parent.parent.parent / "ecg_fm_integration" / "fairseq-signals",
            ]
        else:
            # Notebook context or fallback
            candidates = [
                Path("/home/mithunmanivannan/ecg_fm_integration/fairseq-signals"),
                Path("/home/mithunmanivannan/third_party/WearECG-reconstruction/ecg_fm_integration/fairseq-signals"),
            ]
            
        for fs_path in candidates:
            if fs_path.exists():
                fs_path_str = str(fs_path)
                if fs_path_str not in sys.path:
                    sys.path.insert(0, fs_path_str)
                break
    except Exception:
        pass


_setup_fairseq_path()


class ECGFMFeatureExtractor(nn.Module):
    """
    Frozen ECG-FM feature extractor.

    Returns token embeddings of shape [B, T_enc, D].
    Also provides simple pooled summaries for perceptual supervision / conditioning.
    """

    def __init__(
        self,
        checkpoint_path: str,
        embed_dim: int = 768,
        allow_unexpected_keys: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.allow_unexpected_keys = set(allow_unexpected_keys or [])

        from fairseq_signals.models.ecg_transformer import ECGTransformerModel
        from fairseq_signals.utils.checkpoint_utils import load_checkpoint_to_cpu
        from omegaconf import OmegaConf

        state = load_checkpoint_to_cpu(checkpoint_path)
        cfg = state["cfg"]["model"]
        OmegaConf.set_struct(cfg, False)
        if getattr(cfg, "saliency", None) is None:
            cfg.saliency = False

        self.backbone = ECGTransformerModel.build_model(cfg)
        incompatible = self.backbone.load_state_dict(state["model"], strict=False)

        if incompatible.missing_keys:
            raise RuntimeError(f"ECG-FM checkpoint missing keys: {incompatible.missing_keys}")

        unexpected = set(incompatible.unexpected_keys)
        disallowed = sorted(unexpected - self.allow_unexpected_keys)
        if disallowed:
            raise RuntimeError(f"ECG-FM checkpoint has unsupported unexpected keys: {disallowed}")

        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        self.token_norm = nn.LayerNorm(self.embed_dim)

    def train(self, mode: bool = True):
        super().train(mode)
        # The FM is a frozen teacher/conditioner and should never leave eval mode.
        self.backbone.eval()
        return self

    def extract_tokens(self, x_12: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_12: [B, 12, T], float tensor in ECG units

        Returns:
            tokens: [B, T_enc, D]
        """
        self.backbone.eval()
        if x_12.shape[1] != 12 and x_12.shape[2] == 12:
            x_12 = x_12.transpose(1, 2)
        elif x_12.shape[1] != 12:
            raise ValueError(f"Expected ECG tensor shaped [B, 12, T] or [B, T, 12], got {tuple(x_12.shape)}")

        x_norm = self._full12_zscore(x_12).to(torch.float32)
        device_type = x_norm.device.type
        with torch.amp.autocast(device_type, enabled=False):
            res = self.backbone.extract_features(x_norm, None)
            tokens = res["x"] if isinstance(res, dict) else res
            tokens = self.token_norm(tokens)
        return tokens.to(x_12.dtype)

    @staticmethod
    def _full12_zscore(x_12: torch.Tensor) -> torch.Tensor:
        """
        Per-lead z-score over time for fully observed 12-lead ECG.
        """
        mean = x_12.mean(dim=2, keepdim=True)
        std = x_12.std(dim=2, keepdim=True).clamp(min=1e-6)
        return (x_12 - mean) / std

    @staticmethod
    def pooled_summary(tokens: torch.Tensor) -> torch.Tensor:
        """
        Global pooled summary: [B, D]
        """
        return tokens.mean(dim=1)


# =============================================================================
# EXACT / BASELINE WEARECG BLOCKS
# =============================================================================

class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias: bool = True, out_proj_bias: bool = True):
        super().__init__()
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, causal_mask: bool = False) -> torch.Tensor:
        input_shape = x.shape
        batch_size, sequence_length, _d_embed = input_shape
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)

        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        weight = weight / math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)

        output = weight @ v
        output = output.transpose(1, 2)
        output = output.reshape(input_shape)
        output = self.out_proj(output)
        return output


class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residue = x
        x = x.transpose(-1, -2)
        x = self.attention(x)
        x = x.transpose(-1, -2)
        return x + residue


class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residue = x

        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)

        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        return x + self.residual_layer(residue)


class VAE_Encoder(nn.Module):
    """
    Near-verbatim WearECG encoder.
    Expects x shaped [B, 12, T] or [B, T, 12] if called through wrapper.
    """

    def __init__(
        self,
        *,
        in_channels: int = 12,
        split_latent: bool = False,
        global_latent_channels: int = 2,
        local_latent_channels: int = 2,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.split_latent = bool(split_latent)
        self.global_latent_channels = int(global_latent_channels)
        self.local_latent_channels = int(local_latent_channels)
        self.total_latent_channels = self.global_latent_channels + self.local_latent_channels
        if self.total_latent_channels != 4:
            raise ValueError("VAE_Encoder expects exactly 4 total latent channels.")
        self.blocks = nn.ModuleList(
            [
                nn.Conv1d(self.in_channels, 128, kernel_size=3, padding=1),
                VAE_ResidualBlock(128, 128),
                VAE_ResidualBlock(128, 128),
                nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=0),
                VAE_ResidualBlock(128, 256),
                VAE_ResidualBlock(256, 256),
                nn.Conv1d(256, 256, kernel_size=3, stride=2, padding=0),
                VAE_ResidualBlock(256, 512),
                VAE_ResidualBlock(512, 512),
                nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=0),
                VAE_ResidualBlock(512, 512),
                VAE_ResidualBlock(512, 512),
                VAE_ResidualBlock(512, 512),
                VAE_AttentionBlock(512),
                VAE_ResidualBlock(512, 512),
                nn.GroupNorm(32, 512),
                nn.SiLU(),
            ]
        )
        if self.split_latent:
            self.local_head = nn.Sequential(
                nn.Conv1d(512, 128, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv1d(128, 2 * self.local_latent_channels, kernel_size=1, padding=0),
            )
            self.global_head = nn.Sequential(
                nn.Conv1d(512, 128, kernel_size=1, padding=0),
                nn.SiLU(),
                nn.Conv1d(128, 2 * self.global_latent_channels, kernel_size=1, padding=0),
            )
        else:
            self.output_head = nn.Sequential(
                nn.Conv1d(512, 8, kernel_size=3, padding=1),
                nn.Conv1d(8, 8, kernel_size=1, padding=0),
            )

    def forward(
        self,
        x: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[int, torch.Tensor]]:
        # If caller passed [B, T, 12], convert to [B, 12, T]
        if x.dim() != 3:
            raise ValueError(f"Expected rank-3 input, got shape {tuple(x.shape)}")
        if x.shape[1] != self.in_channels and x.shape[2] == self.in_channels:
            x = x.transpose(1, 2)
        elif x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got shape {tuple(x.shape)}")

        intermediates: Dict[int, torch.Tensor] = {}
        for block_idx, module in enumerate(self.blocks):
            if getattr(module, "stride", None) == (2,):
                x = F.pad(x, (0, 1))
            x = module(x)
            if return_intermediates and block_idx in {3, 6, 9}:
                intermediates[block_idx] = x

        if self.split_latent:
            local_stats = self.local_head(x)
            local_mean, local_log_variance = torch.chunk(local_stats, 2, dim=1)
            global_context = F.adaptive_avg_pool1d(x, 1)
            global_stats = self.global_head(global_context)
            global_mean, global_log_variance = torch.chunk(global_stats, 2, dim=1)
            global_mean = global_mean.expand(-1, -1, local_mean.shape[-1])
            global_log_variance = global_log_variance.expand(-1, -1, local_log_variance.shape[-1])
            mean = torch.cat([global_mean, local_mean], dim=1)
            log_variance = torch.cat([global_log_variance, local_log_variance], dim=1)
        else:
            x = self.output_head(x)
            mean, log_variance = torch.chunk(x, 2, dim=1)
        log_variance = torch.clamp(log_variance, -30, 20)
        variance = log_variance.exp()
        stdev = variance.sqrt()

        if noise is None:
            noise = torch.randn_like(stdev)
        z = mean + stdev * noise
        z = z * LATENT_SCALE

        if return_intermediates:
            return z, mean, log_variance, intermediates
        return z, mean, log_variance


class ConditionalVAE_Decoder(nn.Module):
    """
    WearECG decoder geometry with optional pooled-FM FiLM conditioning.

    Conditioning is deliberately low-risk:
    - pooled ECG-FM summary only
    - applied after residual blocks
    - zero-initialized modulation so decoder starts as baseline
    """

    def __init__(self, cond_dim: int = 0):
        super().__init__()
        self.cond_dim = int(cond_dim)

        self.blocks = nn.ModuleList(
            [
                nn.Conv1d(4, 4, kernel_size=1, padding=0),
                nn.Conv1d(4, 512, kernel_size=3, padding=1),
                VAE_ResidualBlock(512, 512),   # 0
                VAE_AttentionBlock(512),
                VAE_ResidualBlock(512, 512),   # 1
                VAE_ResidualBlock(512, 512),   # 2
                VAE_ResidualBlock(512, 512),   # 3
                VAE_ResidualBlock(512, 512),   # 4
                nn.Upsample(scale_factor=2),
                nn.Conv1d(512, 512, kernel_size=3, padding=1),
                VAE_ResidualBlock(512, 512),   # 5
                VAE_ResidualBlock(512, 512),   # 6
                VAE_ResidualBlock(512, 512),   # 7
                nn.Upsample(scale_factor=2),
                nn.Conv1d(512, 512, kernel_size=3, padding=1),
                VAE_ResidualBlock(512, 512),   # 8
                VAE_ResidualBlock(512, 512),   # 9
                VAE_ResidualBlock(512, 512),   # 10
                nn.Upsample(scale_factor=2),
                nn.Conv1d(512, 512, kernel_size=3, padding=1),
                VAE_ResidualBlock(512, 256),   # 11
                VAE_ResidualBlock(256, 256),   # 12
                VAE_ResidualBlock(256, 256),   # 13
                nn.GroupNorm(32, 256),
                nn.SiLU(),
                nn.Conv1d(256, 12, kernel_size=3, padding=1),
            ]
        )

        residual_out_channels = [
            512, 512, 512, 512, 512,
            512, 512, 512,
            512, 512, 512,
            256, 256, 256,
        ]

        if self.cond_dim > 0:
            self.cond_to_film = nn.ModuleList(
                [nn.Linear(self.cond_dim, 2 * c) for c in residual_out_channels]
            )
            for layer in self.cond_to_film:
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
        else:
            self.cond_to_film = nn.ModuleList()

    def _apply_film(self, x: torch.Tensor, cond_vec: Optional[torch.Tensor], film_idx: int) -> torch.Tensor:
        if cond_vec is None or self.cond_dim == 0:
            return x
        gamma_beta = self.cond_to_film[film_idx](cond_vec)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)
        return x * (1.0 + gamma) + beta

    def forward(self, z: torch.Tensor, cond_vec: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = z / LATENT_SCALE
        film_idx = 0

        for module in self.blocks:
            x = module(x)
            if isinstance(module, VAE_ResidualBlock):
                x = self._apply_film(x, cond_vec, film_idx)
                film_idx += 1

        x = x.transpose(1, 2)
        return x


# =============================================================================
# LOSSES
# =============================================================================

def baseline_loss_function(
    recons: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    kld_weight: float = 1e-4,
    lead_indices: Optional[torch.Tensor] = None,
    missing_lead_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    recons_loss = weighted_reconstruction_mse(
        recons,
        x,
        lead_indices=lead_indices,
        missing_lead_weight=missing_lead_weight,
    )

    q_z_x = Normal(mu, log_var.mul(0.5).exp())
    p_z = Normal(torch.zeros_like(mu), torch.ones_like(log_var))
    kld_loss = kl_divergence(q_z_x, p_z).sum(1).mean()

    loss = recons_loss + kld_weight * kld_loss

    return {
        "loss": loss,
        "recons_loss": recons_loss.detach(),
        "KLD_loss": kld_loss.detach(),
    }


def weighted_reconstruction_mse(
    recons: torch.Tensor,
    target: torch.Tensor,
    *,
    lead_indices: Optional[torch.Tensor] = None,
    missing_lead_weight: float = 1.0,
) -> torch.Tensor:
    """Weighted full-12 reconstruction MSE.

    Defaults to the exact baseline behavior when ``missing_lead_weight == 1.0``.
    When increased, only missing leads receive extra weight, while observed leads
    remain weight 1.0. This keeps the baseline comparison path intact while
    allowing a sparse-reconstruction emphasis ablation.
    """
    if lead_indices is None or float(missing_lead_weight) == 1.0:
        return F.mse_loss(recons, target, reduction="mean")

    if lead_indices.dim() != 2:
        raise ValueError(f"Expected lead_indices shape [B, N_obs], got {tuple(lead_indices.shape)}")
    if recons.dim() != 3 or target.dim() != 3:
        raise ValueError("Expected reconstruction and target tensors shaped [B, T, 12].")
    if recons.shape != target.shape:
        raise ValueError(f"Expected matching reconstruction/target shapes, got {tuple(recons.shape)} vs {tuple(target.shape)}")

    batch_size, _, num_leads = recons.shape
    weights = recons.new_full((batch_size, num_leads), float(missing_lead_weight))
    observed_weight = recons.new_ones((batch_size, lead_indices.shape[1]))
    weights.scatter_(1, lead_indices.long(), observed_weight)
    weights = weights.unsqueeze(1)
    squared_error = (recons - target) ** 2
    time_len = recons.shape[1]
    return (squared_error * weights).sum() / (weights.sum().clamp(min=1.0) * time_len)


def pooled_fm_perceptual_loss(
    fm_model: ECGFMFeatureExtractor,
    x_recon_12: torch.Tensor,
    pooled_true: torch.Tensor,
    *,
    cosine_mix: float = 0.5,
) -> torch.Tensor:
    """
    Pooled FM perceptual loss:
    - MSE in pooled ECG-FM space
    - optional cosine component

    Args:
        x_recon_12: [B, 12, T]
    """
    recon_tokens = fm_model.extract_tokens(x_recon_12)
    recon_pool = fm_model.pooled_summary(recon_tokens)

    mse_term = F.mse_loss(recon_pool, pooled_true, reduction="mean")
    cos_term = 1.0 - F.cosine_similarity(recon_pool, pooled_true, dim=1).mean()

    return (1.0 - cosine_mix) * mse_term + cosine_mix * cos_term


def pooled_feature_loss(
    recon_pool: torch.Tensor,
    true_pool: torch.Tensor,
    *,
    cosine_mix: float = 0.5,
) -> torch.Tensor:
    """Pooled feature loss shared by ECGFM and token-teacher FM-VAE paths."""
    mse_term = F.mse_loss(recon_pool.float(), true_pool.detach().float(), reduction="mean")
    cos_term = 1.0 - F.cosine_similarity(recon_pool.float(), true_pool.detach().float(), dim=1).mean()
    return (1.0 - float(cosine_mix)) * mse_term + float(cosine_mix) * cos_term


# =============================================================================
# EXACT BASELINE WRAPPER
# =============================================================================

class WearECGVAE(nn.Module):
    """
    Exact-ish baseline compatibility wrapper.

    External API:
    - stage1_forward
    - impute_from_regressor
    - forward(..., mode='stage1')
    """

    def __init__(
        self,
        in_channels: int = 12,
        out_channels: int = 12,
        latent_channels: int = 4,
        target_len: int = 5000,
        beta_kl: float = 1e-4,
        chest_weighted: bool = False,
        missing_lead_weight: float = 1.0,
    ):
        super().__init__()
        if in_channels != 12 or out_channels != 12:
            raise ValueError("WearECGVAE expects 12-lead input/output.")
        if latent_channels != 4:
            raise ValueError("Exact WearECG VAE baseline uses latent_channels=4.")

        self.target_len = int(target_len)
        self.beta_kl = float(beta_kl)
        self.chest_weighted = bool(chest_weighted)
        self.missing_lead_weight = float(missing_lead_weight)

        self.encoder = VAE_Encoder()
        self.decoder = ConditionalVAE_Decoder(cond_dim=0)

    def _match_target_len(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] < self.target_len:
            x = F.pad(x, (0, 0, 0, self.target_len - x.shape[1]))
        elif x.shape[1] > self.target_len:
            x = x[:, : self.target_len, :]
        return x

    def stage1_forward(
        self,
        x: torch.Tensor,
        y_full: Optional[torch.Tensor] = None,
        lead_indices=None,
    ) -> Dict[str, torch.Tensor]:
        if x.dim() != 3 or x.shape[1] != 12:
            raise ValueError(f"Expected x shape [B, 12, T], got {tuple(x.shape)}")
        if y_full is not None and (y_full.dim() != 3 or y_full.shape[1] != 12):
            raise ValueError(f"Expected y_full shape [B, 12, T], got {tuple(y_full.shape)}")

        target_12 = y_full if y_full is not None else x
        z, mean, log_var = self.encoder(x)
        recons = self._match_target_len(self.decoder(z, cond_vec=None))
        # Keep loss computation sequence-major, but expose the public wrapper
        # contract as channel-major [B, 12, T].
        y_pred = recons.transpose(1, 2)
        loss = baseline_loss_function(
            recons,
            target_12.transpose(1, 2),
            mean,
            log_var,
            kld_weight=self.beta_kl,
            lead_indices=lead_indices,
            missing_lead_weight=self.missing_lead_weight,
        )

        zero = torch.tensor(0.0, device=x.device)
        return {
            "loss": loss["loss"],
            "decoder_loss": loss["recons_loss"],
            "teacher_loss": zero,
            "align_loss": zero,
            "stft_loss": zero,
            "diff_loss": zero,
            "corr_loss": zero,
            "kl_loss": loss["KLD_loss"],
            "y_target": target_12,
            "y_pred": y_pred,
            "y_pred_reg": y_pred,
            "z_regressed": mean.detach(),
        }

    @torch.no_grad()
    def impute_from_regressor(self, x: torch.Tensor, lead_indices=None) -> Dict[str, torch.Tensor]:
        if x.dim() != 3 or x.shape[1] != 12:
            raise ValueError(f"Expected x shape [B, 12, T], got {tuple(x.shape)}")

        _z, mean, _log_var = self.encoder(x)
        recons = self._match_target_len(self.decoder(mean * LATENT_SCALE, cond_vec=None))
        return {
            "available": True,
            "y_pred": recons.transpose(1, 2),
            "z_latent": mean.detach(),
            "log_var": _log_var.detach(),
        }

    def forward(self, x: torch.Tensor, lead_indices=None, mode: str = "stage1", **kwargs):
        if mode != "stage1":
            raise ValueError("WearECGVAE supports only mode='stage1'.")
        return self.stage1_forward(x, lead_indices=lead_indices, y_full=kwargs.get("y_full"))


# =============================================================================
# FM-INTEGRATED WRAPPER
# =============================================================================

class WearECGFMVAE(nn.Module):
    """
    WearECG baseline + ECG-FM integration.

    Integration strategy:
    1. Preserve the native WearECG encoder-decoder path.
    2. Add frozen ECG-FM perceptual supervision on reconstructions.
    3. Optionally add pooled ECG-FM decoder conditioning.

    External API matches the baseline wrapper.
    """

    def __init__(
        self,
        fm_checkpoint_path: str,
        fm_teacher_encoder: str = "ecgfm",
        teacher_checkpoint: Optional[str] = None,
        teacher_dim: int = 768,
        teacher_token_length: Optional[int] = None,
        teacher_common_token_length: int = 625,
        teacher_layer_mode: str = "last",
        random_teacher_seed: int = 1234,
        in_channels: int = 12,
        out_channels: int = 12,
        latent_channels: int = 4,
        target_len: int = 5000,
        beta_kl: float = 1e-4,
        chest_weighted: bool = False,
        missing_lead_weight: float = 1.0,
        fm_embed_dim: int = 768,
        fm_loss_weight: float = 1e-2,
        fm_cosine_mix: float = 0.5,
        use_decoder_conditioning: bool = False,
        fm_cond_drop_prob: float = 0.0,
        use_latent_alignment: bool = False,
        latent_align_weight: float = 1e-3,
        mask_aware_encoder: bool = True,
        split_latent: bool = True,
        global_latent_channels: int = 2,
        local_latent_channels: int = 2,
        use_multi_scale_align: bool = False,
        multi_scale_align_weight: float = 1e-1,
    ):
        super().__init__()
        if in_channels != 12 or out_channels != 12:
            raise ValueError("WearECGFMVAE expects 12-lead input/output.")
        if latent_channels != 4:
            raise ValueError("This implementation preserves the exact WearECG latent_channels=4 baseline.")

        self.target_len = int(target_len)
        self.beta_kl = float(beta_kl)
        self.chest_weighted = bool(chest_weighted)
        self.missing_lead_weight = float(missing_lead_weight)

        self.fm_loss_weight = float(fm_loss_weight)
        self.fm_cosine_mix = float(fm_cosine_mix)
        self.use_decoder_conditioning = bool(use_decoder_conditioning)
        self.fm_cond_drop_prob = float(fm_cond_drop_prob)
        self.use_latent_alignment = bool(use_latent_alignment)
        self.latent_align_weight = float(latent_align_weight)
        self.fm_embed_dim = int(fm_embed_dim)
        self.mask_aware_encoder = bool(mask_aware_encoder)
        self.split_latent = bool(split_latent)
        self.global_latent_channels = int(global_latent_channels)
        self.local_latent_channels = int(local_latent_channels)
        self.use_multi_scale_align = bool(use_multi_scale_align)
        self.multi_scale_align_weight = float(multi_scale_align_weight)
        self.fm_teacher_encoder = str(fm_teacher_encoder or "ecgfm")
        self.teacher_checkpoint = teacher_checkpoint
        self.teacher_common_token_length = int(teacher_common_token_length)
        self.teacher_layer_mode = str(teacher_layer_mode or "last")
        self.random_teacher_seed = int(random_teacher_seed)
        if self.global_latent_channels + self.local_latent_channels != 4:
            raise ValueError("WearECGFMVAE requires global_latent_channels + local_latent_channels == 4.")

        encoder_in_channels = 24 if self.mask_aware_encoder else 12
        self.encoder = VAE_Encoder(
            in_channels=encoder_in_channels,
            split_latent=self.split_latent,
            global_latent_channels=self.global_latent_channels,
            local_latent_channels=self.local_latent_channels,
        )

        self.fm_teacher_kind = "ecgfm_feature_extractor"
        if self.fm_teacher_encoder == "ecgfm":
            self.fm_model = ECGFMFeatureExtractor(
                checkpoint_path=teacher_checkpoint or fm_checkpoint_path,
                embed_dim=fm_embed_dim,
                allow_unexpected_keys=[
                    "quantizer.vars",
                    "quantizer.weight_proj.weight",
                    "quantizer.weight_proj.bias",
                    "project_q.weight",
                    "project_q.bias",
                    "final_proj.weight",
                    "final_proj.bias",
                ],
            )
            self.fm_embed_dim = int(fm_embed_dim)
        else:
            from src.reconstruction.unified_latents.engineering.token_refiner import build_token_teacher

            self.fm_teacher_kind = "token_teacher"
            self.fm_model = build_token_teacher(
                self.fm_teacher_encoder,
                teacher_checkpoint=teacher_checkpoint,
                teacher_dim=teacher_dim,
                teacher_token_length=teacher_token_length,
                random_seed=self.random_teacher_seed,
                teacher_layer_mode=self.teacher_layer_mode,
            )
            self.fm_embed_dim = int(getattr(self.fm_model, "embed_dim", teacher_dim))

        cond_dim = self.fm_embed_dim if self.use_decoder_conditioning else 0
        self.decoder = ConditionalVAE_Decoder(cond_dim=cond_dim)
        self.latent_projector = nn.Sequential(
            nn.Linear(4, 256),
            nn.SiLU(),
            nn.Linear(256, self.fm_embed_dim),
        )
        self.multi_scale_weights = {3: 0.1, 6: 0.1, 9: 0.1}
        self.scale_projectors = nn.ModuleDict()
        if self.use_multi_scale_align:
            for scale, in_channels in {3: 128, 6: 256, 9: 512}.items():
                self.scale_projectors[str(scale)] = nn.Sequential(
                    nn.Conv1d(in_channels, 256, kernel_size=1),
                    nn.SiLU(),
                    nn.Conv1d(256, self.fm_embed_dim, kernel_size=1),
                )

    def _lead_availability_mask(self, x_12: torch.Tensor, lead_indices) -> torch.Tensor:
        if lead_indices is None:
            return torch.ones_like(x_12)
        if isinstance(lead_indices, (list, tuple)):
            lead_indices = torch.tensor(lead_indices, device=x_12.device)
        if isinstance(lead_indices, torch.Tensor):
            if lead_indices.dim() == 1:
                lead_indices = lead_indices.unsqueeze(0).expand(x_12.size(0), -1)
            elif lead_indices.dim() == 2 and lead_indices.size(0) == 1:
                lead_indices = lead_indices.expand(x_12.size(0), -1)
            if lead_indices.dim() != 2:
                raise ValueError(f"Expected lead_indices shape [N_obs] or [B, N_obs], got {tuple(lead_indices.shape)}")
            mask = x_12.new_zeros((x_12.size(0), 12, 1))
            mask.scatter_(1, lead_indices.long().unsqueeze(-1), 1.0)
            return mask.expand(-1, -1, x_12.shape[-1])
        raise TypeError(f"Unsupported type for lead_indices: {type(lead_indices)}")

    def _prepare_encoder_input(self, x_12: torch.Tensor, lead_indices) -> tuple[torch.Tensor, torch.Tensor]:
        masked = mask_unobserved_leads(x_12, lead_indices) if lead_indices is not None else x_12
        if not self.mask_aware_encoder:
            return masked, masked
        availability = self._lead_availability_mask(masked, lead_indices)
        return torch.cat([masked, availability], dim=1), masked

    def _match_target_len(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] < self.target_len:
            x = F.pad(x, (0, 0, 0, self.target_len - x.shape[1]))
        elif x.shape[1] > self.target_len:
            x = x[:, : self.target_len, :]
        return x

    def _extract_fm_layers(self, x_12: torch.Tensor) -> List[torch.Tensor]:
        """Return one or more frozen teacher token layers aligned for pooled losses."""
        if self.fm_teacher_kind == "ecgfm_feature_extractor":
            return [self.fm_model.extract_tokens(x_12)]

        from src.reconstruction.unified_latents.engineering.token_refiner import align_token_length

        return [
            align_token_length(tokens, self.teacher_common_token_length)
            for tokens in self.fm_model.extract_token_layers(x_12)
        ]

    @staticmethod
    def _pooled_from_layers(token_layers: List[torch.Tensor]) -> List[torch.Tensor]:
        return [tokens.mean(dim=1) for tokens in token_layers]

    @staticmethod
    def _mean_pooled_summary(pooled_layers: List[torch.Tensor]) -> torch.Tensor:
        if len(pooled_layers) == 1:
            return pooled_layers[0]
        return torch.stack([pool.float() for pool in pooled_layers], dim=0).mean(dim=0)

    def _pooled_teacher_loss(
        self,
        x_recon_12: torch.Tensor,
        pooled_true_layers: List[torch.Tensor],
    ) -> torch.Tensor:
        recon_layers = self._extract_fm_layers(x_recon_12)
        pooled_recon_layers = self._pooled_from_layers(recon_layers)
        losses = [
            pooled_feature_loss(recon_pool, true_pool, cosine_mix=self.fm_cosine_mix)
            for recon_pool, true_pool in zip(pooled_recon_layers, pooled_true_layers)
        ]
        if not losses:
            return x_recon_12.new_tensor(0.0)
        return torch.stack(losses).mean()

    def _get_decoder_condition(self, x_12: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.use_decoder_conditioning:
            return None

        with torch.no_grad():
            token_layers = self._extract_fm_layers(x_12)
            cond_vec = self._mean_pooled_summary(self._pooled_from_layers(token_layers))

        if self.training and self.fm_cond_drop_prob > 0.0:
            keep = (torch.rand(cond_vec.size(0), device=cond_vec.device) > self.fm_cond_drop_prob).float()
            cond_vec = cond_vec * keep.unsqueeze(1)

        return cond_vec.to(x_12.dtype)

    def _latent_align_loss(self, mean: torch.Tensor, pooled_true: torch.Tensor) -> torch.Tensor:
        if not self.use_latent_alignment:
            return mean.new_tensor(0.0)
        mean_pool = mean.float().mean(dim=2)
        pred = self.latent_projector(mean_pool)
        return F.mse_loss(pred, pooled_true.detach().float(), reduction="mean")

    def _multi_scale_align_loss(
        self,
        intermediates: Optional[Dict[int, torch.Tensor]],
        true_tokens: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not self.use_multi_scale_align or intermediates is None or true_tokens is None:
            device = true_tokens.device if true_tokens is not None else next(self.parameters()).device
            return torch.tensor(0.0, device=device)
        true_tokens_t = true_tokens.transpose(1, 2).detach().float()
        fm_time = true_tokens_t.shape[-1]
        total = true_tokens_t.new_tensor(0.0)
        for scale, weight in self.multi_scale_weights.items():
            feat = intermediates.get(scale)
            if feat is None:
                continue
            projected = self.scale_projectors[str(scale)](feat.float())
            matched = F.adaptive_avg_pool1d(projected, fm_time)
            total = total + float(weight) * F.mse_loss(matched, true_tokens_t, reduction="mean")
        return self.multi_scale_align_weight * total

    def stage1_forward(
        self,
        x: torch.Tensor,
        y_full: Optional[torch.Tensor] = None,
        lead_indices=None,
    ) -> Dict[str, torch.Tensor]:
        if x.dim() != 3 or x.shape[1] != 12:
            raise ValueError(f"Expected x shape [B, 12, T], got {tuple(x.shape)}")
        if y_full is not None and (y_full.dim() != 3 or y_full.shape[1] != 12):
            raise ValueError(f"Expected y_full shape [B, 12, T], got {tuple(y_full.shape)}")

        target_12 = y_full if y_full is not None else x
        encoder_input, masked_signal = self._prepare_encoder_input(x, lead_indices)
        if self.use_multi_scale_align:
            z, mean, log_var, intermediates = self.encoder(encoder_input, return_intermediates=True)
        else:
            z, mean, log_var = self.encoder(encoder_input)
            intermediates = None

        cond_vec = self._get_decoder_condition(masked_signal)
        recons = self._match_target_len(self.decoder(z, cond_vec=cond_vec))
        # Keep loss computation sequence-major, but expose the public wrapper
        # contract as channel-major [B, 12, T].
        y_pred = recons.transpose(1, 2)

        base_loss = baseline_loss_function(
            recons,
            target_12.transpose(1, 2),
            mean,
            log_var,
            kld_weight=self.beta_kl,
            lead_indices=lead_indices,
            missing_lead_weight=self.missing_lead_weight,
        )

        # FM perceptual supervision
        with torch.no_grad():
            true_layers = self._extract_fm_layers(target_12)
            pooled_true_layers = self._pooled_from_layers(true_layers)
            pooled_true = self._mean_pooled_summary(pooled_true_layers)

        fm_teacher = self._pooled_teacher_loss(
            x_recon_12=y_pred,
            pooled_true_layers=[pool.detach() for pool in pooled_true_layers],
        )
        latent_align = self._latent_align_loss(mean, pooled_true)
        multi_scale_align = self._multi_scale_align_loss(intermediates, true_layers[-1] if true_layers else None)

        total_loss = (
            base_loss["loss"]
            + self.fm_loss_weight * fm_teacher
            + self.latent_align_weight * latent_align
            + multi_scale_align
        )
        zero = x.new_tensor(0.0)

        return {
            "loss": total_loss,
            "decoder_loss": base_loss["recons_loss"],
            "teacher_loss": fm_teacher.detach(),
            "align_loss": latent_align.detach() if self.use_latent_alignment else zero,
            "multi_scale_align_loss": multi_scale_align.detach() if self.use_multi_scale_align else zero,
            "stft_loss": zero,
            "diff_loss": zero,
            "corr_loss": zero,
            "kl_loss": base_loss["KLD_loss"],
            "y_target": target_12,
            "y_pred": y_pred,
            "y_pred_reg": y_pred,
            "z_regressed": mean.detach(),
            "fm_perceptual_loss": fm_teacher.detach(),
            "latent_align_loss": latent_align.detach() if self.use_latent_alignment else zero,
        }

    @torch.no_grad()
    def impute_from_regressor(self, x: torch.Tensor, lead_indices=None) -> Dict[str, torch.Tensor]:
        if x.dim() != 3 or x.shape[1] != 12:
            raise ValueError(f"Expected x shape [B, 12, T], got {tuple(x.shape)}")

        encoder_input, masked_signal = self._prepare_encoder_input(x, lead_indices)
        _z, mean, _log_var = self.encoder(encoder_input)

        cond_vec = self._get_decoder_condition(masked_signal)
        recons = self._match_target_len(self.decoder(mean * LATENT_SCALE, cond_vec=cond_vec))

        return {
            "available": True,
            "y_pred": recons.transpose(1, 2),
            "z_latent": mean.detach(),
            "log_var": _log_var.detach(),
        }

    def forward(self, x: torch.Tensor, lead_indices=None, mode: str = "stage1", **kwargs):
        if mode != "stage1":
            raise ValueError("WearECGFMVAE supports only mode='stage1'.")
        return self.stage1_forward(x, lead_indices=lead_indices, y_full=kwargs.get("y_full"))
