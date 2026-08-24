#!/usr/bin/env python3
"""
WearECG-FM v2.

This version keeps the stable WearECG decoder contract but changes the encoder
and training objective in a more meaningful way than the existing
multi-scale-alignment variant:

- multiresolution input stem to preserve morphology at multiple receptive fields
- dual latent factorization with a global semantic branch and a local morphology branch
- optional decoder FiLM conditioning from frozen ECG-FM pooled features
- explicit engineering losses for spectral shape, temporal derivatives,
  inter-lead coupling, chest progression, and lead algebra consistency
- optional multi-scale FM alignment that is properly weighted
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from unified_latents.engineering.utils.common import mask_unobserved_leads
from unified_latents.engineering.models.vae_fm import (
    LATENT_SCALE,
    ECGFMFeatureExtractor,
    ConditionalVAE_Decoder,
    VAE_AttentionBlock,
    VAE_ResidualBlock,
    WearECGVAE,
    baseline_loss_function,
    pooled_fm_perceptual_loss,
)


def _leadwise_correlation_matrix(x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: [B, 12, T]
    Returns:
        corr: [B, 12, 12]
    """
    x_centered = x - x.mean(dim=2, keepdim=True)
    cov = x_centered @ x_centered.transpose(1, 2)
    denom = torch.sqrt(
        torch.clamp((x_centered**2).sum(dim=2, keepdim=True), min=1e-6)
        * torch.clamp((x_centered**2).sum(dim=2, keepdim=True).transpose(1, 2), min=1e-6)
    )
    return cov / denom.clamp(min=1e-6)


def _first_difference(x: torch.Tensor) -> torch.Tensor:
    return x[:, :, 1:] - x[:, :, :-1]


def _rfft_mag(x: torch.Tensor, n_fft: int) -> torch.Tensor:
    x_fft = torch.fft.rfft(x.float(), n=n_fft, dim=2)
    return torch.abs(x_fft)


def spectral_loss(pred: torch.Tensor, target: torch.Tensor, n_fft: int = 512) -> torch.Tensor:
    pred_mag = _rfft_mag(pred, n_fft)
    tgt_mag = _rfft_mag(target, n_fft)
    return F.l1_loss(torch.log1p(pred_mag), torch.log1p(tgt_mag), reduction="mean")


def diff_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(_first_difference(pred), _first_difference(target), reduction="mean")


def corr_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(_leadwise_correlation_matrix(pred), _leadwise_correlation_matrix(target), reduction="mean")


def chest_progression_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    chest = [6, 7, 8, 9, 10, 11]
    pred_prog = pred[:, chest, :].abs().amax(dim=2)
    target_prog = target[:, chest, :].abs().amax(dim=2)
    return F.l1_loss(pred_prog, target_prog, reduction="mean")


def lead_algebra_consistency_loss(y_full: torch.Tensor) -> torch.Tensor:
    lead_i = y_full[:, 0, :]
    lead_ii = y_full[:, 1, :]
    lead_iii = y_full[:, 2, :]
    avr = y_full[:, 3, :]
    avl = y_full[:, 4, :]
    avf = y_full[:, 5, :]
    loss = y_full.new_tensor(0.0)
    loss = loss + F.l1_loss(lead_iii, lead_ii - lead_i)
    loss = loss + F.l1_loss(avr, -(lead_i + lead_ii) / 2.0)
    loss = loss + F.l1_loss(avl, lead_i - lead_ii / 2.0)
    loss = loss + F.l1_loss(avf, lead_ii - lead_i / 2.0)
    return loss / 4.0


class MultiResolutionStem(nn.Module):
    def __init__(self, in_channels: int = 12, branch_channels: int = 48, out_channels: int = 128):
        super().__init__()
        kernels = (7, 15, 31)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(in_channels, branch_channels, kernel_size=k, padding=k // 2),
                    nn.GroupNorm(16, branch_channels),
                    nn.SiLU(),
                )
                for k in kernels
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(branch_channels * len(kernels), out_channels, kernel_size=1),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([branch(x) for branch in self.branches], dim=1))


class DualPathVAEEncoder(nn.Module):
    """
    Shared multiresolution trunk with factorized latent heads:
    - global branch: pooled semantic state
    - local branch: morphology-preserving state
    Final latent size stays 4 channels for decoder compatibility.
    """

    def __init__(self) -> None:
        super().__init__()
        self.stem = MultiResolutionStem(in_channels=12, branch_channels=48, out_channels=128)
        self.blocks = nn.ModuleList(
            [
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
                VAE_AttentionBlock(512),
                VAE_ResidualBlock(512, 512),
                nn.GroupNorm(32, 512),
                nn.SiLU(),
            ]
        )
        self.local_head = nn.Sequential(
            nn.Conv1d(512, 128, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(128, 4, kernel_size=1),
        )
        self.global_head = nn.Sequential(
            nn.Conv1d(512, 128, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(128, 4, kernel_size=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        if x.dim() != 3:
            raise ValueError(f"Expected rank-3 input, got shape {tuple(x.shape)}")
        if x.shape[1] != 12 and x.shape[2] == 12:
            x = x.transpose(1, 2)
        elif x.shape[1] != 12:
            raise ValueError(f"Expected 12 channels, got shape {tuple(x.shape)}")

        intermediates: Dict[str, torch.Tensor] = {}
        x = self.stem(x)
        if return_intermediates:
            intermediates["stem"] = x

        for idx, module in enumerate(self.blocks):
            if getattr(module, "stride", None) == (2,):
                x = F.pad(x, (0, 1))
            x = module(x)
            if return_intermediates and idx in {2, 5, 8}:
                intermediates[f"down_{idx}"] = x

        local_stats = self.local_head(x)
        global_context = F.adaptive_avg_pool1d(x, 1).expand(-1, -1, x.shape[-1])
        global_stats = self.global_head(global_context)

        mean_local, logvar_local = torch.chunk(local_stats, 2, dim=1)
        mean_global, logvar_global = torch.chunk(global_stats, 2, dim=1)

        mean = torch.cat([mean_global, mean_local], dim=1)
        log_variance = torch.cat([logvar_global, logvar_local], dim=1)
        log_variance = torch.clamp(log_variance, -30, 20)

        stdev = log_variance.mul(0.5).exp()
        if noise is None:
            noise = torch.randn_like(stdev)
        z = (mean + stdev * noise) * LATENT_SCALE

        if return_intermediates:
            intermediates["trunk"] = x
            return z, mean, log_variance, intermediates
        return z, mean, log_variance


class WearECGFMVAEV2(nn.Module):
    def __init__(
        self,
        fm_checkpoint_path: str,
        in_channels: int = 12,
        out_channels: int = 12,
        latent_channels: int = 4,
        target_len: int = 5000,
        beta_kl: float = 1e-4,
        missing_lead_weight: float = 1.0,
        fm_embed_dim: int = 768,
        fm_loss_weight: float = 1e-2,
        fm_cosine_mix: float = 0.5,
        use_decoder_conditioning: bool = False,
        fm_cond_drop_prob: float = 0.0,
        use_latent_alignment: bool = True,
        latent_align_weight: float = 1e-3,
        use_multi_scale_align: bool = False,
        multi_scale_align_weight: float = 0.05,
        spectral_loss_weight: float = 0.10,
        diff_loss_weight: float = 0.25,
        corr_loss_weight: float = 0.20,
        progression_loss_weight: float = 0.15,
        algebra_loss_weight: float = 0.01,
    ) -> None:
        super().__init__()
        if in_channels != 12 or out_channels != 12:
            raise ValueError("WearECGFMVAEV2 expects 12-lead input/output.")
        if latent_channels != 4:
            raise ValueError("WearECGFMVAEV2 preserves latent_channels=4 for decoder compatibility.")

        self.target_len = int(target_len)
        self.beta_kl = float(beta_kl)
        self.missing_lead_weight = float(missing_lead_weight)
        self.fm_loss_weight = float(fm_loss_weight)
        self.fm_cosine_mix = float(fm_cosine_mix)
        self.use_decoder_conditioning = bool(use_decoder_conditioning)
        self.fm_cond_drop_prob = float(fm_cond_drop_prob)
        self.use_latent_alignment = bool(use_latent_alignment)
        self.latent_align_weight = float(latent_align_weight)
        self.use_multi_scale_align = bool(use_multi_scale_align)
        self.multi_scale_align_weight = float(multi_scale_align_weight)
        self.spectral_loss_weight = float(spectral_loss_weight)
        self.diff_loss_weight = float(diff_loss_weight)
        self.corr_loss_weight = float(corr_loss_weight)
        self.progression_loss_weight = float(progression_loss_weight)
        self.algebra_loss_weight = float(algebra_loss_weight)
        self.fm_embed_dim = int(fm_embed_dim)

        self.encoder = DualPathVAEEncoder()
        self.fm_model = ECGFMFeatureExtractor(
            checkpoint_path=fm_checkpoint_path,
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
        cond_dim = fm_embed_dim if self.use_decoder_conditioning else 0
        self.decoder = ConditionalVAE_Decoder(cond_dim=cond_dim)
        self.latent_projector = nn.Sequential(
            nn.Linear(4, 256),
            nn.SiLU(),
            nn.Linear(256, self.fm_embed_dim),
        )
        self.scale_projectors = nn.ModuleDict(
            {
                "stem": nn.Conv1d(128, fm_embed_dim, kernel_size=1),
                "down_2": nn.Conv1d(128, fm_embed_dim, kernel_size=1),
                "down_5": nn.Conv1d(256, fm_embed_dim, kernel_size=1),
                "down_8": nn.Conv1d(512, fm_embed_dim, kernel_size=1),
                "trunk": nn.Conv1d(512, fm_embed_dim, kernel_size=1),
            }
        )

    def _match_target_len(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] < self.target_len:
            x = F.pad(x, (0, 0, 0, self.target_len - x.shape[1]))
        elif x.shape[1] > self.target_len:
            x = x[:, : self.target_len, :]
        return x

    def _get_decoder_condition(self, x_12: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.use_decoder_conditioning:
            return None
        with torch.no_grad():
            tokens = self.fm_model.extract_tokens(x_12)
            cond_vec = self.fm_model.pooled_summary(tokens)
        if self.training and self.fm_cond_drop_prob > 0.0:
            keep = (torch.rand(cond_vec.size(0), device=cond_vec.device) > self.fm_cond_drop_prob).float()
            cond_vec = cond_vec * keep.unsqueeze(1)
        return cond_vec.to(x_12.dtype)

    def _latent_align_loss(self, mean: torch.Tensor, pooled_true: torch.Tensor) -> torch.Tensor:
        if not self.use_latent_alignment:
            return mean.new_tensor(0.0)
        mean_pool = mean.float().mean(dim=2)
        pred = self.latent_projector(mean_pool)
        mse = F.mse_loss(pred, pooled_true.detach().float(), reduction="mean")
        cos = 1.0 - F.cosine_similarity(pred, pooled_true.detach().float(), dim=1).mean()
        return 0.75 * mse + 0.25 * cos

    def _multi_scale_align_loss(
        self,
        intermediates: Optional[Dict[str, torch.Tensor]],
        true_tokens: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not self.use_multi_scale_align or intermediates is None or true_tokens is None:
            if true_tokens is not None:
                return true_tokens.new_tensor(0.0)
            return next(self.parameters()).new_tensor(0.0)

        target = true_tokens.transpose(1, 2).detach().float()
        token_len = target.shape[2]
        total = target.new_tensor(0.0)
        count = 0
        for key, feature in intermediates.items():
            if key not in self.scale_projectors:
                continue
            projector = self.scale_projectors[key]
            projected = projector(feature.float())
            projected = F.adaptive_avg_pool1d(projected, token_len)
            total = total + F.mse_loss(projected, target, reduction="mean")
            count += 1
        if count == 0:
            return target.new_tensor(0.0)
        return total / count

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
        x_enc = mask_unobserved_leads(x, lead_indices) if lead_indices is not None else x
        z, mean, log_var, intermediates = self.encoder(x_enc, return_intermediates=True)
        cond_vec = self._get_decoder_condition(x_enc)
        recons = self._match_target_len(self.decoder(z, cond_vec=cond_vec))
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

        with torch.no_grad():
            true_tokens = self.fm_model.extract_tokens(target_12)
            pooled_true = self.fm_model.pooled_summary(true_tokens)

        fm_teacher = pooled_fm_perceptual_loss(
            fm_model=self.fm_model,
            x_recon_12=y_pred,
            pooled_true=pooled_true.detach(),
            cosine_mix=self.fm_cosine_mix,
        )
        latent_align = self._latent_align_loss(mean, pooled_true)
        multi_scale = self._multi_scale_align_loss(intermediates, true_tokens)
        spectral = spectral_loss(y_pred, target_12)
        delta = diff_loss(y_pred, target_12)
        coupling = corr_loss(y_pred, target_12)
        progression = chest_progression_loss(y_pred, target_12)
        algebra = lead_algebra_consistency_loss(y_pred)

        engineering = (
            self.spectral_loss_weight * spectral
            + self.diff_loss_weight * delta
            + self.corr_loss_weight * coupling
            + self.progression_loss_weight * progression
            + self.algebra_loss_weight * algebra
        )
        total_loss = (
            base_loss["loss"]
            + self.fm_loss_weight * fm_teacher
            + self.latent_align_weight * latent_align
            + self.multi_scale_align_weight * multi_scale
            + engineering
        )
        zero = x.new_tensor(0.0)

        return {
            "loss": total_loss,
            "decoder_loss": base_loss["recons_loss"],
            "teacher_loss": fm_teacher.detach(),
            "align_loss": latent_align.detach() if self.use_latent_alignment else zero,
            "stft_loss": spectral.detach(),
            "diff_loss": delta.detach(),
            "corr_loss": coupling.detach(),
            "kl_loss": base_loss["KLD_loss"],
            "y_target": target_12,
            "y_pred": y_pred,
            "y_pred_reg": y_pred,
            "z_regressed": mean.detach(),
            "fm_perceptual_loss": fm_teacher.detach(),
            "latent_align_loss": latent_align.detach() if self.use_latent_alignment else zero,
            "multi_scale_align_loss": multi_scale.detach(),
            "progression_loss": progression.detach(),
            "algebra_loss": algebra.detach(),
        }

    @torch.no_grad()
    def impute_from_regressor(self, x: torch.Tensor, lead_indices=None) -> Dict[str, torch.Tensor]:
        if x.dim() != 3 or x.shape[1] != 12:
            raise ValueError(f"Expected x shape [B, 12, T], got {tuple(x.shape)}")
        x_enc = mask_unobserved_leads(x, lead_indices) if lead_indices is not None else x
        _z, mean, log_var, _intermediates = self.encoder(x_enc, return_intermediates=True)
        cond_vec = self._get_decoder_condition(x_enc)
        recons = self._match_target_len(self.decoder(mean * LATENT_SCALE, cond_vec=cond_vec))
        return {
            "available": True,
            "y_pred": recons.transpose(1, 2),
            "z_latent": mean.detach(),
            "log_var": log_var.detach(),
        }

    def forward(self, x: torch.Tensor, lead_indices=None, mode: str = "stage1", **kwargs):
        if mode != "stage1":
            raise ValueError("WearECGFMVAEV2 supports only mode='stage1'.")
        return self.stage1_forward(x, lead_indices=lead_indices, y_full=kwargs.get("y_full"))


__all__ = ["WearECGVAE", "WearECGFMVAEV2", "LATENT_SCALE"]
