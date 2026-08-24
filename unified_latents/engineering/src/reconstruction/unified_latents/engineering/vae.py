"""Near-verbatim WearECG VAE modules and loss.

This file intentionally mirrors the third-party WearECG public VAE
implementation as closely as possible. The main local additions are:
- ASCII comments/docstrings
- a small compatibility wrapper (`WearECGVAE`) for local evaluators

The exact baseline training path should use `VAE_Encoder`, `VAE_Decoder`, and
`loss_function` directly.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import kl_divergence
from torch.distributions.normal import Normal

from src.reconstruction.unified_latents.engineering.common import mask_unobserved_leads


LATENT_SCALE = 0.18215


class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, causal_mask=False):
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

        weight /= math.sqrt(self.d_head)
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
        x += residue
        return x


class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
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
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Conv1d(12, 128, kernel_size=3, padding=1),
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
                nn.Conv1d(512, 8, kernel_size=3, padding=1),
                nn.Conv1d(8, 8, kernel_size=1, padding=0),
            ]
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        x = x.transpose(1, 2)

        for module in self.blocks:
            if getattr(module, "stride", None) == (2,):
                x = F.pad(x, (0, 1))
            x = module(x)

        mean, log_variance = torch.chunk(x, 2, dim=1)
        log_variance = torch.clamp(log_variance, -30, 20)
        variance = log_variance.exp()
        stdev = variance.sqrt()

        if noise is None:
            noise = torch.randn(stdev.shape, device=stdev.device)
        x = mean + stdev * noise
        x *= LATENT_SCALE

        return x, mean, log_variance


class VAE_Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Conv1d(4, 4, kernel_size=1, padding=0),
                nn.Conv1d(4, 512, kernel_size=3, padding=1),
                VAE_ResidualBlock(512, 512),
                VAE_AttentionBlock(512),
                VAE_ResidualBlock(512, 512),
                VAE_ResidualBlock(512, 512),
                VAE_ResidualBlock(512, 512),
                VAE_ResidualBlock(512, 512),
                nn.Upsample(scale_factor=2),
                nn.Conv1d(512, 512, kernel_size=3, padding=1),
                VAE_ResidualBlock(512, 512),
                VAE_ResidualBlock(512, 512),
                VAE_ResidualBlock(512, 512),
                nn.Upsample(scale_factor=2),
                nn.Conv1d(512, 512, kernel_size=3, padding=1),
                VAE_ResidualBlock(512, 512),
                VAE_ResidualBlock(512, 512),
                VAE_ResidualBlock(512, 512),
                nn.Upsample(scale_factor=2),
                nn.Conv1d(512, 512, kernel_size=3, padding=1),
                VAE_ResidualBlock(512, 256),
                VAE_ResidualBlock(256, 256),
                VAE_ResidualBlock(256, 256),
                nn.GroupNorm(32, 256),
                nn.SiLU(),
                nn.Conv1d(256, 12, kernel_size=3, padding=1),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x /= LATENT_SCALE

        for module in self.blocks:
            x = module(x)

        x = x.transpose(1, 2)
        return x


def loss_function(
    recons,
    x,
    mu,
    log_var,
    kld_weight=1e-4,
    perceptual_weight=1e-2,
    ecg_founder_model=None,
    device=None,
) -> dict:
    recons_loss = F.mse_loss(recons, x, reduction="mean")

    q_z_x = Normal(mu, log_var.mul(0.5).exp())
    p_z = Normal(torch.zeros_like(mu), torch.ones_like(log_var))
    kld_loss = kl_divergence(q_z_x, p_z).sum(1).mean()
    loss = recons_loss + kld_weight * kld_loss

    return {
        "loss": loss,
        "recons_loss": recons_loss.detach(),
        "KLD_loss": kld_loss.detach(),
        "perceptual_loss": torch.tensor(0.0, device=mu.device),
    }


class WearECGVAE(nn.Module):
    """Compatibility wrapper for local evaluation code.

    This wrapper preserves the exact public modules while exposing the minimal
    local `stage1`/`impute_from_regressor` API expected by the shared evaluator.
    """

    def __init__(self, in_channels: int = 12, out_channels: int = 12, latent_channels: int = 4, target_len: int = 5000, beta_kl: float = 1e-4, chest_weighted: bool = False):
        super().__init__()
        if in_channels != 12 or out_channels != 12:
            raise ValueError("WearECGVAE expects 12-lead input/output.")
        if latent_channels != 4:
            raise ValueError("Exact WearECG VAE baseline uses latent_channels=4.")
        self.target_len = int(target_len)
        self.beta_kl = float(beta_kl)
        self.chest_weighted = bool(chest_weighted)
        self.encoder = VAE_Encoder()
        self.decoder = VAE_Decoder()

    def _match_target_len(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] < self.target_len:
            x = F.pad(x, (0, 0, 0, self.target_len - x.shape[1]))
        elif x.shape[1] > self.target_len:
            x = x[:, : self.target_len, :]
        return x

    def stage1_forward(self, x: torch.Tensor, lead_indices=None, y_full: Optional[torch.Tensor] = None):
        target_12 = y_full if y_full is not None else x
        # Ensure the encoder only sees observed leads
        x_enc = mask_unobserved_leads(x, lead_indices)
        x_seq = x_enc.transpose(1, 2)
        z, mean, log_var = self.encoder(x_seq)
        recons = self._match_target_len(self.decoder(z))
        loss = loss_function(recons, target_12.transpose(1, 2), mean, log_var, kld_weight=self.beta_kl)
        y_pred = recons.transpose(1, 2)
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
            "y_target": x,
            "y_pred": y_pred,
            "y_pred_reg": y_pred,
            "z_regressed": mean.detach(),
        }

    @torch.no_grad()
    def impute_from_regressor(self, x: torch.Tensor, lead_indices=None):
        # Ensure the encoder only sees observed leads
        x_enc = mask_unobserved_leads(x, lead_indices)
        x_seq = x_enc.transpose(1, 2)
        _z, mean, _log_var = self.encoder(x_seq)
        recons = self._match_target_len(self.decoder(mean * LATENT_SCALE))
        return {
            "available": True,
            "y_pred": recons.transpose(1, 2),
            "z_latent": mean.detach(),
        }

    def forward(self, x: torch.Tensor, lead_indices=None, mode: str = "stage1", **kwargs):
        if mode != "stage1":
            raise ValueError("WearECGVAE supports only mode='stage1'.")
        return self.stage1_forward(x, lead_indices=lead_indices, y_full=kwargs.get("y_full"))
