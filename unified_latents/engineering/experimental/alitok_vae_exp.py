"""AliTok tokenizer adapted to 12-lead ECG reconstruction.

This module intentionally reuses AliTok's released attention block, latent-token
expansion helper, and vector quantizer. The local code is only the ECG adapter:
1D patch embedding/unpatching, sparse observed-lead masking, and compatibility
with the WearECG training/evaluation contract.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F

from unified_latents.engineering.utils.common import mask_unobserved_leads
from unified_latents.engineering.experimental.token_refiner import (
    align_token_length,
    build_token_teacher,
    compute_token_loss,
)
from unified_latents.engineering.models.vae_fm import weighted_reconstruction_mse


_ALITOK_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "alitok" / "alitok-main"
_ALITOK_TRAIN_TOKENIZER = _ALITOK_ROOT / "train_tokenizer"
_ALITOK_MODEL = _ALITOK_TRAIN_TOKENIZER / "model"
for _path in (str(_ALITOK_TRAIN_TOKENIZER), str(_ALITOK_MODEL)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# AliTok's tok_utils imports torchvision for image data utilities before
# defining all_gather. This ECG branch only needs all_gather for the released
# VectorQuantizer, so provide the same single-process/distributed helper when
# torchvision is not available in the ECG environment.
try:
    import torchvision  # noqa: F401
except ModuleNotFoundError:
    if "tok_utils" not in sys.modules:
        _tok_utils = types.ModuleType("tok_utils")

        def _all_gather(tensor: torch.Tensor) -> list[torch.Tensor]:
            if not dist.is_available() or not dist.is_initialized():
                return [tensor]
            world_size = dist.get_world_size()
            gathered = [torch.empty_like(tensor) for _ in range(world_size)]
            dist.all_gather(gathered, tensor)
            return gathered

        _tok_utils.all_gather = _all_gather
        sys.modules["tok_utils"] = _tok_utils

from vae_stage1 import (  # type: ignore  # noqa: E402
    ResidualAttentionBlock as AliTokResidualAttentionBlock,
    VectorQuantizer as AliTokVectorQuantizer,
    _expand_token as alitok_expand_token,
)


def _lead_availability_mask(x: torch.Tensor, lead_indices: Optional[torch.Tensor]) -> torch.Tensor:
    mask = torch.zeros_like(x)
    if lead_indices is None:
        return torch.ones_like(x)
    if lead_indices.dim() == 1:
        lead_indices = lead_indices.unsqueeze(0).expand(x.shape[0], -1)
    for batch_idx in range(x.shape[0]):
        mask[batch_idx, lead_indices[batch_idx].long()] = 1.0
    return mask


def _pad_to_patches(x: torch.Tensor, patch_size: int, num_patches: int) -> torch.Tensor:
    target_len = patch_size * num_patches
    if x.shape[-1] < target_len:
        x = F.pad(x, (0, target_len - x.shape[-1]))
    return x[..., :target_len]


def _crop_target_len(x: torch.Tensor, target_len: int) -> torch.Tensor:
    if x.shape[-1] < target_len:
        x = F.pad(x, (0, target_len - x.shape[-1]))
    return x[..., :target_len]


class AliTokECGEncoder(nn.Module):
    """AliTok bidirectional encoder with ECG-shaped patch embedding."""

    def __init__(
        self,
        *,
        target_len: int,
        patch_size: int,
        token_size: int,
        aux_tokens_num: int,
        width: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        input_channels: int = 24,
    ) -> None:
        super().__init__()
        if target_len % patch_size != 0:
            raise ValueError(f"target_len={target_len} must be divisible by patch_size={patch_size}")
        if width % num_heads != 0:
            raise ValueError(f"encoder width={width} must be divisible by heads={num_heads}")

        self.target_len = target_len
        self.patch_size = patch_size
        self.num_patches = target_len // patch_size
        self.aux_tokens_num = aux_tokens_num
        self.num_latent_tokens = self.num_patches + aux_tokens_num
        self.token_size = token_size
        self.width = width
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.patch_embed = nn.Conv2d(
            in_channels=input_channels,
            out_channels=width,
            kernel_size=(1, patch_size),
            stride=(1, patch_size),
            bias=True,
        )
        scale = width**-0.5
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.num_patches, width))
        self.latent_token_positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, width)
        )
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = nn.ModuleList(
            [AliTokResidualAttentionBlock(width, num_heads, mlp_ratio=4.0) for _ in range(num_layers)]
        )
        self.ln_post = nn.LayerNorm(width)
        self.conv_out = nn.Conv2d(width, token_size, kernel_size=1, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        latent_tokens: torch.Tensor,
        prefix_condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        x = _pad_to_patches(x, self.patch_size, self.num_patches).unsqueeze(2)
        x = self.patch_embed(x)
        x = x.reshape(batch_size, self.width, -1).permute(0, 2, 1)
        x = x + self.positional_embedding.to(x.dtype)

        latent_tokens = alitok_expand_token(latent_tokens, batch_size).to(x.dtype)
        latent_tokens = latent_tokens + self.latent_token_positional_embedding.to(x.dtype)
        if prefix_condition is not None:
            if prefix_condition.shape[:2] != (batch_size, self.aux_tokens_num):
                raise ValueError(
                    "Expected prefix_condition shape "
                    f"[{batch_size},{self.aux_tokens_num},D], got {tuple(prefix_condition.shape)}"
                )
            latent_tokens = latent_tokens.clone()
            latent_tokens[:, : self.aux_tokens_num] = (
                latent_tokens[:, : self.aux_tokens_num] + prefix_condition.to(latent_tokens.dtype)
            )
        x = torch.cat([x, latent_tokens], dim=1)

        x = self.ln_pre(x)
        for block in self.transformer:
            x = block(x)
        latent_tokens = self.ln_post(x[:, self.num_patches :])
        latent_tokens = latent_tokens.reshape(batch_size, self.num_latent_tokens, self.width, 1)
        latent_tokens = latent_tokens.permute(0, 2, 3, 1)
        return self.conv_out(latent_tokens)


class AliTokECGCausalDecoder(nn.Module):
    """AliTok stage-1 causal decoder adapted from image patches to ECG patches."""

    def __init__(
        self,
        *,
        target_len: int,
        patch_size: int,
        token_size: int,
        aux_tokens_num: int,
        width: int = 1024,
        num_layers: int = 24,
        num_heads: int = 16,
    ) -> None:
        super().__init__()
        if width % num_heads != 0:
            raise ValueError(f"decoder width={width} must be divisible by heads={num_heads}")
        self.target_len = target_len
        self.patch_size = patch_size
        self.num_patches = target_len // patch_size
        self.aux_tokens_num = aux_tokens_num
        self.num_latent_tokens = self.num_patches + aux_tokens_num
        self.token_size = token_size
        self.width = width
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.decoder_embed = nn.Linear(token_size, width, bias=True)
        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(1, width))
        self.latent_token_positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens + 1, width)
        )
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = nn.ModuleList(
            [AliTokResidualAttentionBlock(width, num_heads, mlp_ratio=4.0) for _ in range(num_layers)]
        )
        self.ln_post = nn.LayerNorm(width)
        self.patch_head = nn.Conv2d(width, 12 * patch_size, 1, padding=0, bias=True)
        self.aux_head = nn.Conv2d(width, 12 * patch_size, 1, padding=0, bias=True)

    def _unpatch(self, x: torch.Tensor, patches: int) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.permute(0, 2, 1).reshape(batch_size, self.width, 1, patches)
        x = self.patch_head(x.contiguous()) if patches == self.num_patches else self.aux_head(x.contiguous())
        x = x.reshape(batch_size, 12, self.patch_size, patches)
        return x.permute(0, 1, 3, 2).reshape(batch_size, 12, patches * self.patch_size)

    def forward(self, z_quantized: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, channels, height, width = z_quantized.shape
        if height != 1 or width != self.num_latent_tokens:
            raise ValueError(f"Expected quantized latents [B,C,1,{self.num_latent_tokens}], got {tuple(z_quantized.shape)}")

        x = z_quantized.reshape(batch_size, channels * height, width).permute(0, 2, 1)
        x = self.decoder_embed(x)
        x = torch.cat([alitok_expand_token(self.class_embedding, batch_size).to(x.dtype), x], dim=1)
        x = x + self.latent_token_positional_embedding.to(x.dtype)

        x = self.ln_pre(x)
        for block in self.transformer:
            x = block(x, is_causal=True)
        x = self.ln_post(x[:, 1:])

        x_aux = x[:, : self.aux_tokens_num]
        x_patch = x[:, -self.num_patches :]
        pred = self._unpatch(x_patch, self.num_patches)
        aux_pred = self._unpatch(x_aux, self.aux_tokens_num)
        return _crop_target_len(pred, self.target_len), aux_pred


class AliTokECGBidirectionalDecoder(nn.Module):
    """AliTok stage-2 bidirectional decoder with released buffer-token logic."""

    def __init__(
        self,
        *,
        target_len: int,
        patch_size: int,
        token_size: int,
        aux_tokens_num: int,
        buffer_tokens: int = 32,
        width: int = 1024,
        num_layers: int = 24,
        num_heads: int = 16,
    ) -> None:
        super().__init__()
        if width % num_heads != 0:
            raise ValueError(f"decoder width={width} must be divisible by heads={num_heads}")
        self.target_len = target_len
        self.patch_size = patch_size
        self.num_patches = target_len // patch_size
        self.aux_tokens_num = aux_tokens_num
        self.num_latent_tokens = self.num_patches + aux_tokens_num
        self.token_size = token_size
        self.width = width
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.buffer_tokens = buffer_tokens

        self.decoder_embed = nn.Linear(token_size, width, bias=True)
        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(buffer_tokens, width))
        self.latent_token_positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens + buffer_tokens, width)
        )
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = nn.ModuleList(
            [AliTokResidualAttentionBlock(width, num_heads, mlp_ratio=4.0) for _ in range(num_layers)]
        )
        self.ln_post = nn.LayerNorm(width)
        self.patch_head = nn.Conv2d(width, 12 * patch_size, 1, padding=0, bias=True)

    def forward(self, z_quantized: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = z_quantized.shape
        if height != 1 or width != self.num_latent_tokens:
            raise ValueError(f"Expected quantized latents [B,C,1,{self.num_latent_tokens}], got {tuple(z_quantized.shape)}")

        x = z_quantized.reshape(batch_size, channels * height, width).permute(0, 2, 1)
        x = self.decoder_embed(x)
        x = torch.cat([alitok_expand_token(self.class_embedding, batch_size).to(x.dtype), x], dim=1)
        x = x + self.latent_token_positional_embedding.to(x.dtype)

        x = self.ln_pre(x)
        for block in self.transformer:
            x = block(x)
        x = self.ln_post(x[:, -self.num_patches :])
        x = x.permute(0, 2, 1).reshape(batch_size, self.width, 1, self.num_patches)
        x = self.patch_head(x.contiguous())
        x = x.reshape(batch_size, 12, self.patch_size, self.num_patches)
        x = x.permute(0, 1, 3, 2).reshape(batch_size, 12, self.num_patches * self.patch_size)
        return _crop_target_len(x, self.target_len)


class ECGAdaptiveAxialBlock(nn.Module):
    """Bidirectional lead/time mixing without quadratic full-grid attention."""

    def __init__(self, width: int, heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.lead_norm = nn.LayerNorm(width)
        self.lead_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.time_norm = nn.LayerNorm(width)
        self.time_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(width)
        hidden = int(width * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Linear(hidden, width),
        )

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        batch, leads, patches, width = grid.shape
        lead_tokens = grid.permute(0, 2, 1, 3).reshape(batch * patches, leads, width)
        normalized = self.lead_norm(lead_tokens)
        lead_tokens = lead_tokens + self.lead_attention(
            normalized, normalized, normalized, need_weights=False
        )[0]
        grid = lead_tokens.reshape(batch, patches, leads, width).permute(0, 2, 1, 3)

        time_tokens = grid.reshape(batch * leads, patches, width)
        normalized = self.time_norm(time_tokens)
        time_tokens = time_tokens + self.time_attention(
            normalized, normalized, normalized, need_weights=False
        )[0]
        grid = time_tokens.reshape(batch, leads, patches, width)
        return grid + self.ffn(self.ffn_norm(grid))


class AliTokECGAIM(nn.Module):
    """ECG-specific Adaptive Inherited Masking model.

    The encoder sees only reliable lead-time patches. Missing tokens are
    reinserted before a lightweight bidirectional axial decoder. This is the
    high-frequency ECG analogue of LSM-2/AIM, with a deterministic limb-lead
    residual prior and explicit lead/time identities.
    """

    def __init__(
        self,
        *,
        target_len: int = 5000,
        patch_size: int = 25,
        width: int = 384,
        encoder_depth: int = 8,
        decoder_depth: int = 4,
        heads: int = 8,
        missing_lead_weight: float = 1.0,
        random_mask_ratio: float = 0.5,
        temporal_mask_ratio: float = 0.25,
        consistency_weight: float = 0.05,
    ) -> None:
        super().__init__()
        if target_len % patch_size:
            raise ValueError("ECG-AIM requires target_len divisible by patch_size")
        if width % heads:
            raise ValueError("ECG-AIM width must be divisible by heads")
        self.architecture = "ecg_aim_v1"
        self.target_len = int(target_len)
        self.patch_size = int(patch_size)
        self.num_patches = self.target_len // self.patch_size
        self.width = int(width)
        self.encoder_depth = int(encoder_depth)
        self.decoder_depth = int(decoder_depth)
        self.heads = int(heads)
        self.missing_lead_weight = float(missing_lead_weight)
        self.random_mask_ratio = float(random_mask_ratio)
        self.temporal_mask_ratio = float(temporal_mask_ratio)
        self.consistency_weight = float(consistency_weight)

        self.patch_projection = nn.Sequential(
            nn.LayerNorm(self.patch_size),
            nn.Linear(self.patch_size, self.width),
        )
        self.baseline_projection = nn.Linear(self.patch_size, self.width, bias=False)
        self.lead_embedding = nn.Parameter(torch.empty(12, self.width))
        self.time_embedding = nn.Parameter(torch.empty(self.num_patches, self.width))
        self.mask_token = nn.Parameter(torch.empty(1, 1, 1, self.width))
        self.mask_type_embedding = nn.Parameter(torch.empty(2, self.width))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.width,
            nhead=self.heads,
            dim_feedforward=4 * self.width,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.encoder_depth,
            norm=nn.LayerNorm(self.width),
            enable_nested_tensor=False,
        )
        self.decoder = nn.ModuleList(
            [ECGAdaptiveAxialBlock(self.width, self.heads) for _ in range(self.decoder_depth)]
        )
        self.output_norm = nn.LayerNorm(self.width)
        self.patch_head = nn.Linear(self.width, self.patch_size)
        self.residual_gain = nn.Parameter(torch.tensor(0.1))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.lead_embedding, std=0.02)
        nn.init.trunc_normal_(self.time_embedding, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.mask_type_embedding, std=0.02)
        nn.init.zeros_(self.patch_head.weight)
        nn.init.zeros_(self.patch_head.bias)

    def _patchify(self, signal: torch.Tensor) -> torch.Tensor:
        signal = _pad_to_patches(signal, self.patch_size, self.num_patches)
        return signal.reshape(signal.shape[0], 12, self.num_patches, self.patch_size)

    def _unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        signal = patches.reshape(patches.shape[0], 12, self.num_patches * self.patch_size)
        return _crop_target_len(signal, self.target_len)

    def _lead_mask(
        self, x: torch.Tensor, lead_indices: Optional[torch.Tensor]
    ) -> torch.Tensor:
        available = _lead_availability_mask(x, lead_indices)[:, :, 0].bool()
        return available[:, :, None].expand(-1, -1, self.num_patches)

    def _artificial_mask(self, inherited: torch.Tensor) -> torch.Tensor:
        artificial = torch.zeros_like(inherited)
        if not self.training:
            return artificial
        batch = inherited.shape[0]
        for index in range(batch):
            observed = torch.where(~inherited[index, :, 0])[0]
            if observed.numel() == 0:
                continue
            strategy = int(torch.randint(0, 3, (), device=inherited.device))
            if strategy == 0:
                draws = torch.rand(
                    observed.numel(), self.num_patches, device=inherited.device
                ) < self.random_mask_ratio
                artificial[index, observed] = draws
            elif strategy == 1:
                span = max(1, int(round(self.num_patches * self.temporal_mask_ratio)))
                start = int(
                    torch.randint(0, self.num_patches - span + 1, (), device=inherited.device)
                )
                artificial[index, observed, start : start + span] = True
            else:
                dropped = observed[
                    torch.randint(0, observed.numel(), (), device=inherited.device)
                ]
                artificial[index, dropped] = True
        return artificial

    def _scale(self, x: torch.Tensor, inherited: torch.Tensor) -> torch.Tensor:
        lead_available = (~inherited[:, :, 0]).to(x.dtype).unsqueeze(-1)
        denominator = lead_available.sum(dim=1, keepdim=True).clamp_min(1.0) * x.shape[-1]
        rms = torch.sqrt(
            (x.square() * lead_available).sum(dim=(1, 2), keepdim=True)
            / denominator
        )
        return rms.clamp_min(1e-3)

    def _limb_prior(self, patches: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
        baseline = torch.where(available.unsqueeze(-1), patches, torch.zeros_like(patches))
        both = available[:, 0] & available[:, 1]
        lead_i, lead_ii = patches[:, 0], patches[:, 1]
        derived = (
            lead_ii - lead_i,
            -0.5 * (lead_i + lead_ii),
            lead_i - 0.5 * lead_ii,
            lead_ii - 0.5 * lead_i,
        )
        for lead, value in zip((2, 3, 4, 5), derived):
            baseline[:, lead] = torch.where(
                both.unsqueeze(-1), value, baseline[:, lead]
            )
        return baseline

    def _encode_grid(
        self, tokens: torch.Tensor, available: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, leads, patches, width = tokens.shape
        encoded_sequences = []
        lengths = []
        for index in range(batch):
            sequence = tokens[index][available[index]]
            encoded_sequences.append(sequence)
            lengths.append(sequence.shape[0])
        max_length = max(lengths)
        padded = tokens.new_zeros(batch, max_length, width)
        padding_mask = torch.ones(batch, max_length, dtype=torch.bool, device=tokens.device)
        for index, sequence in enumerate(encoded_sequences):
            padded[index, : sequence.shape[0]] = sequence
            padding_mask[index, : sequence.shape[0]] = False
        memory = self.encoder(padded, src_key_padding_mask=padding_mask)

        grid = self.mask_token.expand(batch, leads, patches, width).clone()
        grid = grid + self.lead_embedding[None, :, None] + self.time_embedding[None, None]
        for index, length in enumerate(lengths):
            grid[index][available[index]] = memory[index, :length]
        return grid, memory

    def _decode(
        self,
        normalized: torch.Tensor,
        inherited: torch.Tensor,
        artificial: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        patches = self._patchify(normalized)
        available = ~(inherited | artificial)
        positions = self.lead_embedding[None, :, None] + self.time_embedding[None, None]
        tokens = self.patch_projection(patches) + positions
        tokens = tokens + self.mask_type_embedding[0][None, None, None]
        grid, memory = self._encode_grid(tokens, available)
        grid = grid + torch.where(
            inherited.unsqueeze(-1),
            self.mask_type_embedding[1][None, None, None],
            self.mask_type_embedding[0][None, None, None],
        )
        baseline = self._limb_prior(patches, available)
        grid = grid + self.baseline_projection(baseline)
        for block in self.decoder:
            if self.training and grid.device.type == "cpu":
                import torch.utils.checkpoint as cp
                grid = cp.checkpoint(block, grid, use_reentrant=False)
            else:
                grid = block(grid)
        residual = self.patch_head(self.output_norm(grid))
        prediction = baseline + self.residual_gain * residual
        return self._unpatchify(prediction), memory

    def _masked_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        inherited: torch.Tensor,
        artificial: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = inherited.to(prediction.dtype) * self.missing_lead_weight
        weights = weights + artificial.to(prediction.dtype)
        point_weights = weights.repeat_interleave(self.patch_size, dim=-1)[..., : target.shape[-1]]
        squared = (prediction - target).square()
        decoder_loss = (squared * point_weights).sum() / point_weights.sum().clamp_min(1.0)
        artificial_points = artificial.repeat_interleave(self.patch_size, dim=-1)[..., : target.shape[-1]]
        artificial_loss = (squared * artificial_points).sum() / artificial_points.sum().clamp_min(1.0)
        return decoder_loss, artificial_loss

    @staticmethod
    def _limb_consistency(prediction: torch.Tensor) -> torch.Tensor:
        lead_i, lead_ii = prediction[:, 0], prediction[:, 1]
        expected = torch.stack(
            [
                lead_ii - lead_i,
                -0.5 * (lead_i + lead_ii),
                lead_i - 0.5 * lead_ii,
                lead_ii - 0.5 * lead_i,
            ],
            dim=1,
        )
        return F.mse_loss(prediction[:, 2:6], expected)

    def forward(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        y_full: Optional[torch.Tensor] = None,
        lead_indices: Optional[torch.Tensor] = None,
        **_: object,
    ) -> dict[str, torch.Tensor]:
        target = target if target is not None else y_full
        if target is None:
            target = x
        inherited = ~self._lead_mask(x, lead_indices)
        artificial = self._artificial_mask(inherited)
        scale = self._scale(x, inherited)
        normalized_prediction, memory = self._decode(x / scale, inherited, artificial)
        prediction = normalized_prediction * scale
        decoder_loss, artificial_loss = self._masked_loss(
            prediction, target, inherited, artificial
        )
        consistency = self._limb_consistency(prediction)
        loss = decoder_loss + self.consistency_weight * consistency
        zero = loss.new_zeros(())
        latent = memory.transpose(1, 2)
        return {
            "loss": loss,
            "decoder_loss": decoder_loss.detach(),
            "kl_loss": zero.detach(),
            "teacher_loss": zero.detach(),
            "align_loss": zero.detach(),
            "stft_loss": zero.detach(),
            "diff_loss": zero.detach(),
            "corr_loss": zero.detach(),
            "fm_perceptual_loss": zero.detach(),
            "prefix_aux_loss": artificial_loss.detach(),
            "latent_align_loss": zero.detach(),
            "codebook_perplexity": zero.detach(),
            "limb_consistency_loss": consistency.detach(),
            "y_target": target,
            "y_pred": prediction,
            "y_pred_reg": prediction,
            "z_regressed": latent,
            "log_var_regressed": torch.zeros_like(latent),
        }

    @torch.no_grad()
    def impute_from_regressor(
        self,
        x: torch.Tensor,
        lead_indices: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor | bool]:
        inherited = ~self._lead_mask(x, lead_indices)
        artificial = torch.zeros_like(inherited)
        scale = self._scale(x, inherited)
        normalized_prediction, memory = self._decode(x / scale, inherited, artificial)
        prediction = normalized_prediction * scale
        return {
            "available": True,
            "y_pred": prediction,
            "z_latent": memory.transpose(1, 2),
            "log_var": torch.zeros_like(memory.transpose(1, 2)),
        }


class AliTokECGVAE1D(nn.Module):
    """WearECG-compatible AliTok tokenizer/VAE."""

    def __init__(
        self,
        *,
        architecture: str = "stage1_causal",
        target_len: int = 5000,
        patch_size: int = 10,
        token_size: int = 32,
        codebook_size: int = 4096,
        aux_tokens_num: int = 17,
        encoder_width: int = 768,
        decoder_width: int = 1024,
        encoder_depth: int = 12,
        decoder_depth: int = 24,
        encoder_heads: int = 12,
        decoder_heads: int = 16,
        stage2_buffer_tokens: int = 32,
        stage2_mix: float = 0.35,
        missing_lead_weight: float = 1.0,
        quantizer_weight: float = 1.0,
        aux_loss_weight: float = 1.0,
        commitment_cost: float = 0.25,
        clustering_vq: bool = False,
        teacher_encoder: Optional[str] = None,
        teacher_checkpoint: Optional[str] = None,
        teacher_dim: int = 768,
        teacher_token_length: Optional[int] = None,
        teacher_common_token_length: int = 625,
        teacher_layer_mode: str = "last",
        teacher_loss_weight: float = 0.05,
        teacher_loss_mix: float = 0.5,
        random_seed: int = 1234,
    ) -> None:
        super().__init__()
        if architecture not in {"stage1_causal", "stage2_bidir", "stage1_stage2_hybrid"}:
            raise ValueError(f"Unknown AliTok ECG architecture: {architecture}")
        self.architecture = architecture
        self.target_len = target_len
        self.patch_size = patch_size
        self.token_size = token_size
        self.num_patches = target_len // patch_size
        self.aux_tokens_num = aux_tokens_num
        self.num_latent_tokens = self.num_patches + aux_tokens_num
        self.missing_lead_weight = missing_lead_weight
        self.quantizer_weight = quantizer_weight
        self.aux_loss_weight = aux_loss_weight
        self.aux_loss_scale = 10.0 * float(aux_tokens_num) / float(self.num_patches)
        self.stage2_mix = stage2_mix
        self.teacher_encoder = teacher_encoder
        self.teacher_checkpoint = teacher_checkpoint
        self.teacher_common_token_length = int(teacher_common_token_length)
        self.teacher_loss_weight = float(teacher_loss_weight)
        self.teacher_loss_mix = float(teacher_loss_mix)
        self.teacher_layer_mode = str(teacher_layer_mode)

        self.encoder = AliTokECGEncoder(
            target_len=target_len,
            patch_size=patch_size,
            token_size=token_size,
            aux_tokens_num=aux_tokens_num,
            width=encoder_width,
            num_layers=encoder_depth,
            num_heads=encoder_heads,
            input_channels=24,
        )
        scale = encoder_width**-0.5
        self.latent_tokens = nn.Parameter(scale * torch.randn(self.num_latent_tokens, encoder_width))
        self.quantize = AliTokVectorQuantizer(
            codebook_size=codebook_size,
            token_size=token_size,
            commitment_cost=commitment_cost,
            use_l2_norm=False,
            clustering_vq=clustering_vq,
        )
        self.teacher: Optional[nn.Module]
        self.teacher_prefix_proj: Optional[nn.Module]
        self.teacher_layer_logits: Optional[nn.Parameter]
        self.teacher_token_projs: Optional[nn.ModuleList]
        if teacher_encoder:
            self.teacher = build_token_teacher(
                teacher_encoder,
                teacher_checkpoint=teacher_checkpoint,
                teacher_dim=teacher_dim,
                teacher_token_length=teacher_token_length,
                random_seed=random_seed,
                teacher_layer_mode=teacher_layer_mode,
            )
            for param in self.teacher.parameters():
                param.requires_grad = False
            self.teacher.eval()
            teacher_dim_actual = int(getattr(self.teacher, "embed_dim", teacher_dim))
            num_layers = int(getattr(self.teacher, "num_token_layers", 1))
            if num_layers <= 1:
                self.teacher_prefix_proj = nn.Linear(teacher_dim_actual, encoder_width)
                self.teacher_token_projs = None
                self.teacher_layer_logits = None
            else:
                self.teacher_prefix_proj = None
                self.teacher_token_projs = nn.ModuleList(
                    [nn.Linear(teacher_dim_actual, encoder_width) for _ in range(num_layers)]
                )
                self.teacher_layer_logits = nn.Parameter(torch.zeros(num_layers))
        else:
            self.teacher = None
            self.teacher_prefix_proj = None
            self.teacher_token_projs = None
            self.teacher_layer_logits = None
        self.causal_decoder = (
            AliTokECGCausalDecoder(
                target_len=target_len,
                patch_size=patch_size,
                token_size=token_size,
                aux_tokens_num=aux_tokens_num,
                width=decoder_width,
                num_layers=decoder_depth,
                num_heads=decoder_heads,
            )
            if architecture != "stage2_bidir"
            else None
        )
        self.bidir_decoder = (
            AliTokECGBidirectionalDecoder(
                target_len=target_len,
                patch_size=patch_size,
                token_size=token_size,
                aux_tokens_num=aux_tokens_num,
                buffer_tokens=stage2_buffer_tokens,
                width=decoder_width,
                num_layers=decoder_depth,
                num_heads=decoder_heads,
            )
            if architecture != "stage1_causal"
            else None
        )
        self.apply(self._init_weights)
        if architecture == "stage2_bidir":
            self.freeze_stage2_backbone()

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Embedding)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def freeze_stage2_backbone(self) -> None:
        for module in (self.encoder, self.quantize, self.causal_decoder):
            if module is None:
                continue
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
        for module in (self.teacher, self.teacher_prefix_proj, self.teacher_token_projs):
            if module is not None:
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
        if self.teacher_layer_logits is not None:
            self.teacher_layer_logits.requires_grad = False
        self.latent_tokens.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.architecture == "stage2_bidir":
            self.encoder.eval()
            self.quantize.eval()
            if self.causal_decoder is not None:
                self.causal_decoder.eval()
            if self.teacher_prefix_proj is not None:
                self.teacher_prefix_proj.eval()
            if self.teacher_token_projs is not None:
                self.teacher_token_projs.eval()
        if self.teacher is not None:
            self.teacher.eval()
        return self

    def _masked_encoder_input(self, x: torch.Tensor, lead_indices: Optional[torch.Tensor]) -> torch.Tensor:
        observed = mask_unobserved_leads(x, lead_indices, fill_value=0.0)
        availability = _lead_availability_mask(x, lead_indices)
        return torch.cat([observed, availability], dim=1)

    def _teacher_source(self, x: torch.Tensor, lead_indices: Optional[torch.Tensor]) -> torch.Tensor:
        return mask_unobserved_leads(x, lead_indices, fill_value=0.0)

    def _teacher_layers(self, x: torch.Tensor, target_len: int) -> list[torch.Tensor]:
        if self.teacher is None:
            return []
        layers = self.teacher.extract_token_layers(x)
        return [align_token_length(tokens, target_len) for tokens in layers]

    def _teacher_prefix_condition(self, x: torch.Tensor, lead_indices: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if self.teacher is None:
            return None
        with torch.no_grad():
            layers = self._teacher_layers(self._teacher_source(x, lead_indices), self.aux_tokens_num)
        if self.teacher_prefix_proj is not None:
            return self.teacher_prefix_proj(layers[-1].float()).to(x.dtype)
        if self.teacher_token_projs is None or self.teacher_layer_logits is None:
            return None
        weights = F.softmax(self.teacher_layer_logits.float(), dim=0)
        projected = [
            proj(tokens.float()).to(x.dtype) * weights[idx].to(x.dtype)
            for idx, (proj, tokens) in enumerate(zip(self.teacher_token_projs, layers))
        ]
        return torch.stack(projected, dim=0).sum(dim=0)

    def _foundation_token_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.teacher is None or self.teacher_loss_weight <= 0:
            return pred.new_zeros(())
        pred_layers = self._teacher_layers(pred, self.teacher_common_token_length)
        with torch.no_grad():
            target_layers = self._teacher_layers(target, self.teacher_common_token_length)
        losses = [
            compute_token_loss(pred_tokens, target_tokens, mix=self.teacher_loss_mix)
            for pred_tokens, target_tokens in zip(pred_layers, target_layers)
        ]
        if not losses:
            return pred.new_zeros(())
        return torch.stack(losses).mean()

    def encode(self, x: torch.Tensor, lead_indices: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        encoder_input = self._masked_encoder_input(x, lead_indices)
        prefix_condition = self._teacher_prefix_condition(x, lead_indices)
        z = self.encoder(encoder_input, self.latent_tokens, prefix_condition=prefix_condition)
        z_quantized, result_dict = self.quantize(z)
        return z, z_quantized, result_dict

    def decode(self, z_quantized: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.architecture == "stage1_causal":
            if self.causal_decoder is None:
                raise RuntimeError("stage1_causal requires a causal decoder")
            return self.causal_decoder(z_quantized)
        if self.architecture == "stage2_bidir":
            if self.bidir_decoder is None:
                raise RuntimeError("stage2_bidir requires a bidirectional decoder")
            pred = self.bidir_decoder(z_quantized)
            return pred, pred.new_zeros(pred.shape[0], 12, 0)
        if self.causal_decoder is None or self.bidir_decoder is None:
            raise RuntimeError("stage1_stage2_hybrid requires both decoders")
        causal_pred, aux_pred = self.causal_decoder(z_quantized)
        bidir_pred = self.bidir_decoder(z_quantized)
        pred = (1.0 - self.stage2_mix) * causal_pred + self.stage2_mix * bidir_pred
        return pred, aux_pred

    def _aux_loss(self, aux_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if aux_pred.numel() == 0:
            return target.new_zeros(())
        aux_len = min(aux_pred.shape[-1], target.shape[-1])
        return F.mse_loss(aux_pred[..., :aux_len], target[..., :aux_len], reduction="mean")

    @staticmethod
    def _perplexity(result_dict: dict[str, torch.Tensor], codebook_size: int) -> torch.Tensor:
        indices = result_dict.get("min_encoding_indices")
        if indices is None or indices.numel() == 0:
            return torch.tensor(0.0)
        counts = torch.bincount(indices.flatten().long(), minlength=codebook_size)
        avg_probs = counts.float() / max(indices.numel(), 1)
        return torch.exp(-(avg_probs * torch.log(avg_probs + 1e-10)).sum())

    def forward(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        y_full: Optional[torch.Tensor] = None,
        lead_indices: Optional[torch.Tensor] = None,
        **_: object,
    ) -> dict[str, torch.Tensor]:
        target = target if target is not None else y_full
        if target is None:
            target = x
        _, z_quantized, result_dict = self.encode(x, lead_indices)
        y_pred, aux_pred = self.decode(z_quantized)
        y_pred = _crop_target_len(y_pred, target.shape[-1])
        decoder_loss = weighted_reconstruction_mse(
            y_pred.transpose(1, 2),
            target.transpose(1, 2),
            lead_indices=lead_indices,
            missing_lead_weight=self.missing_lead_weight,
        )
        aux_loss = self._aux_loss(aux_pred, target)
        quantizer_loss = result_dict["quantizer_loss"].mean()
        foundation_loss = self._foundation_token_loss(y_pred, target)
        if self.architecture == "stage2_bidir":
            loss = decoder_loss + self.teacher_loss_weight * foundation_loss
        else:
            loss = (
                decoder_loss
                + self.quantizer_weight * quantizer_loss
                + self.aux_loss_weight * self.aux_loss_scale * aux_loss
                + self.teacher_loss_weight * foundation_loss
            )
        zero = loss.new_zeros(())
        codebook_size = int(self.quantize.codebook_size)
        perplexity = self._perplexity(result_dict, codebook_size).to(loss.device)
        return {
            "loss": loss,
            "decoder_loss": decoder_loss.detach(),
            "kl_loss": quantizer_loss.detach(),
            "teacher_loss": foundation_loss.detach(),
            "align_loss": quantizer_loss.detach(),
            "stft_loss": zero.detach(),
            "diff_loss": zero.detach(),
            "corr_loss": zero.detach(),
            "fm_perceptual_loss": foundation_loss.detach(),
            "prefix_aux_loss": aux_loss.detach(),
            "latent_align_loss": quantizer_loss.detach(),
            "codebook_perplexity": perplexity.detach(),
            "y_target": target,
            "y_pred": y_pred,
            "y_pred_reg": y_pred,
            "z_regressed": z_quantized.detach().squeeze(2),
            "log_var_regressed": torch.zeros_like(z_quantized.detach().squeeze(2)),
        }

    @torch.no_grad()
    def impute_from_regressor(
        self,
        x: torch.Tensor,
        lead_indices: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor | bool]:
        _, z_quantized, _ = self.encode(x, lead_indices)
        y_pred, _ = self.decode(z_quantized)
        z_flat = z_quantized.squeeze(2)
        return {
            "available": True,
            "y_pred": _crop_target_len(y_pred, x.shape[-1]),
            "z_latent": z_flat,
            "log_var": torch.zeros_like(z_flat),
        }


def build_alitok_vae_1d(
    *,
    architecture: str = "stage1_causal",
    target_len: int = 5000,
    patch_size: int = 10,
    token_size: int = 32,
    stage2_mix: float = 0.35,
    missing_lead_weight: float = 1.0,
    prefix_tokens: int = 17,
    codebook_size: int = 4096,
    encoder_depth: int = 12,
    decoder_depth: int = 24,
    heads: Optional[int] = None,
    encoder_heads: int = 12,
    decoder_heads: int = 16,
    encoder_width: int = 768,
    decoder_width: int = 1024,
    stage2_buffer_tokens: int = 32,
    quantizer_weight: float = 1.0,
    aux_loss_weight: float = 1.0,
    clustering_vq: bool = True,
    teacher_encoder: Optional[str] = None,
    teacher_checkpoint: Optional[str] = None,
    teacher_dim: int = 768,
    teacher_token_length: Optional[int] = None,
    teacher_common_token_length: int = 625,
    teacher_layer_mode: str = "last",
    teacher_loss_weight: float = 0.05,
    teacher_loss_mix: float = 0.5,
    random_seed: int = 1234,
) -> nn.Module:
    if architecture == "ecg_aim_v1":
        return AliTokECGAIM(
            target_len=target_len,
            patch_size=patch_size,
            width=encoder_width,
            encoder_depth=encoder_depth,
            decoder_depth=decoder_depth,
            heads=encoder_heads if heads is None else heads,
            missing_lead_weight=missing_lead_weight,
        )
    if heads is not None:
        encoder_heads = heads
        decoder_heads = heads
    return AliTokECGVAE1D(
        architecture=architecture,
        target_len=target_len,
        patch_size=patch_size,
        token_size=token_size,
        codebook_size=codebook_size,
        aux_tokens_num=prefix_tokens,
        encoder_width=encoder_width,
        decoder_width=decoder_width,
        encoder_depth=encoder_depth,
        decoder_depth=decoder_depth,
        encoder_heads=encoder_heads,
        decoder_heads=decoder_heads,
        stage2_buffer_tokens=stage2_buffer_tokens,
        stage2_mix=stage2_mix,
        missing_lead_weight=missing_lead_weight,
        quantizer_weight=quantizer_weight,
        aux_loss_weight=aux_loss_weight,
        clustering_vq=clustering_vq,
        teacher_encoder=teacher_encoder,
        teacher_checkpoint=teacher_checkpoint,
        teacher_dim=teacher_dim,
        teacher_token_length=teacher_token_length,
        teacher_common_token_length=teacher_common_token_length,
        teacher_layer_mode=teacher_layer_mode,
        teacher_loss_weight=teacher_loss_weight,
        teacher_loss_mix=teacher_loss_mix,
        random_seed=random_seed,
    )
