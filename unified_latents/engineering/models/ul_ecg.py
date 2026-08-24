"""
Unified Latents ECG Engineering Track
-------------------------------------

Reconstruction-first branch:
    sparse observed ECG + mask -> reconstructive latent regressor -> latent
    -> deterministic decoder -> independent leads
    -> analytical limb-lead derivation -> full 12-lead ECG

Frozen FM usage:
- FM tokens are auxiliary conditioning and semantic regularization.
- FM tokens are NOT the sole reconstruction teacher target.

Key rules:
- The model does NOT freely predict dependent limb leads.
- Dependent limb leads are derived analytically:
    III  = II - I
    aVR  = -(I + II) / 2
    aVL  = I - II / 2
    aVF  = II - I / 2
- The default sparse regime remains I, II, V2.
- The deployed path is deterministic.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.reconstruction.unified_latents.cnvae_core.neural_operations_1d import (
    BNSwishConv,
    Conv1D,
    FactorizedReduce,
    SE,
    UpSample,
)


LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
INDEPENDENT_INDICES = [0, 1, 6, 7, 8, 9, 10, 11]
INDEPENDENT_NAME_TO_SLOT = {lead_idx: slot for slot, lead_idx in enumerate(INDEPENDENT_INDICES)}


def ckpt(fn, *args):
    enabled = torch.is_autocast_enabled()
    dtype = torch.get_autocast_gpu_dtype() if enabled else None

    def wrapped(*inputs):
        if enabled:
            with torch.amp.autocast("cuda", dtype=dtype):
                return fn(*inputs)
        return fn(*inputs)

    return checkpoint(
        wrapped,
        *args,
        use_reentrant=False,
        preserve_rng_state=True,
    )


def _safe_group_count(channels: int, preferred: int = 32) -> int:
    for groups in range(min(preferred, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _mean_cosine_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_n = F.normalize(x.float(), dim=-1)
    y_n = F.normalize(y.float(), dim=-1)
    return (1.0 - (x_n * y_n).sum(dim=-1)).mean()


class NVAECell(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=17, use_se=True):
        super().__init__()
        self.skip = nn.Identity() if in_c == out_c else Conv1D(in_c, out_c, kernel_size=1)
        self.op1 = BNSwishConv(in_c, out_c, kernel_size, padding=kernel_size // 2)
        self.op2 = BNSwishConv(out_c, out_c, kernel_size, padding=kernel_size // 2)
        self.use_se = use_se
        if use_se:
            self.se = SE(in_c, out_c)

    def forward(self, x):
        skip = self.skip(x)
        out = self.op1(x)
        out = self.op2(out)
        if self.use_se:
            out = self.se(out)
        return skip + 0.1 * out


class ViTBlock1D(nn.Module):
    def __init__(self, dim, num_heads=16, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, cross_cond=None):
        if cross_cond is not None:
            attn_out, _ = self.attn(self.norm1(x), cross_cond, cross_cond, need_weights=False)
        else:
            q = self.norm1(x)
            attn_out, _ = self.attn(q, q, q, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class WearECGResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(_safe_group_count(in_channels), in_channels)
        self.conv_1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.groupnorm_2 = nn.GroupNorm(_safe_group_count(out_channels), out_channels)
        self.conv_2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.residual_layer = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        residual = x
        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)
        return x + self.residual_layer(residual)


class WearECGResidualRefiner(nn.Module):
    def __init__(self, in_channels=24, out_channels=len(INDEPENDENT_INDICES)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=3, padding=1),
            WearECGResidualBlock1D(128, 128),
            WearECGResidualBlock1D(128, 128),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            WearECGResidualBlock1D(64, 64),
            nn.GroupNorm(_safe_group_count(64), 64),
            nn.SiLU(),
            nn.Conv1d(64, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class UL_Encoder(nn.Module):
    """
    Reconstruction-oriented teacher latent encoder.
    """

    def __init__(self, in_channels=12, hidden_dim=128, latent_dim=32):
        super().__init__()
        self.conv_in = Conv1D(in_channels, hidden_dim, kernel_size=17, padding=8)
        self.down1 = nn.Sequential(
            FactorizedReduce(hidden_dim, hidden_dim),
            NVAECell(hidden_dim, hidden_dim, kernel_size=17),
        )
        self.down2 = nn.Sequential(
            FactorizedReduce(hidden_dim, hidden_dim * 2),
            NVAECell(hidden_dim * 2, hidden_dim * 2, kernel_size=17),
        )
        self.down3 = nn.Sequential(
            FactorizedReduce(hidden_dim * 2, hidden_dim * 4),
            NVAECell(hidden_dim * 4, hidden_dim * 4, kernel_size=17),
        )
        self.down4 = nn.Sequential(
            FactorizedReduce(hidden_dim * 4, hidden_dim * 4),
            NVAECell(hidden_dim * 4, hidden_dim * 4, kernel_size=17),
        )
        self.process = nn.Sequential(
            NVAECell(hidden_dim * 4, hidden_dim * 4, kernel_size=17),
            NVAECell(hidden_dim * 4, hidden_dim * 4, kernel_size=17),
            NVAECell(hidden_dim * 4, hidden_dim * 4, kernel_size=17),
        )
        self.proj = Conv1D(hidden_dim * 4, latent_dim, kernel_size=1, weight_norm=True)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x):
        h = self.conv_in(x)
        h = self.down1(h)
        h = self.down2(h)
        h = self.down3(h)
        h = self.down4(h)
        h = self.process(h)
        z = self.proj(h)
        z = self.norm(z.transpose(1, 2)).transpose(1, 2)
        if z.shape[-1] > 312:
            z = z[:, :, :312]
        return z


class UL_AlignmentRegressor(nn.Module):
    """
    Predicts reconstructive latent from sparse condition and sparse FM tokens.
    """

    def __init__(self, in_channels=24, hidden_dim=128, latent_dim=32, fm_embed_dim=768):
        super().__init__()
        self.conv_in = Conv1D(in_channels, hidden_dim, kernel_size=17, padding=8)
        self.down1 = nn.Sequential(
            FactorizedReduce(hidden_dim, hidden_dim),
            NVAECell(hidden_dim, hidden_dim, kernel_size=17),
        )
        self.down2 = nn.Sequential(
            FactorizedReduce(hidden_dim, hidden_dim * 2),
            NVAECell(hidden_dim * 2, hidden_dim * 2, kernel_size=17),
        )
        self.down3 = nn.Sequential(
            FactorizedReduce(hidden_dim * 2, hidden_dim * 4),
            NVAECell(hidden_dim * 4, hidden_dim * 4, kernel_size=17),
        )
        self.down4 = nn.Sequential(
            FactorizedReduce(hidden_dim * 4, hidden_dim * 4),
            NVAECell(hidden_dim * 4, hidden_dim * 4, kernel_size=17),
        )
        self.fm_norm = nn.LayerNorm(fm_embed_dim)
        self.fm_proj = nn.Linear(fm_embed_dim, hidden_dim * 4)
        self.process = nn.Sequential(
            NVAECell(hidden_dim * 4, hidden_dim * 4, kernel_size=17),
            NVAECell(hidden_dim * 4, hidden_dim * 4, kernel_size=17),
        )
        self.proj = Conv1D(hidden_dim * 4, latent_dim, kernel_size=1, weight_norm=True)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x_cond, e_fm):
        h = self.conv_in(x_cond)
        h = self.down1(h)
        h = self.down2(h)
        h = self.down3(h)
        h = self.down4(h)
        if h.shape[-1] > 312:
            h = h[:, :, :312]
        h = h + self.fm_proj(self.fm_norm(e_fm)).transpose(1, 2)
        h = self.process(h)
        z = self.proj(h)
        return self.norm(z.transpose(1, 2)).transpose(1, 2)


class UL_DeterministicDecoder(nn.Module):
    """
    Reconstruction decoder driven by reconstructive latent, sparse condition, and FM tokens.
    Outputs only independent leads.
    """

    def __init__(self, in_channels=24, out_channels=len(INDEPENDENT_INDICES), latent_dim=32, fm_embed_dim=768, hidden_dim=128):
        super().__init__()
        self.z_proj = nn.Linear(latent_dim, hidden_dim * 8)
        self.fm_norm = nn.LayerNorm(fm_embed_dim)
        self.fm_proj = nn.Linear(fm_embed_dim, hidden_dim * 8)
        self.conv_in = Conv1D(in_channels, hidden_dim, kernel_size=17, padding=8)
        self.down1 = nn.Sequential(
            FactorizedReduce(hidden_dim, hidden_dim),
            NVAECell(hidden_dim, hidden_dim, kernel_size=17),
        )
        self.down2 = nn.Sequential(
            FactorizedReduce(hidden_dim, hidden_dim * 2),
            NVAECell(hidden_dim * 2, hidden_dim * 2, kernel_size=17),
        )
        self.down3 = nn.Sequential(
            FactorizedReduce(hidden_dim * 2, hidden_dim * 4),
            NVAECell(hidden_dim * 4, hidden_dim * 4, kernel_size=17),
        )
        self.down4 = nn.Sequential(
            FactorizedReduce(hidden_dim * 4, hidden_dim * 4),
            NVAECell(hidden_dim * 4, hidden_dim * 4, kernel_size=17),
        )
        self.to_bot = nn.Linear(hidden_dim * 4, hidden_dim * 8)
        depth = 8
        self.bot_blocks = nn.ModuleList([ViTBlock1D(hidden_dim * 8) for _ in range(depth)])
        self.bot_cross = nn.ModuleList([ViTBlock1D(hidden_dim * 8) for _ in range(depth)])
        self.from_bot = nn.Linear(hidden_dim * 8, hidden_dim * 4)
        self.seq_pos_emb = nn.Parameter(torch.randn(1, 312, hidden_dim * 8) * 0.02)
        self.up4 = UpSample()
        self.merge4 = nn.Sequential(
            Conv1D(hidden_dim * 8, hidden_dim * 4, kernel_size=1),
            NVAECell(hidden_dim * 4, hidden_dim * 4, kernel_size=17),
        )
        self.up3 = UpSample()
        self.merge3 = nn.Sequential(
            Conv1D(hidden_dim * 6, hidden_dim * 2, kernel_size=1),
            NVAECell(hidden_dim * 2, hidden_dim * 2, kernel_size=17),
        )
        self.up2 = UpSample()
        self.merge2 = nn.Sequential(
            Conv1D(hidden_dim * 3, hidden_dim, kernel_size=1),
            NVAECell(hidden_dim, hidden_dim, kernel_size=17),
        )
        self.up1 = UpSample()
        self.merge1 = nn.Sequential(
            Conv1D(hidden_dim * 2, hidden_dim, kernel_size=1),
            NVAECell(hidden_dim, hidden_dim, kernel_size=17),
        )
        self.conv_out = Conv1D(hidden_dim, out_channels, kernel_size=17, padding=8)

    def forward(self, x_cond, z_latent, e_fm):
        z_emb = self.z_proj(z_latent.transpose(1, 2))
        e_fm_emb = self.fm_proj(self.fm_norm(e_fm))
        h1 = self.conv_in(x_cond)
        h2 = self.down1(h1)
        h3 = self.down2(h2)
        h4 = self.down3(h3)
        h5 = self.down4(h4)
        if h5.shape[-1] > 312:
            h5 = h5[:, :, :312]

        h_bot = self.to_bot(h5.transpose(1, 2))
        h_bot = h_bot + z_emb + self.seq_pos_emb
        e_fm_emb = e_fm_emb + self.seq_pos_emb

        for block, cross in zip(self.bot_blocks, self.bot_cross):
            h_bot = ckpt(block, h_bot)
            h_bot = ckpt(lambda x, c: cross(x, cross_cond=c), h_bot, e_fm_emb)

        h_bot = self.from_bot(h_bot).transpose(1, 2)

        u4 = self.up4(h_bot)
        if u4.shape[-1] != h4.shape[-1]:
            u4 = F.pad(u4, (0, h4.shape[-1] - u4.shape[-1]))
        u4 = self.merge4(torch.cat([u4, h4], dim=1))

        u3 = self.up3(u4)
        if u3.shape[-1] != h3.shape[-1]:
            u3 = F.pad(u3, (0, h3.shape[-1] - u3.shape[-1]))
        u3 = self.merge3(torch.cat([u3, h3], dim=1))

        u2 = self.up2(u3)
        if u2.shape[-1] != h2.shape[-1]:
            u2 = F.pad(u2, (0, h2.shape[-1] - u2.shape[-1]))
        u2 = self.merge2(torch.cat([u2, h2], dim=1))

        u1 = self.up1(u2)
        if u1.shape[-1] != h1.shape[-1]:
            u1 = F.pad(u1, (0, h1.shape[-1] - u1.shape[-1]))
        u1 = self.merge1(torch.cat([u1, h1], dim=1))

        return self.conv_out(u1)


class UL_ConditionalBridge(nn.Module):
    def __init__(
        self,
        checkpoint_path,
        embed_dim=768,
        freeze_backbone=True,
        target_len=5000,
        latent_dim=32,
        teacher_loss_weight=0.10,
        reg_loss_weight=1.0,
        align_loss_weight=0.5,
        no_align=False,
        enable_skip_branch=True,
        finetune_mode="anchored",
        use_fm_perceptual=True,
        fm_perceptual_weight=0.10,
    ):
        super().__init__()
        if finetune_mode not in {"anchored", "strict"}:
            raise ValueError("finetune_mode must be 'anchored' or 'strict'.")

        self.target_len = int(target_len)
        self.semantic_dim = int(embed_dim)
        self.latent_dim = int(latent_dim)
        self.use_engineering_losses = True
        self.enable_skip_branch = bool(enable_skip_branch)
        self.finetune_mode = finetune_mode
        self.use_fm_perceptual = bool(use_fm_perceptual)
        self.fm_perceptual_weight = float(fm_perceptual_weight)
        self.teacher_loss_weight = float(teacher_loss_weight)
        self.reg_loss_weight = float(reg_loss_weight)
        self.align_loss_weight = float(align_loss_weight)
        self.freeze_teacher_encoder = False
        self.recon_finetune = False

        ecg_fm_path = os.path.expanduser("~/ecg_fm_integration/fairseq-signals")
        if ecg_fm_path not in sys.path:
            sys.path.append(ecg_fm_path)

        from fairseq_signals.models.ecg_transformer import ECGTransformerModel
        from fairseq_signals.utils.checkpoint_utils import load_checkpoint_to_cpu
        from omegaconf import OmegaConf

        state = load_checkpoint_to_cpu(checkpoint_path)
        cfg = state["cfg"]["model"]
        OmegaConf.set_struct(cfg, False)
        if getattr(cfg, "saliency", None) is None:
            cfg.saliency = False

        self.backbone = ECGTransformerModel.build_model(cfg)
        self.backbone.load_state_dict(state["model"], strict=False)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        self.encoder = UL_Encoder(in_channels=12, hidden_dim=128, latent_dim=self.latent_dim)
        self.latent_to_fm = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.semantic_dim),
        )
        if not no_align:
            self.alignment_head = UL_AlignmentRegressor(
                in_channels=24,
                hidden_dim=128,
                latent_dim=self.latent_dim,
                fm_embed_dim=self.semantic_dim,
            )
        else:
            self.alignment_head = None

        self.decoder = UL_DeterministicDecoder(
            in_channels=24,
            out_channels=len(INDEPENDENT_INDICES),
            latent_dim=self.latent_dim,
            fm_embed_dim=self.semantic_dim,
            hidden_dim=128,
        )

        if self.enable_skip_branch:
            self.obs_to_miss = WearECGResidualRefiner(
                in_channels=24,
                out_channels=len(INDEPENDENT_INDICES),
            )
            self.skip_scale_logit = nn.Parameter(torch.tensor(-1.3862944))
        else:
            self.obs_to_miss = None
            self.skip_scale_logit = None

        class STFTLoss(nn.Module):
            def __init__(self, fft_sizes=[64, 128, 256], hop_sizes=[16, 32, 64], win_lengths=[64, 128, 256]):
                super().__init__()
                self.fft_sizes = fft_sizes
                self.hop_sizes = hop_sizes
                self.win_lengths = win_lengths

            def forward(self, pred, target, lead_weights=None):
                pred = pred.float()
                target = target.float()
                batch_size, num_leads, signal_len = pred.shape
                loss = pred.new_zeros((batch_size, num_leads))
                for fft_size, hop_size, win_length in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
                    window = torch.hann_window(win_length, device=pred.device)
                    pred_stft = torch.stft(
                        pred.reshape(-1, signal_len),
                        n_fft=fft_size,
                        hop_length=hop_size,
                        win_length=win_length,
                        window=window,
                        return_complex=True,
                        center=True,
                    )
                    target_stft = torch.stft(
                        target.reshape(-1, signal_len),
                        n_fft=fft_size,
                        hop_length=hop_size,
                        win_length=win_length,
                        window=window,
                        return_complex=True,
                        center=True,
                    )
                    pred_mag = torch.abs(pred_stft).reshape(batch_size, num_leads, pred_stft.shape[-2], pred_stft.shape[-1])
                    target_mag = torch.abs(target_stft).reshape(batch_size, num_leads, target_stft.shape[-2], target_stft.shape[-1])
                    loss = loss + torch.abs(pred_mag - target_mag).mean(dim=(2, 3))
                    loss = loss + torch.abs(torch.log(pred_mag + 1e-7) - torch.log(target_mag + 1e-7)).mean(dim=(2, 3))
                loss = loss / len(self.fft_sizes)
                if lead_weights is not None:
                    loss = loss * lead_weights.view(1, -1)
                return loss.mean()

        self.stft_loss_fn = STFTLoss()

        full_weights = torch.tensor([0.5, 0.5, 0.10, 0.10, 0.10, 0.10, 1.0, 1.0, 1.5, 2.0, 2.25, 2.5])
        self.register_buffer("full_lead_weights", full_weights.view(1, 12, 1))

    @property
    def skip_scale(self):
        return self.get_skip_scale()

    def get_skip_scale(self):
        if self.skip_scale_logit is None:
            return None
        return torch.sigmoid(self.skip_scale_logit) * 0.25

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        if self.freeze_teacher_encoder or self.recon_finetune:
            self.encoder.eval()
        return self

    def _resolve_indices(self, lead_indices: Optional[torch.Tensor]) -> Tuple[List[int], List[int]]:
        if lead_indices is None:
            obs_indices = [0, 1, 7]
        else:
            obs_indices = lead_indices[0].tolist()
        miss_indices = [i for i in range(12) if i not in obs_indices]
        return obs_indices, miss_indices

    def _make_condition_inputs(self, x: torch.Tensor, obs_indices: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        x_masked = torch.zeros_like(x)
        x_masked[:, obs_indices, :] = x[:, obs_indices, :]
        obs_mask = torch.zeros_like(x_masked)
        obs_mask[:, obs_indices, :] = 1.0
        x_cond = torch.cat([x_masked, obs_mask], dim=1)
        return x_masked, x_cond

    def _extract_fm_tokens(self, x_full: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.backbone.eval()
            return self.backbone.extract_features(x_full.float(), padding_mask=None)["x"]

    def extract_condition(self, x: torch.Tensor, lead_indices: Optional[torch.Tensor] = None):
        obs_indices, miss_indices = self._resolve_indices(lead_indices)
        x_masked, x_cond = self._make_condition_inputs(x, obs_indices)
        z_teacher = self.encoder(x)
        e_fm_full = self._extract_fm_tokens(x)
        e_fm_sparse = self._extract_fm_tokens(x_masked)
        return x_cond, x, z_teacher, e_fm_full, e_fm_sparse, obs_indices, miss_indices

    def _assemble_from_independent_prediction(self, x_true: torch.Tensor, pred_independent: torch.Tensor, obs_indices: Sequence[int]) -> torch.Tensor:
        bsz, _, signal_len = x_true.shape
        y = torch.zeros((bsz, 12, signal_len), device=x_true.device, dtype=x_true.dtype)
        for lead_idx in INDEPENDENT_INDICES:
            if lead_idx in obs_indices:
                y[:, lead_idx, :] = x_true[:, lead_idx, :]
            else:
                slot = INDEPENDENT_NAME_TO_SLOT[lead_idx]
                y[:, lead_idx, :] = pred_independent[:, slot, :].to(x_true.dtype)

        lead_i = y[:, 0, :]
        lead_ii = y[:, 1, :]
        y[:, 2, :] = lead_ii - lead_i
        y[:, 3, :] = -(lead_i + lead_ii) / 2.0
        y[:, 4, :] = lead_i - lead_ii / 2.0
        y[:, 5, :] = lead_ii - lead_i / 2.0

        for idx in obs_indices:
            y[:, idx, :] = x_true[:, idx, :]
        return y

    def _get_canonical_target_mask(self, obs_indices: Sequence[int]) -> List[bool]:
        return [lead_idx not in obs_indices for lead_idx in INDEPENDENT_INDICES]

    def _slice_canonical_missing(self, pred_independent: torch.Tensor, x_true: torch.Tensor, obs_indices: Sequence[int]):
        mask = self._get_canonical_target_mask(obs_indices)
        pred = pred_independent[:, mask, :]
        target_indices = [lead_idx for lead_idx, keep in zip(INDEPENDENT_INDICES, mask) if keep]
        target = x_true[:, target_indices, :]
        weights = self.full_lead_weights[:, target_indices, :]
        return pred, target, weights, target_indices

    def _compute_corr_loss(self, x_pred: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
        xp = x_pred.float()
        xp = xp - xp.mean(dim=-1, keepdim=True)
        xp = xp / (xp.std(dim=-1, keepdim=True) + 1e-6)
        xt = x_true.float()
        xt = xt - xt.mean(dim=-1, keepdim=True)
        xt = xt / (xt.std(dim=-1, keepdim=True) + 1e-6)
        corr_p = torch.matmul(xp, xp.transpose(1, 2)) / x_pred.shape[-1]
        corr_t = torch.matmul(xt, xt.transpose(1, 2)) / x_true.shape[-1]
        return F.l1_loss(corr_p, corr_t)

    def _compute_diff_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, weight_tensor=None) -> torch.Tensor:
        dy_pred = y_pred[:, :, 1:] - y_pred[:, :, :-1]
        dy_true = y_true[:, :, 1:] - y_true[:, :, :-1]
        diff = torch.abs(dy_pred - dy_true)
        if weight_tensor is not None:
            diff = diff * weight_tensor
        return diff.mean()

    def _compute_weighted_recon_loss(self, pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        l1 = (torch.abs(pred - target) * weights).mean()
        mse = (((pred - target) ** 2) * weights).mean()
        return l1 + mse

    def _compute_progression_loss(self, y_pred_full: torch.Tensor, y_true_full: torch.Tensor) -> torch.Tensor:
        chest = [6, 7, 8, 9, 10, 11]
        pred_prog = y_pred_full[:, chest, :].abs().amax(dim=2)
        true_prog = y_true_full[:, chest, :].abs().amax(dim=2)
        return F.l1_loss(pred_prog, true_prog)

    def _compute_algebra_consistency_loss(self, y_full: torch.Tensor) -> torch.Tensor:
        lead_i = y_full[:, 0, :]
        lead_ii = y_full[:, 1, :]
        loss = 0.0
        loss = loss + F.l1_loss(y_full[:, 2, :], lead_ii - lead_i)
        loss = loss + F.l1_loss(y_full[:, 3, :], -(lead_i + lead_ii) / 2.0)
        loss = loss + F.l1_loss(y_full[:, 4, :], lead_i - lead_ii / 2.0)
        loss = loss + F.l1_loss(y_full[:, 5, :], lead_ii - lead_i / 2.0)
        return loss / 4.0

    def _compute_fm_perceptual_loss(self, y_pred_full: torch.Tensor, y_true_full: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z_true = self._extract_fm_tokens(y_true_full)
        z_pred = self._extract_fm_tokens(y_pred_full)
        return F.mse_loss(z_pred.float(), z_true.float()) + 0.25 * _mean_cosine_distance(z_pred, z_true)

    def _compute_fm_latent_alignment_loss(self, z_latent: torch.Tensor, e_fm_full: torch.Tensor) -> torch.Tensor:
        z_proj = self.latent_to_fm(z_latent.transpose(1, 2))
        return F.mse_loss(z_proj.float(), e_fm_full.detach().float()) + 0.25 * _mean_cosine_distance(z_proj, e_fm_full.detach())

    def decode_independent(self, x_cond: torch.Tensor, z_latent: torch.Tensor, e_fm: torch.Tensor) -> torch.Tensor:
        base_pred = self.decoder(x_cond, z_latent, e_fm)
        if self.obs_to_miss is not None:
            skip_pred = self.obs_to_miss(x_cond)
            return base_pred + self.get_skip_scale() * skip_pred
        return base_pred

    def stage1_forward(self, x: torch.Tensor, lead_indices: Optional[torch.Tensor] = None):
        device = x.device
        x_cond, x_target, z_teacher, e_fm_full, e_fm_sparse, obs_indices, _ = self.extract_condition(x, lead_indices)

        if self.alignment_head is not None:
            z_regressed = self.alignment_head(x_cond, e_fm_sparse)
            loss_align = F.mse_loss(z_regressed.float(), z_teacher.detach().float())
            loss_align = loss_align + 0.25 * _mean_cosine_distance(
                z_regressed.transpose(1, 2),
                z_teacher.detach().transpose(1, 2),
            )
            loss_align = loss_align + 0.10 * self._compute_fm_latent_alignment_loss(z_regressed, e_fm_full)
        else:
            z_regressed = None
            loss_align = torch.tensor(0.0, device=device)

        pred_teacher_independent = self.decode_independent(x_cond, z_teacher.detach(), e_fm_full.detach())
        y_pred_teacher_full = self._assemble_from_independent_prediction(x, pred_teacher_independent, obs_indices)
        pred_teacher_missing, target_missing, target_weights, _ = self._slice_canonical_missing(
            pred_teacher_independent, x_target, obs_indices
        )
        loss_decoder_teacher = self._compute_weighted_recon_loss(pred_teacher_missing, target_missing, target_weights)

        loss_decoder_reg = torch.tensor(0.0, device=device)
        loss_stft_reg = torch.tensor(0.0, device=device)
        loss_diff_reg = torch.tensor(0.0, device=device)
        loss_corr_reg = torch.tensor(0.0, device=device)
        loss_progress_reg = torch.tensor(0.0, device=device)
        loss_fm_perc_reg = torch.tensor(0.0, device=device)
        loss_alg_reg = torch.tensor(0.0, device=device)

        if z_regressed is not None:
            pred_reg_independent = self.decode_independent(x_cond, z_regressed, e_fm_sparse)
            y_pred_reg_full = self._assemble_from_independent_prediction(x, pred_reg_independent, obs_indices)
            pred_reg_missing, target_missing, target_weights, _ = self._slice_canonical_missing(
                pred_reg_independent, x_target, obs_indices
            )
            loss_decoder_reg = self._compute_weighted_recon_loss(pred_reg_missing, target_missing, target_weights)

            if self.use_engineering_losses:
                lead_weights = target_weights.squeeze(0).squeeze(-1)
                loss_stft_reg = self.stft_loss_fn(pred_reg_missing, target_missing, lead_weights=lead_weights)
                loss_diff_reg = self._compute_diff_loss(pred_reg_missing, target_missing, target_weights)
                loss_corr_reg = self._compute_corr_loss(pred_reg_missing, target_missing)
                loss_progress_reg = self._compute_progression_loss(y_pred_reg_full, x_target)
                loss_alg_reg = self._compute_algebra_consistency_loss(y_pred_reg_full)
                if self.use_fm_perceptual:
                    loss_fm_perc_reg = self._compute_fm_perceptual_loss(y_pred_reg_full, x_target)

            total_reg_loss = loss_decoder_reg
            total_reg_loss = total_reg_loss + 0.25 * loss_diff_reg
            total_reg_loss = total_reg_loss + 0.10 * loss_stft_reg
            total_reg_loss = total_reg_loss + 0.20 * loss_corr_reg
            total_reg_loss = total_reg_loss + 0.15 * loss_progress_reg
            total_reg_loss = total_reg_loss + 0.01 * loss_alg_reg
            total_reg_loss = total_reg_loss + self.fm_perceptual_weight * loss_fm_perc_reg
        else:
            y_pred_reg_full = None
            total_reg_loss = torch.tensor(0.0, device=device)

        if self.finetune_mode == "strict" or self.recon_finetune:
            total_loss = self.reg_loss_weight * total_reg_loss + self.align_loss_weight * loss_align
        else:
            total_loss = (
                self.teacher_loss_weight * loss_decoder_teacher
                + self.reg_loss_weight * total_reg_loss
                + self.align_loss_weight * loss_align
            )

        y_pred = y_pred_reg_full if y_pred_reg_full is not None else y_pred_teacher_full
        return {
            "loss": total_loss,
            "decoder_loss": loss_decoder_reg,
            "teacher_loss": loss_decoder_teacher,
            "align_loss": loss_align,
            "stft_loss": loss_stft_reg,
            "diff_loss": loss_diff_reg,
            "corr_loss": loss_corr_reg,
            "progress_loss": loss_progress_reg,
            "fm_perceptual_loss": loss_fm_perc_reg,
            "alg_loss": loss_alg_reg,
            "y_target": x_target,
            "y_pred_teacher": y_pred_teacher_full,
            "y_pred_reg": y_pred_reg_full,
            "y_pred": y_pred,
            "z_teacher": z_teacher.detach(),
            "z_regressed": z_regressed.detach() if z_regressed is not None else None,
        }

    @torch.no_grad()
    def impute_from_teacher(self, x: torch.Tensor, lead_indices: Optional[torch.Tensor] = None):
        x_cond, _, z_teacher, e_fm_full, _, obs_indices, _ = self.extract_condition(x, lead_indices)
        pred_teacher_independent = self.decode_independent(x_cond, z_teacher, e_fm_full)
        y_pred = self._assemble_from_independent_prediction(x, pred_teacher_independent, obs_indices)
        return {"y_pred": y_pred, "z_clean": z_teacher}

    @torch.no_grad()
    def impute_from_regressor(self, x: torch.Tensor, lead_indices: Optional[torch.Tensor] = None):
        if self.alignment_head is None:
            return {"available": False, "y_pred": None, "z_latent": None}
        x_cond, _, _, _, e_fm_sparse, obs_indices, _ = self.extract_condition(x, lead_indices)
        z_regressed = self.alignment_head(x_cond, e_fm_sparse)
        pred_independent = self.decode_independent(x_cond, z_regressed, e_fm_sparse)
        y_pred = self._assemble_from_independent_prediction(x, pred_independent, obs_indices)
        return {"available": True, "y_pred": y_pred, "z_latent": z_regressed}

    def forward(self, x: torch.Tensor, lead_indices: Optional[torch.Tensor] = None, mode: str = "stage1", **kwargs):
        if mode != "stage1":
            raise ValueError("Engineering track supports only mode='stage1'.")
        return self.stage1_forward(x, lead_indices=lead_indices)
