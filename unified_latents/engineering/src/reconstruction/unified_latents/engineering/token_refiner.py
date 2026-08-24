"""Sequential token-teacher refiner for frozen WearECG VAE reconstructions."""

from __future__ import annotations

import json
import os
import sys
sys.path.insert(0, '/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/unified_latents/engineering')
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from src.reconstruction.unified_latents.engineering.common import mask_unobserved_leads
from src.reconstruction.unified_latents.engineering.vae_fm import (
    ECGFMFeatureExtractor,
    LATENT_SCALE,
    WearECGVAE,
    weighted_reconstruction_mse,
)


DEFAULT_ECGFM_CKPT = "/home/mithunmanivannan/ecg_fm_integration/checkpoints/mimic_iv_ecg_physionet_pretrained.pt"
DEFAULT_HUBERT_MODEL = "Edoardo-BS/HuBERT-ECG-SSL-Pretrained"
HUBERT_REPO_CODE = "/home/mithunmanivannan/ecg_fm_integration/hubert_ecg_repo/code"
RAW_TEACHER_TOKEN_LENGTHS = {
    "ecgfm": 312,
    "hubert": 936,
    "random_ecgfm": 312,
    "random_hubert": 936,
    "random_ecgfm_arch": 312,
    "random_hubert_arch": 936,
}


class TokenTeacher(nn.Module):
    """Frozen ECG token extractor contract."""

    embed_dim: int
    num_token_layers: int = 1

    def extract_tokens(self, x_12: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def extract_token_layers(self, x_12: torch.Tensor) -> list[torch.Tensor]:
        return [self.extract_tokens(x_12)]


def _resolve_layer_indices(num_layers: int, layer_mode: str) -> list[int]:
    if num_layers < 1:
        return [0]
    mode = str(layer_mode or "last").strip().lower()
    if mode in {"last", "final"}:
        return [num_layers - 1]
    if mode in {"mid,last", "middle,last", "mid_final", "mid+last"}:
        return sorted({max(num_layers // 2 - 1, 0), num_layers - 1})
    if mode in {"early,mid,last", "early_mid_last", "early+mid+last"}:
        return sorted({max(num_layers // 4 - 1, 0), max(num_layers // 2 - 1, 0), num_layers - 1})
    indices: list[int] = []
    for part in mode.split(","):
        part = part.strip()
        if not part:
            continue
        idx = int(part)
        if idx < 0:
            idx = num_layers + idx
        if idx < 0 or idx >= num_layers:
            raise ValueError(f"Layer index {part!r} is out of range for {num_layers} layers")
        indices.append(idx)
    if not indices:
        return [num_layers - 1]
    return sorted(dict.fromkeys(indices))


class ECGFMTokenTeacher(TokenTeacher):
    """Continuous ECG-FM token teacher."""

    def __init__(self, checkpoint_path: str = DEFAULT_ECGFM_CKPT, embed_dim: int = 768, layer_mode: str = "last"):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.extractor = ECGFMFeatureExtractor(
            checkpoint_path=checkpoint_path,
            embed_dim=self.embed_dim,
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
        layers = getattr(getattr(self.extractor.backbone, "encoder", None), "layers", [])
        self.layer_indices = _resolve_layer_indices(len(layers), layer_mode)
        self.num_token_layers = len(self.layer_indices)
        self._freeze()

    def _freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def train(self, mode: bool = True):
        super().train(False)
        return self

    def extract_tokens(self, x_12: torch.Tensor) -> torch.Tensor:
        return self.extractor.extract_tokens(x_12)

    def extract_token_layers(self, x_12: torch.Tensor) -> list[torch.Tensor]:
        layers = getattr(getattr(self.extractor.backbone, "encoder", None), "layers", None)
        if not layers or self.num_token_layers <= 1:
            return [self.extract_tokens(x_12)]

        captured: dict[int, torch.Tensor] = {}
        hooks = []

        def _make_hook(layer_idx: int):
            def _hook(_module, _inputs, output):
                value = output[0] if isinstance(output, tuple) else output
                if isinstance(value, torch.Tensor):
                    captured[layer_idx] = value.detach()
            return _hook

        for idx in self.layer_indices:
            hooks.append(layers[idx].register_forward_hook(_make_hook(idx)))
        try:
            final_tokens = self.extract_tokens(x_12)
        finally:
            for hook in hooks:
                hook.remove()

        token_layers: list[torch.Tensor] = []
        for idx in self.layer_indices:
            tokens = captured.get(idx)
            if tokens is None:
                token_layers.append(final_tokens)
                continue
            if tokens.dim() == 3 and tokens.shape[0] != final_tokens.shape[0]:
                tokens = tokens.transpose(0, 1)
            tokens = self.extractor.token_norm(tokens.float()).to(x_12.dtype)
            token_layers.append(tokens)
        return token_layers


class ECGFMRandomArchTokenTeacher(ECGFMTokenTeacher):
    """Frozen random-weight ECG-FM architecture diagnostic control."""

    def __init__(self, checkpoint_path: str = DEFAULT_ECGFM_CKPT, embed_dim: int = 768, layer_mode: str = "last", seed: int = 1234):
        nn.Module.__init__(self)
        generator_state = torch.random.get_rng_state()
        torch.manual_seed(int(seed))
        self.embed_dim = int(embed_dim)
        from fairseq_signals.models.ecg_transformer import ECGTransformerModel
        from fairseq_signals.utils.checkpoint_utils import load_checkpoint_to_cpu
        from omegaconf import OmegaConf

        state = load_checkpoint_to_cpu(checkpoint_path)
        cfg = state["cfg"]["model"]
        OmegaConf.set_struct(cfg, False)
        if getattr(cfg, "saliency", None) is None:
            cfg.saliency = False
        self.extractor = nn.Module()
        self.extractor.backbone = ECGTransformerModel.build_model(cfg)
        self.extractor.token_norm = nn.LayerNorm(self.embed_dim)
        self.extractor._full12_zscore = ECGFMFeatureExtractor._full12_zscore
        torch.random.set_rng_state(generator_state)
        layers = getattr(getattr(self.extractor.backbone, "encoder", None), "layers", [])
        self.layer_indices = _resolve_layer_indices(len(layers), layer_mode)
        self.num_token_layers = len(self.layer_indices)
        self._freeze()

    def extract_tokens(self, x_12: torch.Tensor) -> torch.Tensor:
        self.extractor.backbone.eval()
        if x_12.shape[1] != 12 and x_12.shape[2] == 12:
            x_12 = x_12.transpose(1, 2)
        elif x_12.shape[1] != 12:
            raise ValueError(f"Expected ECG tensor shaped [B, 12, T] or [B, T, 12], got {tuple(x_12.shape)}")
        x_norm = ECGFMFeatureExtractor._full12_zscore(x_12).to(torch.float32)
        with torch.amp.autocast(x_norm.device.type, enabled=False):
            res = self.extractor.backbone.extract_features(x_norm, None)
            tokens = res["x"] if isinstance(res, dict) else res
            tokens = self.extractor.token_norm(tokens)
        return tokens.to(x_12.dtype)


class HuBERTECGTokenTeacher(TokenTeacher):
    """Continuous hidden-state HuBERT-ECG teacher.

    HuBERT-ECG flattens 12 aligned leads into a single lead-major sequence and
    was trained on 5-second windows. For 10-second PTB-XL tensors we extract two
    non-overlapping windows and concatenate their continuous transformer states.
    """

    def __init__(
        self,
        checkpoint_path: str = DEFAULT_HUBERT_MODEL,
        *,
        repo_code_path: str = HUBERT_REPO_CODE,
        downsampling_factor: Optional[int] = None,
        layer_mode: str = "last",
        load_weights: bool = True,
        random_seed: int = 1234,
    ):
        super().__init__()
        if repo_code_path and repo_code_path not in sys.path:
            sys.path.insert(0, repo_code_path)
        from hubert_ecg import HuBERTECG, HuBERTECGConfig
        from transformers import HubertConfig
        import __main__

        resolved_checkpoint = checkpoint_path
        if not os.path.exists(str(resolved_checkpoint)):
            from huggingface_hub import hf_hub_download

            resolved_checkpoint = hf_hub_download(
                repo_id=checkpoint_path,
                filename="hubert_ecg_base.pt",
            )
        if not hasattr(__main__, "HuBERTECGConfig"):
            setattr(__main__, "HuBERTECGConfig", HuBERTECGConfig)
        checkpoint = torch.load(resolved_checkpoint, map_location="cpu", weights_only=False)
        config = checkpoint.get("model_config")
        vocab_sizes = checkpoint.get("pretraining_vocab_sizes", [100])
        if isinstance(config, HubertConfig) and not isinstance(config, HuBERTECGConfig):
            config = HuBERTECGConfig(
                ensemble_length=len(vocab_sizes),
                vocab_sizes=vocab_sizes,
                **config.to_dict(),
            )
        elif isinstance(config, dict):
            config = HuBERTECGConfig(
                ensemble_length=len(vocab_sizes),
                vocab_sizes=vocab_sizes,
                **config,
            )
        current_defaults = HubertConfig()
        for key, value in current_defaults.to_dict().items():
            if not hasattr(config, key):
                setattr(config, key, value)
        if not hasattr(config, "_experts_implementation_internal"):
            config._experts_implementation_internal = None
        generator_state = torch.random.get_rng_state()
        torch.manual_seed(int(random_seed))
        self.model = HuBERTECG(config)
        torch.random.set_rng_state(generator_state)
        if load_weights:
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.embed_dim = int(self.model.config.hidden_size)
        self.downsampling_factor = downsampling_factor
        self.layer_mode = layer_mode
        num_layers = int(getattr(self.model.config, "num_hidden_layers", 0))
        self.layer_indices = _resolve_layer_indices(num_layers + 1, layer_mode)
        self.num_token_layers = len(self.layer_indices)

    def train(self, mode: bool = True):
        super().train(False)
        self.model.eval()
        return self

    @staticmethod
    def _window_to_hubert_input(x_window: torch.Tensor) -> torch.Tensor:
        return x_window.reshape(x_window.size(0), -1)

    def _maybe_downsample_flat(self, flat: torch.Tensor) -> torch.Tensor:
        if self.downsampling_factor is None:
            return flat
        factor = int(self.downsampling_factor)
        if factor <= 1:
            return flat
        # Differentiable approximation to the repo's scipy decimation for
        # refiner training. The flattened lead-major sequence is the HuBERT
        # input domain, so average pooling is applied there.
        pad = (-flat.shape[-1]) % factor
        if pad:
            flat = F.pad(flat, (0, pad))
        flat = F.avg_pool1d(flat.unsqueeze(1), kernel_size=factor, stride=factor).squeeze(1)
        return flat

    def extract_tokens(self, x_12: torch.Tensor) -> torch.Tensor:
        return self.extract_token_layers(x_12)[-1]

    def extract_token_layers(self, x_12: torch.Tensor) -> list[torch.Tensor]:
        if x_12.shape[1] != 12 and x_12.shape[2] == 12:
            x_12 = x_12.transpose(1, 2)
        if x_12.shape[1] != 12:
            raise ValueError(f"Expected [B,12,T] ECG tensor, got {tuple(x_12.shape)}")
        if x_12.shape[-1] < 5000:
            x_12 = F.pad(x_12, (0, 5000 - x_12.shape[-1]))
        x_12 = x_12[:, :, :5000]
        windows = (x_12[:, :, :2500], x_12[:, :, 2500:5000])
        token_parts_by_layer: list[list[torch.Tensor]] = [[] for _ in self.layer_indices]
        for window in windows:
            flat = self._window_to_hubert_input(window)
            flat = self._maybe_downsample_flat(flat)
            attention_mask = torch.ones_like(flat, dtype=torch.long)
            out = self.model(
                input_values=flat,
                attention_mask=attention_mask,
                output_hidden_states=self.num_token_layers > 1,
                return_dict=True,
            )
            hidden_states = getattr(out, "hidden_states", None)
            if hidden_states is None:
                hidden_states = (out.last_hidden_state,)
                local_indices = [0]
            else:
                local_indices = self.layer_indices
            for out_idx, layer_idx in enumerate(local_indices):
                source_idx = min(layer_idx, len(hidden_states) - 1)
                token_parts_by_layer[out_idx].append(hidden_states[source_idx])
        return [torch.cat(parts, dim=1) for parts in token_parts_by_layer]


class RandomTokenTeacher(TokenTeacher):
    """Frozen random convolutional token teacher matched by shape."""

    def __init__(
        self,
        *,
        embed_dim: int = 768,
        token_length: int = 625,
        seed: int = 1234,
        num_token_layers: int = 1,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.token_length = int(token_length)
        self.num_token_layers = int(num_token_layers)
        generator_state = torch.random.get_rng_state()
        torch.manual_seed(int(seed))
        self.net = nn.Sequential(
            nn.Conv1d(12, 128, kernel_size=17, padding=8),
            nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=9, padding=4),
            nn.GELU(),
            nn.Conv1d(256, self.embed_dim, kernel_size=1),
        )
        self.extra_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(12, 128, kernel_size=17, padding=8),
                    nn.GELU(),
                    nn.Conv1d(128, 256, kernel_size=9, padding=4),
                    nn.GELU(),
                    nn.Conv1d(256, self.embed_dim, kernel_size=1),
                )
                for _ in range(max(self.num_token_layers - 1, 0))
            ]
        )
        torch.random.set_rng_state(generator_state)
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def train(self, mode: bool = True):
        super().train(False)
        return self

    def extract_tokens(self, x_12: torch.Tensor) -> torch.Tensor:
        return self._extract_with_net(self.net, x_12)

    def _extract_with_net(self, net: nn.Module, x_12: torch.Tensor) -> torch.Tensor:
        if x_12.shape[1] != 12 and x_12.shape[2] == 12:
            x_12 = x_12.transpose(1, 2)
        feat = net(x_12.float())
        feat = F.adaptive_avg_pool1d(feat, self.token_length)
        return feat.transpose(1, 2).to(x_12.dtype)

    def extract_token_layers(self, x_12: torch.Tensor) -> list[torch.Tensor]:
        return [self._extract_with_net(self.net, x_12)] + [
            self._extract_with_net(net, x_12) for net in self.extra_nets
        ]


def resolve_teacher_token_length(teacher_encoder: str, teacher_token_length: Optional[int] = None) -> int:
    """Resolve raw teacher token length from the paired teacher type."""
    if teacher_token_length is not None:
        token_length = int(teacher_token_length)
        if token_length < 1:
            raise ValueError(f"teacher_token_length must be >= 1, got {teacher_token_length}")
        return token_length
    if teacher_encoder not in RAW_TEACHER_TOKEN_LENGTHS:
        raise ValueError(f"Unknown teacher_encoder: {teacher_encoder}")
    return RAW_TEACHER_TOKEN_LENGTHS[teacher_encoder]


def build_token_teacher(
    teacher_encoder: str,
    *,
    teacher_checkpoint: Optional[str] = None,
    teacher_dim: int = 768,
    teacher_token_length: Optional[int] = None,
    random_seed: int = 1234,
    teacher_layer_mode: str = "last",
) -> TokenTeacher:
    teacher_token_length = resolve_teacher_token_length(teacher_encoder, teacher_token_length)
    if teacher_encoder == "ecgfm":
        return ECGFMTokenTeacher(
            checkpoint_path=teacher_checkpoint or DEFAULT_ECGFM_CKPT,
            embed_dim=teacher_dim,
            layer_mode=teacher_layer_mode,
        )
    if teacher_encoder == "hubert":
        return HuBERTECGTokenTeacher(
            checkpoint_path=teacher_checkpoint or DEFAULT_HUBERT_MODEL,
            layer_mode=teacher_layer_mode,
        )
    if teacher_encoder == "random_ecgfm_arch":
        return ECGFMRandomArchTokenTeacher(
            checkpoint_path=teacher_checkpoint or DEFAULT_ECGFM_CKPT,
            embed_dim=teacher_dim,
            layer_mode=teacher_layer_mode,
            seed=random_seed,
        )
    if teacher_encoder == "random_hubert_arch":
        return HuBERTECGTokenTeacher(
            checkpoint_path=teacher_checkpoint or DEFAULT_HUBERT_MODEL,
            layer_mode=teacher_layer_mode,
            load_weights=False,
            random_seed=random_seed,
        )
    if teacher_encoder in {"random_ecgfm", "random_hubert"}:
        num_layers = len(_resolve_layer_indices(12, teacher_layer_mode))
        return RandomTokenTeacher(
            embed_dim=teacher_dim,
            token_length=teacher_token_length,
            seed=random_seed,
            num_token_layers=num_layers,
        )
    raise ValueError(f"Unknown teacher_encoder: {teacher_encoder}")


def align_token_length(tokens: torch.Tensor, target_len: Optional[int]) -> torch.Tensor:
    """Linearly align token sequences to a common temporal length."""
    if target_len is None or int(target_len) <= 0 or tokens.shape[1] == int(target_len):
        return tokens
    target_len = int(target_len)
    return F.interpolate(
        tokens.float().transpose(1, 2),
        size=target_len,
        mode="linear",
        align_corners=False,
    ).transpose(1, 2).to(tokens.dtype)


def compute_token_loss(
    pred_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    mix: float = 0.5,
    *,
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Scale-normalized FM token loss for fair ECG-FM/HuBERT comparison."""
    if pred_tokens.shape != target_tokens.shape:
        raise ValueError(f"Token shapes must match, got {tuple(pred_tokens.shape)} and {tuple(target_tokens.shape)}")
    mix = float(mix)
    if not 0.0 <= mix <= 1.0:
        raise ValueError(f"token loss mix must be in [0, 1], got {mix}")
    pred_tokens = pred_tokens.float()
    target_tokens = target_tokens.float()
    if mean is not None and std is not None:
        pred_n = (pred_tokens - mean) / std.clamp(min=1e-6)
        target_n = (target_tokens - mean) / std.clamp(min=1e-6)
    else:
        target_std = target_tokens.std(dim=(0, 1), keepdim=True).clamp(min=1e-6)
        pred_n = pred_tokens / target_std
        target_n = target_tokens / target_std
    mse_term = F.mse_loss(pred_n, target_n)
    cosine_term = 1.0 - F.cosine_similarity(
        pred_n.flatten(0, 1),
        target_n.flatten(0, 1),
        dim=-1,
    ).mean()
    return mix * cosine_term + (1.0 - mix) * mse_term


def _load_frozen_vae(checkpoint_path: Optional[str], *, target_len: int, beta_kl: float, missing_lead_weight: float) -> WearECGVAE:
    vae = WearECGVAE(target_len=target_len, beta_kl=beta_kl, missing_lead_weight=missing_lead_weight)
    if checkpoint_path:
        payload = torch.load(checkpoint_path, map_location="cpu")
        if "encoder_state_dict" in payload and "decoder_state_dict" in payload:
            encoder_state = _remap_legacy_encoder_state(payload["encoder_state_dict"])
            vae.encoder.load_state_dict(encoder_state, strict=True)
            vae.decoder.load_state_dict(payload["decoder_state_dict"], strict=True)
        else:
            state = payload.get("model_state_dict", payload)
            state = _remap_legacy_full_vae_state(state)
            vae.load_state_dict(state, strict=True)
    for param in vae.parameters():
        param.requires_grad = False
    vae.eval()
    return vae


def _remap_legacy_encoder_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map legacy exact-WearECG encoder projection keys to the current wrapper."""
    if "blocks.17.weight" not in state:
        return state
    remapped = dict(state)
    pairs = {
        "blocks.17.weight": "output_head.0.weight",
        "blocks.17.bias": "output_head.0.bias",
        "blocks.18.weight": "output_head.1.weight",
        "blocks.18.bias": "output_head.1.bias",
    }
    for old_key, new_key in pairs.items():
        if old_key in remapped:
            remapped[new_key] = remapped.pop(old_key)
    return remapped


def _remap_legacy_full_vae_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if "encoder.blocks.17.weight" not in state:
        return state
    remapped = dict(state)
    pairs = {
        "encoder.blocks.17.weight": "encoder.output_head.0.weight",
        "encoder.blocks.17.bias": "encoder.output_head.0.bias",
        "encoder.blocks.18.weight": "encoder.output_head.1.weight",
        "encoder.blocks.18.bias": "encoder.output_head.1.bias",
    }
    for old_key, new_key in pairs.items():
        if old_key in remapped:
            remapped[new_key] = remapped.pop(old_key)
    return remapped


class BottleneckAttentionBlock(nn.Module):
    """AliTok-style latent-token block with cross-attention into fixed memories."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.norm_q_coarse = nn.LayerNorm(dim)
        self.norm_m_coarse = nn.LayerNorm(dim)
        self.coarse_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_q_teacher = nn.LayerNorm(dim)
        self.norm_m_teacher = nn.LayerNorm(dim)
        self.teacher_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_self = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, latent: torch.Tensor, coarse_memory: torch.Tensor, teacher_memory: torch.Tensor) -> torch.Tensor:
        q = self.norm_q_coarse(latent)
        m = self.norm_m_coarse(coarse_memory)
        attn_out, _ = self.coarse_attn(q, m, m, need_weights=False)
        latent = latent + attn_out

        q = self.norm_q_teacher(latent)
        m = self.norm_m_teacher(teacher_memory)
        attn_out, _ = self.teacher_attn(q, m, m, need_weights=False)
        latent = latent + attn_out

        q = self.norm_self(latent)
        attn_out, _ = self.self_attn(q, q, q, need_weights=False)
        latent = latent + attn_out
        latent = latent + self.ff(self.norm_ff(latent).float()).to(latent.dtype)
        return latent


class AliTokBottleneckResidualRefiner(nn.Module):
    """Residual decoder that routes teacher/coarse evidence through learned latent tokens.

    This follows the useful AliTok pattern for our setting: learned latent
    tokens form the bottleneck, absorb evidence from signal/teacher tokens, and
    the decoder reconstructs from those latents. It deliberately avoids a
    direct time-aligned teacher-to-waveform shortcut.
    """

    def __init__(
        self,
        *,
        teacher_dim: int,
        refiner_dim: int = 256,
        query_len: int = 256,
        target_len: int = 5000,
        num_heads: int = 8,
        num_token_layers: int = 1,
        num_blocks: int = 2,
    ):
        super().__init__()
        self.query_len = int(query_len)
        self.target_len = int(target_len)
        self.num_token_layers = int(num_token_layers)
        self.coarse_proj = nn.Conv1d(12, refiner_dim, kernel_size=9, padding=4)
        if self.num_token_layers <= 1:
            self.teacher_proj = nn.Linear(teacher_dim, refiner_dim)
        else:
            self.teacher_projs = nn.ModuleList([nn.Linear(teacher_dim, refiner_dim) for _ in range(self.num_token_layers)])
            self.teacher_layer_logits = nn.Parameter(torch.zeros(self.num_token_layers))

        scale = refiner_dim ** -0.5
        self.latent_tokens = nn.Parameter(scale * torch.randn(self.query_len, refiner_dim))
        self.latent_pos = nn.Parameter(scale * torch.randn(self.query_len, refiner_dim))
        self.coarse_pos = nn.Parameter(scale * torch.randn(self.query_len, refiner_dim))
        self.teacher_pos = nn.Parameter(scale * torch.randn(self.query_len, refiner_dim))
        self.blocks = nn.ModuleList([BottleneckAttentionBlock(refiner_dim, num_heads) for _ in range(num_blocks)])
        self.decoder_norm = nn.LayerNorm(refiner_dim)
        self.decoder_blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=refiner_dim,
                    nhead=num_heads,
                    dim_feedforward=4 * refiner_dim,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(max(1, num_blocks // 2))
            ]
        )
        self.to_delta = nn.Sequential(
            nn.Conv1d(refiner_dim, refiner_dim, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(refiner_dim, 12, kernel_size=1),
        )
        nn.init.zeros_(self.to_delta[-1].weight)
        nn.init.zeros_(self.to_delta[-1].bias)

    def _project_teacher_memory(self, tokens: torch.Tensor | list[torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
        if not isinstance(tokens, (list, tuple)):
            return self.teacher_proj(tokens.float()).to(dtype)
        if self.num_token_layers <= 1:
            return self.teacher_proj(tokens[-1].float()).to(dtype)
        weights = F.softmax(self.teacher_layer_logits.float(), dim=0)
        memories = [
            proj(layer_tokens.float()).to(dtype) * weights[idx].to(dtype)
            for idx, (proj, layer_tokens) in enumerate(zip(self.teacher_projs, tokens))
        ]
        return torch.stack(memories, dim=0).sum(dim=0)

    def _pos(self, pos: torch.Tensor, seq_len: int, dtype: torch.dtype) -> torch.Tensor:
        if seq_len == pos.shape[0]:
            return pos.to(dtype).unsqueeze(0)
        return F.interpolate(pos.T.unsqueeze(0), size=seq_len, mode="linear", align_corners=False).transpose(1, 2).to(dtype)

    def _latent_from_memories(self, x_coarse: torch.Tensor, tokens: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        coarse_memory = F.adaptive_avg_pool1d(self.coarse_proj(x_coarse), self.query_len).transpose(1, 2)
        coarse_memory = coarse_memory + self.coarse_pos.to(coarse_memory.dtype).unsqueeze(0)
        teacher_memory = self._project_teacher_memory(tokens, coarse_memory.dtype)
        teacher_memory = teacher_memory + self._pos(self.teacher_pos, teacher_memory.shape[1], teacher_memory.dtype)
        latent = self.latent_tokens.to(coarse_memory.dtype).unsqueeze(0).expand(x_coarse.shape[0], -1, -1)
        latent = latent + self.latent_pos.to(coarse_memory.dtype).unsqueeze(0)
        for block in self.blocks:
            latent = block(latent, coarse_memory, teacher_memory)
        return latent

    def _decode_bidir_latent(self, latent: torch.Tensor) -> torch.Tensor:
        for block in self.decoder_blocks:
            latent = block(latent)
        latent = self.decoder_norm(latent).transpose(1, 2)
        latent = F.interpolate(latent, size=self.target_len, mode="linear", align_corners=False)
        return self.to_delta(latent)

    def forward(self, x_coarse: torch.Tensor, tokens: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        return self._decode_bidir_latent(self._latent_from_memories(x_coarse, tokens))


class AliTokCausalBottleneckResidualRefiner(AliTokBottleneckResidualRefiner):
    """AliTok-inspired refiner with an auxiliary causal decoder over latent tokens.

    The bidirectional decoder remains the inference path. The causal branch
    regularizes the learned latent sequence so each temporal token is useful
    from prefix plus previous-context information, mirroring AliTok's tokenizer
    alignment idea without replacing the working residual refiner.
    """

    def __init__(
        self,
        *,
        teacher_dim: int,
        refiner_dim: int = 256,
        query_len: int = 256,
        target_len: int = 5000,
        num_heads: int = 8,
        num_token_layers: int = 1,
        num_blocks: int = 2,
        prefix_tokens: int = 16,
        stage: str = "causal_align",
    ):
        super().__init__(
            teacher_dim=teacher_dim,
            refiner_dim=refiner_dim,
            query_len=query_len,
            target_len=target_len,
            num_heads=num_heads,
            num_token_layers=num_token_layers,
            num_blocks=num_blocks,
        )
        if prefix_tokens < 1:
            raise ValueError(f"prefix_tokens must be >= 1, got {prefix_tokens}")
        if stage not in {"causal_align", "bidir_refine"}:
            raise ValueError(f"Unknown token_refiner_stage={stage!r}")
        self.prefix_tokens = int(prefix_tokens)
        self.stage = str(stage)

        scale = refiner_dim ** -0.5
        self.causal_prefix = nn.Parameter(scale * torch.randn(self.prefix_tokens, refiner_dim))
        self.causal_pos = nn.Parameter(scale * torch.randn(self.prefix_tokens + self.query_len, refiner_dim))
        self.causal_blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=refiner_dim,
                    nhead=num_heads,
                    dim_feedforward=4 * refiner_dim,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(max(1, num_blocks // 2))
            ]
        )
        self.causal_norm = nn.LayerNorm(refiner_dim)
        self.causal_to_delta = nn.Sequential(
            nn.Conv1d(refiner_dim, refiner_dim, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(refiner_dim, 12, kernel_size=1),
        )
        self.prefix_to_delta = nn.Sequential(
            nn.Conv1d(refiner_dim, refiner_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(refiner_dim, 12, kernel_size=1),
        )
        nn.init.zeros_(self.causal_to_delta[-1].weight)
        nn.init.zeros_(self.causal_to_delta[-1].bias)
        nn.init.zeros_(self.prefix_to_delta[-1].weight)
        nn.init.zeros_(self.prefix_to_delta[-1].bias)
        if self.stage == "bidir_refine":
            self.freeze_causal_alignment_backbone()

    def freeze_causal_alignment_backbone(self) -> None:
        """Freeze the alignment machinery for stage-B bidirectional refinement."""
        for module in (self.coarse_proj, self.blocks, self.causal_blocks, self.causal_norm, self.causal_to_delta, self.prefix_to_delta):
            for param in module.parameters():
                param.requires_grad = False
        for param in (
            self.latent_tokens,
            self.latent_pos,
            self.coarse_pos,
            self.teacher_pos,
            self.causal_prefix,
            self.causal_pos,
        ):
            param.requires_grad = False
        for attr in ("teacher_proj", "teacher_projs", "teacher_layer_logits"):
            value = getattr(self, attr, None)
            if value is None:
                continue
            if isinstance(value, nn.Parameter):
                value.requires_grad = False
            else:
                for param in value.parameters():
                    param.requires_grad = False

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    def _decode_causal_latent(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = latent.shape[0]
        prefix = self.causal_prefix.to(latent.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        seq = torch.cat([prefix, latent], dim=1)
        seq = seq + self.causal_pos.to(seq.dtype).unsqueeze(0)
        attn_mask = self._causal_mask(seq.shape[1], seq.device)
        for block in self.causal_blocks:
            seq = block(seq, src_mask=attn_mask)
        seq = self.causal_norm(seq)
        prefix_hidden = seq[:, : self.prefix_tokens]
        latent_hidden = seq[:, self.prefix_tokens :]

        causal = latent_hidden.transpose(1, 2)
        causal = F.interpolate(causal, size=self.target_len, mode="linear", align_corners=False)
        causal_delta = self.causal_to_delta(causal)

        prefix_len = min(self.target_len, max(self.prefix_tokens, round(self.target_len * self.prefix_tokens / max(self.query_len, 1))))
        prefix_delta = self.prefix_to_delta(prefix_hidden.transpose(1, 2))
        prefix_delta = F.interpolate(prefix_delta, size=prefix_len, mode="linear", align_corners=False)
        return causal_delta, prefix_delta

    def forward_with_aux(self, x_coarse: torch.Tensor, tokens: torch.Tensor | list[torch.Tensor]) -> dict[str, torch.Tensor]:
        latent = self._latent_from_memories(x_coarse, tokens)
        bidir_delta = self._decode_bidir_latent(latent)
        causal_delta, prefix_delta = self._decode_causal_latent(latent)
        return {
            "delta": bidir_delta,
            "causal_delta": causal_delta,
            "prefix_delta": prefix_delta,
        }

    def forward(self, x_coarse: torch.Tensor, tokens: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        return self.forward_with_aux(x_coarse, tokens)["delta"]


class WearECGTokenRefiner(nn.Module):
    """Frozen WearECG VAE + token-conditioned residual correction."""

    def __init__(
        self,
        *,
        frozen_vae_checkpoint: Optional[str],
        teacher_encoder: str,
        teacher_checkpoint: Optional[str] = None,
        target_len: int = 5000,
        beta_kl: float = 1e-4,
        missing_lead_weight: float = 3.0,
        token_loss_weight: float = 0.05,
        token_loss_mix: float = 0.5,
        residual_smoothness_weight: float = 1e-4,
        teacher_dim: int = 768,
        teacher_token_length: Optional[int] = None,
        teacher_common_token_length: int = 625,
        refiner_dim: int = 256,
        query_len: int = 625,
        random_seed: int = 1234,
        use_observed_conditioning: bool = False,
        clamp_observed_output: bool = False,
        token_improvement_margin_weight: float = 0.0,
        token_improvement_margin: float = 0.0,
        teacher_layer_mode: str = "last",
        token_whitening: bool = False,
        causal_alignment: bool = False,
        prefix_tokens: int = 16,
        causal_loss_weight: float = 0.05,
        prefix_aux_loss_weight: float = 0.1,
        refiner_stage: str = "causal_align",
    ):
        super().__init__()
        self.target_len = int(target_len)
        self.beta_kl = float(beta_kl)
        self.missing_lead_weight = float(missing_lead_weight)
        self.token_loss_weight = float(token_loss_weight)
        self.token_loss_mix = float(token_loss_mix)
        self.residual_smoothness_weight = float(residual_smoothness_weight)
        self.teacher_encoder = str(teacher_encoder)
        self.teacher_checkpoint = teacher_checkpoint
        self.frozen_vae_checkpoint = frozen_vae_checkpoint
        self.teacher_token_length = resolve_teacher_token_length(self.teacher_encoder, teacher_token_length)
        self.teacher_common_token_length = int(teacher_common_token_length)
        self.teacher_dim = int(teacher_dim)
        self.use_observed_conditioning = bool(use_observed_conditioning)
        self.clamp_observed_output = bool(clamp_observed_output)
        self.token_improvement_margin_weight = float(token_improvement_margin_weight)
        self.token_improvement_margin = float(token_improvement_margin)
        self.teacher_layer_mode = str(teacher_layer_mode)
        self.token_whitening = bool(token_whitening)
        self.causal_alignment = bool(causal_alignment)
        self.prefix_tokens = int(prefix_tokens)
        self.causal_loss_weight = float(causal_loss_weight)
        self.prefix_aux_loss_weight = float(prefix_aux_loss_weight)
        self.refiner_stage = str(refiner_stage)
        if self.refiner_stage not in {"causal_align", "bidir_refine"}:
            raise ValueError(f"Unknown token_refiner_stage={self.refiner_stage!r}")

        self.frozen_vae = _load_frozen_vae(
            frozen_vae_checkpoint,
            target_len=target_len,
            beta_kl=beta_kl,
            missing_lead_weight=missing_lead_weight,
        )
        self.teacher = build_token_teacher(
            teacher_encoder,
            teacher_checkpoint=teacher_checkpoint,
            teacher_dim=teacher_dim,
            teacher_token_length=self.teacher_token_length,
            random_seed=random_seed,
            teacher_layer_mode=self.teacher_layer_mode,
        )
        teacher_dim_actual = int(getattr(self.teacher, "embed_dim", teacher_dim))
        self.teacher_dim = teacher_dim_actual
        self.num_token_layers = int(getattr(self.teacher, "num_token_layers", 1))
        if self.token_whitening:
            self.register_buffer("token_whiten_mean", torch.zeros(self.num_token_layers, teacher_dim_actual))
            self.register_buffer("token_whiten_std", torch.ones(self.num_token_layers, teacher_dim_actual))
        refiner_kwargs = {
            "teacher_dim": teacher_dim_actual,
            "refiner_dim": refiner_dim,
            "query_len": query_len,
            "target_len": target_len,
            "num_token_layers": self.num_token_layers,
        }
        if self.causal_alignment:
            self.refiner = AliTokCausalBottleneckResidualRefiner(
                **refiner_kwargs,
                prefix_tokens=self.prefix_tokens,
                stage=self.refiner_stage,
            )
        else:
            self.refiner = AliTokBottleneckResidualRefiner(**refiner_kwargs)

    def train(self, mode: bool = True):
        super().train(mode)
        self.frozen_vae.eval()
        self.teacher.eval()
        return self

    def _match_target_len(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] < self.target_len:
            x = F.pad(x, (0, self.target_len - x.shape[-1]))
        elif x.shape[-1] > self.target_len:
            x = x[..., : self.target_len]
        return x

    @staticmethod
    def _replace_observed_leads(base: torch.Tensor, observed_source: torch.Tensor, lead_indices) -> torch.Tensor:
        if lead_indices is None:
            return base
        if lead_indices.dim() != 2:
            raise ValueError(f"Expected lead_indices shape [B,N_obs], got {tuple(lead_indices.shape)}")
        out = base.clone()
        idx = lead_indices.long().to(device=base.device)
        gather_idx = idx.unsqueeze(-1).expand(-1, -1, base.shape[-1])
        observed_values = observed_source.gather(1, gather_idx)
        out.scatter_(1, gather_idx, observed_values)
        return out

    def set_token_whitening_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if not self.token_whitening:
            self.token_whitening = True
            self.register_buffer("token_whiten_mean", mean.detach().float().clone())
            self.register_buffer("token_whiten_std", std.detach().float().clamp(min=1e-6).clone())
            return
        if self.token_whiten_mean.shape != mean.shape:
            raise ValueError(f"Token whitening mean shape mismatch: {tuple(mean.shape)} vs {tuple(self.token_whiten_mean.shape)}")
        self.token_whiten_mean.copy_(mean.detach().float())
        self.token_whiten_std.copy_(std.detach().float().clamp(min=1e-6))

    @torch.no_grad()
    def _coarse_reconstruction(self, x: torch.Tensor, lead_indices=None) -> dict[str, torch.Tensor]:
        out = self.frozen_vae.impute_from_regressor(x, lead_indices=lead_indices)
        if not out.get("available", True):
            raise RuntimeError("Frozen VAE coarse reconstruction is unavailable.")
        y_pred = self._match_target_len(out["y_pred"].float())
        return {**out, "y_pred": y_pred}

    def _teacher_layers(self, x_12: torch.Tensor) -> list[torch.Tensor]:
        layers = self.teacher.extract_token_layers(x_12)
        return [align_token_length(tokens, self.teacher_common_token_length) for tokens in layers]

    def _refine(
        self,
        x_coarse: torch.Tensor,
        *,
        observed_input: Optional[torch.Tensor] = None,
        lead_indices=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        x_cond = x_coarse
        if self.use_observed_conditioning and observed_input is not None:
            x_cond = self._replace_observed_leads(x_coarse, observed_input, lead_indices)
        with torch.no_grad():
            cond_tokens = self._teacher_layers(x_cond)
        aux: dict[str, torch.Tensor] = {}
        if self.causal_alignment and hasattr(self.refiner, "forward_with_aux"):
            refiner_out = self.refiner.forward_with_aux(x_coarse, cond_tokens)
            delta = refiner_out["delta"]
            aux = {
                "causal_delta": refiner_out["causal_delta"],
                "prefix_delta": refiner_out["prefix_delta"],
            }
        else:
            delta = self.refiner(x_coarse, cond_tokens)
        x_final = x_coarse + delta
        if self.clamp_observed_output and observed_input is not None:
            x_final = self._replace_observed_leads(x_final, observed_input, lead_indices)
        return self._match_target_len(x_final), self._match_target_len(delta), self._match_target_len(x_cond), aux

    def _token_layer_loss(self, pred_layers: list[torch.Tensor], target_layers: list[torch.Tensor]) -> torch.Tensor:
        losses = []
        for layer_idx, (pred_tokens, target_tokens) in enumerate(zip(pred_layers, target_layers)):
            mean = std = None
            if self.token_whitening and hasattr(self, "token_whiten_mean"):
                mean = self.token_whiten_mean[layer_idx].view(1, 1, -1).to(pred_tokens.device)
                std = self.token_whiten_std[layer_idx].view(1, 1, -1).to(pred_tokens.device)
            losses.append(compute_token_loss(pred_tokens, target_tokens, mix=self.token_loss_mix, mean=mean, std=std))
        if not losses:
            return pred_layers[-1].new_tensor(0.0)
        return torch.stack(losses).mean()

    def _token_loss(self, x_final: torch.Tensor, target_12: torch.Tensor, x_coarse: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.token_loss_weight <= 0:
            zero = x_final.new_tensor(0.0)
            return zero, zero, zero
        pred_layers = self._teacher_layers(x_final)
        with torch.no_grad():
            true_layers = self._teacher_layers(target_12)
        final_loss = self._token_layer_loss(pred_layers, true_layers)
        if x_coarse is None or self.token_improvement_margin_weight <= 0:
            zero = x_final.new_tensor(0.0)
            return final_loss, zero, zero
        with torch.no_grad():
            coarse_layers = self._teacher_layers(x_coarse)
            coarse_loss = self._token_layer_loss(coarse_layers, true_layers)
        margin_loss = F.relu(final_loss - coarse_loss + self.token_improvement_margin)
        return final_loss, margin_loss, coarse_loss.detach()

    @staticmethod
    def _smoothness_loss(delta: torch.Tensor) -> torch.Tensor:
        if delta.shape[-1] < 2:
            return delta.new_tensor(0.0)
        return delta.diff(dim=-1).abs().mean()

    def _causal_alignment_losses(
        self,
        *,
        x_coarse: torch.Tensor,
        target_12: torch.Tensor,
        aux: dict[str, torch.Tensor],
        lead_indices=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.causal_alignment or self.refiner_stage != "causal_align":
            zero = target_12.new_tensor(0.0)
            return zero, zero
        causal_delta = aux.get("causal_delta")
        prefix_delta = aux.get("prefix_delta")
        causal_loss = target_12.new_tensor(0.0)
        prefix_loss = target_12.new_tensor(0.0)
        if causal_delta is not None and self.causal_loss_weight > 0:
            causal_pred = self._match_target_len(x_coarse + causal_delta)
            causal_loss = weighted_reconstruction_mse(
                causal_pred.transpose(1, 2),
                target_12.transpose(1, 2),
                lead_indices=lead_indices,
                missing_lead_weight=self.missing_lead_weight,
            )
        if prefix_delta is not None and self.prefix_aux_loss_weight > 0 and prefix_delta.shape[-1] > 0:
            prefix_len = min(prefix_delta.shape[-1], target_12.shape[-1], x_coarse.shape[-1])
            prefix_pred = x_coarse[..., :prefix_len] + prefix_delta[..., :prefix_len]
            prefix_loss = weighted_reconstruction_mse(
                prefix_pred.transpose(1, 2),
                target_12[..., :prefix_len].transpose(1, 2),
                lead_indices=lead_indices,
                missing_lead_weight=self.missing_lead_weight,
            )
        return causal_loss, prefix_loss

    def stage1_forward(
        self,
        x: torch.Tensor,
        y_full: Optional[torch.Tensor] = None,
        lead_indices=None,
    ) -> dict[str, torch.Tensor]:
        target_12 = self._match_target_len(y_full if y_full is not None else x)
        coarse_out = self._coarse_reconstruction(x, lead_indices=lead_indices)
        x_coarse = coarse_out["y_pred"]
        x_final, delta, x_cond, aux = self._refine(x_coarse, observed_input=x, lead_indices=lead_indices)
        x_token_final = x_final
        x_token_coarse = x_coarse
        if self.clamp_observed_output:
            x_token_final = self._replace_observed_leads(x_token_final, x, lead_indices)
            x_token_coarse = self._replace_observed_leads(x_token_coarse, x, lead_indices)
        recon_loss = weighted_reconstruction_mse(
            x_final.transpose(1, 2),
            target_12.transpose(1, 2),
            lead_indices=lead_indices,
            missing_lead_weight=self.missing_lead_weight,
        )
        token_loss, token_margin_loss, coarse_token_loss = self._token_loss(x_token_final, target_12, x_token_coarse)
        smooth_loss = self._smoothness_loss(delta)
        causal_loss, prefix_aux_loss = self._causal_alignment_losses(
            x_coarse=x_coarse,
            target_12=target_12,
            aux=aux,
            lead_indices=lead_indices,
        )
        total_loss = (
            recon_loss
            + self.token_loss_weight * token_loss
            + self.token_improvement_margin_weight * token_margin_loss
            + self.residual_smoothness_weight * smooth_loss
            + self.causal_loss_weight * causal_loss
            + self.prefix_aux_loss_weight * prefix_aux_loss
        )
        zero = x.new_tensor(0.0)
        return {
            "loss": total_loss,
            "decoder_loss": recon_loss.detach(),
            "teacher_loss": token_loss.detach(),
            "align_loss": zero,
            "stft_loss": zero,
            "diff_loss": smooth_loss.detach(),
            "corr_loss": zero,
            "kl_loss": zero,
            "fm_perceptual_loss": token_loss.detach(),
            "token_margin_loss": token_margin_loss.detach(),
            "coarse_token_loss": coarse_token_loss.detach(),
            "causal_alignment_loss": causal_loss.detach(),
            "prefix_aux_loss": prefix_aux_loss.detach(),
            "latent_align_loss": zero,
            "multi_scale_align_loss": zero,
            "y_target": target_12,
            "y_pred": x_final,
            "y_pred_reg": x_final,
            "y_coarse": x_coarse.detach(),
            "x_teacher_cond": x_cond.detach(),
            "delta": delta.detach(),
            "z_regressed": coarse_out.get("z_latent", zero).detach() if isinstance(coarse_out.get("z_latent"), torch.Tensor) else None,
        }

    @torch.no_grad()
    def impute_from_regressor(self, x: torch.Tensor, lead_indices=None) -> dict[str, torch.Tensor]:
        coarse_out = self._coarse_reconstruction(x, lead_indices=lead_indices)
        x_final, delta, x_cond, _aux = self._refine(coarse_out["y_pred"], observed_input=x, lead_indices=lead_indices)
        return {
            "available": True,
            "y_pred": x_final,
            "y_coarse": coarse_out["y_pred"],
            "x_teacher_cond": x_cond,
            "delta": delta,
            "z_latent": coarse_out.get("z_latent"),
            "log_var": coarse_out.get("log_var"),
        }

    def forward(self, x: torch.Tensor, lead_indices=None, mode: str = "stage1", **kwargs):
        if mode != "stage1":
            raise ValueError("WearECGTokenRefiner supports only mode='stage1'.")
        if x.dim() != 3 or x.shape[1] != 12:
            raise ValueError(f"Expected x shape [B,12,T], got {tuple(x.shape)}")
        return self.stage1_forward(x, y_full=kwargs.get("y_full"), lead_indices=lead_indices)


def probe_token_teacher_shapes(
    sample: torch.Tensor,
    *,
    output_path: str,
    ecgfm_checkpoint: str = DEFAULT_ECGFM_CKPT,
    hubert_checkpoint: str = DEFAULT_HUBERT_MODEL,
    teacher_common_token_length: int = 625,
    frozen_vae_checkpoint: Optional[str] = None,
    pathology_pair: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
) -> dict[str, object]:
    """Record Phase 0 teacher shape and variance checks."""
    device = sample.device
    report: dict[str, object] = {}
    real_token_shapes: dict[str, tuple[int, int]] = {}
    for name, teacher in [
        ("ecgfm", ECGFMTokenTeacher(ecgfm_checkpoint)),
        ("hubert", HuBERTECGTokenTeacher(hubert_checkpoint)),
    ]:
        teacher = teacher.to(device)
        with torch.no_grad():
            tokens = teacher.extract_tokens(sample)
            aligned = align_token_length(tokens, teacher_common_token_length)
        real_token_shapes[name] = (int(tokens.shape[1]), int(tokens.shape[2]))
        report[name] = {
            "shape": list(tokens.shape),
            "token_length": int(tokens.shape[1]),
            "embed_dim": int(tokens.shape[2]),
            "aligned_shape": list(aligned.shape),
            "aligned_token_length": int(aligned.shape[1]),
            "param_count": int(sum(p.numel() for p in teacher.parameters())),
            "finite": bool(torch.isfinite(tokens).all().item()),
            "mean_abs": float(tokens.float().abs().mean().item()),
            "std": float(tokens.float().std().item()),
            "per_feature_std_mean": float(tokens.float().std(dim=(0, 1)).mean().item()),
        }
        if name == "hubert":
            report[name]["variance_check"] = "pass" if report[name]["std"] > 0.1 else "fail"
        if name == "ecgfm" and pathology_pair is not None:
            sr_sample, mi_sample = pathology_pair
            sr_sample = sr_sample.to(device=device, dtype=sample.dtype)
            mi_sample = mi_sample.to(device=device, dtype=sample.dtype)
            with torch.no_grad():
                sr_tokens = align_token_length(teacher.extract_tokens(sr_sample), teacher_common_token_length)
                mi_tokens = align_token_length(teacher.extract_tokens(mi_sample), teacher_common_token_length)
                cosine = F.cosine_similarity(
                    sr_tokens.mean(dim=1).float(),
                    mi_tokens.mean(dim=1).float(),
                    dim=-1,
                ).mean()
            report[name]["sinus_vs_mi_cosine"] = float(cosine.item())
            report[name]["pathology_cosine_check"] = "pass" if float(cosine.item()) < 0.98 else "fail"
        del teacher
    for random_name, real_name in [("random_ecgfm", "ecgfm"), ("random_hubert", "hubert")]:
        token_len, embed_dim = real_token_shapes[real_name]
        teacher = RandomTokenTeacher(embed_dim=embed_dim, token_length=token_len).to(device)
        with torch.no_grad():
            tokens = teacher.extract_tokens(sample)
            aligned = align_token_length(tokens, teacher_common_token_length)
        expected_shape = [sample.shape[0], token_len, embed_dim]
        report[random_name] = {
            "shape": list(tokens.shape),
            "expected_shape": expected_shape,
            "shape_check": "pass" if list(tokens.shape) == expected_shape else "fail",
            "token_length": int(tokens.shape[1]),
            "embed_dim": int(tokens.shape[2]),
            "aligned_shape": list(aligned.shape),
            "param_count": int(sum(p.numel() for p in teacher.parameters())),
            "finite": bool(torch.isfinite(tokens).all().item()),
            "mean_abs": float(tokens.float().abs().mean().item()),
            "std": float(tokens.float().std().item()),
        }
        del teacher
    report["common_alignment"] = {
        "teacher_common_token_length": int(teacher_common_token_length),
        "ecgfm_raw_token_length": real_token_shapes["ecgfm"][0],
        "hubert_raw_token_length": real_token_shapes["hubert"][0],
        "raw_lengths_match": bool(real_token_shapes["ecgfm"][0] == real_token_shapes["hubert"][0]),
    }
    if frozen_vae_checkpoint:
        vae = _load_frozen_vae(frozen_vae_checkpoint, target_len=sample.shape[-1], beta_kl=1e-4, missing_lead_weight=3.0)
        trainable = int(sum(p.numel() for p in vae.parameters() if p.requires_grad))
        report["frozen_vae"] = {
            "checkpoint": frozen_vae_checkpoint,
            "trainable_parameter_count": trainable,
            "freeze_check": "pass" if trainable == 0 else "fail",
        }
        del vae
    if pathology_pair is None:
        report["pathology_probe_status"] = "fallback_unlabeled"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report
