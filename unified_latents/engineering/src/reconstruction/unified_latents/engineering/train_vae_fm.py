"""Primary consolidated trainer for the exact WearECG and FM-VAE models."""

from __future__ import annotations

import argparse
import math
import os
import sys
sys.path.insert(0, '/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/unified_latents/engineering')
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

sys.path.append(os.getcwd())

from src.reconstruction.unified_latents.engineering.common import (
    AugmentedTensorDataset,
    CombinedTensorFolderDataset,
    TensorFolderDataset,
    cleanup_partial_checkpoints,
    mask_unobserved_leads,
    prune_epoch_checkpoints,
    write_best_summary,
    write_json,
    write_run_artifacts,
    write_warm_start_summary,
)
from src.reconstruction.unified_latents.engineering.eval_reconstruction import evaluate_reconstruction
from src.reconstruction.unified_latents.engineering.regimes import (
    LEAD_NAMES,
    format_lead_set,
    get_missing_indices,
    make_lead_indices,
    resolve_obs_leads,
)
from src.reconstruction.unified_latents.engineering.vae_fm import WearECGVAE, WearECGFMVAE
from src.reconstruction.unified_latents.engineering.token_refiner import (
    DEFAULT_HUBERT_MODEL,
    WearECGTokenRefiner,
    resolve_teacher_token_length,
)
try:
    from src.reconstruction.unified_latents.engineering.alitok_vae_exp import build_alitok_vae_1d
except ModuleNotFoundError:
    build_alitok_vae_1d = None


DEFAULT_FM_CKPT = "/home/mithunmanivannan/ecg_fm_integration/checkpoints/mimic_iv_ecg_physionet_pretrained.pt"
_WANDB_WARNED_PREFIXES: set[str] = set()


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool) -> None:
    dest = name.replace("-", "_")
    flag_names = {f"--{name}"}
    if "_" in name:
        flag_names.add(f"--{name.replace('_', '-')}")
    parser.add_argument(*sorted(flag_names), dest=dest, nargs="?", const=True, type=str_to_bool)
    negative_flag_names = {f"--no-{name}"}
    if "_" in name:
        negative_flag_names.add(f"--no-{name.replace('_', '-')}")
    parser.add_argument(*sorted(negative_flag_names), dest=dest, nargs="?", const=False, type=str_to_bool)
    parser.set_defaults(**{dest: default})


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--model_family",
        type=str,
        choices=["baseline", "wearecg", "fm_vae", "token_refiner", "alitok", "alitok_stage2", "alitok_hybrid"],
        default="fm_vae",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_lr", "--max-lr", dest="max_lr", type=float, default=5e-5)
    parser.add_argument("--target_len", "--target-len", dest="target_len", type=int, default=5000)
    parser.add_argument("--latent_channels", "--latent-channels", dest="latent_channels", type=int, default=4)
    parser.add_argument("--beta_kl", "--beta-kl", dest="beta_kl", type=float, default=1e-4)
    parser.add_argument(
        "--beta_kl_schedule",
        "--beta-kl-schedule",
        dest="beta_kl_schedule",
        type=str,
        choices=["constant", "linear_warmup"],
        default="constant",
    )
    parser.add_argument("--beta_kl_start", "--beta-kl-start", dest="beta_kl_start", type=float, default=None)
    parser.add_argument("--beta_kl_end", "--beta-kl-end", dest="beta_kl_end", type=float, default=None)
    parser.add_argument(
        "--beta_kl_warmup_steps",
        "--beta-kl-warmup-steps",
        dest="beta_kl_warmup_steps",
        type=int,
        default=0,
    )
    parser.add_argument("--missing_lead_weight", "--missing-lead-weight", dest="missing_lead_weight", type=float, default=1.0)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--regime", type=str, choices=["current", "wearecg", "historical"], default="wearecg")
    parser.add_argument("--obs_leads", type=str, default=None)
    parser.add_argument("--run_tag", "--run-tag", dest="run_tag", type=str, default="")
    parser.add_argument("--split", type=str, choices=["val", "test"], default="val")
    parser.add_argument("--train_splits", "--train-splits", dest="train_splits", type=str, default="train")
    parser.add_argument("--eval_split", "--eval-split", dest="eval_split", type=str, choices=["val", "test"], default=None)
    add_bool_arg(parser, "train_augmentation", False)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_logs", action="store_true")
    parser.add_argument("--save_training_state", action="store_true")
    parser.add_argument("--train_num_workers", "--train-num-workers", dest="train_num_workers", type=int, default=8)
    parser.add_argument("--val_num_workers", "--val-num-workers", dest="val_num_workers", type=int, default=4)
    parser.add_argument(
        "--keep_epoch_checkpoints",
        "--keep-epoch-checkpoints",
        dest="keep_epoch_checkpoints",
        type=int,
        default=2,
    )
    add_bool_arg(parser, "full_val_reconstruction", False)
    parser.add_argument("--fm_checkpoint", "--fm-checkpoint", dest="fm_checkpoint", type=str, default=DEFAULT_FM_CKPT)
    parser.add_argument("--frozen_vae_checkpoint", "--frozen-vae-checkpoint", dest="frozen_vae_checkpoint", type=str, default=None)
    parser.add_argument(
        "--teacher_encoder",
        "--teacher-encoder",
        dest="teacher_encoder",
        type=str,
        choices=["ecgfm", "hubert", "random_ecgfm", "random_hubert", "random_ecgfm_arch", "random_hubert_arch"],
        default="ecgfm",
    )
    parser.add_argument("--teacher_checkpoint", "--teacher-checkpoint", dest="teacher_checkpoint", type=str, default=None)
    parser.add_argument("--teacher_dim", "--teacher-dim", dest="teacher_dim", type=int, default=768)
    parser.add_argument("--teacher_token_length", "--teacher-token-length", dest="teacher_token_length", type=int, default=None)
    parser.add_argument("--token_loss_weight", "--token-loss-weight", dest="token_loss_weight", type=float, default=0.05)
    parser.add_argument("--token_loss_mix", "--token-loss-mix", dest="token_loss_mix", type=float, default=0.5)
    parser.add_argument(
        "--teacher_common_token_length",
        "--teacher-common-token-length",
        dest="teacher_common_token_length",
        type=int,
        default=625,
    )
    parser.add_argument("--residual_smoothness_weight", "--residual-smoothness-weight", dest="residual_smoothness_weight", type=float, default=1e-4)
    parser.add_argument("--refiner_dim", "--refiner-dim", dest="refiner_dim", type=int, default=256)
    parser.add_argument("--refiner_query_len", "--refiner-query-len", dest="refiner_query_len", type=int, default=625)
    parser.add_argument("--random_teacher_seed", "--random-teacher-seed", dest="random_teacher_seed", type=int, default=1234)
    add_bool_arg(parser, "token_refiner_v2", False)
    add_bool_arg(parser, "token_refiner_observed_conditioning", False)
    add_bool_arg(parser, "token_refiner_clamp_observed", False)
    add_bool_arg(parser, "token_refiner_causal_alignment", False)
    parser.add_argument("--token_refiner_prefix_tokens", "--token-refiner-prefix-tokens", dest="token_refiner_prefix_tokens", type=int, default=16)
    parser.add_argument("--token_refiner_causal_loss_weight", "--token-refiner-causal-loss-weight", dest="token_refiner_causal_loss_weight", type=float, default=0.05)
    parser.add_argument("--token_refiner_prefix_aux_loss_weight", "--token-refiner-prefix-aux-loss-weight", dest="token_refiner_prefix_aux_loss_weight", type=float, default=0.1)
    parser.add_argument(
        "--token_refiner_stage",
        "--token-refiner-stage",
        dest="token_refiner_stage",
        choices=["causal_align", "bidir_refine"],
        default="causal_align",
    )
    add_bool_arg(parser, "token_whitening", False)
    parser.add_argument("--token_whitening_batches", "--token-whitening-batches", dest="token_whitening_batches", type=int, default=64)
    parser.add_argument("--teacher_layer_mode", "--teacher-layer-mode", dest="teacher_layer_mode", type=str, default="last")
    parser.add_argument("--token_improvement_margin_weight", "--token-improvement-margin-weight", dest="token_improvement_margin_weight", type=float, default=0.0)
    parser.add_argument("--token_improvement_margin", "--token-improvement-margin", dest="token_improvement_margin", type=float, default=0.0)
    parser.add_argument(
        "--token_loss_schedule",
        "--token-loss-schedule",
        dest="token_loss_schedule",
        choices=["constant", "warmup_ramp"],
        default="constant",
    )
    parser.add_argument("--token_loss_warmup_epochs", "--token-loss-warmup-epochs", dest="token_loss_warmup_epochs", type=int, default=3)
    parser.add_argument("--token_loss_ramp_epochs", "--token-loss-ramp-epochs", dest="token_loss_ramp_epochs", type=int, default=17)
    parser.add_argument("--token_loss_target_weight", "--token-loss-target-weight", dest="token_loss_target_weight", type=float, default=0.1)
    parser.add_argument("--fm_loss_weight", "--fm-loss-weight", dest="fm_loss_weight", type=float, default=1e-2)
    parser.add_argument("--fm_cosine_mix", "--fm-cosine-mix", dest="fm_cosine_mix", type=float, default=0.5)
    parser.add_argument("--fm_cond_drop_prob", "--fm-cond-drop-prob", dest="fm_cond_drop_prob", type=float, default=0.0)
    parser.add_argument("--latent_align_weight", "--latent-align-weight", dest="latent_align_weight", type=float, default=1e-3)
    parser.add_argument("--multi_scale_align_weight", "--multi-scale-align-weight", dest="multi_scale_align_weight", type=float, default=1e-1)
    parser.add_argument("--global_latent_channels", "--global-latent-channels", dest="global_latent_channels", type=int, default=2)
    parser.add_argument("--local_latent_channels", "--local-latent-channels", dest="local_latent_channels", type=int, default=2)
    parser.add_argument("--alitok_patch_size", "--alitok-patch-size", dest="alitok_patch_size", type=int, default=10)
    parser.add_argument("--alitok_token_size", "--alitok-token-size", dest="alitok_token_size", type=int, default=32)
    parser.add_argument("--alitok_stage2_mix", "--alitok-stage2-mix", dest="alitok_stage2_mix", type=float, default=0.35)
    parser.add_argument("--alitok_prefix_tokens", "--alitok-prefix-tokens", dest="alitok_prefix_tokens", type=int, default=17)
    parser.add_argument("--alitok_codebook_size", "--alitok-codebook-size", dest="alitok_codebook_size", type=int, default=4096)
    parser.add_argument("--alitok_encoder_depth", "--alitok-encoder-depth", dest="alitok_encoder_depth", type=int, default=12)
    parser.add_argument("--alitok_decoder_depth", "--alitok-decoder-depth", dest="alitok_decoder_depth", type=int, default=24)
    parser.add_argument("--alitok_encoder_width", "--alitok-encoder-width", dest="alitok_encoder_width", type=int, default=768)
    parser.add_argument("--alitok_decoder_width", "--alitok-decoder-width", dest="alitok_decoder_width", type=int, default=1024)
    parser.add_argument("--alitok_encoder_heads", "--alitok-encoder-heads", dest="alitok_encoder_heads", type=int, default=12)
    parser.add_argument("--alitok_decoder_heads", "--alitok-decoder-heads", dest="alitok_decoder_heads", type=int, default=16)
    parser.add_argument("--alitok_heads", "--alitok-heads", dest="alitok_heads", type=int, default=None)
    parser.add_argument("--alitok_stage2_buffer_tokens", "--alitok-stage2-buffer-tokens", dest="alitok_stage2_buffer_tokens", type=int, default=32)
    add_bool_arg(parser, "alitok_clustering_vq", True)
    add_bool_arg(parser, "fm_perceptual", True)
    add_bool_arg(parser, "fm_decoder_conditioning", False)
    add_bool_arg(parser, "fm_latent_align", False)
    add_bool_arg(parser, "fm_multi_scale_align", False)
    add_bool_arg(parser, "mask_aware_encoder", True)
    add_bool_arg(parser, "split_latent", True)
    add_bool_arg(parser, "freeze_vae", True)
    args = parser.parse_args()
    if args.model_family == "wearecg":
        args.model_family = "baseline"
        args.model_family_tag = "wearecg"
    else:
        args.model_family_tag = args.model_family if args.model_family != "baseline" else "baseline"
    if args.eval_split is not None:
        args.split = args.eval_split
    if args.keep_epoch_checkpoints < 1:
        raise ValueError("keep_epoch_checkpoints must be >= 1")
    args.obs_lead_indices = resolve_obs_leads(args.regime, args.obs_leads)
    if args.model_family in {"baseline", "fm_vae"} and args.latent_channels != 4:
        raise ValueError("WearECG baseline and FM-VAE preserve latent_channels=4.")
    if args.model_family != "fm_vae":
        # Keep non-FM runs deterministic and avoid accidental FM flag leakage.
        args.fm_perceptual = False
        args.fm_decoder_conditioning = False
        args.fm_latent_align = False
        args.fm_multi_scale_align = False
        args.fm_loss_weight = 0.0
        args.fm_cosine_mix = 0.0
        args.fm_cond_drop_prob = 0.0
        args.latent_align_weight = 0.0
        args.multi_scale_align_weight = 0.0
    if args.model_family == "token_refiner":
        if not args.frozen_vae_checkpoint:
            raise ValueError("--frozen_vae_checkpoint is required for --model_family token_refiner")
        if not args.freeze_vae:
            raise ValueError("token_refiner requires --freeze_vae true; sequential Phase 2 keeps the VAE frozen.")
        if args.teacher_common_token_length < 1:
            raise ValueError("--teacher_common_token_length must be >= 1")
        if not 0.0 <= args.token_loss_mix <= 1.0:
            raise ValueError("--token_loss_mix must be in [0, 1]")
        if args.token_refiner_prefix_tokens < 1:
            raise ValueError("--token_refiner_prefix_tokens must be >= 1")
        if args.token_refiner_causal_loss_weight < 0:
            raise ValueError("--token_refiner_causal_loss_weight must be >= 0")
        if args.token_refiner_prefix_aux_loss_weight < 0:
            raise ValueError("--token_refiner_prefix_aux_loss_weight must be >= 0")
        if args.token_refiner_stage == "bidir_refine" and not args.token_refiner_causal_alignment:
            raise ValueError("--token_refiner_stage bidir_refine requires --token_refiner_causal_alignment true")
        args.teacher_token_length = resolve_teacher_token_length(args.teacher_encoder, args.teacher_token_length)
        if args.token_refiner_v2:
            args.token_refiner_observed_conditioning = True
            args.token_refiner_clamp_observed = True
            args.token_whitening = True
            if args.refiner_query_len == 625:
                args.refiner_query_len = 256
            if args.teacher_layer_mode == "last":
                args.teacher_layer_mode = "mid,last"
            if args.token_loss_schedule == "constant":
                args.token_loss_schedule = "warmup_ramp"
            args.token_loss_target_weight = max(float(args.token_loss_target_weight), 0.1)

    # KL schedule normalization.
    if args.beta_kl_schedule == "constant":
        if args.beta_kl_start is None:
            args.beta_kl_start = float(args.beta_kl)
        if args.beta_kl_end is None:
            args.beta_kl_end = float(args.beta_kl)
        args.beta_kl_warmup_steps = 0
    else:
        if args.beta_kl_start is None:
            args.beta_kl_start = 0.0
        if args.beta_kl_end is None:
            args.beta_kl_end = float(args.beta_kl)
        if args.beta_kl_warmup_steps <= 0:
            raise ValueError("beta_kl_warmup_steps must be > 0 when beta_kl_schedule=linear_warmup")

    # Keep beta_kl equal to the schedule end value for backward compatibility.
    args.beta_kl = float(args.beta_kl_end)
    return args


def get_baseline_selector(metrics: dict[str, float]) -> tuple[float, float, float]:
    return (
        -metrics.get("val/mse_reg", float("inf")),
        -metrics.get("val/mae_reg", float("inf")),
        -metrics.get("val/rmse_reg", float("inf")),
    )


def current_beta_kl(args: argparse.Namespace, step: int) -> float:
    if args.beta_kl_schedule == "constant":
        return float(args.beta_kl)
    warmup_steps = max(int(args.beta_kl_warmup_steps), 1)
    progress = min(max(int(step), 0), warmup_steps) / warmup_steps
    start = float(args.beta_kl_start)
    end = float(args.beta_kl_end)
    return start + (end - start) * progress


def safe_wandb_log(payload: dict[str, float], *, step: int | None = None, prefix: str = "") -> bool:
    if wandb.run is None:
        return False
    try:
        wandb.log(payload, step=step)
        return True
    except Exception as exc:
        warn_key = prefix or "default"
        if warn_key not in _WANDB_WARNED_PREFIXES:
            print(f"[W&B] log failed during {warn_key}; continuing ({exc})")
            _WANDB_WARNED_PREFIXES.add(warn_key)
        return False


def _count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _named_metric(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key, float("nan"))
    if isinstance(value, torch.Tensor):
        return float(value.item())
    return float(value)


def current_token_loss_weight(args: argparse.Namespace, epoch_idx: int) -> float:
    if args.model_family != "token_refiner" or args.token_loss_schedule == "constant":
        return float(args.token_loss_weight)
    warmup = max(int(args.token_loss_warmup_epochs), 0)
    ramp = max(int(args.token_loss_ramp_epochs), 1)
    target = float(args.token_loss_target_weight)
    if epoch_idx < warmup:
        return 0.0
    progress = min(max((epoch_idx - warmup + 1) / ramp, 0.0), 1.0)
    return target * progress


@torch.no_grad()
def compute_token_whitening_stats(
    model: torch.nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if args.model_family != "token_refiner" or not getattr(args, "token_whitening", False):
        return None
    if not hasattr(model, "_teacher_layers"):
        return None

    model.eval()
    sums = None
    sumsq = None
    count = 0
    max_batches = int(args.token_whitening_batches)
    for batch_idx, (_x, y, _meta) in enumerate(tqdm(train_loader, desc="Token-whiten-stats")):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        y = y.to(device, dtype=torch.float32, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            layers = model._teacher_layers(y)
        layer_means = []
        layer_sqs = []
        for tokens in layers:
            tokens_f = tokens.float()
            layer_means.append(tokens_f.sum(dim=(0, 1)))
            layer_sqs.append((tokens_f * tokens_f).sum(dim=(0, 1)))
        batch_count = int(layers[0].shape[0] * layers[0].shape[1])
        batch_sum = torch.stack(layer_means)
        batch_sumsq = torch.stack(layer_sqs)
        sums = batch_sum if sums is None else sums + batch_sum
        sumsq = batch_sumsq if sumsq is None else sumsq + batch_sumsq
        count += batch_count
    if sums is None or sumsq is None or count <= 0:
        return None
    mean = sums / count
    var = (sumsq / count - mean * mean).clamp(min=1e-6)
    std = torch.sqrt(var)
    model.set_token_whitening_stats(mean, std)
    return mean.detach().cpu(), std.detach().cpu()


def _print_warm_start_summary(summary: dict[str, Any]) -> None:
    print(
        "[WarmStart] "
        f"type={summary['initialization_type']} "
        f"loaded={summary['loaded_tensor_count']} "
        f"skipped={summary['skipped_tensor_count']} "
        f"loaded_prefixes={summary['loaded_prefixes'][:8]} "
        f"skipped_prefixes={summary['skipped_prefixes_sample'][:8]}"
    )


def _print_debug_banner(
    args: argparse.Namespace,
    device: torch.device,
    train_dataset,
    val_dataset,
    save_dir: str,
    model: torch.nn.Module,
) -> None:
    total_params, trainable_params = _count_parameters(model)
    learned_target_indices = get_missing_indices(args.obs_lead_indices)
    learned_target_names = [LEAD_NAMES[idx] for idx in learned_target_indices]
    print("[Startup]")
    print(f"  device={device}")
    print(f"  model_family={args.model_family}")
    print(f"  regime={args.regime}")
    print(f"  obs_leads={[LEAD_NAMES[idx] for idx in args.obs_lead_indices]}")
    print(f"  learned_targets={learned_target_names}")
    print(f"  train_size={len(train_dataset)} val_size={len(val_dataset)}")
    print(f"  save_dir={save_dir}")
    print(f"  params_total={total_params} params_trainable={trainable_params}")
    print(
        "  fm_flags="
        f"perceptual={bool(args.fm_perceptual)} "
        f"decoder_conditioning={bool(args.fm_decoder_conditioning)} "
        f"latent_align={bool(args.fm_latent_align)}"
    )
    if args.model_family == "fm_vae":
        print(
            "  fm_config="
            f"checkpoint={args.fm_checkpoint} "
            f"fm_loss_weight={args.fm_loss_weight} "
            f"fm_cosine_mix={args.fm_cosine_mix} "
            f"latent_align_weight={args.latent_align_weight} "
            f"multi_scale_align_weight={args.multi_scale_align_weight} "
            f"cond_drop={args.fm_cond_drop_prob}"
        )
        print(
            "  encoder_config="
            f"mask_aware={bool(args.mask_aware_encoder)} "
            f"split_latent={bool(args.split_latent)} "
            f"global_latent={args.global_latent_channels} "
            f"local_latent={args.local_latent_channels}"
        )
    if args.model_family == "token_refiner":
        print(
            "  token_refiner_config="
            f"frozen_vae={args.frozen_vae_checkpoint} "
            f"teacher={args.teacher_encoder} "
            f"teacher_checkpoint={args.teacher_checkpoint or DEFAULT_HUBERT_MODEL if args.teacher_encoder == 'hubert' else args.teacher_checkpoint} "
            f"token_loss_weight={args.token_loss_weight} "
            f"token_loss_mix={args.token_loss_mix} "
            f"teacher_common_token_length={args.teacher_common_token_length} "
            f"smoothness={args.residual_smoothness_weight} "
            f"v2={bool(args.token_refiner_v2)} "
            f"observed_conditioning={bool(args.token_refiner_observed_conditioning)} "
            f"clamp_observed={bool(args.token_refiner_clamp_observed)} "
            f"teacher_layer_mode={args.teacher_layer_mode} "
            f"token_whitening={bool(args.token_whitening)} "
            f"token_loss_schedule={args.token_loss_schedule} "
            f"refiner={'alitok_causal_bottleneck' if bool(args.token_refiner_causal_alignment) else 'alitok_bottleneck'} "
            f"refiner_dim={args.refiner_dim} "
            f"query_len={args.refiner_query_len} "
            f"causal_alignment={bool(args.token_refiner_causal_alignment)} "
            f"prefix_tokens={args.token_refiner_prefix_tokens} "
            f"causal_loss_weight={args.token_refiner_causal_loss_weight} "
            f"prefix_aux_loss_weight={args.token_refiner_prefix_aux_loss_weight} "
            f"stage={args.token_refiner_stage}"
        )
    if args.model_family in {"alitok", "alitok_stage2", "alitok_hybrid"} and getattr(args, "teacher_encoder", None):
        print(
            "  alitok_teacher_config="
            f"teacher={args.teacher_encoder} "
            f"teacher_checkpoint={args.teacher_checkpoint} "
            f"teacher_dim={args.teacher_dim} "
            f"teacher_layer_mode={args.teacher_layer_mode} "
            f"teacher_loss_weight={args.token_loss_weight} "
            f"teacher_loss_mix={args.token_loss_mix} "
            f"teacher_common_token_length={args.teacher_common_token_length}"
        )
    print(f"  recon_weighting=observed:1.0 missing:{args.missing_lead_weight}")
    print(
        "  kl_schedule="
        f"mode={args.beta_kl_schedule} "
        f"start={args.beta_kl_start} "
        f"end={args.beta_kl_end} "
        f"warmup_steps={args.beta_kl_warmup_steps}"
    )
    print(
        "  selector="
        "lowest val/mse_reg -> val/mae_reg -> val/rmse_reg"
    )


def _print_first_batch_debug(
    x: torch.Tensor,
    y: torch.Tensor,
    out: dict[str, torch.Tensor],
    grad_norm: float | None = None,
) -> None:
    y_pred = out["y_pred"]
    z_reg = out["z_regressed"]
    print(
        "[DebugBatch] "
        f"x_shape={tuple(x.shape)} "
        f"x_min={x.min().item():.4f} x_max={x.max().item():.4f} "
        f"x_mean={x.mean().item():.4f} x_std={x.std().item():.4f} "
        f"y_shape={tuple(y.shape)} "
        f"y_min={y.min().item():.4f} y_max={y.max().item():.4f} "
        f"y_mean={y.mean().item():.4f} y_std={y.std().item():.4f} "
        f"y_pred_shape={tuple(y_pred.shape)} "
        f"y_pred_min={y_pred.min().item():.4f} y_pred_max={y_pred.max().item():.4f} "
        f"z_shape={tuple(z_reg.shape) if z_reg is not None else None}"
    )
    print(
        "[DebugLoss] "
        f"total={out['loss'].item():.4f} "
        f"decoder={out['decoder_loss'].item():.4f} "
        f"kl={out['kl_loss'].item():.6f} "
        f"fm={out.get('fm_perceptual_loss', out['teacher_loss']).item():.6f} "
        f"latent_align={out.get('latent_align_loss', out['align_loss']).item():.6f}"
    )
    if grad_norm is not None:
        print(f"[DebugGrad] grad_norm={grad_norm:.6f}")


def _print_validation_summary(
    metrics: dict[str, Any],
    args: argparse.Namespace,
    step: int,
    current_selector: tuple[float, float, float],
) -> None:
    learned_target_indices = get_missing_indices(args.obs_lead_indices)
    learned_target_names = [LEAD_NAMES[idx] for idx in learned_target_indices]
    print(f"\n[Validation] Step {step}")
    print(
        "  Regressor "
        f"R2={_named_metric(metrics, 'val/r2_regressor'):.4f} "
        f"MSE={_named_metric(metrics, 'val/mse_reg'):.6f} "
        f"MAE={_named_metric(metrics, 'val/mae_reg'):.6f} "
        f"RMSE={_named_metric(metrics, 'val/rmse_reg'):.6f}"
    )
    print(
        "  Losses "
        f"decoder={_named_metric(metrics, 'val/decoder_loss'):.6f} "
        f"kl={_named_metric(metrics, 'val/kl_loss'):.6f} "
        f"beta_kl={_named_metric(metrics, 'val/beta_kl'):.6e} "
        f"fm={_named_metric(metrics, 'val/fm_perceptual_loss'):.6f} "
        f"latent_align={_named_metric(metrics, 'val/latent_align_loss'):.6f}"
    )
    print(f"  Learned targets: {learned_target_names}")
    print(
        "[ValidationSummary] "
        f"step={step} "
        f"mse={_named_metric(metrics, 'val/mse_reg'):.6f} "
        f"mae={_named_metric(metrics, 'val/mae_reg'):.6f} "
        f"rmse={_named_metric(metrics, 'val/rmse_reg'):.6f} "
        f"r2_reg={_named_metric(metrics, 'val/r2_regressor'):.4f} "
        f"v4_v6={_named_metric(metrics, 'val/r2_reg_v4_v6_mean'):.4f} "
        f"v3_v6={_named_metric(metrics, 'val/r2_reg_v3_v6_mean'):.4f} "
        f"selector={current_selector}"
    )
    for lead_name in learned_target_names:
        print(
            "    "
            f"{lead_name} | "
            f"r2={_named_metric(metrics, f'val/lead_r2_reg_{lead_name}'):.4f} "
            f"mae={_named_metric(metrics, f'val/mae_reg_{lead_name}'):.5f} "
            f"mse={_named_metric(metrics, f'val/mse_reg_{lead_name}'):.5f} "
            f"rmse={_named_metric(metrics, f'val/rmse_reg_{lead_name}'):.5f} "
            f"corr={_named_metric(metrics, f'val/corr_reg_{lead_name}'):.4f}"
        )
    chest_keys = [
        "val/r2_reg_chest_mean",
        "val/r2_reg_lateral_mean",
        "val/r2_reg_v4_v6_mean",
        "val/r2_reg_v3_v6_mean",
        "val/mae_reg_chest_mean",
        "val/rmse_reg_chest_mean",
        "val/corr_reg_chest_mean",
    ]
    printed_any = False
    for key in chest_keys:
        if key in metrics:
            if not printed_any:
                print("  Aggregates")
                printed_any = True
            print(f"    {key.split('/', 1)[1]}={_named_metric(metrics, key):.6f}")
    morph_keys = [
        "val/clinical_reg_recon_chest_mean_beat_corr",
        "val/clinical_reg_recon_chest_mean_beat_rmse",
        "val/clinical_reg_recon_chest_mean_r_peak_timing_error_ms",
        "val/clinical_reg_full_chest_rwave_progression_mae",
        "val/clinical_reg_full_chest_rwave_progression_corr",
        "val/clinical_reg_recon_chest_rwave_progression_mae",
        "val/clinical_reg_recon_chest_rwave_progression_corr",
    ]
    morph_present = [key for key in morph_keys if key in metrics]
    if morph_present:
        print("  Morphology")
        for key in morph_present:
            print(f"    {key.split('/', 1)[1]}={_named_metric(metrics, key):.6f}")


def fm_features_active(args: argparse.Namespace) -> bool:
    if args.model_family == "token_refiner":
        return True
    if args.model_family in {"alitok", "alitok_stage2", "alitok_hybrid"} and getattr(args, "teacher_encoder", None):
        return True
    return args.model_family == "fm_vae" and (
        args.fm_perceptual or args.fm_decoder_conditioning or args.fm_latent_align or args.fm_multi_scale_align
    )


def architecture_version(args: argparse.Namespace) -> str:
    if args.model_family == "fm_vae":
        if getattr(args, "teacher_encoder", "ecgfm") != "ecgfm":
            return "fm_vae_generic_teacher_v2"
        return "fm_vae_mask_split_v1"
    if args.model_family == "token_refiner":
        if getattr(args, "token_refiner_causal_alignment", False):
            return "token_refiner_alitok_causal_bottleneck_v1"
        return "token_refiner_alitok_bottleneck_v1"
    if args.model_family == "alitok":
        return "alitok_stage1_causal"
    if args.model_family == "alitok_stage2":
        return "alitok_stage2_bidir_decoder"
    if args.model_family == "alitok_hybrid":
        return "alitok_stage1_stage2_hybrid"
    return "wearecg_vae_exact"


def build_run_metadata(args: argparse.Namespace) -> dict[str, Any]:
    if args.model_family == "baseline":
        model_family = "wearecg_vae"
    elif args.model_family == "alitok":
        model_family = "alitok_stage1"
    elif args.model_family == "alitok_stage2":
        model_family = "alitok_stage2"
    elif args.model_family == "alitok_hybrid":
        model_family = "alitok_stage1_stage2_hybrid"
    elif args.model_family == "token_refiner":
        model_family = "token_refiner"
    else:
        model_family = "fm_vae"
    return {
        "family": "engineering",
        "experiment_family": "engineering",
        "model_family": model_family,
        "comparison_protocol": "wear_ecg_exact_regime",
        "baseline_semantics": "wear_ecg_public_exact_modules",
        "primary_selector": "lowest_val_mse_then_lowest_val_mae_then_lowest_val_rmse",
        "epoch_validation_mode": "full_reconstruction" if args.full_val_reconstruction else "lightweight",
        "keep_epoch_checkpoints": int(args.keep_epoch_checkpoints),
        "regime": args.regime,
        "run_tag": args.run_tag,
        "obs_leads": [LEAD_NAMES[idx] for idx in args.obs_lead_indices],
        "obs_lead_indices": args.obs_lead_indices,
        "num_observed_leads": len(args.obs_lead_indices),
        "lead_regime": f"{len(args.obs_lead_indices)}lead",
        "split": args.split,
        "train_splits": [part.strip() for part in args.train_splits.split(",") if part.strip()],
        "eval_split": args.split,
        "train_augmentation": bool(args.train_augmentation),
        "beta_kl": args.beta_kl,
        "beta_kl_schedule": args.beta_kl_schedule,
        "beta_kl_start": args.beta_kl_start,
        "beta_kl_end": args.beta_kl_end,
        "beta_kl_warmup_steps": args.beta_kl_warmup_steps,
        "missing_lead_weight": args.missing_lead_weight,
        "latent_channels": 4,
        "target_len": args.target_len,
        "fm_checkpoint": args.fm_checkpoint if args.model_family == "fm_vae" else None,
        "frozen_vae_checkpoint": args.frozen_vae_checkpoint if args.model_family == "token_refiner" else None,
        "teacher_encoder": args.teacher_encoder if args.model_family in {"fm_vae", "token_refiner", "alitok", "alitok_stage2", "alitok_hybrid"} else None,
        "teacher_checkpoint": args.teacher_checkpoint if args.model_family in {"fm_vae", "token_refiner", "alitok", "alitok_stage2", "alitok_hybrid"} else None,
        "teacher_dim": args.teacher_dim if args.model_family in {"fm_vae", "token_refiner", "alitok", "alitok_stage2", "alitok_hybrid"} else None,
        "teacher_token_length": args.teacher_token_length if args.model_family in {"fm_vae", "token_refiner"} else None,
        "token_loss_weight": args.token_loss_weight if args.model_family in {"token_refiner", "alitok", "alitok_stage2", "alitok_hybrid"} else 0.0,
        "token_loss_mix": args.token_loss_mix if args.model_family in {"token_refiner", "alitok", "alitok_stage2", "alitok_hybrid"} else 0.0,
        "teacher_common_token_length": args.teacher_common_token_length if args.model_family in {"fm_vae", "token_refiner", "alitok", "alitok_stage2", "alitok_hybrid"} else None,
        "teacher_layer_mode": args.teacher_layer_mode if args.model_family in {"fm_vae", "token_refiner", "alitok", "alitok_stage2", "alitok_hybrid"} else "last",
        "residual_smoothness_weight": args.residual_smoothness_weight if args.model_family == "token_refiner" else 0.0,
        "refiner_dim": args.refiner_dim if args.model_family == "token_refiner" else None,
        "refiner_query_len": args.refiner_query_len if args.model_family == "token_refiner" else None,
        "refiner_impl": (
            "alitok_causal_bottleneck" if args.model_family == "token_refiner" and bool(args.token_refiner_causal_alignment)
            else "alitok_bottleneck" if args.model_family == "token_refiner"
            else None
        ),
        "token_refiner_causal_alignment": bool(args.token_refiner_causal_alignment) if args.model_family == "token_refiner" else False,
        "token_refiner_prefix_tokens": args.token_refiner_prefix_tokens if args.model_family == "token_refiner" else 0,
        "token_refiner_causal_loss_weight": args.token_refiner_causal_loss_weight if args.model_family == "token_refiner" else 0.0,
        "token_refiner_prefix_aux_loss_weight": args.token_refiner_prefix_aux_loss_weight if args.model_family == "token_refiner" else 0.0,
        "token_refiner_stage": args.token_refiner_stage if args.model_family == "token_refiner" else None,
        "token_refiner_v2": bool(args.token_refiner_v2) if args.model_family == "token_refiner" else False,
        "token_refiner_observed_conditioning": bool(args.token_refiner_observed_conditioning) if args.model_family == "token_refiner" else False,
        "token_refiner_clamp_observed": bool(args.token_refiner_clamp_observed) if args.model_family == "token_refiner" else False,
        "token_whitening": bool(args.token_whitening) if args.model_family == "token_refiner" else False,
        "token_whitening_batches": args.token_whitening_batches if args.model_family == "token_refiner" else 0,
        "token_improvement_margin_weight": args.token_improvement_margin_weight if args.model_family == "token_refiner" else 0.0,
        "token_improvement_margin": args.token_improvement_margin if args.model_family == "token_refiner" else 0.0,
        "token_loss_schedule": args.token_loss_schedule if args.model_family == "token_refiner" else "constant",
        "token_loss_warmup_epochs": args.token_loss_warmup_epochs if args.model_family == "token_refiner" else 0,
        "token_loss_ramp_epochs": args.token_loss_ramp_epochs if args.model_family == "token_refiner" else 0,
        "token_loss_target_weight": args.token_loss_target_weight if args.model_family == "token_refiner" else 0.0,
        "checkpoint_contains_fm_backbone": False if args.model_family in {"fm_vae", "token_refiner"} else True,
        "random_teacher_seed": args.random_teacher_seed if args.model_family in {"fm_vae", "token_refiner", "alitok", "alitok_stage2", "alitok_hybrid"} else None,
        "fm_loss_weight": args.fm_loss_weight if args.model_family == "fm_vae" else 0.0,
        "fm_cosine_mix": args.fm_cosine_mix if args.model_family == "fm_vae" else 0.0,
        "fm_features_active": fm_features_active(args),
        "fm_perceptual": bool(args.fm_perceptual) if args.model_family == "fm_vae" else False,
        "fm_decoder_conditioning": bool(args.fm_decoder_conditioning) if args.model_family == "fm_vae" else False,
        "fm_cond_drop_prob": args.fm_cond_drop_prob if args.model_family == "fm_vae" else 0.0,
        "fm_latent_align": bool(args.fm_latent_align) if args.model_family == "fm_vae" else False,
        "latent_align_weight": args.latent_align_weight if args.model_family == "fm_vae" else 0.0,
        "fm_multi_scale_align": bool(args.fm_multi_scale_align) if args.model_family == "fm_vae" else False,
        "multi_scale_align_weight": args.multi_scale_align_weight if args.model_family == "fm_vae" else 0.0,
        "architecture_version": architecture_version(args),
        "mask_aware_encoder": bool(args.mask_aware_encoder) if args.model_family == "fm_vae" else False,
        "split_latent": bool(args.split_latent) if args.model_family == "fm_vae" else False,
        "global_latent_channels": args.global_latent_channels if args.model_family == "fm_vae" else 0,
        "local_latent_channels": args.local_latent_channels if args.model_family == "fm_vae" else 0,
        "alitok_patch_size": args.alitok_patch_size if args.model_family in {"alitok", "alitok_stage2", "alitok_hybrid"} else None,
        "alitok_token_size": args.alitok_token_size if args.model_family in {"alitok", "alitok_stage2", "alitok_hybrid"} else None,
        "alitok_prefix_tokens": args.alitok_prefix_tokens if args.model_family in {"alitok", "alitok_stage2", "alitok_hybrid"} else None,
        "alitok_codebook_size": args.alitok_codebook_size if args.model_family in {"alitok", "alitok_stage2", "alitok_hybrid"} else None,
        "alitok_encoder_depth": args.alitok_encoder_depth if args.model_family in {"alitok", "alitok_stage2", "alitok_hybrid"} else None,
        "alitok_decoder_depth": args.alitok_decoder_depth if args.model_family in {"alitok", "alitok_stage2", "alitok_hybrid"} else None,
        "alitok_encoder_width": args.alitok_encoder_width if args.model_family in {"alitok", "alitok_stage2", "alitok_hybrid"} else None,
        "alitok_decoder_width": args.alitok_decoder_width if args.model_family in {"alitok", "alitok_stage2", "alitok_hybrid"} else None,
        "alitok_encoder_heads": args.alitok_encoder_heads if args.model_family in {"alitok", "alitok_stage2", "alitok_hybrid"} else None,
        "alitok_decoder_heads": args.alitok_decoder_heads if args.model_family in {"alitok", "alitok_stage2", "alitok_hybrid"} else None,
        "alitok_heads": args.alitok_heads if args.model_family in {"alitok", "alitok_stage2", "alitok_hybrid"} else None,
        "alitok_stage2_buffer_tokens": args.alitok_stage2_buffer_tokens if args.model_family in {"alitok", "alitok_stage2", "alitok_hybrid"} else None,
        "alitok_clustering_vq": bool(args.alitok_clustering_vq) if args.model_family in {"alitok", "alitok_stage2", "alitok_hybrid"} else None,
        "alitok_stage2_mix": args.alitok_stage2_mix if args.model_family == "alitok_hybrid" else None,
        "has_regressor_path": True,
        "has_teacher_path": False,
        "has_diffusion_rollout": False,
        "has_direct_prior": False,
        "initialization_type": "scratch" if not args.resume else "full_warm_start",
    }


def build_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    if args.model_family == "baseline":
        model = WearECGVAE(
            latent_channels=4,
            target_len=args.target_len,
            beta_kl=args.beta_kl,
            missing_lead_weight=args.missing_lead_weight,
        )
    elif args.model_family == "fm_vae":
        model = WearECGFMVAE(
            fm_checkpoint_path=args.fm_checkpoint,
            fm_teacher_encoder=args.teacher_encoder,
            teacher_checkpoint=args.teacher_checkpoint,
            teacher_dim=args.teacher_dim,
            teacher_token_length=args.teacher_token_length,
            teacher_common_token_length=args.teacher_common_token_length,
            teacher_layer_mode=args.teacher_layer_mode,
            random_teacher_seed=args.random_teacher_seed,
            latent_channels=4,
            target_len=args.target_len,
            beta_kl=args.beta_kl,
            missing_lead_weight=args.missing_lead_weight,
            fm_loss_weight=args.fm_loss_weight if args.fm_perceptual else 0.0,
            fm_cosine_mix=args.fm_cosine_mix,
            use_decoder_conditioning=args.fm_decoder_conditioning,
            fm_cond_drop_prob=args.fm_cond_drop_prob,
            use_latent_alignment=args.fm_latent_align,
            latent_align_weight=args.latent_align_weight,
            use_multi_scale_align=args.fm_multi_scale_align,
            multi_scale_align_weight=args.multi_scale_align_weight,
            mask_aware_encoder=args.mask_aware_encoder,
            split_latent=args.split_latent,
            global_latent_channels=args.global_latent_channels,
            local_latent_channels=args.local_latent_channels,
        )
    elif args.model_family == "token_refiner":
        model = WearECGTokenRefiner(
            frozen_vae_checkpoint=args.frozen_vae_checkpoint,
            teacher_encoder=args.teacher_encoder,
            teacher_checkpoint=args.teacher_checkpoint,
            target_len=args.target_len,
            beta_kl=args.beta_kl,
            missing_lead_weight=args.missing_lead_weight,
            token_loss_weight=args.token_loss_weight,
            token_loss_mix=args.token_loss_mix,
            residual_smoothness_weight=args.residual_smoothness_weight,
            teacher_dim=args.teacher_dim,
            teacher_token_length=args.teacher_token_length,
            teacher_common_token_length=args.teacher_common_token_length,
            refiner_dim=args.refiner_dim,
            query_len=args.refiner_query_len,
            random_seed=args.random_teacher_seed,
            use_observed_conditioning=args.token_refiner_observed_conditioning,
            clamp_observed_output=args.token_refiner_clamp_observed,
            token_improvement_margin_weight=args.token_improvement_margin_weight,
            token_improvement_margin=args.token_improvement_margin,
            teacher_layer_mode=args.teacher_layer_mode,
            token_whitening=args.token_whitening,
            causal_alignment=args.token_refiner_causal_alignment,
            prefix_tokens=args.token_refiner_prefix_tokens,
            causal_loss_weight=args.token_refiner_causal_loss_weight,
            prefix_aux_loss_weight=args.token_refiner_prefix_aux_loss_weight,
            refiner_stage=args.token_refiner_stage,
        )
    elif args.model_family == "alitok":
        if build_alitok_vae_1d is None:
            raise RuntimeError("AliTok model requested, but alitok_vae_exp.py is not available in this checkout.")
        model = build_alitok_vae_1d(
            architecture="stage1_causal",
            target_len=args.target_len,
            patch_size=args.alitok_patch_size,
            token_size=args.alitok_token_size,
            missing_lead_weight=args.missing_lead_weight,
            prefix_tokens=args.alitok_prefix_tokens,
            codebook_size=args.alitok_codebook_size,
            encoder_depth=args.alitok_encoder_depth,
            decoder_depth=args.alitok_decoder_depth,
            heads=args.alitok_heads,
            encoder_heads=args.alitok_encoder_heads,
            decoder_heads=args.alitok_decoder_heads,
            encoder_width=args.alitok_encoder_width,
            decoder_width=args.alitok_decoder_width,
            stage2_buffer_tokens=args.alitok_stage2_buffer_tokens,
            clustering_vq=args.alitok_clustering_vq,
            teacher_encoder=args.teacher_encoder if getattr(args, "teacher_encoder", None) else None,
            teacher_checkpoint=args.teacher_checkpoint,
            teacher_dim=args.teacher_dim,
            teacher_token_length=getattr(args, "teacher_token_length", None),
            teacher_common_token_length=args.teacher_common_token_length,
            teacher_layer_mode=args.teacher_layer_mode,
            teacher_loss_weight=args.token_loss_weight,
            teacher_loss_mix=args.token_loss_mix,
            random_seed=args.random_teacher_seed,
        )
    elif args.model_family == "alitok_stage2":
        if build_alitok_vae_1d is None:
            raise RuntimeError("AliTok stage2 requested, but alitok_vae_exp.py is not available in this checkout.")
        model = build_alitok_vae_1d(
            architecture="stage2_bidir",
            target_len=args.target_len,
            patch_size=args.alitok_patch_size,
            token_size=args.alitok_token_size,
            missing_lead_weight=args.missing_lead_weight,
            prefix_tokens=args.alitok_prefix_tokens,
            codebook_size=args.alitok_codebook_size,
            encoder_depth=args.alitok_encoder_depth,
            decoder_depth=args.alitok_decoder_depth,
            heads=args.alitok_heads,
            encoder_heads=args.alitok_encoder_heads,
            decoder_heads=args.alitok_decoder_heads,
            encoder_width=args.alitok_encoder_width,
            decoder_width=args.alitok_decoder_width,
            stage2_buffer_tokens=args.alitok_stage2_buffer_tokens,
            clustering_vq=args.alitok_clustering_vq,
            teacher_encoder=args.teacher_encoder if getattr(args, "teacher_encoder", None) else None,
            teacher_checkpoint=args.teacher_checkpoint,
            teacher_dim=args.teacher_dim,
            teacher_token_length=getattr(args, "teacher_token_length", None),
            teacher_common_token_length=args.teacher_common_token_length,
            teacher_layer_mode=args.teacher_layer_mode,
            teacher_loss_weight=args.token_loss_weight,
            teacher_loss_mix=args.token_loss_mix,
            random_seed=args.random_teacher_seed,
        )
    elif args.model_family == "alitok_hybrid":
        if build_alitok_vae_1d is None:
            raise RuntimeError("AliTok hybrid requested, but alitok_vae_exp.py is not available in this checkout.")
        model = build_alitok_vae_1d(
            architecture="stage1_stage2_hybrid",
            target_len=args.target_len,
            patch_size=args.alitok_patch_size,
            token_size=args.alitok_token_size,
            stage2_mix=args.alitok_stage2_mix,
            missing_lead_weight=args.missing_lead_weight,
            prefix_tokens=args.alitok_prefix_tokens,
            codebook_size=args.alitok_codebook_size,
            encoder_depth=args.alitok_encoder_depth,
            decoder_depth=args.alitok_decoder_depth,
            heads=args.alitok_heads,
            encoder_heads=args.alitok_encoder_heads,
            decoder_heads=args.alitok_decoder_heads,
            encoder_width=args.alitok_encoder_width,
            decoder_width=args.alitok_decoder_width,
            stage2_buffer_tokens=args.alitok_stage2_buffer_tokens,
            clustering_vq=args.alitok_clustering_vq,
            teacher_encoder=args.teacher_encoder if getattr(args, "teacher_encoder", None) else None,
            teacher_checkpoint=args.teacher_checkpoint,
            teacher_dim=args.teacher_dim,
            teacher_token_length=getattr(args, "teacher_token_length", None),
            teacher_common_token_length=args.teacher_common_token_length,
            teacher_layer_mode=args.teacher_layer_mode,
            teacher_loss_weight=args.token_loss_weight,
            teacher_loss_mix=args.token_loss_mix,
            random_seed=args.random_teacher_seed,
        )
    else:
        raise ValueError(f"Unknown model_family: {args.model_family}")
    return model.to(device, dtype=torch.float32)


def _batch_r2_per_lead(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    ss_res = ((pred - true) ** 2).sum(dim=2)
    centered = true - true.mean(dim=2, keepdim=True)
    ss_tot = (centered ** 2).sum(dim=2)
    valid = ss_tot > 1e-6
    ratio = torch.zeros_like(ss_res)
    ratio[valid] = ss_res[valid] / ss_tot[valid]
    return 1.0 - ratio


def _batch_corr_per_lead(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    pred_centered = pred - pred.mean(dim=2, keepdim=True)
    true_centered = true - true.mean(dim=2, keepdim=True)
    numerator = (pred_centered * true_centered).sum(dim=2)
    denominator = torch.sqrt(
        (pred_centered**2).sum(dim=2).clamp(min=1e-8)
        * (true_centered**2).sum(dim=2).clamp(min=1e-8)
    ).clamp(min=1e-8)
    return numerator / denominator


def lightweight_validate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    metrics: dict[str, float] = {}
    lead_names = LEAD_NAMES
    total_samples = 0
    total_decoder = 0.0
    total_kl = 0.0
    total_fm = 0.0
    total_latent_align = 0.0
    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    total_r2 = 0.0
    lead_mse = torch.zeros(len(lead_names), dtype=torch.float64, device=device)
    lead_mae = torch.zeros(len(lead_names), dtype=torch.float64, device=device)
    lead_rmse = torch.zeros(len(lead_names), dtype=torch.float64, device=device)
    lead_r2 = torch.zeros(len(lead_names), dtype=torch.float64, device=device)
    lead_corr = torch.zeros(len(lead_names), dtype=torch.float64, device=device)

    with torch.no_grad():
        for x, y, _meta in tqdm(val_loader, desc="Validate-light"):
            x = x.to(device, dtype=torch.float32, non_blocking=True)
            y = y.to(device, dtype=torch.float32, non_blocking=True)
            batch_size = x.size(0)
            lead_indices = make_lead_indices(args.obs_lead_indices, batch_size, device)
            x = mask_unobserved_leads(x, args.obs_lead_indices)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(x, y_full=y, lead_indices=lead_indices, mode="stage1")

            y_pred = out["y_pred"].float()
            y_true = y.float()
            if y_true.shape[-1] != y_pred.shape[-1]:
                y_true = y_true[..., : y_pred.shape[-1]] if y_true.shape[-1] > y_pred.shape[-1] else F.pad(y_true, (0, y_pred.shape[-1] - y_true.shape[-1]))
            batch_mse_per_lead = ((y_pred - y_true) ** 2).mean(dim=(0, 2)).double()
            batch_mae_per_lead = (y_pred - y_true).abs().mean(dim=(0, 2)).double()
            batch_rmse_per_lead = torch.sqrt(batch_mse_per_lead)
            batch_r2_per_lead = _batch_r2_per_lead(y_pred, y_true).mean(dim=0).double()
            batch_corr_per_lead = _batch_corr_per_lead(y_pred, y_true).mean(dim=0).double()

            total_samples += batch_size
            total_decoder += float(out["decoder_loss"].item()) * batch_size
            total_kl += float(out["kl_loss"].item()) * batch_size
            total_fm += float(out.get("fm_perceptual_loss", out["teacher_loss"]).item()) * batch_size
            total_latent_align += float(out.get("latent_align_loss", out["align_loss"]).item()) * batch_size
            total_loss += float(out["loss"].item()) * batch_size
            total_mse += float(batch_mse_per_lead.mean().item()) * batch_size
            total_mae += float(batch_mae_per_lead.mean().item()) * batch_size
            total_rmse += float(batch_rmse_per_lead.mean().item()) * batch_size
            total_r2 += float(batch_r2_per_lead.mean().item()) * batch_size
            lead_mse += batch_mse_per_lead * batch_size
            lead_mae += batch_mae_per_lead * batch_size
            lead_rmse += batch_rmse_per_lead * batch_size
            lead_r2 += batch_r2_per_lead * batch_size
            lead_corr += batch_corr_per_lead * batch_size

    denom = max(total_samples, 1)
    metrics["val/total_loss"] = total_loss / denom
    metrics["val/decoder_loss"] = total_decoder / denom
    metrics["val/kl_loss"] = total_kl / denom
    metrics["val/fm_perceptual_loss"] = total_fm / denom
    metrics["val/latent_align_loss"] = total_latent_align / denom
    metrics["val/mse_reg"] = total_mse / denom
    metrics["val/mae_reg"] = total_mae / denom
    metrics["val/rmse_reg"] = total_rmse / denom
    metrics["val/r2_regressor"] = total_r2 / denom
    metrics["val/eval_mode"] = "lightweight"

    lead_mse /= denom
    lead_mae /= denom
    lead_rmse /= denom
    lead_r2 /= denom
    lead_corr /= denom
    for idx, lead_name in enumerate(lead_names):
        metrics[f"val/lead_r2_reg_{lead_name}"] = float(lead_r2[idx].item())
        metrics[f"val/mae_reg_{lead_name}"] = float(lead_mae[idx].item())
        metrics[f"val/mse_reg_{lead_name}"] = float(lead_mse[idx].item())
        metrics[f"val/rmse_reg_{lead_name}"] = float(lead_rmse[idx].item())
        metrics[f"val/corr_reg_{lead_name}"] = float(lead_corr[idx].item())

    def _mean_for(leads: list[str], values: torch.Tensor) -> float:
        picked = [float(values[lead_names.index(lead)].item()) for lead in leads]
        return sum(picked) / len(picked)

    metrics["val/r2_reg_chest_mean"] = _mean_for(["V1", "V2", "V3", "V4", "V5", "V6"], lead_r2)
    metrics["val/r2_reg_lateral_mean"] = _mean_for(["V4", "V5", "V6"], lead_r2)
    metrics["val/r2_reg_v4_v6_mean"] = _mean_for(["V4", "V5", "V6"], lead_r2)
    metrics["val/r2_reg_v3_v6_mean"] = _mean_for(["V3", "V4", "V5", "V6"], lead_r2)
    metrics["val/mae_reg_chest_mean"] = _mean_for(["V1", "V2", "V3", "V4", "V5", "V6"], lead_mae)
    metrics["val/rmse_reg_chest_mean"] = _mean_for(["V1", "V2", "V3", "V4", "V5", "V6"], lead_rmse)
    metrics["val/corr_reg_chest_mean"] = _mean_for(["V1", "V2", "V3", "V4", "V5", "V6"], lead_corr)
    return metrics


def _load_checkpoint_state_strict(model: torch.nn.Module, state_dict: dict[str, Any], model_family: str) -> None:
    model_state = model.state_dict()
    model_keys = set(model_state.keys())
    if model_family == "alitok_stage2":
        matching_state = {
            key: value
            for key, value in state_dict.items()
            if key in model_state and model_state[key].shape == value.shape
        }
        if not matching_state:
            raise RuntimeError("AliTok stage2 warm start found no compatible tensors to load.")
        model.load_state_dict(matching_state, strict=False)
        return

    if model_family == "fm_vae":
        excluded_prefixes = ("fm_model.",)
    elif model_family == "token_refiner":
        excluded_prefixes = ("frozen_vae.", "teacher.")
    elif model_family in {"alitok", "alitok_stage2", "alitok_hybrid"}:
        excluded_prefixes = ("teacher.",)
    else:
        excluded_prefixes = ()
    required_model_keys = {key for key in model_keys if not key.startswith(excluded_prefixes)}

    provided_keys = set(state_dict.keys())
    missing_keys = sorted(required_model_keys - provided_keys)
    unexpected_keys = sorted(key for key in (provided_keys - model_keys) if not key.startswith(excluded_prefixes))
    shape_mismatches = sorted(
        key for key in (required_model_keys & provided_keys) if model_state[key].shape != state_dict[key].shape
    )

    if missing_keys or unexpected_keys or shape_mismatches:
        raise RuntimeError(
            "Checkpoint/model mismatch. "
            f"missing={missing_keys[:12]} unexpected={unexpected_keys[:12]} "
            f"shape_mismatches={shape_mismatches[:12]}"
        )

    loadable_state = {
        key: value
        for key, value in state_dict.items()
        if key in model_keys and not key.startswith(excluded_prefixes)
    }
    incompatible = model.load_state_dict(loadable_state, strict=False)
    remaining_missing = [
        key for key in getattr(incompatible, "missing_keys", [])
        if not key.startswith(excluded_prefixes)
    ]
    remaining_unexpected = list(getattr(incompatible, "unexpected_keys", []))
    if remaining_missing or remaining_unexpected:
        raise RuntimeError(
            "Unexpected incompatibility after guarded checkpoint load. "
            f"missing={remaining_missing[:12]} unexpected={remaining_unexpected[:12]}"
        )


def build_warm_start_summary(args: argparse.Namespace, ckpt: dict[str, Any], model: torch.nn.Module) -> dict[str, Any]:
    saved_model_state = ckpt.get("model_state_dict", {})
    current_state = model.state_dict()
    loaded_keys = []
    skipped_keys = []
    for key, value in saved_model_state.items():
        if key not in current_state or current_state[key].shape != value.shape:
            skipped_keys.append(key)
        else:
            loaded_keys.append(key)
    init_type = "scratch"
    if args.resume:
        init_type = "partial_warm_start" if skipped_keys else "full_warm_start"
    return {
        "checkpoint_path": args.resume,
        "loaded_tensor_count": len(loaded_keys),
        "skipped_tensor_count": len(skipped_keys),
        "loaded_prefixes": sorted({key.split(".")[0] for key in loaded_keys}),
        "skipped_prefixes_sample": sorted({key.split(".")[0] for key in skipped_keys})[:20],
        "loaded_keys_sample": sorted(loaded_keys)[:20],
        "skipped_keys_sample": sorted(skipped_keys)[:20],
        "initialization_type": init_type,
    }


def extract_checkpoint_state(model: torch.nn.Module, args: argparse.Namespace) -> dict[str, Any]:
    state = model.state_dict()
    if args.model_family == "fm_vae":
        state = {key: value for key, value in state.items() if not key.startswith("fm_model.")}
    if args.model_family == "token_refiner":
        state = {
            key: value
            for key, value in state.items()
            if not key.startswith("frozen_vae.") and not key.startswith("teacher.")
        }
    if args.model_family in {"alitok", "alitok_stage2", "alitok_hybrid"}:
        state = {key: value for key, value in state.items() if not key.startswith("teacher.")}

    # AliTok variants are large; serialize checkpoint tensors in fp16 on CPU
    # to reduce disk footprint and avoid filesystem write failures.
    if args.model_family in {"alitok", "alitok_stage2", "alitok_hybrid"}:
        compact_state: dict[str, Any] = {}
        for key, value in state.items():
            tensor = value.detach().cpu()
            if torch.is_floating_point(tensor):
                tensor = tensor.half()
            compact_state[key] = tensor
        return compact_state

    return state


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lead_tag = format_lead_set(args.obs_lead_indices)
    model_family_tag = getattr(args, "model_family_tag", args.model_family if args.model_family != "baseline" else "baseline")
    run_tag_suffix = f"_{args.run_tag}" if args.run_tag else ""
    fallback_missing_weight_tag = f"_mw{args.missing_lead_weight:g}" if args.missing_lead_weight != 1.0 else ""
    name_suffix = run_tag_suffix if args.run_tag else fallback_missing_weight_tag
    run_name = f"wearecg_fm_{model_family_tag}_{lead_tag}_bs{args.batch_size}{name_suffix}"

    wandb_enabled = False
    try:
        wandb.init(
            project="WearECG-FM-Scientific-Production",
            name=run_name,
            config=vars(args),
        )
        wandb_enabled = wandb.run is not None
    except Exception as exc:
        print(f"[W&B] init failed; continuing without W&B ({exc})")
        wandb_enabled = False

    base_dir = "/home/mithunmanivannan/data/ptb_xl/tensors"
    train_split_names = [part.strip() for part in args.train_splits.split(",") if part.strip()]
    if not train_split_names:
        raise ValueError("--train_splits must include at least one split name")
    train_dataset = CombinedTensorFolderDataset([f"{base_dir}/{split_name}" for split_name in train_split_names])
    if args.train_augmentation:
        train_dataset = AugmentedTensorDataset(train_dataset, target_len=args.target_len)
    val_dataset = TensorFolderDataset(f"{base_dir}/{args.split}")
    if args.debug:
        train_dataset = torch.utils.data.Subset(train_dataset, range(64))
        val_dataset = torch.utils.data.Subset(val_dataset, range(64))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.train_num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.train_num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.val_num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.val_num_workers > 0,
    )

    save_dir = f"checkpoints/wearecg_fm/engineering_{model_family_tag}_{lead_tag}_bs{args.batch_size}{name_suffix}"
    os.makedirs(save_dir, exist_ok=True)
    cleanup_partial_checkpoints(save_dir)

    model = build_model(args, device)
    token_whiten_stats = compute_token_whitening_stats(model, train_loader, device, args)
    if token_whiten_stats is not None:
        mean, std = token_whiten_stats
        write_json(
            os.path.join(save_dir, "token_whitening_stats.json"),
            {
                "teacher_encoder": args.teacher_encoder,
                "teacher_layer_mode": args.teacher_layer_mode,
                "num_token_layers": int(mean.shape[0]),
                "teacher_dim": int(mean.shape[1]),
                "token_whitening_batches": int(args.token_whitening_batches),
                "mean_abs_mean": float(mean.abs().mean().item()),
                "std_mean": float(std.mean().item()),
                "std_min": float(std.min().item()),
            },
        )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.max_lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.2,
    )

    start_epoch = 0
    best_selector = None
    warm_start_summary = None
    if args.resume:
        print(f"Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        _load_checkpoint_state_strict(model, ckpt["model_state_dict"], args.model_family)
        if args.model_family == "alitok_stage2":
            print("AliTok stage2 warm-start loaded model tensors only; optimizer, scheduler, and epoch are reset.")
        elif args.model_family == "token_refiner" and getattr(args, "token_refiner_stage", "causal_align") == "bidir_refine":
            print("Token-refiner bidir_refine loaded model tensors only; optimizer, scheduler, and epoch are reset.")
        elif "optimizer_state_dict" in ckpt and "scheduler_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt["epoch"]
            best_selector = tuple(ckpt.get("best_selector_tuple", [])) or None
        warm_start_summary = build_warm_start_summary(args, ckpt, model)
        write_warm_start_summary(save_dir, warm_start_summary)

    metadata = build_run_metadata(args)
    write_json(os.path.join(save_dir, "run_metadata.json"), metadata)
    step_counter = start_epoch * len(train_loader)
    opt_step = 0
    debug_first_batch_printed = False

    if hasattr(model, "beta_kl"):
        model.beta_kl = current_beta_kl(args, step_counter)

    if args.debug or args.debug_logs:
        _print_debug_banner(args, device, train_dataset, val_dataset, save_dir, model)
        if warm_start_summary is not None:
            _print_warm_start_summary(warm_start_summary)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        token_weight_now = current_token_loss_weight(args, epoch)
        if args.model_family == "token_refiner":
            model.token_loss_weight = token_weight_now
        if args.debug or args.debug_logs:
            print(
                f"\n[EpochStart] epoch={epoch + 1}/{args.epochs} "
                f"lr={optimizer.param_groups[0]['lr']:.8f} token_loss_weight={token_weight_now:.6f}"
            )
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for x, y, _meta in pbar:
            x = x.to(device, dtype=torch.float32, non_blocking=True)
            y = y.to(device, dtype=torch.float32, non_blocking=True)
            lead_indices = make_lead_indices(args.obs_lead_indices, x.size(0), device)
            beta_kl_now = current_beta_kl(args, step_counter)
            if hasattr(model, "beta_kl"):
                model.beta_kl = beta_kl_now

            # Mask missing leads so the model sees only the observed leads on input.
            x = mask_unobserved_leads(x, args.obs_lead_indices)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(x, y_full=y, lead_indices=lead_indices, mode="stage1")
                loss = out["loss"]

            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                1.0,
            )
            if not torch.isfinite(grad_norm):
                print(f"Non-finite grad_norm at step {step_counter}. Skipping batch.")
                optimizer.zero_grad(set_to_none=True)
            else:
                if (args.debug or args.debug_logs) and not debug_first_batch_printed:
                    _print_first_batch_debug(x, y, out, float(grad_norm))
                    debug_first_batch_printed = True
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1

            decoder_loss_val = float(out["decoder_loss"].item())
            kl_loss_val = float(out["kl_loss"].item())
            fm_loss_val = float(out.get("fm_perceptual_loss", out["teacher_loss"]).item())
            latent_align_val = float(out.get("latent_align_loss", out["align_loss"]).item())
            total_loss_val = float(loss.item())
            if not all(math.isfinite(v) for v in [decoder_loss_val, kl_loss_val, fm_loss_val, latent_align_val, total_loss_val]):
                print(f"Non-finite loss component at step {step_counter}. Skipping batch log.")
                step_counter += 1
                continue

            if wandb_enabled:
                safe_wandb_log(
                    {
                        "train/decoder_loss": decoder_loss_val,
                        "train/kl_loss": kl_loss_val,
                        "train/beta_kl": beta_kl_now,
                        "train/fm_perceptual_loss": fm_loss_val,
                        "train/latent_align_loss": latent_align_val,
                        "train/token_margin_loss": float(out.get("token_margin_loss", torch.tensor(0.0)).item()),
                        "train/coarse_token_loss": float(out.get("coarse_token_loss", torch.tensor(0.0)).item()),
                        "train/causal_alignment_loss": float(out.get("causal_alignment_loss", torch.tensor(0.0)).item()),
                        "train/prefix_aux_loss": float(out.get("prefix_aux_loss", torch.tensor(0.0)).item()),
                        "train/token_loss_weight": float(token_weight_now),
                        "train/total_loss": total_loss_val,
                        "train/grad_norm": float(grad_norm),
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "step": step_counter,
                        "opt_step": opt_step,
                    },
                    step=step_counter,
                    prefix="train",
                )
            pbar.set_postfix(
                {
                    "Loss": f"{total_loss_val:.3f}",
                    "Dec": f"{decoder_loss_val:.3f}",
                    "KL": f"{kl_loss_val:.4f}",
                    "Beta": f"{beta_kl_now:.2e}",
                    "FM": f"{fm_loss_val:.4f}",
                }
            )
            step_counter += 1

        beta_kl_val = current_beta_kl(args, step_counter)
        if hasattr(model, "beta_kl"):
            model.beta_kl = beta_kl_val

        if args.full_val_reconstruction:
            model.eval()
            metrics = evaluate_reconstruction(
                model,
                val_loader,
                device,
                args.obs_lead_indices,
                split="val",
                step=step_counter,
                model_family=args.model_family if args.model_family != "baseline" else "wearecg_vae",
                log_to_wandb=False,
            )
        else:
            metrics = lightweight_validate(model, val_loader, device, args)
        metrics["val/beta_kl"] = beta_kl_val
        write_run_artifacts(save_dir, metadata, metrics)
        current_selector = get_baseline_selector(metrics)
        _print_validation_summary(metrics, args, step_counter, current_selector)
        is_best = best_selector is None or current_selector > best_selector
        if is_best:
            best_selector = current_selector

        ckpt_path = os.path.join(save_dir, f"ul_ecp_ep{epoch + 1}.pt")
        tmp_ckpt_path = ckpt_path + ".tmp"
        save_dict = {
            "epoch": epoch + 1,
            "model_state_dict": extract_checkpoint_state(model, args),
            "best_selector_tuple": list(best_selector) if best_selector is not None else None,
            "current_selector": list(current_selector),
            "model_family": args.model_family if args.model_family != "baseline" else "wearecg_vae",
            "run_tag": args.run_tag,
            "obs_leads": args.obs_lead_indices,
            "regime": args.regime,
            "beta_kl": args.beta_kl,
            "beta_kl_schedule": args.beta_kl_schedule,
            "beta_kl_start": args.beta_kl_start,
            "beta_kl_end": args.beta_kl_end,
            "beta_kl_warmup_steps": args.beta_kl_warmup_steps,
            "beta_kl_current": beta_kl_val,
            "missing_lead_weight": args.missing_lead_weight,
            "latent_channels": 4,
            "target_len": args.target_len,
            "fm_checkpoint": args.fm_checkpoint if args.model_family == "fm_vae" else None,
            "frozen_vae_checkpoint": args.frozen_vae_checkpoint if args.model_family == "token_refiner" else None,
            "teacher_encoder": args.teacher_encoder if args.model_family in {"fm_vae", "token_refiner", "alitok", "alitok_stage2", "alitok_hybrid"} else None,
            "teacher_checkpoint": args.teacher_checkpoint if args.model_family in {"fm_vae", "token_refiner", "alitok", "alitok_stage2", "alitok_hybrid"} else None,
            "teacher_dim": args.teacher_dim if args.model_family in {"fm_vae", "token_refiner", "alitok", "alitok_stage2", "alitok_hybrid"} else None,
            "teacher_token_length": args.teacher_token_length if args.model_family in {"fm_vae", "token_refiner"} else None,
            "token_loss_weight": args.token_loss_weight if args.model_family in {"token_refiner", "alitok", "alitok_stage2", "alitok_hybrid"} else 0.0,
            "token_loss_mix": args.token_loss_mix if args.model_family in {"token_refiner", "alitok", "alitok_stage2", "alitok_hybrid"} else 0.0,
            "teacher_common_token_length": args.teacher_common_token_length if args.model_family in {"fm_vae", "token_refiner", "alitok", "alitok_stage2", "alitok_hybrid"} else None,
            "teacher_layer_mode": args.teacher_layer_mode if args.model_family in {"fm_vae", "token_refiner", "alitok", "alitok_stage2", "alitok_hybrid"} else "last",
            "residual_smoothness_weight": args.residual_smoothness_weight if args.model_family == "token_refiner" else 0.0,
            "refiner_dim": args.refiner_dim if args.model_family == "token_refiner" else None,
            "refiner_query_len": args.refiner_query_len if args.model_family == "token_refiner" else None,
            "refiner_impl": (
                "alitok_causal_bottleneck" if args.model_family == "token_refiner" and bool(args.token_refiner_causal_alignment)
                else "alitok_bottleneck" if args.model_family == "token_refiner"
                else None
            ),
            "token_refiner_causal_alignment": bool(args.token_refiner_causal_alignment) if args.model_family == "token_refiner" else False,
            "token_refiner_prefix_tokens": args.token_refiner_prefix_tokens if args.model_family == "token_refiner" else 0,
            "token_refiner_causal_loss_weight": args.token_refiner_causal_loss_weight if args.model_family == "token_refiner" else 0.0,
            "token_refiner_prefix_aux_loss_weight": args.token_refiner_prefix_aux_loss_weight if args.model_family == "token_refiner" else 0.0,
            "token_refiner_stage": args.token_refiner_stage if args.model_family == "token_refiner" else None,
            "random_teacher_seed": args.random_teacher_seed if args.model_family in {"fm_vae", "token_refiner", "alitok", "alitok_stage2", "alitok_hybrid"} else None,
            "token_refiner_v2": bool(args.token_refiner_v2) if args.model_family == "token_refiner" else False,
            "token_refiner_observed_conditioning": bool(args.token_refiner_observed_conditioning) if args.model_family == "token_refiner" else False,
            "token_refiner_clamp_observed": bool(args.token_refiner_clamp_observed) if args.model_family == "token_refiner" else False,
            "token_whitening": bool(args.token_whitening) if args.model_family == "token_refiner" else False,
            "token_whitening_batches": args.token_whitening_batches if args.model_family == "token_refiner" else 0,
            "token_improvement_margin_weight": args.token_improvement_margin_weight if args.model_family == "token_refiner" else 0.0,
            "token_improvement_margin": args.token_improvement_margin if args.model_family == "token_refiner" else 0.0,
            "token_loss_schedule": args.token_loss_schedule if args.model_family == "token_refiner" else "constant",
            "token_loss_warmup_epochs": args.token_loss_warmup_epochs if args.model_family == "token_refiner" else 0,
            "token_loss_ramp_epochs": args.token_loss_ramp_epochs if args.model_family == "token_refiner" else 0,
            "token_loss_target_weight": args.token_loss_target_weight if args.model_family == "token_refiner" else 0.0,
            "checkpoint_contains_fm_backbone": False if args.model_family in {"fm_vae", "token_refiner"} else True,
            "fm_perceptual": bool(args.fm_perceptual) if args.model_family == "fm_vae" else False,
            "fm_loss_weight": args.fm_loss_weight if args.model_family == "fm_vae" and args.fm_perceptual else 0.0,
            "fm_cosine_mix": args.fm_cosine_mix if args.model_family == "fm_vae" else 0.0,
            "use_decoder_conditioning": bool(args.fm_decoder_conditioning) if args.model_family == "fm_vae" else False,
            "fm_cond_drop_prob": args.fm_cond_drop_prob if args.model_family == "fm_vae" else 0.0,
            "use_latent_alignment": bool(args.fm_latent_align) if args.model_family == "fm_vae" else False,
            "latent_align_weight": args.latent_align_weight if args.model_family == "fm_vae" else 0.0,
            "fm_multi_scale_align": bool(args.fm_multi_scale_align) if args.model_family == "fm_vae" else False,
            "multi_scale_align_weight": args.multi_scale_align_weight if args.model_family == "fm_vae" else 0.0,
            "architecture_version": architecture_version(args),
            "mask_aware_encoder": bool(args.mask_aware_encoder) if args.model_family == "fm_vae" else False,
            "split_latent": bool(args.split_latent) if args.model_family == "fm_vae" else False,
            "global_latent_channels": args.global_latent_channels if args.model_family == "fm_vae" else 0,
            "local_latent_channels": args.local_latent_channels if args.model_family == "fm_vae" else 0,
            "alitok_patch_size": args.alitok_patch_size if args.model_family in {"alitok", "alitok_stage2", "alitok_hybrid"} else None,
            "alitok_token_size": args.alitok_token_size if args.model_family in {"alitok", "alitok_stage2", "alitok_hybrid"} else None,
            "alitok_prefix_tokens": args.alitok_prefix_tokens if args.model_family in {"alitok", "alitok_stage2", "alitok_hybrid"} else None,
            "alitok_codebook_size": args.alitok_codebook_size if args.model_family in {"alitok", "alitok_stage2", "alitok_hybrid"} else None,
            "alitok_encoder_depth": args.alitok_encoder_depth if args.model_family in {"alitok", "alitok_stage2", "alitok_hybrid"} else None,
            "alitok_decoder_depth": args.alitok_decoder_depth if args.model_family in {"alitok", "alitok_stage2", "alitok_hybrid"} else None,
            "alitok_encoder_width": args.alitok_encoder_width if args.model_family in {"alitok", "alitok_stage2", "alitok_hybrid"} else None,
            "alitok_decoder_width": args.alitok_decoder_width if args.model_family in {"alitok", "alitok_stage2", "alitok_hybrid"} else None,
            "alitok_encoder_heads": args.alitok_encoder_heads if args.model_family in {"alitok", "alitok_stage2", "alitok_hybrid"} else None,
            "alitok_decoder_heads": args.alitok_decoder_heads if args.model_family in {"alitok", "alitok_stage2", "alitok_hybrid"} else None,
            "alitok_heads": args.alitok_heads if args.model_family in {"alitok", "alitok_stage2", "alitok_hybrid"} else None,
            "alitok_stage2_buffer_tokens": args.alitok_stage2_buffer_tokens if args.model_family in {"alitok", "alitok_stage2", "alitok_hybrid"} else None,
            "alitok_clustering_vq": bool(args.alitok_clustering_vq) if args.model_family in {"alitok", "alitok_stage2", "alitok_hybrid"} else None,
            "alitok_stage2_mix": args.alitok_stage2_mix if args.model_family == "alitok_hybrid" else None,
            "fm_features_active": fm_features_active(args),
            "comparison_protocol": "wear_ecg_exact_regime",
            "val_metrics_snapshot": {
                "val/r2_reg_v4_v6_mean": float(metrics.get("val/r2_reg_v4_v6_mean", float("nan"))),
                "val/rmse_reg": float(metrics.get("val/rmse_reg", float("nan"))),
            },
        }
        if args.save_training_state:
            save_dict["optimizer_state_dict"] = optimizer.state_dict()
            save_dict["scheduler_state_dict"] = scheduler.state_dict()

        prune_epoch_checkpoints(save_dir, keep_latest=args.keep_epoch_checkpoints)
        torch.save(save_dict, tmp_ckpt_path)
        os.replace(tmp_ckpt_path, ckpt_path)
        print(f"Saved: {ckpt_path}")

        if is_best:
            best_ckpt_path = os.path.join(save_dir, "ul_ecp_best.pt")
            if os.path.exists(best_ckpt_path):
                os.remove(best_ckpt_path)
            try:
                # Avoid duplicate multi-GB writes by linking best -> epoch ckpt.
                os.link(ckpt_path, best_ckpt_path)
            except OSError:
                best_tmp_ckpt_path = best_ckpt_path + ".tmp"
                torch.save(save_dict, best_tmp_ckpt_path)
                os.replace(best_tmp_ckpt_path, best_ckpt_path)
            write_best_summary(save_dir, current_selector, metrics, epoch + 1)
            print(f"Updated best checkpoint: {best_ckpt_path} | selector={current_selector}")

        if wandb_enabled:
            safe_wandb_log(metrics, step=step_counter, prefix="val")

        prune_epoch_checkpoints(save_dir, keep_latest=args.keep_epoch_checkpoints)

        if args.debug or args.debug_logs:
            print(
                "[EpochEnd] "
                f"epoch={epoch + 1} "
                f"best_selector={best_selector} "
                f"lr={optimizer.param_groups[0]['lr']:.8f}"
            )

    if wandb_enabled:
        try:
            wandb.finish()
        except Exception as exc:
            print(f"[W&B] finish failed: {exc}")

    print(f"Training complete: epochs={args.epochs} final_step={step_counter} save_dir={save_dir}")


if __name__ == "__main__":
    train(get_args())
