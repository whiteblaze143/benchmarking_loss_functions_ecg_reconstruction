"""Train the exact WearECG baseline or the FM-assisted WearECG VAE."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from scripts.bootstrap_paths import ECG_FM_ROOT, setup_import_paths

setup_import_paths(include_fairseq=True)

from unified_latents.engineering.utils.common import (
    TensorFolderDataset,
    cleanup_partial_checkpoints,
    mask_unobserved_leads,
    prune_epoch_checkpoints,
    write_best_summary,
    write_json,
    write_run_artifacts,
    write_warm_start_summary,
)
from unified_latents.engineering.eval.eval_reconstruction import evaluate_reconstruction
from unified_latents.engineering.utils.regimes import (
    LEAD_NAMES,
    format_lead_set,
    get_missing_indices,
    make_lead_indices,
    resolve_obs_leads,
)
from unified_latents.engineering.models.vae_fm_memopt import WearECGVAE, WearECGFMVAE


DEFAULT_FM_CKPT = str(ECG_FM_ROOT / "checkpoints" / "mimic_iv_ecg_physionet_pretrained.pt")
_WANDB_WARNED_PREFIXES: set[str] = set()


def add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool) -> None:
    dest = name.replace("-", "_")
    parser.add_argument(f"--{name}", dest=dest, action="store_true")
    parser.add_argument(f"--no-{name}", dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model_family", type=str, choices=["baseline", "fm_vae"], default="fm_vae")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_lr", type=float, default=5e-5)
    parser.add_argument("--target_len", type=int, default=5000)
    parser.add_argument("--latent_channels", type=int, default=4)
    parser.add_argument("--beta_kl", type=float, default=1e-4)
    parser.add_argument("--missing_lead_weight", type=float, default=1.0)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--regime", type=str, choices=["current", "wearecg", "historical"], default="wearecg")
    parser.add_argument("--obs_leads", type=str, default=None)
    parser.add_argument("--run_tag", type=str, default="")
    parser.add_argument("--split", type=str, choices=["val"], default="val")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_logs", action="store_true")
    parser.add_argument("--save_training_state", action="store_true")
    parser.add_argument("--fm_checkpoint", type=str, default=DEFAULT_FM_CKPT)
    parser.add_argument("--fm_loss_weight", type=float, default=1e-2)
    parser.add_argument("--fm_cosine_mix", type=float, default=0.5)
    parser.add_argument("--fm_cond_drop_prob", type=float, default=0.0)
    parser.add_argument("--latent_align_weight", type=float, default=1e-3)
    add_bool_arg(parser, "fm_perceptual", True)
    add_bool_arg(parser, "fm_decoder_conditioning", False)
    add_bool_arg(parser, "fm_latent_align", False)
    args, _ = parser.parse_known_args()
    args.obs_lead_indices = resolve_obs_leads(args.regime, args.obs_leads)
    if args.latent_channels != 4:
        raise ValueError("WearECG baseline and FM-VAE preserve latent_channels=4.")
    return args


def get_baseline_selector(metrics: dict[str, float]) -> tuple[float, float, float]:
    return (
        -metrics.get("val/mse_reg", float("inf")),
        -metrics.get("val/mae_reg", float("inf")),
        -metrics.get("val/rmse_reg", float("inf")),
    )


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
            f"cond_drop={args.fm_cond_drop_prob}"
        )
    print(f"  recon_weighting=observed:1.0 missing:{args.missing_lead_weight}")
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
    return args.model_family == "fm_vae" and (
        args.fm_perceptual or args.fm_decoder_conditioning or args.fm_latent_align
    )


def build_run_metadata(args: argparse.Namespace) -> dict[str, Any]:
    model_family = "fm_vae" if args.model_family == "fm_vae" else "wearecg_vae"
    return {
        "family": "engineering",
        "experiment_family": "engineering",
        "model_family": model_family,
        "comparison_protocol": "wear_ecg_exact_regime",
        "baseline_semantics": "wear_ecg_public_exact_modules",
        "primary_selector": "lowest_val_mse_then_lowest_val_mae_then_lowest_val_rmse",
        "regime": args.regime,
        "run_tag": args.run_tag,
        "obs_leads": [LEAD_NAMES[idx] for idx in args.obs_lead_indices],
        "obs_lead_indices": args.obs_lead_indices,
        "num_observed_leads": len(args.obs_lead_indices),
        "lead_regime": f"{len(args.obs_lead_indices)}lead",
        "split": args.split,
        "beta_kl": args.beta_kl,
        "missing_lead_weight": args.missing_lead_weight,
        "latent_channels": 4,
        "target_len": args.target_len,
        "fm_checkpoint": args.fm_checkpoint if args.model_family == "fm_vae" else None,
        "checkpoint_contains_fm_backbone": False if args.model_family == "fm_vae" else True,
        "fm_loss_weight": args.fm_loss_weight if args.model_family == "fm_vae" else 0.0,
        "fm_cosine_mix": args.fm_cosine_mix if args.model_family == "fm_vae" else 0.0,
        "fm_features_active": fm_features_active(args),
        "fm_perceptual": bool(args.fm_perceptual) if args.model_family == "fm_vae" else False,
        "fm_decoder_conditioning": bool(args.fm_decoder_conditioning) if args.model_family == "fm_vae" else False,
        "fm_cond_drop_prob": args.fm_cond_drop_prob if args.model_family == "fm_vae" else 0.0,
        "fm_latent_align": bool(args.fm_latent_align) if args.model_family == "fm_vae" else False,
        "latent_align_weight": args.latent_align_weight if args.model_family == "fm_vae" else 0.0,
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
    else:
        model = WearECGFMVAE(
            fm_checkpoint_path=args.fm_checkpoint,
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
        )
    return model.to(device, dtype=torch.float32)


def _load_checkpoint_state_strict(model: torch.nn.Module, state_dict: dict[str, Any], model_family: str) -> None:
    model_keys = set(model.state_dict().keys())
    if model_family == "fm_vae":
        excluded_prefixes = ("fm_model.backbone.",)
        required_model_keys = {key for key in model_keys if not key.startswith(excluded_prefixes)}
    else:
        excluded_prefixes = ()
        required_model_keys = model_keys

    provided_keys = set(state_dict.keys())
    missing_keys = sorted(required_model_keys - provided_keys)
    unexpected_keys = sorted(provided_keys - model_keys)
    shape_mismatches = sorted(
        key for key in (required_model_keys & provided_keys) if model.state_dict()[key].shape != state_dict[key].shape
    )
    if missing_keys or unexpected_keys or shape_mismatches:
        raise RuntimeError(
            "Checkpoint/model mismatch. "
            f"missing={missing_keys[:12]} unexpected={unexpected_keys[:12]} "
            f"shape_mismatches={shape_mismatches[:12]}"
        )

    incompatible = model.load_state_dict(state_dict, strict=False)
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
    if args.model_family != "fm_vae":
        return state
    return {key: value for key, value in state.items() if not key.startswith("fm_model.backbone.")}


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lead_tag = format_lead_set(args.obs_lead_indices)
    model_family_tag = "fm_vae" if args.model_family == "fm_vae" else "baseline"
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
    train_dataset = TensorFolderDataset(f"{base_dir}/train")
    val_dataset = TensorFolderDataset(f"{base_dir}/{args.split}")
    if args.debug:
        train_dataset = torch.utils.data.Subset(train_dataset, range(64))
        val_dataset = torch.utils.data.Subset(val_dataset, range(64))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    save_dir = f"checkpoints/wearecg_fm/engineering_{model_family_tag}_{lead_tag}_bs{args.batch_size}{name_suffix}"
    os.makedirs(save_dir, exist_ok=True)
    cleanup_partial_checkpoints(save_dir)

    model = build_model(args, device)
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
        if "optimizer_state_dict" in ckpt and "scheduler_state_dict" in ckpt:
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

    if args.debug or args.debug_logs:
        _print_debug_banner(args, device, train_dataset, val_dataset, save_dir, model)
        if warm_start_summary is not None:
            _print_warm_start_summary(warm_start_summary)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        if args.debug or args.debug_logs:
            print(f"\n[EpochStart] epoch={epoch + 1}/{args.epochs} lr={optimizer.param_groups[0]['lr']:.8f}")
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for x, y, _meta in pbar:
            x = x.to(device, dtype=torch.float32, non_blocking=True)
            y = y.to(device, dtype=torch.float32, non_blocking=True)
            lead_indices = make_lead_indices(args.obs_lead_indices, x.size(0), device)

            # Mask out missing leads so the encoder only sees observed channels.
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
                        "train/fm_perceptual_loss": fm_loss_val,
                        "train/latent_align_loss": latent_align_val,
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
                    "FM": f"{fm_loss_val:.4f}",
                }
            )
            step_counter += 1

        eval_model = build_model(args, device)
        eval_model.load_state_dict(model.state_dict(), strict=True)
        eval_model = eval_model.to(device, dtype=torch.float32)
        metrics = evaluate_reconstruction(
            eval_model,
            val_loader,
            device,
            args.obs_lead_indices,
            split="val",
            step=step_counter,
            model_family="fm_vae" if args.model_family == "fm_vae" else "wearecg_vae",
            log_to_wandb=False,
        )
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
            "model_family": "fm_vae" if args.model_family == "fm_vae" else "wearecg_vae",
            "run_tag": args.run_tag,
            "obs_leads": args.obs_lead_indices,
            "regime": args.regime,
            "beta_kl": args.beta_kl,
            "missing_lead_weight": args.missing_lead_weight,
            "latent_channels": 4,
            "target_len": args.target_len,
            "fm_checkpoint": args.fm_checkpoint if args.model_family == "fm_vae" else None,
            "checkpoint_contains_fm_backbone": False if args.model_family == "fm_vae" else True,
            "fm_perceptual": bool(args.fm_perceptual) if args.model_family == "fm_vae" else False,
            "fm_loss_weight": args.fm_loss_weight if args.model_family == "fm_vae" and args.fm_perceptual else 0.0,
            "fm_cosine_mix": args.fm_cosine_mix if args.model_family == "fm_vae" else 0.0,
            "use_decoder_conditioning": bool(args.fm_decoder_conditioning) if args.model_family == "fm_vae" else False,
            "fm_cond_drop_prob": args.fm_cond_drop_prob if args.model_family == "fm_vae" else 0.0,
            "use_latent_alignment": bool(args.fm_latent_align) if args.model_family == "fm_vae" else False,
            "latent_align_weight": args.latent_align_weight if args.model_family == "fm_vae" else 0.0,
            "fm_features_active": fm_features_active(args),
            "comparison_protocol": "wear_ecg_exact_regime",
        }
        if args.save_training_state:
            save_dict["optimizer_state_dict"] = optimizer.state_dict()
            save_dict["scheduler_state_dict"] = scheduler.state_dict()

        prune_epoch_checkpoints(save_dir, keep_latest=2)
        torch.save(save_dict, tmp_ckpt_path)
        os.replace(tmp_ckpt_path, ckpt_path)
        print(f"Saved: {ckpt_path}")

        if is_best:
            best_ckpt_path = os.path.join(save_dir, "ul_ecp_best.pt")
            best_tmp_ckpt_path = best_ckpt_path + ".tmp"
            torch.save(save_dict, best_tmp_ckpt_path)
            os.replace(best_tmp_ckpt_path, best_ckpt_path)
            write_best_summary(save_dir, current_selector, metrics, epoch + 1)
            print(f"Updated best checkpoint: {best_ckpt_path} | selector={current_selector}")

        if wandb_enabled:
            safe_wandb_log(metrics, step=step_counter, prefix="val")

        prune_epoch_checkpoints(save_dir, keep_latest=2)

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


if __name__ == "__main__":
    train(get_args())
import sys
from pathlib import Path
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from scripts.bootstrap_paths import setup_import_paths
setup_import_paths()
