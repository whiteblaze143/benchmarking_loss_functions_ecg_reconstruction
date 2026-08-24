"""Unified evaluator for engineering reconstruction checkpoints."""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
sys.path.insert(0, '/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/unified_latents/engineering')

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.getcwd())

from src.reconstruction.evaluate_functions.fiducials import FS_HZ, compute_reconstruction_morphology_metrics
from src.reconstruction.unified_latents.engineering.common import (
    CHEST_INDICES_ALL,
    CHEST_LEADS,
    LATERAL_LEADS,
    RECON_FOCUS_LEADS,
    TensorFolderDataset,
    V3_V6_LEADS,
    V4_V6_LEADS,
    compute_batch_corr_per_lead,
    compute_batch_mae,
    compute_batch_mae_per_lead,
    compute_batch_mse,
    compute_batch_mse_per_lead,
    compute_batch_r2,
    compute_batch_r2_per_lead,
    compute_batch_rmse,
    compute_batch_rmse_per_lead,
    compute_rwave_progression_metrics,
    mask_unobserved_leads,
    mean_for_leads,
    to_serializable_metrics,
    write_json,
    write_run_artifacts,
)
from src.reconstruction.unified_latents.engineering.regimes import (
    LEAD_NAMES,
    format_lead_set,
    get_missing_indices,
    make_lead_indices,
    resolve_obs_leads,
)


def _autocast_context(device):
    if device.type != "cuda":
        return contextlib.nullcontext()
    return torch.amp.autocast("cuda", dtype=torch.bfloat16)


def _checkpoint_state_dict(path: str):
    ckpt = torch.load(path, map_location="cpu")
    return ckpt.get("model_state_dict", ckpt)


def _checkpoint_payload(path: str):
    return torch.load(path, map_location="cpu")


def _named_metric(metrics: dict[str, object], key: str) -> float:
    value = metrics.get(key, float("nan"))
    if isinstance(value, torch.Tensor):
        return float(value.item())
    return float(value)


def _payload_or_default(payload: dict[str, object], key: str, default):
    value = payload.get(key, default)
    return default if value is None else value


def _payload_path_or_default(payload: dict[str, object], key: str, default):
    value = payload.get(key, default)
    if value is None:
        return default
    if isinstance(value, str) and value and not os.path.exists(value) and default:
        return default
    return value


_WANDB_WARNED_PREFIXES: set[str] = set()


def safe_wandb_log(metrics: dict[str, float], *, step: int | None = None, prefix: str = "") -> bool:
    try:
        import wandb
    except Exception:
        return False

    if wandb.run is None:
        return False

    try:
        wandb.log(metrics, step=step)
        return True
    except Exception as exc:
        warn_key = prefix or "default"
        if warn_key not in _WANDB_WARNED_PREFIXES:
            print(f"[W&B] log failed during {warn_key}; continuing with local artifacts ({exc})")
            _WANDB_WARNED_PREFIXES.add(warn_key)
        return False


def _load_checkpoint_state_strict(model: torch.nn.Module, state_dict: dict[str, object], model_family: str) -> None:
    model_state = model.state_dict()
    model_keys = set(model_state.keys())
    if model_family == "fm_vae":
        excluded_prefixes = ("fm_model.",)
    elif model_family == "token_refiner":
        excluded_prefixes = ("frozen_vae.", "teacher.")
    elif model_family in {"alitok", "alitok_stage1"}:
        excluded_prefixes = ("bidir_decoder.",)
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


def _print_eval_summary(
    metrics: dict[str, object],
    *,
    split: str,
    model_family: str,
    obs_indices: list[int],
    debug: bool = False,
) -> None:
    learned_target_indices = get_missing_indices(obs_indices)
    learned_target_names = [LEAD_NAMES[idx] for idx in learned_target_indices]
    print(f"\n[EvalSummary] split={split} model_family={model_family}")
    print(f"  observed={ [LEAD_NAMES[idx] for idx in obs_indices] }")
    print(f"  learned_targets={learned_target_names}")
    print(
        "  core "
        f"r2_reg={_named_metric(metrics, f'{split}/r2_regressor'):.4f} "
        f"mse={_named_metric(metrics, f'{split}/mse_reg'):.6f} "
        f"mae={_named_metric(metrics, f'{split}/mae_reg'):.6f} "
        f"rmse={_named_metric(metrics, f'{split}/rmse_reg'):.6f}"
    )
    print(
        "  losses "
        f"decoder={_named_metric(metrics, f'{split}/decoder_loss'):.6f} "
        f"kl={_named_metric(metrics, f'{split}/kl_loss'):.6f} "
        f"fm={_named_metric(metrics, f'{split}/fm_perceptual_loss'):.6f} "
        f"latent_align={_named_metric(metrics, f'{split}/latent_align_loss'):.6f}"
    )
    if f"{split}/r2_teacher_clean" in metrics:
        print(
            "  teacher "
            f"r2={_named_metric(metrics, f'{split}/r2_teacher_clean'):.4f} "
            f"mse_z={_named_metric(metrics, f'{split}/mse_z_reg'):.6f}"
        )
    print(
        "  aggregates "
        f"v4_v6={_named_metric(metrics, f'{split}/r2_reg_v4_v6_mean'):.4f} "
        f"v3_v6={_named_metric(metrics, f'{split}/r2_reg_v3_v6_mean'):.4f} "
        f"chest={_named_metric(metrics, f'{split}/r2_reg_chest_mean'):.4f} "
        f"lateral={_named_metric(metrics, f'{split}/r2_reg_lateral_mean'):.4f}"
    )
    if f"{split}/clinical_reg_recon_chest_mean_beat_corr" in metrics:
        print(
            "  morphology "
            f"beat_corr={_named_metric(metrics, f'{split}/clinical_reg_recon_chest_mean_beat_corr'):.6f} "
            f"beat_rmse={_named_metric(metrics, f'{split}/clinical_reg_recon_chest_mean_beat_rmse'):.6f} "
            f"r_peak_ms={_named_metric(metrics, f'{split}/clinical_reg_recon_chest_mean_r_peak_timing_error_ms'):.6f}"
        )
    if not debug:
        return

    for lead_name in learned_target_names:
        print(
            "    "
            f"{lead_name} | "
            f"r2={_named_metric(metrics, f'{split}/lead_r2_reg_{lead_name}'):.4f} "
            f"mae={_named_metric(metrics, f'{split}/mae_reg_{lead_name}'):.5f} "
            f"mse={_named_metric(metrics, f'{split}/mse_reg_{lead_name}'):.5f} "
            f"rmse={_named_metric(metrics, f'{split}/rmse_reg_{lead_name}'):.5f} "
            f"corr={_named_metric(metrics, f'{split}/corr_reg_{lead_name}'):.4f}"
        )
    focus_keys = [
        f"{split}/clinical_reg_recon_V4_mean_beat_corr",
        f"{split}/clinical_reg_recon_V4_mean_beat_rmse",
        f"{split}/clinical_reg_recon_V4_mean_r_peak_timing_error_ms",
        f"{split}/clinical_reg_recon_V5_mean_beat_corr",
        f"{split}/clinical_reg_recon_V5_mean_beat_rmse",
        f"{split}/clinical_reg_recon_V5_mean_r_peak_timing_error_ms",
        f"{split}/clinical_reg_recon_V6_mean_beat_corr",
        f"{split}/clinical_reg_recon_V6_mean_beat_rmse",
        f"{split}/clinical_reg_recon_V6_mean_r_peak_timing_error_ms",
    ]
    present_focus = [key for key in focus_keys if key in metrics]
    if present_focus:
        print("  focus_morphology")
        for key in present_focus:
            print(f"    {key.split('/', 1)[1]}={_named_metric(metrics, key):.6f}")


def evaluate_reconstruction(
    model,
    val_loader,
    device,
    obs_indices: list[int],
    *,
    split: str = "val",
    step: int | None = None,
    model_family: str = "hybrid",
    log_to_wandb: bool = False,
    fast_eval: bool = False,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    gen_target_indices = get_missing_indices(obs_indices)
    lead_names = [LEAD_NAMES[idx] for idx in gen_target_indices]

    recon_chest_indices = [idx for idx in gen_target_indices if idx in [8, 9, 10, 11]]
    if not recon_chest_indices:
        recon_chest_indices = [idx for idx in gen_target_indices if idx in CHEST_INDICES_ALL]

    teacher_available = model_family == "hybrid" and hasattr(model, "impute_from_teacher")
    teacher_r2_total = 0.0
    teacher_r2_leads = [0.0] * len(gen_target_indices)
    reg_r2_total = 0.0
    reg_r2_leads = [0.0] * len(gen_target_indices)
    reg_mae_leads = [0.0] * len(gen_target_indices)
    reg_mse_leads = [0.0] * len(gen_target_indices)
    reg_rmse_leads = [0.0] * len(gen_target_indices)
    reg_corr_leads = [0.0] * len(gen_target_indices)

    val_loss_decoder = 0.0
    val_loss_teacher = 0.0
    val_loss_align = 0.0
    val_loss_stft = 0.0
    val_loss_diff = 0.0
    val_loss_corr = 0.0
    val_loss_kl = 0.0
    val_loss_fm_perceptual = 0.0
    val_loss_latent_align = 0.0
    val_loss_multi_scale = 0.0

    mse_z_reg_total = 0.0
    reg_mae_total = 0.0
    reg_mse_total = 0.0
    reg_rmse_total = 0.0
    reg_batches = 0
    latent_batches = 0

    reg_morph_total = {
        "samples_with_beats": 0.0,
        "mean_beat_rmse": 0.0,
        "mean_beat_corr": 0.0,
        "mean_r_amp_error": 0.0,
        "mean_r_peak_timing_error_ms": 0.0,
    }
    reg_morph_batches = 0
    reg_progression_total = {"rwave_progression_mae": 0.0, "rwave_progression_corr": 0.0}
    reg_progression_batches = 0
    reg_progression_recon_total = {"rwave_progression_mae": 0.0, "rwave_progression_corr": 0.0}
    reg_progression_recon_batches = 0
    reg_morph_focus_total = {
        lead_name: {
            "mean_beat_rmse": 0.0,
            "mean_beat_corr": 0.0,
            "mean_r_peak_timing_error_ms": 0.0,
        }
        for lead_name, lead_idx in RECON_FOCUS_LEADS
        if lead_idx in gen_target_indices
    }
    reg_morph_focus_batches = {lead_name: 0 for lead_name, lead_idx in RECON_FOCUS_LEADS if lead_idx in gen_target_indices}
    with torch.no_grad():
        for x, y, _ in tqdm(val_loader, desc=f"Evaluate-{split}"):
            x = x.to(device, dtype=torch.float32, non_blocking=True)
            y = y.to(device, dtype=torch.float32, non_blocking=True)
            lead_indices = make_lead_indices(obs_indices, x.size(0), device)

            # Ensure inference uses only observed leads (mask missing leads)
            x_masked = mask_unobserved_leads(x, obs_indices)

            with _autocast_context(device):
                out = model(x_masked, y_full=y, lead_indices=lead_indices, mode="stage1")
                reg_out = model.impute_from_regressor(x_masked, lead_indices=lead_indices)
                teacher_out = model.impute_from_teacher(x_masked, lead_indices=lead_indices) if teacher_available else None

            if not reg_out.get("available", True):
                raise RuntimeError("Primary reconstruction path is unavailable during evaluation.")

            y_target_miss = y[:, gen_target_indices, :]
            y_pred_reg = reg_out["y_pred"]
            reg_pred_miss = y_pred_reg[:, gen_target_indices, :]
            z_reg = reg_out.get("z_latent")

            teacher_pred_miss = None
            z_teacher = None
            if teacher_out is not None:
                y_pred_teacher = teacher_out["y_pred"]
                teacher_pred_miss = y_pred_teacher[:, gen_target_indices, :]
                z_teacher = teacher_out.get("z_clean")

            # Evaluation uses autocast for speed, but downstream morphology helpers
            # expect float tensors when they leave PyTorch for NumPy-based fiducials.
            y_float = y.float()
            y_pred_reg_float = y_pred_reg.float()
            if y_float.shape[-1] != y_pred_reg_float.shape[-1]:
                if y_float.shape[-1] > y_pred_reg_float.shape[-1]:
                    y_float = y_float[..., : y_pred_reg_float.shape[-1]]
                else:
                    y_float = torch.nn.functional.pad(y_float, (0, y_pred_reg_float.shape[-1] - y_float.shape[-1]))
                y_target_miss = y_float[:, gen_target_indices, :]

            val_loss_decoder += float(out.get("decoder_loss", torch.tensor(0.0, device=device)).item())
            val_loss_teacher += float(out.get("teacher_loss", torch.tensor(0.0, device=device)).item())
            val_loss_align += float(out.get("align_loss", torch.tensor(0.0, device=device)).item())
            val_loss_stft += float(out.get("stft_loss", torch.tensor(0.0, device=device)).item())
            val_loss_diff += float(out.get("diff_loss", torch.tensor(0.0, device=device)).item())
            val_loss_corr += float(out.get("corr_loss", torch.tensor(0.0, device=device)).item())
            val_loss_kl += float(out.get("kl_loss", torch.tensor(0.0, device=device)).item())
            val_loss_fm_perceptual += float(out.get("fm_perceptual_loss", out.get("teacher_loss", torch.tensor(0.0, device=device))).item())
            val_loss_latent_align += float(out.get("latent_align_loss", out.get("align_loss", torch.tensor(0.0, device=device))).item())
            val_loss_multi_scale += float(out.get("multi_scale_align_loss", torch.tensor(0.0, device=device)).item())

            reg_r2_total += compute_batch_r2(reg_pred_miss, y_target_miss).item()
            reg_mae_total += compute_batch_mae(reg_pred_miss, y_target_miss).item()
            reg_mse_total += compute_batch_mse(reg_pred_miss, y_target_miss).item()
            reg_rmse_total += compute_batch_rmse(reg_pred_miss, y_target_miss).item()
            reg_batches += 1

            if teacher_pred_miss is not None:
                teacher_r2_total += compute_batch_r2(teacher_pred_miss, y_target_miss).item()
            if z_teacher is not None and z_reg is not None:
                mse_z_reg_total += torch.nn.functional.mse_loss(z_reg, z_teacher).item()
                latent_batches += 1

            teacher_lead_vals = compute_batch_r2_per_lead(teacher_pred_miss, y_target_miss) if teacher_pred_miss is not None else None
            reg_lead_vals = compute_batch_r2_per_lead(reg_pred_miss, y_target_miss)
            reg_mae_vals = compute_batch_mae_per_lead(reg_pred_miss, y_target_miss)
            reg_mse_vals = compute_batch_mse_per_lead(reg_pred_miss, y_target_miss)
            reg_rmse_vals = compute_batch_rmse_per_lead(reg_pred_miss, y_target_miss)
            reg_corr_vals = compute_batch_corr_per_lead(reg_pred_miss, y_target_miss)

            if not fast_eval:
                morph_metrics = compute_reconstruction_morphology_metrics(y_float, y_pred_reg_float, lead_indices=recon_chest_indices, fs=FS_HZ)
                progression_metrics_full = compute_rwave_progression_metrics(y_float, y_pred_reg_float, chest_indices=CHEST_INDICES_ALL)
                progression_metrics_recon = compute_rwave_progression_metrics(y_float, y_pred_reg_float, chest_indices=recon_chest_indices)
            else:
                morph_metrics = None
                progression_metrics_full = None
                progression_metrics_recon = None

            for i in range(len(gen_target_indices)):
                if teacher_lead_vals is not None:
                    teacher_r2_leads[i] += teacher_lead_vals[i]
                reg_r2_leads[i] += reg_lead_vals[i]
                reg_mae_leads[i] += reg_mae_vals[i]
                reg_mse_leads[i] += reg_mse_vals[i]
                reg_rmse_leads[i] += reg_rmse_vals[i]
                reg_corr_leads[i] += reg_corr_vals[i]

            if not fast_eval:
                if morph_metrics is not None and not torch.isnan(torch.tensor(morph_metrics["mean_beat_rmse"])):
                    reg_morph_batches += 1
                    for key in reg_morph_total:
                        reg_morph_total[key] += morph_metrics[key]
                if progression_metrics_full is not None and not torch.isnan(torch.tensor(progression_metrics_full["rwave_progression_mae"])):
                    reg_progression_batches += 1
                    for key in reg_progression_total:
                        reg_progression_total[key] += progression_metrics_full[key]
                if progression_metrics_recon is not None and not torch.isnan(torch.tensor(progression_metrics_recon["rwave_progression_mae"])):
                    reg_progression_recon_batches += 1
                    for key in reg_progression_recon_total:
                        reg_progression_recon_total[key] += progression_metrics_recon[key]

            if not fast_eval:
                for lead_name, lead_idx in RECON_FOCUS_LEADS:
                    if lead_idx not in gen_target_indices:
                        continue
                    per_lead_metrics = compute_reconstruction_morphology_metrics(y_float, y_pred_reg_float, lead_indices=[lead_idx], fs=FS_HZ)
                    if torch.isnan(torch.tensor(per_lead_metrics["mean_beat_rmse"])):
                        continue
                    reg_morph_focus_batches[lead_name] += 1
                    reg_morph_focus_total[lead_name]["mean_beat_rmse"] += per_lead_metrics["mean_beat_rmse"]
                    reg_morph_focus_total[lead_name]["mean_beat_corr"] += per_lead_metrics["mean_beat_corr"]
                    reg_morph_focus_total[lead_name]["mean_r_peak_timing_error_ms"] += per_lead_metrics["mean_r_peak_timing_error_ms"]

    n = max(reg_batches, 1)
    teacher_r2_total = teacher_r2_total / n if teacher_available else float("nan")
    reg_r2_total /= n
    reg_mae_total /= n
    reg_mse_total /= n
    reg_rmse_total /= n
    val_loss_decoder /= n
    val_loss_teacher /= n
    val_loss_align /= n
    val_loss_stft /= n
    val_loss_diff /= n
    val_loss_corr /= n
    val_loss_kl /= n
    val_loss_fm_perceptual /= n
    val_loss_latent_align /= n
    val_loss_multi_scale /= n
    mse_z_reg_total = mse_z_reg_total / max(latent_batches, 1) if latent_batches > 0 else float("nan")

    for i in range(len(gen_target_indices)):
        if teacher_available:
            teacher_r2_leads[i] /= n
        reg_r2_leads[i] /= n
        reg_mae_leads[i] /= n
        reg_mse_leads[i] /= n
        reg_rmse_leads[i] /= n
        reg_corr_leads[i] /= n

    metrics = {
        f"{split}/decoder_loss": val_loss_decoder,
        f"{split}/teacher_loss": val_loss_teacher,
        f"{split}/align_loss": val_loss_align,
        f"{split}/stft_loss": val_loss_stft,
        f"{split}/diff_loss": val_loss_diff,
        f"{split}/corr_loss": val_loss_corr,
        f"{split}/kl_loss": val_loss_kl,
        f"{split}/fm_perceptual_loss": val_loss_fm_perceptual,
        f"{split}/latent_align_loss": val_loss_latent_align,
        f"{split}/multi_scale_align_loss": val_loss_multi_scale,
        f"{split}/r2_regressor": reg_r2_total,
        f"{split}/mae_reg": reg_mae_total,
        f"{split}/mse_reg": reg_mse_total,
        f"{split}/rmse_reg": reg_rmse_total,
    }
    if teacher_available:
        metrics[f"{split}/r2_teacher_clean"] = teacher_r2_total
    if latent_batches > 0:
        metrics[f"{split}/mse_z_reg"] = mse_z_reg_total

    for i, lead in enumerate(lead_names):
        if teacher_available:
            metrics[f"{split}/lead_r2_teach_{lead}"] = teacher_r2_leads[i]
        metrics[f"{split}/lead_r2_reg_{lead}"] = reg_r2_leads[i]
        metrics[f"{split}/mae_reg_{lead}"] = reg_mae_leads[i]
        metrics[f"{split}/mse_reg_{lead}"] = reg_mse_leads[i]
        metrics[f"{split}/rmse_reg_{lead}"] = reg_rmse_leads[i]
        metrics[f"{split}/corr_reg_{lead}"] = reg_corr_leads[i]

    if teacher_available:
        teacher_chest_mean = mean_for_leads(teacher_r2_leads, lead_names, CHEST_LEADS)
        teacher_lateral_mean = mean_for_leads(teacher_r2_leads, lead_names, LATERAL_LEADS)
        if teacher_chest_mean is not None:
            metrics[f"{split}/r2_teach_chest_mean"] = teacher_chest_mean
        if teacher_lateral_mean is not None:
            metrics[f"{split}/r2_teach_lateral_mean"] = teacher_lateral_mean

    if not fast_eval:
        reg_chest_mean = mean_for_leads(reg_r2_leads, lead_names, CHEST_LEADS)
        reg_lateral_mean = mean_for_leads(reg_r2_leads, lead_names, LATERAL_LEADS)
        reg_v4_v6_mean = mean_for_leads(reg_r2_leads, lead_names, V4_V6_LEADS)
        reg_v3_v6_mean = mean_for_leads(reg_r2_leads, lead_names, V3_V6_LEADS)
        reg_mae_chest_mean = mean_for_leads(reg_mae_leads, lead_names, CHEST_LEADS)
        reg_rmse_chest_mean = mean_for_leads(reg_rmse_leads, lead_names, CHEST_LEADS)
        reg_corr_chest_mean = mean_for_leads(reg_corr_leads, lead_names, CHEST_LEADS)
        reg_mae_lateral_mean = mean_for_leads(reg_mae_leads, lead_names, LATERAL_LEADS)
        reg_rmse_lateral_mean = mean_for_leads(reg_rmse_leads, lead_names, LATERAL_LEADS)
        reg_corr_lateral_mean = mean_for_leads(reg_corr_leads, lead_names, LATERAL_LEADS)
    else:
        reg_chest_mean = None
        reg_lateral_mean = None
        reg_v4_v6_mean = None
        reg_v3_v6_mean = None
        reg_mae_chest_mean = None
        reg_rmse_chest_mean = None
        reg_corr_chest_mean = None
        reg_mae_lateral_mean = None
        reg_rmse_lateral_mean = None
        reg_corr_lateral_mean = None

    if reg_chest_mean is not None:
        metrics[f"{split}/r2_reg_chest_mean"] = reg_chest_mean
    if reg_lateral_mean is not None:
        metrics[f"{split}/r2_reg_lateral_mean"] = reg_lateral_mean
    if reg_v4_v6_mean is not None:
        metrics[f"{split}/r2_reg_v4_v6_mean"] = reg_v4_v6_mean
    if reg_v3_v6_mean is not None:
        metrics[f"{split}/r2_reg_v3_v6_mean"] = reg_v3_v6_mean
    if reg_mae_chest_mean is not None:
        metrics[f"{split}/mae_reg_chest_mean"] = reg_mae_chest_mean
        metrics[f"{split}/rmse_reg_chest_mean"] = reg_rmse_chest_mean
        metrics[f"{split}/corr_reg_chest_mean"] = reg_corr_chest_mean
    if reg_mae_lateral_mean is not None:
        metrics[f"{split}/mae_reg_lateral_mean"] = reg_mae_lateral_mean
        metrics[f"{split}/rmse_reg_lateral_mean"] = reg_rmse_lateral_mean
        metrics[f"{split}/corr_reg_lateral_mean"] = reg_corr_lateral_mean
    if teacher_available:
        metrics[f"{split}/gap_reg_vs_teacher"] = reg_r2_total - teacher_r2_total
        if reg_chest_mean is not None and f"{split}/r2_teach_chest_mean" in metrics:
            metrics[f"{split}/gap_reg_vs_teacher_chest_mean"] = reg_chest_mean - metrics[f"{split}/r2_teach_chest_mean"]
        if reg_lateral_mean is not None and f"{split}/r2_teach_lateral_mean" in metrics:
            metrics[f"{split}/gap_reg_vs_teacher_lateral_mean"] = reg_lateral_mean - metrics[f"{split}/r2_teach_lateral_mean"]

    if reg_morph_batches > 0:
        for key, total in reg_morph_total.items():
            metrics[f"{split}/clinical_reg_recon_chest_{key}"] = total / reg_morph_batches
    if reg_progression_batches > 0:
        for key, total in reg_progression_total.items():
            metrics[f"{split}/clinical_reg_full_chest_{key}"] = total / reg_progression_batches
    if reg_progression_recon_batches > 0:
        for key, total in reg_progression_recon_total.items():
            metrics[f"{split}/clinical_reg_recon_chest_{key}"] = total / reg_progression_recon_batches
    for lead_name, lead_totals in reg_morph_focus_total.items():
        batch_count = reg_morph_focus_batches[lead_name]
        if batch_count <= 0:
            continue
        for key, total in lead_totals.items():
            metrics[f"{split}/clinical_reg_recon_{lead_name}_{key}"] = total / batch_count

    if log_to_wandb:
        safe_wandb_log(metrics, step=step, prefix="eval")

    if was_training:
        model.train()
    return metrics


def build_eval_metadata(args, obs_indices: list[int], payload: dict[str, object] | None = None) -> dict[str, object]:
    external_status = "gated_internal_only" if (args.include_sunnybrook or args.include_ecgfounder) else "not_requested"
    metadata = {
        "family": "engineering",
        "model_family": args.model_family,
        "regime": args.regime,
        "obs_leads": [LEAD_NAMES[idx] for idx in obs_indices],
        "obs_lead_indices": obs_indices,
        "split": args.split,
        "checkpoint": args.checkpoint,
        "include_sunnybrook": bool(args.include_sunnybrook),
        "include_ecgfounder": bool(args.include_ecgfounder),
        "fast_eval": bool(args.fast_eval),
        "external_status": external_status,
    }
    if args.model_family in {"fm_vae", "token_refiner"}:
        metadata.update(
            {
                "comparison_protocol": "wear_ecg_exact_regime",
                "fm_features_active": True,
                "missing_lead_weight": (payload or {}).get("missing_lead_weight", args.missing_lead_weight),
                "checkpoint_contains_fm_backbone": False,
            }
        )
    if args.model_family == "token_refiner":
        metadata.update(
            {
                "frozen_vae_checkpoint": (payload or {}).get("frozen_vae_checkpoint"),
                "teacher_encoder": (payload or {}).get("teacher_encoder"),
                "teacher_checkpoint": (payload or {}).get("teacher_checkpoint"),
                "token_loss_weight": (payload or {}).get("token_loss_weight"),
                "token_loss_mix": (payload or {}).get("token_loss_mix"),
                "teacher_token_length": (payload or {}).get("teacher_token_length"),
                "teacher_common_token_length": (payload or {}).get("teacher_common_token_length"),
                "residual_smoothness_weight": (payload or {}).get("residual_smoothness_weight"),
                "refiner_impl": (payload or {}).get("refiner_impl", "alitok_bottleneck"),
                "token_refiner_v2": bool((payload or {}).get("token_refiner_v2", False)),
                "token_refiner_observed_conditioning": bool((payload or {}).get("token_refiner_observed_conditioning", False)),
                "token_refiner_clamp_observed": bool((payload or {}).get("token_refiner_clamp_observed", False)),
                "token_refiner_causal_alignment": bool((payload or {}).get("token_refiner_causal_alignment", False)),
                "token_refiner_prefix_tokens": (payload or {}).get("token_refiner_prefix_tokens", 0),
                "token_refiner_causal_loss_weight": (payload or {}).get("token_refiner_causal_loss_weight", 0.0),
                "token_refiner_prefix_aux_loss_weight": (payload or {}).get("token_refiner_prefix_aux_loss_weight", 0.0),
                "token_refiner_stage": (payload or {}).get("token_refiner_stage"),
                "token_whitening": bool((payload or {}).get("token_whitening", False)),
                "teacher_layer_mode": (payload or {}).get("teacher_layer_mode", "last"),
                "token_improvement_margin_weight": (payload or {}).get("token_improvement_margin_weight", 0.0),
            }
        )
    if args.model_family in {"alitok", "alitok_stage1", "alitok_stage2", "alitok_hybrid"}:
        metadata.update(
            {
                "comparison_protocol": "wear_ecg_exact_regime",
                "fm_features_active": False,
                "missing_lead_weight": (payload or {}).get("missing_lead_weight", args.missing_lead_weight),
                "architecture_version": (payload or {}).get("architecture_version"),
                "alitok_patch_size": (payload or {}).get("alitok_patch_size"),
                "alitok_token_size": (payload or {}).get("alitok_token_size"),
                "alitok_prefix_tokens": (payload or {}).get("alitok_prefix_tokens"),
                "alitok_codebook_size": (payload or {}).get("alitok_codebook_size"),
                "alitok_encoder_depth": (payload or {}).get("alitok_encoder_depth"),
                "alitok_decoder_depth": (payload or {}).get("alitok_decoder_depth"),
                "alitok_encoder_width": (payload or {}).get("alitok_encoder_width"),
                "alitok_decoder_width": (payload or {}).get("alitok_decoder_width"),
                "alitok_encoder_heads": (payload or {}).get("alitok_encoder_heads"),
                "alitok_decoder_heads": (payload or {}).get("alitok_decoder_heads"),
                "alitok_heads": (payload or {}).get("alitok_heads"),
                "alitok_stage2_buffer_tokens": (payload or {}).get("alitok_stage2_buffer_tokens"),
                "alitok_clustering_vq": (payload or {}).get("alitok_clustering_vq"),
                "alitok_stage2_mix": (payload or {}).get("alitok_stage2_mix"),
            }
        )
    return metadata


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--model_family",
        type=str,
        choices=[
            "hybrid",
            "wearecg_vae",
            "fm_vae",
            "token_refiner",
            "alitok",
            "alitok_stage1",
            "alitok_stage2",
            "alitok_hybrid",
        ],
        required=True,
    )
    parser.add_argument("--regime", type=str, choices=["current", "wearecg", "historical"], default="current")
    parser.add_argument("--obs_leads", type=str, default=None)
    parser.add_argument("--split", type=str, choices=["val", "test"], default="val")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--target_len", type=int, default=5000)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--latent_channels", type=int, default=64)
    parser.add_argument("--missing_lead_weight", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--fm_checkpoint", type=str, default="ecg_fm_integration/checkpoints/mimic_iv_ecg_physionet_pretrained.pt")
    parser.add_argument("--frozen_vae_checkpoint", type=str, default=None)
    parser.add_argument(
        "--teacher_encoder",
        type=str,
        choices=["ecgfm", "hubert", "random_ecgfm", "random_hubert", "random_ecgfm_arch", "random_hubert_arch"],
        default="ecgfm",
    )
    parser.add_argument("--teacher_checkpoint", type=str, default=None)
    parser.add_argument("--teacher_dim", type=int, default=768)
    parser.add_argument("--teacher_token_length", type=int, default=None)
    parser.add_argument("--token_loss_weight", type=float, default=0.05)
    parser.add_argument("--token_loss_mix", type=float, default=0.5)
    parser.add_argument("--teacher_common_token_length", type=int, default=625)
    parser.add_argument("--residual_smoothness_weight", type=float, default=1e-4)
    parser.add_argument("--refiner_dim", type=int, default=256)
    parser.add_argument("--refiner_query_len", type=int, default=625)
    parser.add_argument("--token_refiner_causal_alignment", action="store_true", default=False)
    parser.add_argument("--token_refiner_prefix_tokens", type=int, default=16)
    parser.add_argument("--token_refiner_causal_loss_weight", type=float, default=0.05)
    parser.add_argument("--token_refiner_prefix_aux_loss_weight", type=float, default=0.1)
    parser.add_argument("--token_refiner_stage", choices=["causal_align", "bidir_refine"], default="causal_align")
    parser.add_argument("--alitok_patch_size", type=int, default=10)
    parser.add_argument("--alitok_token_size", type=int, default=32)
    parser.add_argument("--alitok_prefix_tokens", type=int, default=17)
    parser.add_argument("--alitok_codebook_size", type=int, default=4096)
    parser.add_argument("--alitok_encoder_depth", type=int, default=12)
    parser.add_argument("--alitok_decoder_depth", type=int, default=24)
    parser.add_argument("--alitok_encoder_width", type=int, default=768)
    parser.add_argument("--alitok_decoder_width", type=int, default=1024)
    parser.add_argument("--alitok_encoder_heads", type=int, default=12)
    parser.add_argument("--alitok_decoder_heads", type=int, default=16)
    parser.add_argument("--alitok_heads", type=int, default=None)
    parser.add_argument("--alitok_stage2_buffer_tokens", type=int, default=32)
    parser.add_argument("--alitok_clustering_vq", action="store_true", default=True)
    parser.add_argument("--alitok_stage2_mix", type=float, default=0.35)
    parser.add_argument("--include_sunnybrook", action="store_true")
    parser.add_argument("--include_ecgfounder", action="store_true")
    parser.add_argument("--fast_eval", action="store_true", help="Skip CPU-heavy morphology metrics.")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.model_family == "token_refiner":
        from src.reconstruction.unified_latents.engineering.token_refiner import resolve_teacher_token_length

        args.teacher_token_length = resolve_teacher_token_length(args.teacher_encoder, args.teacher_token_length)
    return args


def load_model(args, device):
    payload = _checkpoint_payload(args.checkpoint)
    state_dict = payload.get("model_state_dict", payload)
    if args.model_family == "hybrid":
        from src.reconstruction.unified_latents.engineering.ul_ecg import UL_ConditionalBridge

        fm_checkpoint = payload.get("fm_checkpoint", args.fm_checkpoint)
        model = UL_ConditionalBridge(
            checkpoint_path=fm_checkpoint,
            freeze_backbone=True,
            target_len=payload.get("target_len", args.target_len),
            teacher_loss_weight=payload.get("teacher_loss_weight", 0.10),
            reg_loss_weight=payload.get("reg_loss_weight", 1.0),
            align_loss_weight=payload.get("align_loss_weight", 0.5),
            finetune_mode=payload.get("finetune_mode", "anchored"),
            use_fm_perceptual=True,
            fm_perceptual_weight=payload.get("fm_perceptual_weight", 0.10),
        )
        model.freeze_teacher_encoder = bool(payload.get("freeze_teacher_encoder", False))
        if model.freeze_teacher_encoder:
            for p in model.encoder.parameters():
                p.requires_grad = False
            model.encoder.eval()
    elif args.model_family == "wearecg_vae":
        from src.reconstruction.unified_latents.engineering.vae_fm import WearECGVAE

        model = WearECGVAE(
            latent_channels=payload.get("latent_channels", args.latent_channels),
            target_len=payload.get("target_len", args.target_len),
            beta_kl=payload.get("beta_kl", 1e-4),
            missing_lead_weight=payload.get("missing_lead_weight", args.missing_lead_weight),
        )
        if "encoder_state_dict" in payload and "decoder_state_dict" in payload:
            model.encoder.load_state_dict(payload["encoder_state_dict"], strict=True)
            model.decoder.load_state_dict(payload["decoder_state_dict"], strict=True)
            return model.to(device, dtype=torch.float32)
    elif args.model_family == "fm_vae":
        from src.reconstruction.unified_latents.engineering.vae_fm import WearECGFMVAE

        model = WearECGFMVAE(
            fm_checkpoint_path=payload.get("fm_checkpoint", args.fm_checkpoint),
            fm_teacher_encoder=payload.get("teacher_encoder", args.teacher_encoder),
            teacher_checkpoint=payload.get("teacher_checkpoint", args.teacher_checkpoint),
            teacher_dim=payload.get("teacher_dim", args.teacher_dim),
            teacher_token_length=payload.get("teacher_token_length", args.teacher_token_length),
            teacher_common_token_length=payload.get("teacher_common_token_length", args.teacher_common_token_length),
            teacher_layer_mode=payload.get("teacher_layer_mode", args.teacher_layer_mode),
            random_teacher_seed=payload.get("random_teacher_seed", 1234),
            latent_channels=payload.get("latent_channels", 4),
            target_len=payload.get("target_len", args.target_len),
            beta_kl=payload.get("beta_kl", 1e-4),
            missing_lead_weight=payload.get("missing_lead_weight", args.missing_lead_weight),
            fm_loss_weight=payload.get("fm_loss_weight", 1e-2),
            fm_cosine_mix=payload.get("fm_cosine_mix", 0.5),
            use_decoder_conditioning=payload.get("use_decoder_conditioning", False),
            fm_cond_drop_prob=payload.get("fm_cond_drop_prob", 0.0),
            use_latent_alignment=payload.get("use_latent_alignment", False),
            latent_align_weight=payload.get("latent_align_weight", 1e-3),
            use_multi_scale_align=payload.get("fm_multi_scale_align", False),
            multi_scale_align_weight=payload.get("multi_scale_align_weight", 1e-1),
            mask_aware_encoder=payload.get("mask_aware_encoder", True),
            split_latent=payload.get("split_latent", True),
            global_latent_channels=payload.get("global_latent_channels", 2),
            local_latent_channels=payload.get("local_latent_channels", 2),
        )
    elif args.model_family == "token_refiner":
        from src.reconstruction.unified_latents.engineering.token_refiner import WearECGTokenRefiner

        model = WearECGTokenRefiner(
            frozen_vae_checkpoint=_payload_path_or_default(payload, "frozen_vae_checkpoint", args.frozen_vae_checkpoint),
            teacher_encoder=payload.get("teacher_encoder", args.teacher_encoder),
            teacher_checkpoint=payload.get("teacher_checkpoint", args.teacher_checkpoint),
            target_len=payload.get("target_len", args.target_len),
            beta_kl=payload.get("beta_kl", 1e-4),
            missing_lead_weight=payload.get("missing_lead_weight", args.missing_lead_weight),
            token_loss_weight=payload.get("token_loss_weight", args.token_loss_weight),
            token_loss_mix=payload.get("token_loss_mix", args.token_loss_mix),
            residual_smoothness_weight=payload.get("residual_smoothness_weight", args.residual_smoothness_weight),
            teacher_dim=payload.get("teacher_dim", args.teacher_dim),
            teacher_token_length=payload.get("teacher_token_length", args.teacher_token_length),
            teacher_common_token_length=payload.get("teacher_common_token_length", args.teacher_common_token_length),
            refiner_dim=payload.get("refiner_dim", args.refiner_dim),
            query_len=payload.get("refiner_query_len", args.refiner_query_len),
            random_seed=payload.get("random_teacher_seed", 1234),
            use_observed_conditioning=payload.get("token_refiner_observed_conditioning", False),
            clamp_observed_output=payload.get("token_refiner_clamp_observed", False),
            token_improvement_margin_weight=payload.get("token_improvement_margin_weight", 0.0),
            token_improvement_margin=payload.get("token_improvement_margin", 0.0),
            teacher_layer_mode=payload.get("teacher_layer_mode", "last"),
            token_whitening=payload.get("token_whitening", False),
            causal_alignment=payload.get("token_refiner_causal_alignment", args.token_refiner_causal_alignment),
            prefix_tokens=payload.get("token_refiner_prefix_tokens", args.token_refiner_prefix_tokens),
            causal_loss_weight=payload.get("token_refiner_causal_loss_weight", args.token_refiner_causal_loss_weight),
            prefix_aux_loss_weight=payload.get("token_refiner_prefix_aux_loss_weight", args.token_refiner_prefix_aux_loss_weight),
            refiner_stage=payload.get("token_refiner_stage", args.token_refiner_stage),
        )
    elif args.model_family in {"alitok", "alitok_stage1", "alitok_stage2", "alitok_hybrid"}:
        from src.reconstruction.unified_latents.engineering.alitok_vae_exp import build_alitok_vae_1d

        if args.model_family in {"alitok", "alitok_stage1"}:
            architecture = "stage1_causal"
        elif args.model_family == "alitok_stage2":
            architecture = "stage2_bidir"
        else:
            architecture = "stage1_stage2_hybrid"
        model = build_alitok_vae_1d(
            architecture=architecture,
            target_len=_payload_or_default(payload, "target_len", args.target_len),
            patch_size=_payload_or_default(payload, "alitok_patch_size", args.alitok_patch_size),
            token_size=_payload_or_default(payload, "alitok_token_size", args.alitok_token_size),
            stage2_mix=_payload_or_default(payload, "alitok_stage2_mix", args.alitok_stage2_mix),
            missing_lead_weight=_payload_or_default(payload, "missing_lead_weight", args.missing_lead_weight),
            prefix_tokens=_payload_or_default(payload, "alitok_prefix_tokens", args.alitok_prefix_tokens),
            codebook_size=_payload_or_default(payload, "alitok_codebook_size", args.alitok_codebook_size),
            encoder_depth=_payload_or_default(payload, "alitok_encoder_depth", args.alitok_encoder_depth),
            decoder_depth=_payload_or_default(payload, "alitok_decoder_depth", args.alitok_decoder_depth),
            heads=_payload_or_default(payload, "alitok_heads", args.alitok_heads),
            encoder_heads=_payload_or_default(payload, "alitok_encoder_heads", args.alitok_encoder_heads),
            decoder_heads=_payload_or_default(payload, "alitok_decoder_heads", args.alitok_decoder_heads),
            encoder_width=_payload_or_default(payload, "alitok_encoder_width", args.alitok_encoder_width),
            decoder_width=_payload_or_default(payload, "alitok_decoder_width", args.alitok_decoder_width),
            stage2_buffer_tokens=_payload_or_default(payload, "alitok_stage2_buffer_tokens", args.alitok_stage2_buffer_tokens),
            clustering_vq=_payload_or_default(payload, "alitok_clustering_vq", args.alitok_clustering_vq),
        )
    _load_checkpoint_state_strict(model, state_dict, args.model_family)
    return model.to(device, dtype=torch.float32)


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_indices = resolve_obs_leads(args.regime, args.obs_leads)
    payload = _checkpoint_payload(args.checkpoint)
    base_dir = "data/ptb_xl/tensors"
    dataset = TensorFolderDataset(f"{base_dir}/{args.split}")
    if args.debug:
        dataset = torch.utils.data.Subset(dataset, range(64))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    model = load_model(args, device)
    metrics = evaluate_reconstruction(
        model,
        loader,
        device,
        obs_indices,
        split=args.split,
        model_family=args.model_family,
        fast_eval=bool(args.fast_eval),
    )

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(args.checkpoint),
        f"eval_{args.model_family}_{args.split}_{format_lead_set(obs_indices)}",
    )
    metadata = build_eval_metadata(args, obs_indices, payload)
    if args.model_family == "fm_vae":
        metadata.update(
            {
                "fm_checkpoint": payload.get("fm_checkpoint", args.fm_checkpoint),
                "teacher_encoder": payload.get("teacher_encoder", args.teacher_encoder),
                "teacher_checkpoint": payload.get("teacher_checkpoint", args.teacher_checkpoint),
                "teacher_dim": payload.get("teacher_dim", args.teacher_dim),
                "teacher_token_length": payload.get("teacher_token_length", args.teacher_token_length),
                "teacher_common_token_length": payload.get("teacher_common_token_length", args.teacher_common_token_length),
                "teacher_layer_mode": payload.get("teacher_layer_mode", args.teacher_layer_mode),
                "random_teacher_seed": payload.get("random_teacher_seed", 1234),
                "fm_perceptual": payload.get("fm_perceptual", payload.get("fm_loss_weight", 0.0) > 0.0),
                "fm_loss_weight": payload.get("fm_loss_weight", 1e-2),
                "fm_cosine_mix": payload.get("fm_cosine_mix", 0.5),
                "fm_decoder_conditioning": payload.get("use_decoder_conditioning", False),
                "fm_latent_align": payload.get("use_latent_alignment", False),
                "fm_multi_scale_align": payload.get("fm_multi_scale_align", False),
                "multi_scale_align_weight": payload.get("multi_scale_align_weight", 1e-1),
                "architecture_version": payload.get("architecture_version", "fm_vae_mask_split_v1"),
                "mask_aware_encoder": payload.get("mask_aware_encoder", True),
                "split_latent": payload.get("split_latent", True),
                "global_latent_channels": payload.get("global_latent_channels", 2),
                "local_latent_channels": payload.get("local_latent_channels", 2),
                "missing_lead_weight": payload.get("missing_lead_weight", args.missing_lead_weight),
                "checkpoint_contains_fm_backbone": False,
            }
        )
    if args.model_family == "token_refiner":
        metadata.update(
            {
                "frozen_vae_checkpoint": payload.get("frozen_vae_checkpoint", args.frozen_vae_checkpoint),
                "teacher_encoder": payload.get("teacher_encoder", args.teacher_encoder),
                "teacher_checkpoint": payload.get("teacher_checkpoint", args.teacher_checkpoint),
                "teacher_dim": payload.get("teacher_dim", args.teacher_dim),
                "teacher_token_length": payload.get("teacher_token_length", args.teacher_token_length),
                "token_loss_weight": payload.get("token_loss_weight", args.token_loss_weight),
                "token_loss_mix": payload.get("token_loss_mix", args.token_loss_mix),
                "teacher_common_token_length": payload.get("teacher_common_token_length", args.teacher_common_token_length),
                "residual_smoothness_weight": payload.get("residual_smoothness_weight", args.residual_smoothness_weight),
                "refiner_dim": payload.get("refiner_dim", args.refiner_dim),
                "refiner_query_len": payload.get("refiner_query_len", args.refiner_query_len),
                "refiner_impl": payload.get("refiner_impl", "alitok_bottleneck"),
                "checkpoint_contains_fm_backbone": False,
            }
        )
    if args.model_family in {"alitok", "alitok_stage1", "alitok_stage2", "alitok_hybrid"}:
        metadata.update(
            {
                "architecture_version": payload.get("architecture_version"),
                "missing_lead_weight": payload.get("missing_lead_weight", args.missing_lead_weight),
                "alitok_patch_size": payload.get("alitok_patch_size", args.alitok_patch_size),
                "alitok_token_size": payload.get("alitok_token_size", args.alitok_token_size),
                "alitok_prefix_tokens": payload.get("alitok_prefix_tokens", args.alitok_prefix_tokens),
                "alitok_codebook_size": payload.get("alitok_codebook_size", args.alitok_codebook_size),
                "alitok_encoder_depth": payload.get("alitok_encoder_depth", args.alitok_encoder_depth),
                "alitok_decoder_depth": payload.get("alitok_decoder_depth", args.alitok_decoder_depth),
                "alitok_encoder_width": payload.get("alitok_encoder_width", args.alitok_encoder_width),
                "alitok_decoder_width": payload.get("alitok_decoder_width", args.alitok_decoder_width),
                "alitok_encoder_heads": payload.get("alitok_encoder_heads", args.alitok_encoder_heads),
                "alitok_decoder_heads": payload.get("alitok_decoder_heads", args.alitok_decoder_heads),
                "alitok_heads": payload.get("alitok_heads", args.alitok_heads),
                "alitok_stage2_buffer_tokens": payload.get("alitok_stage2_buffer_tokens", args.alitok_stage2_buffer_tokens),
                "alitok_clustering_vq": payload.get("alitok_clustering_vq", args.alitok_clustering_vq),
                "alitok_stage2_mix": payload.get("alitok_stage2_mix", args.alitok_stage2_mix),
                "checkpoint_contains_fm_backbone": True,
            }
        )
    if args.include_sunnybrook or args.include_ecgfounder:
        print("External panels were requested but remain gated until an internal chest-metric win is promoted.")
    write_run_artifacts(output_dir, metadata, metrics)
    write_json(os.path.join(output_dir, "eval_metadata.json"), metadata)
    print(f"Saved evaluation to {output_dir}")
    _print_eval_summary(
        metrics,
        split=args.split,
        model_family=args.model_family,
        obs_indices=obs_indices,
        debug=bool(args.debug),
    )
    for key in [
        f"{args.split}/r2_reg_v4_v6_mean",
        f"{args.split}/r2_reg_v3_v6_mean",
        f"{args.split}/r2_regressor",
        f"{args.split}/r2_teacher_clean",
    ]:
        if key in metrics:
            print(f"{key}: {metrics[key]:.4f}")


if __name__ == "__main__":
    main()
