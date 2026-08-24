from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
from pathlib import Path
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from scripts.bootstrap_paths import setup_import_paths
setup_import_paths()

import numpy as np
import neurokit2 as nk
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb


from unified_latents.engineering.models.ul_ecg import UL_ConditionalBridge
from src.reconstruction.evaluate_functions.fiducials import FS_HZ

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_NAME_TO_INDEX = {name: idx for idx, name in enumerate(LEAD_NAMES)}
INDEPENDENT_INDICES = [0, 1, 6, 7, 8, 9, 10, 11]


class TensorFolderDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path):
        self.files = sorted(glob.glob(os.path.join(folder_path, "*.pt")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        t = torch.load(self.files[idx], weights_only=True)
        t = torch.clamp(t, min=-5.0, max=5.0)
        return t, t.clone(), torch.zeros(5)


def compute_batch_r2(out, tgt):
    ssr = ((out - tgt) ** 2).sum(dim=2)
    sst = ((tgt - tgt.mean(dim=2, keepdim=True)) ** 2).sum(dim=2)
    r2 = 1.0 - ssr / torch.clamp(sst, min=0.1)
    return torch.clamp(r2, min=-100.0).mean()


def compute_batch_r2_per_lead(out, tgt):
    _, num_leads, _ = out.shape
    values = []
    for i in range(num_leads):
        pred = out[:, i, :]
        true = tgt[:, i, :]
        ssr = ((pred - true) ** 2).sum(dim=1)
        sst = ((true - true.mean(dim=1, keepdim=True)) ** 2).sum(dim=1)
        r2 = 1.0 - ssr / torch.clamp(sst, min=0.1)
        values.append(torch.clamp(r2, min=-100.0).mean().item())
    return values


def compute_batch_mae(out, tgt):
    return (out - tgt).abs().mean()


def compute_batch_mae_per_lead(out, tgt):
    return (out - tgt).abs().mean(dim=(0, 2)).tolist()


def compute_batch_rmse(out, tgt):
    return torch.sqrt(F.mse_loss(out, tgt))


def compute_batch_rmse_per_lead(out, tgt):
    return torch.sqrt(((out - tgt) ** 2).mean(dim=(0, 2))).tolist()


def compute_batch_corr_per_lead(out, tgt):
    pred = out - out.mean(dim=2, keepdim=True)
    true = tgt - tgt.mean(dim=2, keepdim=True)
    numerator = (pred * true).sum(dim=2)
    denom = torch.sqrt((pred ** 2).sum(dim=2) * (true ** 2).sum(dim=2)).clamp(min=1e-8)
    corr = (numerator / denom).mean(dim=0)
    return corr.tolist()


def extract_beat_windows(signal_1d, r_peaks, fs=500, window_ms=400):
    window_samples = int(window_ms * fs / 1000)
    half_window = window_samples // 2
    beats = []
    for peak in r_peaks:
        start = peak - half_window
        end = peak + half_window
        if start >= 0 and end <= len(signal_1d):
            beats.append(signal_1d[start:end])
    return np.array(beats) if beats else None


def detect_r_peaks_neurokit(signal_1d, fs=FS_HZ):
    try:
        _, rpeaks = nk.ecg_peaks(signal_1d, sampling_rate=int(fs))
    except Exception:
        return np.array([], dtype=int)
    peaks = rpeaks.get("ECG_R_Peaks", [])
    return np.asarray(peaks, dtype=int)


def compute_beat_morphology_metrics(gt_12lead, pred_12lead, lead_indices, fs=FS_HZ):
    gt_np = gt_12lead.detach().float().cpu().numpy() if torch.is_tensor(gt_12lead) else gt_12lead
    pred_np = pred_12lead.detach().float().cpu().numpy() if torch.is_tensor(pred_12lead) else pred_12lead

    if not lead_indices:
        return {
            "samples_with_beats": 0.0,
            "mean_beat_rmse": float("nan"),
            "mean_beat_corr": float("nan"),
            "mean_r_amp_error": float("nan"),
            "mean_r_peak_timing_error_ms": float("nan"),
        }

    beat_rmse = []
    beat_corr = []
    r_amp_error = []
    r_peak_timing_error_ms = []
    samples_with_beats = 0

    for b in range(gt_np.shape[0]):
        for lead_idx in lead_indices:
            gt_lead = gt_np[b, lead_idx]
            pred_lead = pred_np[b, lead_idx]
            r_peaks = detect_r_peaks_neurokit(gt_lead, fs=fs)
            if len(r_peaks) < 2:
                continue
            gt_beats = extract_beat_windows(gt_lead, r_peaks, fs=fs)
            pred_beats = extract_beat_windows(pred_lead, r_peaks, fs=fs)
            if gt_beats is None or pred_beats is None or len(gt_beats) != len(pred_beats):
                continue
            samples_with_beats += 1
            local_radius = max(1, int(0.05 * fs))
            for peak, gt_beat, pred_beat in zip(r_peaks, gt_beats, pred_beats):
                beat_rmse.append(float(np.sqrt(np.mean((gt_beat - pred_beat) ** 2))))
                if np.std(gt_beat) > 1e-6 and np.std(pred_beat) > 1e-6:
                    beat_corr.append(float(np.corrcoef(gt_beat, pred_beat)[0, 1]))
                center = len(gt_beat) // 2
                gt_amp = abs(float(gt_beat[center]))
                pred_amp = abs(float(pred_beat[center]))
                if gt_amp > 1e-6:
                    r_amp_error.append(abs(gt_amp - pred_amp) / gt_amp)
                start = max(0, peak - local_radius)
                end = min(len(pred_lead), peak + local_radius + 1)
                pred_local_peak = start + int(np.argmax(np.abs(pred_lead[start:end])))
                r_peak_timing_error_ms.append(abs(pred_local_peak - peak) * 1000.0 / fs)

    return {
        "samples_with_beats": float(samples_with_beats),
        "mean_beat_rmse": float(np.mean(beat_rmse)) if beat_rmse else float("nan"),
        "mean_beat_corr": float(np.mean(beat_corr)) if beat_corr else float("nan"),
        "mean_r_amp_error": float(np.mean(r_amp_error)) if r_amp_error else float("nan"),
        "mean_r_peak_timing_error_ms": float(np.mean(r_peak_timing_error_ms)) if r_peak_timing_error_ms else float("nan"),
    }


def compute_rwave_progression_metrics(gt_12lead, pred_12lead, chest_indices):
    if len(chest_indices) < 2:
        return {
            "rwave_progression_mae": float("nan"),
            "rwave_progression_corr": float("nan"),
        }
    gt_chest = gt_12lead[:, chest_indices, :].abs().amax(dim=2)
    pred_chest = pred_12lead[:, chest_indices, :].abs().amax(dim=2)
    progression_mae = (pred_chest - gt_chest).abs().mean().item()
    corr_vals = []
    for gt_row, pred_row in zip(gt_chest, pred_chest):
        if gt_row.std().item() > 1e-6 and pred_row.std().item() > 1e-6:
            corr_vals.append(torch.corrcoef(torch.stack([gt_row, pred_row]))[0, 1].item())
    progression_corr = sum(corr_vals) / len(corr_vals) if corr_vals else float("nan")
    return {
        "rwave_progression_mae": progression_mae,
        "rwave_progression_corr": progression_corr,
    }


def parse_obs_leads(obs_leads_arg):
    try:
        names = [name.strip() for name in obs_leads_arg.split(",") if name.strip()]
        indices = [LEAD_NAME_TO_INDEX[name] for name in names]
    except KeyError as exc:
        raise ValueError(f"Unknown lead name: {exc.args[0]}.") from exc
    if not indices:
        raise ValueError("--obs_leads must specify at least one lead.")
    if len(indices) > 3:
        raise ValueError("Engineering trainer supports at most 3 observed leads.")
    if len(indices) != len(set(indices)):
        raise ValueError("--obs_leads contains duplicates.")
    return indices


def get_missing_indices(obs_indices):
    return [idx for idx in range(len(LEAD_NAMES)) if idx not in obs_indices]


def format_lead_set(obs_indices):
    return "-".join(LEAD_NAMES[idx] for idx in obs_indices)


def make_lead_indices(obs_indices, batch_size, device):
    return torch.tensor([obs_indices], device=device).expand(batch_size, -1)


def mean_for_leads(values, lead_names, selected_leads):
    picked = [value for value, lead in zip(values, lead_names) if lead in selected_leads]
    if not picked:
        return None
    return sum(picked) / len(picked)


def to_serializable_metrics(metrics):
    serializable = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            serializable[key] = float(value.item())
        else:
            serializable[key] = float(value)
    return serializable


def write_run_artifacts(save_dir, args, metrics):
    os.makedirs(save_dir, exist_ok=True)
    metadata = {
        "family": "engineering",
        "experiment_family": "engineering",
        "model_family": "hybrid",
        "primary_selector": "regressor_v4_v6_then_v3_v6_then_rmse_then_global_r2_then_latent_mse",
        "obs_leads": [LEAD_NAMES[idx] for idx in args.obs_lead_indices],
        "obs_lead_indices": args.obs_lead_indices,
        "num_observed_leads": len(args.obs_lead_indices),
        "lead_regime": f"{len(args.obs_lead_indices)}lead",
        "split": args.split,
        "recon_finetune": args.recon_finetune,
        "has_regressor_path": True,
        "has_diffusion_rollout": False,
        "has_direct_prior": False,
        "teacher_target": "full_ecg_reconstructive_latent_with_fm_aux",
        "latent_dim": 32,
        "dependent_limb_leads_derived_analytically": True,
        "target_len": args.target_len,
        "teacher_loss_weight": args.teacher_loss_weight,
        "reg_loss_weight": args.reg_loss_weight,
        "align_loss_weight": args.align_loss_weight,
        "fm_perceptual_weight": args.fm_perceptual_weight,
        "finetune_mode": args.finetune_mode,
        "fm_checkpoint": args.fm_checkpoint,
        "run_tag": args.run_tag,
    }
    serializable_metrics = to_serializable_metrics(metrics)
    with open(os.path.join(save_dir, "run_metadata.json"), "w", encoding="ascii") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=True)
    with open(os.path.join(save_dir, "latest_metrics.json"), "w", encoding="ascii") as fh:
        json.dump(serializable_metrics, fh, indent=2, sort_keys=True)
    with open(os.path.join(save_dir, "latest_metrics.csv"), "w", encoding="ascii", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        for key in sorted(serializable_metrics):
            writer.writerow([key, serializable_metrics[key]])


def get_selector_tuple(metrics):
    return (
        metrics.get("val/r2_reg_v4_v6_mean", float("-inf")),
        metrics.get("val/r2_reg_v3_v6_mean", float("-inf")),
        -metrics.get("val/rmse_reg", float("inf")),
        metrics.get("val/r2_regressor", float("-inf")),
        -metrics.get("val/mse_z_reg", float("inf")),
    )


def write_best_summary(save_dir, score_tuple, metrics, epoch):
    payload = {
        "epoch": int(epoch),
        "selector_tuple": list(score_tuple),
        "metrics": to_serializable_metrics(metrics),
    }
    with open(os.path.join(save_dir, "best_summary.json"), "w", encoding="ascii") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def prune_epoch_checkpoints(save_dir, keep_latest=2):
    all_ckpts = sorted(glob.glob(os.path.join(save_dir, "ul_ecp_ep*.pt")), key=os.path.getmtime)
    if len(all_ckpts) <= keep_latest:
        return
    for old_ckpt in all_ckpts[:-keep_latest]:
        try:
            os.remove(old_ckpt)
        except OSError:
            pass


def cleanup_partial_checkpoints(save_dir):
    for tmp_path in glob.glob(os.path.join(save_dir, "*.tmp")):
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def load_compatible_model_state(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    source_state = ckpt["model_state_dict"]
    target_state = model.state_dict()
    compatible_state = {}
    skipped = []
    for key, value in source_state.items():
        if key not in target_state or target_state[key].shape != value.shape:
            skipped.append(key)
            continue
        compatible_state[key] = value
    missing, unexpected = model.load_state_dict(compatible_state, strict=False)
    return ckpt, skipped, missing, unexpected


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--target_len", type=int, default=5000)
    parser.add_argument("--loss_factor", type=float, default=1.5)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--stage1_ckpt", type=str, default=None)
    parser.add_argument("--obs_leads", type=str, default="I,II,V2")
    parser.add_argument("--split", type=str, choices=["val", "test"], default="val")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_logs", action="store_true")
    parser.add_argument("--recon_finetune", action="store_true")
    parser.add_argument("--teacher_loss_weight", type=float, default=0.10)
    parser.add_argument("--reg_loss_weight", type=float, default=1.0)
    parser.add_argument("--align_loss_weight", type=float, default=0.5)
    parser.add_argument("--fm_perceptual_weight", type=float, default=0.10)
    parser.add_argument("--accumulate_grad_batches", type=int, default=4)
    parser.add_argument("--save_training_state", action="store_true")
    parser.add_argument("--finetune_mode", type=str, choices=["anchored", "strict"], default="anchored")
    parser.add_argument("--fm_checkpoint", type=str, default="ecg_fm_integration/checkpoints/mimic_iv_ecg_physionet_pretrained.pt")
    parser.add_argument("--run_tag", type=str, default="")
    args = parser.parse_args()
    args.obs_lead_indices = parse_obs_leads(args.obs_leads)
    return args


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return total, trainable, frozen


def print_debug_banner(args, device, train_dataset, val_dataset, save_dir, model, skipped=None):
    total, trainable, frozen = count_parameters(model)
    print("\n[Debug] Training configuration")
    print(f"  device={device}")
    print(f"  obs_leads={[LEAD_NAMES[idx] for idx in args.obs_lead_indices]} ({args.obs_lead_indices})")
    print(f"  split={args.split} | target_len={args.target_len}")
    print(f"  recon_finetune={args.recon_finetune} | finetune_mode={args.finetune_mode}")
    print(
        "  loss_weights="
        f"(teacher={args.teacher_loss_weight}, reg={args.reg_loss_weight}, "
        f"align={args.align_loss_weight}, fm_perc={args.fm_perceptual_weight})"
    )
    print(
        f"  dataset_sizes=(train={len(train_dataset)}, val={len(val_dataset)}) | "
        f"batch_size={args.batch_size} | grad_accum={args.accumulate_grad_batches}"
    )
    print(f"  save_dir={save_dir}")
    print(f"  fm_checkpoint={args.fm_checkpoint}")
    print(f"  params(total={total:,}, trainable={trainable:,}, frozen={frozen:,})")
    if model.get_skip_scale() is not None:
        print(f"  initial_skip_scale={model.get_skip_scale().item():.6f}")
    if skipped is not None:
        print(f"  warm_start_skipped_tensors={len(skipped)}")


def evaluate(model, val_loader, device, step, args):
    model.eval()
    obs_indices = args.obs_lead_indices
    gen_target_indices = get_missing_indices(obs_indices)
    learned_target_indices = [idx for idx in gen_target_indices if idx in INDEPENDENT_INDICES]
    lead_names = [LEAD_NAMES[idx] for idx in gen_target_indices]
    learned_target_names = [LEAD_NAMES[idx] for idx in learned_target_indices]

    chest_leads = {"V1", "V2", "V3", "V4", "V5", "V6"}
    lateral_leads = {"V4", "V5", "V6"}
    v4_v6_leads = {"V4", "V5", "V6"}
    v3_v6_leads = {"V3", "V4", "V5", "V6"}
    chest_indices_all = [6, 7, 8, 9, 10, 11]
    recon_chest_indices = [idx for idx in gen_target_indices if idx in [8, 9, 10, 11]]
    if not recon_chest_indices:
        recon_chest_indices = [idx for idx in gen_target_indices if idx in chest_indices_all]

    teacher_r2_total = 0.0
    reg_r2_total = 0.0
    teacher_r2_leads = [0.0] * len(gen_target_indices)
    reg_r2_leads = [0.0] * len(gen_target_indices)
    reg_mae_leads = [0.0] * len(gen_target_indices)
    reg_rmse_leads = [0.0] * len(gen_target_indices)
    reg_corr_leads = [0.0] * len(gen_target_indices)

    val_loss_decoder = 0.0
    val_loss_teacher = 0.0
    val_loss_align = 0.0
    val_loss_stft = 0.0
    val_loss_diff = 0.0
    val_loss_corr = 0.0
    val_loss_progress = 0.0
    val_loss_fm_perc = 0.0
    val_loss_alg = 0.0

    mse_z_reg_total = 0.0
    reg_mae_total = 0.0
    reg_rmse_total = 0.0
    reg_batches = 0

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

    with torch.no_grad():
        for x, y, _ in tqdm(val_loader, desc="Validation"):
            x = x.to(device, dtype=torch.float32, non_blocking=True)
            y = y.to(device, dtype=torch.float32, non_blocking=True)
            lead_indices = make_lead_indices(obs_indices, x.size(0), device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(x, lead_indices=lead_indices, mode="stage1")
                teacher_out = model.impute_from_teacher(x, lead_indices=lead_indices)
                reg_out = model.impute_from_regressor(x, lead_indices=lead_indices)

            y_target_miss = y[:, gen_target_indices, :]
            y_pred_teacher = teacher_out["y_pred"]
            y_pred_reg = reg_out["y_pred"]
            z_teacher = teacher_out["z_clean"]
            z_reg = reg_out["z_latent"]

            teacher_pred_miss = y_pred_teacher[:, gen_target_indices, :]
            reg_pred_miss = y_pred_reg[:, gen_target_indices, :]

            val_loss_decoder += out["decoder_loss"].item()
            val_loss_teacher += out["teacher_loss"].item()
            val_loss_align += out["align_loss"].item()
            val_loss_stft += out["stft_loss"].item()
            val_loss_diff += out["diff_loss"].item()
            val_loss_corr += out["corr_loss"].item()
            val_loss_progress += out.get("progress_loss", torch.tensor(0.0)).item()
            val_loss_fm_perc += out.get("fm_perceptual_loss", torch.tensor(0.0)).item()
            val_loss_alg += out.get("alg_loss", torch.tensor(0.0)).item()

            teacher_r2_total += compute_batch_r2(teacher_pred_miss, y_target_miss).item()
            reg_r2_total += compute_batch_r2(reg_pred_miss, y_target_miss).item()
            mse_z_reg_total += F.mse_loss(z_reg, z_teacher).item()
            reg_mae_total += compute_batch_mae(reg_pred_miss, y_target_miss).item()
            reg_rmse_total += compute_batch_rmse(reg_pred_miss, y_target_miss).item()
            reg_batches += 1

            teacher_lead_vals = compute_batch_r2_per_lead(teacher_pred_miss, y_target_miss)
            reg_lead_vals = compute_batch_r2_per_lead(reg_pred_miss, y_target_miss)
            reg_mae_vals = compute_batch_mae_per_lead(reg_pred_miss, y_target_miss)
            reg_rmse_vals = compute_batch_rmse_per_lead(reg_pred_miss, y_target_miss)
            reg_corr_vals = compute_batch_corr_per_lead(reg_pred_miss, y_target_miss)
            morph_metrics = compute_beat_morphology_metrics(y, y_pred_reg, lead_indices=recon_chest_indices)
            progression_metrics_full = compute_rwave_progression_metrics(y, y_pred_reg, chest_indices=chest_indices_all)
            progression_metrics_recon = compute_rwave_progression_metrics(y, y_pred_reg, chest_indices=recon_chest_indices)

            for i in range(len(gen_target_indices)):
                teacher_r2_leads[i] += teacher_lead_vals[i]
                reg_r2_leads[i] += reg_lead_vals[i]
                reg_mae_leads[i] += reg_mae_vals[i]
                reg_rmse_leads[i] += reg_rmse_vals[i]
                reg_corr_leads[i] += reg_corr_vals[i]

            if not math.isnan(morph_metrics["mean_beat_rmse"]):
                reg_morph_batches += 1
                for key in reg_morph_total:
                    reg_morph_total[key] += morph_metrics[key]
            if not math.isnan(progression_metrics_full["rwave_progression_mae"]):
                reg_progression_batches += 1
                for key in reg_progression_total:
                    reg_progression_total[key] += progression_metrics_full[key]
            if not math.isnan(progression_metrics_recon["rwave_progression_mae"]):
                reg_progression_recon_batches += 1
                for key in reg_progression_recon_total:
                    reg_progression_recon_total[key] += progression_metrics_recon[key]

    n = max(reg_batches, 1)
    teacher_r2_total /= n
    reg_r2_total /= n
    mse_z_reg_total /= n
    reg_mae_total /= n
    reg_rmse_total /= n
    val_loss_decoder /= n
    val_loss_teacher /= n
    val_loss_align /= n
    val_loss_stft /= n
    val_loss_diff /= n
    val_loss_corr /= n
    val_loss_progress /= n
    val_loss_fm_perc /= n
    val_loss_alg /= n

    for i in range(len(gen_target_indices)):
        teacher_r2_leads[i] /= n
        reg_r2_leads[i] /= n
        reg_mae_leads[i] /= n
        reg_rmse_leads[i] /= n
        reg_corr_leads[i] /= n

    print(f"\n[Validation] Step {step}")
    print(f"  R2 Teacher-CleanDecode: {teacher_r2_total:.4f} [decoder upper bound]")
    print(f"  R2 Regressor: {reg_r2_total:.4f} | Latent MSE: {mse_z_reg_total:.6f}")
    print(f"  Regressor MAE/RMSE: {reg_mae_total:.4f} / {reg_rmse_total:.4f}")
    print(f"  Learned targets: {learned_target_names}")

    metrics = {
        "val/decoder_loss": val_loss_decoder,
        "val/teacher_loss": val_loss_teacher,
        "val/align_loss": val_loss_align,
        "val/stft_loss": val_loss_stft,
        "val/diff_loss": val_loss_diff,
        "val/corr_loss": val_loss_corr,
        "val/progress_loss": val_loss_progress,
        "val/fm_perceptual_loss": val_loss_fm_perc,
        "val/alg_loss": val_loss_alg,
        "val/r2_teacher_clean": teacher_r2_total,
        "val/r2_regressor": reg_r2_total,
        "val/mse_z_reg": mse_z_reg_total,
        "val/mae_reg": reg_mae_total,
        "val/rmse_reg": reg_rmse_total,
    }

    for i, lead in enumerate(lead_names):
        metrics[f"val/lead_r2_teach_{lead}"] = teacher_r2_leads[i]
        metrics[f"val/lead_r2_reg_{lead}"] = reg_r2_leads[i]
        metrics[f"val/mae_reg_{lead}"] = reg_mae_leads[i]
        metrics[f"val/rmse_reg_{lead}"] = reg_rmse_leads[i]
        metrics[f"val/corr_reg_{lead}"] = reg_corr_leads[i]
        print(f"    {lead} | reg: {reg_r2_leads[i]:.4f} | teach: {teacher_r2_leads[i]:.4f}")

    teacher_chest_mean = mean_for_leads(teacher_r2_leads, lead_names, chest_leads)
    teacher_lateral_mean = mean_for_leads(teacher_r2_leads, lead_names, lateral_leads)
    reg_chest_mean = mean_for_leads(reg_r2_leads, lead_names, chest_leads)
    reg_lateral_mean = mean_for_leads(reg_r2_leads, lead_names, lateral_leads)
    reg_v4_v6_mean = mean_for_leads(reg_r2_leads, lead_names, v4_v6_leads)
    reg_v3_v6_mean = mean_for_leads(reg_r2_leads, lead_names, v3_v6_leads)
    reg_mae_chest_mean = mean_for_leads(reg_mae_leads, lead_names, chest_leads)
    reg_rmse_chest_mean = mean_for_leads(reg_rmse_leads, lead_names, chest_leads)
    reg_corr_chest_mean = mean_for_leads(reg_corr_leads, lead_names, chest_leads)
    reg_mae_lateral_mean = mean_for_leads(reg_mae_leads, lead_names, lateral_leads)
    reg_rmse_lateral_mean = mean_for_leads(reg_rmse_leads, lead_names, lateral_leads)
    reg_corr_lateral_mean = mean_for_leads(reg_corr_leads, lead_names, lateral_leads)

    if teacher_chest_mean is not None:
        metrics["val/r2_teach_chest_mean"] = teacher_chest_mean
    if teacher_lateral_mean is not None:
        metrics["val/r2_teach_lateral_mean"] = teacher_lateral_mean
    if reg_chest_mean is not None:
        metrics["val/r2_reg_chest_mean"] = reg_chest_mean
        metrics["val/gap_reg_vs_teacher_chest_mean"] = reg_chest_mean - teacher_chest_mean
    if reg_lateral_mean is not None:
        metrics["val/r2_reg_lateral_mean"] = reg_lateral_mean
        metrics["val/gap_reg_vs_teacher_lateral_mean"] = reg_lateral_mean - teacher_lateral_mean
    if reg_v4_v6_mean is not None:
        metrics["val/r2_reg_v4_v6_mean"] = reg_v4_v6_mean
    if reg_v3_v6_mean is not None:
        metrics["val/r2_reg_v3_v6_mean"] = reg_v3_v6_mean
    if reg_mae_chest_mean is not None:
        metrics["val/mae_reg_chest_mean"] = reg_mae_chest_mean
        metrics["val/rmse_reg_chest_mean"] = reg_rmse_chest_mean
        metrics["val/corr_reg_chest_mean"] = reg_corr_chest_mean
    if reg_mae_lateral_mean is not None:
        metrics["val/mae_reg_lateral_mean"] = reg_mae_lateral_mean
        metrics["val/rmse_reg_lateral_mean"] = reg_rmse_lateral_mean
        metrics["val/corr_reg_lateral_mean"] = reg_corr_lateral_mean
    metrics["val/gap_reg_vs_teacher"] = reg_r2_total - teacher_r2_total

    if reg_morph_batches > 0:
        for key, total in reg_morph_total.items():
            metrics[f"val/clinical_reg_recon_chest_{key}"] = total / reg_morph_batches
    if reg_progression_batches > 0:
        for key, total in reg_progression_total.items():
            metrics[f"val/clinical_reg_full_chest_{key}"] = total / reg_progression_batches
    if reg_progression_recon_batches > 0:
        for key, total in reg_progression_recon_total.items():
            metrics[f"val/clinical_reg_recon_chest_{key}"] = total / reg_progression_recon_batches

    print(
        "[ValidationSummary] "
        f"step={step} "
        f"r2_reg={metrics.get('val/r2_regressor', float('nan')):.4f} "
        f"r2_teach={metrics.get('val/r2_teacher_clean', float('nan')):.4f} "
        f"v4_v6={metrics.get('val/r2_reg_v4_v6_mean', float('nan')):.4f} "
        f"v3_v6={metrics.get('val/r2_reg_v3_v6_mean', float('nan')):.4f} "
        f"rmse={metrics.get('val/rmse_reg', float('nan')):.4f} "
        f"mse_z={metrics.get('val/mse_z_reg', float('nan')):.4f}"
    )

    if wandb.run is not None:
        wandb.log(metrics, step=step)

    model.train()
    return metrics


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lead_tag = format_lead_set(args.obs_lead_indices)
    run_kind = "recon_finetune" if args.recon_finetune else "recon_train"

    wandb.init(
        project="cNVAE-ECG-Scientific-Production",
        name=(
            f"ul_engineering_{run_kind}_{lead_tag}_bs{args.batch_size}_lf{args.loss_factor}"
            + (f"_{args.run_tag}" if args.run_tag else "")
        ),
        config=vars(args),
    )

    base_dir = "data/ptb_xl/tensors"
    train_dataset = TensorFolderDataset(f"{base_dir}/train")
    val_dataset = TensorFolderDataset(f"{base_dir}/{args.split}")
    if args.debug:
        train_dataset = torch.utils.data.Subset(train_dataset, range(64))
        val_dataset = torch.utils.data.Subset(val_dataset, range(64))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = UL_ConditionalBridge(
        checkpoint_path=args.fm_checkpoint,
        freeze_backbone=True,
        target_len=args.target_len,
        teacher_loss_weight=args.teacher_loss_weight,
        reg_loss_weight=args.reg_loss_weight,
        align_loss_weight=args.align_loss_weight,
        finetune_mode=args.finetune_mode,
        use_fm_perceptual=True,
        fm_perceptual_weight=args.fm_perceptual_weight,
    ).to(device, dtype=torch.float32)

    skipped = None
    if args.stage1_ckpt:
        print(f"Loading checkpoint weights from {args.stage1_ckpt}")
        _, skipped, _, _ = load_compatible_model_state(model, args.stage1_ckpt, device)
        if skipped:
            print(f"Skipped {len(skipped)} incompatible tensors while loading warm-start weights.")

    model.recon_finetune = bool(args.recon_finetune)

    if args.recon_finetune:
        # Strictly freeze FM backbone; there is no trainable teacher waveform encoder anymore.
        for p in model.backbone.parameters():
            p.requires_grad = False
        model.backbone.eval()

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    best_selector = None
    if args.resume:
        print(f"Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if "optimizer_state_dict" in ckpt and "scheduler_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"]
        best_selector = tuple(ckpt.get("best_selector_tuple", [])) or None

    step_counter = start_epoch * len(train_loader)
    opt_step = 0
    run_tag_suffix = f"_{args.run_tag}" if args.run_tag else ""
    save_dir = f"checkpoints/ul_ecg/engineering_{lead_tag}_{run_kind}_bs{args.batch_size}_lf{args.loss_factor}{run_tag_suffix}"
    os.makedirs(save_dir, exist_ok=True)
    cleanup_partial_checkpoints(save_dir)
    debug_first_batch_printed = False

    model.train()
    if args.debug or args.debug_logs:
        print_debug_banner(args, device, train_dataset, val_dataset, save_dir, model, skipped=skipped)
    for epoch in range(start_epoch, args.epochs):
        print(f"\n[EpochStart] epoch={epoch + 1}/{args.epochs}")
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for i, (x, _, _) in enumerate(pbar):
            x = x.to(device, dtype=torch.float32, non_blocking=True)
            lead_indices = make_lead_indices(args.obs_lead_indices, x.size(0), device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(x, lead_indices=lead_indices, mode="stage1")
                loss = outputs["loss"] / args.accumulate_grad_batches

            loss.backward()

            if (args.debug or args.debug_logs) and not debug_first_batch_printed:
                print(
                    "[DebugBatch] "
                    f"x_shape={tuple(x.shape)} x_min={x.min().item():.4f} x_max={x.max().item():.4f} "
                    f"y_pred_shape={tuple(outputs['y_pred'].shape)} "
                    f"y_teacher_shape={tuple(outputs['y_pred_teacher'].shape)} "
                    f"z_teacher_shape={tuple(outputs['z_teacher'].shape)} "
                    f"z_reg_shape={tuple(outputs['z_regressed'].shape) if outputs['z_regressed'] is not None else None}"
                )
                print(
                    "[DebugLoss] "
                    f"decoder={outputs['decoder_loss'].item():.4f} "
                    f"teacher={outputs['teacher_loss'].item():.4f} "
                    f"align={outputs['align_loss'].item():.4f} "
                    f"stft={outputs['stft_loss'].item():.4f} "
                    f"diff={outputs['diff_loss'].item():.4f} "
                    f"corr={outputs['corr_loss'].item():.4f} "
                    f"progress={outputs.get('progress_loss', torch.tensor(0.0)).item():.4f} "
                    f"fm_perc={outputs.get('fm_perceptual_loss', torch.tensor(0.0)).item():.4f} "
                    f"alg={outputs.get('alg_loss', torch.tensor(0.0)).item():.6f}"
                )

            if (i + 1) % args.accumulate_grad_batches == 0 or (i + 1) == len(train_loader):
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(grad_norm):
                    print(f"Non-finite grad_norm at step {step_counter}. Skipping batch.")
                    optimizer.zero_grad(set_to_none=True)
                else:
                    if (args.debug or args.debug_logs) and not debug_first_batch_printed:
                        print(f"[DebugGrad] grad_norm={float(grad_norm):.6f}")
                        debug_first_batch_printed = True
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    opt_step += 1

            decoder_loss_val = outputs["decoder_loss"].item()
            teacher_loss_val = outputs["teacher_loss"].item()
            align_loss_val = outputs["align_loss"].item()
            stft_loss_val = outputs["stft_loss"].item()
            diff_loss_val = outputs["diff_loss"].item()
            corr_loss_val = outputs["corr_loss"].item()
            progress_loss_val = outputs.get("progress_loss", torch.tensor(0.0)).item()
            fm_perc_loss_val = outputs.get("fm_perceptual_loss", torch.tensor(0.0)).item()
            alg_loss_val = outputs.get("alg_loss", torch.tensor(0.0)).item()

            finite_losses = (
                math.isfinite(decoder_loss_val)
                and math.isfinite(teacher_loss_val)
                and math.isfinite(align_loss_val)
                and math.isfinite(stft_loss_val)
                and math.isfinite(diff_loss_val)
                and math.isfinite(corr_loss_val)
                and math.isfinite(progress_loss_val)
                and math.isfinite(fm_perc_loss_val)
                and math.isfinite(alg_loss_val)
            )
            if not finite_losses:
                print(f"Non-finite component loss at step {step_counter}. Skipping batch.")
                step_counter += 1
                continue

            wandb.log(
                {
                    "train/decoder_loss": decoder_loss_val,
                    "train/teacher_loss": teacher_loss_val,
                    "train/align_loss": align_loss_val,
                    "train/stft_loss": stft_loss_val,
                    "train/diff_loss": diff_loss_val,
                    "train/corr_loss": corr_loss_val,
                    "train/progress_loss": progress_loss_val,
                    "train/fm_perceptual_loss": fm_perc_loss_val,
                    "train/alg_loss": alg_loss_val,
                    "train/skip_scale": model.get_skip_scale().item() if model.get_skip_scale() is not None else 0.0,
                    "train/lr": scheduler.get_last_lr()[0],
                    "step": step_counter,
                    "opt_step": opt_step,
                }
            )

            pbar.set_postfix(
                {
                    "Dec": f"{decoder_loss_val:.2f}",
                    "Teach": f"{teacher_loss_val:.2f}",
                    "Align": f"{align_loss_val:.2f}",
                    "STFT": f"{stft_loss_val:.2f}",
                }
            )
            step_counter += 1

        scheduler.step()
        print(f"[EpochEnd] epoch={epoch + 1} lr={scheduler.get_last_lr()[0]:.8f}")

        if (epoch + 1) % 5 == 0 or epoch == 0 or args.debug:
            metrics = evaluate(model, val_loader, device, step_counter, args)
            write_run_artifacts(save_dir, args, metrics)
            current_selector = get_selector_tuple(metrics)
            is_best = best_selector is None or current_selector > best_selector
            if is_best:
                best_selector = current_selector

            ckpt_path = os.path.join(save_dir, f"ul_ecp_ep{epoch + 1}.pt")
            tmp_ckpt_path = ckpt_path + ".tmp"
            state_dict_to_save = {k: v for k, v in model.state_dict().items() if not k.startswith("backbone.")}
            save_dict = {
                "epoch": epoch + 1,
                "model_state_dict": state_dict_to_save,
                "best_selector_tuple": list(best_selector) if best_selector is not None else None,
                "current_selector": list(current_selector),
                "obs_leads": args.obs_lead_indices,
                "model_family": "hybrid",
                "target_len": args.target_len,
                "teacher_target": "full_ecg_reconstructive_latent_with_fm_aux",
                "latent_dim": 32,
                "teacher_loss_weight": args.teacher_loss_weight,
                "reg_loss_weight": args.reg_loss_weight,
                "align_loss_weight": args.align_loss_weight,
                "fm_perceptual_weight": args.fm_perceptual_weight,
                "finetune_mode": args.finetune_mode,
                "fm_checkpoint": args.fm_checkpoint,
                "run_tag": args.run_tag,
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
                save_dict["best_selector_tuple"] = list(best_selector)
                torch.save(save_dict, best_tmp_ckpt_path)
                os.replace(best_tmp_ckpt_path, best_ckpt_path)
                write_best_summary(save_dir, current_selector, metrics, epoch + 1)
                print(f"Updated best checkpoint: {best_ckpt_path} | selector={current_selector}")

            prune_epoch_checkpoints(save_dir, keep_latest=2)


if __name__ == "__main__":
    train(get_args())
