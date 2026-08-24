"""Zero-shot smartwatch ECG generalization evaluator.

This script evaluates reconstruction checkpoints on the PhysioNet
"ECG-Capable Smartwatches" dataset. It treats smartwatch records as a
single observed wearable lead and Philips TC30 as the 12-lead reference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import wfdb

import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from scripts.bootstrap_paths import setup_import_paths
setup_import_paths()

from unified_latents.engineering.eval.eval_reconstruction import load_model
from unified_latents.engineering.utils.regimes import LEAD_NAME_TO_INDEX, LEAD_NAMES, make_lead_indices


DEFAULT_ZIP = Path(
    "/home/mithunmanivannan/"
    "electrocardiogram-capable-smartwatches-assessing-their-clinical-accuracy-and-application-1.0.0.zip"
)
DEFAULT_DATA_ROOT = Path("/home/mithunmanivannan/data/physionet_smartwatches/ecg-capable-smartwatches-1.0.0")
DEFAULT_FROZEN_VAE = (
    "/home/mithunmanivannan/checkpoints/ul_ecg/"
    "engineering_wearecg_exact_II-V1-V5_bs64_lf1.5_ep10_canonical_baseline/ul_ecp_best.pt"
)
ARCHIVE_TOP = "electrocardiogram-capable-smartwatches-assessing-their-clinical-accuracy-and-application-1.0.0"
REFERENCE_DEVICE = "philips_tc30"
WEARABLE_DEVICES = ("applewatch_serie8", "fitbitsense2", "samsunggalaxy6", "withingsscanwatch")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    checkpoint: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/smartwatch_generalization"))
    parser.add_argument(
        "--model",
        action="append",
        nargs=3,
        metavar=("NAME", "MODEL_FAMILY", "CHECKPOINT"),
        help="Model to evaluate. Repeatable. Example: --model v2 token_refiner checkpoints/.../ul_ecp_best.pt",
    )
    parser.add_argument("--target-len", type=int, default=5000)
    parser.add_argument("--crop-seconds", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--calibration-records-per-device", type=int, default=32)
    parser.add_argument("--plot-examples", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def prepare_dataset(zip_path: Path, data_root: Path) -> Path:
    if (data_root / "RECORDS").exists():
        return data_root
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing PhysioNet smartwatch zip: {zip_path}")
    extract_parent = data_root.parent
    extract_parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_parent)
    extracted = extract_parent / ARCHIVE_TOP
    if extracted != data_root:
        if data_root.exists():
            shutil.rmtree(data_root)
        extracted.rename(data_root)
    return data_root


def read_records(data_root: Path) -> list[str]:
    records_path = data_root / "RECORDS"
    with records_path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def build_pair_manifest(data_root: Path, max_records: int | None = None) -> list[dict[str, str]]:
    records = read_records(data_root)
    by_device: dict[str, set[str]] = {}
    for record in records:
        device, rel = record.split("/", 1)
        by_device.setdefault(device, set()).add(rel)
    reference_rels = by_device.get(REFERENCE_DEVICE, set())
    rows: list[dict[str, str]] = []
    for device in WEARABLE_DEVICES:
        for rel in sorted(by_device.get(device, set()) & reference_rels):
            test_family = rel.split("/", 1)[0]
            rows.append(
                {
                    "device": device,
                    "test_family": test_family,
                    "test_name": rel.split("/")[1] if "/" in rel else "",
                    "record_id": rel.rsplit("/", 1)[-1],
                    "relative_record": rel,
                    "wearable_record": str(data_root / device / rel),
                    "reference_record": str(data_root / REFERENCE_DEVICE / rel),
                }
            )
    if max_records is not None:
        rows = rows[: max(0, int(max_records))]
    return rows


def load_record(record_base: str) -> tuple[np.ndarray, dict[str, Any]]:
    signal, fields = wfdb.rdsamp(record_base)
    signal = np.asarray(signal, dtype=np.float32).T
    return signal, fields


def crop_center_seconds(signal: np.ndarray, fields: dict[str, Any], seconds: float | None) -> np.ndarray:
    if seconds is None or seconds <= 0:
        return signal
    fs = float(fields.get("fs", 500))
    target = int(round(float(seconds) * fs))
    if target <= 0 or signal.shape[-1] == target:
        return signal
    if signal.shape[-1] > target:
        start = max(0, (signal.shape[-1] - target) // 2)
        return signal[..., start : start + target]
    pad = target - signal.shape[-1]
    left = pad // 2
    return np.pad(signal, ((0, 0), (left, pad - left)), mode="edge")


def resample_1d(x: np.ndarray, target_len: int) -> np.ndarray:
    t = torch.from_numpy(np.asarray(x, dtype=np.float32)).view(1, 1, -1)
    y = F.interpolate(t, size=target_len, mode="linear", align_corners=False)
    return y.view(-1).numpy()


def resample_2d(x: np.ndarray, target_len: int) -> np.ndarray:
    t = torch.from_numpy(np.asarray(x, dtype=np.float32)).unsqueeze(0)
    y = F.interpolate(t, size=target_len, mode="linear", align_corners=False)
    return y.squeeze(0).numpy()


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x - float(np.mean(x))) / max(float(np.std(x)), 1e-6)


def best_lag_and_corr(source: np.ndarray, target: np.ndarray, max_lag: int | None = None) -> tuple[int, float]:
    a = zscore(source)
    b = zscore(target)
    if max_lag is None:
        max_lag = min(len(a) // 2, 1500)
    best_lag = 0
    best_corr = 0.0
    best_abs_corr = -float("inf")
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            aa = a[-lag:]
            bb = b[: len(aa)]
        elif lag > 0:
            aa = a[:-lag]
            bb = b[lag:]
        else:
            aa = a
            bb = b
        if len(aa) < 64:
            continue
        if np.std(aa) <= 1e-8 or np.std(bb) <= 1e-8:
            corr = 0.0
        else:
            corr = float(np.corrcoef(aa, bb)[0, 1])
        if abs(corr) > best_abs_corr:
            best_lag, best_corr = lag, corr
            best_abs_corr = abs(corr)
    return best_lag, best_corr


def shift_signal(x: np.ndarray, lag: int) -> np.ndarray:
    y = np.zeros_like(x)
    if lag < 0:
        y[:lag] = x[-lag:]
    elif lag > 0:
        y[lag:] = x[:-lag]
    else:
        y = x.copy()
    return y


def waveform_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    err = pred - target
    mse = float(np.mean(err**2))
    mae = float(np.mean(np.abs(err)))
    rmse = float(math.sqrt(mse))
    denom = float(np.sum((target - np.mean(target)) ** 2))
    r2 = float(1.0 - np.sum(err**2) / max(denom, 1e-8))
    corr = float(np.corrcoef(pred, target)[0, 1]) if np.std(pred) > 1e-8 and np.std(target) > 1e-8 else 0.0
    return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2, "corr": corr}


def find_r_peaks(x: np.ndarray, fs: int = 500) -> np.ndarray:
    from scipy.signal import find_peaks

    sig = zscore(x)
    distance = max(1, int(0.22 * fs))
    prominence = max(0.3, float(np.std(sig)) * 0.5)
    peaks, _ = find_peaks(sig, distance=distance, prominence=prominence)
    return peaks.astype(int)


def morphology_metrics(pred: np.ndarray, target: np.ndarray, *, fs: int = 500, family: str = "") -> dict[str, float]:
    peaks = find_r_peaks(target, fs=fs)
    if len(peaks) == 0:
        return {
            "r_amp_mae": float("nan"),
            "hr_error_bpm": float("nan"),
            "st_j_error_mv": float("nan"),
            "square_corr": float("nan"),
        }
    safe_peaks = peaks[(peaks >= 0) & (peaks < len(target))]
    r_amp_mae = float(np.mean(np.abs(pred[safe_peaks] - target[safe_peaks]))) if len(safe_peaks) else float("nan")
    if len(peaks) >= 2:
        ref_hr = 60.0 * fs / max(float(np.median(np.diff(peaks))), 1.0)
        pred_peaks = find_r_peaks(pred, fs=fs)
        pred_hr = 60.0 * fs / max(float(np.median(np.diff(pred_peaks))), 1.0) if len(pred_peaks) >= 2 else float("nan")
        hr_error = abs(pred_hr - ref_hr) if math.isfinite(pred_hr) else float("nan")
    else:
        hr_error = float("nan")
    j_offset = int(0.08 * fs)
    baseline_offset = int(0.16 * fs)
    st_errors = []
    for peak in safe_peaks:
        j = peak + j_offset
        base = peak - baseline_offset
        if 0 <= base < len(target) and 0 <= j < len(target):
            ref_st = target[j] - target[base]
            pred_st = pred[j] - pred[base]
            st_errors.append(abs(pred_st - ref_st))
    st_error = float(np.mean(st_errors)) if st_errors else float("nan")
    square_corr = waveform_metrics(pred, target)["corr"] if family == "sqr-2hz" else float("nan")
    return {
        "r_amp_mae": r_amp_mae,
        "hr_error_bpm": float(hr_error),
        "st_j_error_mv": st_error,
        "square_corr": square_corr,
    }


def calibrate_devices(
    rows: list[dict[str, str]],
    target_len: int,
    limit_per_device: int,
    crop_seconds: float,
) -> dict[str, dict[str, Any]]:
    calibration: dict[str, dict[str, Any]] = {}
    for device in WEARABLE_DEVICES:
        candidates = {"I": [], "II": []}
        device_rows = [row for row in rows if row["device"] == device][:limit_per_device]
        for row in device_rows:
            wearable, wearable_fields = load_record(row["wearable_record"])
            reference, fields = load_record(row["reference_record"])
            wearable = crop_center_seconds(wearable, wearable_fields, crop_seconds)
            reference = crop_center_seconds(reference, fields, crop_seconds)
            wearable_1d = resample_1d(wearable[0], target_len)
            sig_names = list(fields.get("sig_name", LEAD_NAMES))
            for lead_name in candidates:
                if lead_name not in sig_names:
                    continue
                ref = resample_1d(reference[sig_names.index(lead_name)], target_len)
                _lag, corr = best_lag_and_corr(wearable_1d, ref)
                candidates[lead_name].append(corr)
        summary = {
            lead: {
                "mean_corr": float(np.mean(vals)) if vals else 0.0,
                "mean_abs_corr": float(np.mean(np.abs(vals))) if vals else 0.0,
                "polarity": -1.0 if vals and float(np.mean(vals)) < 0 else 1.0,
                "n": len(vals),
            }
            for lead, vals in candidates.items()
        }
        chosen = max(summary, key=lambda lead: summary[lead]["mean_abs_corr"])
        calibration[device] = {
            "chosen_reference_lead": chosen,
            "chosen_reference_index": LEAD_NAME_TO_INDEX[chosen],
            "polarity": summary[chosen]["polarity"],
            "candidates": summary,
        }
    return calibration


def make_eval_args(model: ModelSpec) -> SimpleNamespace:
    return SimpleNamespace(
        checkpoint=str(model.checkpoint),
        model_family=model.family,
        target_len=5000,
        latent_dim=32,
        latent_channels=64,
        missing_lead_weight=1.0,
        fm_checkpoint="ecg_fm_integration/checkpoints/mimic_iv_ecg_physionet_pretrained.pt",
        frozen_vae_checkpoint=DEFAULT_FROZEN_VAE,
        teacher_encoder="ecgfm",
        teacher_checkpoint=None,
        teacher_dim=768,
        teacher_token_length=None,
        token_loss_weight=0.05,
        token_loss_mix=0.5,
        teacher_common_token_length=625,
        residual_smoothness_weight=1e-4,
        refiner_dim=256,
        refiner_query_len=256,
        token_refiner_causal_alignment=False,
        token_refiner_prefix_tokens=16,
        token_refiner_causal_loss_weight=0.05,
        token_refiner_prefix_aux_loss_weight=0.1,
        token_refiner_stage="causal_align",
        alitok_patch_size=10,
        alitok_token_size=32,
        alitok_prefix_tokens=17,
        alitok_codebook_size=4096,
        alitok_encoder_depth=12,
        alitok_decoder_depth=24,
        alitok_encoder_width=768,
        alitok_decoder_width=1024,
        alitok_encoder_heads=12,
        alitok_decoder_heads=16,
        alitok_heads=None,
        alitok_stage2_buffer_tokens=32,
        alitok_clustering_vq=True,
        alitok_stage2_mix=0.35,
    )


def run_model(model: torch.nn.Module, x_12: np.ndarray, obs_idx: int, device: torch.device) -> tuple[np.ndarray, np.ndarray | None]:
    x = torch.from_numpy(x_12).unsqueeze(0).to(device=device, dtype=torch.float32)
    lead_indices = make_lead_indices([obs_idx], 1, device)
    with torch.no_grad():
        out = model.impute_from_regressor(x, lead_indices=lead_indices)
    pred = out["y_pred"].detach().float().cpu().numpy()[0]
    coarse = out.get("y_coarse")
    coarse_np = coarse.detach().float().cpu().numpy()[0] if isinstance(coarse, torch.Tensor) else None
    return pred, coarse_np


def evaluate_rows(
    rows: list[dict[str, str]],
    calibration: dict[str, dict[str, Any]],
    models: list[ModelSpec],
    *,
    target_len: int,
    batch_size: int,
    device: torch.device,
    output_dir: Path,
    plot_examples: int,
    crop_seconds: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    loaded_models: list[tuple[ModelSpec, torch.nn.Module]] = []
    for spec in models:
        model = load_model(make_eval_args(spec), device)
        model.eval()
        loaded_models.append((spec, model))

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    per_record: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []

    for row_idx, row in enumerate(rows):
        wearable, wearable_fields = load_record(row["wearable_record"])
        reference, ref_fields = load_record(row["reference_record"])
        wearable = crop_center_seconds(wearable, wearable_fields, crop_seconds)
        reference = crop_center_seconds(reference, ref_fields, crop_seconds)
        reference_12 = resample_2d(reference, target_len)
        sig_names = list(ref_fields.get("sig_name", LEAD_NAMES))
        ordered_reference = np.zeros((12, target_len), dtype=np.float32)
        for lead_idx, lead_name in enumerate(LEAD_NAMES):
            if lead_name in sig_names:
                ordered_reference[lead_idx] = reference_12[sig_names.index(lead_name)]

        device_cal = calibration[row["device"]]
        obs_name = device_cal["chosen_reference_lead"]
        obs_idx = int(device_cal["chosen_reference_index"])
        wearable_1d = resample_1d(wearable[0], target_len) * float(device_cal["polarity"])
        ref_obs = ordered_reference[obs_idx]
        lag, raw_corr = best_lag_and_corr(wearable_1d, ref_obs)
        wearable_aligned = shift_signal(wearable_1d, lag)
        x_12 = np.zeros((12, target_len), dtype=np.float32)
        x_12[obs_idx] = np.clip(wearable_aligned, -5.0, 5.0)

        raw_metric = {
            **row,
            "chosen_reference_lead": obs_name,
            "chosen_reference_index": obs_idx,
            "polarity": float(device_cal["polarity"]),
            "alignment_lag_samples": lag,
            "alignment_corr": raw_corr,
        }
        raw_metric.update({f"raw_obs_{key}": value for key, value in waveform_metrics(wearable_aligned, ref_obs).items()})
        raw_metric.update({f"raw_obs_{key}": value for key, value in morphology_metrics(wearable_aligned, ref_obs, family=row["test_family"]).items()})
        raw_rows.append(raw_metric)

        for spec, model in loaded_models:
            pred, coarse = run_model(model, x_12, obs_idx, device)
            missing = [idx for idx in range(12) if idx != obs_idx]
            all_metrics = waveform_metrics(pred.reshape(-1), ordered_reference.reshape(-1))
            miss_metrics = waveform_metrics(pred[missing].reshape(-1), ordered_reference[missing].reshape(-1))
            obs_metrics = waveform_metrics(pred[obs_idx], ref_obs)
            obs_morph = morphology_metrics(pred[obs_idx], ref_obs, family=row["test_family"])
            record = {
                **row,
                "model_name": spec.name,
                "model_family": spec.family,
                "checkpoint": str(spec.checkpoint),
                "chosen_reference_lead": obs_name,
                "chosen_reference_index": obs_idx,
                "alignment_lag_samples": lag,
                "raw_alignment_corr": raw_corr,
            }
            record.update({f"all12_{key}": value for key, value in all_metrics.items()})
            record.update({f"missing11_{key}": value for key, value in miss_metrics.items()})
            record.update({f"obs_lead_{key}": value for key, value in obs_metrics.items()})
            record.update({f"obs_lead_{key}": value for key, value in obs_morph.items()})
            if coarse is not None:
                coarse_metrics = waveform_metrics(coarse[missing].reshape(-1), ordered_reference[missing].reshape(-1))
                record.update({f"coarse_missing11_{key}": value for key, value in coarse_metrics.items()})
            per_record.append(record)

            if row_idx < plot_examples:
                t = np.arange(target_len) / 500.0
                fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
                axes[0].plot(t, ref_obs, label=f"Philips {obs_name}", linewidth=1.0)
                axes[0].plot(t, wearable_aligned, label=f"{row['device']} raw aligned", linewidth=0.8, alpha=0.8)
                axes[0].plot(t, pred[obs_idx], label=f"{spec.name} {obs_name}", linewidth=0.8, alpha=0.8)
                axes[0].legend(loc="upper right", fontsize=8)
                for ax, lead_name in zip(axes[1:], ["V4", "V6"]):
                    lead_idx = LEAD_NAME_TO_INDEX[lead_name]
                    ax.plot(t, ordered_reference[lead_idx], label=f"Philips {lead_name}", linewidth=1.0)
                    ax.plot(t, pred[lead_idx], label=f"{spec.name} {lead_name}", linewidth=0.8, alpha=0.8)
                    ax.legend(loc="upper right", fontsize=8)
                axes[-1].set_xlabel("seconds")
                fig.suptitle(f"{row['device']} {row['relative_record']} | {spec.name}")
                fig.tight_layout()
                safe_name = f"{row_idx:03d}_{row['device']}_{row['record_id']}_{spec.name}.png".replace("/", "_")
                fig.savefig(plot_dir / safe_name, dpi=150)
                plt.close(fig)

    return per_record, raw_rows


def summarize(rows: list[dict[str, Any]], group_cols: list[str], metric_cols: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[col] for col in group_cols), []).append(row)
    summary = []
    for key, items in sorted(groups.items()):
        out = {col: value for col, value in zip(group_cols, key)}
        out["n_records"] = len(items)
        for metric in metric_cols:
            vals = [float(item[metric]) for item in items if metric in item and math.isfinite(float(item[metric]))]
            out[f"{metric}_mean"] = float(np.mean(vals)) if vals else float("nan")
            out[f"{metric}_std"] = float(np.std(vals)) if vals else float("nan")
        summary.append(out)
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_error_bar_plots(output_dir: Path, model_summary: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for metric in ["missing11_mse", "obs_lead_r_amp_mae", "obs_lead_st_j_error_mv", "obs_lead_hr_error_bpm"]:
        rows = [row for row in model_summary if f"{metric}_mean" in row and math.isfinite(float(row[f"{metric}_mean"]))]
        if not rows:
            continue
        labels = [f"{row['model_name']}\\n{row['device']}" for row in rows]
        means = [float(row[f"{metric}_mean"]) for row in rows]
        stds = [float(row.get(f"{metric}_std", 0.0)) for row in rows]
        fig, ax = plt.subplots(figsize=(max(8, len(rows) * 0.65), 4))
        ax.bar(np.arange(len(rows)), means, yerr=stds, capsize=3)
        ax.set_xticks(np.arange(len(rows)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(metric)
        fig.tight_layout()
        fig.savefig(plot_dir / f"device_errorbar_{metric}.png", dpi=150)
        plt.close(fig)


def parse_model_specs(args: argparse.Namespace) -> list[ModelSpec]:
    specs = []
    for item in args.model or []:
        name, family, checkpoint = item
        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"Missing checkpoint for model {name}: {path}")
        specs.append(ModelSpec(name=name, family=family, checkpoint=path))
    return specs


def main() -> None:
    args = parse_args()
    data_root = prepare_dataset(args.zip_path, args.data_root)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = build_pair_manifest(data_root, max_records=args.max_records)
    write_csv(output_dir / "paired_record_manifest.csv", rows)
    calibration = calibrate_devices(rows, args.target_len, args.calibration_records_per_device, args.crop_seconds)
    (output_dir / "device_lead_calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")

    metadata = {
        "dataset": "Electrocardiogram-Capable Smartwatches: Assessing Their Clinical Accuracy and Application",
        "physionet_doi": "10.13026/7018-y383",
        "physionet_standard_citation": "Goldberger et al. (2000), Circulation 101(23):e215-e220.",
        "data_root": str(data_root),
        "target_len": args.target_len,
        "crop_seconds": args.crop_seconds,
        "n_pairs": len(rows),
        "devices": list(WEARABLE_DEVICES),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if args.prepare_only:
        print(f"Prepared smartwatch dataset and manifest: {output_dir}")
        return

    specs = parse_model_specs(args)
    if not specs:
        print(f"No --model entries supplied. Wrote manifest/calibration only: {output_dir}")
        return

    device = torch.device(args.device)
    per_record, raw_rows = evaluate_rows(
        rows,
        calibration,
        specs,
        target_len=args.target_len,
        batch_size=args.batch_size,
        device=device,
        output_dir=output_dir,
        plot_examples=args.plot_examples,
        crop_seconds=args.crop_seconds,
    )
    write_csv(output_dir / "raw_wearable_vs_philips_metrics.csv", raw_rows)
    write_csv(output_dir / "per_record_metrics.csv", per_record)

    model_metric_cols = [
        "all12_mse",
        "all12_mae",
        "all12_rmse",
        "all12_r2",
        "all12_corr",
        "missing11_mse",
        "missing11_mae",
        "missing11_rmse",
        "missing11_r2",
        "missing11_corr",
        "obs_lead_mse",
        "obs_lead_corr",
        "obs_lead_r_amp_mae",
        "obs_lead_hr_error_bpm",
        "obs_lead_st_j_error_mv",
        "obs_lead_square_corr",
        "coarse_missing11_mse",
    ]
    raw_metric_cols = [
        "raw_obs_mse",
        "raw_obs_mae",
        "raw_obs_rmse",
        "raw_obs_r2",
        "raw_obs_corr",
        "raw_obs_r_amp_mae",
        "raw_obs_hr_error_bpm",
        "raw_obs_st_j_error_mv",
        "raw_obs_square_corr",
    ]
    device_summary = summarize(per_record, ["model_name", "device"], model_metric_cols)
    family_summary = summarize(per_record, ["model_name", "device", "test_family"], model_metric_cols)
    raw_device_summary = summarize(raw_rows, ["device"], raw_metric_cols)
    raw_family_summary = summarize(raw_rows, ["device", "test_family"], raw_metric_cols)
    write_csv(output_dir / "per_device_summary.csv", device_summary)
    write_csv(output_dir / "per_test_family_summary.csv", family_summary)
    write_csv(output_dir / "raw_per_device_summary.csv", raw_device_summary)
    write_csv(output_dir / "raw_per_test_family_summary.csv", raw_family_summary)
    write_error_bar_plots(output_dir, device_summary)
    print(f"Smartwatch generalization evaluation complete: {output_dir}")


if __name__ == "__main__":
    main()
