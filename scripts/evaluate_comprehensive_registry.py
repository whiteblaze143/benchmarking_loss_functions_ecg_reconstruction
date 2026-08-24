#!/usr/bin/env python3
"""Evaluate every registered reconstruction model under one PTB-XL protocol.

The evaluator deliberately separates observed-lead and missing-lead metrics.
It also uses ECG IDs from tensor filenames to join real diagnostic labels,
age, and sex from ptbxl_database.csv. No synthetic demographic masks or
placeholder labels are permitted.
"""

from __future__ import annotations

import argparse
import ast
import gc
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.signal import find_peaks
from sklearn.metrics import roc_auc_score
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from scripts.bootstrap_paths import setup_import_paths, ECG_FM_ROOT
setup_import_paths(include_fairseq=True)

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from scripts.clinical_metrics import (
    multilabel_classification_metrics,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def json_safe(value: Any) -> Any:
    """Convert non-finite metrics to JSON null without hiding valid zeros."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]


class TensorECGDataset(Dataset):
    def __init__(self, data_dir: Path, metadata: pd.DataFrame, max_samples: int = 0):
        self.files = sorted(data_dir.glob("*.pt"), key=lambda p: int(p.stem))
        if max_samples > 0:
            self.files = self.files[:max_samples]
        self.metadata = metadata

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        path = self.files[index]
        ecg_id = int(path.stem)
        signal = torch.load(path, weights_only=True).float()
        if signal.shape != (12, 5000):
            raise ValueError(f"Expected [12,5000] tensor at {path}, got {tuple(signal.shape)}")
        row = self.metadata.loc[ecg_id]
        return (
            signal,
            ecg_id,
            row["ecgfounder_labels"],
            row["labels"],
            float(row["age"]),
            int(row["sex"]),
        )


def _diagnostic_labels(code_string: str, statements: pd.DataFrame) -> np.ndarray:
    labels = np.zeros(len(CLASS_NAMES), dtype=np.float32)
    try:
        codes = ast.literal_eval(str(code_string))
    except (ValueError, SyntaxError):
        return labels
    for code, likelihood in codes.items():
        if float(likelihood) < 50 or code not in statements.index:
            continue
        diagnostic_class = statements.loc[code].get("diagnostic_class")
        if diagnostic_class in CLASS_NAMES:
            labels[CLASS_NAMES.index(diagnostic_class)] = 1.0
    return labels


def load_metadata(
    ptbxl_csv: Path,
    statements_csv: Path,
    ecgfounder_labels_csv: Path,
) -> pd.DataFrame:
    from scripts.ecgfounder_classifier import attach_ptbxl_labels

    frame = pd.read_csv(ptbxl_csv).set_index("ecg_id")
    statements = pd.read_csv(statements_csv, index_col=0)
    frame["labels"] = [
        _diagnostic_labels(value, statements) for value in frame["scp_codes"]
    ]
    frame["age"] = frame["age"].fillna(frame["age"].median())
    frame["sex"] = frame["sex"].fillna(0).astype(int)
    return attach_ptbxl_labels(frame, ecgfounder_labels_csv)


@dataclass
class ReconstructionAdapter:
    spec: dict[str, Any]
    model: torch.nn.Module
    device: torch.device

    @property
    def observed(self) -> list[int]:
        return [int(x) for x in self.spec.get("observed_leads", [0, 1, 7])]

    @torch.inference_mode()
    def reconstruct(
        self,
        target: torch.Tensor,
        preserve_observed: list[int] | None = None,
    ) -> torch.Tensor:
        kind = self.spec["kind"]
        target = target.to(self.device, non_blocking=True)
        if kind == "unet":
            padded = F.pad(target, (0, 120))
            output = self.model(padded[:, self.observed, :])
            reconstruction = output[..., :5000]
        elif kind == "m2":
            padded = F.pad(target, (0, 120))
            reconstruction = self.model(padded[:, self.observed, :])[:, :12, :5000]
        elif kind == "cnvae":
            x = target[:, self.observed, :].to(torch.float32)
            lead_indices = torch.tensor(
                self.observed, device=self.device, dtype=torch.long
            ).unsqueeze(0).expand(x.shape[0], -1)
            output = self.model(
                x,
                lead_indices=lead_indices,
                target_len=5000,
                use_posterior=True,
                deterministic=True,
                kl_balance=True,
            )
            reconstruction = output["recon"].float()
        elif kind in {"msvae", "alitok"}:
            from unified_latents.engineering.utils.common import mask_unobserved_leads
            from unified_latents.engineering.utils.regimes import make_lead_indices

            masked = mask_unobserved_leads(target, self.observed)
            lead_indices = make_lead_indices(self.observed, target.shape[0], self.device)
            if hasattr(self.model, "impute_from_regressor"):
                reconstruction = self.model.impute_from_regressor(
                    masked, lead_indices=lead_indices
                )["y_pred"].float()
            else:
                output = self.model(
                    masked, y_full=target, lead_indices=lead_indices, mode="stage1"
                )
                reconstruction = output["y_pred"].float()
        else:
            raise ValueError(f"Unsupported model kind: {kind}")

        # Match lengths in case the model's internal padding (e.g. patchification or downsampling)
        # resulted in a slightly longer output sequence than the input sequence.
        min_len = min(reconstruction.shape[-1], target.shape[-1])
        reconstruction = reconstruction[..., :min_len]
        target_trunc = target[..., :min_len]

        # This benchmark is an imputation task: acquired leads are known and
        # must pass through unchanged. Enforcing that contract here prevents
        # architecture-specific output behavior from biasing clinical/all-lead
        # metrics; primary missing-lead metrics are unaffected.
        passthrough = self.observed if preserve_observed is None else preserve_observed
        invalid_passthrough = sorted(set(passthrough) - set(self.observed))
        if invalid_passthrough:
            raise ValueError(
                "Passthrough leads must be part of the model input contract: "
                f"{invalid_passthrough}"
            )
        reconstruction[:, passthrough, :] = target_trunc[
            :, passthrough, :
        ].to(reconstruction.dtype)
        return reconstruction


def normalize_compiled_state_dict(state: dict[str, Any]) -> dict[str, Any]:
    """Remove the wrapper prefix emitted by ``torch.compile`` checkpoints.

    The training process may save either the compiled module state
    (``_orig_mod.*`` keys) or an already-normalized state.  Evaluation must
    accept both representations without weakening strict key/shape checking.
    """
    if not isinstance(state, dict):
        return state
    prefix = "_orig_mod."
    if not any(key.startswith(prefix) for key in state):
        return state
    normalized = {
        key[len(prefix):] if key.startswith(prefix) else key: value
        for key, value in state.items()
    }
    if len(normalized) != len(state):
        raise RuntimeError("Compiled checkpoint key normalization caused a collision")
    return normalized


def load_adapter(spec: dict[str, Any], device: torch.device) -> ReconstructionAdapter:
    kind = spec["kind"]
    model_id = spec["id"]
    
    try:
        from scripts.checkpoint_store import load_checkpoint_with_identity
        checkpoint, identity = load_checkpoint_with_identity(
            model_id=model_id,
            weights_only=False
        )
    except Exception as e:
        # Fallback to direct path
        checkpoint_path = PROJECT_ROOT / spec["checkpoint"]
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found locally or remotely for {model_id}: {e}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    kind = spec["kind"]
    if kind == "unet":
        from scripts.train_mcma_3lead import MCMAModel

        model = MCMAModel(in_channels=3, out_channels=12)
        state = checkpoint
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=True)
    elif kind == "m2":
        from scripts.train_m2_uncertainty import UncertaintyMCMAModel

        model = UncertaintyMCMAModel(in_channels=3, out_channels=12)
        state = checkpoint
        model.load_state_dict(state, strict=True)
    elif kind == "cnvae":
        from scripts.baselines.train_cnvae import get_model
        args = checkpoint["args"]
        fm_checkpoint = resolve_ecgfm_checkpoint(
            getattr(args, "fm_checkpoint", checkpoint.get("fm_checkpoint", ""))
        )
        model = get_model(
            "ecgfm",
            torch.device("cpu"),
            target_len=int(getattr(args, "target_len", 5000)),
            fm_checkpoint=str(fm_checkpoint),
        )
        incompatible = model.load_state_dict(checkpoint["bridge_state_dict"], strict=False)
        allowed_missing_prefixes = ("backbone.", "decoder.backbone.")
        disallowed_missing = [
            key
            for key in incompatible.missing_keys
            if not key.startswith(allowed_missing_prefixes)
        ]
        if disallowed_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "Invalid cNVAE checkpoint state: "
                f"missing={disallowed_missing}, unexpected={incompatible.unexpected_keys}"
            )
    elif kind in {"msvae", "alitok"}:
        from unified_latents.engineering.experimental.Multi_Scale_VAE import WearECGVAE

        if kind == "msvae":
            model = WearECGVAE(
                latent_channels=4,
                target_len=int(checkpoint.get("target_len", 5000)),
                beta_kl=float(checkpoint.get("beta_kl", 1e-4)),
                missing_lead_weight=float(checkpoint.get("missing_lead_weight", 1.0)),
            )
        else:
            from unified_latents.engineering.experimental.alitok_vae_exp import build_alitok_vae_1d

            architecture = checkpoint.get("alitok_architecture", "stage1_causal")
            aim_architecture = architecture == "ecg_aim_v1"
            model = build_alitok_vae_1d(
                architecture=architecture,
                target_len=int(checkpoint.get("target_len", 5000)),
                patch_size=int(checkpoint.get("alitok_patch_size", 25 if aim_architecture else 10)),
                token_size=int(checkpoint.get("alitok_token_size", 32)),
                prefix_tokens=int(checkpoint.get("alitok_prefix_tokens", 17)),
                codebook_size=int(checkpoint.get("alitok_codebook_size", 4096)),
                encoder_width=int(checkpoint.get("alitok_width", 768)),
                decoder_width=int(checkpoint.get("alitok_width", 768 if aim_architecture else 1024)),
                encoder_depth=int(checkpoint.get("alitok_encoder_depth", 8 if aim_architecture else 12)),
                decoder_depth=int(checkpoint.get("alitok_decoder_depth", 4 if aim_architecture else 24)),
                encoder_heads=int(checkpoint.get("alitok_heads", 12)),
                decoder_heads=int(checkpoint.get("alitok_heads", 12 if aim_architecture else 16)),
                missing_lead_weight=float(checkpoint.get("missing_lead_weight", 1.0)),
                clustering_vq=bool(checkpoint.get("alitok_clustering_vq", False)),
            )
        state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        state = normalize_compiled_state_dict(state)
        model.load_state_dict(state, strict=True)
    else:
        raise ValueError(f"Unknown kind: {kind}")
    model = model.to(device).eval()
    return ReconstructionAdapter(spec=spec, model=model, device=device)


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_ecgfm_checkpoint(value: str | Path) -> Path:
    """Resolve legacy saved ECG-FM paths into this self-contained project.

    Older cNVAE checkpoints persisted an absolute path to the former
    ``~/ecg_fm_integration`` checkout. The model weights remain valid after
    migration, so use that path when it still exists and otherwise resolve the
    same checkpoint filename from the vendored project-local integration.
    """
    saved = Path(value).expanduser()
    candidates = [saved]
    if not saved.is_absolute():
        candidates.append(PROJECT_ROOT / saved)
    candidates.append(ECG_FM_ROOT / "checkpoints" / saved.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Unable to resolve ECG-FM checkpoint saved in cNVAE metadata. "
        f"Tried: {[str(candidate) for candidate in candidates]}"
    )


class StreamingSignalMetrics:
    def __init__(self, observed: list[int]):
        self.observed = set(observed)
        self.record_count = 0
        self.point_count = 0
        self.derivative_point_count = 0
        self.sse = np.zeros(12, dtype=np.float64)
        self.sae = np.zeros(12, dtype=np.float64)
        self.target_energy = np.zeros(12, dtype=np.float64)
        self.deriv_sse = np.zeros(12, dtype=np.float64)
        self.corr_sum = np.zeros(12, dtype=np.float64)
        self.r2_sum = np.zeros(12, dtype=np.float64)
        self.samples = np.zeros(12, dtype=np.int64)

    def update(self, target: torch.Tensor, recon: torch.Tensor) -> None:
        target = target.float()
        recon = recon.float()
        error = recon - target
        self.sse += error.square().sum(dim=(0, 2)).cpu().double().numpy()
        self.sae += error.abs().sum(dim=(0, 2)).cpu().double().numpy()
        self.target_energy += target.square().sum(dim=(0, 2)).cpu().double().numpy()
        d_error = torch.diff(recon, dim=-1) - torch.diff(target, dim=-1)
        self.deriv_sse += d_error.square().sum(dim=(0, 2)).cpu().double().numpy()
        centered_t = target - target.mean(dim=-1, keepdim=True)
        centered_r = recon - recon.mean(dim=-1, keepdim=True)
        covariance = (centered_t * centered_r).sum(dim=-1)
        denominator = torch.sqrt(
            centered_t.square().sum(dim=-1) * centered_r.square().sum(dim=-1)
        ).clamp_min(1e-8)
        corr = covariance / denominator
        r2 = 1.0 - error.square().sum(dim=-1) / centered_t.square().sum(dim=-1).clamp_min(1e-8)
        finite = torch.isfinite(corr) & torch.isfinite(r2)
        self.corr_sum += torch.where(finite, corr, 0).sum(dim=0).cpu().double().numpy()
        self.r2_sum += torch.where(finite, r2, 0).sum(dim=0).cpu().double().numpy()
        self.samples += finite.sum(dim=0).cpu().numpy()
        self.record_count += int(target.shape[0])
        self.point_count += int(target.shape[0] * target.shape[-1])
        self.derivative_point_count += int(
            target.shape[0] * max(target.shape[-1] - 1, 0)
        )

    def finalize(self) -> dict[str, Any]:
        points_per_lead = max(self.point_count, 1)
        deriv_points = max(self.derivative_point_count, 1)
        mse = self.sse / points_per_lead
        mae = self.sae / points_per_lead
        rmse = np.sqrt(mse)
        corr = self.corr_sum / np.maximum(self.samples, 1)
        r2 = self.r2_sum / np.maximum(self.samples, 1)
        deriv_mse = self.deriv_sse / deriv_points
        snr = 10.0 * np.log10(
            np.maximum(self.target_energy, 1e-12) / np.maximum(self.sse, 1e-12)
        )
        missing = [idx for idx in range(12) if idx not in self.observed]
        observed = sorted(self.observed)

        def mean(values: np.ndarray, indices: list[int]) -> float:
            return float(np.mean(values[indices])) if indices else float("nan")

        return {
            "n_samples": self.record_count,
            "per_lead": {
                LEAD_NAMES[i]: {
                    "mse": float(mse[i]),
                    "rmse": float(rmse[i]),
                    "mae": float(mae[i]),
                    "pearson": float(corr[i]),
                    "r2": float(r2[i]),
                    "derivative_mse": float(deriv_mse[i]),
                    "snr_db": float(snr[i]),
                }
                for i in range(12)
            },
            "all_leads": {
                "mse": float(np.mean(mse)),
                "rmse": float(np.mean(rmse)),
                "mae": float(np.mean(mae)),
                "pearson": float(np.mean(corr)),
                "r2": float(np.mean(r2)),
                "derivative_mse": float(np.mean(deriv_mse)),
                "snr_db": float(np.mean(snr)),
            },
            "missing_leads": {
                "indices": missing,
                "names": [LEAD_NAMES[i] for i in missing],
                "mse": mean(mse, missing),
                "rmse": mean(rmse, missing),
                "mae": mean(mae, missing),
                "pearson": mean(corr, missing),
                "r2": mean(r2, missing),
                "derivative_mse": mean(deriv_mse, missing),
                "snr_db": mean(snr, missing),
            },
            "observed_leads": {
                "indices": observed,
                "names": [LEAD_NAMES[i] for i in observed],
                "mse": mean(mse, observed),
                "rmse": mean(rmse, observed),
                "mae": mean(mae, observed),
                "pearson": mean(corr, observed),
                "r2": mean(r2, observed),
            },
        }


def record_signal_metrics(
    target: torch.Tensor,
    recon: torch.Tensor,
    ecg_ids: torch.Tensor,
    observed: list[int],
    condition: str,
) -> list[dict[str, Any]]:
    """Return paired record-level metrics for bootstrap and interaction analysis."""
    missing = [index for index in range(12) if index not in observed]
    target = target[:, missing].float()
    recon = recon[:, missing].float()
    error = recon - target
    mse = error.square().mean(dim=(1, 2))
    mae = error.abs().mean(dim=(1, 2))
    derivative_mse = (
        torch.diff(recon, dim=-1) - torch.diff(target, dim=-1)
    ).square().mean(dim=(1, 2))
    centered_target = target - target.mean(dim=-1, keepdim=True)
    centered_recon = recon - recon.mean(dim=-1, keepdim=True)
    covariance = (centered_target * centered_recon).sum(dim=-1)
    denominator = torch.sqrt(
        centered_target.square().sum(dim=-1)
        * centered_recon.square().sum(dim=-1)
    ).clamp_min(1e-8)
    pearson = (covariance / denominator).mean(dim=1)
    r2 = (
        1.0
        - error.square().sum(dim=-1)
        / centered_target.square().sum(dim=-1).clamp_min(1e-8)
    ).mean(dim=1)
    snr = 10.0 * torch.log10(
        target.square().sum(dim=(1, 2)).clamp_min(1e-12)
        / error.square().sum(dim=(1, 2)).clamp_min(1e-12)
    )
    return [
        {
            "ecg_id": int(ecg_id),
            "condition": condition,
            "mse": float(mse[index]),
            "rmse": float(torch.sqrt(mse[index])),
            "mae": float(mae[index]),
            "pearson": float(pearson[index]),
            "r2": float(r2[index]),
            "snr_db": float(snr[index]),
            "derivative_mse": float(derivative_mse[index]),
        }
        for index, ecg_id in enumerate(ecg_ids.tolist())
    ]


def classification_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    return multilabel_classification_metrics(
        probs,
        labels,
        class_names,
        thresholds=0.5,
        threshold_source="fixed_predefined_0.5",
    )


def subgroup_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    ages: np.ndarray,
    sexes: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    groups = {
        # PTB-XL metadata follows the dataset convention: male=0, female=1.
        "female": sexes == 1,
        "male": sexes == 0,
        "age_lt_40": ages < 40,
        "age_40_65": (ages >= 40) & (ages <= 65),
        "age_gt_65": ages > 65,
    }
    output: dict[str, Any] = {}
    group_masks: dict[str, np.ndarray] = {}
    valid_tasks: dict[str, set[int]] = {}
    for name, mask in groups.items():
        if int(mask.sum()) < 20:
            continue
        group_masks[name] = mask
        valid_tasks[name] = {
            index
            for index in range(labels.shape[1])
            if np.unique(labels[mask, index]).size == 2
        }
        macro_auroc = float(np.mean([
            roc_auc_score(labels[mask, task], probs[mask, task])
            for task in sorted(valid_tasks[name])
        ])) if valid_tasks[name] else None
        output[name] = {
            "n": int(mask.sum()),
            "macro_auroc": macro_auroc,
            "n_independently_evaluable_tasks": len(valid_tasks[name]),
        }

    def comparable_gap(names: tuple[str, ...]) -> dict[str, Any]:
        present = [name for name in names if name in group_masks]
        if len(present) < 2:
            return {
                "value": None,
                "n_common_tasks": 0,
                "common_task_indices": [],
                "status": "insufficient_groups",
            }
        common = sorted(set.intersection(*(valid_tasks[name] for name in present)))
        if not common:
            return {
                "value": None,
                "n_common_tasks": 0,
                "common_task_indices": [],
                "status": "no_common_evaluable_tasks",
            }
        macros: dict[str, float] = {}
        for name in present:
            mask = group_masks[name]
            macros[name] = float(np.mean([
                roc_auc_score(labels[mask, task], probs[mask, task])
                for task in common
            ]))
            output[name][f"macro_auroc_common_{'_'.join(names)}"] = macros[name]
        return {
            "value": float(max(macros.values()) - min(macros.values())),
            "n_common_tasks": len(common),
            "common_task_indices": common,
            "group_macro_aurocs": macros,
            "status": "comparable_common_task_set",
        }

    gender_gap = comparable_gap(("female", "male"))
    age_gap = comparable_gap(("age_lt_40", "age_40_65", "age_gt_65"))
    output["gaps"] = {
        "gender_auroc_gap": gender_gap["value"],
        "age_auroc_gap": age_gap["value"],
        "gender_comparability": gender_gap,
        "age_comparability": age_gap,
        "sex_encoding": "PTB-XL: male=0, female=1",
    }
    return output


def morphology_metrics(
    targets: list[np.ndarray],
    recons: list[np.ndarray],
    observed: list[int],
    diagnostic_labels: list[np.ndarray] | None = None,
    fs: int = 500,
) -> dict[str, Any]:
    missing = [index for index in range(12) if index not in observed]
    regions = {
        "anterior": [index for index in (6, 7, 8, 9) if index in missing],
        "lateral": [index for index in (0, 4, 10, 11) if index in missing],
        "inferior": [index for index in (1, 2, 5) if index in missing],
    }
    bands = {"0.5_40_hz": (0.5, 40.0), "40_100_hz": (40.0, 100.0), "100_150_hz": (100.0, 150.0)}
    qrs_corrs: list[float] = []
    st_corrs: list[float] = []
    j_errors: list[float] = []
    timing_errors: list[float] = []
    waveform_corrs: list[float] = []
    region_corrs: dict[str, list[float]] = {name: [] for name in regions}
    spectral_errors: dict[str, list[float]] = {name: [] for name in bands}
    class_qrs: dict[str, list[float]] = {name: [] for name in CLASS_NAMES}
    successful_records = 0
    attempted_records = len(targets)

    def correlation(left: np.ndarray, right: np.ndarray) -> float | None:
        if left.size < 2 or np.std(left) * np.std(right) <= 1e-10:
            return None
        value = float(np.corrcoef(left.ravel(), right.ravel())[0, 1])
        return value if np.isfinite(value) else None

    for record_index, (target, recon) in enumerate(zip(targets, recons)):
        reference = target[1]
        scale = max(float(np.std(reference)), 1e-6)
        peaks, _ = find_peaks(
            reference,
            distance=int(0.25 * fs),
            prominence=0.5 * scale,
        )
        valid = peaks[(peaks >= int(0.05 * fs)) & (peaks + int(0.24 * fs) < target.shape[-1])]
        if valid.size == 0:
            continue
        successful_records += 1
        record_qrs: list[float] = []
        for lead in missing:
            truth = target[lead]
            pred = recon[lead]
            predicted_peaks, _ = find_peaks(
                pred,
                distance=int(0.25 * fs),
                prominence=0.5 * max(float(np.std(truth)), 1e-6),
            )
            if predicted_peaks.size:
                nearest = np.min(np.abs(predicted_peaks[:, None] - valid[None, :]), axis=0)
                timing_errors.extend((nearest * 1000.0 / fs).tolist())

            qrs_truth = np.concatenate([truth[p - int(0.05 * fs):p + int(0.10 * fs)] for p in valid])
            qrs_pred = np.concatenate([pred[p - int(0.05 * fs):p + int(0.10 * fs)] for p in valid])
            st_truth = np.concatenate([truth[p + int(0.12 * fs):p + int(0.24 * fs)] for p in valid])
            st_pred = np.concatenate([pred[p + int(0.12 * fs):p + int(0.24 * fs)] for p in valid])
            qrs_corr = correlation(qrs_truth, qrs_pred)
            st_corr = correlation(st_truth, st_pred)
            waveform_corr = correlation(truth, pred)
            if qrs_corr is not None:
                qrs_corrs.append(qrs_corr)
                record_qrs.append(qrs_corr)
            if st_corr is not None:
                st_corrs.append(st_corr)
            if waveform_corr is not None:
                waveform_corrs.append(waveform_corr)
            j_indices = valid + int(0.10 * fs)
            j_errors.extend(np.abs(pred[j_indices] - truth[j_indices]).tolist())

            frequencies = np.fft.rfftfreq(truth.size, d=1.0 / fs)
            truth_power = np.abs(np.fft.rfft(truth)) ** 2
            pred_power = np.abs(np.fft.rfft(pred)) ** 2
            for name, (low, high) in bands.items():
                band_mask = (frequencies >= low) & (frequencies < high)
                reference_power = max(float(truth_power[band_mask].sum()), 1e-12)
                spectral_errors[name].append(abs(float(pred_power[band_mask].sum()) - reference_power) / reference_power)

        for region, lead_indices in regions.items():
            if lead_indices:
                value = correlation(target[lead_indices], recon[lead_indices])
                if value is not None:
                    region_corrs[region].append(value)
        if diagnostic_labels is not None and record_qrs:
            for class_index, class_name in enumerate(CLASS_NAMES):
                if diagnostic_labels[record_index][class_index] > 0:
                    class_qrs[class_name].append(float(np.mean(record_qrs)))

    return {
        "protocol": {
            "qrs_window_ms": [-50, 100],
            "st_window_ms": [120, 240],
            "j_point_ms": 100,
            "sample_rate_hz": fs,
            "evaluated_leads": [LEAD_NAMES[index] for index in missing],
        },
        "detector_coverage": {
            "attempted_records": attempted_records,
            "successful_records": successful_records,
            "fraction": successful_records / attempted_records if attempted_records else float("nan"),
        },
        "qrs_correlation": float(np.mean(qrs_corrs)) if qrs_corrs else float("nan"),
        "st_correlation": float(np.mean(st_corrs)) if st_corrs else float("nan"),
        "j_point_amplitude_mae": float(np.mean(j_errors)) if j_errors else float("nan"),
        "r_peak_timing_mae_ms": float(np.mean(timing_errors)) if timing_errors else float("nan"),
        "waveform_correlation": float(np.mean(waveform_corrs)) if waveform_corrs else float("nan"),
        "n_peak_matches": len(timing_errors),
        "regional_correlation": {
            name: float(np.mean(values)) if values else float("nan")
            for name, values in region_corrs.items()
        },
        "spectral_relative_error": {
            name: float(np.mean(values)) if values else float("nan")
            for name, values in spectral_errors.items()
        },
        "class_stratified_qrs_correlation": {
            name: {"n": len(values), "mean": float(np.mean(values)) if values else float("nan")}
            for name, values in class_qrs.items()
        },
    }


def morphology_record_metrics(
    targets: list[np.ndarray],
    recons: list[np.ndarray],
    ecg_ids: list[int],
    observed: list[int],
    fs: int = 500,
) -> list[dict[str, Any]]:
    """Record-level standardized QRS/ST/J metrics for paired inference."""
    missing = [index for index in range(12) if index not in observed]
    rows: list[dict[str, Any]] = []
    for target, recon, ecg_id in zip(targets, recons, ecg_ids):
        reference = target[1]
        peaks, _ = find_peaks(
            reference,
            distance=int(0.25 * fs),
            prominence=0.5 * max(float(np.std(reference)), 1e-6),
        )
        peaks = peaks[(peaks >= int(0.05 * fs)) & (peaks + int(0.24 * fs) < target.shape[-1])]
        row = {"ecg_id": int(ecg_id), "morphology_detected": bool(peaks.size)}
        if peaks.size:
            qrs_truth = np.concatenate([
                target[lead, peak - int(0.05 * fs):peak + int(0.10 * fs)]
                for lead in missing for peak in peaks
            ])
            qrs_recon = np.concatenate([
                recon[lead, peak - int(0.05 * fs):peak + int(0.10 * fs)]
                for lead in missing for peak in peaks
            ])
            st_truth = np.concatenate([
                target[lead, peak + int(0.12 * fs):peak + int(0.24 * fs)]
                for lead in missing for peak in peaks
            ])
            st_recon = np.concatenate([
                recon[lead, peak + int(0.12 * fs):peak + int(0.24 * fs)]
                for lead in missing for peak in peaks
            ])
            row.update({
                "qrs_correlation": float(np.corrcoef(qrs_truth, qrs_recon)[0, 1]),
                "st_correlation": float(np.corrcoef(st_truth, st_recon)[0, 1]),
                "j_point_amplitude_mae": float(np.mean([
                    abs(recon[lead, peak + int(0.10 * fs)] - target[lead, peak + int(0.10 * fs)])
                    for lead in missing for peak in peaks
                ])),
            })
        else:
            row.update({"qrs_correlation": float("nan"), "st_correlation": float("nan"), "j_point_amplitude_mae": float("nan")})
        rows.append(row)
    return rows


def save_example_plot(model_id: str, target: np.ndarray, recon: np.ndarray, output_dir: Path) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{model_id}.png"
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    for axis, lead in zip(axes, (1, 6, 10)):
        axis.plot(target[lead], label="target", linewidth=0.8)
        axis.plot(recon[lead], label="reconstruction", linewidth=0.8, alpha=0.8)
        axis.set_ylabel(LEAD_NAMES[lead])
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("sample")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


@torch.inference_mode()
@torch.no_grad()
def evaluate_model(
    spec: dict[str, Any],
    loader: DataLoader,
    device: torch.device,
    classifier: torch.nn.Module | None,
    class_names: list[str],
    parity_classifier: torch.nn.Module | None,
    parity_class_names: list[str],
    stress_conditions: list[Any],
    nstdb_noises: dict[str, np.ndarray],
    robustness_samples: int,
    robustness_condition_batch: int,
    morphology_samples: int,
    plot_dir: Path,
    record_dir: Path,
) -> dict[str, Any]:
    adapter = load_adapter(spec, device)
    signal_metrics = StreamingSignalMetrics(adapter.observed)
    from scripts.ecgfounder_classifier import preprocess_ecgfounder
    from scripts.robustness_stress import apply_condition

    robust_clean_metrics = StreamingSignalMetrics(adapter.observed)
    robust_metrics = {
        condition.id: StreamingSignalMetrics(adapter.observed)
        for condition in stress_conditions
    }
    probabilities: list[np.ndarray] = []
    reference_probabilities: list[np.ndarray] = []
    parity_probabilities: list[np.ndarray] = []
    parity_labels: list[np.ndarray] = []
    parity_ids: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    ages_all: list[np.ndarray] = []
    sexes_all: list[np.ndarray] = []
    ecg_ids_all: list[np.ndarray] = []
    morphology_target: list[np.ndarray] = []
    morphology_recon: list[np.ndarray] = []
    morphology_labels: list[np.ndarray] = []
    morphology_ids: list[int] = []
    record_rows: list[dict[str, Any]] = []
    robust_seen = 0
    robust_probabilities: dict[str, list[np.ndarray]] = {
        condition.id: [] for condition in stress_conditions
    }
    example_plot = None

    for target, ecg_ids, labels, diagnostic_labels, ages, sexes in tqdm(loader, desc=spec["id"]):
        target = target.to(device, non_blocking=True)
        recon = adapter.reconstruct(target)
        if recon.shape != target.shape:
            raise RuntimeError(
                f"{spec['id']} produced {tuple(recon.shape)} for target {tuple(target.shape)}"
            )
        if not torch.isfinite(recon).all():
            raise RuntimeError(f"{spec['id']} produced NaN/Inf values")
        signal_metrics.update(target.cpu(), recon.cpu())
        record_rows.extend(
            record_signal_metrics(target.cpu(), recon.cpu(), ecg_ids, adapter.observed, "clean")
        )

        if robust_seen < robustness_samples:
            take = min(target.shape[0], robustness_samples - robust_seen)
            robust_clean_metrics.update(target[:take].cpu(), recon[:take].cpu())
            for condition_start in range(0, len(stress_conditions), robustness_condition_batch):
                condition_group = stress_conditions[
                    condition_start:condition_start + robustness_condition_batch
                ]
                noisy_group = [
                    apply_condition(
                        target[:take],
                        adapter.observed,
                        ecg_ids[:take],
                        condition,
                        nstdb_noises,
                    )
                    for condition in condition_group
                ]
                reconstructed_group = adapter.reconstruct(torch.cat(noisy_group, dim=0))
                if reconstructed_group.shape[0] != take * len(condition_group):
                    raise RuntimeError("Robustness condition batching changed output batch size")
                for group_index, condition in enumerate(condition_group):
                    noisy_recon = reconstructed_group[
                        group_index * take:(group_index + 1) * take
                    ]
                    robust_metrics[condition.id].update(
                        target[:take].cpu(), noisy_recon.cpu()
                    )
                    record_rows.extend(record_signal_metrics(
                        target[:take].cpu(),
                        noisy_recon.cpu(),
                        ecg_ids[:take],
                        adapter.observed,
                        condition.id,
                    ))
                    if classifier is not None:
                        robust_probabilities[condition.id].append(
                            torch.sigmoid(
                                classifier(preprocess_ecgfounder(noisy_recon))
                            ).cpu().numpy()
                        )
            robust_seen += take

        remaining = morphology_samples - len(morphology_target)
        if remaining > 0:
            take = min(target.shape[0], remaining)
            morphology_target.extend(target[:take].cpu().numpy())
            morphology_recon.extend(recon[:take].cpu().numpy())
            morphology_labels.extend(diagnostic_labels[:take].numpy())
            morphology_ids.extend(int(value) for value in ecg_ids[:take].tolist())

        if example_plot is None:
            example_plot = save_example_plot(
                spec["id"], target[0].cpu().numpy(), recon[0].cpu().numpy(), plot_dir
            )

        if classifier is not None:
            logits = classifier(preprocess_ecgfounder(recon))
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            reference_probabilities.append(
                torch.sigmoid(classifier(preprocess_ecgfounder(target))).cpu().numpy()
            )
            labels_all.append(labels.numpy())
            ages_all.append(ages.numpy())
            sexes_all.append(sexes.numpy())
            ecg_ids_all.append(ecg_ids.numpy())
        if parity_classifier is not None:
            from scripts.five_superclass_parity import preprocess as preprocess_parity
            parity_probabilities.append(
                torch.sigmoid(parity_classifier(preprocess_parity(recon))).cpu().numpy()
            )
            parity_labels.append(diagnostic_labels.numpy())
            parity_ids.append(ecg_ids.numpy())

    clean = signal_metrics.finalize()
    robust_clean = robust_clean_metrics.finalize() if robust_seen else {}
    robust_conditions: dict[str, Any] = {}
    for condition in stress_conditions:
        stressed = robust_metrics[condition.id].finalize() if robust_seen else {}
        condition_result: dict[str, Any] = {
            "source": condition.source,
            "noise_type": condition.noise_type,
            "snr_db": condition.snr_db,
            "n_samples": robust_seen,
            "signal": stressed,
            "missing_lead_mse_delta": (
                stressed["missing_leads"]["mse"]
                - robust_clean["missing_leads"]["mse"]
                if stressed
                else float("nan")
            ),
        }
        if classifier is not None and robust_probabilities[condition.id]:
            condition_result["clinical"] = classification_metrics(
                np.concatenate(robust_probabilities[condition.id]),
                np.concatenate(labels_all)[:robust_seen],
                class_names,
            )
        robust_conditions[condition.id] = condition_result
    result: dict[str, Any] = {
        "id": spec["id"],
        "family": spec["family"],
        "loss": spec["loss"],
        "kind": spec["kind"],
        "checkpoint": spec["checkpoint"],
        "observed_leads": spec["observed_leads"],
        "signal": clean,
        "robustness": {
            "n_samples": robust_seen,
            "clean_subset": robust_clean,
            "conditions": robust_conditions,
        },
        "morphology": morphology_metrics(
            morphology_target, morphology_recon, adapter.observed, morphology_labels
        ),
        "example_plot": example_plot,
    }
    if probabilities:
        probs = np.concatenate(probabilities)
        labels_np = np.concatenate(labels_all)
        ages_np = np.concatenate(ages_all)
        sexes_np = np.concatenate(sexes_all)
        result["clinical"] = classification_metrics(probs, labels_np, class_names)
        result["fairness"] = subgroup_metrics(
            probs, labels_np, ages_np, sexes_np, class_names
        )
        reference_probs = np.concatenate(reference_probabilities)
        result["clinical_reference"] = classification_metrics(
            reference_probs, labels_np, class_names
        )
        clinical_path = record_dir / f"{spec['id']}__ecgfounder.parquet"
        clinical_frame = pd.DataFrame({
            "ecg_id": np.concatenate(ecg_ids_all).astype(np.int64),
            "probabilities": list(probs.astype(np.float32)),
            "reference_probabilities": list(reference_probs.astype(np.float32)),
            "labels": list(labels_np.astype(np.float32)),
            "age": ages_np.astype(np.float32),
            "sex": sexes_np.astype(np.int8),
        })
        record_dir.mkdir(parents=True, exist_ok=True)
        clinical_frame.to_parquet(clinical_path, index=False)
        result["ecgfounder_per_record_parquet"] = str(clinical_path.relative_to(PROJECT_ROOT))
    else:
        result["clinical"] = {"status": "oracle_skipped"}
        result["fairness"] = {"status": "oracle_skipped"}

    if parity_probabilities:
        parity_probs = np.concatenate(parity_probabilities)
        parity_labels_np = np.concatenate(parity_labels)
        result["paper_parity_clinical"] = classification_metrics(
            parity_probs, parity_labels_np, parity_class_names
        )
        parity_path = record_dir / f"{spec['id']}__five_superclass.parquet"
        pd.DataFrame({
            "ecg_id": np.concatenate(parity_ids).astype(np.int64),
            "probabilities": list(parity_probs.astype(np.float32)),
            "labels": list(parity_labels_np.astype(np.float32)),
        }).to_parquet(parity_path, index=False)
        result["paper_parity_per_record_parquet"] = str(
            parity_path.relative_to(PROJECT_ROOT)
        )
    else:
        result["paper_parity_clinical"] = {"status": "parity_classifier_skipped"}

    morphology_by_id = {
        row["ecg_id"]: row
        for row in morphology_record_metrics(
            morphology_target, morphology_recon, morphology_ids, adapter.observed
        )
    }
    for row in record_rows:
        if row["condition"] == "clean" and row["ecg_id"] in morphology_by_id:
            row.update({key: value for key, value in morphology_by_id[row["ecg_id"]].items() if key != "ecg_id"})
    record_dir.mkdir(parents=True, exist_ok=True)
    record_path = record_dir / f"{spec['id']}.parquet"
    pd.DataFrame.from_records(record_rows).to_parquet(record_path, index=False)
    result["per_record_parquet"] = str(record_path.relative_to(PROJECT_ROOT))

    del adapter
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="experiment_queue/comprehensive_v1/model_registry.json")
    parser.add_argument("--output", default="results/comprehensive/comprehensive_results.json")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip_oracle", action="store_true")
    parser.add_argument("--skip_robustness", action="store_true")
    parser.add_argument("--robustness_samples", type=int, default=256)
    parser.add_argument("--robustness_condition_batch", type=int, default=4)
    parser.add_argument("--morphology_samples", type=int, default=128)
    args = parser.parse_args()
    if args.robustness_condition_batch < 1:
        raise ValueError("--robustness_condition_batch must be positive")

    registry_path = (PROJECT_ROOT / args.registry).resolve()
    registry = json.loads(registry_path.read_text())
    selected = registry["models"]
    if args.models:
        requested = set(args.models)
        selected = [model for model in selected if model["id"] in requested]
        missing = requested - {model["id"] for model in selected}
        if missing:
            raise ValueError(f"Unknown registry model IDs: {sorted(missing)}")
    missing_checkpoints = [
        model["checkpoint"]
        for model in selected
        if not (PROJECT_ROOT / model["checkpoint"]).exists()
    ]
    if missing_checkpoints:
        raise FileNotFoundError(
            "Registered checkpoints are missing; refusing partial evaluation:\n"
            + "\n".join(missing_checkpoints)
        )

    metadata = load_metadata(
        PROJECT_ROOT / registry["ptbxl_csv"],
        PROJECT_ROOT / registry["scp_statements_csv"],
        resolve_path(registry["ecgfounder_labels_csv"]),
    )
    dataset = TensorECGDataset(
        PROJECT_ROOT / registry["data_dir"], metadata, max_samples=args.max_samples
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    device = torch.device(args.device)
    classifier = None
    parity_classifier = None
    parity_class_names: list[str] = []
    from scripts.ecgfounder_classifier import load_ecgfounder, load_task_names
    from scripts.robustness_stress import (
        build_conditions,
        load_fitbit_noise,
        load_nstdb_noise,
    )

    class_names = load_task_names(resolve_path(registry["ecgfounder_tasks"]))
    if not args.skip_oracle:
        classifier = load_ecgfounder(
            resolve_path(registry["ecgfounder_repo"]),
            resolve_path(registry["ecgfounder_checkpoint"]),
            device,
            len(class_names),
        )
        from scripts.five_superclass_parity import load_classifier as load_parity_classifier
        parity_classifier, parity_class_names = load_parity_classifier(
            resolve_path(registry["five_superclass_backbone"]),
            resolve_path(registry["five_superclass_checkpoint"]),
            device,
        )
    fitbit_provenance = None
    fitbit_noises: dict[str, np.ndarray] = {}
    fitbit_root = resolve_path(registry.get("fitbit_noise_dir", "data/fitbit_noise"))
    if not args.skip_robustness and (fitbit_root / "PROVENANCE.json").exists():
        fitbit_noises, fitbit_provenance = load_fitbit_noise(fitbit_root)
    stress_conditions = (
        []
        if args.skip_robustness
        else build_conditions(include_fitbit=fitbit_provenance is not None)
    )
    nstdb_noises = (
        load_nstdb_noise(resolve_path(registry["nstdb_dir"]))
        if stress_conditions
        else {}
    )
    nstdb_noises.update(fitbit_noises)

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_output_path = output_path.with_suffix(output_path.suffix + ".partial")
    plot_dir = output_path.parent / "plots"
    record_dir = output_path.parent / "per_record"
    payload: dict[str, Any] = {
        "schema_version": 2,
        "protocol": {
            "dataset": "PTB-XL",
            "split": registry.get("split", "test"),
            "n_samples": len(dataset),
            "tensor_shape": [12, 5000],
            "sample_rate_hz": 500,
            "classifier": "ECGFounder 12-lead frozen 150-task head",
            "paper_parity_classifier": "separately retrained frozen ECG-FM five-superclass head",
            "labels": class_names,
            "robustness_conditions": [condition.id for condition in stress_conditions],
            "robustness_pairing": "FNV1a(condition_id,ecg_id,lead,base_seed=1337)",
            "robustness_condition_batch": args.robustness_condition_batch,
            "fitbit_noise_provenance": fitbit_provenance,
            "fitbit_noise_status": (
                "verified_and_included"
                if fitbit_provenance is not None
                else "not_included_missing_verified_extraction_provenance"
            ),
            "demographics": ["age", "sex"],
            "missing_lead_metrics_primary": True,
        },
        "models": {},
    }
    for spec in selected:
        payload["models"][spec["id"]] = evaluate_model(
            spec,
            loader,
            device,
            classifier,
            class_names,
            parity_classifier,
            parity_class_names,
            stress_conditions,
            nstdb_noises,
            0 if args.skip_robustness else args.robustness_samples,
            args.robustness_condition_batch,
            args.morphology_samples,
            plot_dir,
            record_dir,
        )
        partial_output_path.write_text(
            json.dumps(json_safe(payload), indent=2, allow_nan=False)
        )
    os.replace(partial_output_path, output_path)
    print(f"Saved comprehensive results: {output_path}")


if __name__ == "__main__":
    main()
