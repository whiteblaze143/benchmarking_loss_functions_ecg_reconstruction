"""Shared engineering utilities for reconstruction-first experiments."""

from __future__ import annotations

import csv
import glob
import json
import os
from typing import Any, Optional, Union

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset

from src.reconstruction.unified_latents.engineering.regimes import LEAD_NAMES


CHEST_LEADS = {"V1", "V2", "V3", "V4", "V5", "V6"}
LATERAL_LEADS = {"V4", "V5", "V6"}
V4_V6_LEADS = {"V4", "V5", "V6"}
V3_V6_LEADS = {"V3", "V4", "V5", "V6"}
CHEST_INDICES_ALL = [6, 7, 8, 9, 10, 11]
RECON_FOCUS_LEADS = [("V4", 9), ("V5", 10), ("V6", 11)]


class TensorFolderDataset(torch.utils.data.Dataset):
    """Simple PT tensor dataset used by engineering trainers and evaluators."""

    def __init__(self, folder_path: str):
        self.files = sorted(glob.glob(os.path.join(folder_path, "*.pt")))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        t = torch.load(self.files[idx], weights_only=True)
        t = torch.clamp(t, min=-5.0, max=5.0)
        return t, t.clone(), torch.zeros(5)


class CombinedTensorFolderDataset(ConcatDataset):
    """Concatenate multiple tensor split folders without moving files."""

    def __init__(self, folder_paths: list[str]):
        datasets = [TensorFolderDataset(folder_path) for folder_path in folder_paths]
        empty = [path for path, dataset in zip(folder_paths, datasets) if len(dataset) == 0]
        if empty:
            raise ValueError(f"No .pt tensors found in split folder(s): {empty}")
        super().__init__(datasets)
        self.folder_paths = list(folder_paths)


class AugmentedTensorDataset(torch.utils.data.Dataset):
    """Lightweight ECG augmentation wrapper for PTB-XL tensor folders."""

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        *,
        target_len: int = 5000,
        amp_min: float = 0.8,
        amp_max: float = 1.2,
        noise_std_frac: float = 0.02,
        resize_min: float = 0.95,
        resize_max: float = 1.05,
    ):
        self.dataset = dataset
        self.target_len = int(target_len)
        self.amp_min = float(amp_min)
        self.amp_max = float(amp_max)
        self.noise_std_frac = float(noise_std_frac)
        self.resize_min = float(resize_min)
        self.resize_max = float(resize_max)

    def __len__(self) -> int:
        return len(self.dataset)

    def _resize_and_match(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.empty((), dtype=x.dtype).uniform_(self.resize_min, self.resize_max).item()
        new_len = max(8, int(round(x.shape[-1] * scale)))
        x = F.interpolate(x.unsqueeze(0), size=new_len, mode="linear", align_corners=False).squeeze(0)
        if new_len > self.target_len:
            start = torch.randint(0, new_len - self.target_len + 1, ()).item()
            x = x[:, start : start + self.target_len]
        elif new_len < self.target_len:
            pad_total = self.target_len - new_len
            pad_left = torch.randint(0, pad_total + 1, ()).item()
            x = F.pad(x, (pad_left, pad_total - pad_left))
        return x

    def __getitem__(self, idx: int):
        x, y, meta = self.dataset[idx]
        x = self._resize_and_match(x.float())
        amp = torch.empty((x.shape[0], 1), dtype=x.dtype).uniform_(self.amp_min, self.amp_max)
        x = x * amp
        lead_std = x.std(dim=1, keepdim=True).clamp(min=1e-6)
        x = x + torch.randn_like(x) * lead_std * self.noise_std_frac
        x = torch.clamp(x, min=-5.0, max=5.0)
        # The target follows the same augmented waveform so the task remains
        # sparse-to-full reconstruction under realistic nuisance variation.
        return x, x.clone(), meta


def mask_unobserved_leads(
    x: torch.Tensor,
    obs_lead_indices: Optional[Union[torch.Tensor, list[int]]],
    fill_value: float = 0.0,
) -> torch.Tensor:
    """Mask a 12-lead ECG tensor so only observed leads remain.

    This is used during training and evaluation when only a subset of leads
    are available as input (e.g., 3-lead inference). The missing leads are
    overwritten with a constant value (default 0.0) so the encoder cannot
    "cheat" by seeing the full 12-lead signal.

    Args:
        x: Tensor of shape [B, 12, T].
        obs_lead_indices: Either a list of lead indices or a tensor of shape
            [B, N_obs] / [N_obs]. When a tensor is provided it is permitted to
            have a batch size of 1 and will be broadcast across the batch.
        fill_value: Value to fill for unobserved leads.

    Returns:
        Tensor of shape [B, 12, T] with unobserved leads filled.
    """

    if x.dim() != 3 or x.size(1) != 12:
        raise ValueError(f"Expected x shape [B, 12, T], got {tuple(x.shape)}")

    if obs_lead_indices is None:
        return x

    if isinstance(obs_lead_indices, (list, tuple)):
        obs_lead_indices = torch.tensor(obs_lead_indices, device=x.device)

    if isinstance(obs_lead_indices, torch.Tensor):
        if obs_lead_indices.dim() == 1:
            obs_lead_indices = obs_lead_indices.unsqueeze(0).expand(x.size(0), -1)
        elif obs_lead_indices.dim() == 2 and obs_lead_indices.size(0) == 1:
            obs_lead_indices = obs_lead_indices.expand(x.size(0), -1)

        if obs_lead_indices.dim() != 2:
            raise ValueError(
                f"Expected obs_lead_indices to be shape [N_obs] or [B, N_obs], got {tuple(obs_lead_indices.shape)}"
            )

        if obs_lead_indices.numel() == 0:
            return torch.full_like(x, float(fill_value))

        mask = x.new_zeros((x.size(0), 12, 1))
        # We assume the observed-lead set is the same for all batch items.
        if obs_lead_indices.dim() == 2:
            # If it's [B, N_obs], assume all rows are identical (common regime)
            obs_lead_indices = obs_lead_indices[0]
        mask[:, obs_lead_indices.long(), :] = 1.0
        return x * mask + float(fill_value) * (1.0 - mask)

    raise TypeError(
        f"Unsupported type for obs_lead_indices: {type(obs_lead_indices)}. "
        "Expected list[int] or Tensor."
    )


def compute_batch_r2(out: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    ssr = ((out - tgt) ** 2).sum(dim=2)
    sst = ((tgt - tgt.mean(dim=2, keepdim=True)) ** 2).sum(dim=2)
    r2 = 1.0 - ssr / torch.clamp(sst, min=0.1)
    return torch.clamp(r2, min=-100.0).mean()


def compute_batch_r2_per_lead(out: torch.Tensor, tgt: torch.Tensor) -> list[float]:
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


def compute_batch_mae(out: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    return (out - tgt).abs().mean()


def compute_batch_mae_per_lead(out: torch.Tensor, tgt: torch.Tensor) -> list[float]:
    return (out - tgt).abs().mean(dim=(0, 2)).tolist()


def compute_batch_mse(out: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    return ((out - tgt) ** 2).mean()


def compute_batch_mse_per_lead(out: torch.Tensor, tgt: torch.Tensor) -> list[float]:
    return ((out - tgt) ** 2).mean(dim=(0, 2)).tolist()


def compute_batch_rmse(out: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(F.mse_loss(out, tgt))


def compute_batch_rmse_per_lead(out: torch.Tensor, tgt: torch.Tensor) -> list[float]:
    return torch.sqrt(((out - tgt) ** 2).mean(dim=(0, 2))).tolist()


def compute_batch_corr_per_lead(out: torch.Tensor, tgt: torch.Tensor) -> list[float]:
    pred = out - out.mean(dim=2, keepdim=True)
    true = tgt - tgt.mean(dim=2, keepdim=True)
    numerator = (pred * true).sum(dim=2)
    denom = torch.sqrt((pred ** 2).sum(dim=2) * (true ** 2).sum(dim=2)).clamp(min=1e-8)
    corr = (numerator / denom).mean(dim=0)
    return corr.tolist()


def compute_rwave_progression_metrics(
    gt_12lead: torch.Tensor, pred_12lead: torch.Tensor, chest_indices: list[int]
) -> dict[str, float]:
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


def mean_for_leads(values: list[float], lead_names: list[str], selected_leads: set[str]) -> float | None:
    picked = [value for value, lead in zip(values, lead_names) if lead in selected_leads]
    if not picked:
        return None
    return sum(picked) / len(picked)


def to_serializable_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    serializable = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            serializable[key] = float(value.item())
        elif isinstance(value, (int, float, str, bool)) or value is None:
            serializable[key] = value
        else:
            serializable[key] = value
    return serializable


def write_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="ascii") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def write_run_artifacts(save_dir: str, metadata: dict[str, Any], metrics: dict[str, Any]) -> None:
    os.makedirs(save_dir, exist_ok=True)
    serializable_metrics = to_serializable_metrics(metrics)
    write_json(os.path.join(save_dir, "run_metadata.json"), metadata)
    write_json(os.path.join(save_dir, "latest_metrics.json"), serializable_metrics)
    with open(os.path.join(save_dir, "latest_metrics.csv"), "w", encoding="ascii", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        for key in sorted(serializable_metrics):
            writer.writerow([key, serializable_metrics[key]])


def get_selector_tuple(metrics: dict[str, float]) -> tuple[float, float, float, float, float]:
    return (
        metrics.get("val/r2_reg_v4_v6_mean", float("-inf")),
        metrics.get("val/r2_reg_v3_v6_mean", float("-inf")),
        -metrics.get("val/rmse_reg", float("inf")),
        metrics.get("val/r2_regressor", float("-inf")),
        -metrics.get("val/mse_z_reg", float("inf")),
    )


def write_best_summary(save_dir: str, score_tuple: tuple[float, ...], metrics: dict[str, Any], epoch: int) -> None:
    payload = {
        "epoch": int(epoch),
        "selector_tuple": list(score_tuple),
        "metrics": to_serializable_metrics(metrics),
    }
    write_json(os.path.join(save_dir, "best_summary.json"), payload)


def prune_epoch_checkpoints(save_dir: str, keep_latest: int = 2) -> None:
    all_ckpts = sorted(glob.glob(os.path.join(save_dir, "ul_ecp_ep*.pt")), key=os.path.getmtime)
    if len(all_ckpts) <= keep_latest:
        return
    for old_ckpt in all_ckpts[:-keep_latest]:
        try:
            os.remove(old_ckpt)
        except OSError:
            pass


def cleanup_partial_checkpoints(save_dir: str) -> None:
    for tmp_path in glob.glob(os.path.join(save_dir, "*.tmp")):
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def _extract_model_state(ckpt: dict[str, Any]) -> dict[str, Any]:
    if "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    return ckpt


def load_compatible_model_state(model, checkpoint_path: str, device) -> tuple[dict[str, Any], dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    source_state = _extract_model_state(ckpt)
    target_state = model.state_dict()
    compatible_state = {}
    skipped = []
    for key, value in source_state.items():
        if key not in target_state or target_state[key].shape != value.shape:
            skipped.append(key)
            continue
        compatible_state[key] = value
    missing, unexpected = model.load_state_dict(compatible_state, strict=False)

    loaded_keys = sorted(compatible_state.keys())
    skipped_sorted = sorted(skipped)
    loaded_prefixes = sorted({key.split(".")[0] for key in loaded_keys})
    skipped_prefixes = sorted({key.split(".")[0] for key in skipped_sorted})
    if not checkpoint_path:
        init_type = "scratch"
    elif skipped_sorted:
        init_type = "partial_warm_start"
    else:
        init_type = "full_warm_start"

    warm_start_summary = {
        "checkpoint_path": checkpoint_path,
        "loaded_tensor_count": len(loaded_keys),
        "skipped_tensor_count": len(skipped_sorted),
        "loaded_prefixes": loaded_prefixes,
        "skipped_prefixes_sample": skipped_prefixes[:20],
        "loaded_keys_sample": loaded_keys[:20],
        "skipped_keys_sample": skipped_sorted[:20],
        "missing_keys_sample": sorted(missing)[:20],
        "unexpected_keys_sample": sorted(unexpected)[:20],
        "initialization_type": init_type,
    }
    return ckpt, warm_start_summary


def write_warm_start_summary(save_dir: str, summary: dict[str, Any]) -> None:
    os.makedirs(save_dir, exist_ok=True)
    write_json(os.path.join(save_dir, "warm_start_summary.json"), summary)
