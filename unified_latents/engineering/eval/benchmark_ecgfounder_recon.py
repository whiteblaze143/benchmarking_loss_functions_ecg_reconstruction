#!/usr/bin/env python3
"""ECGFounder audit for reconstruction models on PTB-XL and Sunnybrook.

This script compares:
- original 12-lead ECGs
- exact baseline reconstructions
- FM reconstructions

using the frozen 150-task ECGFounder classifier already vendored in this repo.

PTB-XL:
- evaluates classification AUROC / average precision against the provided
  `ptbxl_label.csv` labels from the ECGFounder repo
- reports probability-preservation correlation vs original ECGs

Sunnybrook:
- no 150-task ground-truth labels are available
- reports probability-preservation metrics vs original ECGs only
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import wfdb
from scipy.signal import butter, filtfilt, iirnotch, medfilt
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from scripts.bootstrap_paths import setup_import_paths, ECG_FM_ROOT
setup_import_paths(include_fairseq=True)
ECGFOUNDER_REPO = ECG_FM_ROOT / "ecgfounder_repo"
if str(ECGFOUNDER_REPO) not in sys.path:
    sys.path.insert(0, str(ECGFOUNDER_REPO))
from net1d import Net1D  # type: ignore

from unified_latents.engineering.eval.benchmark_sunnybrook_all_features import (
    LEAD_ORDER,
    load_reconstruction_model,
    load_sunnybrook_record,
    parse_obs_leads,
)


DEFAULT_BASELINE_CKPT = (
    "/home/mithunmanivannan/checkpoints/ul_ecg/"
    "engineering_wearecg_exact_II-V1-V5_bs64_lf1.5_ep10_canonical_baseline/"
    "ul_ecp_best.pt"
)
DEFAULT_FM_CKPT = (
    "/home/mithunmanivannan/checkpoints/wearecg_fm/"
    "engineering_fm_vae_II-V1-V5_bs64_diagmasksplitv1_mw2_fml5e-2_mix0_dc0_drop0_la0_law0/"
    "ul_ecp_best.pt"
)
DEFAULT_PTBXL_ROOT = "/home/mithunmanivannan/data/ptb_xl"
DEFAULT_PTBXL_LABEL_CSV = (
    str(ECGFOUNDER_REPO / "csv" / "ptbxl_label.csv")
)
DEFAULT_TASKS = str(ECGFOUNDER_REPO / "tasks.txt")
DEFAULT_SUNNYBROOK_DIR = "/home/mithunmanivannan/data/sunnybrook"
DEFAULT_OUTPUT_ROOT = "/home/mithunmanivannan/reports/ecgfounder_recon_audit"
DEFAULT_PTBXL_SCREEN_INDEX = "ptbxl_screen_4096_seed42.csv"
ECGFOUNDER_CKPT = (
    str(ECGFOUNDER_REPO / "checkpoint" / "12_lead_ECGFounder.pth")
)

INTEREST_KEYWORDS = (
    "AXIS",
    "BUNDLE BRANCH BLOCK",
    "FASCICULAR BLOCK",
    "INTRAVENTRICULAR",
    "PACEMAKER",
    "PACED",
    "AV BLOCK",
    "HEART BLOCK",
    "CONDUCTION",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ECGFounder reconstruction audit.")
    parser.add_argument("--baseline-checkpoint", default=DEFAULT_BASELINE_CKPT)
    parser.add_argument("--candidate-checkpoint", default=DEFAULT_FM_CKPT)
    parser.add_argument("--ptbxl-root", default=DEFAULT_PTBXL_ROOT)
    parser.add_argument("--ptbxl-label-csv", default=DEFAULT_PTBXL_LABEL_CSV)
    parser.add_argument("--ptbxl-index-csv", default=DEFAULT_PTBXL_SCREEN_INDEX)
    parser.add_argument("--tasks-path", default=DEFAULT_TASKS)
    parser.add_argument("--sunnybrook-dir", default=DEFAULT_SUNNYBROOK_DIR)
    parser.add_argument("--obs-leads", default="II,V1,V5")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--ptbxl-limit", type=int, default=None)
    parser.add_argument("--sunnybrook-limit", type=int, default=None)
    parser.add_argument("--screen-subset-size", type=int, default=4096)
    parser.add_argument("--screen-seed", type=int, default=42)
    parser.add_argument("--screen-min-class-count", type=int, default=25)
    parser.add_argument("--use-ptbxl-index", dest="use_ptbxl_index", action="store_true")
    parser.add_argument("--no-use-ptbxl-index", dest="use_ptbxl_index", action="store_false")
    parser.add_argument(
        "--allow-generate-ptbxl-index",
        dest="allow_generate_ptbxl_index",
        action="store_true",
    )
    parser.add_argument(
        "--no-allow-generate-ptbxl-index",
        dest="allow_generate_ptbxl_index",
        action="store_false",
    )
    parser.set_defaults(use_ptbxl_index=True, allow_generate_ptbxl_index=True)
    parser.add_argument("--skip-ptbxl", action="store_true")
    parser.add_argument("--skip-sunnybrook", action="store_true")
    return parser.parse_args()


def sanitize_name(path: str) -> str:
    return Path(path).resolve().parent.name


def checkpoint_run_tag(path: str) -> str:
    resolved = Path(path).resolve()
    return f"{resolved.parent.name}__{resolved.stem}"


def load_ecgfounder(device: torch.device) -> Net1D:
    model = Net1D(
        in_channels=12,
        base_filters=64,
        ratio=1,
        filter_list=[64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
        kernel_size=16,
        stride=2,
        groups_width=16,
        verbose=False,
        use_bn=False,
        use_do=False,
        n_classes=150,
    )
    checkpoint = torch.load(ECGFOUNDER_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval().to(device)
    return model


def load_tasks(tasks_path: str) -> list[str]:
    with open(tasks_path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


class PTBXLECGFounderDataset(Dataset):
    def __init__(self, root: str, labels_df: pd.DataFrame):
        self.root = Path(root)
        self.df = labels_df.reset_index(drop=True).copy()
        self.labels = np.stack(self.df["label"].map(json.loads).to_numpy()).astype(np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        row = self.df.iloc[idx]
        rel = str(row["filename_hr"])
        record_path = self.root / rel
        signal, _ = wfdb.rdsamp(str(record_path))
        signal = signal.T.astype(np.float32)
        if signal.shape[1] < 5000:
            signal = np.pad(signal, ((0, 0), (0, 5000 - signal.shape[1])))
        elif signal.shape[1] > 5000:
            signal = signal[:, :5000]
        return (
            torch.from_numpy(signal),
            torch.from_numpy(self.labels[idx]),
            rel,
        )


def collate_ptbxl(batch: list[tuple[torch.Tensor, torch.Tensor, str]]):
    signals, labels, names = zip(*batch)
    return torch.stack(list(signals), dim=0), torch.stack(list(labels), dim=0), list(names)


def ecgfounder_preprocess_ptbxl(signals: np.ndarray) -> np.ndarray:
    mean = signals.mean(axis=(1, 2), keepdims=True)
    std = signals.std(axis=(1, 2), keepdims=True)
    std = np.clip(std, 1e-8, None)
    return ((signals - mean) / std).astype(np.float32)


def ecgfounder_preprocess_external(signal: np.ndarray, fs: int = 500) -> np.ndarray:
    b_notch, a_notch = iirnotch(50, 30, fs)
    sig_filtered = filtfilt(b_notch, a_notch, signal, axis=-1)
    b_bp, a_bp = butter(N=4, Wn=[0.67, 40], btype="bandpass", fs=fs)
    sig_filtered = filtfilt(b_bp, a_bp, sig_filtered, axis=-1)
    kernel_size = int(0.4 * fs) + 1
    if kernel_size % 2 == 0:
        kernel_size += 1
    baseline = np.stack([medfilt(ch, kernel_size=kernel_size) for ch in sig_filtered], axis=0)
    sig_pre = sig_filtered - baseline
    mean = sig_pre.mean()
    std = max(float(sig_pre.std()), 1e-8)
    return ((sig_pre - mean) / std).astype(np.float32)


def infer_probs(model: Net1D, signals: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    probs = []
    with torch.no_grad():
        for start in range(0, len(signals), batch_size):
            batch = torch.from_numpy(signals[start : start + batch_size]).to(device=device, dtype=torch.float32)
            logits = model(batch)
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs, axis=0) if probs else np.zeros((0, 150), dtype=np.float32)


def safe_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return math.nan
    return float(roc_auc_score(y_true, y_pred))


def safe_ap(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.sum() == 0:
        return math.nan
    return float(average_precision_score(y_true, y_pred))


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def summarize_task_metrics(
    task_names: list[str],
    labels: np.ndarray,
    orig_probs: np.ndarray,
    baseline_probs: np.ndarray,
    candidate_probs: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for idx, task in enumerate(task_names):
        y = labels[:, idx]
        rows.append(
            {
                "task_index": idx,
                "task_name": task,
                "positives": int(y.sum()),
                "negatives": int((1 - y).sum()),
                "orig_auroc": safe_auc(y, orig_probs[:, idx]),
                "baseline_auroc": safe_auc(y, baseline_probs[:, idx]),
                "candidate_auroc": safe_auc(y, candidate_probs[:, idx]),
                "orig_ap": safe_ap(y, orig_probs[:, idx]),
                "baseline_ap": safe_ap(y, baseline_probs[:, idx]),
                "candidate_ap": safe_ap(y, candidate_probs[:, idx]),
                "baseline_prob_corr_vs_orig": safe_corr(orig_probs[:, idx], baseline_probs[:, idx]),
                "candidate_prob_corr_vs_orig": safe_corr(orig_probs[:, idx], candidate_probs[:, idx]),
            }
        )
    df = pd.DataFrame(rows)
    df["candidate_minus_baseline_auroc"] = df["candidate_auroc"] - df["baseline_auroc"]
    df["candidate_minus_baseline_ap"] = df["candidate_ap"] - df["baseline_ap"]
    df["candidate_minus_baseline_prob_corr"] = (
        df["candidate_prob_corr_vs_orig"] - df["baseline_prob_corr_vs_orig"]
    )
    df["interest_task"] = df["task_name"].str.upper().apply(
        lambda name: any(keyword in name for keyword in INTEREST_KEYWORDS)
    )
    return df


def macro_mean(series: pd.Series) -> float:
    valid = pd.to_numeric(series, errors="coerce").dropna()
    return float(valid.mean()) if len(valid) else math.nan


def _dedupe_preserve_order(values: list[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        iv = int(value)
        if iv in seen:
            continue
        seen.add(iv)
        out.append(iv)
    return out


def resolve_index_path(index_csv: str, output_root: str) -> Path:
    path = Path(index_csv)
    if path.is_absolute():
        return path
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate
    return Path(output_root) / path


def build_or_load_ptbxl_subset(
    labels_df: pd.DataFrame,
    index_csv: str,
    output_root: str,
    subset_size: int,
    seed: int,
    allow_generate: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, Path, str]:
    index_path = resolve_index_path(index_csv, output_root)
    label_filenames = labels_df["filename_hr"].astype(str).tolist()
    filename_to_idx = {name: idx for idx, name in enumerate(label_filenames)}

    if index_path.exists():
        index_df = pd.read_csv(index_path)
        if "index_in_label_csv" in index_df.columns:
            indices = _dedupe_preserve_order(index_df["index_in_label_csv"].dropna().astype(int).tolist())
        elif "label_index" in index_df.columns:
            indices = _dedupe_preserve_order(index_df["label_index"].dropna().astype(int).tolist())
        elif "filename_hr" in index_df.columns:
            names = index_df["filename_hr"].astype(str).tolist()
            indices = _dedupe_preserve_order([filename_to_idx[name] for name in names if name in filename_to_idx])
        elif "record" in index_df.columns:
            names = index_df["record"].astype(str).tolist()
            indices = _dedupe_preserve_order([filename_to_idx[name] for name in names if name in filename_to_idx])
        else:
            raise RuntimeError(
                f"Unsupported PTB-XL index format in {index_path}. "
                "Expected one of: index_in_label_csv, label_index, filename_hr, record"
            )
        source = "existing_index_file"
    else:
        if not allow_generate:
            raise FileNotFoundError(
                f"PTB-XL index file not found: {index_path}. "
                "Pass --allow-generate-ptbxl-index to create it deterministically."
            )
        n = min(int(subset_size), len(labels_df))
        rng = np.random.default_rng(int(seed))
        sampled = rng.choice(len(labels_df), size=n, replace=False).tolist()
        indices = _dedupe_preserve_order(sampled)
        source = "generated_deterministic"

    valid_indices = [idx for idx in indices if 0 <= idx < len(labels_df)]
    if not valid_indices:
        raise RuntimeError("PTB-XL screen subset is empty after resolving index file.")

    subset_df = labels_df.iloc[valid_indices].copy()
    subset_definition = pd.DataFrame(
        {
            "subset_rank": np.arange(len(valid_indices), dtype=int),
            "index_in_label_csv": valid_indices,
            "filename_hr": subset_df["filename_hr"].astype(str).tolist(),
            "subset_source": source,
            "screen_seed": int(seed),
            "screen_subset_size": int(len(valid_indices)),
            "index_file": str(index_path),
        }
    )

    if source == "generated_deterministic":
        index_path.parent.mkdir(parents=True, exist_ok=True)
        subset_definition[["subset_rank", "index_in_label_csv", "filename_hr"]].to_csv(index_path, index=False)

    return subset_df, subset_definition, index_path, source


def summarize_ptbxl(
    task_metrics: pd.DataFrame,
    orig_probs: np.ndarray,
    baseline_probs: np.ndarray,
    candidate_probs: np.ndarray,
    screen_min_class_count: int,
) -> dict[str, Any]:
    interest = task_metrics[task_metrics["interest_task"]].copy()
    valid_tasks = task_metrics[
        (task_metrics["positives"] >= int(screen_min_class_count))
        & (task_metrics["negatives"] >= int(screen_min_class_count))
    ].copy()
    interest_valid = valid_tasks[valid_tasks["interest_task"]].copy()
    return {
        "n_records": int(orig_probs.shape[0]),
        "n_tasks": int(task_metrics.shape[0]),
        "n_interest_tasks": int(interest.shape[0]),
        "n_valid_auroc_tasks": int(valid_tasks.shape[0]),
        "n_valid_interest_tasks": int(interest_valid.shape[0]),
        "screen_min_class_count": int(screen_min_class_count),
        "macro_auroc": macro_mean(valid_tasks["candidate_auroc"]),
        "interest_macro_auroc": macro_mean(interest_valid["candidate_auroc"]),
        "macro_ap": macro_mean(valid_tasks["candidate_ap"]),
        "candidate_beats_baseline_on_valid_tasks": int(
            (valid_tasks["candidate_minus_baseline_auroc"] > 0).sum()
        ),
        "baseline_macro_auroc_valid": macro_mean(valid_tasks["baseline_auroc"]),
        "baseline_interest_macro_auroc_valid": macro_mean(interest_valid["baseline_auroc"]),
        "baseline_macro_ap_valid": macro_mean(valid_tasks["baseline_ap"]),
        "orig_macro_auroc_valid": macro_mean(valid_tasks["orig_auroc"]),
        "orig_macro_ap_valid": macro_mean(valid_tasks["orig_ap"]),
        "candidate_macro_auroc_valid": macro_mean(valid_tasks["candidate_auroc"]),
        "candidate_macro_ap_valid": macro_mean(valid_tasks["candidate_ap"]),
        "orig_macro_auroc": macro_mean(task_metrics["orig_auroc"]),
        "baseline_macro_auroc": macro_mean(task_metrics["baseline_auroc"]),
        "candidate_macro_auroc": macro_mean(task_metrics["candidate_auroc"]),
        "orig_macro_ap": macro_mean(task_metrics["orig_ap"]),
        "baseline_macro_ap": macro_mean(task_metrics["baseline_ap"]),
        "candidate_macro_ap": macro_mean(task_metrics["candidate_ap"]),
        "baseline_global_prob_corr_vs_orig": safe_corr(orig_probs.ravel(), baseline_probs.ravel()),
        "candidate_global_prob_corr_vs_orig": safe_corr(orig_probs.ravel(), candidate_probs.ravel()),
        "candidate_beats_baseline_on_auroc_tasks": int((task_metrics["candidate_minus_baseline_auroc"] > 0).sum()),
        "candidate_beats_baseline_on_ap_tasks": int((task_metrics["candidate_minus_baseline_ap"] > 0).sum()),
        "interest_baseline_macro_auroc": macro_mean(interest["baseline_auroc"]),
        "interest_candidate_macro_auroc": macro_mean(interest["candidate_auroc"]),
        "interest_baseline_prob_corr_vs_orig": macro_mean(interest["baseline_prob_corr_vs_orig"]),
        "interest_candidate_prob_corr_vs_orig": macro_mean(interest["candidate_prob_corr_vs_orig"]),
    }


def run_ptbxl_audit(
    baseline_model: torch.nn.Module,
    candidate_model: torch.nn.Module,
    ecgfounder: Net1D,
    device: torch.device,
    obs_indices: list[int],
    batch_size: int,
    ptbxl_root: str,
    ptbxl_label_csv: str,
    tasks: list[str],
    limit: int | None,
    ptbxl_index_csv: str,
    use_ptbxl_index: bool,
    allow_generate_ptbxl_index: bool,
    screen_subset_size: int,
    screen_seed: int,
    output_root: str,
    screen_min_class_count: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels_df = pd.read_csv(ptbxl_label_csv)
    if use_ptbxl_index:
        selected_df, subset_definition, resolved_index_path, subset_source = build_or_load_ptbxl_subset(
            labels_df=labels_df,
            index_csv=ptbxl_index_csv,
            output_root=output_root,
            subset_size=screen_subset_size,
            seed=screen_seed,
            allow_generate=allow_generate_ptbxl_index,
        )
    else:
        selected_df = labels_df.copy()
        subset_definition = pd.DataFrame(
            {
                "subset_rank": np.arange(len(selected_df), dtype=int),
                "index_in_label_csv": np.arange(len(selected_df), dtype=int),
                "filename_hr": selected_df["filename_hr"].astype(str).tolist(),
                "subset_source": "full_dataset",
                "screen_seed": int(screen_seed),
                "screen_subset_size": int(len(selected_df)),
                "index_file": "<disabled>",
            }
        )
        resolved_index_path = Path("<disabled>")
        subset_source = "full_dataset"

    if limit is not None:
        selected_df = selected_df.iloc[:limit].copy()
        subset_definition = subset_definition.iloc[:limit].copy().reset_index(drop=True)
        subset_definition["subset_rank"] = np.arange(len(subset_definition), dtype=int)

    dataset = PTBXLECGFounderDataset(ptbxl_root, selected_df)
    if dataset.labels.shape[1] != len(tasks):
        raise RuntimeError(
            f"PTB-XL label width {dataset.labels.shape[1]} does not match task count {len(tasks)}"
        )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_ptbxl)
    lead_idx_tensor = None
    all_labels = []
    all_orig_probs = []
    all_baseline_probs = []
    all_candidate_probs = []

    for signals, labels, _names in tqdm(loader, desc="PTB-XL ECGFounder audit"):
        signals = signals.to(device=device, dtype=torch.float32)
        if lead_idx_tensor is None or lead_idx_tensor.shape[0] != signals.shape[0]:
            lead_idx_tensor = torch.tensor([obs_indices] * signals.shape[0], device=device)
        with torch.no_grad():
            baseline_out = baseline_model.impute_from_regressor(signals, lead_indices=lead_idx_tensor)
            candidate_out = candidate_model.impute_from_regressor(signals, lead_indices=lead_idx_tensor)
        orig_np = signals.cpu().numpy()
        baseline_np = baseline_out["y_pred"].detach().cpu().numpy()
        candidate_np = candidate_out["y_pred"].detach().cpu().numpy()

        orig_probs = infer_probs(ecgfounder, ecgfounder_preprocess_ptbxl(orig_np), device, batch_size)
        baseline_probs = infer_probs(ecgfounder, ecgfounder_preprocess_ptbxl(baseline_np), device, batch_size)
        candidate_probs = infer_probs(ecgfounder, ecgfounder_preprocess_ptbxl(candidate_np), device, batch_size)

        all_labels.append(labels.numpy())
        all_orig_probs.append(orig_probs)
        all_baseline_probs.append(baseline_probs)
        all_candidate_probs.append(candidate_probs)

    labels_np = np.concatenate(all_labels, axis=0)
    orig_np = np.concatenate(all_orig_probs, axis=0)
    baseline_np = np.concatenate(all_baseline_probs, axis=0)
    candidate_np = np.concatenate(all_candidate_probs, axis=0)
    task_metrics = summarize_task_metrics(tasks, labels_np, orig_np, baseline_np, candidate_np)
    task_metrics["valid_screen_task"] = (
        (task_metrics["positives"] >= int(screen_min_class_count))
        & (task_metrics["negatives"] >= int(screen_min_class_count))
    )
    task_definition = task_metrics[
        [
            "task_index",
            "task_name",
            "positives",
            "negatives",
            "interest_task",
            "valid_screen_task",
        ]
    ].copy()
    summary = summarize_ptbxl(
        task_metrics,
        orig_np,
        baseline_np,
        candidate_np,
        screen_min_class_count=screen_min_class_count,
    )
    summary["ptbxl_subset_source"] = subset_source
    summary["ptbxl_index_csv"] = str(resolved_index_path)
    return summary, task_metrics, subset_definition, task_definition


def run_sunnybrook_audit(
    baseline_model: torch.nn.Module,
    candidate_model: torch.nn.Module,
    ecgfounder: Net1D,
    device: torch.device,
    obs_indices: list[int],
    batch_size: int,
    sunnybrook_dir: str,
    tasks: list[str],
    limit: int | None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    xmls = sorted(Path(sunnybrook_dir).glob("*.xml"))
    if limit is not None:
        xmls = xmls[:limit]
    rows = []
    lead_idx_tensor = None
    orig_all = []
    baseline_all = []
    candidate_all = []

    for xml_path in tqdm(xmls, desc="Sunnybrook ECGFounder audit"):
        signal = load_sunnybrook_record(xml_path)
        x = torch.from_numpy(signal).unsqueeze(0).to(device=device, dtype=torch.float32)
        if lead_idx_tensor is None:
            lead_idx_tensor = torch.tensor([obs_indices], device=device)
        with torch.no_grad():
            baseline_out = baseline_model.impute_from_regressor(x, lead_indices=lead_idx_tensor)
            candidate_out = candidate_model.impute_from_regressor(x, lead_indices=lead_idx_tensor)

        orig_np = ecgfounder_preprocess_external(signal)
        baseline_np = ecgfounder_preprocess_external(baseline_out["y_pred"].squeeze(0).detach().cpu().numpy())
        candidate_np = ecgfounder_preprocess_external(candidate_out["y_pred"].squeeze(0).detach().cpu().numpy())

        orig_probs = infer_probs(ecgfounder, orig_np[None, ...], device, batch_size)[0]
        baseline_probs = infer_probs(ecgfounder, baseline_np[None, ...], device, batch_size)[0]
        candidate_probs = infer_probs(ecgfounder, candidate_np[None, ...], device, batch_size)[0]

        orig_all.append(orig_probs)
        baseline_all.append(baseline_probs)
        candidate_all.append(candidate_probs)

        top_orig = np.argsort(orig_probs)[::-1][:5]
        top_base = np.argsort(baseline_probs)[::-1][:5]
        top_cand = np.argsort(candidate_probs)[::-1][:5]
        rows.append(
            {
                "file": xml_path.name,
                "baseline_prob_corr_vs_orig": safe_corr(orig_probs, baseline_probs),
                "candidate_prob_corr_vs_orig": safe_corr(orig_probs, candidate_probs),
                "baseline_top5_overlap": int(len(set(top_orig) & set(top_base))),
                "candidate_top5_overlap": int(len(set(top_orig) & set(top_cand))),
                "orig_top5_tasks": " | ".join(tasks[i] for i in top_orig),
                "baseline_top5_tasks": " | ".join(tasks[i] for i in top_base),
                "candidate_top5_tasks": " | ".join(tasks[i] for i in top_cand),
            }
        )

    per_record = pd.DataFrame(rows)
    orig_all_np = np.stack(orig_all, axis=0) if orig_all else np.zeros((0, len(tasks)))
    baseline_all_np = np.stack(baseline_all, axis=0) if baseline_all else np.zeros((0, len(tasks)))
    candidate_all_np = np.stack(candidate_all, axis=0) if candidate_all else np.zeros((0, len(tasks)))

    summary = {
        "n_records": int(len(xmls)),
        "baseline_global_prob_corr_vs_orig": safe_corr(orig_all_np.ravel(), baseline_all_np.ravel()),
        "candidate_global_prob_corr_vs_orig": safe_corr(orig_all_np.ravel(), candidate_all_np.ravel()),
        "baseline_mean_record_prob_corr_vs_orig": macro_mean(per_record["baseline_prob_corr_vs_orig"]) if len(per_record) else math.nan,
        "candidate_mean_record_prob_corr_vs_orig": macro_mean(per_record["candidate_prob_corr_vs_orig"]) if len(per_record) else math.nan,
        "baseline_mean_top5_overlap": macro_mean(per_record["baseline_top5_overlap"]) if len(per_record) else math.nan,
        "candidate_mean_top5_overlap": macro_mean(per_record["candidate_top5_overlap"]) if len(per_record) else math.nan,
    }
    return summary, per_record


def write_interpretation(
    output_dir: Path,
    baseline_name: str,
    candidate_name: str,
    ptbxl_summary: dict[str, Any] | None,
    sunnybrook_summary: dict[str, Any] | None,
) -> None:
    lines = [
        "# ECGFounder Reconstruction Audit",
        "",
        f"- Baseline checkpoint: `{baseline_name}`",
        f"- Candidate checkpoint: `{candidate_name}`",
        "",
    ]
    if ptbxl_summary is not None:
        lines.extend(
            [
                "## PTB-XL",
                "",
                f"- Baseline macro AUROC: `{ptbxl_summary['baseline_macro_auroc']:.6f}`",
                f"- Candidate macro AUROC: `{ptbxl_summary['candidate_macro_auroc']:.6f}`",
                f"- Baseline global probability correlation vs original: `{ptbxl_summary['baseline_global_prob_corr_vs_orig']:.6f}`",
                f"- Candidate global probability correlation vs original: `{ptbxl_summary['candidate_global_prob_corr_vs_orig']:.6f}`",
                f"- Candidate beats baseline on AUROC for `{ptbxl_summary['candidate_beats_baseline_on_auroc_tasks']}/{ptbxl_summary['n_tasks']}` tasks",
                "",
            ]
        )
    if sunnybrook_summary is not None:
        lines.extend(
            [
                "## Sunnybrook",
                "",
                f"- Baseline global probability correlation vs original: `{sunnybrook_summary['baseline_global_prob_corr_vs_orig']:.6f}`",
                f"- Candidate global probability correlation vs original: `{sunnybrook_summary['candidate_global_prob_corr_vs_orig']:.6f}`",
                f"- Baseline mean top-5 task overlap: `{sunnybrook_summary['baseline_mean_top5_overlap']:.3f}`",
                f"- Candidate mean top-5 task overlap: `{sunnybrook_summary['candidate_mean_top5_overlap']:.3f}`",
                "",
            ]
        )
    (output_dir / "interpretation.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    obs_indices = parse_obs_leads(args.obs_leads)

    baseline_model, _baseline_payload, _ = load_reconstruction_model(args.baseline_checkpoint, device, "auto")
    candidate_model, _candidate_payload, _ = load_reconstruction_model(args.candidate_checkpoint, device, "auto")
    ecgfounder = load_ecgfounder(device)
    tasks = load_tasks(args.tasks_path)

    baseline_name = sanitize_name(args.baseline_checkpoint)
    candidate_name = sanitize_name(args.candidate_checkpoint)
    baseline_tag = checkpoint_run_tag(args.baseline_checkpoint)
    candidate_tag = checkpoint_run_tag(args.candidate_checkpoint)
    output_dir = Path(args.output_root) / f"{baseline_tag}_vs_{candidate_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    ptbxl_summary = None
    sunnybrook_summary = None

    if not args.skip_ptbxl:
        ptbxl_summary, ptbxl_task_metrics, ptbxl_subset_definition, ptbxl_task_definition = run_ptbxl_audit(
            baseline_model=baseline_model,
            candidate_model=candidate_model,
            ecgfounder=ecgfounder,
            device=device,
            obs_indices=obs_indices,
            batch_size=args.batch_size,
            ptbxl_root=args.ptbxl_root,
            ptbxl_label_csv=args.ptbxl_label_csv,
            tasks=tasks,
            limit=args.ptbxl_limit,
            ptbxl_index_csv=args.ptbxl_index_csv,
            use_ptbxl_index=args.use_ptbxl_index,
            allow_generate_ptbxl_index=args.allow_generate_ptbxl_index,
            screen_subset_size=args.screen_subset_size,
            screen_seed=args.screen_seed,
            output_root=args.output_root,
            screen_min_class_count=args.screen_min_class_count,
        )
        pd.DataFrame([ptbxl_summary]).to_csv(output_dir / "ptbxl_summary.csv", index=False)
        ptbxl_task_metrics.to_csv(output_dir / "ptbxl_task_metrics.csv", index=False)
        ptbxl_subset_definition.to_csv(output_dir / "screen_subset_definition.csv", index=False)
        ptbxl_task_definition.to_csv(output_dir / "screen_task_definition.csv", index=False)
        pd.DataFrame(
            [
                {
                    "baseline_checkpoint": str(Path(args.baseline_checkpoint).resolve()),
                    "candidate_checkpoint": str(Path(args.candidate_checkpoint).resolve()),
                    "macro_auroc": ptbxl_summary.get("macro_auroc", math.nan),
                    "interest_macro_auroc": ptbxl_summary.get("interest_macro_auroc", math.nan),
                    "macro_ap": ptbxl_summary.get("macro_ap", math.nan),
                    "candidate_beats_baseline_on_valid_tasks": ptbxl_summary.get(
                        "candidate_beats_baseline_on_valid_tasks", 0
                    ),
                    "n_valid_auroc_tasks": ptbxl_summary.get("n_valid_auroc_tasks", 0),
                }
            ]
        ).to_csv(output_dir / "checkpoint_screen_summary.csv", index=False)
        ptbxl_interest = ptbxl_task_metrics[ptbxl_task_metrics["interest_task"]].copy()
        ptbxl_interest.sort_values("candidate_minus_baseline_auroc", ascending=False).to_csv(
            output_dir / "ptbxl_interest_task_metrics.csv", index=False
        )

    if not args.skip_sunnybrook:
        sunnybrook_summary, sunnybrook_records = run_sunnybrook_audit(
            baseline_model=baseline_model,
            candidate_model=candidate_model,
            ecgfounder=ecgfounder,
            device=device,
            obs_indices=obs_indices,
            batch_size=args.batch_size,
            sunnybrook_dir=args.sunnybrook_dir,
            tasks=tasks,
            limit=args.sunnybrook_limit,
        )
        pd.DataFrame([sunnybrook_summary]).to_csv(output_dir / "sunnybrook_summary.csv", index=False)
        sunnybrook_records.to_csv(output_dir / "sunnybrook_record_metrics.csv", index=False)

    write_interpretation(output_dir, baseline_name, candidate_name, ptbxl_summary, sunnybrook_summary)
    print(f"Saved ECGFounder audit to {output_dir}")


if __name__ == "__main__":
    main()
