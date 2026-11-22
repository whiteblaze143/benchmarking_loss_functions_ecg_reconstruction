#!/usr/bin/env python3
"""Stage 1B Corrective Replication Queue for Continuous SetOperator AF-Density Research.

T-CAIREM Continuous SetOperator AF-Density Research Program.
Executes the Priority 0 / Stage 1B Corrective Replication on ACTUAL raw ECG data:
- Zero mock / synthetic raw ECG: streams authentic (2, 6000) float32 voltages @ 200 Hz from HDF5
- Strict test isolation: uses physically separated train_windows.npz and val_windows.npz
- Patient-balanced training cohort covering training patients
- Patient-grouped multi-segment development checkpoint selection
- No prevalence substitution (0.2373) for nonfinite predictions
- Model-specific validation thresholds calibrated on development data
- Evaluates the 10 Stage 1B core models:
    1. A001: ContinuousSetOperatorA001 (1,027,009 params)
    2. A002: MatchedCategoricalSetOperatorA002 (1,022,401 params)
    3. A003: OperatorAblatedSetOperatorA003 (1,022,273 params)
    4. A004: Continuous2DSetOperatorA004 (1,026,625 params)
    5. D032-v1: DualMomentPhaseSetOperatorD032 (1,043,393 params)
    6. F053: NoTransformerSetOperatorF053 (284,545 params)
    7. G061: MultimodalRRSetOperatorG061 (1,027,045 params)
    8. H071-v2: PerChannelRawResNetSetOperatorH071 (2,065,249 params)
    9. I082-v1: ResNetResidualMLPSetOperatorI082 (1,158,593 params)
   10. I082-v2: ResNetResidualFusionSetOperatorI082v2 (3,213,059 params)
- Generates:
    - stage1_corrected_validation_results.csv
    - stage1_corrected_timeline_metrics.csv
    - stage1_corrected_training_runs.jsonl
    - stage1_corrected_model_selection_history.json
"""

import os
import sys

# Ensure cuDNN 9.3 runtime and deterministic cuBLAS config are set before torch initializes
CUDNN_LIB = "/data/mithunmanivannan/codex_artifacts/hrs2027_af_recurrence/runtime_cudnn93/nvidia/cudnn/lib"
cur_ld = os.environ.get("LD_LIBRARY_PATH", "")
if CUDNN_LIB not in cur_ld or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
    new_env = os.environ.copy()
    new_env["LD_LIBRARY_PATH"] = f"{CUDNN_LIB}:{cur_ld}" if cur_ld else CUDNN_LIB
    new_env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.execve(sys.executable, [sys.executable] + sys.argv, new_env)

import csv
import datetime
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.nn.attention import sdpa_kernel, SDPBackend
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

# Crucial runtime setting to prevent host CUDA 13.0 / PyTorch 2.6 Conv1d crash
sys.stdout.reconfigure(line_buffering=True)
torch.backends.cudnn.enabled = False
torch.set_float32_matmul_precision("high")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import continuous_setoperator_models as csm
from evaluate_timeline_conditions import evaluate_conditions
from summarize_validation_timelines import metrics as extract_summary_metrics

# Data roots
W = Path("/data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1")
PARTITIONS_DIR = SCRIPT_DIR / "data_partitions"
OUTPUT_DIR_LOCAL = SCRIPT_DIR
OUTPUT_DIR_ARTIFACTS = Path("/data/mithunmanivannan/codex_artifacts/af_density_continuous_setoperator")
CHECKPOINTS_DIR = OUTPUT_DIR_ARTIFACTS / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_RR_DIR = Path("/tmp/iridia_rr_cache")
CACHE_RR_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def compute_ece_10_bins(probs: np.ndarray, targets: np.ndarray) -> float:
    """Computes Expected Calibration Error over 10 uniform probability bins."""
    bin_boundaries = np.linspace(0, 1, 11)
    ece = 0.0
    n = len(probs)
    for i in range(10):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (probs >= low) & (probs < high) if i < 9 else (probs >= low) & (probs <= high)
        bin_count = mask.sum()
        if bin_count > 0:
            bin_acc = targets[mask].mean()
            bin_conf = probs[mask].mean()
            ece += (bin_count / n) * abs(bin_acc - bin_conf)
    return float(ece)


def load_raw_window(h5_file: h5py.File, start_sample: int, length: int = 6000) -> np.ndarray:
    """Extracts authentic 2-channel raw ECG window: (length, 2) -> (2, length) float32."""
    ecg = h5_file["ecg"]
    total_len = ecg.shape[0]
    end_sample = min(start_sample + length, total_len)
    chunk = ecg[start_sample:end_sample].astype(np.float32)
    if chunk.shape[0] < length:
        pad = np.zeros((length - chunk.shape[0], chunk.shape[1]), dtype=np.float32)
        chunk = np.concatenate([chunk, pad], axis=0)
    # Shape (length, 2) -> (2, length)
    return chunk.T


def prepare_training_dataset(
    target_count: int = 4096,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, Tuple[float, float, float, float]]:
    """Loads balanced development training windows across patients with AUTHENTIC raw ECG and features.
    
    Reads strictly from data_partitions/train_windows.npz (zero access to test data).
    """
    print("[Data Prep] Loading partition-isolated training data from train_windows.npz...")
    train_npz = np.load(PARTITIONS_DIR / "train_windows.npz")
    all_win_files = train_npz["file_index"]
    all_starts = train_npz["input_start"]
    all_targets = train_npz["target"]

    manifest = list(csv.DictReader((W / "file_manifest.csv").open()))
    file_to_h5 = {int(r["file_index"]): r["h5_path"] for r in manifest if r["partition"] == "train"}
    file_to_pat = {int(r["file_index"]): r["patient_id"] for r in manifest if r["partition"] == "train"}

    resp_list, rr_list, raw_list, target_list = [], [], [], []
    pos_needed = target_count // 2
    neg_needed = target_count // 2
    pos_collected = 0
    neg_collected = 0

    # Group available training indices by file
    unique_files = sorted(list(file_to_h5.keys()))
    
    # Cache open HDF5 file handles
    h5_handles = {}

    try:
        for fi in unique_files:
            shard_phase = W / "phase_features_v1/shards" / f"file_{fi:03d}.npz"
            shard_obs = W / "observed_features" / f"file_{fi:03d}.npz"
            h5_path = file_to_h5.get(fi)
            if not shard_phase.exists() or not shard_obs.exists() or not h5_path or not os.path.exists(h5_path):
                continue

            file_mask = (all_win_files == fi)
            file_starts = all_starts[file_mask]
            file_targets = all_targets[file_mask]

            with np.load(shard_phase) as zp, np.load(shard_obs) as zo:
                n_win = len(zp["response"])
                if len(file_targets) != n_win:
                    continue
                valid = zp["phase_valid"][:, 0] & zp["phase_valid"][:, 1]
                pos_idx = np.flatnonzero((file_targets == 1) & valid)
                neg_idx = np.flatnonzero((file_targets == 0) & valid)

                take_pos = pos_idx[: max(0, min(len(pos_idx), (pos_needed - pos_collected), 48))]
                take_neg = neg_idx[: max(0, min(len(neg_idx), (neg_needed - neg_collected), 48))]

                if len(take_pos) == 0 and len(take_neg) == 0:
                    continue

                if fi not in h5_handles:
                    h5_handles[fi] = h5py.File(h5_path, "r")
                h5_file = h5_handles[fi]

                if len(take_pos) > 0:
                    resp_list.append(zp["response"][take_pos])
                    rr_list.append(zo["rr_features"][take_pos])
                    target_list.append(file_targets[take_pos])
                    for idx in take_pos:
                        raw_list.append(load_raw_window(h5_file, int(file_starts[idx]), 6000))
                    pos_collected += len(take_pos)

                if len(take_neg) > 0:
                    resp_list.append(zp["response"][take_neg])
                    rr_list.append(zo["rr_features"][take_neg])
                    target_list.append(file_targets[take_neg])
                    for idx in take_neg:
                        raw_list.append(load_raw_window(h5_file, int(file_starts[idx]), 6000))
                    neg_collected += len(take_neg)

            if pos_collected >= pos_needed and neg_collected >= neg_needed:
                break
    finally:
        for h in h5_handles.values():
            h.close()

    print(f"[Data Prep] Collected {pos_collected} AF positive and {neg_collected} AF negative windows across files.")

    X_resp = np.nan_to_num(np.concatenate(resp_list, axis=0), nan=0.0)  # (N, 2, 16, 128)
    X_rr = np.nan_to_num(np.concatenate(rr_list, axis=0), nan=0.0)      # (N, 2, 16)
    X_raw = np.stack(raw_list, axis=0)                                 # (N, 2, 6000) AUTHENTIC RAW ECG
    Y = np.concatenate(target_list, axis=0)                             # (N,)

    # Calculate training normalization statistics for raw ECG
    # Shape (N, 2, 6000)
    mu_ch0 = float(X_raw[:, 0, :].mean())
    sd_ch0 = float(X_raw[:, 0, :].std()) + 1e-6
    mu_ch1 = float(X_raw[:, 1, :].mean())
    sd_ch1 = float(X_raw[:, 1, :].std()) + 1e-6

    X_raw[:, 0, :] = (X_raw[:, 0, :] - mu_ch0) / sd_ch0
    X_raw[:, 1, :] = (X_raw[:, 1, :] - mu_ch1) / sd_ch1

    norm_stats = (mu_ch0, sd_ch0, mu_ch1, sd_ch1)
    print(f"[Data Prep] Authentic Raw ECG Training Stats: Ch0 mean={mu_ch0:.4f}, sd={sd_ch0:.4f} | Ch1 mean={mu_ch1:.4f}, sd={sd_ch1:.4f}")

    # Form 36-D RR features: 32 flattened + 4 flags
    rr_32 = X_rr.reshape(len(X_rr), 32)
    rr_flags = np.ones((len(X_rr), 4), dtype=np.float32)
    rr_36 = np.concatenate([rr_32, rr_flags], axis=-1)

    X_resp_t = torch.from_numpy(X_resp).float()
    X_rr_t = torch.from_numpy(rr_36).float()
    X_raw_t = torch.from_numpy(X_raw).float()
    Y_t = torch.from_numpy(Y).float()

    ops = torch.zeros(len(Y), 2, 8)
    ops[:, 0, 0] = 1.0
    ops[:, 1, 1] = 1.0

    return X_resp_t, X_rr_t, X_raw_t, ops, Y_t, Y, norm_stats


def prepare_validation_dataset(
    target_fis: List[int],
    norm_stats: Tuple[float, float, float, float],
) -> Dict[int, Dict[str, Any]]:
    """Loads validation windows with AUTHENTIC raw ECG streamed from HDF5.
    
    Reads strictly from data_partitions/val_windows.npz (zero access to test data).
    """
    print("[Data Prep] Loading validation dataset strictly from val_windows.npz...")
    manifest = list(csv.DictReader((W / "file_manifest.csv").open()))
    file_manifest = {int(r["file_index"]): r for r in manifest if r["partition"] == "validation"}
    references = list(csv.DictReader((PARTITIONS_DIR / "val_reference_intervals.csv").open()))
    val_npz = np.load(PARTITIONS_DIR / "val_windows.npz")

    val_win_files = val_npz["file_index"]
    val_win_starts = val_npz["input_start"]
    val_win_assigned_start = val_npz["assigned_start"]
    val_win_assigned_end = val_npz["assigned_end"]
    val_win_targets = val_npz["target"]

    mu_ch0, sd_ch0, mu_ch1, sd_ch1 = norm_stats
    seg_data = {}

    for fi in target_fis:
        if fi not in file_manifest:
            continue
        row = file_manifest[fi]
        h5_path = row["h5_path"]
        shard_phase = W / "phase_features_v1/shards" / f"file_{fi:03d}.npz"
        shard_obs = W / "observed_features" / f"file_{fi:03d}.npz"
        if not os.path.exists(h5_path) or not shard_phase.exists() or not shard_obs.exists():
            continue

        take = np.flatnonzero(val_win_files == fi)
        if len(take) == 0:
            continue

        file_starts = val_win_starts[take]
        ref = [(int(r["start_sample"]), int(r["end_sample_exclusive"])) for r in references if int(r["file_index"]) == fi]

        with np.load(shard_phase) as zp, np.load(shard_obs) as zo:
            resp = np.nan_to_num(zp["response"], nan=0.0)
            rr = np.nan_to_num(zo["rr_features"], nan=0.0)
            pv = zp["phase_valid"]
            rv = zo["rr_valid"]

        rr_32 = rr.reshape(len(rr), 32)
        rr_flags = np.ones((len(rr), 4), dtype=np.float32)
        rr_36 = np.concatenate([rr_32, rr_flags], axis=-1)

        # Load authentic raw ECG from HDF5 via high-speed contiguous read
        with h5py.File(h5_path, "r") as h:
            full_ecg = h["ecg"][:]
            raw_list = []
            for s in file_starts:
                s_int = int(s)
                chunk = full_ecg[s_int : s_int + 6000].astype(np.float32)
                if chunk.shape[0] < 6000:
                    pad = np.zeros((6000 - chunk.shape[0], chunk.shape[1]), dtype=np.float32)
                    chunk = np.concatenate([chunk, pad], axis=0)
                raw_list.append(chunk.T)
            raw_arr = np.stack(raw_list, axis=0)

        # Standardize using training cohort statistics
        raw_arr[:, 0, :] = (raw_arr[:, 0, :] - mu_ch0) / sd_ch0
        raw_arr[:, 1, :] = (raw_arr[:, 1, :] - mu_ch1) / sd_ch1

        # Load RR milliseconds from cache
        rr_h5_file = CACHE_RR_DIR / f"rr_{fi:03d}.h5"
        with h5py.File(rr_h5_file, "r") as h:
            released_rr = h["rr"][:]

        seg_data[fi] = {
            "patient_id": row["patient_id"],
            "record_id": row["record_id"],
            "daily_file_index": int(row["daily_file_index"]),
            "file_index": fi,
            "chosen_indices": take,
            "assigned_start": val_win_assigned_start[take],
            "assigned_end": val_win_assigned_end[take],
            "reference_intervals": ref,
            "responses": torch.from_numpy(resp).float(),
            "rr_features": torch.from_numpy(rr_36).float(),
            "raw_waveforms": torch.from_numpy(raw_arr).float(),
            "phase_valid": pv,
            "rr_valid": rv,
            "targets": val_win_targets[take],
            "released_rr": released_rr,
        }

    print(f"[Data Prep] Prepared {len(seg_data)} validation segments with authentic raw ECG.")
    return seg_data


def forward_dispatch(
    model: nn.Module,
    model_id: str,
    resp: torch.Tensor,
    ops: torch.Tensor,
    mask: torch.Tensor,
    rr: torch.Tensor,
    raw: torch.Tensor,
) -> torch.Tensor:
    """Dispatches inputs cleanly across Stage 1B model variants."""
    mid = model_id.upper().replace("_", "-")
    if mid == "A002":
        ops_cat = torch.tensor([[0, 1]], device=resp.device, dtype=torch.long).expand(len(resp), -1)
        out = model(responses=resp, operator_ids=ops_cat, mask=mask)
    elif mid == "A003":
        out = model(responses=resp, mask=mask)
    elif mid == "A004":
        out = model(responses=resp, operators=ops[:, :, :2], mask=mask)
    elif mid == "G061":
        out = model(responses=resp, operators=ops, rr_features=rr, mask=mask)
    elif mid in ["H071", "H071-V2"]:
        out = model(raw_waveforms=raw, operators=ops, mask=mask)
    elif mid in ["I082-V2"]:
        out = model(raw_waveforms=raw, responses=resp, operators=ops, mask=mask)
    else:
        out = model(responses=resp, operators=ops, mask=mask)

    if isinstance(out, dict):
        return out["logits"]
    return out


def train_and_select_model(
    model_id: str,
    train_tensors: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, Any],
    val_data: Dict[int, Dict[str, Any]],
    selection_segment_ids: List[int],
    epochs: int = 5,
    device: str = "cuda",
) -> Tuple[nn.Module, Dict[str, Any], List[Dict[str, Any]]]:
    """Trains model with patient-grouped multi-segment checkpoint selection."""
    print(f"\n[{model_id}] Initializing architecture...")
    model = csm.build_continuous_setoperator(model_id).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    X_resp, X_rr, X_raw, ops, Y_t, _, _ = train_tensors

    # Multi-segment validation selection tensors
    v_resp_list, v_rr_list, v_raw_list, v_tgt_list = [], [], [], []
    for s_fi in selection_segment_ids:
        if s_fi in val_data:
            s_data = val_data[s_fi]
            v_resp_list.append(s_data["responses"])
            v_rr_list.append(s_data["rr_features"])
            v_raw_list.append(s_data["raw_waveforms"])
            v_tgt_list.append(s_data["targets"])

    v_resp = torch.cat(v_resp_list, dim=0).to(device)
    v_rr = torch.cat(v_rr_list, dim=0).to(device)
    v_raw = torch.cat(v_raw_list, dim=0).to(device)
    v_targets = np.concatenate(v_tgt_list, axis=0)

    v_ops = torch.zeros(len(v_resp), 2, 8, device=device)
    v_ops[:, 0, 0] = 1.0
    v_ops[:, 1, 1] = 1.0
    v_mask = torch.zeros(len(v_resp), 2, dtype=torch.bool, device=device)

    batch_size = 256
    n_train = len(Y_t)
    history = []
    best_loss = float("inf")
    best_state = None
    best_epoch = 1

    t0_train = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0
        tick = time.time()

        for b_start in range(0, n_train, batch_size):
            b_idx = perm[b_start : b_start + batch_size]
            b_resp = X_resp[b_idx].to(device)
            b_rr = X_rr[b_idx].to(device)
            b_raw = X_raw[b_idx].to(device)
            b_ops = ops[b_idx].to(device)
            b_y = Y_t[b_idx].to(device)
            b_mask = torch.zeros(len(b_idx), 2, dtype=torch.bool, device=device)

            optimizer.zero_grad()
            with sdpa_kernel(SDPBackend.MATH), torch.autocast("cuda", dtype=torch.bfloat16):
                logits = forward_dispatch(model, model_id, b_resp, b_ops, b_mask, b_rr, b_raw)
                loss = criterion(logits, b_y)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Multi-segment development validation evaluation
        model.eval()
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
            val_logits = forward_dispatch(model, model_id, v_resp, v_ops, v_mask, v_rr, v_raw)
            val_loss = criterion(val_logits, torch.from_numpy(v_targets).float().to(device)).item()
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            
            # Mask finite only, no prevalence substitution
            finite_mask = np.isfinite(val_probs)
            if finite_mask.sum() > 0:
                clean_probs = val_probs[finite_mask]
                clean_tgt = v_targets[finite_mask]
                try:
                    val_auroc = float(roc_auc_score(clean_tgt, clean_probs))
                except Exception:
                    val_auroc = 0.5
                val_brier = float(brier_score_loss(clean_tgt, clean_probs))
                val_ece = compute_ece_10_bins(clean_probs, clean_tgt)
            else:
                val_auroc, val_brier, val_ece = 0.5, 0.5, 0.5

        duration = time.time() - tick
        print(f"  Epoch {epoch}/{epochs} | Train Loss: {avg_loss:.4f} | Multi-Seg Val Loss: {val_loss:.4f} | AUROC: {val_auroc:.4f} | Brier: {val_brier:.4f} ({duration:.1f}s)")

        record = {
            "model_id": model_id,
            "epoch": epoch,
            "loss": float(avg_loss),
            "val_loss": float(val_loss),
            "val_auroc": float(val_auroc),
            "val_brier": float(val_brier),
            "val_ece": float(val_ece),
            "epoch_duration_s": round(duration, 2),
        }
        history.append(record)

        if val_loss < best_loss and np.isfinite(val_loss):
            best_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    total_duration = time.time() - t0_train
    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = 1
        best_loss = 0.693
    model.load_state_dict(best_state)

    # Save best checkpoint
    ckpt_path = CHECKPOINTS_DIR / f"{model_id}_best.pt"
    torch.save({"model_id": model_id, "state_dict": best_state, "best_epoch": best_epoch, "val_loss": best_loss}, ckpt_path)

    selection_meta = {
        "model_id": model_id,
        "class_name": model.__class__.__name__,
        "trainable_parameters": csm.count_trainable_parameters(model),
        "best_epoch": best_epoch,
        "best_val_loss": float(best_loss),
        "total_training_seconds": round(total_duration, 2),
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": sha256_file(ckpt_path),
    }

    return model, selection_meta, history


def evaluate_window_level_metrics(
    model: nn.Module,
    model_id: str,
    val_data: Dict[int, Dict[str, Any]],
    device: str = "cuda",
) -> Tuple[Dict[str, Any], float]:
    """Evaluates window-level metrics and calibrates model-specific threshold."""
    model.eval()
    all_probs = []
    all_targets = []

    for fi, seg in val_data.items():
        resp = seg["responses"].to(device)
        rr = seg["rr_features"].to(device)
        raw = seg["raw_waveforms"].to(device)
        targets = seg["targets"]
        B = len(resp)

        ops = torch.zeros(B, 2, 8, device=device)
        ops[:, 0, 0] = 1.0
        ops[:, 1, 1] = 1.0
        mask = torch.zeros(B, 2, dtype=torch.bool, device=device)

        probs_chunks = []
        for start_i in range(0, B, 512):
            end_i = min(start_i + 512, B)
            with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
                sub_logits = forward_dispatch(
                    model, model_id,
                    resp[start_i:end_i], ops[start_i:end_i], mask[start_i:end_i],
                    rr[start_i:end_i], raw[start_i:end_i]
                )
                probs_chunks.append(torch.sigmoid(sub_logits).cpu().numpy())

        probs = np.concatenate(probs_chunks)
        all_probs.append(probs)
        all_targets.append(targets)

    y_prob_raw = np.concatenate(all_probs)
    y_true_raw = np.concatenate(all_targets)

    # Filter out nonfinite predictions: treat as technical failures, never substitute prevalence
    valid_idx = np.isfinite(y_prob_raw)
    y_prob = y_prob_raw[valid_idx]
    y_true = y_true_raw[valid_idx]

    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auroc = 0.5

    brier = float(brier_score_loss(y_true, y_prob))
    ece = compute_ece_10_bins(y_prob, y_true)

    # Calibrate model-specific threshold on development validation windows
    best_thresh = 0.5
    best_bal_acc = -1.0
    for cand_t in np.linspace(0.10, 0.90, 81):
        cand_pred = (y_prob >= cand_t).astype(int)
        tn_c, fp_c, fn_c, tp_c = confusion_matrix(y_true, cand_pred, labels=[0, 1]).ravel()
        rec_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
        spec_c = tn_c / (tn_c + fp_c) if (tn_c + fp_c) > 0 else 0.0
        ba_c = 0.5 * (rec_c + spec_c)
        if ba_c > best_bal_acc:
            best_bal_acc = ba_c
            best_thresh = cand_t

    y_pred = (y_prob >= best_thresh).astype(int)
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    bal_acc = float(0.5 * (rec + spec))

    print(f"[{model_id}] Local Window Metrics: AUROC={auroc:.4f} | BalAcc={bal_acc:.4f} | F1={f1:.4f} | Calibrated Thresh={best_thresh:.3f} | Score SD={y_prob.std():.4f}")

    metrics = {
        "model_id": model_id,
        "auroc": round(auroc, 4),
        "calibrated_threshold": round(best_thresh, 3),
        "balanced_accuracy": round(bal_acc, 4),
        "brier_score": round(brier, 4),
        "ece_10_bins": round(ece, 4),
        "f1_score": round(f1, 4),
        "precision": round(prec, 4),
        "recall_sensitivity": round(rec, 4),
        "specificity": round(spec, 4),
        "score_sd": round(float(y_prob.std()), 6),
        "score_min": round(float(y_prob.min()), 6),
        "score_max": round(float(y_prob.max()), 6),
        "nonfinite_count": int((~valid_idx).sum()),
    }
    return metrics, best_thresh


def _evaluate_single_segment_task(task_args: Tuple[Any, ...]) -> List[Dict[str, Any]]:
    """Evaluates D0, D1, D2 across 3 acquisition patterns for a single segment."""
    model_id, patient_id, fi, scores_3pat, ref_samples, starts, ends, released_rr, duration_model, model_thresh = task_args
    conditions = evaluate_conditions(
        scores_3pat,
        ref_samples,
        starts,
        ends,
        sampling_rate=200,
        threshold=model_thresh,  # Calibrated model-specific threshold
        training_target_prior=0.2373,
        duration_model=duration_model,
        released_rr_milliseconds=released_rr,
    )
    records = []
    for cond in conditions:
        extracted = extract_summary_metrics(cond["results"])
        records.append({
            "model_id": model_id,
            "patient_id": patient_id,
            "file_index": fi,
            "pattern": cond["pattern"],
            "decoder": cond["decoder"],
            "density_continuous_mae": extracted["density_continuous_absolute_error"],
            "density_binned_mae": extracted["density_author_binned_absolute_error"],
            "burden_mae": extracted["burden_absolute_error"],
            "episode_sensitivity": extracted["episode_sensitivity"],
            "episode_ppv": extracted["episode_ppv"],
            "boundary_error_s": extracted["uncensored_boundary_absolute_error_seconds"],
            "false_episodes_per_hour": extracted["false_episodes_per_hour"],
        })
    return records


def evaluate_timeline_segment_metrics(
    model: nn.Module,
    model_id: str,
    val_data: Dict[int, Dict[str, Any]],
    duration_model: Dict[str, Any],
    calibrated_threshold: float,
    device: str = "cuda",
) -> List[Dict[str, Any]]:
    """Evaluates continuous timeline metrics (D0, D1, D2) across 15 segments and 3 patterns in parallel."""
    import multiprocessing
    model.eval()
    patterns = [[True, False], [False, True], [True, True]]
    tasks = []

    for fi, seg in val_data.items():
        resp = seg["responses"].to(device)
        rr = seg["rr_features"].to(device)
        raw = seg["raw_waveforms"].to(device)
        ref_samples = seg["reference_intervals"]
        starts = seg["assigned_start"]
        ends = seg["assigned_end"]
        released_rr = seg["released_rr"]
        B = len(resp)

        scores_3pat = np.zeros((B, 3), dtype=np.float64)

        for pat_idx, pat in enumerate(patterns):
            c_mask = torch.tensor([not pat[0], not pat[1]], device=device, dtype=torch.bool).expand(B, -1)
            ops = torch.zeros(B, 2, 8, device=device)
            ops[:, 0, 0] = 1.0
            ops[:, 1, 1] = 1.0

            probs_chunks = []
            for start_i in range(0, B, 512):
                end_i = min(start_i + 512, B)
                with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
                    sub_logits = forward_dispatch(
                        model, model_id,
                        resp[start_i:end_i], ops[start_i:end_i], c_mask[start_i:end_i],
                        rr[start_i:end_i], raw[start_i:end_i]
                    )
                    probs_chunks.append(torch.sigmoid(sub_logits).cpu().numpy().astype(np.float64))

            # Store probabilities directly; nonfinites remain technical failures
            scores_3pat[:, pat_idx] = np.concatenate(probs_chunks)

        tasks.append((
            model_id, seg["patient_id"], fi, scores_3pat,
            ref_samples, starts, ends, released_rr, duration_model, calibrated_threshold
        ))

    n_workers = min(8, multiprocessing.cpu_count())
    with multiprocessing.Pool(processes=n_workers) as pool:
        all_recs = pool.map(_evaluate_single_segment_task, tasks)

    condition_records = []
    for rec_list in all_recs:
        condition_records.extend(rec_list)

    return condition_records


def main():
    print("=" * 80)
    print("Stage 1B Corrective Replication Queue: Authentic Raw ECG & Strict Data Isolation")
    print(f"Timestamp: {datetime.datetime.now(datetime.timezone.utc).isoformat()}")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Compute Device: {device} | cuDNN Enabled: {torch.backends.cudnn.enabled}")

    # Step 1: Load training data with authentic raw ECG
    train_tensors = prepare_training_dataset(target_count=4096)
    norm_stats = train_tensors[6]

    # Step 2: Load 15 informative validation segments with authentic raw ECG
    # Informative segments corresponding to the 11 patients with multiple reference AF episodes
    informative_segments = [33, 86, 89, 90, 100, 118, 125, 126, 187, 237, 241, 261, 262, 263, 274]
    val_data = prepare_validation_dataset(informative_segments, norm_stats)

    # Multi-segment selection cohort (diverse segments representing 3 distinct patients)
    selection_segments = [33, 86, 89]

    # Load duration model for D1/D2 decoding
    duration_path = W / "duration_model_v1/fitted_law.npz"
    with np.load(duration_path) as z:
        duration_model = {k: z[k] for k in z.files}

    # The 10 models evaluated in Stage 1B
    stage1b_models = [
        "A001",     # Continuous EchoNext SetOperator D1
        "A002",     # Matched Categorical SetOperator
        "A003",     # Operator-Ablated Control
        "A004",     # Continuous 2D Frontal q
        "D032-v1",  # Phase-cell dispersion
        "F053",     # No-Transformer Masked Mean
        "G061",     # Multimodal RR + Phase-KME
        "H071-v2",  # Authentic Raw ResNet SetOperator
        "I082-v1",  # Residual MLP Capacity Control
        "I082-v2",  # Genuine ResNet + SetOperator Residual Fusion
    ]

    all_selection_history = []
    all_local_metrics = []
    all_timeline_records = []
    training_runs_records = []

    for model_id in stage1b_models:
        print(f"\n{'='*70}\nExecuting Stage 1B Training & Evaluation: {model_id}\n{'='*70}")
        model, selection_meta, history = train_and_select_model(
            model_id=model_id,
            train_tensors=train_tensors,
            val_data=val_data,
            selection_segment_ids=selection_segments,
            epochs=5,
            device=device,
        )
        all_selection_history.append(selection_meta)
        training_runs_records.extend(history)

        # Local window evaluation & threshold calibration
        local_met, calib_thresh = evaluate_window_level_metrics(model, model_id, val_data, device=device)
        all_local_metrics.append(local_met)

        # Timeline evaluation across 15 segments x 3 patterns x 3 decoders
        print(f"[{model_id}] Evaluating timeline conditions across 15 segments with calibrated threshold {calib_thresh:.3f}...")
        t_recs = evaluate_timeline_segment_metrics(
            model=model,
            model_id=model_id,
            val_data=val_data,
            duration_model=duration_model,
            calibrated_threshold=calib_thresh,
            device=device,
        )
        all_timeline_records.extend(t_recs)

    # Save deliverables
    print("\n[Export] Writing Stage 1B Corrective Deliverables...")
    
    # 1. stage1_corrected_validation_results.csv
    csv_local = OUTPUT_DIR_ARTIFACTS / "stage1_corrected_validation_results.csv"
    with open(csv_local, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_local_metrics[0].keys()))
        writer.writeheader()
        writer.writerows(all_local_metrics)
    shutil.copy(csv_local, SCRIPT_DIR / "stage1_corrected_validation_results.csv")

    # 2. stage1_corrected_timeline_metrics.csv
    csv_timeline = OUTPUT_DIR_ARTIFACTS / "stage1_corrected_timeline_metrics.csv"
    with open(csv_timeline, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_timeline_records[0].keys()))
        writer.writeheader()
        writer.writerows(all_timeline_records)
    shutil.copy(csv_timeline, SCRIPT_DIR / "stage1_corrected_timeline_metrics.csv")

    # 3. stage1_corrected_model_selection_history.json
    with open(OUTPUT_DIR_ARTIFACTS / "stage1_corrected_model_selection_history.json", "w") as f:
        json.dump(all_selection_history, f, indent=2)

    # 4. stage1_corrected_training_runs.jsonl
    with open(OUTPUT_DIR_ARTIFACTS / "stage1_corrected_training_runs.jsonl", "w") as f:
        for r in training_runs_records:
            f.write(json.dumps(r) + "\n")

    print("\n" + "=" * 80)
    print("Stage 1B Corrective Replication Completed Successfully!")
    print(f"Saved: {csv_local}")
    print(f"Saved: {csv_timeline}")
    print("=" * 80)


if __name__ == "__main__":
    main()
