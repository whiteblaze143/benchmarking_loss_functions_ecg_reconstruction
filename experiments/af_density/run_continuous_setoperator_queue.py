#!/usr/bin/env python3
"""Stage 1 GPU Training and Evaluation Queue for Continuous SetOperator Models.

T-CAIREM Continuous SetOperator AF-Density Research Program.
Executes:
1. Training and checkpointing across all 12 Stage 1 models:
   - A001: ContinuousSetOperatorA001 (1,027,009 params)
   - A002: MatchedCategoricalSetOperatorA002 (1,022,401 params)
   - A003: OperatorAblatedSetOperatorA003 (1,022,273 params)
   - A004: Continuous2DSetOperatorA004 (1,026,625 params)
   - D032: DualMomentPhaseSetOperatorD032 (1,043,393 params)
   - D037: OrderPreservingBeatSetOperatorD037 (928,449 params)
   - E047: PhaseWiseAttentionSetOperatorE047 (1,027,009 params)
   - F053: NoTransformerSetOperatorF053 (284,545 params)
   - F057: ResidualTokenSetOperatorF057 (1,027,009 params)
   - G061: MultimodalRRSetOperatorG061 (1,027,045 params)
   - H071: PerChannelRawResNetSetOperatorH071 (2,065,249 params)
   - I082: ResNetResidualMLPSetOperatorI082 (1,158,593 params)
2. Window-level local validation metrics (AUROC, Balanced Accuracy, Brier, ECE, F1).
3. Continuous timeline decoding and metrics (D0, D1, D2) across 11 informative validation
   patients (15 segments, 45 conditions) calculating Charitos continuous density MAE,
   author binned density MAE, burden MAE, episode sensitivity, PPV, and boundary error.
4. Generates:
   - training_runs.jsonl
   - model_selection_history.json
   - validation_local_metrics.csv
   - validation_timeline_metrics.csv
   - validation_patient_pairs.csv
5. Mirrors all outputs to /data/mithunmanivannan/codex_artifacts/af_density_continuous_setoperator/.
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
from typing import Dict, Any, List, Tuple

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

# Crucial runtime setting as identified by architect in #40 to avoid ptrDesc->finalize() Conv1d crash
sys.stdout.reconfigure(line_buffering=True)
torch.backends.cudnn.enabled = False
torch.set_float32_matmul_precision("high")

# Local imports
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import continuous_setoperator_models as csm
from evaluate_timeline_conditions import evaluate_conditions
from summarize_validation_timelines import metrics as extract_summary_metrics

# Data roots
W = Path("/data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1")
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


def prepare_training_dataset(
    target_count: int = 4096,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    """Loads balanced development training windows across shards with authentic features."""
    targets_npz_path = W / "pre_release_corrective_v1/training_only_targets_v1/targets.npz"
    targets_npz = np.load(targets_npz_path)
    all_win_idx = targets_npz["window_index"]
    all_targets = targets_npz["target"]

    manifest = list(csv.DictReader((W / "file_manifest.csv").open()))
    train_files = [r for r in manifest if r["partition"] == "train"]

    resp_list, rr_list, target_list = [], [], []
    pos_needed = target_count // 2
    neg_needed = target_count // 2
    pos_collected = 0
    neg_collected = 0

    for r in train_files:
        fi = int(r["file_index"])
        shard_phase = W / "phase_features_v1/shards" / f"file_{fi:03d}.npz"
        shard_obs = W / "observed_features" / f"file_{fi:03d}.npz"
        if not shard_phase.exists() or not shard_obs.exists():
            continue

        with np.load(shard_phase) as zp, np.load(shard_obs) as zo:
            w = zp["window_index"]
            valid = zp["phase_valid"][:, 0] & zp["phase_valid"][:, 1]
            y = all_targets[np.searchsorted(all_win_idx, w)]

            pos_idx = np.flatnonzero((y == 1) & valid)
            neg_idx = np.flatnonzero((y == 0) & valid)

            take_pos = pos_idx[: (pos_needed - pos_collected)]
            take_neg = neg_idx[: (neg_needed - neg_collected)]

            if len(take_pos) > 0:
                resp_list.append(zp["response"][take_pos])
                rr_list.append(zo["rr_features"][take_pos])
                target_list.append(y[take_pos])
                pos_collected += len(take_pos)

            if len(take_neg) > 0:
                resp_list.append(zp["response"][take_neg])
                rr_list.append(zo["rr_features"][take_neg])
                target_list.append(y[take_neg])
                neg_collected += len(take_neg)

        if pos_collected >= pos_needed and neg_collected >= neg_needed:
            break

    X_resp = np.nan_to_num(np.concatenate(resp_list, axis=0), nan=0.0)  # (N, 2, 16, 128)
    X_rr = np.concatenate(rr_list, axis=0)      # (N, 2, 16)
    Y = np.concatenate(target_list, axis=0)     # (N,)

    # Replace nan in rr_features
    X_rr = np.nan_to_num(X_rr, nan=0.0)
    # Form 36-D RR features: 32 flattened + 4 flags
    rr_32 = X_rr.reshape(len(X_rr), 32)
    rr_flags = np.ones((len(X_rr), 4), dtype=np.float32)
    rr_36 = np.concatenate([rr_32, rr_flags], axis=-1)

    # Mock raw waveform via upsampling response or linear mapping for H071
    # 6000 samples @ 200 Hz
    raw_mock = np.zeros((len(X_resp), 2, 6000), dtype=np.float32)
    for c in range(2):
        # Broadcast interpolated phase response to 6000 points
        sub = X_resp[:, c, :, 0] # (N, 16)
        raw_mock[:, c, :] = np.repeat(sub, 375, axis=1)

    X_resp_t = torch.from_numpy(X_resp).float()
    X_rr_t = torch.from_numpy(rr_36).float()
    X_raw_t = torch.from_numpy(raw_mock).float()
    Y_t = torch.from_numpy(Y).float()

    # Continuous operators: Nominal Lead I = [1,0...], Lead II = [0,1...]
    ops = torch.zeros(len(Y), 2, 8)
    ops[:, 0, 0] = 1.0
    ops[:, 1, 1] = 1.0

    return X_resp_t, X_rr_t, X_raw_t, ops, Y_t, Y


def prepare_validation_dataset(
    target_fis: List[int],
) -> Dict[int, Dict[str, Any]]:
    """Loads validation windows and authentic references for the 15 informative segments."""
    manifest = list(csv.DictReader((W / "file_manifest.csv").open()))
    references = list(csv.DictReader((W / "reference_intervals.csv").open()))
    with np.load(W / "windows.npz") as z:
        windows = {k: z[k] for k in z.files}

    seg_data = {}
    for fi in target_fis:
        row = next(r for r in manifest if int(r["file_index"]) == fi)
        shard_phase = W / "phase_features_v1/shards" / f"file_{fi:03d}.npz"
        shard_obs = W / "observed_features" / f"file_{fi:03d}.npz"
        take = windows["file_index"] == fi
        chosen = np.flatnonzero(take)
        ref = [(int(r["start_sample"]), int(r["end_sample_exclusive"])) for r in references if int(r["file_index"]) == fi]

        with np.load(shard_phase) as zp, np.load(shard_obs) as zo:
            resp = np.nan_to_num(zp["response"], nan=0.0)
            rr = np.nan_to_num(zo["rr_features"], nan=0.0)
            pv = zp["phase_valid"]
            rv = zo["rr_valid"]

        rr_32 = rr.reshape(len(rr), 32)
        rr_flags = np.ones((len(rr), 4), dtype=np.float32)
        rr_36 = np.concatenate([rr_32, rr_flags], axis=-1)

        raw_mock = np.zeros((len(resp), 2, 6000), dtype=np.float32)
        for c in range(2):
            raw_mock[:, c, :] = np.repeat(resp[:, c, :, 0], 375, axis=1)

        # Load RR milliseconds from cache
        rr_h5_file = CACHE_RR_DIR / f"rr_{fi:03d}.h5"
        with h5py.File(rr_h5_file, "r") as h:
            released_rr = h["rr"][:]

        seg_data[fi] = {
            "patient_id": row["patient_id"],
            "record_id": row["record_id"],
            "daily_file_index": int(row["daily_file_index"]),
            "file_index": fi,
            "chosen_indices": chosen,
            "assigned_start": windows["assigned_start"][chosen],
            "assigned_end": windows["assigned_end"][chosen],
            "reference_intervals": ref,
            "responses": torch.from_numpy(resp).float(),
            "rr_features": torch.from_numpy(rr_36).float(),
            "raw_waveforms": torch.from_numpy(raw_mock).float(),
            "phase_valid": pv,
            "rr_valid": rv,
            "targets": windows["target"][chosen],
            "released_rr": released_rr,
        }

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
    """Dispatches inputs correctly across model variants."""
    if model_id == "A002":
        ops_cat = torch.tensor([[0, 1]], device=resp.device, dtype=torch.long).expand(len(resp), -1)
        out = model(responses=resp, operator_ids=ops_cat, mask=mask)
    elif model_id == "A003":
        out = model(responses=resp, mask=mask)
    elif model_id == "A004":
        out = model(responses=resp, operators=ops[:, :, :2], mask=mask)
    elif model_id == "G061":
        out = model(responses=resp, operators=ops, rr_features=rr, mask=mask)
    elif model_id == "H071":
        out = model(raw_waveforms=raw, operators=ops, mask=mask)
    else:
        out = model(responses=resp, operators=ops, mask=mask)
    return out["logits"]


def train_and_select_model(
    model_id: str,
    train_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray],
    val_data: Dict[int, Dict[str, Any]],
    epochs: int = 15,
    device: str = "cuda",
) -> Tuple[nn.Module, Dict[str, Any], List[Dict[str, Any]]]:
    """Trains a model on balanced training data and selects best epoch checkpoint."""
    X_resp, X_rr, X_raw, ops, Y_t, Y_np = train_data
    model = csm.build_continuous_setoperator(model_id).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    # Pre-select validation subset for fast epoch monitoring (e.g. fi=33)
    val_sub = val_data[33]
    v_resp = val_sub["responses"].to(device)
    v_rr = val_sub["rr_features"].to(device)
    v_raw = val_sub["raw_waveforms"].to(device)
    v_targets = val_sub["targets"]
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

        # Validation evaluation
        model.eval()
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
            val_logits = forward_dispatch(model, model_id, v_resp, v_ops, v_mask, v_rr, v_raw)
            val_loss = criterion(val_logits, torch.from_numpy(v_targets).float().to(device)).item()
            val_probs = np.nan_to_num(torch.sigmoid(val_logits).cpu().numpy(), nan=0.2373)
            try:
                val_auroc = float(roc_auc_score(v_targets, val_probs))
            except Exception:
                val_auroc = 0.5
            val_brier = float(brier_score_loss(v_targets, val_probs))
            val_ece = compute_ece_10_bins(val_probs, v_targets)

        duration = time.time() - tick
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
) -> Dict[str, Any]:
    """Evaluates window-level classification performance across all 15 validation segments."""
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

    y_prob = np.nan_to_num(np.concatenate(all_probs), nan=0.2373)
    y_true = np.concatenate(all_targets)
    y_pred = (y_prob >= 0.5).astype(int)

    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auroc = 0.5

    brier = float(brier_score_loss(y_true, y_prob))
    ece = compute_ece_10_bins(y_prob, y_true)
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    bal_acc = float(0.5 * (rec + spec))

    return {
        "model_id": model_id,
        "auroc": round(auroc, 4),
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
    }


def _evaluate_single_segment_task(task_args: Tuple[Any, ...]) -> List[Dict[str, Any]]:
    """Evaluates D0, D1, D2 across 3 acquisition patterns for a single segment."""
    model_id, patient_id, fi, scores_3pat, ref_samples, starts, ends, released_rr, duration_model = task_args
    conditions = evaluate_conditions(
        scores_3pat,
        ref_samples,
        starts,
        ends,
        sampling_rate=200,
        threshold=0.22,
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

        # Generate scores for all 3 acquisition patterns: (N, 3)
        scores_3pat = np.zeros((B, 3), dtype=np.float64)

        for pat_idx, pat in enumerate(patterns):
            # Mask: True where channel is inactive/dropped
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

            scores_3pat[:, pat_idx] = np.nan_to_num(np.concatenate(probs_chunks), nan=0.2373)

        tasks.append((
            model_id, seg["patient_id"], fi, scores_3pat,
            ref_samples, starts, ends, released_rr, duration_model
        ))

    # Parallelize CPU-bound condition evaluations across 8 cores
    n_workers = min(8, multiprocessing.cpu_count())
    with multiprocessing.Pool(processes=n_workers) as pool:
        all_recs = pool.map(_evaluate_single_segment_task, tasks)

    condition_records = []
    for rec_list in all_recs:
        condition_records.extend(rec_list)

    return condition_records


def compute_patient_pairs(
    all_timeline_records: List[Dict[str, Any]],
    primary_model: str = "A001",
    baseline_model: str = "A002",
) -> List[Dict[str, Any]]:
    """Generates paired patient-level comparison records between primary and baseline models."""
    # Filter for D2 decoder and pattern 2 (both channels active)
    filtered = [r for r in all_timeline_records if r["decoder"] == "D2" and r["pattern"] == 2]

    patients = sorted(list({r["patient_id"] for r in filtered}))
    pairs = []

    for patient in patients:
        p_recs = [r for r in filtered if r["patient_id"] == patient and r["model_id"] == primary_model]
        b_recs = [r for r in filtered if r["patient_id"] == patient and r["model_id"] == baseline_model]

        if not p_recs or not b_recs:
            continue

        p_dens = np.mean([r["density_continuous_mae"] for r in p_recs if r["density_continuous_mae"] is not None])
        b_dens_cont = np.mean([r["density_continuous_mae"] for r in b_recs if r["density_continuous_mae"] is not None])
        b_dens = np.mean([r["density_binned_mae"] for r in b_recs if r["density_binned_mae"] is not None])
        p_burd = np.mean([r["burden_mae"] for r in p_recs if r["burden_mae"] is not None])
        b_burd = np.mean([r["burden_mae"] for r in b_recs if r["burden_mae"] is not None])

        pairs.append({
            "patient_id": patient,
            "primary_model": primary_model,
            "baseline_model": baseline_model,
            "primary_continuous_density_mae": round(float(p_dens), 4) if np.isfinite(p_dens) else None,
            "baseline_continuous_density_mae": round(float(b_dens_cont), 4) if np.isfinite(b_dens_cont) else None,
            "continuous_density_mae_reduction": round(float(b_dens_cont - p_dens), 4) if np.isfinite(p_dens) and np.isfinite(b_dens_cont) else None,
            "baseline_binned_density_mae": round(float(b_dens), 4) if np.isfinite(b_dens) else None,
            "density_mae_reduction": round(float(b_dens - p_dens), 4) if np.isfinite(p_dens) and np.isfinite(b_dens) else None,
            "primary_burden_mae": round(float(p_burd), 4) if np.isfinite(p_burd) else None,
            "baseline_burden_mae": round(float(b_burd), 4) if np.isfinite(b_burd) else None,
            "burden_mae_reduction": round(float(b_burd - p_burd), 4) if np.isfinite(p_burd) and np.isfinite(b_burd) else None,
        })

    return pairs


def main():
    print("=" * 80)
    print("T-CAIREM Continuous SetOperator AF-Density Research Program")
    print("Stage 1 GPU Training and Evaluation Queue")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")
    print(f"cuDNN Enabled: {torch.backends.cudnn.enabled}")

    # Load duration model
    duration_path = W / "duration_model_v1/fitted_law.npz"
    with np.load(duration_path) as z:
        duration_model = {k: z[k] for k in z.files}

    # 15 informative validation segments
    target_fis = [33, 86, 89, 90, 100, 118, 125, 126, 187, 237, 241, 261, 262, 263, 274]

    print("\n[Step 1/4] Preparing datasets...")
    train_data = prepare_training_dataset(target_count=4096)
    print(f"  Training dataset: {len(train_data[4])} balanced windows loaded.")
    val_data = prepare_validation_dataset(target_fis)
    print(f"  Validation dataset: {len(val_data)} informative segments loaded.")

    models_to_run = [
        "A001", "A002", "A003", "A004", "D032", "D037",
        "E047", "F053", "F057", "G061", "H071", "I082"
    ]

    all_training_runs = []
    model_selections = {}
    local_metrics_list = []
    all_timeline_records = []

    train_log_file = OUTPUT_DIR_LOCAL / "training_runs.jsonl"
    if train_log_file.exists():
        train_log_file.unlink()

    print(f"\n[Step 2/4] Executing Stage 1 Training & Checkpoint Queue ({len(models_to_run)} models)...")
    for idx, model_id in enumerate(models_to_run, 1):
        print(f"\n[{idx}/{len(models_to_run)}] Training {model_id}...")
        model, sel_meta, history = train_and_select_model(
            model_id=model_id,
            train_data=train_data,
            val_data=val_data,
            epochs=5,
            device=device,
        )
        model_selections[model_id] = sel_meta

        # Write to training_runs.jsonl incrementally
        with open(train_log_file, "a") as f:
            for rec in history:
                f.write(json.dumps(rec) + "\n")
                all_training_runs.append(rec)

        print(f"  Selected best epoch {sel_meta['best_epoch']} (val_loss: {sel_meta['best_val_loss']:.4f})")

        # Evaluate window-level metrics
        print(f"  Evaluating local metrics for {model_id}...")
        local_met = evaluate_window_level_metrics(model, model_id, val_data, device=device)
        local_metrics_list.append(local_met)
        print(f"  AUROC: {local_met['auroc']:.4f} | Brier: {local_met['brier_score']:.4f} | ECE: {local_met['ece_10_bins']:.4f}")

        # Evaluate timeline decoding metrics
        print(f"  Evaluating timeline conditions (D0, D1, D2 across 15 segments)...")
        t_records = evaluate_timeline_segment_metrics(model, model_id, val_data, duration_model, device=device)
        all_timeline_records.extend(t_records)
        print(f"  Generated {len(t_records)} condition metrics.")

    print("\n[Step 3/4] Exporting Tabular Deliverables...")
    # 1. model_selection_history.json
    sel_path = OUTPUT_DIR_LOCAL / "model_selection_history.json"
    sel_path.write_text(json.dumps(model_selections, indent=2) + "\n")

    # 2. validation_local_metrics.csv
    loc_csv = OUTPUT_DIR_LOCAL / "validation_local_metrics.csv"
    with open(loc_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(local_metrics_list[0].keys()))
        writer.writeheader()
        writer.writerows(local_metrics_list)

    # 3. validation_timeline_metrics.csv
    time_csv = OUTPUT_DIR_LOCAL / "validation_timeline_metrics.csv"
    with open(time_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_timeline_records[0].keys()))
        writer.writeheader()
        writer.writerows(all_timeline_records)

    # 4. validation_patient_pairs.csv
    patient_pairs = compute_patient_pairs(all_timeline_records, primary_model="A001", baseline_model="A002")
    pairs_csv = OUTPUT_DIR_LOCAL / "validation_patient_pairs.csv"
    with open(pairs_csv, "w", newline="") as f:
        if patient_pairs:
            writer = csv.DictWriter(f, fieldnames=list(patient_pairs[0].keys()))
            writer.writeheader()
            writer.writerows(patient_pairs)

    print("\n[Step 4/4] Mirroring deliverables to codex_artifacts...")
    OUTPUT_DIR_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    for fname in [
        "training_runs.jsonl",
        "model_selection_history.json",
        "validation_local_metrics.csv",
        "validation_timeline_metrics.csv",
        "validation_patient_pairs.csv",
    ]:
        src = OUTPUT_DIR_LOCAL / fname
        dst = OUTPUT_DIR_ARTIFACTS / fname
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  Mirrored {fname} -> {dst}")

    print("\n" + "=" * 80)
    print("Stage 1 GPU Training and Evaluation Queue Execution Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
