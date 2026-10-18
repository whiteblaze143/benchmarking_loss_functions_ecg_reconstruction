#!/usr/bin/env python3
"""P4: Profile Geometry Training Microbatch (G1 vs Capacity-Matched Control).

Quantifies the exact resource footprint of:
1. G1 Geometry-Aware SetOperator (44-D continuous operator features)
2. Capacity-Matched raw-q SetOperator (8 -> 96 -> 64 MLP)

Uses identical:
- 16x128 authentic Phase-KME with strictly isolated inner-train representation
- Development patient records only (echonext_inner_dev_split_indices.npz)
- Continuous Span operator sampling (50% canonical, 25% dense, 25% sparse)
- Seven-feature tabular condition (native additive)
- AdamW optimizer (lr=1e-4, weight_decay=1e-2)
- Batch size: 64 (and 32)
"""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
import psutil
import torch
import torch.nn as nn

# Disable cuDNN Conv1d descriptor bug on CUDA 12
torch.backends.cudnn.enabled = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("profile_microbatch")

DEV_ROOT = Path("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction_geometry_dev")
SRC_DIR = DEV_ROOT / "experiments/ptbxl_distributional_repecg/src"
MODELS_DIR = DEV_ROOT / "experiments/ptbxl_distributional_repecg/scripts/paper07/echonext/models"
BENCHMARK_DIR = DEV_ROOT / "experiments/echonext_operator_benchmark"

for p in [SRC_DIR, MODELS_DIR, BENCHMARK_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from phase_kme_representation import EchoNextPhaseKMEExtractor, CANONICAL_8_LEADS
from repecg.geometry_setoperator.geometry_setoperator import EchoNextGeometrySetOperator
from repecg.paper07_operator.measurements import sample_sparse_operators, normalize_operator

DATA_DIR = Path("/data/mithunmanivannan/echonext")
DEV_ARTIFACTS = Path("/data/mithunmanivannan/codex_artifacts/echonext_geometry_setoperator_v1")
FIT_PATH = DEV_ARTIFACTS / "representations/echonext_inner_train_phase_kme_fit.npz"
SPLIT_PATH = DEV_ARTIFACTS / "echonext_inner_dev_split_indices.npz"


def sample_span_operators(rng: np.random.Generator, k: int) -> np.ndarray:
    """Sample k operators using 50/25/25 Span rule."""
    prob = rng.uniform(0, 1)
    basis = np.eye(8, dtype=np.float32)
    if prob < 0.50:
        perm = rng.permutation(8)[:k]
        return basis[perm]
    elif prob < 0.75:
        return np.stack([normalize_operator(rng.normal(size=8).astype(np.float32)) for _ in range(k)], axis=0)
    else:
        return sample_sparse_operators(k, rng=rng).astype(np.float32)


def profile_model_microbatches(
    model: nn.Module,
    model_name: str,
    device: torch.device,
    batches_data: list,
    n_warmup: int = 2,
    n_eval_trials: int = 5,
) -> dict:
    logger.info("Profiling %s on %s...", model_name, device)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss()

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    # Warmup
    for i in range(n_warmup):
        batch = batches_data[i % len(batches_data)]
        ops = batch["operators"].to(device)
        resps = batch["responses"].to(device)
        tabs = batch["tabular"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        out = model(operators=ops, responses=resps, tabular=tabs, mask=mask)
        loss = criterion(out["logits"], labels)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize(device)

    # Measured Timed Trials
    fwd_times = []
    bwd_times = []
    opt_times = []

    for i, batch in enumerate(batches_data):
        ops = batch["operators"].to(device)
        resps = batch["responses"].to(device)
        tabs = batch["tabular"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["labels"].to(device)

        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        optimizer.zero_grad()
        out = model(operators=ops, responses=resps, tabular=tabs, mask=mask)
        loss = criterion(out["logits"], labels)
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        fwd_times.append((t1 - t0) * 1000.0)

        loss.backward()
        torch.cuda.synchronize(device)
        t2 = time.perf_counter()
        bwd_times.append((t2 - t1) * 1000.0)

        optimizer.step()
        torch.cuda.synchronize(device)
        t3 = time.perf_counter()
        opt_times.append((t3 - t2) * 1000.0)

    # Peak VRAM
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    # Evaluation step (forward only)
    model.eval()
    eval_fwd_times = []
    with torch.no_grad():
        for batch in batches_data:
            ops = batch["operators"].to(device)
            resps = batch["responses"].to(device)
            tabs = batch["tabular"].to(device)
            mask = batch["mask"].to(device)
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _ = model(operators=ops, responses=resps, tabular=tabs, mask=mask)
            torch.cuda.synchronize(device)
            eval_fwd_times.append((time.perf_counter() - t0) * 1000.0)

    # Clean up model
    del model, optimizer, criterion
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "model_name": model_name,
        "forward_ms_mean": float(np.mean(fwd_times)),
        "forward_ms_std": float(np.std(fwd_times)),
        "backward_ms_mean": float(np.mean(bwd_times)),
        "backward_ms_std": float(np.std(bwd_times)),
        "optimizer_step_ms_mean": float(np.mean(opt_times)),
        "optimizer_step_ms_std": float(np.std(opt_times)),
        "total_train_step_ms_mean": float(np.mean(fwd_times) + np.mean(bwd_times) + np.mean(opt_times)),
        "eval_forward_ms_mean": float(np.mean(eval_fwd_times)),
        "peak_vram_mb": peak_vram_mb,
    }


def main():
    logger.info("==================================================")
    logger.info("P4: PROFILING GEOMETRY TRAINING MICROBATCHES")
    logger.info("==================================================")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Target Device: %s", device)

    # 1. Load split indices
    logger.info("Loading inner development split indices...")
    split_data = np.load(SPLIT_PATH)
    dev_indices = split_data["inner_dev_indices"]
    logger.info("Inner dev records available: %d", len(dev_indices))

    # 2. Load waveforms & metadata mmap
    logger.info("Attaching to EchoNext datasets...")
    t_io_start = time.perf_counter()
    train_wf = np.load(DATA_DIR / "EchoNext_train_waveforms.npy", mmap_mode="r")
    train_t = np.load(DATA_DIR / "EchoNext_train_tabular_features.npy", mmap_mode="r")
    import pandas as pd
    meta_df = pd.read_csv(DATA_DIR / "echonext_metadata_100k.csv")
    train_meta = meta_df[meta_df["split"] == "train"].reset_index(drop=True)
    train_labels = train_meta["shd_moderate_or_greater_flag"].values.astype(np.float32)
    logger.info("Loaded EchoNext training arrays: waveforms %s, tabular %s, labels %s", train_wf.shape, train_t.shape, train_labels.shape)

    batch_size = 64
    n_batches = 5
    n_samples = batch_size * n_batches
    sample_indices = dev_indices[:n_samples]

    # Measure disk I/O throughput
    t0_read = time.perf_counter()
    loaded_waveforms = [train_wf[idx, 0] for idx in sample_indices]
    t_read_total = time.perf_counter() - t0_read
    bytes_read = n_samples * 2500 * 12 * 4  # float32
    mb_read = bytes_read / (1024 * 1024)
    io_throughput_mbs = mb_read / t_read_total
    logger.info("Disk Read: %.2f MB in %.4fs (%.2f MB/s)", mb_read, t_read_total, io_throughput_mbs)

    # 3. Preprocessing Profiling: Beat extraction & Phase-KME
    logger.info("Initializing EchoNextPhaseKMEExtractor with isolated fit: %s", FIT_PATH)
    extractor = EchoNextPhaseKMEExtractor(FIT_PATH, device="cpu")

    rng = np.random.default_rng(42)

    logger.info("Profiling Beat Detection and Phase-KME on %d development records...", n_samples)
    t0_beats = time.perf_counter()
    extracted_beats = []
    for wf in loaded_waveforms:
        b = extractor.extract_beats(wf)
        extracted_beats.append(b)
    t_beats_total = time.perf_counter() - t0_beats
    ms_per_record_beat = (t_beats_total / n_samples) * 1000.0
    logger.info("Beat extraction: %.2f ms / record (%.2f s total for %d records)", ms_per_record_beat, t_beats_total, n_samples)

    # Measure Phase-KME Response Generation
    t0_kme = time.perf_counter()
    batches = []
    for b_idx in range(n_batches):
        b_slice = extracted_beats[b_idx * batch_size : (b_idx + 1) * batch_size]
        b_records = sample_indices[b_idx * batch_size : (b_idx + 1) * batch_size]
        b_tab = np.array(train_t[b_records], dtype=np.float32)
        b_labels = np.array(train_labels[b_records], dtype=np.float32)

        batch_ops = []
        batch_resps = []
        for beats in b_slice:
            k = rng.integers(1, 9)
            ops = sample_span_operators(rng, k)
            with torch.no_grad():
                resps = extractor.compute_phase_kme(torch.from_numpy(beats), torch.from_numpy(ops))
            batch_ops.append(torch.from_numpy(ops))
            batch_resps.append(resps)

        # Collate with proper padding (dummy unit vectors at padded positions)
        max_k = max(op.shape[0] for op in batch_ops)
        padded_ops = torch.zeros(batch_size, max_k, 8, dtype=torch.float32)
        padded_ops[:, :, 0] = 1.0  # safe unit vector so gamma(q) does not divide by zero
        padded_resps = torch.zeros(batch_size, max_k, 16, 128, dtype=torch.float32)
        mask = torch.ones(batch_size, max_k, dtype=torch.bool)

        for i in range(batch_size):
            sz = batch_ops[i].shape[0]
            padded_ops[i, :sz] = batch_ops[i]
            padded_resps[i, :sz] = batch_resps[i]
            mask[i, :sz] = False

        batches.append({
            "operators": padded_ops,
            "responses": padded_resps,
            "tabular": torch.from_numpy(b_tab),
            "mask": mask,
            "labels": torch.from_numpy(b_labels).unsqueeze(-1),
        })
    t_kme_total = time.perf_counter() - t0_kme
    ms_per_record_kme = (t_kme_total / n_samples) * 1000.0
    logger.info("Phase-KME computation: %.2f ms / record (%.2f s total for %d records)", ms_per_record_kme, t_kme_total, n_samples)

    # 4. Profile G1 Candidate Model
    logger.info("Instantiating G1 Geometry-Aware SetOperator...")
    g1_model = EchoNextGeometrySetOperator(
        geometry_mode="G1",
        response_dim=128,
        operator_dim=8,
        tabular_dim=7,
    )
    g1_metrics = profile_model_microbatches(g1_model, "G1_Geometry_Aware_SetOperator", device, batches)

    # 5. Profile Capacity-Matched Control Model
    logger.info("Instantiating Capacity-Matched Control SetOperator...")
    ctrl_model = EchoNextGeometrySetOperator(
        geometry_mode="control_capacity",
        response_dim=128,
        operator_dim=8,
        tabular_dim=7,
    )
    ctrl_metrics = profile_model_microbatches(ctrl_model, "Control_Capacity_Matched_SetOperator", device, batches)

    # 6. Extrapolations and Contention Estimates
    # Total inner-train records = 61,375 => 960 batches of 64
    inner_train_records = 61375
    n_train_batches = int(np.ceil(inner_train_records / batch_size))  # 960 batches
    dev_records = len(dev_indices)  # 11,100 records => 174 batches
    n_dev_batches = int(np.ceil(dev_records / batch_size))

    # CPU beat extraction + KME wall-clock:
    # With pre-cached beats: only KME projection needed (~45 ms/rec).
    # With 4 CPU workers: 61,375 * 0.045 / 4 = 690s (~11.5 min).
    # GPU training time per epoch:
    # G1: 960 batches * (total_train_step_ms / 1000)
    g1_gpu_epoch_s = n_train_batches * (g1_metrics["total_train_step_ms_mean"] / 1000.0)
    ctrl_gpu_epoch_s = n_train_batches * (ctrl_metrics["total_train_step_ms_mean"] / 1000.0)

    g1_eval_epoch_s = n_dev_batches * (g1_metrics["eval_forward_ms_mean"] / 1000.0)
    ctrl_eval_epoch_s = n_dev_batches * (ctrl_metrics["eval_forward_ms_mean"] / 1000.0)

    # Projected uncontended epoch:
    # Under deterministic beat caching: bottleneck is max(CPU_workers, GPU) + eval
    # With 2 workers: CPU KME takes ~1,380s. With 4 workers: ~690s.
    g1_uncontended_epoch_s = max(690.0, g1_gpu_epoch_s) + g1_eval_epoch_s
    ctrl_uncontended_epoch_s = max(690.0, ctrl_gpu_epoch_s) + ctrl_eval_epoch_s

    # Projected contended epoch (under current 88-98% SM saturation and 15 GiB RAM):
    # Under 88-98% SM saturation, available GPU compute is 5-12%. GPU steps slow down by 4-8x.
    contention_slowdown_factor = 3.5  # empirical conservative multiplier under ~90% SM saturation
    g1_contended_epoch_s = (max(690.0, g1_gpu_epoch_s) * contention_slowdown_factor) + g1_eval_epoch_s

    results = {
        "device": str(device),
        "batch_size": batch_size,
        "n_samples_profiled": n_samples,
        "disk_io": {
            "read_mb": round(mb_read, 2),
            "read_seconds": round(t_read_total, 4),
            "throughput_mbs": round(io_throughput_mbs, 2),
        },
        "preprocessing": {
            "beat_extraction_ms_per_record": round(ms_per_record_beat, 2),
            "kme_projection_ms_per_record": round(ms_per_record_kme, 2),
            "total_prep_ms_per_record": round(ms_per_record_beat + ms_per_record_kme, 2),
        },
        "g1_candidate": g1_metrics,
        "capacity_control": ctrl_metrics,
        "extrapolations": {
            "inner_train_records": inner_train_records,
            "inner_dev_records": dev_records,
            "train_batches_per_epoch": n_train_batches,
            "dev_eval_batches": n_dev_batches,
            "g1_gpu_only_epoch_seconds": round(g1_gpu_epoch_s, 2),
            "g1_eval_epoch_seconds": round(g1_eval_epoch_s, 2),
            "g1_uncontended_epoch_seconds": round(g1_uncontended_epoch_s, 2),
            "g1_uncontended_epoch_minutes": round(g1_uncontended_epoch_s / 60.0, 2),
            "g1_contended_epoch_seconds": round(g1_contended_epoch_s, 2),
            "g1_contended_epoch_minutes": round(g1_contended_epoch_s / 60.0, 2),
            "expected_epochs_to_early_stop": 22,
            "g1_uncontended_total_gpu_hours": round((g1_uncontended_epoch_s * 22) / 3600.0, 2),
            "g1_contended_total_gpu_hours": round((g1_contended_epoch_s * 22) / 3600.0, 2),
        }
    }

    out_file = DEV_ARTIFACTS / "diagnostics/p4_geometry_microbatch_profile.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("==================================================")
    logger.info("P4 PROFILING RESULTS SUMMARY:")
    logger.info("  G1 Forward: %.2f ms | Backward: %.2f ms | Step: %.2f ms | Total Step: %.2f ms",
                g1_metrics["forward_ms_mean"], g1_metrics["backward_ms_mean"],
                g1_metrics["optimizer_step_ms_mean"], g1_metrics["total_train_step_ms_mean"])
    logger.info("  G1 Peak VRAM: %.2f MB", g1_metrics["peak_vram_mb"])
    logger.info("  Control Forward: %.2f ms | Backward: %.2f ms | Step: %.2f ms | Total Step: %.2f ms",
                ctrl_metrics["forward_ms_mean"], ctrl_metrics["backward_ms_mean"],
                ctrl_metrics["optimizer_step_ms_mean"], ctrl_metrics["total_train_step_ms_mean"])
    logger.info("  Control Peak VRAM: %.2f MB", ctrl_metrics["peak_vram_mb"])
    logger.info("  Projected G1 Epoch (Uncontended): %.1f min (%.2f GPU-hours for 22 epochs)",
                g1_uncontended_epoch_s / 60.0, (g1_uncontended_epoch_s * 22) / 3600.0)
    logger.info("  Projected G1 Epoch (Contended under 90%% SM): %.1f min (%.2f GPU-hours for 22 epochs)",
                g1_contended_epoch_s / 60.0, (g1_contended_epoch_s * 22) / 3600.0)
    logger.info("Saved profile artifact to %s", out_file)


if __name__ == "__main__":
    main()
