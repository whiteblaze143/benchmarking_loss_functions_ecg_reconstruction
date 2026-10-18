#!/usr/bin/env python3
"""Detailed Preprocessing and Training Throughput Profile for EchoNext Phase-KME.

Measures the exact wall-clock time breakdown per batch (batch_size=64) and extrapolates
to a full training epoch (N=72,475 records, 1,133 batches):
1. Raw waveform beat detection and phase normalization (detect_rpeaks + phase_normalize)
2. Phase-KME response computation given pre-extracted beats
3. GPU forward pass, loss calculation, backward pass, and optimizer step
4. Validation evaluation throughput (N=4,626 records)
5. Comparison: On-the-fly beat extraction vs Pre-cached beats throughput
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn

# Disable cuDNN Conv1d descriptor bug on CUDA 12
torch.backends.cudnn.enabled = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("profile_phase_kme")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = REPO_ROOT / "experiments/ptbxl_distributional_repecg/src"
MODELS_DIR = REPO_ROOT / "experiments/ptbxl_distributional_repecg/scripts/paper07/echonext/models"
BENCHMARK_DIR = REPO_ROOT / "experiments/echonext_operator_benchmark"

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(MODELS_DIR))
sys.path.insert(0, str(BENCHMARK_DIR))

from phase_kme_representation import EchoNextPhaseKMEExtractor, DEFAULT_FIT_PATH, CANONICAL_8_LEADS
from setoperator_echonext import build_setoperator_echonext
from repecg.paper07_operator.measurements import normalize_operator, sample_sparse_operators

DATA_DIR = Path("/data/mithunmanivannan/echonext")
OUT_DIR = Path("/data/mithunmanivannan/codex_artifacts/echonext_three_way_v1/diagnostics")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    logger.info("==================================================")
    logger.info("PROFILING ECHONEXT PHASE-KME TRAINING THROUGHPUT")
    logger.info("==================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # 1. Load sample waveforms
    logger.info("Loading sample waveforms from training split...")
    train_wf = np.load(DATA_DIR / "EchoNext_train_waveforms.npy", mmap_mode="r")
    batch_size = 64
    n_sample_batches = 5
    n_total_samples = batch_size * n_sample_batches

    sample_records = [train_wf[i, 0][:, CANONICAL_8_LEADS].astype(np.float32) for i in range(n_total_samples)]
    sample_tabs = torch.randn(batch_size, 7, device=device)
    sample_labels = torch.randint(0, 2, (batch_size, 1), device=device, dtype=torch.float32)

    extractor = EchoNextPhaseKMEExtractor(DEFAULT_FIT_PATH, device="cpu")
    model = build_setoperator_echonext("SO-ECG+Tab-Native").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    rng = np.random.default_rng(42)

    # -------------------------------------------------------------------------
    # Component 1: CPU Beat Detection + Phase Normalization
    # -------------------------------------------------------------------------
    logger.info("Profiling Component 1: Beat Detection & RR Normalization on CPU...")
    t0 = time.time()
    all_beats = []
    for rec in sample_records:
        beats = extractor.extract_beats(rec)
        all_beats.append(beats)
    t_beat_total = time.time() - t0
    t_beat_per_rec = (t_beat_total / n_total_samples) * 1000.0
    t_beat_per_batch = t_beat_total / n_sample_batches
    logger.info("  Beat detection: %.2f ms / record (%.2f s / batch of %d)", t_beat_per_rec, t_beat_per_batch, batch_size)

    # -------------------------------------------------------------------------
    # Component 2: Phase-KME Computation (given beats)
    # -------------------------------------------------------------------------
    logger.info("Profiling Component 2: Phase-KME Projection & Kernel Features...")
    t0 = time.time()
    for i in range(n_sample_batches):
        batch_slice = all_beats[i * batch_size : (i + 1) * batch_size]
        for b in batch_slice:
            # Sample k in [1, 8] operators
            k = rng.integers(1, 9)
            ops = sample_sparse_operators(k, rng).astype(np.float32)
            with torch.no_grad():
                resps = extractor.compute_phase_kme(torch.from_numpy(b), torch.from_numpy(ops))
    t_kme_total = time.time() - t0
    t_kme_per_rec = (t_kme_total / n_total_samples) * 1000.0
    t_kme_per_batch = t_kme_total / n_sample_batches
    logger.info("  Phase-KME computation: %.2f ms / record (%.2f s / batch of %d)", t_kme_per_rec, t_kme_per_batch, batch_size)

    # -------------------------------------------------------------------------
    # Component 3: GPU Forward + Backward + Optimizer Step
    # -------------------------------------------------------------------------
    logger.info("Profiling Component 3: GPU Forward & Backward passes on A100...")
    # Prepare dummy batch on GPU
    dummy_ops = torch.randn(batch_size, 8, 8, device=device)
    dummy_resps = torch.randn(batch_size, 8, 16, 128, device=device)
    dummy_mask = torch.zeros(batch_size, 8, dtype=torch.bool, device=device)

    # Warmup
    for _ in range(5):
        optimizer.zero_grad()
        out = model(operators=dummy_ops, responses=dummy_resps, tabular=sample_tabs, mask=dummy_mask)
        loss = criterion(out["logits"], sample_labels)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    t0 = time.time()
    n_gpu_trials = 50
    for _ in range(n_gpu_trials):
        optimizer.zero_grad()
        out = model(operators=dummy_ops, responses=dummy_resps, tabular=sample_tabs, mask=dummy_mask)
        loss = criterion(out["logits"], sample_labels)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    t_gpu_total = time.time() - t0
    t_gpu_per_batch = (t_gpu_total / n_gpu_trials)
    t_gpu_per_rec = (t_gpu_per_batch / batch_size) * 1000.0
    logger.info("  GPU Forward+Backward step: %.2f ms / batch (%.2f ms / record)", t_gpu_per_batch * 1000.0, t_gpu_per_rec)

    # -------------------------------------------------------------------------
    # Component 4: Validation Evaluation Throughput (N=4,626)
    # -------------------------------------------------------------------------
    logger.info("Profiling Component 4: GPU Evaluation Forward pass...")
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(n_gpu_trials):
            out = model(operators=dummy_ops, responses=dummy_resps, tabular=sample_tabs, mask=dummy_mask)
    torch.cuda.synchronize()
    t_eval_per_batch = (time.time() - t0) / n_gpu_trials
    val_batches = int(np.ceil(4626 / batch_size))
    t_val_total = val_batches * t_eval_per_batch
    logger.info("  GPU Eval step: %.2f ms / batch (Extrapolated full Val set: %.2f s)", t_eval_per_batch * 1000.0, t_val_total)

    # -------------------------------------------------------------------------
    # Extrapolations across Full Epoch (72,475 records = 1,133 batches of 64)
    # -------------------------------------------------------------------------
    total_train_records = 72475
    n_train_batches = int(np.ceil(total_train_records / batch_size))  # 1133

    # Case A: On-the-fly un-cached beat detection + KME + GPU (with 4 DataLoader workers)
    # CPU time dominates: (t_beat_per_batch + t_kme_per_batch) / 4 workers
    epoch_sec_uncached = (n_train_batches * (t_beat_per_batch + t_kme_per_batch) / 4.0) + (n_train_batches * t_gpu_per_batch) + t_val_total
    epoch_min_uncached = epoch_sec_uncached / 60.0

    # Case B: Pre-cached beats on disk (only KME projection + GPU forward/backward)
    epoch_sec_cached_beats = (n_train_batches * t_kme_per_batch / 4.0) + (n_train_batches * t_gpu_per_batch) + t_val_total
    epoch_min_cached_beats = epoch_sec_cached_beats / 60.0

    # Case C: GPU-accelerated batch Phase-KME or cached canonical KME
    epoch_sec_fast = (n_train_batches * t_gpu_per_batch * 1.5) + t_val_total
    epoch_min_fast = epoch_sec_fast / 60.0

    logger.info("==================================================")
    logger.info("FULL EPOCH RUNTIME FORECAST (N=72,475 records):")
    logger.info("  Case A (Uncached on-the-fly beat detection): %.1f min / epoch (%.1f hours for 25 epochs)", epoch_min_uncached, epoch_min_uncached * 25 / 60)
    logger.info("  Case B (Pre-cached beats, dynamic KME):      %.1f min / epoch (%.1f hours for 25 epochs)", epoch_min_cached_beats, epoch_min_cached_beats * 25 / 60)
    logger.info("  Case C (Optimized batched Phase-KME pipeline): %.1f min / epoch (%.1f hours for 25 epochs)", epoch_min_fast, epoch_min_fast * 25 / 60)
    logger.info("==================================================")

    profile_results = {
        "hardware": "NVIDIA A100-PCIE-40GB",
        "batch_size": batch_size,
        "n_train_records": total_train_records,
        "n_train_batches": n_train_batches,
        "t_beat_detection_ms_per_record": t_beat_per_rec,
        "t_kme_computation_ms_per_record": t_kme_per_rec,
        "t_gpu_fwd_bwd_ms_per_batch": t_gpu_per_batch * 1000.0,
        "t_gpu_eval_ms_per_batch": t_eval_per_batch * 1000.0,
        "t_val_total_sec": t_val_total,
        "epoch_forecast_uncached_min": epoch_min_uncached,
        "epoch_forecast_cached_beats_min": epoch_min_cached_beats,
        "epoch_forecast_optimized_min": epoch_min_fast,
    }

    out_file = OUT_DIR / "echonext_phase_kme_throughput_profile.json"
    with open(out_file, "w") as f:
        json.dump(profile_results, f, indent=2)
    logger.info("Saved complete throughput profile to %s", out_file)


if __name__ == "__main__":
    main()
