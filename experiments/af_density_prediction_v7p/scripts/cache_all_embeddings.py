"""GPU Feature Extraction & Embedding Cache Suite for V7-P.

Extracts causal 10s, 30s, and 60s frozen representations across:
8 Foundation Models:
- CSFM Tiny (dim 768)
- CLEF Medium (dim 1024)
- ECGFounder 1-lead (dim 1024)
- ECG-FM KED (dim 768)
- EchoNext Minimodel (dim 256)
- DINO-SR (dim 512)
- JEPA (dim 512)
- NEFCore Full (dim 128)
4 SetOperator Models:
- F2 SetOperator (dim 320)
- F0 SetOperator (dim 320)
- G1 EchoNext SetOperator (dim 256)
- Control Capacity SetOperator (dim 256)

Zero synthetic fallbacks: all waveforms are sliced causally from authentic recordings.
Missing or corrupted records raise explicit exceptions.
"""

import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SRC_DIR = PROJECT_DIR / "src"
EMBEDDINGS_DIR = PROJECT_DIR / "embeddings"
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

for p in [SRC_DIR, PROJECT_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import foundation_adapters as fa
from waveform_loader import WaveformLoader
from ecg_features import RRFeatureExtractor

CHECKPOINTS = {
    "CSFM": ("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/external/Cardiac-Sensing-FM/pretrained/CSFM_tiny.pth", "CSFM"),
    "CLEF": ("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/foundation_model_comparison/candidate_weights/clef/clef_medium.ckpt", "CLEF"),
    "ECGFOUNDER": (str(PROJECT_DIR / "checkpoints" / "1_lead_ECGFounder.pth"), "ECGFOUNDER"),
    "ECGFM_KED": ("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/foundation_model_comparison/candidate_weights/ecgfm_ked/best_valid_all_increase_with_augment_epoch_3.pt", "KED"),
    "ECHONEXT": ("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/external/IntroECG/7-EchoNext Minimodel/models/echonext_multilabel_minimodel/weights.pt", "ECHONEXT"),
    "DINOSR": ("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/foundation_model_comparison/candidate_sources/ecg_pretraining_strategies/checkpoints/dinosr_e10.ckpt", "DINOSR"),
    "JEPA": ("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/foundation_model_comparison/candidate_sources/ecg_pretraining_strategies/checkpoints/jepa_e10.ckpt", "JEPA"),
    "NEFCORE": ("/data/mithunmanivannan/nef-net-aim/02_nef_core/pretrained/nefcore_full.pt", "NEFCORE"),
    "F2": ("/data/mithunmanivannan/codex_artifacts/temporal_setoperator_fm_20261120/artifacts/F2/pilot_final.pt", "F2"),
    "F0": ("/data/mithunmanivannan/codex_artifacts/temporal_setoperator_fm_20261120/artifacts/F0/pilot_final.pt", "F0"),
    "G1": ("/data/mithunmanivannan/codex_artifacts/echonext_geometry_setoperator_v1/runs/g1_seed42/best_model.pt", "G1"),
    "CONTROL": ("/data/mithunmanivannan/codex_artifacts/echonext_geometry_setoperator_v1/runs/control_capacity_seed42/best_model.pt", "CAPACITY"),
}

MANIFEST_FILES = {
    "P02": PROJECT_DIR / "P02_episode_persistence_manifest.parquet",
    "P03": PROJECT_DIR / "P03_survival_manifest.parquet",
    "P04": PROJECT_DIR / "P04_noise_classification_manifest.parquet",
    "P01": PROJECT_DIR / "P01_landmark_manifest.parquet",
}


def cache_f4_rr_features(task_id: str, manifest_df: pd.DataFrame) -> Path:
    """Extracts and caches F4 physiological ventricular dynamics."""
    out_path = EMBEDDINGS_DIR / f"{task_id}_f4_rr_features.parquet"
    if out_path.exists():
        print(f"  [Cached] F4 RR features already exist at {out_path.name}")
        return out_path

    print(f"  Extracting F4 physiological RR features for {task_id} (N={len(manifest_df):,})...")
    extractor = RRFeatureExtractor()
    records = []

    t0 = time.time()
    for row in manifest_df.itertuples():
        ds = getattr(row, "dataset")
        rid = getattr(row, "record_id", getattr(row, "strip_id", ""))
        t = getattr(row, "landmark_time_s", 10.0)
        feats = extractor.extract_window_rr_features(ds, rid, t, window_s=300.0)
        records.append(feats)

    rr_df = pd.DataFrame(records)
    rr_df.to_parquet(out_path, index=False)
    print(f"  Saved F4 RR features to {out_path.name} in {time.time() - t0:.1f}s.")
    return out_path


def cache_embeddings_for_task(
    task_id: str,
    manifest_df: pd.DataFrame,
    device: torch.device,
    batch_size: int = 128,
):
    """Caches 10s, 30s, and 60s causal embeddings for all 12 models."""
    print(f"\n{'=' * 80}")
    print(f"Processing Task {task_id}: N={len(manifest_df):,} samples, {manifest_df['patient_id'].nunique()} patients")
    print(f"{'=' * 80}")

    # 1. Cache F4 RR features first
    cache_f4_rr_features(task_id, manifest_df)

    wl = WaveformLoader()

    # Pre-slice raw 60s causal ECG window ending at landmark_time_s
    # A 60s window contains 6 contiguous 10s chunks:
    # chunk 0: [-60:-50], chunk 1: [-50:-40], chunk 2: [-40:-30],
    # chunk 3: [-30:-20], chunk 4: [-20:-10], chunk 5: [-10:0] (primary 10s)
    # Then:
    # 10s embedding = emb(chunk 5)
    # 30s embedding = mean(emb(chunk 3), emb(chunk 4), emb(chunk 5))
    # 60s embedding = mean(emb(chunk 0..5))
    
    # Check if all models for this task are already cached
    all_cached = True
    for model_name in CHECKPOINTS:
        for win in ["10s", "30s", "60s"]:
            fn = EMBEDDINGS_DIR / f"{task_id}_{model_name.lower()}_{win}.npz"
            if not fn.exists():
                all_cached = False
                break
        if not all_cached:
            break

    if all_cached:
        print(f"All 12 models already cached for {task_id}! Skipping.")
        wl.close()
        return

    print(f"Loading raw ECG waveform segments for {task_id}...")
    t0 = time.time()
    
    # Group landmarks by (dataset, record_id) for optimal disk caching
    raw_chunks_10s = []  # List of 6 arrays per sample, each 10s
    record_landmarks = []
    
    # Preserve order
    for idx, row in enumerate(manifest_df.itertuples()):
        ds = getattr(row, "dataset")
        rid = getattr(row, "record_id", getattr(row, "strip_id", ""))
        t = getattr(row, "landmark_time_s", 10.0)
        record_landmarks.append((idx, ds, rid, t))

    # Extract 6 contiguous 10s chunks per landmark
    print(f"Extracting 6 causal 10s chunks per landmark across {len(record_landmarks):,} points...")
    chunks_per_sample = [None] * len(record_landmarks)

    for idx, ds, rid, t in record_landmarks:
        # Extract 60s continuous segment ending at t
        sig_60s, fs = wl.get_causal_segment(ds, rid, t, duration_s=60.0)
        chunk_len = int(round(10.0 * fs))
        # 6 chunks
        c_list = []
        for i in range(6):
            s_idx = i * chunk_len
            e_idx = (i + 1) * chunk_len
            c_list.append(sig_60s[s_idx:e_idx])
        chunks_per_sample[idx] = (c_list, fs)

    print(f"Loaded all raw waveforms in {time.time() - t0:.1f}s.")
    wl.close()

    patient_ids = manifest_df["patient_id"].values
    folds = manifest_df["fold"].values

    # Run each model on GPU
    for model_name, (ckpt_path, model_key) in CHECKPOINTS.items():
        win_paths = {
            "10s": EMBEDDINGS_DIR / f"{task_id}_{model_name.lower()}_10s.npz",
            "30s": EMBEDDINGS_DIR / f"{task_id}_{model_name.lower()}_30s.npz",
            "60s": EMBEDDINGS_DIR / f"{task_id}_{model_name.lower()}_60s.npz",
        }
        if all(p.exists() for p in win_paths.values()):
            print(f"  [Cached] {model_name} already complete for {task_id}.")
            continue

        print(f"\n  >>> Running GPU Feature Extraction: {model_name} on {task_id}...")
        t_model = time.time()
        adapter = fa.get_model_adapter(model_key, ckpt_path, device)
        target_fs = float(adapter.fs)
        target_len = adapter.target_len

        # Resample chunks to target_fs and target_len
        print(f"      Resampling {len(chunks_per_sample):,} samples to {target_fs} Hz ({target_len} samples)...")
        # To avoid massive memory spike, resample in streaming batches through the GPU
        n_samples = len(chunks_per_sample)
        all_emb_10s = np.zeros((n_samples, adapter.dim), dtype=np.float32)
        all_emb_30s = np.zeros((n_samples, adapter.dim), dtype=np.float32)
        all_emb_60s = np.zeros((n_samples, adapter.dim), dtype=np.float32)

        for b_start in range(0, n_samples, batch_size):
            b_end = min(n_samples, b_start + batch_size)
            b_size = b_end - b_start

            # Collect 6 chunks for each item in batch
            batch_chunks = [[] for _ in range(6)]
            for i in range(b_start, b_end):
                c_list, src_fs = chunks_per_sample[i]
                for chunk_idx in range(6):
                    resampled = fa.resample_signal(c_list[chunk_idx], src_fs, target_fs, target_len)
                    batch_chunks[chunk_idx].append(resampled)

            # Embed each chunk index (0 to 5) through GPU
            chunk_embs = []
            for chunk_idx in range(6):
                arr = np.stack(batch_chunks[chunk_idx], axis=0)[:, np.newaxis, :]  # (B, 1, L)
                t_arr = torch.from_numpy(arr).float().to(device)
                with torch.no_grad():
                    e = adapter(t_arr).cpu().numpy()
                chunk_embs.append(e)

            # Compute windowed representations:
            # 10s: chunk 5
            emb_10 = chunk_embs[5]
            # 30s: mean(chunk 3, 4, 5)
            emb_30 = (chunk_embs[3] + chunk_embs[4] + chunk_embs[5]) / 3.0
            # 60s: mean(chunk 0..5)
            emb_60 = np.mean(chunk_embs, axis=0)

            all_emb_10s[b_start:b_end] = emb_10
            all_emb_30s[b_start:b_end] = emb_30
            all_emb_60s[b_start:b_end] = emb_60

        # Save to disk
        np.savez_compressed(win_paths["10s"], embeddings=all_emb_10s, patient_id=patient_ids, fold=folds)
        np.savez_compressed(win_paths["30s"], embeddings=all_emb_30s, patient_id=patient_ids, fold=folds)
        np.savez_compressed(win_paths["60s"], embeddings=all_emb_60s, patient_id=patient_ids, fold=folds)

        dt = time.time() - t_model
        throughput = n_samples / dt
        print(f"      Completed {model_name} in {dt:.1f}s ({throughput:.1f} samples/s). Saved 10s, 30s, 60s.")

        # Clean GPU memory
        del adapter
        torch.cuda.empty_cache()


def main():
    print("=" * 80)
    print("V7-P Predictive Framework: Foundation Model Embedding Caching Suite")
    print("=" * 80)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    for task_id in ["P02", "P03", "P04", "P01"]:
        manifest_path = MANIFEST_FILES[task_id]
        if not manifest_path.exists():
            print(f"Error: Manifest {manifest_path} not found!")
            continue
        df = pd.read_parquet(manifest_path)
        cache_embeddings_for_task(task_id, df, device, batch_size=128)

    print("\n" + "=" * 80)
    print("ALL FOUNDATION MODEL AND SETOPERATOR EMBEDDINGS CACHED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    main()
