#!/usr/bin/env python3
"""Extract frozen GraphECG latent embeddings for the PTB-XL repeated patient cohort."""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Batch
import wfdb

# CRITICAL for A100 / PyTorch 2.6
torch.backends.cudnn.enabled = False

REPO_DIR = Path(__file__).resolve().parents[4]
EXPERIMENT_DIR = REPO_DIR / "experiments/ptbxl_distributional_repecg"
sys.path.insert(0, str(EXPERIMENT_DIR / "src"))

from graphECG_author_code.graph import ECGGraphBuilder
from graphECG_author_code.model import GraphECG


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("extract_graphecg")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_file, mode="w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract GraphECG Latents for PTB-XL Repeated Cohort")
    parser.add_argument("--ptbxl-data", type=Path, default=Path("data/ptbxl"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=EXPERIMENT_DIR / "outputs/graphecg/graphecg_ptbxl_best.pt",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/prism_cache/graphecg_multi_z.npz"),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(args.output_path.parent / "extract_graphecg.log")

    logger.info("=" * 60)
    logger.info("Extracting GraphECG Latents for Repeated Cohort")
    logger.info(f"PTB-XL Root:  {args.ptbxl_data}")
    logger.info(f"Checkpoint:   {args.checkpoint}")
    logger.info(f"Output:       {args.output_path}")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # 1. Load Repeated Cohort Metadata
    df_meta = pd.read_csv(args.ptbxl_data / "ptbxl_database.csv", index_col="ecg_id")
    df_meta["recording_date"] = pd.to_datetime(df_meta["recording_date"])
    pat_counts = df_meta["patient_id"].value_counts()
    multi_pats = pat_counts[pat_counts >= 2].index.values
    df_multi = df_meta[df_meta["patient_id"].isin(multi_pats)].sort_values(["patient_id", "recording_date"]).copy()
    logger.info(f"Repeated Cohort: {len(multi_pats):,} patients, {len(df_multi):,} ECG records.")

    # 2. Instantiate and Load GraphECG Model
    logger.info(f"Instantiating GraphECG model and loading weights from {args.checkpoint}...")
    model = GraphECG(
        node_dim=128,
        edge_dim=192,
        hidden_dim=192,
        num_layers=3,
        tabular_dim=0,
        num_classes=5,
        dropout=0.0,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)
    model.eval()
    logger.info("GraphECG model loaded successfully.")

    # 3. Batch Graph Extraction
    builder = ECGGraphBuilder()
    records = df_multi.reset_index().to_dict("records")
    embeddings = []
    ecg_ids = []

    t0 = time.time()
    logger.info(f"Starting extraction for {len(records)} records in batches of {args.batch_size}...")

    with torch.no_grad():
        for i in range(0, len(records), args.batch_size):
            batch_slice = records[i : i + args.batch_size]
            graphs = []
            batch_ids = []

            for r in batch_slice:
                ecg_id = int(r["ecg_id"])
                fn = args.ptbxl_data / str(r["filename_lr"])
                sig, _ = wfdb.rdsamp(str(fn))  # (1000, 12)
                sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                mean = sig.mean(axis=0, keepdims=True)
                std = sig.std(axis=0, keepdims=True) + 1e-6
                sig = (sig - mean) / std

                # build_from_array expects (12, T)
                g = builder.build_from_array(sig.T, bidirectional=True)
                graphs.append(g)
                batch_ids.append(ecg_id)

            batch_pyg = Batch.from_data_list(graphs).to(device)
            out = model(batch_pyg)
            emb = out["embedding"].cpu().numpy().astype(np.float32)
            embeddings.append(emb)
            ecg_ids.extend(batch_ids)

            if (i // args.batch_size) % 5 == 0 or (i + args.batch_size) >= len(records):
                logger.info(f"Processed {min(i + args.batch_size, len(records))}/{len(records)} records ({time.time() - t0:.1f}s)...")

    z_all = np.concatenate(embeddings, axis=0)
    ecg_ids_arr = np.array(ecg_ids, dtype=np.int64)

    logger.info(f"GraphECG extraction complete! Shape: {z_all.shape}, Time: {time.time() - t0:.2f}s")

    np.savez_compressed(
        args.output_path,
        graphecg_z=z_all,
        ecg_ids=ecg_ids_arr,
    )
    logger.info(f"Saved GraphECG latents to {args.output_path}")


if __name__ == "__main__":
    main()
