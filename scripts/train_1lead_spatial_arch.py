#!/usr/bin/env python3
"""Trainer for Patient-Conditioned Spatial Architecture ECG-AIM (1->12 Reconstruction)."""

from __future__ import annotations

import argparse, copy, csv, gc, hashlib, io, json, math, os, random, shutil, signal, sqlite3, sys, tempfile, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from unified_latents.engineering.experimental.patient_spatial_ecg_aim import PatientSpatialECGAIM
from scripts.common_loss import CombinatorialCompositeLoss
from scripts.wavelet_ssl_queue import sha256_file, atomic_json

# ------------------------- Dataset Definitions -------------------------------

class PTBXLMetadataDataset(Dataset):
    """PTB-XL Dataset paired with patient metadata table."""
    def __init__(self, tensor_dir: str, metadata_df: pd.DataFrame, shuffle_metadata: bool = False):
        self.root = Path(tensor_dir)
        self.files = sorted(self.root.glob("*.pt"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
        self.metadata_lookup = {}
        
        # Meta columns: age_missing, sex_missing, height_missing, weight_missing, age_z, sex, height_z, weight_z, bmi_z
        meta_cols = ["age_z", "sex", "height_z", "weight_z", "age_missing", "sex_missing", "height_missing", "weight_missing", "bmi_z"]
        
        for _, row in metadata_df.iterrows():
            ecg_id = int(row["ecg_id"])
            vec = row[meta_cols].values.astype(np.float32)
            self.metadata_lookup[ecg_id] = vec
            
        self.all_meta_vecs = list(self.metadata_lookup.values())
        self.shuffle_metadata = shuffle_metadata

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        ecg_id = int(path.stem) if path.stem.isdigit() else idx
        x = torch.load(path, map_location="cpu", weights_only=True).float()
        
        if self.shuffle_metadata:
            meta = random.choice(self.all_meta_vecs)
        else:
            meta = self.metadata_lookup.get(ecg_id, np.zeros(9, dtype=np.float32))
            
        return {
            "waveform": x, # [12, 5000]
            "metadata": torch.tensor(meta, dtype=torch.float32), # [9]
            "ecg_id": ecg_id
        }


class DelineationDataset(Dataset):
    def __init__(self, tensor_dir: str):
        self.root = Path(tensor_dir)
        self.files = sorted(self.root.glob("*.pt"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = torch.load(self.files[idx], map_location="cpu", weights_only=False)
        return {
            "waveform": torch.as_tensor(d["waveform"], dtype=torch.float32),
            "segmentation": torch.as_tensor(d["segmentation"], dtype=torch.long),
            "seg_valid": torch.as_tensor(d.get("seg_valid", d["segmentation"] != -1), dtype=torch.bool),
        }

# ------------------------- Search Matrix Definitions -------------------------

CELLS_48 = [
    # Block A — Anchors & Metadata
    {"name": "A00_base_frozen", "mode": "A00_base_frozen"},
    {"name": "A01_param_matched", "mode": "A01_param_matched", "width": 864},
    {"name": "A02_meta_age_sex_concat", "mode": "A02_meta_age_sex_concat"},
    {"name": "A03_meta_all4_concat", "mode": "A03_meta_all4_concat"},
    {"name": "A04_meta_patient_token", "mode": "A04_meta_patient_token"},
    {"name": "A05_meta_lead_embedding", "mode": "A05_meta_lead_embedding"},
    {"name": "A06_meta_film_grid", "mode": "A06_meta_film_grid"},
    {"name": "A07_meta_film_leadwise", "mode": "A07_meta_film_leadwise"},
    {"name": "A08_meta_target_query", "mode": "A08_meta_target_query"},
    {"name": "A09_meta_hyperdecoder", "mode": "A09_meta_hyperdecoder"},
    {"name": "A10_meta_age_sex_film", "mode": "A10_meta_age_sex_film"},
    {"name": "A11_meta_all4_bmi_film", "mode": "A11_meta_all4_bmi_film"},
    {"name": "A12_meta_shuffled_control", "mode": "A12_meta_shuffled_control", "shuffle_meta": True},

    # Block B — Target-Query Spatial Decoders
    {"name": "B13_target_query_1x", "mode": "B13_target_query_1x"},
    {"name": "B14_target_query_2x", "mode": "B14_target_query_2x"},
    {"name": "B15_target_query_selfattn", "mode": "B15_target_query_selfattn"},
    {"name": "B16_target_query_gated", "mode": "B16_target_query_gated"},

    # Block C — Hierarchical Lead Structure
    {"name": "C17_plane_tokens", "mode": "C17_plane_tokens"},
    {"name": "C18_plane_experts", "mode": "C18_plane_experts"},
    {"name": "C19_plane_shared_then_split", "mode": "C19_plane_shared_then_split"},
    {"name": "C20_precordial_refiner", "mode": "C20_precordial_refiner"},

    # Block D — Spatial Graph Decoders
    {"name": "D21_graph_fixed", "mode": "D21_graph_fixed"},
    {"name": "D22_graph_geometry", "mode": "D22_graph_geometry"},
    {"name": "D23_graph_learnable", "mode": "D23_graph_learnable"},
    {"name": "D24_graph_patient", "mode": "D24_graph_patient"},

    # Block E — Target Specialization
    {"name": "E25_shared_head", "mode": "E25_shared_head"},
    {"name": "E26_per_lead_heads", "mode": "E26_per_lead_heads"},
    {"name": "E27_shared_plus_refiner", "mode": "E27_shared_plus_refiner"},
    {"name": "E28_grouped_heads", "mode": "E28_grouped_heads"},

    # Block F — Low-Dimensional Cardiac Basis
    {"name": "F29_basis_k3", "mode": "F29_basis_k3"},
    {"name": "F30_basis_k6", "mode": "F30_basis_k6"},
    {"name": "F31_basis_k8", "mode": "F31_basis_k8"},
    {"name": "F32_basis6_residual", "mode": "F32_basis6_residual"},

    # Block G — Universal Single-Lead Pretraining
    {"name": "G33_random_source8", "mode": "G33_random_source8", "random_source": True},
    {"name": "G34_balanced_source8", "mode": "G34_balanced_source8", "balanced_source": True},
    {"name": "G35_source8_then_I", "mode": "G35_source8_then_I", "source8_pretrain": True},
    {"name": "G36_source8_geometry", "mode": "G36_source8_geometry", "source8_geom": True},

    # Block H — Masked-Channel Curricula
    {"name": "H37_one_random_lead", "mode": "H37_one_random_lead"},
    {"name": "H38_random_1to3", "mode": "H38_random_1to3"},
    {"name": "H39_random_1to8", "mode": "H39_random_1to8"},
    {"name": "H40_curriculum_8_4_2_1", "mode": "H40_curriculum_8_4_2_1"},
    {"name": "H41_fullmask_then_I", "mode": "H41_fullmask_then_I"},

    # Block I — Same-Lead Deterministic Representations
    {"name": "I42_raw_d1", "mode": "I42_raw_d1"},
    {"name": "I43_raw_d1_d2", "mode": "I43_raw_d1_d2"},
    {"name": "I44_multires_raw", "mode": "I44_multires_raw"},
    {"name": "I45_full_plus_medianbeat", "mode": "I45_full_plus_medianbeat"},
    {"name": "I46_full_plus_beatstack", "mode": "I46_full_plus_beatstack"},
    {"name": "I47_rphase_encoding", "mode": "I47_rphase_encoding"},
]

# ------------------------- Metrics & Evaluation ------------------------------

def compute_pearson(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # pred, target: [B, L, 5000]
    p_mean = pred.mean(dim=-1, keepdim=True)
    t_mean = target.mean(dim=-1, keepdim=True)
    p_sub = pred - p_mean
    t_sub = target - t_mean
    cov = (p_sub * t_sub).sum(dim=-1)
    var_p = (p_sub ** 2).sum(dim=-1)
    var_t = (t_sub ** 2).sum(dim=-1)
    corr = cov / torch.sqrt(var_p * var_t + eps)
    return corr # [B, L]

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, obs_lead: int = 0) -> Dict[str, float]:
    model.eval()
    all_corrs = []
    
    with torch.no_grad():
        for batch in loader:
            y = batch["waveform"].to(device) # [B, 12, 5000]
            meta = batch["metadata"].to(device) # [B, 9]
            x_src = y[:, obs_lead:obs_lead+1, :] # [B, 1, 5000]
            
            out = model(x_src, obs_lead_idx=obs_lead, metadata=meta)
            pred = out["reconstruction"]
            
            corr = compute_pearson(pred, y) # [B, 12]
            all_corrs.append(corr.cpu())
            
    all_corrs = torch.cat(all_corrs, dim=0).numpy() # [N, 12]
    
    # Missing leads mask (all leads except observed)
    missing_mask = np.ones(12, dtype=bool)
    missing_mask[obs_lead] = False
    
    missing_corrs = all_corrs[:, missing_mask].mean(axis=1) # [N]
    
    precordial_corrs = all_corrs[:, 6:12].mean(axis=1) # V1-V6
    v1_v3_corrs = all_corrs[:, 6:9].mean(axis=1) # V1-V3
    
    return {
        "val_missing_pearson": float(np.mean(missing_corrs)),
        "val_missing_pearson_p05": float(np.quantile(missing_corrs, 0.05)),
        "val_precordial_pearson": float(np.mean(precordial_corrs)),
        "val_v1_v3_pearson": float(np.mean(v1_v3_corrs)),
        "val_lead_I_pearson": float(np.mean(all_corrs[:, 0])),
        "val_lead_II_pearson": float(np.mean(all_corrs[:, 1])),
    }

# ------------------------- Training Loop -------------------------------------

def train_spatial_arch(args) -> Dict[str, Any]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device} | Cell: {args.run_name}")
    
    # Load metadata parquet
    meta_path = Path(args.metadata_path)
    df_meta = pd.read_parquet(meta_path)
    
    tr_meta = df_meta[df_meta["split"] == "train"]
    va_meta = df_meta[df_meta["split"] == "val"]
    
    data_dir = Path(args.data_dir)
    tr_ds = PTBXLMetadataDataset(str(data_dir / "train"), tr_meta, shuffle_metadata=args.shuffle_metadata)
    va_ds = PTBXLMetadataDataset(str(data_dir / "val"), va_meta, shuffle_metadata=False)
    
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device.type=="cuda"))
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type=="cuda"))
    
    # Instantiate Model
    model = PatientSpatialECGAIM(
        mode=args.cell_mode,
        width=getattr(args, "width", 768),
        predict_delineation=True
    ).to(device)
    
    # Unchanged reconstruction criterion: 1110000
    criterion = CombinatorialCompositeLoss(args.factorial_mask)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(tr_loader))
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    best_score = -1.0
    best_metrics = {}
    
    epochs = 1 if args.quick_verify else args.epochs
    
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(tr_loader, desc=f"Epoch {ep}/{epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            y = batch["waveform"].to(device) # [B, 12, 5000]
            meta = batch["metadata"].to(device) # [B, 9]
            
            # Select observed source lead
            obs_lead = args.observed_lead
            if args.random_source:
                obs_lead = random.choice([0, 1, 6, 7, 8, 9, 10, 11])
                
            x_src = y[:, obs_lead:obs_lead+1, :]
            
            optimizer.zero_grad()
            out = model(x_src, obs_lead_idx=obs_lead, metadata=meta)
            pred = out["reconstruction"]
            
            loss, mse, corr, deriv, _, _, _, _ = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "corr": f"{corr.item():.4f}"})
            
            if args.quick_verify and batch_idx >= 1:
                break
                
        metrics = evaluate(model, va_loader, device, obs_lead=args.observed_lead)
        score = metrics["val_missing_pearson"]
        print(f"Epoch {ep} Validation -> Mean r: {score:.4f} | p05: {metrics['val_missing_pearson_p05']:.4f} | V1-V3: {metrics['val_v1_v3_pearson']:.4f}")
        
        if score > best_score:
            best_score = score
            best_metrics = {"epoch": ep, **metrics}
            
    summary = {
        "run_name": args.run_name,
        "cell_mode": args.cell_mode,
        "epochs": epochs,
        "best_score": best_score,
        **best_metrics
    }
    
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "_SUCCESS.json").write_text(json.dumps({"status": "completed", "run_name": args.run_name, "summary": summary}, indent=2) + "\n")
    
    return summary

# ------------------------- CLI Parser ----------------------------------------

def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="spatial_arch_cell")
    p.add_argument("--output-dir", default="refine-logs/spatial_arch_1lead_v1/runs/cell")
    p.add_argument("--data-dir", default="data/ptb_xl/tensors")
    p.add_argument("--data-manifest", default="refine-logs/ptbxl_tensor_content_manifest.json")
    p.add_argument("--metadata-path", default="refine-logs/spatial_arch_1lead_v1/assets/ptbxl_patient_metadata.parquet")
    p.add_argument("--cell-mode", default="A00_base_frozen")
    p.add_argument("--factorial-mask", default="1110000")
    p.add_argument("--observed-lead", type=int, default=0)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick-verify", action="store_true")
    p.add_argument("--shuffle-metadata", action="store_true")
    p.add_argument("--random-source", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--width", type=int, default=768)
    return p

if __name__ == "__main__":
    args = parser().parse_args()
    train_spatial_arch(args)
