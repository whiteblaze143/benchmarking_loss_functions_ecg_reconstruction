#!/usr/bin/env python3
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.train_mcma_3lead import MCMAModel, PTBXLDataset
from unified_latents.engineering.models.fm_v2 import (
    MCMAModel_FM_V2,
    STMEM_Prior,
    MERL_Prior,
    KED_Prior,
    ECGFounder_Prior,
    HuBERTECG_Prior,
    ECGFM_Prior
)

def train_model(model, train_loader, val_loader, device, epochs=15, run_name="default"):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    ckpt_dir = f"/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/checkpoints/fm_v2_experiments/{run_name}"
    os.makedirs(ckpt_dir, exist_ok=True)
    
    print(f"\n--- Starting run: {run_name} ---")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item()
                
        t_loss = train_loss / len(train_loader)
        v_loss = val_loss / len(val_loader)
        print(f"[{run_name}] Epoch {epoch+1}/{epochs} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f}")
        
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), f"{ckpt_dir}/best_model.pt")
            
    print(f"Finished {run_name}. Best Val Loss: {best_val_loss:.4f}\n")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_dir = "/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/data/ptbxl_tensors"
    print("Loading datasets...")
    train_dataset = PTBXLDataset(f"{data_dir}/train")
    val_dataset = PTBXLDataset(f"{data_dir}/val")
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    experiments = [
        # F0: None (current raw+wavelet) - clean baseline
        ("F0_Baseline", lambda: MCMAModel(in_channels=3, out_channels=12)),
        
        # F1: ST-MEM (global FiLM) - clean
        ("F1_STMEM_FiLM", lambda: MCMAModel_FM_V2(
            in_channels=3, fm_class=STMEM_Prior, use_film=True, use_residual=False)),
        
        # F2: ST-MEM (FiLM + coarse residual) - primary
        ("F2_STMEM_Full", lambda: MCMAModel_FM_V2(
            in_channels=3, fm_class=STMEM_Prior, use_film=True, use_residual=True)),
        
        # F3: MERL (global FiLM) - clean
        ("F3_MERL_FiLM", lambda: MCMAModel_FM_V2(
            in_channels=3, fm_class=MERL_Prior, use_film=True, use_residual=False)),
        
        # F4: KED (global FiLM) - clean
        ("F4_KED_FiLM", lambda: MCMAModel_FM_V2(
            in_channels=3, fm_class=KED_Prior, use_film=True, use_residual=False)),
        
        # F5: ECGFounder-1L (global FiLM) - clean supervised control
        ("F5_ECGFounder_FiLM", lambda: MCMAModel_FM_V2(
            in_channels=3, fm_class=ECGFounder_Prior, use_film=True, use_residual=False)),
            
        # F6: HuBERT-ECG BASE (FiLM + residual) - exploratory (PTB-XL exposure)
        ("F6_HuBERTECG_Full", lambda: MCMAModel_FM_V2(
            in_channels=3, fm_class=HuBERTECG_Prior, use_film=True, use_residual=True)),
            
        # F7: ECG-FM (FiLM + residual) - exploratory (PTB-XL exposure)
        ("F7_ECGFM_Full", lambda: MCMAModel_FM_V2(
            in_channels=3, fm_class=ECGFM_Prior, use_film=True, use_residual=True)),
            
        # F8: Random-init capacity control for the winning architecture
        ("F8_RandomInit_Control", lambda: (
            model := MCMAModel_FM_V2(in_channels=3, fm_class=STMEM_Prior, use_film=True, use_residual=True),
            [setattr(p, 'requires_grad', True) for p in model.fm.parameters()],
            nn.init.normal_(model.fm.dummy_param), # Force a non-frozen fake state
            model
        )[3]),
    ]
    
    for run_name, model_fn in experiments:
        try:
            model = model_fn()
            train_model(model, train_loader, val_loader, device, epochs=15, run_name=run_name)
        except Exception as e:
            print(f"Failed to run {run_name}: {e}")

if __name__ == "__main__":
    main()
