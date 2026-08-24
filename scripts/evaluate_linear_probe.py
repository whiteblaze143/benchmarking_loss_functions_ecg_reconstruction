#!/usr/bin/env python3
import os
import sys
import glob
import logging
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from tqdm import tqdm
import sqlite3

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.bootstrap_paths import setup_import_paths
setup_import_paths(include_fairseq=True)

from scripts.train_factorial_multimodel import build_architecture
from scripts.echonext_classifier import load_echonext_test_metadata, SHD_TASKS, SHD_LABEL_COLUMNS
from scripts.evaluate_echonext import EchoNextWaveforms, resolve_waveform

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def extract_unet_features(model, x):
    # MCMAModel bottleneck extraction
    e0 = model.down0(x)
    e1 = model.down1(e0)
    e2 = model.down2(e1)
    e3 = model.down3(e2)
    e4 = model.down4(e3)
    e5 = model.down5(e4) # Shape: (B, 512, L)
    # Global average pooling
    features = e5.mean(dim=-1)
    return features

def extract_msvae_features(model, x):
    # WearECGVAE encoder returns mu, log_var
    # Forward pass signature depends on whether it's unmasked or masked
    # In WearECGVAE, we can just use the encoder
    mu, logvar = model.encode(x)
    return mu.view(x.shape[0], -1)

def extract_ecg_aim_features(model, x):
    # alitok_vae_1d encoder returns encoder_out
    encoder_out = model.encoder(x)
    # usually it returns a dict with 'encoder_out' or similar, let's assume dict or tensor
    if isinstance(encoder_out, dict):
        z = encoder_out.get('encoder_out', encoder_out.get('z', None))
    else:
        z = encoder_out
    
    if hasattr(model, 'quantizer') and model.quantizer is not None:
        pass # we can use z directly
    
    # Global average pooling if it's 3D (B, C, L) or (B, L, C)
    if z.ndim == 3:
        if z.shape[1] > z.shape[2]: # likely B, L, C
            features = z.mean(dim=1)
        else: # likely B, C, L
            features = z.mean(dim=2)
    else:
        features = z.view(z.shape[0], -1)
    return features

def extract_features(model, arch_name, data_loader, device):
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in tqdm(data_loader, desc=f"Extracting {arch_name} features"):
            batch_x = batch_x.to(device)
            
            if arch_name == "unet":
                features = extract_unet_features(model, batch_x)
            elif arch_name == "msvae":
                features = extract_msvae_features(model, batch_x)
            elif arch_name == "ecg_aim":
                features = extract_ecg_aim_features(model, batch_x)
            else:
                raise ValueError(f"Unknown architecture: {arch_name}")
                
            all_features.append(features.cpu().numpy())
            all_labels.append(batch_y.numpy())
            
    return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, waveforms, labels, input_indices=[0, 1, 7]):
        self.waveforms = waveforms
        self.labels = labels
        self.input_indices = input_indices
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        x_all = self.waveforms[idx] # (12, L) or (L, 12)
        if x_all.shape[-1] == 12:
            x_all = x_all.T
            
        x_all = torch.tensor(x_all, dtype=torch.float32)
        
        # padding for specific models if needed, assume input length is 2500 for EchoNext
        # we interpolate to 5120 for UNet or 5000 for MSVAE/ECG-AIM if necessary, but 
        # let's assume we resize batch dynamically or rely on dataset 
        
        x_in = x_all[self.input_indices, :]
        y = self.labels[idx]
        return x_in, y

def pad_or_interpolate(x, target_len):
    # x is (B, C, L)
    if x.shape[-1] == target_len:
        return x
    return torch.nn.functional.interpolate(x, size=target_len, mode='linear', align_corners=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, required=True, choices=["unet", "msvae", "ecg_aim"])
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--target", type=str, default="lvef_lte_45_flag", help="Target label for evaluation")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data
    # For linear probing we need train and test sets. 
    # EchoNext test metadata is available. For demonstration, we'll split the 'test' set 
    # into train/test (80/20) since we don't have the EchoNext train split script handy,
    # or we can just run cross-validation on the available data.
    
    echonext_dir = _ROOT / "data" / "echonext"
    if not echonext_dir.exists():
        logging.error("EchoNext data not found.")
        return
        
    metadata_path = echonext_dir / "metadata.csv"
    transformer_path = echonext_dir / "tabular_transformer.pkl"
    
    df, mean, scale = load_echonext_test_metadata(metadata_path, transformer_path)
    
    waveform_path = resolve_waveform(echonext_dir)
    wave_data = np.load(waveform_path, mmap_mode="r")
    
    provenance_path = waveform_path.with_name("provenance.json")
    with open(provenance_path, "r") as f:
        provenance = json.load(f)
        
    waveforms = EchoNextWaveforms(wave_data, provenance)
    
    # Target: lvef_lte_45_flag as an example SHD task
    target_col = args.target
    labels = df[target_col].values
    
    # Filter out NaNs in labels
    valid_idx = ~np.isnan(labels)
    labels = labels[valid_idx]
    
    # Load all valid waveforms into memory (it's 5442 samples, should fit)
    valid_waveforms = waveforms.batch(0, len(waveforms))[valid_idx]
    
    dataset = SimpleDataset(valid_waveforms, labels)
    
    # Train/Test Split (80/20)
    np.random.seed(42)
    indices = np.random.permutation(len(dataset))
    split_idx = int(0.8 * len(indices))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    
    train_subset = torch.utils.data.Subset(dataset, train_idx)
    test_subset = torch.utils.data.Subset(dataset, test_idx)
    
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=args.batch_size, shuffle=False)
    
    # 2. Build Model
    model = build_architecture(args.arch, device)
    
    state_dict = torch.load(args.ckpt, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
    elif "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
        
    # Remove module. prefix if present
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    
    # We need to wrap data_loader to interpolate to 5120 (unet) or 5000 (msvae/ecg_aim)
    target_len = 5120 if args.arch == "unet" else 5000
    
    def wrapped_loader(loader):
        for x, y in loader:
            x = pad_or_interpolate(x, target_len)
            yield x, y
            
    # 3. Extract Features
    X_train, y_train = extract_features(model, args.arch, wrapped_loader(train_loader), device)
    X_test, y_test = extract_features(model, args.arch, wrapped_loader(test_loader), device)
    
    # 4. Train Linear Probe
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_train_scaled, y_train)
    
    y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
    y_pred = clf.predict(X_test_scaled)
    
    auprc = average_precision_score(y_test, y_pred_proba)
    auroc = roc_auc_score(y_test, y_pred_proba)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    
    logging.info(f"Results for {args.arch}:")
    logging.info(f"AUPRC: {auprc:.4f}")
    logging.info(f"AUROC: {auroc:.4f}")
    logging.info(f"Macro-F1: {macro_f1:.4f}")

if __name__ == "__main__":
    main()
