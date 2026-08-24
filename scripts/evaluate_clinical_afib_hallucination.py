#!/usr/bin/env python3
"""Evaluate clinical AFIB hallucination metrics for ECG reconstruction models.

This script tests the hypothesis that MSE-based models hallucinate P-waves 
during Atrial Fibrillation (AF) because they regress to the Normal Sinus Rhythm (SR) mean.
This hallucination causes severe false-negative diagnostic failures for AF wearables.

It calculates:
1. P-Wave Hallucination (Count & Amplitude) for AFIB vs SR.
2. RR Interval Irregularity (Coefficient of Variation) for AFIB vs SR.
"""

import json
import logging
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import neurokit2 as nk
from tqdm import tqdm
import warnings
import concurrent.futures

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.bootstrap_paths import setup_import_paths
setup_import_paths()

from torch.utils.data import Dataset, DataLoader
from scripts.evaluate_comprehensive_registry import load_adapter

class AFIBSubsetDataset(Dataset):
    def __init__(self, data_dir, afib_idx, sr_idx):
        # Maps index string to path
        all_files = {int(p.stem): p for p in Path(data_dir).glob("*.pt")}
        self.samples = []
        for idx in afib_idx:
            if idx in all_files:
                self.samples.append((all_files[idx], 1)) # 1 = AFIB
        for idx in sr_idx:
            if idx in all_files:
                self.samples.append((all_files[idx], 0)) # 0 = SR
                
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, index):
        path, label = self.samples[index]
        signal = torch.load(path, weights_only=True).float()
        return (signal, label, int(path.stem))

def extract_afib_features(signal: np.ndarray, fs: int = 500):
    """Extract AFIB features using neurokit2."""
    try:
        signals, info = nk.ecg_process(signal, sampling_rate=fs)
        
        p_peaks = np.where(signals["ECG_P_Peaks"] == 1)[0]
        p_count = len(p_peaks)
        p_amp = np.mean(signal[p_peaks]) if p_count > 0 else 0.0
            
        r_peaks = info["ECG_R_Peaks"]
        if len(r_peaks) > 1:
            rr_intervals = np.diff(r_peaks) / fs
            rr_cv = np.std(rr_intervals) / np.mean(rr_intervals)
        else:
            rr_cv = 0.0
            
        return {
            "p_count": p_count,
            "p_amp": p_amp * 1000, # convert to uV
            "rr_cv": rr_cv
        }
    except Exception:
        return None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = Path(__file__).resolve().parents[1]
    results_json = project_root / "results/factorial_v4/clean_results.json"
    
    with open(results_json) as f:
        registry = json.load(f)
        
    model_ids = list(registry["models"].keys())
    # model_ids = ["unet__e0c0m0d0__s42"]
    
    # 1. Get Subset Indices
    df = pd.read_csv(project_root / "data/ptb_xl/ptbxl_database.csv", index_col="ecg_id")
    test_df = df[df["strat_fold"] == 10]
    afib_idx = test_df[test_df["scp_codes"].str.contains("AFIB")].index.tolist()
    sr_idx = test_df[test_df["scp_codes"].str.contains("SR") & ~test_df["scp_codes"].str.contains("AFIB")].index.tolist()
    # Balance classes (152 AFIB, 152 SR)
    sr_idx = sr_idx[:len(afib_idx)]
    
    logging.info(f"Evaluating AFIB Hallucination on {len(afib_idx)} AFIB and {len(sr_idx)} SR test samples across {len(model_ids)} models.")
    
    data_dir = project_root / "data/ptb_xl/tensors/test"
    dataset = AFIBSubsetDataset(data_dir, afib_idx, sr_idx)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    out_dir = project_root / "results/factorial_v4/clinical_afib"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_file = out_dir / "clinical_afib_metrics.csv"
    
    if summary_file.exists():
        summary_file.unlink()
        
    # We will write the CSV progressively
    with open(summary_file, 'w') as f:
        f.write("model_id,afib_p_count_diff,afib_p_amp_diff,sr_p_count_diff,sr_p_amp_diff,afib_rr_cv_diff,sr_rr_cv_diff\n")
    
    executor = concurrent.futures.ProcessPoolExecutor()
    
    for model_id in model_ids:
        logging.info(f"Evaluating {model_id}...")
        spec = registry["models"][model_id]
        
        checkpoint_path = Path(spec["checkpoint"])
        if not checkpoint_path.is_absolute():
            spec["checkpoint"] = str(project_root / checkpoint_path)
            
        try:
            adapter = load_adapter(spec, device)
        except Exception as e:
            logging.error(f"Failed to load {model_id}: {e}")
            continue
        
        # We will extract features from Lead V1 (index 6) as it is best for P-waves.
        # Wait, the reconstructed leads are III, aVR, aVL, aVF, V1, V3, V4, V5, V6.
        # V1 is at index 6 in the 12-lead array. Yes, Lead V1 is arguably the best for AFIB detection.
        LEAD_IDX = 6
        
        metrics = {
            "afib": {"gt_count": [], "recon_count": [], "gt_amp": [], "recon_amp": [], "gt_rr": [], "recon_rr": []},
            "sr":   {"gt_count": [], "recon_count": [], "gt_amp": [], "recon_amp": [], "gt_rr": [], "recon_rr": []}
        }
        
        with torch.inference_mode():
            for batch in tqdm(loader, desc=f"Processing {model_id}"):
                target = batch[0].to(device)
                labels = batch[1].numpy()
                try:
                    recon = adapter.reconstruct(target)
                except Exception as e:
                    continue
                
                target_np = target.cpu().numpy()
                recon_np = recon.cpu().numpy()
                
                flat_target = [target_np[i, LEAD_IDX] for i in range(target_np.shape[0])]
                flat_recon = [recon_np[i, LEAD_IDX] for i in range(recon_np.shape[0])]
                
                target_results = list(executor.map(extract_afib_features, flat_target))
                recon_results = list(executor.map(extract_afib_features, flat_recon))
                
                for t_res, r_res, label in zip(target_results, recon_results, labels):
                    if t_res is not None and r_res is not None:
                        group = "afib" if label == 1 else "sr"
                        metrics[group]["gt_count"].append(t_res["p_count"])
                        metrics[group]["recon_count"].append(r_res["p_count"])
                        metrics[group]["gt_amp"].append(t_res["p_amp"])
                        metrics[group]["recon_amp"].append(r_res["p_amp"])
                        metrics[group]["gt_rr"].append(t_res["rr_cv"])
                        metrics[group]["recon_rr"].append(r_res["rr_cv"])
                        
        if len(metrics["afib"]["gt_count"]) == 0:
            logging.error(f"No valid AFIB features extracted for {model_id}")
            continue
            
        # Compute differences (Recon - GT)
        # For AFIB P-count, a positive difference means Hallucination (Recon has more P-waves than GT).
        afib_p_count_diff = np.mean(np.array(metrics["afib"]["recon_count"]) - np.array(metrics["afib"]["gt_count"]))
        afib_p_amp_diff = np.mean(np.array(metrics["afib"]["recon_amp"]) - np.array(metrics["afib"]["gt_amp"]))
        afib_rr_cv_diff = np.mean(np.array(metrics["afib"]["recon_rr"]) - np.array(metrics["afib"]["gt_rr"]))
        
        sr_p_count_diff = np.mean(np.array(metrics["sr"]["recon_count"]) - np.array(metrics["sr"]["gt_count"]))
        sr_p_amp_diff = np.mean(np.array(metrics["sr"]["recon_amp"]) - np.array(metrics["sr"]["gt_amp"]))
        sr_rr_cv_diff = np.mean(np.array(metrics["sr"]["recon_rr"]) - np.array(metrics["sr"]["gt_rr"]))
        
        # Append to CSV
        with open(summary_file, 'a') as f:
            f.write(f"{model_id},{afib_p_count_diff:.4f},{afib_p_amp_diff:.4f},{sr_p_count_diff:.4f},{sr_p_amp_diff:.4f},{afib_rr_cv_diff:.4f},{sr_rr_cv_diff:.4f}\n")
            
        logging.info(f"{model_id} - AFIB P-wave Hallucination Count: {afib_p_count_diff:.2f}, Amplitude Diff: {afib_p_amp_diff:.2f}uV")

if __name__ == "__main__":
    main()
