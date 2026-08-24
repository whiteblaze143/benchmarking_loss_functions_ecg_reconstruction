#!/usr/bin/env python3
"""Evaluate clinical QTc metrics for ECG reconstruction models.

This script tests the hypothesis from Ansari et al. (Circulation 2026) that
MSE-based models smooth out clinical extremes like prolonged QTc, and MMD 
recovers this sensitivity.

It calculates:
1. Mean Absolute Error (MAE) of QTc
2. Pearson r correlation of QTc
3. AUROC / AUPRC / F1 for Prolonged QTc Classification
"""

import json
import logging
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import neurokit2 as nk
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
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

class SimpleECGDataset(Dataset):
    def __init__(self, data_dir):
        self.files = sorted(Path(data_dir).glob("*.pt"), key=lambda p: int(p.stem))
    def __len__(self):
        return len(self.files)
    def __getitem__(self, index):
        path = self.files[index]
        signal = torch.load(path, weights_only=True).float()
        return (signal, int(path.stem))

def extract_qtc(signal: np.ndarray, fs: int = 500):
    """Extract QTc and QRS duration using neurokit2."""
    try:
        signals, info = nk.ecg_process(signal, sampling_rate=fs)
        cleaned = signals["ECG_Clean"].values
        rpeaks = info["ECG_R_Peaks"]
        
        if len(rpeaks) < 2:
            return None
            
        _, waves = nk.ecg_delineate(cleaned, rpeaks, sampling_rate=fs, method="dwt")
        
        rr_intervals = np.diff(rpeaks) / fs
        mean_rr = np.nanmean(rr_intervals)
        
        onsets = np.array(waves["ECG_R_Onsets"])
        toffsets = np.array(waves["ECG_T_Offsets"])
        roffsets = np.array(waves["ECG_R_Offsets"])
        
        qt_intervals = []
        qrs_durations = []
        
        for p in rpeaks:
            onset_cands = onsets[onsets < p]
            if len(onset_cands) == 0 or np.isnan(onset_cands[-1]):
                continue
            onset = onset_cands[-1]
            
            toffset_cands = toffsets[toffsets > p]
            if len(toffset_cands) == 0 or np.isnan(toffset_cands[0]):
                continue
            toffset = toffset_cands[0]
            
            qt = (toffset - onset) / fs
            if 0.2 < qt < 0.8:
                qt_intervals.append(qt)
                
            roffset_cands = roffsets[roffsets > p]
            if len(roffset_cands) > 0 and not np.isnan(roffset_cands[0]):
                roffset = roffset_cands[0]
                if roffset > onset:
                    qrs_durations.append((roffset - onset) / fs)
                    
        if not qt_intervals:
            return None
            
        mean_qt = np.nanmean(qt_intervals)
        mean_qrs = np.nanmean(qrs_durations) if qrs_durations else 0.1
        qtc = mean_qt / np.sqrt(mean_rr)
        
        return {
            "qt_ms": mean_qt * 1000,
            "rr_ms": mean_rr * 1000,
            "qtc_ms": qtc * 1000,
            "qrs_ms": mean_qrs * 1000
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
    logging.info(f"Evaluating {len(model_ids)} models for Clinical QTc metrics.")
    
    data_dir = project_root / "data/ptb_xl/tensors/test"
    dataset = SimpleECGDataset(data_dir)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    out_dir = project_root / "results/factorial_v4/clinical_qtc"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_file = out_dir / "clinical_qtc_metrics.csv"
    
    if summary_file.exists():
        summary_file.unlink()
        
    # We will write the CSV progressively
    with open(summary_file, 'w') as f:
        f.write("model_id,mae_qtc,pearson_r,auroc_prolonged,auprc_prolonged,f1_prolonged,sens_prolonged,spec_prolonged,n_samples,n_prolonged\n")
    
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
        
        # We will extract QTc from Lead V6 (index 11) for simplicity as it is a reconstructed lead commonly used for QT.
        LEAD_IDX = 11
        
        target_qtc_list = []
        recon_qtc_list = []
        target_qrs_list = []
        recon_qrs_list = []
        
        with torch.inference_mode():
            for batch in tqdm(loader, desc=f"Processing {model_id}"):
                target = batch[0].to(device)
                try:
                    recon = adapter.reconstruct(target)
                except Exception as e:
                    continue
                
                target_np = target.cpu().numpy()
                recon_np = recon.cpu().numpy()
                
                # Flatten batch for lead V6
                flat_target = [target_np[i, LEAD_IDX] for i in range(target_np.shape[0])]
                flat_recon = [recon_np[i, LEAD_IDX] for i in range(recon_np.shape[0])]
                
                # Extract target QTc
                target_results = list(executor.map(extract_qtc, flat_target))
                # Extract recon QTc
                recon_results = list(executor.map(extract_qtc, flat_recon))
                
                for t_res, r_res in zip(target_results, recon_results):
                    if t_res is not None and r_res is not None:
                        # Drop extreme outliers (e.g. QTc > 1000ms usually implies neurokit failure)
                        if 200 < t_res["qtc_ms"] < 1000 and 200 < r_res["qtc_ms"] < 1000:
                            target_qtc_list.append(t_res["qtc_ms"])
                            recon_qtc_list.append(r_res["qtc_ms"])
                            target_qrs_list.append(t_res["qrs_ms"])
                            recon_qrs_list.append(r_res["qrs_ms"])
                            
        if len(target_qtc_list) == 0:
            logging.error(f"No valid QTc extracted for {model_id}")
            continue
            
        # Compute MAE
        target_qtc = np.array(target_qtc_list)
        recon_qtc = np.array(recon_qtc_list)
        target_qrs = np.array(target_qrs_list)
        
        mae = np.mean(np.abs(target_qtc - recon_qtc))
        
        # Compute Pearson r
        r, _ = stats.pearsonr(target_qtc, recon_qtc)
        
        # Define Prolonged QTc criteria (Dynamic Threshold)
        # > 500ms for narrow QRS (<= 100ms)
        # > 550ms for wide QRS (> 100ms)
        thresholds = np.where(target_qrs > 100, 550.0, 500.0)
        
        y_true = (target_qtc > thresholds).astype(int)
        y_score = recon_qtc # Raw QTc as score for AUROC
        y_pred = (recon_qtc > thresholds).astype(int)
        
        n_prolonged = np.sum(y_true)
        n_samples = len(y_true)
        
        if n_prolonged > 0 and n_prolonged < n_samples:
            auroc = roc_auc_score(y_true, y_score)
            auprc = average_precision_score(y_true, y_score)
            f1 = f1_score(y_true, y_pred)
            
            # Sensitivity / Specificity
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            
            sens = tp / (tp + fn) if (tp+fn) > 0 else 0
            spec = tn / (tn + fp) if (tn+fp) > 0 else 0
        else:
            auroc = float('nan')
            auprc = float('nan')
            f1 = float('nan')
            sens = float('nan')
            spec = float('nan')
            
        # Append to CSV
        with open(summary_file, 'a') as f:
            f.write(f"{model_id},{mae:.4f},{r:.4f},{auroc:.4f},{auprc:.4f},{f1:.4f},{sens:.4f},{spec:.4f},{n_samples},{n_prolonged}\n")
            
        logging.info(f"{model_id} - MAE: {mae:.2f}ms, r: {r:.3f}, AUROC: {auroc:.3f}, F1: {f1:.3f}")

if __name__ == "__main__":
    main()
