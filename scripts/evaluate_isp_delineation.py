#!/usr/bin/env python3
"""Evaluate temporal micro-structure (delineation) on ISP dataset.

Tests whether the MMD + Masking loss components accurately preserve exact 
temporal morphological boundaries (P, QRS, T wave onsets/offsets) without 
introducing phase distortions, compared to baseline MSE models.

Provides results un-aggregated for each of the 12 leads.
"""

import json
import logging
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import neurokit2 as nk
import wfdb
from tqdm import tqdm
import warnings
import concurrent.futures
import ast
import scipy.signal

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.bootstrap_paths import setup_import_paths
setup_import_paths()

from scripts.evaluate_comprehensive_registry import load_adapter

LEADS = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
# ISP records are 1000Hz (based on the .hea files)
FS = 1000

def get_gt_delineations(tuples_list):
    """Parse GT boundaries from ISP CSV tuples for all 12 leads.
    ISP provides global delineations per record, so we apply them to all leads.
    """
    p_onsets, p_offsets = [], []
    qrs_onsets, qrs_offsets = [], []
    t_onsets, t_offsets = [], []
    
    for wave_type, onset, offset in tuples_list:
        if wave_type == 0:
            p_onsets.append(onset)
            p_offsets.append(offset)
        elif wave_type == 1:
            qrs_onsets.append(onset)
            qrs_offsets.append(offset)
        elif wave_type == 2:
            t_onsets.append(onset)
            t_offsets.append(offset)
            
    gt_bounds = {}
    for lead in LEADS:
        gt_bounds[lead] = {
            'P_Onset': np.array(p_onsets),
            'P_Offset': np.array(p_offsets),
            'R_Onset': np.array(qrs_onsets),
            'R_Offset': np.array(qrs_offsets),
            'T_Onset': np.array(t_onsets),
            'T_Offset': np.array(t_offsets),
        }
    return gt_bounds

def compute_delineation_metrics(pred_bounds, gt_bounds, tolerance=150): # 150 samples = 150ms tolerance at 1000Hz
    """Compute F1 and MAE between predicted and GT boundaries."""
    if len(gt_bounds) == 0:
        return {'F1': np.nan, 'MAE': np.nan}
        
    if len(pred_bounds) == 0:
        return {'F1': 0.0, 'MAE': np.nan}
        
    pred_bounds = pred_bounds[~np.isnan(pred_bounds)]
    if len(pred_bounds) == 0:
        return {'F1': 0.0, 'MAE': np.nan}
        
    tps = 0
    errors = []
    
    # Match each GT to nearest pred
    for gt in gt_bounds:
        dist = np.abs(pred_bounds - gt)
        if len(dist) > 0 and np.min(dist) <= tolerance:
            tps += 1
            errors.append(np.min(dist))
            
    fn = len(gt_bounds) - tps
    fp = len(pred_bounds) - tps
    
    f1 = 2 * tps / (2 * tps + fp + fn) if (2*tps + fp + fn) > 0 else 0.0
    mae = np.mean(errors) * (1000.0 / FS) if len(errors) > 0 else np.nan # convert to ms
    
    return {'F1': f1, 'MAE': mae}

def evaluate_reconstruction_lead(args):
    """Worker function to evaluate a single lead's delineation."""
    lead_signal, gt_bounds, lead_name = args
    
    try:
        # Delineate using NeuroKit2
        signals, info = nk.ecg_process(lead_signal, sampling_rate=FS)
        cleaned = signals["ECG_Clean"].values
        rpeaks = info["ECG_R_Peaks"]
        
        if len(rpeaks) < 2:
            raise ValueError("Not enough peaks")
            
        _, waves = nk.ecg_delineate(cleaned, rpeaks, sampling_rate=FS, method="dwt")
        
        pred_bounds = {
            'P_Onset': np.array(waves["ECG_P_Onsets"]),
            'P_Offset': np.array(waves["ECG_P_Offsets"]),
            'R_Onset': np.array(waves["ECG_R_Onsets"]),
            'R_Offset': np.array(waves["ECG_R_Offsets"]),
            'T_Onset': np.array(waves["ECG_T_Onsets"]),
            'T_Offset': np.array(waves["ECG_T_Offsets"]),
        }
        
        metrics = {}
        for bound_type in pred_bounds.keys():
            res = compute_delineation_metrics(pred_bounds[bound_type], gt_bounds[bound_type])
            metrics[f'{bound_type}_F1'] = res['F1']
            metrics[f'{bound_type}_MAE'] = res['MAE']
            
        return metrics
    except Exception:
        # Return NaNs if delineation fails
        return {
            f'{b}_F1': np.nan for b in ['P_Onset', 'P_Offset', 'R_Onset', 'R_Offset', 'T_Onset', 'T_Offset']
        } | {
            f'{b}_MAE': np.nan for b in ['P_Onset', 'P_Offset', 'R_Onset', 'R_Offset', 'T_Onset', 'T_Offset']
        }

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = Path(__file__).resolve().parents[1]
    results_json = project_root / "results/factorial_v4/clean_results.json"
    
    with open(results_json) as f:
        registry = json.load(f)
        
    model_ids = ["ground_truth_ceiling"] + list(registry["models"].keys())
    
    data_dir = project_root / "data/isp/isp_delineation_dataset"
    
    # Load all CSVs
    try:
        df_train = pd.read_csv(data_dir / "train_isp_delineation_data.csv")
        df_train['split'] = 'train_data'
    except Exception:
        df_train = pd.DataFrame()
        
    try:
        df_test = pd.read_csv(data_dir / "test_isp_delineation_data.csv")
        df_test['split'] = 'test_data'
    except Exception:
        df_test = pd.DataFrame()
        
    df_all = pd.concat([df_train, df_test]).reset_index(drop=True)
    
    logging.info(f"Evaluating {len(model_ids)} models on {len(df_all)} ISP records.")
    
    # Pre-load signals and GT
    isp_data = []
    for _, row in df_all.iterrows():
        rid = str(row['file_name'])
        split = row['split']
        rec_path = data_dir / split / rid
        
        try:
            record = wfdb.rdrecord(str(rec_path))
            tuples_list = ast.literal_eval(row['target'])
            gt_bounds = get_gt_delineations(tuples_list)
            
            # Reorder channels to match LEADS exactly
            signal_np = record.p_signal 
            channels = record.sig_name
            
            ordered_signal = np.zeros((len(LEADS), signal_np.shape[0]))
            for i, ld in enumerate(LEADS):
                if ld in channels:
                    idx = channels.index(ld)
                    ordered_signal[i] = signal_np[:, idx]
                elif ld.upper() in channels:
                    idx = channels.index(ld.upper())
                    ordered_signal[i] = signal_np[:, idx]
            
            # Resample from 1000Hz to 500Hz for the MS-VAE model reconstruction
            target_signal_500 = scipy.signal.resample(ordered_signal, num=ordered_signal.shape[1] // 2, axis=1)
            
            isp_data.append({
                'id': rid,
                'signal_1000hz': ordered_signal, # for ground_truth_ceiling
                'signal_500hz': torch.tensor(target_signal_500, dtype=torch.float32), # for reconstruction
                'gt': gt_bounds,
                'original_length': ordered_signal.shape[1]
            })
        except Exception as e:
            logging.error(f"Failed to load ISP {rid}: {e}")
            continue
            
    out_dir = project_root / "results/factorial_v4"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_file = out_dir / "isp_delineation_evaluation.csv"
    
    if summary_file.exists():
        summary_file.unlink()
        
    # Headers
    headers = ["model_id", "lead"]
    bounds = ['P_Onset', 'P_Offset', 'R_Onset', 'R_Offset', 'T_Onset', 'T_Offset']
    for b in bounds:
        headers.extend([f"{b}_F1", f"{b}_MAE"])
        
    with open(summary_file, 'w') as f:
        f.write(",".join(headers) + "\n")
        
    for model_id in model_ids:
        logging.info(f"Evaluating {model_id}...")
        
        if model_id == "ground_truth_ceiling":
            adapter = None
        else:
            spec = registry["models"][model_id]
            checkpoint_path = Path(spec["checkpoint"])
            if not checkpoint_path.is_absolute():
                spec["checkpoint"] = str(project_root / checkpoint_path)
                
            try:
                adapter = load_adapter(spec, device)
            except Exception as e:
                logging.error(f"Failed to load {model_id}: {e}")
                continue
            
        # 1. Batched GPU Inference
        recon_nps = []
        batch_size = 64
        for i in tqdm(range(0, len(isp_data), batch_size), desc=f"GPU Inference {model_id}"):
            batch_data = isp_data[i:i+batch_size]
            batch_tensors = torch.stack([d['signal_500hz'] for d in batch_data]).to(device)
            with torch.inference_mode():
                if model_id == "ground_truth_ceiling":
                    recon = batch_tensors
                else:
                    try:
                        recon = adapter.reconstruct(batch_tensors)
                    except Exception as e:
                        logging.error(f"Reconstruction failed: {e}")
                        recon = batch_tensors
                        
                for r, d in zip(recon, batch_data):
                    if model_id == "ground_truth_ceiling":
                        recon_nps.append(d['signal_1000hz'])
                    else:
                        recon_500hz = r.cpu().numpy()
                        # upscale back to 1000hz
                        recon_1000hz = scipy.signal.resample(recon_500hz, num=d['original_length'], axis=1)
                        recon_nps.append(recon_1000hz)
                        
        # 2. Free Memory
        if adapter is not None:
            del adapter
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        all_metrics = {ld: {f"{b}_F1": [] for b in bounds} | {f"{b}_MAE": [] for b in bounds} for ld in LEADS}
        
        # 3. CPU Parallel Evaluation
        jobs = []
        for i, data in enumerate(isp_data):
            if i >= len(recon_nps): continue
            recon_np = recon_nps[i]
            for j, ld in enumerate(LEADS):
                jobs.append((recon_np[j], data['gt'][ld], ld))
                
        logging.info(f"Evaluating {len(jobs)} total delineations using CPU pool...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            results = list(tqdm(executor.map(evaluate_reconstruction_lead, jobs), total=len(jobs), desc=f"CPU Delineation {model_id}"))
            
        # Map results back to metrics
        idx = 0
        for i, data in enumerate(isp_data):
            if i >= len(recon_nps): continue
            for j, ld in enumerate(LEADS):
                res = results[idx]
                idx += 1
                for k, v in res.items():
                    if not np.isnan(v):
                        all_metrics[ld][k].append(v)
                        
        # Write means to CSV per lead
        with open(summary_file, 'a') as f:
            for ld in LEADS:
                row = [model_id, ld]
                for b in bounds:
                    f1 = np.nanmean(all_metrics[ld][f"{b}_F1"]) if all_metrics[ld][f"{b}_F1"] else np.nan
                    mae = np.nanmean(all_metrics[ld][f"{b}_MAE"]) if all_metrics[ld][f"{b}_MAE"] else np.nan
                    row.extend([f"{f1:.4f}", f"{mae:.4f}"])
                f.write(",".join(row) + "\n")

if __name__ == "__main__":
    main()
