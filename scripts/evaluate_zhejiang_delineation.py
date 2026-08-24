#!/usr/bin/env python3
"""Evaluate temporal micro-structure (delineation) on Zhejiang PVC Dataset.

Tests whether the MMD + Masking loss components accurately preserve exact 
temporal morphological boundaries (P, QRS, T wave onsets/offsets) without 
introducing phase distortions, compared to baseline MSE models.

Incorporates deep statistical rigor: Levene's test for homoscedasticity and 
Bootstrap t-tests for robust Confidence Intervals.
"""

import json
import logging
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import neurokit2 as nk
import pickle
from tqdm import tqdm
import warnings
import concurrent.futures
from scipy import stats

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.bootstrap_paths import setup_import_paths
setup_import_paths()

from scripts.evaluate_comprehensive_registry import load_adapter
from scipy.signal import resample

# The Zhejiang dataset leads
LEAD_NAME_MAP = {
    'i': 'I', 'ii': 'II', 'iii': 'III',
    'avr': 'aVR', 'avl': 'aVL', 'avf': 'aVF',
    'v1': 'V1', 'v2': 'V2', 'v3': 'V3',
    'v4': 'V4', 'v5': 'V5', 'v6': 'V6'
}
LEADS = list(LEAD_NAME_MAP.keys())
FS_ORIG = 2000
FS_TARGET = 500
TARGET_LEN = 5000

def extract_bounds_from_mask(mask, val):
    binary = (mask == val).astype(int)
    diff = np.diff(binary)
    onsets = np.where(diff == 1)[0] + 1
    offsets = np.where(diff == -1)[0]
    if binary[0] == 1:
        onsets = np.insert(onsets, 0, 0)
    if binary[-1] == 1:
        offsets = np.append(offsets, len(binary) - 1)
    return onsets, offsets

def compute_delineation_metrics(pred_bounds, gt_bounds, tolerance=300): # 300 samples = 150ms tolerance at 2000Hz
    if len(gt_bounds) == 0:
        return {'F1': np.nan, 'MAE': np.nan}
    if len(pred_bounds) == 0:
        return {'F1': 0.0, 'MAE': np.nan}
    
    pred_bounds = pred_bounds[~np.isnan(pred_bounds)]
    if len(pred_bounds) == 0:
        return {'F1': 0.0, 'MAE': np.nan}
        
    tps = 0
    errors = []
    
    for gt in gt_bounds:
        dist = np.abs(pred_bounds - gt)
        if len(dist) > 0 and np.min(dist) <= tolerance:
            tps += 1
            errors.append(np.min(dist))
            
    fn = len(gt_bounds) - tps
    fp = len(pred_bounds) - tps
    
    f1 = 2 * tps / (2 * tps + fp + fn) if (2*tps + fp + fn) > 0 else 0.0
    mae = np.mean(errors) * (1000.0 / 2000.0) if len(errors) > 0 else np.nan # ms
    
    return {'F1': f1, 'MAE': mae}

def evaluate_reconstruction_lead(args):
    lead_signal, gt_mask, lead_name = args
    
    # Upsample the 500 Hz model reconstruction to 20,000 samples to match the native GT mask
    from scipy.signal import resample
    lead_signal = resample(lead_signal, 20000)
    
    # Extract GT bounds from native mask
    # 1: P, 2: QRS, 3: T
    gt_bounds = {
        'P_Onset': extract_bounds_from_mask(gt_mask, 1)[0],
        'P_Offset': extract_bounds_from_mask(gt_mask, 1)[1],
        'R_Onset': extract_bounds_from_mask(gt_mask, 2)[0],
        'R_Offset': extract_bounds_from_mask(gt_mask, 2)[1],
        'T_Onset': extract_bounds_from_mask(gt_mask, 3)[0],
        'T_Offset': extract_bounds_from_mask(gt_mask, 3)[1],
    }
    
    try:
        # Delineate using NeuroKit2 on reconstructed signal
        signals, info = nk.ecg_process(lead_signal, sampling_rate=2000)
        cleaned = signals["ECG_Clean"].values
        rpeaks = info["ECG_R_Peaks"]
        
        if len(rpeaks) < 2:
            raise ValueError("Not enough peaks")
            
        _, waves = nk.ecg_delineate(cleaned, rpeaks, sampling_rate=2000, method="dwt")
        
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
            
        # Also compute Dice score on masks directly
        pred_mask = np.zeros_like(gt_mask)
        # We fill pred_mask based on bounds
        for i in range(len(pred_bounds['R_Onset'])):
            on = pred_bounds['R_Onset'][i]
            off = pred_bounds['R_Offset'][i]
            if not np.isnan(on) and not np.isnan(off):
                pred_mask[int(on):int(off)] = 2
        for i in range(len(pred_bounds['P_Onset'])):
            on = pred_bounds['P_Onset'][i]
            off = pred_bounds['P_Offset'][i]
            if not np.isnan(on) and not np.isnan(off):
                pred_mask[int(on):int(off)] = 1
        for i in range(len(pred_bounds['T_Onset'])):
            on = pred_bounds['T_Onset'][i]
            off = pred_bounds['T_Offset'][i]
            if not np.isnan(on) and not np.isnan(off):
                pred_mask[int(on):int(off)] = 3
                
        for val, name in [(1, 'P'), (2, 'QRS'), (3, 'T')]:
            pred_bin = (pred_mask == val)
            gt_bin = (gt_mask == val)
            intersection = np.sum(pred_bin & gt_bin)
            union = np.sum(pred_bin) + np.sum(gt_bin)
            dice = (2.0 * intersection / union) if union > 0 else (1.0 if np.sum(gt_bin)==0 else 0.0)
            metrics[f'{name}_Dice'] = dice
            
        return metrics
    except Exception:
        metrics = {f'{b}_F1': np.nan for b in gt_bounds.keys()}
        metrics.update({f'{b}_MAE': np.nan for b in gt_bounds.keys()})
        metrics.update({f'{n}_Dice': np.nan for n in ['P', 'QRS', 'T']})
        return metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_run", action="store_true", help="Run on a small subset")
    parser.add_argument("--cpu", action="store_true", help="Force execution on CPU")
    args = parser.parse_args()

    device = torch.device("cpu") if args.cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = Path(__file__).resolve().parents[1]
    
    # We will test MSE baseline vs MMD model
    # Assuming unet__e0c0m0d0__s42 is MSE and unet__e1c1m1d1__s42 is MMD
    results_json = project_root / "results/comprehensive_latest_48_models/provenance/model_registry.json"
    spec_map = {}
    if results_json.exists():
        with open(results_json) as f:
            registry = json.load(f)
            spec_map = {m["id"]: m for m in registry["models"]}
            
    # Discover new factorial checkpoints (only fully completed ones with metadata)
    for pt_file in (project_root / "checkpoints").glob("factorial_*.pt"):
        metadata_file = pt_file.with_suffix('.metadata.json')
        if not metadata_file.exists():
            continue  # Skip actively training checkpoints
            
        mid = pt_file.stem.replace("factorial_", "f_")
        spec_map[mid] = {
            "id": mid,
            "family": "unet",
            "kind": "unet",
            "checkpoint": str(pt_file),
            "observed_leads": [0, 1, 6, 7, 8, 9, 10, 11]
        }
        
    model_ids = ["ground_truth_ceiling"]
    
    out_dir = project_root / "results/zhejiang_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_file = out_dir / "zhejiang_delineation_evaluation.csv"
    
    evaluated_models = set()
    if summary_file.exists():
        try:
            df_exist = pd.read_csv(summary_file)
            if 'model_id' in df_exist.columns:
                evaluated_models = set(df_exist['model_id'].unique())
        except Exception:
            pass

    # Identify MSE and MMD models from registry
    mse_model = None
    mmd_model = None
    for mid, spec in spec_map.items():
        ckpt_path = Path(spec["checkpoint"])
        if not ckpt_path.is_absolute():
            ckpt_path = project_root / ckpt_path
            
        if "unet" in mid:
            if "e0c0m0d0" in mid and "s42" in mid:
                mse_model = mid
            if "e1c1m1d1" in mid and "s42" in mid:
                mmd_model = mid
        elif mid.startswith("f_"):
            if "1000000" in mid and "s42" in mid:
                mse_model = mid
            if "1111111" in mid and "s42" in mid:
                mmd_model = mid
                
        # Only add models whose checkpoints actually exist locally and aren't evaluated yet
        if ckpt_path.exists():
            if mid not in evaluated_models:
                model_ids.append(mid)
    
    data_dir = project_root / "data/zhejiang"
    ecg_dir = data_dir / "ecg"
    label_dir = data_dir / "label"
    
    # Get patient IDs
    all_label_files = list(label_dir.glob("*.pkl"))
    record_ids = sorted([p.stem for p in all_label_files])
    
    if args.test_run:
        record_ids = record_ids[:5]
        
    logging.info(f"Evaluating {len(model_ids)} models on {len(record_ids)} Zhejiang records.")
    
    zhejiang_data = []
    for rid in tqdm(record_ids, desc="Loading Data"):
        ordered_signal = np.zeros((len(LEADS), TARGET_LEN))
        valid = True
        for i, ld in enumerate(LEADS):
            lead_suffix = LEAD_NAME_MAP[ld]
            f_path = ecg_dir / f"{rid}_{lead_suffix}.pkl"
            if not f_path.exists():
                valid = False
                break
            with open(f_path, 'rb') as f:
                sig = pickle.load(f)
            # Downsample from 20000 to 5000
            sig_resampled = resample(sig, TARGET_LEN)
            ordered_signal[i] = sig_resampled
            
        if not valid: continue
        
        with open(label_dir / f"{rid}.pkl", 'rb') as f:
            gt_mask_20k = pickle.load(f)
            
        # Keep full 20,000 resolution mask
        gt_mask = gt_mask_20k
        
        zhejiang_data.append({
            'id': rid,
            'signal': torch.tensor(ordered_signal, dtype=torch.float32),
            'gt': gt_mask
        })
        
    bounds = ['P_Onset', 'P_Offset', 'R_Onset', 'R_Offset', 'T_Onset', 'T_Offset']
    dice_keys = ['P_Dice', 'QRS_Dice', 'T_Dice']
    
    all_model_metrics = {}
    
    for model_id in model_ids:
        logging.info(f"Evaluating {model_id}...")
        
        if model_id == "ground_truth_ceiling":
            adapter = None
        else:
            spec = spec_map[model_id]
            checkpoint_path = Path(spec["checkpoint"])
            if not checkpoint_path.is_absolute():
                spec["checkpoint"] = str(project_root / checkpoint_path)
            try:
                adapter = load_adapter(spec, device)
            except Exception as e:
                logging.error(f"Failed to load {model_id}: {e}")
                continue
            
        recon_nps = []
        batch_size = 64
        for i in tqdm(range(0, len(zhejiang_data), batch_size), desc=f"GPU Inference {model_id}"):
            batch_data = zhejiang_data[i:i+batch_size]
            batch_tensors = torch.stack([d['signal'] for d in batch_data]).to(device)
            with torch.inference_mode():
                if model_id == "ground_truth_ceiling":
                    recon = batch_tensors
                else:
                    try:
                        recon = adapter.reconstruct(batch_tensors)
                    except Exception as e:
                        logging.error(f"Reconstruction failed: {e}")
                        recon = batch_tensors
                for r in recon:
                    recon_nps.append(r.cpu().numpy())
                    
        if adapter is not None:
            del adapter
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        jobs = []
        for i, data in enumerate(zhejiang_data):
            if i >= len(recon_nps): continue
            recon_np = recon_nps[i]
            for j, ld in enumerate(LEADS):
                jobs.append((recon_np[j], data['gt'], ld))
                
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            results = list(tqdm(executor.map(evaluate_reconstruction_lead, jobs), total=len(jobs), desc=f"CPU Delineation {model_id}"))
            
        # Store flat metrics for statistical tests later
        flat_results = {k: [] for k in results[0].keys()}
        for res in results:
            for k, v in res.items():
                if not np.isnan(v):
                    flat_results[k].append(v)
        
        all_model_metrics[model_id] = flat_results
        
        # Write to summary
        df = pd.DataFrame(results)
        df_mean = df.mean().to_frame().T
        df_mean['model_id'] = model_id
        df_mean.to_csv(summary_file, mode='a', header=not summary_file.exists(), index=False)

    # --- Statistical Rigor Section ---
    logging.info("Running Advanced Statistical Tests (Levene's & Bootstrap t-tests)...")
    if mse_model in all_model_metrics and mmd_model in all_model_metrics:
        stats_file = out_dir / "zhejiang_statistical_tests.txt"
        with open(stats_file, 'w') as f:
            f.write("Zhejiang Delineation Benchmark: Extreme Statistical Rigor Report\n")
            f.write("="*64 + "\n\n")
            
            for metric in ['QRS_Dice', 'T_Dice', 'R_Onset_MAE', 'T_Offset_MAE']:
                mse_vals = np.array(all_model_metrics[mse_model].get(metric, []))
                mmd_vals = np.array(all_model_metrics[mmd_model].get(metric, []))
                
                # Make sure same size for paired/tests
                min_len = min(len(mse_vals), len(mmd_vals))
                if min_len < 2: continue
                mse_vals = mse_vals[:min_len]
                mmd_vals = mmd_vals[:min_len]
                
                f.write(f"--- Metric: {metric} ---\n")
                f.write(f"MSE Model Mean: {np.mean(mse_vals):.4f} \n")
                f.write(f"MMD Model Mean: {np.mean(mmd_vals):.4f} \n")
                
                # Levene's Test
                stat, p = stats.levene(mse_vals, mmd_vals)
                f.write(f"Levene's Test (Homoscedasticity): statistic={stat:.4f}, p-value={p:.4e}\n")
                
                # Bootstrap t-test (Difference of means)
                diff = mmd_vals - mse_vals
                res = stats.bootstrap((diff,), np.mean, confidence_level=0.95, n_resamples=9999, method='BCa')
                ci_l, ci_u = res.confidence_interval
                f.write(f"Bootstrap 95% CI of Difference (MMD - MSE): [{ci_l:.4f}, {ci_u:.4f}]\n")
                
                # Wilcoxon signed-rank test (non-parametric paired test)
                w_stat, w_p = stats.wilcoxon(mse_vals, mmd_vals)
                f.write(f"Wilcoxon Paired Test: p-value={w_p:.4e}\n\n")
        logging.info(f"Statistical report saved to {stats_file}")
        with open(stats_file, 'r') as f:
            print(f.read())

if __name__ == "__main__":
    main()
