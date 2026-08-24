#!/home/mithunmanivannan/.venv/bin/python3
"""Evaluate regression to the mean for ECG reconstruction on all models and fiducials.

This script tests the hypothesis from Presacan et al. (2025) that deep learning
reconstruction of 12-lead ECGs from limited leads regress to the mean. It compares
all 48 models across 22 different fiducial markers (11 per lead for V3 and V6).
"""

import json
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import neurokit2 as nk
from scipy.signal import find_peaks
from scipy import stats
from tqdm import tqdm
import warnings

# Suppress neurokit warnings about messy signals
warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.bootstrap_paths import setup_import_paths
setup_import_paths()

from torch.utils.data import Dataset, DataLoader

class SimpleECGDataset(Dataset):
    def __init__(self, data_dir):
        self.files = sorted(Path(data_dir).glob("*.pt"), key=lambda p: int(p.stem))
    def __len__(self):
        return len(self.files)
    def __getitem__(self, index):
        path = self.files[index]
        signal = torch.load(path, weights_only=True).float()
        return (signal, int(path.stem))

from scripts.evaluate_comprehensive_registry import load_adapter
from unified_latents.engineering.src.evaluation.fiducial_bland_altman import (
    bland_altman_for_fiducial,
    plot_bland_altman,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

def find_all_fiducial_amplitudes(signal: np.ndarray, fs: int = 500):
    """Extract all fiducial amplitudes using neurokit's ecg_process pipeline."""
    try:
        # ecg_process returns a DataFrame of the processed signals and an info dict.
        # It handles cleaning, peak detection, and delineation.
        # We use method='neurokit' (default).
        signals, info = nk.ecg_process(signal, sampling_rate=fs)
        
        # We want to compare the amplitudes of the original signal at the points
        # where ecg_process detected features.
        # We extract the indices of 1s for each morphological feature column.
        features = [
            "ECG_P_Onsets", "ECG_P_Peaks", "ECG_P_Offsets",
            "ECG_Q_Peaks", "ECG_R_Onsets", "ECG_R_Peaks", "ECG_R_Offsets",
            "ECG_S_Peaks", "ECG_T_Onsets", "ECG_T_Peaks", "ECG_T_Offsets"
        ]
        
        amplitudes = {}
        for feature in features:
            if feature in signals.columns:
                # Get the indices where the feature is marked as 1
                indices = np.where(signals[feature] == 1)[0]
                short_k = feature.replace("ECG_", "")
                if len(indices) > 0:
                    amplitudes[short_k] = signal[indices]
                else:
                    amplitudes[short_k] = np.array([])
                    
        return amplitudes
    except Exception as e:
        # If neurokit fails on a noisy signal
        return {}


import concurrent.futures

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    project_root = Path(__file__).resolve().parents[1]
    results_json = project_root / "results/factorial_v4/clean_results.json"
    
    with open(results_json) as f:
        registry = json.load(f)
        
    model_ids = list(registry["models"].keys())
    logging.info(f"Evaluating all {len(model_ids)} models.")
    
    logging.info("Loading PTB-XL test set...")
    dataset = SimpleECGDataset(
        data_dir=project_root / "data/ptb_xl/tensors/test"
    )
    # Use larger batch size and num_workers since inference is fast,
    # but neurokit2 processing takes most of the time.
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    leads_to_evaluate = {
        "I": 0, "II": 1, "III": 2, "aVR": 3, "aVL": 4, "aVF": 5,
        "V1": 6, "V2": 7, "V3": 8, "V4": 9, "V5": 10, "V6": 11
    }
    
    out_dir = project_root / "results/factorial_v4/bland_altman_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    summary_file = out_dir / "regression_to_mean_summary.csv"
    if summary_file.exists():
        summary_file.unlink()
        
    # Pre-create executor
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
        
        # Nested dict: lead -> fiducial -> list of values
        real_amps = {lead: {} for lead in leads_to_evaluate}
        recon_amps = {lead: {} for lead in leads_to_evaluate}
        
        with torch.inference_mode():
            for batch in tqdm(loader):
                target = batch[0].to(device)
                try:
                    recon = adapter.reconstruct(target)
                except Exception as e:
                    logging.error(f"Inference error on {model_id}: {e}")
                    continue
                
                target_np = target.cpu().numpy()
                recon_np = recon.cpu().numpy()
                
                target_uv = target_np * 1000.0
                recon_uv = recon_np * 1000.0
                
                B = target.shape[0]
                
                # Flatten signals for parallel processing
                flat_target = []
                flat_recon = []
                lead_names_order = list(leads_to_evaluate.keys())
                
                for b in range(B):
                    for lead_name in lead_names_order:
                        lead_idx = leads_to_evaluate[lead_name]
                        flat_target.append(target_uv[b, lead_idx])
                        flat_recon.append(recon_uv[b, lead_idx])
                
                all_signals = flat_target + flat_recon
                
                # Execute in parallel
                results = list(executor.map(find_all_fiducial_amplitudes, all_signals))
                
                target_results = results[:len(flat_target)]
                recon_results = results[len(flat_target):]
                
                # Reconstruct into nested dicts
                idx = 0
                for b in range(B):
                    for lead_name in lead_names_order:
                        real_fid = target_results[idx]
                        recon_fid = recon_results[idx]
                        idx += 1
                        
                        common_keys = set(real_fid.keys()).intersection(set(recon_fid.keys()))
                        for k in common_keys:
                            if k not in real_amps[lead_name]:
                                real_amps[lead_name][k] = []
                                recon_amps[lead_name][k] = []
                                
                            real_k = real_fid[k]
                            recon_k = recon_fid[k]
                            
                            n_beats = min(len(real_k), len(recon_k))
                            for i in range(n_beats):
                                # Only keep pairs where both are not NaN
                                if not np.isnan(real_k[i]) and not np.isnan(recon_k[i]):
                                    real_amps[lead_name][k].append(real_k[i])
                                    recon_amps[lead_name][k].append(recon_k[i])
                                    
        # Compute stats for this model
        summary_rows = []
        for lead_name in leads_to_evaluate:
            for fid_name in real_amps[lead_name]:
                real_arr = np.array(real_amps[lead_name][fid_name])
                recon_arr = np.array(recon_amps[lead_name][fid_name])
                
                if len(real_arr) < 5:
                    continue
                    
                ba_stats = bland_altman_for_fiducial(real_arr, recon_arr)
                
                var_real = np.var(real_arr, ddof=1)
                var_recon = np.var(recon_arr, ddof=1)
                
                _, p_var = stats.levene(real_arr, recon_arr)
                _, p_mean = stats.ttest_ind(real_arr, recon_arr, equal_var=False)
                
                summary_rows.append({
                    "model_id": model_id,
                    "lead": lead_name,
                    "fiducial": fid_name,
                    "n_beats": len(real_arr),
                    "mean_real_uv": np.mean(real_arr),
                    "mean_recon_uv": np.mean(recon_arr),
                    "p_mean": p_mean,
                    "var_real": var_real,
                    "var_recon": var_recon,
                    "variance_ratio": var_recon / max(var_real, 1e-12),
                    "p_var": p_var,
                    "ba_bias": ba_stats.bias,
                    "ba_slope": ba_stats.slope,
                    "ba_robust_slope": ba_stats.robust_slope,
                })
                
        if summary_rows:
            df = pd.DataFrame(summary_rows)
            df.to_csv(summary_file, mode='a', header=not summary_file.exists(), index=False)
            
    logging.info(f"Analysis complete. Results saved to {out_dir}")

if __name__ == "__main__":
    main()
