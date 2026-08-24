#!/home/mithunmanivannan/.venv/bin/python3
"""Evaluate inter-lead correlation and error percentiles (Presacan et al. 2025 parity)."""

import json
import logging
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from scipy import stats
from tqdm import tqdm

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

def get_peak_amplitude(signal, lead_idx):
    """Extremely simplified peak extraction for interlead correlation. 
    Finds max (R-peak approximation) and min (S-peak approximation)."""
    # For a 10s ECG, just taking the global max is a crude but effective R-peak approximation
    # for the purpose of cross-lead correlation shift.
    lead_sig = signal[lead_idx]
    r_peak = np.max(lead_sig)
    s_peak = np.min(lead_sig)
    return r_peak, s_peak

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = Path(__file__).resolve().parents[1]
    
    with open(project_root / "results/factorial_v4/clean_results.json") as f:
        registry = json.load(f)
        
    # We will only evaluate a few key models to save time and demonstrate the effect.
    # Unet baseline, unet + mmd, unet + masking, msvae + masking + mmd
    key_models = [
        "unet__e0c0m0d0__s42", # Baseline MSE
        "unet__e0c0m0d1__s42", # MMD added
        "unet__e0c0m1d0__s42", # Masking added
        "msvae__e0c0m1d1__s42" # Full robust model
    ]
    
    dataset = SimpleECGDataset(data_dir=project_root / "data/ptb_xl/tensors/test")
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    
    out_dir = project_root / "results/factorial_v4/paper_parity"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    summary_rows = []
    
    for model_id in key_models:
        if model_id not in registry["models"]:
            continue
        logging.info(f"Evaluating Inter-lead & Percentiles for {model_id}...")
        spec = registry["models"][model_id]
        
        checkpoint_path = Path(spec["checkpoint"])
        if not checkpoint_path.is_absolute():
            spec["checkpoint"] = str(project_root / checkpoint_path)
            
        adapter = load_adapter(spec, device)
        
        # Collect for R2 correlation
        # lead 0 is I, lead 8 is V3
        real_I_R, real_V3_R = [], []
        recon_I_R, recon_V3_R = [], []
        
        # Collect for error percentiles (V3 R-peak)
        errors_V3_R = []
        
        with torch.inference_mode():
            for batch in tqdm(loader, desc=f"{model_id}"):
                target = batch[0].to(device)
                recon = adapter.reconstruct(target)
                
                target_np = target.cpu().numpy() * 1000.0 # uV
                recon_np = recon.cpu().numpy() * 1000.0 # uV
                
                B = target.shape[0]
                for b in range(B):
                    # Real
                    r_I_real, _ = get_peak_amplitude(target_np[b], 0) # Lead I
                    r_V3_real, _ = get_peak_amplitude(target_np[b], 8) # Lead V3
                    # Recon
                    r_I_recon, _ = get_peak_amplitude(recon_np[b], 0)
                    r_V3_recon, _ = get_peak_amplitude(recon_np[b], 8)
                    
                    real_I_R.append(r_I_real)
                    real_V3_R.append(r_V3_real)
                    recon_I_R.append(r_I_recon)
                    recon_V3_R.append(r_V3_recon)
                    
                    errors_V3_R.append(r_V3_recon - r_V3_real)
                    
        # Calculate Inter-lead R2 (Lead I vs V3)
        # Real
        slope_real, intercept_real, r_value_real, p_value_real, std_err_real = stats.linregress(real_I_R, real_V3_R)
        r2_real = r_value_real**2
        
        # Recon
        slope_recon, intercept_recon, r_value_recon, p_value_recon, std_err_recon = stats.linregress(recon_I_R, recon_V3_R)
        r2_recon = r_value_recon**2
        
        # Calculate Error Percentiles (V3 R-peak)
        err_5 = np.percentile(errors_V3_R, 5)
        err_10 = np.percentile(errors_V3_R, 10)
        err_90 = np.percentile(errors_V3_R, 90)
        err_95 = np.percentile(errors_V3_R, 95)
        
        summary_rows.append({
            "model_id": model_id,
            "r2_LeadI_V3_real": r2_real,
            "r2_LeadI_V3_recon": r2_recon,
            "err_5th": err_5,
            "err_10th": err_10,
            "err_90th": err_90,
            "err_95th": err_95,
            "mean_error_uV": np.mean(errors_V3_R)
        })
        
    df = pd.DataFrame(summary_rows)
    df.to_csv(out_dir / "interlead_percentiles_summary.csv", index=False)
    print("\n--- RESULTS ---")
    print(df.to_string())

if __name__ == "__main__":
    main()
