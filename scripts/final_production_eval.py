
import os
import sys
from pathlib import Path
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from scripts.bootstrap_paths import setup_import_paths
setup_import_paths()
import torch
import numpy as np
import pandas as pd
import neurokit2 as nk
from tqdm import tqdm
from scipy.stats import pearsonr
import json


from src.reconstruction.learn_functions.bridge import ModularBridge
from src.data.multi_source_dataset import MultiSourceECGDataset

def compute_all_nk2_metrics(tgt, pred, fs=500):
    clean_tgt = nk.ecg_clean(tgt[1], sampling_rate=fs, method="neurokit")
    try:
        _, rpeaks = nk.ecg_peaks(clean_tgt, sampling_rate=fs)
        _, waves = nk.ecg_delineate(clean_tgt, rpeaks, sampling_rate=fs, method="dwt")
        
        on = waves['ECG_R_Onsets']
        off = waves['ECG_R_Offsets'] # J-point
        t_on = waves['ECG_T_Onsets']
        
        qrs_c, st_c = [], []
        j_errs = []
        
        for i in range(len(rpeaks['ECG_R_Peaks'])):
            # QRS Corr
            if not np.isnan(on[i]) and not np.isnan(off[i]):
                t_seg = tgt[:, int(on[i]):int(off[i])]
                p_seg = pred[:, int(on[i]):int(off[i])]
                if t_seg.size > 10:
                    r, _ = pearsonr(t_seg.flatten(), p_seg.flatten())
                    if not np.isnan(r): qrs_c.append(r)
                
                # J-point Error (uV)
                jp = int(off[i])
                j_errs.append(np.mean(np.abs(pred[:, jp] - tgt[:, jp])) * 10000)
            
            # ST Corr
            s_end = t_on[i]
            if np.isnan(s_end): s_end = off[i] + int(0.12 * fs)
            
            if not np.isnan(off[i]) and not np.isnan(s_end):
                t_seg = tgt[:, int(off[i]):int(s_end)]
                p_seg = pred[:, int(off[i]):int(s_end)]
                if t_seg.size > 10:
                    r, _ = pearsonr(t_seg.flatten(), p_seg.flatten())
                    if not np.isnan(r): st_c.append(r)
        
        return np.mean(qrs_c), np.mean(st_c), np.mean(j_errs)
    except Exception:
        return np.nan, np.nan, np.nan

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sources = [{"name": "PTB-XL", "path": "data/ptb_xl/tensors", "format": "pt"}]
    dataset = MultiSourceECGDataset(split='test', sources=sources, target_len=5000, normalization='min_max', input_leads=["I"])
    print(f"Dataset Size: {len(dataset)}")
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
    
    # Production Bridge (Version C)
    bridge = ModularBridge(
        fa_mae_checkpoint='checkpoints/fa_mae_production/checkpoint_ep50.pt',
        device=str(device)
    )
    bridge.load_state_dict(torch.load('checkpoints/bridge_production/bridge_ep50.pt', map_location=device))
    bridge.eval()
    
    stats = {'bridge': []}
    
    print("Running Final End-to-End Production Evaluation (200 samples)...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader)):
            if i > 12: break # ~200 samples
            x = batch['input'].to(device)
            y = batch['target'].to(device)
            
            p_bridge = bridge(x) # Bridge takes Lead I
            
            for b in range(x.shape[0]):
                qb, sb, jb = compute_all_nk2_metrics(y[b].cpu().numpy(), p_bridge[b].cpu().numpy())
                if not np.isnan(qb): 
                    stats['bridge'].append({'qrs': qb, 'st': sb, 'j': jb})
                else:
                    print(f"Sample {i*16 + b} failed NK2 metrics (NaN returned)")
                    
    print(f"Final Count of Validated Samples: {len(stats['bridge'])}")
    
    dfb = pd.DataFrame(stats['bridge']).mean()
    
    report = {
        'Bridge-Production': dfb.to_dict(),
        'Metadata': {
            'n_samples': len(stats['bridge']),
            'checkpoint': 'checkpoints/bridge_production/bridge_ep50.pt'
        }
    }
    
    print("\n=== Final Standardized Results ===")
    print(json.dumps(report, indent=4))
    
    with open('results/final_production_report.json', 'w') as f:
        json.dump(report, f, indent=4)

if __name__ == "__main__":
    main()
