
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


from src.reconstruction.learn_functions.wrappers import MasonWrapper
from src.data.multi_source_dataset import MultiSourceECGDataset

def load_mason_ckpt(wrapper, path, device):
    state = torch.load(path, map_location=device, weights_only=False)
    new_state = {}
    for k, v in state.items():
        if k.startswith('mason.'):
            new_state[k[len('mason.'):]] = v
        else:
            new_state[k] = v
    wrapper.model.load_state_dict(new_state, strict=False)

def compute_territory_nk2(tgt, pred, leads_indices, fs=500):
    clean_tgt = nk.ecg_clean(tgt[1], sampling_rate=fs, method="neurokit")
    try:
        _, rpeaks = nk.ecg_peaks(clean_tgt, sampling_rate=fs)
        _, waves = nk.ecg_delineate(clean_tgt, rpeaks, sampling_rate=fs, method="dwt")
        onsets = waves['ECG_R_Onsets']
        offsets = waves['ECG_R_Offsets']
        
        corrs = []
        for i in range(len(rpeaks['ECG_R_Peaks'])):
            on, off = onsets[i], offsets[i]
            if np.isnan(on) or np.isnan(off): continue
            # Correlate only selected leads in the QRS region
            t_seg = tgt[leads_indices, int(on):int(off)]
            p_seg = pred[leads_indices, int(on):int(off)]
            if t_seg.size > 5:
                r, _ = pearsonr(t_seg.flatten(), p_seg.flatten())
                if not np.isnan(r): corrs.append(r)
        return np.mean(corrs)
    except Exception:
        return np.nan

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sources = [{"name": "PTB-XL", "path": "data/ptbxl_tensors", "format": "pt"}]
    dataset = MultiSourceECGDataset(split='val', sources=sources, target_len=5000, normalization='min_max', input_leads=["I", "II", "V2"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
    
    m0 = MasonWrapper(device=device)
    load_mason_ckpt(m0, 'checkpoints/M0_seed42.pt', device)
    m0.eval()
    
    m1 = MasonWrapper(device=device)
    load_mason_ckpt(m1, 'checkpoints/M1_Final.pt', device)
    m1.eval()
    
    # Anatomical index map (0-11)
    # I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
    anterior = [6, 7, 8, 9] # V1-V4
    lateral = [0, 4, 10, 11] # I, aVL, V5-V6
    inferior = [1, 2, 5] # II, III, aVF
    
    res = {'m0': {'ant': [], 'lat': [], 'inf': []}, 'm1': {'ant': [], 'lat': [], 'inf': []}}
    
    print("Running Anatomical NK2 Analysis (200 samples)...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader)):
            if i > 3: break
            x = batch['input'].to(device)
            y = batch['target'].to(device)
            
            p0 = m0(x)
            p1 = m1(x)
            
            for b in range(x.shape[0]):
                y_np, p0_np, p1_np = y[b].cpu().numpy(), p0[b].cpu().numpy(), p1[b].cpu().numpy()
                
                # M0
                a0 = compute_territory_nk2(y_np, p0_np, anterior)
                l0 = compute_territory_nk2(y_np, p0_np, lateral)
                f0 = compute_territory_nk2(y_np, p0_np, inferior)
                
                # M1
                a1 = compute_territory_nk2(y_np, p1_np, anterior)
                l1 = compute_territory_nk2(y_np, p1_np, lateral)
                f1 = compute_territory_nk2(y_np, p1_np, inferior)
                
                if not np.isnan(a0):
                    res['m0']['ant'].append(a0); res['m0']['lat'].append(l0); res['m0']['inf'].append(f0)
                    res['m1']['ant'].append(a1); res['m1']['lat'].append(l1); res['m1']['inf'].append(f1)
                    
    print("\n=== Anatomical Territory NK2 Correlation ===")
    out = {}
    for mod in ['m0', 'm1']:
        out[mod] = {k: np.nanmean(v) for k, v in res[mod].items()}
    print(pd.DataFrame(out))

if __name__ == "__main__":
    main()
