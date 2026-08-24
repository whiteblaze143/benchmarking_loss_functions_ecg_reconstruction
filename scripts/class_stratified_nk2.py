
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

def compute_fidelity_nk2(tgt, pred, fs=500):
    clean_tgt = nk.ecg_clean(tgt[1], sampling_rate=fs, method="neurokit")
    try:
        _, rpeaks = nk.ecg_peaks(clean_tgt, sampling_rate=fs)
        _, waves = nk.ecg_delineate(clean_tgt, rpeaks, sampling_rate=fs, method="dwt")
        
        onsets = waves['ECG_R_Onsets']
        offsets = waves['ECG_R_Offsets']
        
        qrs_corrs = []
        for i in range(len(rpeaks['ECG_R_Peaks'])):
            on, off = onsets[i], offsets[i]
            if np.isnan(on) or np.isnan(off): continue
            t_seg = tgt[:, int(on):int(off)]
            p_seg = pred[:, int(on):int(off)]
            if t_seg.size > 10:
                r, _ = pearsonr(t_seg.flatten(), p_seg.flatten())
                if not np.isnan(r): qrs_corrs.append(r)
        return np.mean(qrs_corrs)
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
    
    classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    results = {c: {'m0': [], 'm1': []} for c in classes}
    
    print("Running Class-Stratified NK2 Analysis (300 samples)...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader)):
            if i > 18: break
            x = batch['input'].to(device)
            y = batch['target'].to(device)
            labels = batch['label'] # (B, 5)
            
            p0 = m0(x)
            p1 = m1(x)
            
            for b in range(x.shape[0]):
                q0 = compute_fidelity_nk2(y[b].cpu().numpy(), p0[b].cpu().numpy())
                q1 = compute_fidelity_nk2(y[b].cpu().numpy(), p1[b].cpu().numpy())
                
                # Assign to major class (highest label value or active classes)
                active_classes = np.where(labels[b] > 0.5)[0]
                for c_idx in active_classes:
                    c_name = classes[c_idx]
                    if not np.isnan(q0): results[c_name]['m0'].append(q0)
                    if not np.isnan(q1): results[c_name]['m1'].append(q1)
                    
    final_rpt = []
    for c in classes:
        final_rpt.append({
            'Class': c,
            'M0': np.mean(results[c]['m0']),
            'M1-HPO': np.mean(results[c]['m1']),
            'Count': len(results[c]['m0'])
        })
    
    df = pd.DataFrame(final_rpt)
    print("\n=== Class-Stratified NK2 QRS Correlation ===")
    print(df)
    df.to_csv('results/class_stratified_nk2.csv', index=False)

if __name__ == "__main__":
    main()
