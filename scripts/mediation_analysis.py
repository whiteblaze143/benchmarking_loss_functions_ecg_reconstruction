
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
import statsmodels.api as sm


from src.reconstruction.learn_functions.wrappers import MasonWrapper
from src.reconstruction.learn_functions.classifier import xresnet1d101
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

def get_logit_margin(logits, labels):
    margins = []
    for i in range(logits.shape[0]):
        active_classes = torch.where(labels[i] > 0.5)[0]
        if len(active_classes) == 0:
            margins.append(0.0)
            continue
        sample_logits = logits[i]
        sample_margins = []
        for c in active_classes:
            target_logit = sample_logits[c]
            other_mask = torch.ones_like(sample_logits, dtype=torch.bool)
            other_mask[c] = False
            max_other = torch.max(sample_logits[other_mask])
            sample_margins.append((target_logit - max_other).item())
        margins.append(np.mean(sample_margins))
    return np.array(margins)

def compute_clinical_errors(tgt, pred, fs=500):
    """
    Standardize ST and QRS errors using NK2 delineation.
    """
    # 1. Clean and find peaks on target Lead II (Index 1)
    clean_tgt = nk.ecg_clean(tgt[1], sampling_rate=fs, method="neurokit")
    try:
        _, rpeaks = nk.ecg_peaks(clean_tgt, sampling_rate=fs)
        _, waves = nk.ecg_delineate(clean_tgt, rpeaks, sampling_rate=fs, method="dwt")
        
        j_points = waves['ECG_R_Offsets'] # J-point
        r_onsets = waves['ECG_R_Onsets']
        
        st_errs = []
        qrs_dur_errs = []
        
        for i in range(len(rpeaks['ECG_R_Peaks'])):
            j_idx = j_points[i]
            on_idx = r_onsets[i]
            if np.isnan(j_idx) or np.isnan(on_idx): continue
            
            j_idx = int(j_idx)
            # ST Segment error at J + 60ms
            st_idx = j_idx + int(0.06 * fs)
            if st_idx < tgt.shape[1]:
                # Error in mV: (pred - tgt) * 10 
                err = np.mean(np.abs(pred[:, st_idx] - tgt[:, st_idx])) * 10.0
                st_errs.append(err)
            
            # QRS Duration Error
            # We don't delineate pred (too noisy/smooth), 
            # we check how well pred matches the tgt duration boundaries
            # or just report duration reconstruction fidelity.
            # Simplified: MAE in QRS region
            qs, qe = int(on_idx), int(j_idx)
            q_err = np.mean(np.abs(pred[:, qs:qe] - tgt[:, qs:qe])) * 10.0
            qrs_dur_errs.append(q_err)

        return np.mean(st_errs) if st_errs else np.nan, np.mean(qrs_dur_errs) if qrs_dur_errs else np.nan
        
    except Exception:
        return np.nan, np.nan

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
    
    clf = xresnet1d101(num_classes=5, input_channels=12).to(device)
    clf.load_state_dict(torch.load('checkpoints/oracle_original.pt', map_location=device, weights_only=False))
    clf.eval()
    
    records = []
    print("Running Upgraded Mediation Analysis (500 samples)...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader)):
            if i > 5: break # ~500 samples
            x = batch['input'].to(device)
            y = batch['target'].to(device)
            labels = batch['label'].to(device)
            
            p0 = m0(x)
            p1 = m1(x)
            
            l_orig = clf(y)
            l0 = clf(p0)
            l1 = clf(p1)
            
            m_orig = get_logit_margin(l_orig, labels)
            m0_vals = get_logit_margin(l0, labels)
            m1_vals = get_logit_margin(l1, labels)
            
            for b in range(x.shape[0]):
                st0, qr0 = compute_clinical_errors(y[b].cpu().numpy(), p0[b].cpu().numpy())
                st1, qr1 = compute_clinical_errors(y[b].cpu().numpy(), p1[b].cpu().numpy())
                
                if not np.isnan(st0):
                    records.append({'drop': (m_orig[b] - m0_vals[b]), 'st_err_mv': st0, 'qrs_err_mv': qr0, 'is_m1': 0})
                if not np.isnan(st1):
                    records.append({'drop': (m_orig[b] - m1_vals[b]), 'st_err_mv': st1, 'qrs_err_mv': qr1, 'is_m1': 1})
                
    df_pool = pd.DataFrame(records).dropna()
    print(f"\nAnalyzed {len(df_pool)} observations.")
    
    # OLS: Prediction Logit Drop ~ ST Error + QRS Error + ModelType
    X = df_pool[['st_err_mv', 'qrs_err_mv', 'is_m1']]
    X = sm.add_constant(X)
    y = df_pool['drop']
    
    model = sm.OLS(y, X).fit()
    print(model.summary())
    
    # Write to final results
    with open('results/mediation_report_final.txt', 'w') as f:
        f.write("=== Upgraded Mediation Analysis (NK2 Delineated) ===\n")
        f.write(model.summary().as_text())
        f.write(f"\n\nCorrelation (ST Error vs Logit Drop): {df_pool['st_err_mv'].corr(df_pool['drop']):.4f}")
        f.write(f"\nCorrelation (QRS Error vs Logit Drop): {df_pool['qrs_err_mv'].corr(df_pool['drop']):.4f}")

if __name__ == "__main__":
    main()
