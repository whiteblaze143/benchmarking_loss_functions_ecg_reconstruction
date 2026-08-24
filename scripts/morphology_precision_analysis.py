
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

def compute_precision_metrics(tgt, pred, fs=500):
    """
    Calculate amplitude errors at J, J+60, R-peak and lead-stratified correlation.
    """
    leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    
    # 1. Lead-stratified Pearson Correlation
    lead_corrs = []
    for l in range(12):
        if np.std(tgt[l]) > 1e-6:
            r, _ = pearsonr(tgt[l], pred[l])
            lead_corrs.append(r)
        else:
            lead_corrs.append(np.nan)
            
    # 2. Amplitude Errors at Fiducials (NK2)
    clean_tgt = nk.ecg_clean(tgt[1], sampling_rate=fs, method="neurokit") # Lead II reference
    try:
        _, rpeaks = nk.ecg_peaks(clean_tgt, sampling_rate=fs)
        _, waves = nk.ecg_delineate(clean_tgt, rpeaks, sampling_rate=fs, method="dwt")
        
        j_points = waves['ECG_R_Offsets']
        r_peaks = rpeaks['ECG_R_Peaks']
        
        j_errs = []
        j60_errs = []
        r_errs = []
        
        # Unit conversion: 1.0 Norm Unit = 10 mV = 10000 uV
        # Error in uV = Error_norm * 10000
        
        for i in range(len(r_peaks)):
            # R-peak
            rp = int(r_peaks[i])
            r_errs.append(np.mean(np.abs(pred[:, rp] - tgt[:, rp])) * 10000)
            
            # J-point
            jp = j_points[i]
            if not np.isnan(jp):
                jp = int(jp)
                j_errs.append(np.mean(np.abs(pred[:, jp] - tgt[:, jp])) * 10000)
                
                # J+60ms
                j60 = jp + int(0.06 * fs)
                if j60 < tgt.shape[1]:
                    j60_errs.append(np.mean(np.abs(pred[:, j60] - tgt[:, j60])) * 10000)
                    
        return lead_corrs, np.mean(j_errs), np.mean(j60_errs), np.mean(r_errs)
    except Exception:
        return lead_corrs, np.nan, np.nan, np.nan

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
    
    all_corrs0, all_corrs1 = [], []
    fiducials0, fiducials1 = [], []
    
    print("Running Morphology Precision Analysis (500 samples)...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader)):
            if i > 5: break
            x = batch['input'].to(device)
            y = batch['target'].to(device)
            
            p0 = m0(x)
            p1 = m1(x)
            
            for b in range(x.shape[0]):
                c0, j0, j60_0, r0 = compute_precision_metrics(y[b].cpu().numpy(), p0[b].cpu().numpy())
                c1, j1, j60_1, r1 = compute_precision_metrics(y[b].cpu().numpy(), p1[b].cpu().numpy())
                
                all_corrs0.append(c0)
                all_corrs1.append(c1)
                fiducials0.append([j0, j60_0, r0])
                fiducials1.append([j1, j60_1, r1])
                
    # 1. Lead-stratified report
    leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    df_c0 = pd.DataFrame(all_corrs0, columns=leads).mean()
    df_c1 = pd.DataFrame(all_corrs1, columns=leads).mean()
    
    print("\n=== Lead-Stratified Pearson Correlation ===")
    lead_rpt = pd.DataFrame({"M0": df_c0, "M1-HPO": df_c1})
    print(lead_rpt)
    
    # 2. Fiducial report
    df_f0 = pd.DataFrame(fiducials0, columns=["J-point", "J+60ms", "R-peak"]).mean()
    df_f1 = pd.DataFrame(fiducials1, columns=["J-point", "J+60ms", "R-peak"]).mean()
    
    print("\n=== Amplitude Error at Fiducials (uV) ===")
    fid_rpt = pd.DataFrame({"M0 (uV)": df_f0, "M1-HPO (uV)": df_f1})
    fid_rpt["Reduction (%)"] = (1 - fid_rpt["M1-HPO (uV)"] / fid_rpt["M0 (uV)"]) * 100
    print(fid_rpt)
    
    # Save results
    lead_rpt.to_csv('results/lead_stratified_correlation.csv')
    fid_rpt.to_csv('results/fiducial_amplitude_errors_uv.csv')
    
    with open('results/morphology_precision_summary.txt', 'w') as f:
        f.write("=== Morphology Precision Report ===\n\n")
        f.write("Lead-Stratified Correlation:\n")
        f.write(lead_rpt.to_string())
        f.write("\n\nFiducial Amplitude Errors (uV):\n")
        f.write(fid_rpt.to_string())

if __name__ == "__main__":
    main()
