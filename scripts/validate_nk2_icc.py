import logging
from pathlib import Path
import numpy as np
import pandas as pd
import neurokit2 as nk
import wfdb
import ast
import json
import pickle
from tqdm import tqdm
import warnings
import pingouin as pg
from scipy.signal import resample

warnings.filterwarnings("ignore")

project_root = Path(__file__).resolve().parents[1]
LEADS = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
LEAD_NAME_MAP = {
    'i': 'I', 'ii': 'II', 'iii': 'III', 'avr': 'aVR', 'avl': 'aVL', 'avf': 'aVF',
    'v1': 'V1', 'v2': 'V2', 'v3': 'V3', 'v4': 'V4', 'v5': 'V5', 'v6': 'V6'
}

def get_ludb_gt(record_path):
    gt_bounds = {}
    for lead in LEADS:
        try:
            ann = wfdb.rdann(str(record_path), extension=lead)
            symbols = np.array(ann.symbol)
            samples = np.array(ann.sample)
            
            p_onsets, p_offsets = [], []
            qrs_onsets, qrs_offsets = [], []
            t_onsets, t_offsets = [], []
            
            for i in range(1, len(symbols) - 1):
                sym = symbols[i]
                if sym in ['p', 'N', 't']:
                    onset_cand = samples[i-1] if symbols[i-1] == '(' else None
                    offset_cand = samples[i+1] if symbols[i+1] == ')' else None
                    if onset_cand is not None and offset_cand is not None:
                        if sym == 'p':
                            p_onsets.append(onset_cand)
                            p_offsets.append(offset_cand)
                        elif sym == 'N':
                            qrs_onsets.append(onset_cand)
                            qrs_offsets.append(offset_cand)
                        elif sym == 't':
                            t_onsets.append(onset_cand)
                            t_offsets.append(offset_cand)
            gt_bounds[lead] = {
                'P_Onset': np.array(p_onsets), 'P_Offset': np.array(p_offsets),
                'R_Onset': np.array(qrs_onsets), 'R_Offset': np.array(qrs_offsets),
                'T_Onset': np.array(t_onsets), 'T_Offset': np.array(t_offsets),
            }
        except:
            gt_bounds[lead] = {b: np.array([]) for b in ['P_Onset', 'P_Offset', 'R_Onset', 'R_Offset', 'T_Onset', 'T_Offset']}
    return gt_bounds

def get_isp_gt(tuples_list):
    gt_bounds = {b: [] for b in ['P_Onset', 'P_Offset', 'R_Onset', 'R_Offset', 'T_Onset', 'T_Offset']}
    for t in tuples_list:
        start_idx = int(t[0])
        end_idx = int(t[1])
        label = t[2]
        if label == 'p':
            gt_bounds['P_Onset'].append(start_idx)
            gt_bounds['P_Offset'].append(end_idx)
        elif label == 'qrs':
            gt_bounds['R_Onset'].append(start_idx)
            gt_bounds['R_Offset'].append(end_idx)
        elif label == 't':
            gt_bounds['T_Onset'].append(start_idx)
            gt_bounds['T_Offset'].append(end_idx)
    return {k: np.array(v) for k, v in gt_bounds.items()}

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

def match_pairs(pred_bounds, gt_bounds, tolerance):
    pairs = []
    if len(gt_bounds) == 0 or len(pred_bounds) == 0:
        return pairs
    pred_bounds = pred_bounds[~np.isnan(pred_bounds)]
    
    for gt in gt_bounds:
        dist = np.abs(pred_bounds - gt)
        if len(dist) > 0 and np.min(dist) <= tolerance:
            idx = np.argmin(dist)
            pairs.append((gt, pred_bounds[idx]))
    return pairs

def process_lead(lead_signal, fs):
    try:
        signals, info = nk.ecg_process(lead_signal, sampling_rate=fs)
        cleaned = signals["ECG_Clean"].values
        rpeaks = info["ECG_R_Peaks"]
        if len(rpeaks) < 2: return None
        _, waves = nk.ecg_delineate(cleaned, rpeaks, sampling_rate=fs, method="dwt")
        return {
            'P_Onset': np.array(waves["ECG_P_Onsets"]), 'P_Offset': np.array(waves["ECG_P_Offsets"]),
            'R_Onset': np.array(waves["ECG_R_Onsets"]), 'R_Offset': np.array(waves["ECG_R_Offsets"]),
            'T_Onset': np.array(waves["ECG_T_Onsets"]), 'T_Offset': np.array(waves["ECG_T_Offsets"]),
        }
    except:
        return None

results = []

print("Processing LUDB (500Hz, max 50 records)...")
ludb_dir = project_root / "data/ludb"
ludb_records = sorted(list(set([p.stem for p in ludb_dir.glob("*.dat")])))[:50]
for rid in tqdm(ludb_records):
    rec_path = ludb_dir / rid
    try:
        record = wfdb.rdrecord(str(rec_path))
        gt_bounds = get_ludb_gt(rec_path)
        sig = record.p_signal
        channels = record.sig_name
        for i, ld in enumerate(LEADS):
            if ld in channels:
                idx = channels.index(ld)
            elif ld.upper() in channels:
                idx = channels.index(ld.upper())
            else: continue
            
            nk_bounds = process_lead(sig[:, idx], 500)
            if nk_bounds:
                for b_type in nk_bounds.keys():
                    pairs = match_pairs(nk_bounds[b_type], gt_bounds[ld][b_type], 75)
                    for pt in pairs:
                        results.append({'dataset': 'ludb', 'boundary': b_type, 'true': pt[0], 'pred': pt[1]})
    except Exception as e:
        pass

print("Processing ISP (1000Hz, max 50 records)...")
isp_dir = project_root / "data/isp/isp_delineation_dataset"
try:
    df_isp = pd.read_csv(isp_dir / "train_isp_delineation_data.csv").head(50)
    for _, row in tqdm(df_isp.iterrows(), total=len(df_isp)):
        rid = str(row['file_name'])
        rec_path = isp_dir / "train_data" / rid
        try:
            record = wfdb.rdrecord(str(rec_path))
            tuples_list = ast.literal_eval(row['target'])
            gt_bounds = get_isp_gt(tuples_list)
            sig = record.p_signal
            channels = record.sig_name
            # For ISP, GT is same for all leads, we can just process lead II as representative
            ld = 'ii'
            idx = channels.index(ld) if ld in channels else channels.index(ld.upper())
            nk_bounds = process_lead(sig[:, idx], 1000)
            if nk_bounds:
                for b_type in nk_bounds.keys():
                    pairs = match_pairs(nk_bounds[b_type], gt_bounds[b_type], 150)
                    for pt in pairs:
                        results.append({'dataset': 'isp', 'boundary': b_type, 'true': pt[0], 'pred': pt[1]})
        except Exception as e:
            pass
except:
    pass

print("Processing ZHEJIANG (2000Hz, max 50 records)...")
zhe_ecg_dir = project_root / "data/zhejiang/ecg"
zhe_label_dir = project_root / "data/zhejiang/label"
zhe_records = sorted([p.stem for p in zhe_label_dir.glob("*.pkl")])[:50]
for rid in tqdm(zhe_records):
    try:
        with open(zhe_label_dir / f"{rid}.pkl", 'rb') as f:
            gt_mask_20k = pickle.load(f)
            
        gt_bounds = {
            'P_Onset': extract_bounds_from_mask(gt_mask_20k, 1)[0],
            'P_Offset': extract_bounds_from_mask(gt_mask_20k, 1)[1],
            'R_Onset': extract_bounds_from_mask(gt_mask_20k, 2)[0],
            'R_Offset': extract_bounds_from_mask(gt_mask_20k, 2)[1],
            'T_Onset': extract_bounds_from_mask(gt_mask_20k, 3)[0],
            'T_Offset': extract_bounds_from_mask(gt_mask_20k, 3)[1],
        }
        
        # Load lead II
        lead_suffix = LEAD_NAME_MAP['ii']
        with open(zhe_ecg_dir / f"{rid}_{lead_suffix}.pkl", 'rb') as f:
            sig = pickle.load(f)
            
        nk_bounds = process_lead(sig, 2000)
        if nk_bounds:
            for b_type in nk_bounds.keys():
                pairs = match_pairs(nk_bounds[b_type], gt_bounds[b_type], 300)
                for pt in pairs:
                    results.append({'dataset': 'zhejiang', 'boundary': b_type, 'true': pt[0], 'pred': pt[1]})
    except Exception as e:
        pass

df = pd.DataFrame(results)

print("\n=== ICC Results ===")
out_md = []
out_md.append("| Dataset | Boundary | ICC1 | ICC2 | ICC3 | N Pairs |")
out_md.append("| :--- | :--- | :--- | :--- | :--- | :--- |")

for ds in ['ludb', 'isp', 'zhejiang']:
    for b_type in ['P_Onset', 'P_Offset', 'R_Onset', 'R_Offset', 'T_Onset', 'T_Offset']:
        sub = df[(df['dataset'] == ds) & (df['boundary'] == b_type)]
        if len(sub) < 10:
            continue
            
        icc_df = pd.DataFrame({
            'Target': np.repeat(np.arange(len(sub)), 2),
            'Rater': np.tile(['Physician', 'NK2'], len(sub)),
            'Score': np.column_stack((sub['true'].values, sub['pred'].values)).flatten()
        })
        
        icc = pg.intraclass_corr(data=icc_df, targets='Target', raters='Rater', ratings='Score')
        
        icc1 = icc[icc['Type'] == 'ICC(1,1)'].iloc[0]['ICC'] if 'ICC(1,1)' in icc['Type'].values else float('nan')
        icc2 = icc[icc['Type'] == 'ICC(A,1)'].iloc[0]['ICC'] if 'ICC(A,1)' in icc['Type'].values else float('nan')
        icc3 = icc[icc['Type'] == 'ICC(C,1)'].iloc[0]['ICC'] if 'ICC(C,1)' in icc['Type'].values else float('nan')
        
        out_md.append(f"| {ds.upper()} | {b_type} | {icc1:.3f} | {icc2:.3f} | {icc3:.3f} | {len(sub)} |")

with open(project_root / 'nk2_validation_icc.md', 'w') as f:
    f.write("# NeuroKit2 Validation vs Physician Annotations (ICC)\n\n")
    f.write("\n".join(out_md))

print("Done. Saved to nk2_validation_icc.md")
