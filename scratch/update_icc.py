ˇimport sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.bootstrap_paths import setup_import_paths
setup_import_paths()

import logging
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
            
            p_peaks = []
            r_peaks = []
            t_peaks = []
            
            for i in range(1, len(symbols) - 1):
                sym = symbols[i]
                if sym in ['p', 'N', 't']:
                    peak = samples[i]
                    if sym == 'p':
                        p_peaks.append(peak)
                    elif sym == 'N':
                        r_peaks.append(peak)
                    elif sym == 't':
                        t_peaks.append(peak)
            gt_bounds[lead] = {
                'P_Peak': np.array(p_peaks),
                'R_Peak': np.array(r_peaks),
                'T_Peak': np.array(t_peaks),
            }
        except:
            gt_bounds[lead] = {b: np.array([]) for b in ['P_Peak', 'R_Peak', 'T_Peak']}
    return gt_bounds

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
            'P_Peak': np.array(waves.get("ECG_P_Peaks", [])),
            'R_Peak': np.array(rpeaks),
            'T_Peak': np.array(waves.get("ECG_T_Peaks", [])),
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

df = pd.DataFrame(results)

print("\n=== ICC Results (Peaks) ===")
out_md = []
out_md.append("| Dataset | Boundary | ICC1 | ICC2 | ICC3 | N Pairs |")
out_md.append("| :--- | :--- | :--- | :--- | :--- | :--- |")

for ds in ['ludb']:
    for b_type in ['P_Peak', 'R_Peak', 'T_Peak']:
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
        print(out_md[-1])
