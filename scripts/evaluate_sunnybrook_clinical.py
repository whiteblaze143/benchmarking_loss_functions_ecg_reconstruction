import os
import sys
import json
import logging
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import neurokit2 as nk
import sierraecg
import matplotlib.pyplot as plt
import concurrent.futures

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

MASON_MIN = -2.5
MASON_AMP = 5.0

def normalize_mason(lead, min_value=None, amplitude=None):
    min_value = MASON_MIN if min_value is None else min_value
    amplitude = MASON_AMP if amplitude is None else amplitude
    norm_lead = (lead - min_value) / amplitude
    if isinstance(norm_lead, torch.Tensor):
        return torch.clamp(norm_lead, 0.0, 1.0)
    return np.clip(norm_lead, 0.0, 1.0)

def denormalize_mason(lead, min_value=None, amplitude=None):
    min_value = MASON_MIN if min_value is None else min_value
    amplitude = MASON_AMP if amplitude is None else amplitude
    return lead * amplitude + min_value

from scipy.signal import butter, filtfilt

def _resample_mason_style(sig_mv, target_len):
    new_x = (np.arange(target_len, dtype=np.float64) + 1) / target_len
    x = (np.arange(sig_mv.shape[1], dtype=np.float64) + 1) / sig_mv.shape[1]
    out = np.stack([np.interp(new_x, x, sig_mv[i]) for i in range(sig_mv.shape[0])], axis=0)
    return out.astype(np.float32)

def _bandpass_mv(sig_mv, low_hz, high_hz, fs, order=4):
    nyq = 0.5 * fs
    low = max(0.001, low_hz / nyq)
    high = min(0.99, high_hz / nyq)
    if low >= high:
        return sig_mv
    b, a = butter(order, [low, high], btype="band")
    out = np.zeros_like(sig_mv, dtype=np.float64)
    for i in range(sig_mv.shape[0]):
        out[i] = filtfilt(b, a, sig_mv[i].astype(np.float64))
    return out.astype(np.float32)

def load_sunnybrook_record(xml_path, target_len=5000, resample_to=None, bandpass_hz=None, fs=None):
    try:
        f = sierraecg.read_file(str(xml_path))
        signal_map = {lead.label: lead.samples for lead in f.leads}
        
        LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        sig = np.stack([signal_map[l] for l in LEAD_ORDER]).astype(np.float64)
        sig_mv = sig / 1000.0
        
        if resample_to is not None:
            sig_mv = _resample_mason_style(sig_mv, resample_to)
            curr_len = resample_to
        else:
            curr_len = sig_mv.shape[1]
            if curr_len < target_len:
                sig_mv = np.concatenate([sig_mv, np.zeros((12, target_len - curr_len))], axis=1)
            elif curr_len > target_len:
                sig_mv = sig_mv[:, :target_len]
                
        if bandpass_hz is not None:
            low_hz, high_hz = bandpass_hz[0], bandpass_hz[1]
            if fs is None:
                fs = 250 if (curr_len == 2500) else 500
            sig_mv = _bandpass_mv(sig_mv, low_hz, high_hz, fs)
            
        sig_norm = normalize_mason(sig_mv)
        sig_norm_tensor = torch.tensor(sig_norm, dtype=torch.float32)
        
        return sig_norm_tensor, sig_mv
    except Exception as e:
        print(f"ERROR loading {xml_path}: {e}")
        return None, None

from scripts.evaluate_comprehensive_registry import load_adapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

LEADS = {
    "I": 0, "II": 1, "III": 2, "aVR": 3, "aVL": 4, "aVF": 5,
    "V1": 6, "V2": 7, "V3": 8, "V4": 9, "V5": 10, "V6": 11
}

def parse_xli_features(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = {"p": "http://www3.medical.philips.com"}
    features = {}
    
    # Global metrics
    for glob in root.findall(".//p:groupmeasurement", ns):
        for child in glob:
            tag = child.tag.split("}")[-1]
            if child.text and child.text.strip():
                try:
                    features[f"global_{tag}"] = float(child.text)
                except ValueError:
                    pass
                
    # Lead-specific metrics
    for lead in root.findall(".//p:leadmeasurement", ns):
        leadname = lead.attrib.get("leadname")
        if not leadname: continue
        for child in lead:
            tag = child.tag.split("}")[-1]
            if child.text and child.text.strip():
                try:
                    features[f"{leadname}_{tag}"] = float(child.text)
                except ValueError:
                    pass
    return features

def extract_nk2_features(sig_mv):
    """Extract clinical features for all leads using neurokit2"""
    features = {}
    
    for lead_name, lead_idx in LEADS.items():
        try:
            signals, info = nk.ecg_process(sig_mv[lead_idx], sampling_rate=500)
            
            if "ECG_R_Peaks" in signals:
                r_peaks = np.where(signals["ECG_R_Peaks"] == 1)[0]
                if len(r_peaks) > 0:
                    features[f"{lead_name}_ramp"] = np.mean(sig_mv[lead_idx, r_peaks]) * 1000.0
                    
            if "ECG_T_Peaks" in signals:
                t_peaks = np.where(signals["ECG_T_Peaks"] == 1)[0]
                if len(t_peaks) > 0:
                    features[f"{lead_name}_tamp"] = np.mean(sig_mv[lead_idx, t_peaks]) * 1000.0
                    
            if "ECG_S_Peaks" in signals:
                s_peaks = np.where(signals["ECG_S_Peaks"] == 1)[0]
                if len(s_peaks) > 0:
                    features[f"{lead_name}_samp"] = np.mean(sig_mv[lead_idx, s_peaks]) * 1000.0
                    
            if "ECG_R_Offsets" in signals:
                j_points = np.where(signals["ECG_R_Offsets"] == 1)[0]
                if len(j_points) > 0:
                    st80_idx = np.clip(j_points + 40, 0, sig_mv.shape[1] - 1)
                    features[f"{lead_name}_st80"] = np.mean(sig_mv[lead_idx, st80_idx]) * 1000.0
        except Exception as e:
            pass
            
    # Global features (derived from lead II)
    try:
        signals, info = nk.ecg_process(sig_mv[1], sampling_rate=500)
        
        if "ECG_R_Onsets" in signals and "ECG_R_Offsets" in signals:
            onsets = np.where(signals["ECG_R_Onsets"] == 1)[0]
            offsets = np.where(signals["ECG_R_Offsets"] == 1)[0]
            if len(onsets) > 0 and len(offsets) > 0:
                min_len = min(len(onsets), len(offsets))
                durations = (offsets[:min_len] - onsets[:min_len]) / 500.0 * 1000.0
                features["global_meanqrsdur"] = np.mean(durations)
                
        if "ECG_R_Onsets" in signals and "ECG_T_Offsets" in signals:
            onsets = np.where(signals["ECG_R_Onsets"] == 1)[0]
            t_offsets = np.where(signals["ECG_T_Offsets"] == 1)[0]
            if len(onsets) > 0 and len(t_offsets) > 0:
                min_len = min(len(onsets), len(t_offsets))
                durations = (t_offsets[:min_len] - onsets[:min_len]) / 500.0 * 1000.0
                features["global_meanqtint"] = np.mean(durations)
    except Exception as e:
        pass
        
    return features


def plot_bland_altman(diffs, means, xlabel, ylabel, title, out_path):
    plt.figure(figsize=(8, 6))
    plt.scatter(means, diffs, alpha=0.5, color='b')
    
    md = np.nanmean(diffs)
    sd = np.nanstd(diffs)
    
    plt.axhline(md, color='gray', linestyle='--')
    plt.axhline(md + 1.96*sd, color='gray', linestyle=':')
    plt.axhline(md - 1.96*sd, color='gray', linestyle=':')
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def process_record(xml_path):
    xli_feats = parse_xli_features(xml_path)
    sig_norm, sig_mv = load_sunnybrook_record(xml_path, target_len=5000, fs=500)
    
    if sig_norm is None:
        return None
        
    real_open_feats = extract_nk2_features(sig_mv)
    
    return {
        "path": xml_path.name,
        "sig_norm": sig_norm,
        "xli": xli_feats,
        "real_open": real_open_feats
    }

def process_recon(item):
    if item is None:
        return None
    rec, recon_mv, model_id = item
    recon_open_feats = extract_nk2_features(recon_mv)
    
    row = {"model": model_id, "file": rec["path"]}
    
    for k in ["V3_ramp", "V3_tamp", "V3_samp", "V3_st80", "V6_ramp", "V6_tamp", "V6_samp", "V6_st80", "global_meanqrsdur", "global_meanqtint"]:
        row[f"ep1_real_open_{k}"] = rec["real_open"].get(k, np.nan)
        row[f"ep1_recon_open_{k}"] = recon_open_feats.get(k, np.nan)
        row[f"ep1_err_{k}"] = row[f"ep1_recon_open_{k}"] - row[f"ep1_real_open_{k}"]
        
        row[f"ep2_xli_{k}"] = rec["xli"].get(k, np.nan)
        row[f"ep2_err_{k}"] = rec["real_open"].get(k, np.nan) - row[f"ep2_xli_{k}"]
        
        row[f"ep3_err_{k}"] = recon_open_feats.get(k, np.nan) - row[f"ep2_xli_{k}"]
    return row

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data_dir = project_root / "data/sunnybrook_12_lead_ecg_samples"
    xml_files = sorted(list(data_dir.glob("ECG*.xml")))
    logging.info(f"Found {len(xml_files)} XML files.")
    
    results_json = project_root / "results/factorial_v4/clean_results.json"
    with open(results_json) as f:
        registry = json.load(f)
        
    model_ids = list(registry["models"].keys())
    
    out_dir = project_root / "results/sunnybrook_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Pre-load all records and extract Real Open and XLI features
    records = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_record, path): path for path in xml_files}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res is not None:
                records.append(res)
        
    logging.info(f"Successfully processed {len(records)} records.")
    
    all_metrics = []
    
    # 2. Evaluate all models
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
            
        model_metrics = []
        
        recon_mvs = []
        with torch.inference_mode():
            for rec in records:
                target_tensor = rec["sig_norm"].unsqueeze(0).to(device)
                try:
                    recon_tensor = adapter.reconstruct(target_tensor)
                except Exception as e:
                    logging.error(f"Inference error on {model_id}: {e}")
                    recon_mvs.append(None)
                    continue
                    
                recon_norm = recon_tensor[0].cpu().numpy()
                recon_mv = denormalize_mason(recon_norm)
                recon_mvs.append((rec, recon_mv, model_id))
                
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for row in executor.map(process_recon, recon_mvs):
                if row is not None:
                    model_metrics.append(row)
                
        df = pd.DataFrame(model_metrics)
        df.to_csv(out_dir / f"{model_id}_sunnybrook_clinical.csv", index=False)
        all_metrics.extend(model_metrics)
        
        plot_bland_altman(
            diffs=df["ep1_err_V3_ramp"],
            means=(df["ep1_real_open_V3_ramp"] + df["ep1_recon_open_V3_ramp"])/2,
            xlabel="Mean V3 R-Amplitude (uV)",
            ylabel="Recon - Real (uV)",
            title=f"EP1: {model_id} V3 R-Amp",
            out_path=out_dir / f"{model_id}_EP1_V3_ramp.png"
        )
        
    full_df = pd.DataFrame(all_metrics)
    full_df.to_csv(out_dir / "all_models_sunnybrook_clinical.csv", index=False)
    logging.info(f"Done. Saved to {out_dir}")

if __name__ == "__main__":
    main()
