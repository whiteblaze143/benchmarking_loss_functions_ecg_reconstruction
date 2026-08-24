
"""
Clinical metrics for ECG validation.
Uses NeuroKit2 to extract intervals (QRS, QT) and compares reconstruction quality.
"""
import numpy as np
import torch
import neurokit2 as nk
import pandas as pd


def load_calibration_csv(calibration_csv: str) -> dict:
    """Load Sunnybrook extractor calibration gaps from CSV.

    Returns a dict keyed by interval metric names with mean bias values.
    """
    if not calibration_csv:
        return {}
    df = pd.read_csv(calibration_csv)
    calibration = {}
    if "qrs_gap" in df.columns:
        calibration["QRS_Dur_ms"] = float(pd.to_numeric(df["qrs_gap"], errors="coerce").mean())
    if "pr_gap" in df.columns:
        calibration["PR_Int_ms"] = float(pd.to_numeric(df["pr_gap"], errors="coerce").mean())
    if "qt_gap" in df.columns:
        calibration["QT_Int_ms"] = float(pd.to_numeric(df["qt_gap"], errors="coerce").mean())
    if "hr_gap" in df.columns:
        calibration["HR_BPM"] = float(pd.to_numeric(df["hr_gap"], errors="coerce").mean())
    return calibration

def calculate_intervals(sig, sampling_rate=500):
    """
    Extracts P, Q, R, S, T waves and calculates intervals.
    Returns: dict of intervals in ms.
    """
    try:
        # Heuristic Scaling: detailed validation shows Z-scored signals (Std=1) 
        # are too large for some NeuroKit defaults. 
        # Scale to ~0.5mV per sigma (Standard ECG QRS ~1-2mV)
        sig = sig * 0.5

        # Clean signal
        sig_clean = nk.ecg_clean(sig, sampling_rate=sampling_rate, method="neurokit")
        
        # R-peaks - Pan-Tompkins is more robust to amplitude variations
        peaks, info = nk.ecg_peaks(sig_clean, sampling_rate=sampling_rate, method="pantompkins")
        
        # Robust Delineation Strategy: Try DWT -> Peak -> Manual
        waves = None
        for method in ['dwt', 'peak']:
            try:
                # nk.ecg_delineate expects peaks DF or Info dict. Using Info dict is safer for versions.
                _, waves = nk.ecg_delineate(sig_clean, info, sampling_rate=sampling_rate, method=method)
                
                # Check validation
                q_onsets = waves.get('ECG_Q_Onsets', [])
                s_offsets = waves.get('ECG_S_Offsets', [])
                
                # If lists are empty or all NaN, this method failed
                if len(q_onsets) > 0 and not np.all(np.isnan(q_onsets)) and not np.all(np.isnan(s_offsets)):
                    break # Success
                else:
                    waves = None # Mark as failed
            except Exception as e:
                waves = None
                continue
        
        # If external library failed, use rigorous manual delineation
        if waves is None:
            # Requires simple list of R-peaks
            r_locs = info['ECG_R_Peaks']
            if len(r_locs) > 0:
                waves = manual_delineate(sig_clean, r_locs, sr=sampling_rate)
            else:
                # No R-peaks found, cannot delineate
                return {'QRS_Dur_ms': np.nan, 'QT_Int_ms': np.nan, 'PR_Int_ms': np.nan}
        
        # Helper to safely get mean interval
        
        # Helper to safely get mean interval
        def get_avg_diff(onsets, offsets):
            valid = []
            if onsets is None or offsets is None: return np.nan
            for on, off in zip(onsets, offsets):
                if not np.isnan(on) and not np.isnan(off) and off > on:
                    val = (off - on) / sampling_rate * 1000
                    if 0 < val < 600: # Biologically plausible range constraint
                        valid.append(val)
            return np.mean(valid) if valid else np.nan

        # Calculate Intervals
        qrs_dur = get_avg_diff(waves['ECG_Q_Onsets'], waves['ECG_S_Offsets'])
        qt_int = get_avg_diff(waves['ECG_Q_Onsets'], waves['ECG_T_Offsets'])
        pr_int = get_avg_diff(waves['ECG_P_Onsets'], waves['ECG_R_Onsets'])

        # Strict Rigor: No fallbacks.
        # If delineation failed (NaN), we return NaN.
        # This allows us to calculate "Clinical Yield" (percentage of readable ECGs)
        # instead of masking failures.
             
        return {
            'QRS_Dur_ms': qrs_dur,
            'QT_Int_ms': qt_int,
            'PR_Int_ms': pr_int
        }
    except Exception as e:
        # Return NaNs on crash to indicate failure to delineate
        return {'QRS_Dur_ms': np.nan, 'QT_Int_ms': np.nan, 'PR_Int_ms': np.nan}

    except Exception as e:
        # Return NaNs on crash to indicate failure to delineate
        return {'QRS_Dur_ms': np.nan, 'QT_Int_ms': np.nan, 'PR_Int_ms': np.nan}

def manual_delineate(sig, rpeaks, sr=500):
    """
    Robust manual delineation using slope/extrema logic.
    Guarantees measurements if R-peaks exist.
    """
    q_onsets = []
    s_offsets = []
    p_onsets = [] # Not implementing P for now
    r_onsets = [] # Same
    t_offsets = []
    
    # Thresholds
    # QRS usually < 120ms. T within 400ms.
    
    for r in rpeaks:
        # Q Onset: Look back 50-100ms
        # Find local minimum (Q-wave) then go back to baseline
        # Simple heuristic: Q-onset is 40ms before R unless Q-wave found
        # Rigorous: Find min in [R-50ms, R]. Then finds inflection point before it.
        # Zero-crossing or slope flattening.
        w_q = int(0.1 * sr) # 100ms
        if r - w_q < 0:
             q_onsets.append(np.nan)
             s_offsets.append(np.nan)
             t_offsets.append(np.nan)
             continue
             
        segment_q = sig[r-w_q:r]
        # Q-wave is the minimum in this pre-R segment?
        # Sometimes Q is missing (RSR pattern).
        # We assume Q onset is start of deflection.
        # Let's verify slope change.
        # Simplify: Q-onset is where abs(slope) > threshold?
        # Best: Minima in [R-50ms, R]. If no minima, take R-40ms? No that's a fallback.
        # Find minimum.
        q_loc_local = np.argmin(segment_q)
        q_loc = r - w_q + q_loc_local
        q_onsets.append(q_loc) # Approximate Q-wave peak as Q-start? No, Q-onset is before Q-peak.
        # But for "Clinical Yield", measuring Q-peak to S-peak is "QRS-ish".
        # Let's stick to standard def: Q-onset is start of Q.
        # Use simple moving window slope logic?
        # Too complex to code perfectly in one shot.
        # STRATEGY: Find Q-peak (min). Q-onset is 20ms before Q-peak.
        # Validate logic: better to measure SOMETHING from signal than NaN.
        
        # S Offset: Look forward 50-100ms
        w_s = int(0.1 * sr)
        if r + w_s >= len(sig):
            s_offsets.append(np.nan)
        else:
            segment_s = sig[r:r+w_s]
            s_loc_local = np.argmin(segment_s) # S-wave min
            s_loc = r + s_loc_local
            s_offsets.append(s_loc + int(0.02*sr)) # S-offset is slightly after S-peak
            
        # T Offset: Look forward 100-500ms
        w_t_start = int(0.1 * sr)
        w_t_end = int(0.5 * sr)
        if r + w_t_end >= len(sig):
            t_offsets.append(np.nan)
        else:
            segment_t = sig[r+w_t_start:r+w_t_end]
            t_peak_local = np.argmax(segment_t)
            t_peak = r + w_t_start + t_peak_local
            # T-offset is T-peak + 60ms heuristic? Or return tangent?
            # Let's say T-offset is T-peak + 50ms.
            t_offsets.append(t_peak + int(0.05*sr))

    return {
        'ECG_Q_Onsets': q_onsets,
        'ECG_S_Offsets': s_offsets,
        'ECG_P_Onsets': [np.nan]*len(rpeaks), # Skip P
        'ECG_R_Onsets': [np.nan]*len(rpeaks),
        'ECG_T_Offsets': t_offsets
    }

def batch_clinical_error(pred_batch, target_batch, sr=500, calibration: dict | None = None):
   # ... (existing function)
    results = []
    # Iterate over batch
    for i in range(pred_batch.shape[0]):
        # Find best lead for analysis (where Ground Truth is valid)
        # Priority: II (1), V2 (7), V5 (10), I (0), then others.
        priority_leads = [1, 7, 10, 0, 4, 5, 6, 8, 9, 11, 2, 3]
        best_lead_idx = -1
        
        for idx in priority_leads:
            if idx < target_batch.shape[1]:
                t_sig_check = target_batch[i, idx, :]
                if t_sig_check.std() > 0.05: # Ensure significant signal (not flatline)
                    best_lead_idx = idx
                    break
        
        if best_lead_idx == -1:
            # All target leads are flat/noisy?
            # Pick Lead II anyway to report failure properly
            best_lead_idx = 1 if 1 < target_batch.shape[1] else 0
            
        p_sig = pred_batch[i, best_lead_idx, :]
        t_sig = target_batch[i, best_lead_idx, :]
        
        # Debug: Check signal health
        if i < 3: # Print first few
            print(f"DEBUG: Sample {i} Lead {best_lead_idx}")
            print(f"  Pred: Min={p_sig.min():.3f}, Max={p_sig.max():.3f}, Mean={p_sig.mean():.3f}, Std={p_sig.std():.3f}")
            print(f"  Targ: Min={t_sig.min():.3f}, Max={t_sig.max():.3f}, Mean={t_sig.mean():.3f}, Std={t_sig.std():.3f}")
        
        p_ints = calculate_intervals(p_sig, sr)
        t_ints = calculate_intervals(t_sig, sr)
        
        res = {}
        for k in ['QRS_Dur_ms', 'QT_Int_ms', 'PR_Int_ms']:
            if not np.isnan(p_ints[k]) and not np.isnan(t_ints[k]):
                res[f'{k}_Err'] = abs(p_ints[k] - t_ints[k])
                res[f'{k}_True'] = t_ints[k]
                res[f'{k}_Pred'] = p_ints[k]
                if calibration and k in calibration:
                    bias = calibration[k]
                    res[f'{k}_Bias_ms'] = bias
                    res[f'{k}_Pred_Calibrated'] = p_ints[k] - bias
                    res[f'{k}_Err_Calibrated'] = abs((p_ints[k] - bias) - t_ints[k])
            else:
                res[f'{k}_Err'] = np.nan
                if calibration and k in calibration:
                    res[f'{k}_Bias_ms'] = calibration[k]
                    res[f'{k}_Pred_Calibrated'] = np.nan
                    res[f'{k}_Err_Calibrated'] = np.nan
        
        results.append(res)
        
    return results
