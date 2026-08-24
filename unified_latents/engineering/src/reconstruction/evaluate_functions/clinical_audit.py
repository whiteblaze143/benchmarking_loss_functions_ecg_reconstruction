#!/usr/bin/env python3
"""
Sunnybrook Clinical Evaluation (PRD Section 5)
===============================================
Mason-faithful zero-shot evaluation on Sunnybrook clinical data.

Implements:
- 5.1  Feature-Extractor Calibration (Primary CRR)
- 5.2  Morphology Delta (QRS, QTc, PR, Axis, HR, ST80)
- 5.3  MISLDS Stress Test (ECG010)
- 5.4  Physics-Consistency Gates (Einthoven, Augmented Identity)
- 5.4.1 Waveform Fidelity (R2, MSE, Pearson)

Data protocol: Mason-faithful
- Sunnybrook XML -> mV (divide by 1000) -> normalize_mason [0,1]
- Model input: [0,1] normalized (I, II, V3)
- Target: denormalized to mV for loss/metrics
"""

import os
import sys
sys.path.insert(0, '/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/unified_latents/engineering')
import argparse
import json
import numpy as np
import pandas as pd
import torch
import warnings
from pathlib import Path
from tqdm import tqdm

# Project root
sys.path.append(os.getcwd())

# -- src/reconstruction imports --
from src.reconstruction.learn_functions.fam_ecg import UniversalSpatialFusionAdapter
from src.reconstruction.learn_functions.losses import MasonR2Loss
from src.reconstruction.util_functions.mason_12lead import (
    MASON_INPUT_LIMB_V3,
    normalize_mason,
    denormalize_mason,
    mason_batch_r2_loss,
    get_twelve_keys,
)
from src.reconstruction.learn_functions.reconstruction_functions import (
    calculate_lead_r2,
    calculate_pearson,
)

# Lazy import get_model to avoid pulling in all backbone dependencies at module level
def _get_model(*args, **kwargs):
    from src.reconstruction.training_functions.train_universal_bridge import get_model
    return get_model(*args, **kwargs)

# Third-party
import sierraecg
from scipy.signal import butter, filtfilt

# Mason 12-lead order (mason_12lead.get_twelve_keys): index 3 = aVL, 4 = aVR
LEAD_ORDER = get_twelve_keys()

# PRD 5.4: Hardware noise floor thresholds (uV)
EINTHOVEN_NOISE_FLOOR_UV = 389
AUGMENTED_NOISE_FLOOR_UV = 296

# PRD 5.2: Clinical feature tolerances
CLINICAL_TOLERANCES = {
    'heart_rate':   {'unit': 'bpm', 'tol': 2.0},
    'qrs_duration': {'unit': 'ms',  'tol': 10.0},
    'qtc':          {'unit': 'ms',  'tol': 20.0},
    'pr_interval':  {'unit': 'ms',  'tol': 15.0},
    'qrs_axis':     {'unit': 'deg', 'tol': 15.0},
}
ST80_TOLERANCE_UV = 50.0

# Diagnosis mapping (Sunnybrook codes -> subgroups)
PATHOLOGICAL_GROUPS = {
    'afib_afl':  {'AFIB', 'AFIB0', 'AFLT2'},
    'mi':        {'IMIC', 'POIC', 'ALMI', 'IMI', 'AMI'},
    'mislds':    {'MISLDS'},
}


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def _resample_mason_style(sig_mv, target_len):
    """
    Resample (12, L) signal in mV to (12, target_len) using Mason's exact method.
    From third_party/ecg_reconstruction/load_functions/load_leads.py process_leads:
      x = (np.arange(actual_sample_num) + 1) / actual_sample_num
      new_x = (np.arange(sample_num) + 1) / sample_num
      twelve_leads = [np.interp(new_x, x, lead) for lead in twelve_leads]
    """
    actual = sig_mv.shape[1]
    if actual == target_len:
        return sig_mv
    x = (np.arange(actual, dtype=np.float64) + 1) / actual
    new_x = (np.arange(target_len, dtype=np.float64) + 1) / target_len
    out = np.stack([np.interp(new_x, x, sig_mv[i]) for i in range(sig_mv.shape[0])], axis=0)
    return out.astype(np.float32)


def _bandpass_mv(sig_mv, low_hz, high_hz, fs, order=4):
    """
    Apply zero-phase bandpass (Butterworth) to (12, L) signal in mV.
    Aligns spectral content with common ECG preprocessing (e.g. 0.5–40 Hz).
    """
    nyq = 0.5 * fs
    # Use small floor (0.001) so 0.5 Hz at 250 Hz is actually 0.5 Hz; avoid 0 for numerical stability
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
    """
    Load Sunnybrook Sierra XML -> Mason-normalized tensor.

    Args:
        xml_path: path to Sierra XML
        target_len: length for pad/crop when resample_to is None (default 5000)
        resample_to: if set (e.g. 2500), resample to this length Mason-style (np.interp)
                     instead of pad/crop, so preprocessing matches Mason training data.
        bandpass_hz: optional (low_hz, high_hz) to apply zero-phase bandpass before normalize.
                     Common ECG band 0.5–40 Hz aligns with many pipelines and can improve R²
                     when comparing to models trained on band-limited data.
        fs: sample rate in Hz for bandpass. If None and bandpass_hz set: 2500->250, 5000->500.

    Returns:
        signal_norm: (12, L) tensor in [0, 1] (Mason normalization)
        signal_mv:   (12, L) numpy array in mV (for clinical feature extraction)
        None, None on failure
    """
    try:
        f = sierraecg.read_file(str(xml_path))
        signal_map = {lead.label: lead.samples for lead in f.leads}

        if not all(l in signal_map for l in LEAD_ORDER):
            print(f"  WARNING: {xml_path.name} missing leads, skipping")
            return None, None

        # Stack in canonical order -> (12, samples), raw units from XML
        sig = np.stack([signal_map[l] for l in LEAD_ORDER]).astype(np.float64)

        # Sierra XML samples are in uV (5 uV/bit) -> convert to mV
        # Same as Mason's extract_twelve_leads: / 1000
        sig_mv = sig / 1000.0

        if resample_to is not None:
            # Mason-style: resample to fixed length (same as process_leads in load_leads.py)
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

        # Mason normalization: mV -> [0, 1]
        sig_norm = normalize_mason(sig_mv)  # numpy, clipped [0, 1]
        sig_norm_tensor = torch.tensor(sig_norm, dtype=torch.float32)

        return sig_norm_tensor, sig_mv

    except Exception as e:
        print(f"  ERROR loading {xml_path}: {e}")
        return None, None


def classify_record(diag_str):
    """Classify a record into pathological subgroups based on diagnosis codes."""
    if pd.isna(diag_str) or diag_str.strip() == '':
        return 'normal'

    codes = {c.strip() for c in diag_str.split(',')}

    if codes & PATHOLOGICAL_GROUPS['mislds']:
        return 'mislds'
    if codes & PATHOLOGICAL_GROUPS['afib_afl']:
        return 'afib_afl'
    if codes & PATHOLOGICAL_GROUPS['mi']:
        return 'mi'
    return 'normal'


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_waveform_fidelity(recon_mv, gt_mv):
    """
    Compute waveform fidelity metrics in mV space.

    Args:
        recon_mv, gt_mv: (1, 12, L) tensors in mV
    Returns:
        dict with per-lead and global metrics
    """
    # Mason R2 (global-mean SST)
    mason_loss = mason_batch_r2_loss(recon_mv, gt_mv).item()
    mason_r2 = -mason_loss

    # Per-lead R2 (standard, per-sample mean)
    lead_r2 = calculate_lead_r2(recon_mv, gt_mv)[0]  # (12,)
    per_lead_r2 = {LEAD_ORDER[i]: lead_r2[i].item() for i in range(12)}

    # MSE in mV
    mse = torch.mean((recon_mv - gt_mv) ** 2).item()

    # Pearson correlation per lead
    pearson_scores = {}
    for i in range(12):
        r = recon_mv[0, i].cpu()
        g = gt_mv[0, i].cpu()
        r_cent = r - r.mean()
        g_cent = g - g.mean()
        num = (r_cent * g_cent).sum()
        den = torch.sqrt((r_cent ** 2).sum() * (g_cent ** 2).sum() + 1e-8)
        pearson_scores[LEAD_ORDER[i]] = (num / den).item()

    return {
        'mason_r2': mason_r2,
        'mse_mv': mse,
        'per_lead_r2': per_lead_r2,
        'mean_r2': np.mean(list(per_lead_r2.values())),
        'per_lead_pearson': pearson_scores,
        'mean_pearson': np.mean(list(pearson_scores.values())),
    }


def compute_physics_gates(recon_mv):
    """
    Compute Einthoven and Augmented Identity physics gates.

    Args:
        recon_mv: (1, 12, L) tensor in mV
    Returns:
        dict with RMS errors in uV and pass/fail
    """
    r = recon_mv[0]  # (12, L)
    I, II, III = r[0], r[1], r[2]
    aVR, aVL, aVF = r[3], r[4], r[5]

    # Einthoven: I + III - II = 0 (or equivalently II = I + III)
    einthoven_residual = II - (I + III)
    einthoven_rms_mv = torch.sqrt(torch.mean(einthoven_residual ** 2)).item()
    einthoven_rms_uv = einthoven_rms_mv * 1000.0

    # Augmented identity: aVR + (I + II)/2 = 0
    aug_residual = aVR + (I + II) / 2.0
    aug_rms_mv = torch.sqrt(torch.mean(aug_residual ** 2)).item()
    aug_rms_uv = aug_rms_mv * 1000.0

    return {
        'einthoven_rms_uv': einthoven_rms_uv,
        'einthoven_pass': einthoven_rms_uv <= EINTHOVEN_NOISE_FLOOR_UV,
        'augmented_rms_uv': aug_rms_uv,
        'augmented_pass': aug_rms_uv <= AUGMENTED_NOISE_FLOOR_UV,
    }


def extract_clinical_features_nk2(signal_mv, fs=500):
    """
    Extract clinical features from a 12-lead signal using NeuroKit2.

    Args:
        signal_mv: (12, L) numpy array in mV
    Returns:
        dict of clinical features, or None on failure
    """
    try:
        import neurokit2 as nk
    except ImportError:
        return None

    features = {}
    lead_ii = signal_mv[1]  # Lead II for rhythm analysis

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Process Lead II for global intervals
            ecg_signals, info = nk.ecg_process(lead_ii, sampling_rate=fs)

        # Heart Rate
        if 'ECG_Rate' in ecg_signals.columns:
            hr_values = ecg_signals['ECG_Rate'].dropna()
            if len(hr_values) > 0:
                features['heart_rate'] = float(hr_values.median())

        # R-peaks for interval computation
        rpeaks = info.get('ECG_R_Peaks', [])
        if len(rpeaks) >= 2:
            rr_intervals = np.diff(rpeaks) / fs * 1000  # ms

            # QRS Duration (from delineation)
            if 'ECG_Q_Peaks' in ecg_signals.columns and 'ECG_S_Peaks' in ecg_signals.columns:
                q_peaks = ecg_signals['ECG_Q_Peaks'].dropna()
                s_peaks = ecg_signals['ECG_S_Peaks'].dropna()
                # Use delineation waves if available
                try:
                    waves = nk.ecg_delineate(ecg_signals['ECG_Clean'], rpeaks, sampling_rate=fs, method='dwt')
                    if isinstance(waves, tuple):
                        waves = waves[1]  # dict of arrays
                    if 'ECG_Q_Peaks' in waves and 'ECG_S_Peaks' in waves:
                        q = np.array(waves['ECG_Q_Peaks'])
                        s = np.array(waves['ECG_S_Peaks'])
                        valid = ~np.isnan(q) & ~np.isnan(s)
                        if valid.any():
                            qrs_dur = (s[valid] - q[valid]) / fs * 1000
                            features['qrs_duration'] = float(np.median(qrs_dur))
                except Exception:
                    pass

            # PR Interval
            try:
                waves = nk.ecg_delineate(ecg_signals['ECG_Clean'], rpeaks, sampling_rate=fs, method='dwt')
                if isinstance(waves, tuple):
                    waves = waves[1]
                if 'ECG_P_Onsets' in waves and 'ECG_R_Onsets' in waves:
                    p_on = np.array(waves['ECG_P_Onsets'])
                    r_on = np.array(waves['ECG_R_Onsets'])
                    valid = ~np.isnan(p_on) & ~np.isnan(r_on)
                    if valid.any():
                        pr = (r_on[valid] - p_on[valid]) / fs * 1000
                        pr = pr[(pr > 50) & (pr < 400)]
                        if len(pr) > 0:
                            features['pr_interval'] = float(np.median(pr))
            except Exception:
                pass

            # QT/QTc
            try:
                waves = nk.ecg_delineate(ecg_signals['ECG_Clean'], rpeaks, sampling_rate=fs, method='dwt')
                if isinstance(waves, tuple):
                    waves = waves[1]
                if 'ECG_R_Onsets' in waves and 'ECG_T_Offsets' in waves:
                    r_on = np.array(waves['ECG_R_Onsets'])
                    t_off = np.array(waves['ECG_T_Offsets'])
                    valid = ~np.isnan(r_on) & ~np.isnan(t_off)
                    if valid.any():
                        qt = (t_off[valid] - r_on[valid]) / fs * 1000
                        qt = qt[(qt > 200) & (qt < 600)]
                        if len(qt) > 0:
                            qt_median = float(np.median(qt))
                            features['qt_interval'] = qt_median
                            # Bazett correction
                            rr_sec = np.median(rr_intervals) / 1000
                            if rr_sec > 0:
                                features['qtc'] = qt_median / np.sqrt(rr_sec)
            except Exception:
                pass

    except Exception:
        pass

    # QRS Axis (from leads I and aVF)
    try:
        lead_i = signal_mv[0]
        lead_avf = signal_mv[5]
        # Net QRS area as proxy
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, info_i = nk.ecg_process(lead_i, sampling_rate=fs)
            _, info_avf = nk.ecg_process(lead_avf, sampling_rate=fs)

        rp_i = info_i.get('ECG_R_Peaks', [])
        rp_avf = info_avf.get('ECG_R_Peaks', [])

        if len(rp_i) >= 2 and len(rp_avf) >= 2:
            # Approximate net QRS amplitude
            net_i = np.mean([lead_i[max(0, p-10):p+10].sum() for p in rp_i[:5] if p < len(lead_i) - 10])
            net_avf = np.mean([lead_avf[max(0, p-10):p+10].sum() for p in rp_avf[:5] if p < len(lead_avf) - 10])
            axis_rad = np.arctan2(net_avf, net_i)
            features['qrs_axis'] = float(np.degrees(axis_rad))
    except Exception:
        pass

    # ST80 per precordial lead (V1-V6, indices 6-11)
    st80_values = {}
    for i, lead_name in enumerate(LEAD_ORDER[6:12], start=6):
        try:
            lead_sig = signal_mv[i]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ecg_s, info_lead = nk.ecg_process(lead_sig, sampling_rate=fs)
            rp = info_lead.get('ECG_R_Peaks', [])
            if len(rp) >= 2:
                # ST80 = amplitude 80ms after J-point (approx R-peak + 40ms + 80ms = R+60 samples at 500Hz)
                st80_samples = []
                offset = int(0.12 * fs)  # 120ms after R-peak ~ J+80ms
                for p in rp:
                    idx = p + offset
                    if idx < len(lead_sig):
                        st80_samples.append(lead_sig[idx])
                if st80_samples:
                    st80_values[f'{lead_name}_st_80'] = float(np.median(st80_samples) * 1000)  # to uV
        except Exception:
            pass

    features['st80'] = st80_values
    return features


def compute_morphology_delta(gt_features, recon_features):
    """
    Compute clinical morphology deltas (PRD 5.2).

    Args:
        gt_features, recon_features: dicts from extract_clinical_features_nk2
    Returns:
        dict of {feature: {gt, recon, delta, tolerance, pass}}
    """
    deltas = {}
    for feat, spec in CLINICAL_TOLERANCES.items():
        gt_val = gt_features.get(feat)
        recon_val = recon_features.get(feat)
        if gt_val is not None and recon_val is not None:
            delta = abs(recon_val - gt_val)
            deltas[feat] = {
                'gt': gt_val,
                'recon': recon_val,
                'delta': delta,
                'unit': spec['unit'],
                'tolerance': spec['tol'],
                'pass': delta <= spec['tol'],
            }

    # ST80 deltas (per precordial lead)
    gt_st80 = gt_features.get('st80', {})
    recon_st80 = recon_features.get('st80', {})
    st80_deltas = {}
    for key in gt_st80:
        if key in recon_st80:
            delta = abs(recon_st80[key] - gt_st80[key])
            st80_deltas[key] = {
                'gt': gt_st80[key],
                'recon': recon_st80[key],
                'delta': delta,
                'unit': 'uV',
                'tolerance': ST80_TOLERANCE_UV,
                'pass': delta <= ST80_TOLERANCE_UV,
            }
    deltas['st80'] = st80_deltas

    return deltas


# ---------------------------------------------------------------------------
# Main Evaluation
# ---------------------------------------------------------------------------

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== Sunnybrook Clinical Evaluation ({args.backbone}) ===")
    print(f"    Device: {device}")
    print(f"    Data: Mason-faithful ([0,1] input, mV target)")

    # 1. Load model
    print("\nLoading bridge model...")
    bridge = _get_model(args.backbone, device, target_len=5000)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    if 'bridge_state_dict' in ckpt:
        bridge.load_state_dict(ckpt['bridge_state_dict'])
        adapter_state = ckpt.get('adapter_state_dict')
    elif 'bridge' in ckpt:
        bridge.load_state_dict(ckpt['bridge'])
        adapter_state = ckpt.get('adapter')
    else:
        bridge.load_state_dict(ckpt)
        adapter_state = None

    bridge.eval()

    adapter = UniversalSpatialFusionAdapter(dim=bridge.embed_dim).to(device)
    if adapter_state:
        adapter.load_state_dict(adapter_state)
    adapter.eval()

    # 2. Load data
    data_dir = Path(args.data_dir)
    xml_files = sorted(list(data_dir.glob("*.xml")))
    print(f"Found {len(xml_files)} XML records in {data_dir}")

    # Load reference features
    master_csv = Path(args.master_csv)
    if master_csv.exists():
        df_master = pd.read_csv(master_csv)
        print(f"Loaded master features: {len(df_master)} records, {len(df_master.columns)} columns")
    else:
        df_master = pd.DataFrame()
        print("WARNING: Master feature CSV not found. Morphology deltas vs Philips XLI unavailable.")

    loss_fn = MasonR2Loss()

    # 3. Per-record evaluation
    all_results = []

    for xml_path in tqdm(xml_files, desc="Evaluating"):
        fname = xml_path.name
        record = {'file': fname}

        # Load data (Mason normalization)
        sig_norm, sig_mv = load_sunnybrook_record(xml_path)
        if sig_norm is None:
            record['status'] = 'load_failed'
            all_results.append(record)
            continue

        # Determine subgroup
        if not df_master.empty:
            row = df_master[df_master['file'] == fname]
            if not row.empty:
                diag = row.iloc[0].get('diag_codes', '')
                record['diag_codes'] = diag
                record['subgroup'] = classify_record(diag)
            else:
                record['diag_codes'] = ''
                record['subgroup'] = 'normal'
        else:
            record['subgroup'] = 'unknown'

        # Prepare tensors
        x_norm = sig_norm[MASON_INPUT_LIMB_V3].unsqueeze(0).to(device)  # [1, 3, L] I, II, V3
        gt_norm = sig_norm.unsqueeze(0).to(device)                      # [1, 12, L]
        gt_mv = denormalize_mason(gt_norm)                              # [1, 12, L] mV

        # Reconstruct
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                recon = bridge(x_norm, lead_indices=MASON_INPUT_LIMB_V3, adapter=adapter)

        # recon is in mV space (model trained to output mV)
        recon_cpu = recon.cpu()
        gt_mv_cpu = gt_mv.cpu()

        # --- Waveform Fidelity (PRD 5.4.1) ---
        fidelity = compute_waveform_fidelity(recon_cpu, gt_mv_cpu)
        record.update({
            'mason_r2': fidelity['mason_r2'],
            'mean_r2': fidelity['mean_r2'],
            'mse_mv': fidelity['mse_mv'],
            'mean_pearson': fidelity['mean_pearson'],
        })
        record['per_lead_r2'] = fidelity['per_lead_r2']
        record['per_lead_pearson'] = fidelity['per_lead_pearson']

        # --- Physics Gates (PRD 5.4) ---
        physics = compute_physics_gates(recon_cpu)
        record.update({
            'einthoven_rms_uv': physics['einthoven_rms_uv'],
            'einthoven_pass': physics['einthoven_pass'],
            'augmented_rms_uv': physics['augmented_rms_uv'],
            'augmented_pass': physics['augmented_pass'],
        })

        # --- Clinical Morphology (PRD 5.2) ---
        recon_mv_np = recon_cpu[0].numpy()
        gt_features = extract_clinical_features_nk2(sig_mv)
        recon_features = extract_clinical_features_nk2(recon_mv_np)

        if gt_features and recon_features:
            morph_delta = compute_morphology_delta(gt_features, recon_features)
            record['morphology_delta'] = morph_delta

            # Also compare against Philips XLI reference (Secondary CRR)
            if not df_master.empty and not row.empty:
                xli_ref = row.iloc[0]
                record['xli_comparison'] = {}
                for feat in ['heart_rate', 'qrs_duration', 'qtc', 'pr_interval', 'qrs_axis']:
                    xli_val = xli_ref.get(feat)
                    recon_val = recon_features.get(feat)
                    if xli_val is not None and recon_val is not None and not pd.isna(xli_val):
                        record['xli_comparison'][feat] = {
                            'xli': float(xli_val),
                            'recon': recon_val,
                            'delta': abs(recon_val - float(xli_val)),
                        }

        record['status'] = 'ok'
        all_results.append(record)

    # 4. Aggregate and print results
    ok_results = [r for r in all_results if r.get('status') == 'ok']
    n = len(ok_results)
    print(f"\n{'='*60}")
    print(f"SUNNYBROOK EVALUATION SUMMARY ({args.backbone})")
    print(f"{'='*60}")
    print(f"Records evaluated: {n}/{len(xml_files)}")

    if n == 0:
        print("No records evaluated successfully.")
        return

    # Waveform Fidelity
    print(f"\n--- Waveform Fidelity ---")
    r2s = [r['mason_r2'] for r in ok_results]
    mses = [r['mse_mv'] for r in ok_results]
    pearsons = [r['mean_pearson'] for r in ok_results]
    print(f"  Mason R2 (global-mean SST): {np.mean(r2s):.4f} +/- {np.std(r2s):.4f}")
    print(f"  Per-lead R2 (mean):          {np.mean([r['mean_r2'] for r in ok_results]):.4f}")
    print(f"  MSE (mV):                    {np.mean(mses):.6f}")
    print(f"  Pearson correlation:         {np.mean(pearsons):.4f} +/- {np.std(pearsons):.4f}")

    # Per-lead breakdown
    print(f"\n  Per-Lead R2:")
    for lead in LEAD_ORDER:
        vals = [r['per_lead_r2'].get(lead, float('nan')) for r in ok_results]
        print(f"    {lead:>4s}: {np.nanmean(vals):.4f}")

    # Physics Gates
    print(f"\n--- Physics Gates ---")
    einth = [r['einthoven_rms_uv'] for r in ok_results]
    aug = [r['augmented_rms_uv'] for r in ok_results]
    einth_pass = sum(1 for r in ok_results if r['einthoven_pass'])
    aug_pass = sum(1 for r in ok_results if r['augmented_pass'])
    print(f"  Einthoven RMS:    {np.mean(einth):.1f} uV (threshold: {EINTHOVEN_NOISE_FLOOR_UV} uV)")
    print(f"  Einthoven Pass:   {einth_pass}/{n}")
    print(f"  Augmented RMS:    {np.mean(aug):.1f} uV (threshold: {AUGMENTED_NOISE_FLOOR_UV} uV)")
    print(f"  Augmented Pass:   {aug_pass}/{n}")

    # Clinical Morphology
    print(f"\n--- Clinical Morphology Delta (Primary CRR) ---")
    for feat, spec in CLINICAL_TOLERANCES.items():
        deltas = []
        passes = 0
        total = 0
        for r in ok_results:
            md = r.get('morphology_delta', {})
            if feat in md:
                deltas.append(md[feat]['delta'])
                if md[feat]['pass']:
                    passes += 1
                total += 1
        if deltas:
            print(f"  {feat:>15s}: MAE={np.mean(deltas):7.2f} {spec['unit']:>3s}  "
                  f"(tol={spec['tol']:.0f})  Pass: {passes}/{total}")
        else:
            print(f"  {feat:>15s}: -- (extraction failed)")

    # ST80
    st80_deltas_all = []
    for r in ok_results:
        md = r.get('morphology_delta', {})
        st80 = md.get('st80', {})
        for key, val in st80.items():
            st80_deltas_all.append(val['delta'])
    if st80_deltas_all:
        st80_pass = sum(1 for d in st80_deltas_all if d <= ST80_TOLERANCE_UV)
        print(f"  {'ST80 (V1-V6)':>15s}: MAE={np.mean(st80_deltas_all):7.2f} uV   "
              f"(tol={ST80_TOLERANCE_UV:.0f})  Pass: {st80_pass}/{len(st80_deltas_all)}")

    # Pathological Subgroups
    print(f"\n--- Pathological Subgroups ---")
    subgroups = {}
    for r in ok_results:
        sg = r.get('subgroup', 'unknown')
        if sg not in subgroups:
            subgroups[sg] = []
        subgroups[sg].append(r)

    for sg_name, sg_records in sorted(subgroups.items()):
        sg_r2 = [r['mason_r2'] for r in sg_records]
        sg_pearson = [r['mean_pearson'] for r in sg_records]
        files = [r['file'] for r in sg_records]
        print(f"  {sg_name:>10s} (n={len(sg_records)}): "
              f"R2={np.mean(sg_r2):.4f}, Pearson={np.mean(sg_pearson):.4f}  "
              f"Files: {', '.join(files)}")

    # MISLDS stress test (ECG010)
    mislds = [r for r in ok_results if r.get('subgroup') == 'mislds']
    if mislds:
        print(f"\n--- MISLDS Stress Test (ECG010) ---")
        for r in mislds:
            print(f"  File: {r['file']}")
            print(f"  R2: {r['mason_r2']:.4f}, Pearson: {r['mean_pearson']:.4f}")
            print(f"  Einthoven: {r['einthoven_rms_uv']:.1f} uV, Augmented: {r['augmented_rms_uv']:.1f} uV")
            md = r.get('morphology_delta', {})
            if 'qrs_axis' in md:
                print(f"  QRS Axis: GT={md['qrs_axis']['gt']:.1f}, Recon={md['qrs_axis']['recon']:.1f}, "
                      f"Delta={md['qrs_axis']['delta']:.1f} deg")

    print(f"\n{'='*60}")

    # 5. Save outputs
    os.makedirs(args.output_dir, exist_ok=True)

    # Per-record CSV (flat columns)
    csv_rows = []
    for r in all_results:
        flat = {
            'file': r['file'],
            'status': r.get('status', 'unknown'),
            'subgroup': r.get('subgroup', ''),
            'diag_codes': r.get('diag_codes', ''),
            'mason_r2': r.get('mason_r2', ''),
            'mean_r2': r.get('mean_r2', ''),
            'mse_mv': r.get('mse_mv', ''),
            'mean_pearson': r.get('mean_pearson', ''),
            'einthoven_rms_uv': r.get('einthoven_rms_uv', ''),
            'einthoven_pass': r.get('einthoven_pass', ''),
            'augmented_rms_uv': r.get('augmented_rms_uv', ''),
            'augmented_pass': r.get('augmented_pass', ''),
        }
        # Per-lead R2
        for lead in LEAD_ORDER:
            flat[f'r2_{lead}'] = r.get('per_lead_r2', {}).get(lead, '')
            flat[f'pearson_{lead}'] = r.get('per_lead_pearson', {}).get(lead, '')
        # Morphology deltas
        for feat in CLINICAL_TOLERANCES:
            md = r.get('morphology_delta', {})
            if feat in md:
                flat[f'{feat}_delta'] = md[feat]['delta']
                flat[f'{feat}_pass'] = md[feat]['pass']
        csv_rows.append(flat)

    csv_path = os.path.join(args.output_dir, f"sunnybrook_eval_{args.backbone}.csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"Saved per-record CSV: {csv_path}")

    # Summary JSON
    summary = {
        'backbone': args.backbone,
        'checkpoint': args.checkpoint,
        'n_records': n,
        'waveform_fidelity': {
            'mason_r2_mean': float(np.mean(r2s)),
            'mason_r2_std': float(np.std(r2s)),
            'mse_mv_mean': float(np.mean(mses)),
            'pearson_mean': float(np.mean(pearsons)),
            'pearson_std': float(np.std(pearsons)),
        },
        'physics_gates': {
            'einthoven_rms_uv_mean': float(np.mean(einth)),
            'einthoven_pass_rate': einth_pass / n,
            'augmented_rms_uv_mean': float(np.mean(aug)),
            'augmented_pass_rate': aug_pass / n,
        },
        'subgroups': {
            sg: {
                'n': len(recs),
                'mason_r2_mean': float(np.mean([r['mason_r2'] for r in recs])),
                'pearson_mean': float(np.mean([r['mean_pearson'] for r in recs])),
            }
            for sg, recs in subgroups.items()
        },
    }

    json_path = os.path.join(args.output_dir, f"sunnybrook_eval_{args.backbone}_summary.json")
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary JSON: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sunnybrook Clinical Evaluation (PRD Section 5)")
    parser.add_argument("--backbone", required=True, choices=['ecgfm', 'hubert', 'ecgfounder'])
    parser.add_argument("--checkpoint", required=True, help="Path to trained bridge checkpoint")
    parser.add_argument("--data_dir", default="data/sunnybrook", help="Directory with Sunnybrook XML files")
    parser.add_argument("--master_csv", default="data/sunnybrook_master_hyperfeatures.csv",
                        help="Path to Philips XLI reference features CSV")
    parser.add_argument("--output_dir", default="results", help="Output directory for CSV and JSON")
    args = parser.parse_args()

    evaluate(args)
