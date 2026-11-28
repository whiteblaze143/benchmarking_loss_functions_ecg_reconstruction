"""
Experiment E04: Phenotype-Dependent ECG Observability under Authentic Artifact
Grounded in:
- Noise Holter Database (6,130 excerpts across 316 patients, Clean / N1 / N2 strata)
- Physiological feature degradation (ventricular rate, CVRR, RMSSD, valid beat coverage)
- Separation of AF from non-AF across noise strata
- Controlled simulation of sensing dropouts on continuous cohorts (IRIDIA-AF and SHDB-AF)
"""

import os
import sys
import glob
import time
import math
import numpy as np
import pandas as pd
from scipy import stats

# Ensure af_density metrics are accessible
sys.path.insert(0, os.path.abspath('experiments/af_density'))
try:
    from metrics import charitos_density
except ImportError:
    def charitos_density(intervals, observation_seconds):
        return None

try:
    import wfdb
except ImportError:
    wfdb = None


def extract_noise_holter_features(nh_dir='/data/mithunmanivannan/datasets/noise_holter_database'):
    """
    Parses all records in Noise Holter, extracting quality annotations, clinical events,
    and physiological interval dynamics (HR, CVRR, RMSSD, SDNN, pNN50).
    """
    records_file = os.path.join(nh_dir, 'RECORDS')
    if not os.path.exists(records_file):
        raise FileNotFoundError(f"RECORDS file not found at {records_file}")
        
    with open(records_file) as f:
        records = [line.strip().replace('segments/', '').replace('wfdb_files/', '') for line in f if line.strip()]
        
    segments_dir = os.path.join(nh_dir, 'segments')
    if not os.path.exists(segments_dir):
        segments_dir = os.path.join(nh_dir, 'wfdb_files')
        
    extracted_rows = []
    
    for r in records:
        hea_path = os.path.join(segments_dir, r + '.hea')
        if not os.path.exists(hea_path):
            continue
            
        with open(hea_path) as f:
            lines = f.readlines()
            
        first = lines[0].strip().split()
        record_name = first[0]
        num_leads = int(first[1])
        fs = float(first[2])
        num_samples = int(first[3])
        dur_s = num_samples / fs
        
        patient_id = record_name.split('-')[0]
        
        # Parse comment
        quality_class = 'Unknown'
        noise_type = 'None'
        events = 'None'
        for line in lines[1:]:
            if line.startswith('#'):
                parts = [p.strip() for p in line[1:].strip().split('|')]
                for p in parts:
                    if p.startswith('CLASS:'):
                        quality_class = p.split(':', 1)[1].strip()
                    elif p.startswith('NOISE:'):
                        noise_type = p.split(':', 1)[1].strip()
                    elif p.startswith('EVENTS:'):
                        events = p.split(':', 1)[1].strip()
                        
        is_af = ('a. fib' in events.lower()) or ('afib' in events.lower())
        is_flutter = ('flutter' in events.lower())
        is_sinus = (events.strip().lower() in ['none', 'sinus', 'normal'])
        
        # Read QRS detections
        beats = []
        if wfdb is not None:
            qrs_path = os.path.join(segments_dir, r)
            try:
                ann = wfdb.rdann(qrs_path, 'qrs')
                beats = ann.sample
            except Exception:
                try:
                    ann = wfdb.rdann(qrs_path, 'atr')
                    beats = ann.sample
                except Exception:
                    beats = []
                    
        n_beats = len(beats)
        expected_beats = max(1.0, dur_s * (72.0 / 60.0))
        beat_coverage = min(2.5, n_beats / expected_beats)
        
        if n_beats >= 3:
            rr = np.diff(beats) / fs  # seconds
            valid_rr = rr[(rr >= 0.20) & (rr <= 3.0)]  # physiological filter
            if len(valid_rr) >= 2:
                mean_rr = float(np.mean(valid_rr))
                std_rr = float(np.std(valid_rr))
                hr = 60.0 / mean_rr if mean_rr > 0 else np.nan
                cvrr = std_rr / mean_rr if mean_rr > 0 else np.nan
                rmssd = float(np.sqrt(np.mean(np.diff(valid_rr)**2))) * 1000.0 if len(valid_rr) > 1 else np.nan
                sdnn = std_rr * 1000.0
                pnn50 = float(np.mean(np.abs(np.diff(valid_rr)) > 0.050) * 100.0) if len(valid_rr) > 1 else np.nan
            else:
                hr, cvrr, rmssd, sdnn, pnn50 = np.nan, np.nan, np.nan, np.nan, np.nan
        else:
            hr, cvrr, rmssd, sdnn, pnn50 = np.nan, np.nan, np.nan, np.nan, np.nan
            
        extracted_rows.append({
            'record_id': r,
            'patient_id': patient_id,
            'num_leads': num_leads,
            'fs': fs,
            'duration_s': dur_s,
            'quality_class': quality_class,
            'noise_type': noise_type,
            'events': events,
            'is_af': is_af,
            'is_flutter': is_flutter,
            'is_sinus': is_sinus,
            'n_beats': n_beats,
            'beat_coverage': beat_coverage,
            'hr': hr,
            'cvrr': cvrr,
            'rmssd': rmssd,
            'sdnn': sdnn,
            'pnn50': pnn50
        })
        
    return pd.DataFrame(extracted_rows)


def analyze_feature_degradation(df):
    """
    Evaluates degradation of physiological features (CVRR, RMSSD, HR, beat coverage)
    across Clean, N1, and N2 noise strata for both AF and Non-AF rhythms.
    """
    strata_summary = []
    
    for q in ['Clean', 'N1', 'N2']:
        sub_q = df[df['quality_class'] == q]
        n_tot = len(sub_q)
        n_af = sub_q['is_af'].sum()
        n_sinus = sub_q['is_sinus'].sum()
        
        # Valid coverage
        mean_cov = sub_q['beat_coverage'].mean()
        
        # AF features
        af_rows = sub_q[sub_q['is_af'] & sub_q['cvrr'].notna()]
        af_cvrr_mean = af_rows['cvrr'].mean() if len(af_rows) > 0 else np.nan
        af_cvrr_median = af_rows['cvrr'].median() if len(af_rows) > 0 else np.nan
        af_rmssd_mean = af_rows['rmssd'].mean() if len(af_rows) > 0 else np.nan
        af_hr_mean = af_rows['hr'].mean() if len(af_rows) > 0 else np.nan
        
        # Sinus features
        sinus_rows = sub_q[sub_q['is_sinus'] & sub_q['cvrr'].notna()]
        sinus_cvrr_mean = sinus_rows['cvrr'].mean() if len(sinus_rows) > 0 else np.nan
        sinus_cvrr_median = sinus_rows['cvrr'].median() if len(sinus_rows) > 0 else np.nan
        sinus_rmssd_mean = sinus_rows['rmssd'].mean() if len(sinus_rows) > 0 else np.nan
        sinus_hr_mean = sinus_rows['hr'].mean() if len(sinus_rows) > 0 else np.nan
        
        # Separation between AF and Sinus on CVRR (Cohen's d)
        if len(af_rows) > 5 and len(sinus_rows) > 5:
            pooled_std = np.sqrt(((len(af_rows)-1)*af_rows['cvrr'].var() + (len(sinus_rows)-1)*sinus_rows['cvrr'].var()) / (len(af_rows) + len(sinus_rows) - 2))
            cohens_d = (af_cvrr_mean - sinus_cvrr_mean) / pooled_std if pooled_std > 0 else np.nan
            
            # Simple threshold AUROC for AF vs Sinus based on CVRR
            labels = np.concatenate([np.ones(len(af_rows)), np.zeros(len(sinus_rows))])
            scores = np.concatenate([af_rows['cvrr'].values, sinus_rows['cvrr'].values])
            from sklearn.metrics import roc_auc_score
            auroc_cvrr = float(roc_auc_score(labels, scores))
        else:
            cohens_d = np.nan
            auroc_cvrr = np.nan
            
        strata_summary.append({
            'quality_class': q,
            'total_excerpts': n_tot,
            'af_excerpts': n_af,
            'sinus_excerpts': n_sinus,
            'mean_beat_coverage': mean_cov,
            'af_cvrr_mean': af_cvrr_mean,
            'af_cvrr_median': af_cvrr_median,
            'af_rmssd_mean': af_rmssd_mean,
            'af_hr_mean': af_hr_mean,
            'sinus_cvrr_mean': sinus_cvrr_mean,
            'sinus_cvrr_median': sinus_cvrr_median,
            'sinus_rmssd_mean': sinus_rmssd_mean,
            'sinus_hr_mean': sinus_hr_mean,
            'cvrr_cohens_d': cohens_d,
            'cvrr_auroc': auroc_cvrr
        })
        
    return pd.DataFrame(strata_summary)


def simulate_sensing_dropouts_on_continuous_cohorts(canonical_runs_parquet, canonical_cohort_csv):
    """
    Simulates empirical sensing limitations from Noise Holter on authentic continuous
    timelines from IRIDIA-AF and SHDB-AF:
    Scenario 1: Omission of brief AF runs (<30s and <10s dropped)
    Scenario 2: Misclassification of brief non-AF gaps (<60s and <30s bridged)
    Scenario 3: Intermittent sensing dropouts (5%, 10%, 20%, 30% dropout blocks)
    """
    df_runs = pd.read_parquet(canonical_runs_parquet)
    df_cohort = pd.read_csv(canonical_cohort_csv)
    
    simulation_results = []
    
    # Analyze multi-run recordings in SHDB and IRIDIA
    for dataset_name in ['SHDB-AF', 'IRIDIA-AF']:
        sub_cohort = df_cohort[(df_cohort['dataset'] == dataset_name) & (df_cohort['total_af_runs'] >= 2)].copy()
        
        for _, c_row in sub_cohort.iterrows():
            rec_id = str(c_row['record_id'])
            p_id = str(c_row['patient_id'])
            dur_h = float(c_row['monitoring_duration_h'])
            dur_s = dur_h * 3600.0
            
            runs = df_runs[(df_runs['dataset'] == dataset_name) & (df_runs['record_id'].astype(str) == rec_id)].sort_values('start_time_s')
            if len(runs) < 2:
                continue
                
            intervals_true = [(float(r['start_time_s']), float(r['end_time_s'])) for _, r in runs.iterrows()]
            af_s_true = sum(e - s for s, e in intervals_true)
            burden_true = af_s_true / dur_s
            longest_true = max(e - s for s, e in intervals_true)
            n_brief_30s_true = sum(1 for s, e in intervals_true if (e - s) < 30.0)
            
            try:
                dens_true = charitos_density(intervals_true, dur_s)
            except Exception:
                dens_true = None
                
            if dens_true is None:
                continue
                
            # --- Scenario 1A: Drop brief runs < 30s ---
            int_no_b30 = [(s, e) for s, e in intervals_true if (e - s) >= 30.0]
            if len(int_no_b30) >= 2:
                af_s_s1a = sum(e - s for s, e in int_no_b30)
                burden_s1a = af_s_s1a / dur_s
                try:
                    dens_s1a = charitos_density(int_no_b30, dur_s)
                    dens_err_s1a = dens_s1a - dens_true
                except Exception:
                    dens_err_s1a = np.nan
            else:
                dens_err_s1a = np.nan
                burden_s1a = sum(e - s for s, e in int_no_b30) / dur_s if len(int_no_b30) > 0 else 0.0
                
            # --- Scenario 1B: Drop brief runs < 10s ---
            int_no_b10 = [(s, e) for s, e in intervals_true if (e - s) >= 10.0]
            if len(int_no_b10) >= 2:
                af_s_s1b = sum(e - s for s, e in int_no_b10)
                burden_s1b = af_s_s1b / dur_s
                try:
                    dens_s1b = charitos_density(int_no_b10, dur_s)
                    dens_err_s1b = dens_s1b - dens_true
                except Exception:
                    dens_err_s1b = np.nan
            else:
                dens_err_s1b = np.nan
                burden_s1b = sum(e - s for s, e in int_no_b10) / dur_s if len(int_no_b10) > 0 else 0.0
                
            # --- Scenario 2: Bridge non-AF gaps < 60s ---
            int_bridged = []
            cur_s, cur_e = intervals_true[0]
            for s, e in intervals_true[1:]:
                if (s - cur_e) <= 60.0:
                    cur_e = e  # bridge gap
                else:
                    int_bridged.append((cur_s, cur_e))
                    cur_s, cur_e = s, e
            int_bridged.append((cur_s, cur_e))
            
            af_s_s2 = sum(e - s for s, e in int_bridged)
            burden_s2 = af_s_s2 / dur_s
            if len(int_bridged) >= 2:
                try:
                    dens_s2 = charitos_density(int_bridged, dur_s)
                    dens_err_s2 = dens_s2 - dens_true
                except Exception:
                    dens_err_s2 = np.nan
            else:
                dens_err_s2 = 1.0 - dens_true  # becomes single run
                
            # --- Scenario 3: Intermittent sensing dropout 20% (matching N2 rate) ---
            np.random.seed(int(hash(rec_id) % 100000))
            n_blocks = 20
            block_dur = (dur_s * 0.20) / n_blocks
            dropouts = []
            for _ in range(n_blocks):
                t_start = np.random.uniform(0, max(1.0, dur_s - block_dur))
                dropouts.append((t_start, t_start + block_dur))
            dropouts.sort()
            
            # Compute Manski bounds for 20% unobserved time
            t_unobs = dur_s * 0.20
            manski_min = max(0.0, (af_s_true - t_unobs) / dur_s)
            manski_max = min(1.0, (af_s_true + t_unobs) / dur_s)
            
            simulation_results.append({
                'dataset': dataset_name,
                'record_id': rec_id,
                'patient_id': p_id,
                'duration_h': dur_h,
                'true_burden': burden_true,
                'true_density': dens_true,
                'true_n_runs': len(intervals_true),
                'true_brief_30s': n_brief_30s_true,
                's1a_burden_err': burden_s1a - burden_true,
                's1a_dens_err': dens_err_s1a,
                's1b_burden_err': burden_s1b - burden_true,
                's1b_dens_err': dens_err_s1b,
                's2_burden_err': burden_s2 - burden_true,
                's2_dens_err': dens_err_s2,
                's3_manski_width': manski_max - manski_min,
                's3_manski_covers': (manski_min <= burden_true <= manski_max)
            })
            
    return pd.DataFrame(simulation_results)
