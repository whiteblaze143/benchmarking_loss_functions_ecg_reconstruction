#!/usr/bin/env python3
"""
Aggregate all models trained exclusively on Lead I into a single comprehensive CSV document.
Captures all evaluation metrics across:
- PTB-XL Validation Multi-Task Reconstruction
- PTB-XL Delineation & Segmentation (IoU, Dice, F1)
- PTB-XL 12-Lead Individual Decompositions (Pearson, RMSE, MAE, SNR)
- PTB-XL Anatomical Subgroups (Chest, Limb, Septal, Anterior, Lateral, High Lateral, Inferior)
- Russian Database (RDB) Blinded External Generalization (Signal robustness, boundary timing MAEs, wave IoUs)
"""

import os
import sys
import re
import json
import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'book'))
from live_results import load_onelead_queue, load_summary_tree, load_spatial_training_logs, load_onelead_rdb, read_sql, source_path

OUTPUT_CSV = ROOT / 'results' / 'lead1_all_models_comprehensive_metrics.csv'

def clean_arch_name(name: str) -> str:
    n = re.sub(r'^conv(10|15)e_', '', name)
    n = re.sub(r'^spatial_1lead_', '', n)
    n = re.sub(r'_s\d+_l[01]$', '', n)
    n = re.sub(r'_\d{7}_s\d+_l[01]$', '', n)
    return n

def extract_seed(name: str) -> int | None:
    m = re.search(r'_s(\d+)_', name)
    return int(m.group(1)) if m else None

def extract_mask(name: str) -> str:
    m = re.search(r'_(\d{7})_', name)
    if m:
        return m.group(1)
    if '1110000' in name or 'conv' in name:
        return '1110000'
    return '-'

def main():
    print("Collecting all Lead 1 models across repositories and databases...")
    all_models = {}

    # --------------------------------------------------------------------------
    # 1. Convergence Runs (10e and 15e)
    # --------------------------------------------------------------------------
    c10_df = load_summary_tree('refine-logs/convergence_10e/runs')
    c10_l0 = c10_df[c10_df.observed_lead == 'I'] if not c10_df.empty else pd.DataFrame()
    for _, r in c10_l0.iterrows():
        mid = r['run_name']
        if mid.startswith('D'):
            track = 'Stage A 3D Theta Spatial Benchmark'
            epochs = 10
        elif 'zscore' in mid:
            track = '15-Epoch Normalization Control'
            epochs = 15
        elif 'conv15e' in mid:
            track = '15-Epoch Convergence'
            epochs = 15
        else:
            track = '10-Epoch Convergence'
            epochs = 10
        all_models[mid] = {
            'model_id': mid,
            'study_track': track,
            'architecture': clean_arch_name(mid),
            'observed_lead': 'I',
            'factorial_mask': '1110000',
            'seed': extract_seed(mid) or 42,
            'epochs_trained': epochs,
            'status': r.get('run_state', 'completed'),
            'val_missing_pearson': r.get('val_missing_pearson'),
            'val_missing_pearson_p05': r.get('val_missing_pearson_p05'),
            'val_recon_loss': r.get('val_recon_loss'),
            'miou_wave': r.get('miou_wave'),
            'P_iou': r.get('P_iou'),
            'QRS_iou': r.get('QRS_iou'),
            'T_iou': r.get('T_iou'),
            'macro_f1_wave': r.get('macro_f1_wave'),
            'boundary_f1_smoke': r.get('boundary_f1_smoke'),
            'train_recon': r.get('train_recon'),
            'train_ssl': r.get('train_ssl'),
            'train_ce': r.get('train_ce'),
            'train_dice': r.get('train_dice'),
            'train_boundary': r.get('train_boundary'),
            'train_fid': r.get('train_fid'),
            'train_total': r.get('train_total'),
            'train_grad_norm': r.get('train_grad_norm'),
            'train_consistency': r.get('train_consistency'),
        }

    # --------------------------------------------------------------------------
    # 2. 3-Epoch Multi-Task ECGAIM Screening Runs
    # --------------------------------------------------------------------------
    q_runs = load_onelead_queue(source_path('refine-logs/wavelet_ssl_1110000/full/queue.sqlite'))
    q_l0 = q_runs[q_runs.observed_lead == 'I'] if not q_runs.empty else pd.DataFrame()
    for _, r in q_l0.iterrows():
        mid = r['id']
        if mid not in all_models:
            all_models[mid] = {
                'model_id': mid,
                'study_track': '3-Epoch Wavelet & SSL Screening',
                'architecture': clean_arch_name(mid),
                'observed_lead': 'I',
                'factorial_mask': '1110000',
                'seed': extract_seed(mid) or 42,
                'epochs_trained': 3,
                'status': r.get('status', 'completed'),
                'val_missing_pearson': r.get('metric.val_missing_pearson'),
                'val_missing_pearson_p05': r.get('metric.val_missing_pearson_p05'),
                'val_recon_loss': r.get('metric.val_recon_loss'),
                'miou_wave': r.get('metric.miou_wave'),
                'P_iou': r.get('metric.P_iou'),
                'QRS_iou': r.get('metric.QRS_iou'),
                'T_iou': r.get('metric.T_iou'),
                'macro_f1_wave': r.get('metric.macro_f1_wave'),
                'boundary_f1_smoke': r.get('metric.boundary_f1_smoke'),
                'train_recon': r.get('metric.train_recon'),
                'train_ssl': r.get('metric.train_ssl'),
                'train_ce': r.get('metric.train_ce'),
                'train_dice': r.get('metric.train_dice'),
                'train_boundary': r.get('metric.train_boundary'),
                'train_fid': r.get('metric.train_fid'),
                'train_total': r.get('metric.train_total'),
                'train_grad_norm': r.get('metric.train_grad_norm'),
                'train_consistency': r.get('metric.train_consistency'),
            }

    # --------------------------------------------------------------------------
    # 3. Spatial Architecture Grid Runs
    # --------------------------------------------------------------------------
    sp = load_spatial_training_logs('refine-logs/queue_spatial_1lead/jobs')
    sp_l0 = sp[sp.observed_lead == 'I'] if not sp.empty else pd.DataFrame()
    for _, r in sp_l0.iterrows():
        mid = r['run_name']
        if mid not in all_models:
            all_models[mid] = {
                'model_id': mid,
                'study_track': '3-Epoch Spatial Architecture Grid',
                'architecture': r.get('variant', clean_arch_name(mid)),
                'observed_lead': 'I',
                'factorial_mask': r.get('factorial_mask', extract_mask(mid)),
                'seed': r.get('seed', extract_seed(mid) or 42),
                'epochs_trained': int(r.get('epochs_logged', 3)) if pd.notna(r.get('epochs_logged')) else 3,
                'status': r.get('status', 'completed'),
                'val_missing_pearson': r.get('best_val_missing_pearson') or r.get('final_val_missing_pearson'),
                'val_missing_pearson_p05': np.nan,
                'val_recon_loss': r.get('final_val_loss'),
                'miou_wave': np.nan,
                'P_iou': np.nan,
                'QRS_iou': np.nan,
                'T_iou': np.nan,
                'macro_f1_wave': np.nan,
                'boundary_f1_smoke': np.nan,
                'train_recon': np.nan,
                'train_ssl': np.nan,
                'train_ce': np.nan,
                'train_dice': np.nan,
                'train_boundary': np.nan,
                'train_fid': np.nan,
                'train_total': np.nan,
                'train_grad_norm': np.nan,
                'train_consistency': np.nan,
            }

    # --------------------------------------------------------------------------
    # 3.5 Reference Ground Truth Ceiling on Original RDB Waveforms
    # --------------------------------------------------------------------------
    all_models['__original_l0__'] = {
        'model_id': '__original_l0__',
        'study_track': 'Ground Truth Ceiling (Original Waveforms)',
        'architecture': 'Original 12-Lead Ground Truth (Upper Bound)',
        'observed_lead': 'All 12 Leads',
        'factorial_mask': 'original',
        'seed': 42,
        'epochs_trained': 0,
        'status': 'reference_ceiling',
        'val_missing_pearson': 1.0,
        'val_missing_pearson_p05': 1.0,
        'val_recon_loss': 0.0,
        'miou_wave': np.nan,
        'P_iou': np.nan,
        'QRS_iou': np.nan,
        'T_iou': np.nan,
        'macro_f1_wave': np.nan,
        'boundary_f1_smoke': np.nan,
        'train_recon': np.nan,
        'train_ssl': np.nan,
        'train_ce': np.nan,
        'train_dice': np.nan,
        'train_boundary': np.nan,
        'train_fid': np.nan,
        'train_total': np.nan,
        'train_grad_norm': np.nan,
        'train_consistency': np.nan,
    }

    # Convert to DataFrame
    df = pd.DataFrame(list(all_models.values()))
    print(f"Total base Lead 1 models aggregated: {len(df)}")

    # --------------------------------------------------------------------------
    # 4. Integrate PTB-XL Anatomical Subgroups (from convergence_per_lead_evaluation_v1)
    # --------------------------------------------------------------------------
    try:
        con_pl = sqlite3.connect('file:results/convergence_per_lead_evaluation_v1/compact.sqlite?mode=ro', uri=True)
        pl_evals = pd.read_sql('SELECT * FROM evaluations WHERE observed_lead=\"I\"', con_pl)
        if not pl_evals.empty:
            subgroup_cols = [
                'mean_all_missing_r', 'p05_all_missing_r', 'mean_chest_r', 'p05_chest_r',
                'mean_limb_r', 'p05_limb_r', 'mean_septal_r', 'mean_anterior_r',
                'mean_lateral_chest_r', 'mean_high_lateral_r', 'mean_inferior_r'
            ]
            for c in subgroup_cols:
                if c in pl_evals:
                    sub_map = pl_evals.set_index('model_id')[c].to_dict()
                    df[c] = df['model_id'].map(sub_map)
        else:
            for c in ['mean_all_missing_r', 'p05_all_missing_r', 'mean_chest_r', 'p05_chest_r', 'mean_limb_r', 'p05_limb_r', 'mean_septal_r', 'mean_anterior_r', 'mean_lateral_chest_r', 'mean_high_lateral_r', 'mean_inferior_r']:
                df[c] = np.nan
    except Exception as e:
        print(f"Warning loading per-lead evaluations: {e}")
        for c in ['mean_all_missing_r', 'p05_all_missing_r', 'mean_chest_r', 'p05_chest_r', 'mean_limb_r', 'p05_limb_r', 'mean_septal_r', 'mean_anterior_r', 'mean_lateral_chest_r', 'mean_high_lateral_r', 'mean_inferior_r']:
            df[c] = np.nan

    # --------------------------------------------------------------------------
    # 5. Integrate PTB-XL 12-Lead Individual Metrics (from per_lead_metrics)
    # --------------------------------------------------------------------------
    try:
        pl_metrics = pd.read_sql('SELECT * FROM per_lead_metrics', con_pl)
        leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        for lead in leads:
            sub = pl_metrics[pl_metrics.lead_name == lead]
            if not sub.empty:
                df[f'pearson_lead_{lead}'] = df['model_id'].map(sub.set_index('model_id')['mean_pearson'].to_dict())
                df[f'p05_lead_{lead}'] = df['model_id'].map(sub.set_index('model_id')['p05_pearson'].to_dict())
                df[f'p50_lead_{lead}'] = df['model_id'].map(sub.set_index('model_id')['p50_pearson'].to_dict())
                df[f'p95_lead_{lead}'] = df['model_id'].map(sub.set_index('model_id')['p95_pearson'].to_dict())
                df[f'rmse_lead_{lead}'] = df['model_id'].map(sub.set_index('model_id')['rmse'].to_dict())
                df[f'mae_lead_{lead}'] = df['model_id'].map(sub.set_index('model_id')['mae'].to_dict())
                df[f'snr_lead_{lead}'] = df['model_id'].map(sub.set_index('model_id')['snr_db'].to_dict())
            else:
                for metric in ['pearson', 'p05', 'p50', 'p95', 'rmse', 'mae', 'snr']:
                    df[f'{metric}_lead_{lead}'] = np.nan
    except Exception as e:
        print(f"Warning loading per-lead metrics: {e}")

    # --------------------------------------------------------------------------
    # 6. Integrate Convergence RDB Metrics (from results/convergence_rdb_semiseg_v1)
    # --------------------------------------------------------------------------
    try:
        con_crdb = sqlite3.connect('file:results/convergence_rdb_semiseg_v1/compact.sqlite?mode=ro', uri=True)
        crdb_ev = pd.read_sql('SELECT * FROM evaluations WHERE model_id LIKE \"%_l0\"', con_crdb)
        crdb_bd = pd.read_sql('SELECT * FROM boundary_summaries WHERE model_id LIKE \"%_l0\" AND stage=\"full\" AND lead_group=\"all_missing\"', con_crdb)
        
        # Primary RDB evaluations
        if not crdb_ev.empty:
            for c in ['primary_mean_micro_f1_20ms', 'primary_signal_pearson_p05', 'miou_wave', 'p_iou', 'qrs_iou', 't_iou', 'p_dice', 'qrs_dice', 't_dice']:
                if c in crdb_ev:
                    col_name = f'rdb_{c}' if not c.startswith('primary') else ('rdb_boundary_micro_f1_20ms' if 'micro_f1' in c else 'rdb_signal_pearson_p05')
                    df[col_name] = df['model_id'].map(crdb_ev.set_index('model_id')[c].to_dict())
        
        # Boundary MAEs and F1s
        if not crdb_bd.empty:
            for b in ['P_onset', 'P_offset', 'QRS_onset', 'QRS_offset', 'T_onset', 'T_offset']:
                b_sub = crdb_bd[crdb_bd.boundary == b]
                if not b_sub.empty:
                    df[f'rdb_mae_{b}_ms'] = df['model_id'].map(b_sub.set_index('model_id')['mae_ms'].to_dict())
                    df[f'rdb_f1_{b}'] = df['model_id'].map(b_sub.set_index('model_id')['micro_f1_20ms'].to_dict())
    except Exception as e:
        print(f"Warning loading convergence RDB: {e}")

    # --------------------------------------------------------------------------
    # 7. Integrate Screened Spatial RDB Metrics (from results/onelead_rdb_semiseg_screened_v1)
    # --------------------------------------------------------------------------
    try:
        rdb_scr = load_onelead_rdb(source_path('results/onelead_rdb_semiseg_screened_v1/compact.sqlite'))
        rdb_ev_s = rdb_scr['evaluations']
        rdb_bd_s = rdb_scr['boundary_summaries']
        rdb_rg_s = rdb_scr['region_summaries']

        bd_full_s = rdb_bd_s[(rdb_bd_s.stage == 'full') & (rdb_bd_s.lead_group == 'all_missing')].copy()
        p_mae_s = bd_full_s.pivot_table(index='model_id', columns='boundary', values='mae_ms')
        p_f1_s = bd_full_s.pivot_table(index='model_id', columns='boundary', values='micro_f1_20ms')

        rg_full_s = rdb_rg_s[(rdb_rg_s.stage == 'full') & (rdb_rg_s.lead_group == 'all_missing')].copy()
        p_dice_s = rg_full_s.pivot_table(index='model_id', columns='wave', values='dice')
        p_iou_s = rg_full_s.pivot_table(index='model_id', columns='wave', values='iou')

        for _, r in rdb_ev_s[rdb_ev_s.model_id.str.contains('_l0')].iterrows():
            mid = r['model_id']
            idx = df[df.model_id == mid].index
            if len(idx) > 0:
                if pd.isna(df.loc[idx, 'rdb_boundary_micro_f1_20ms'].values[0]):
                    df.loc[idx, 'rdb_boundary_micro_f1_20ms'] = r.get('primary_mean_micro_f1_20ms')
                if pd.isna(df.loc[idx, 'rdb_signal_pearson_p05'].values[0]):
                    df.loc[idx, 'rdb_signal_pearson_p05'] = r.get('primary_signal_pearson_p05')
                
                if mid in p_mae_s.index:
                    for b in ['P_onset', 'P_offset', 'QRS_onset', 'QRS_offset', 'T_onset', 'T_offset']:
                        if b in p_mae_s.columns and pd.isna(df.loc[idx, f'rdb_mae_{b}_ms'].values[0]):
                            df.loc[idx, f'rdb_mae_{b}_ms'] = p_mae_s.loc[mid, b]
                        if b in p_f1_s.columns and pd.isna(df.loc[idx, f'rdb_f1_{b}'].values[0]):
                            df.loc[idx, f'rdb_f1_{b}'] = p_f1_s.loc[mid, b]

                if mid in p_dice_s.index:
                    for w in ['P', 'QRS', 'T']:
                        if w in p_dice_s.columns and pd.isna(df.loc[idx, f'rdb_{w.lower()}_dice'].values[0]):
                            df.loc[idx, f'rdb_{w.lower()}_dice'] = p_dice_s.loc[mid, w]
                        if w in p_iou_s.columns and pd.isna(df.loc[idx, f'rdb_{w.lower()}_iou'].values[0]):
                            df.loc[idx, f'rdb_{w.lower()}_iou'] = p_iou_s.loc[mid, w]
    except Exception as e:
        print(f"Warning loading screened RDB: {e}")

    # Sort logically: Track priority, then val_missing_pearson descending
    track_order = {
        '15-Epoch Convergence': 1,
        '15-Epoch Normalization Control': 2,
        'Stage A 3D Theta Spatial Benchmark': 3,
        '10-Epoch Convergence': 4,
        '3-Epoch Wavelet & SSL Screening': 5,
        '3-Epoch Spatial Architecture Grid': 6
    }
    df['track_order'] = df['study_track'].map(track_order).fillna(99)
    df = df.sort_values(['track_order', 'val_missing_pearson'], ascending=[True, False]).drop(columns=['track_order'])

    # Write Master CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Master CSV successfully written to: {OUTPUT_CSV}")
    print(f"Rows: {len(df)} | Columns: {len(df.columns)}")

if __name__ == '__main__':
    main()
