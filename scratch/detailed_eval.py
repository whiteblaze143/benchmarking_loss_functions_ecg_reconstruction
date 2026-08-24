import pandas as pd
from pathlib import Path
import os
import numpy as np

base_dir = Path("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/results/factorial_mixed_level/external_delineation_generation_bound")
datasets = ['ludb', 'isp', 'zhejiang']
LOSS_NAMES = ['L_anchor', 'L_lead', 'L_edge', 'L_fft', 'L_ed', 'L_fd', 'L_mmd']

results = []
for model_dir in base_dir.iterdir():
    if not model_dir.is_dir() or not model_dir.name.startswith('f_'):
        continue
    model_id = model_dir.name
    mask = model_id.split('_')[1].zfill(7)
    
    for ds in datasets:
        summary_path = model_dir / ds / 'summary.csv'
        if not summary_path.exists(): continue
        
        try:
            df = pd.read_csv(summary_path)
            
            # Boundary Metrics
            bounds_df = df[df['metric_family'] == 'boundary']
            if len(bounds_df) == 0: continue
            
            p_onset_mae = bounds_df[bounds_df['boundary'] == 'P_Onset']['timing_mae_ms_header_derived'].values
            qrs_onset_mae = bounds_df[bounds_df['boundary'] == 'R_Onset']['timing_mae_ms_header_derived'].values
            t_offset_mae = bounds_df[bounds_df['boundary'] == 'T_Offset']['timing_mae_ms_header_derived'].values
            
            micro_f1 = bounds_df['micro_f1'].mean()
            
            # Wave Dice
            waves_df = df[df['metric_family'] == 'wave_overlap']
            qrs_dice = waves_df[waves_df['wave'] == 'QRS']['macro_dice'].values
            
            res_dict = {
                'model_id': model_id,
                'mask': mask,
                'dataset': ds,
                'micro_f1': micro_f1,
                'p_onset_mae': p_onset_mae[0] if len(p_onset_mae)>0 else None,
                'qrs_onset_mae': qrs_onset_mae[0] if len(qrs_onset_mae)>0 else None,
                't_offset_mae': t_offset_mae[0] if len(t_offset_mae)>0 else None,
                'qrs_dice': qrs_dice[0] if len(qrs_dice)>0 else None
            }
            # Add mask bits explicitly
            for i, lname in enumerate(LOSS_NAMES):
                res_dict[lname] = int(mask[i])
                
            results.append(res_dict)
        except Exception as e:
            pass

res_df = pd.DataFrame(results)

print("=== GRANULAR TOP MODELS BY DATASET ===")
for ds in datasets:
    print(f"\n--- Top 5 Models for {ds.upper()} (Ranked by Boundary Micro F1) ---")
    ds_df = res_df[res_df['dataset'] == ds].sort_values('micro_f1', ascending=False).head(5)
    print(ds_df[['model_id', 'micro_f1', 'p_onset_mae', 'qrs_dice'] + LOSS_NAMES].to_string(index=False))

print("\n\n=== LOSS TERM MARGINAL CONTRIBUTIONS (Average Micro F1 effect when active) ===")
marginal_df = res_df.groupby('dataset').mean(numeric_only=True)
for ds in datasets:
    print(f"\n--- {ds.upper()} ---")
    ds_df = res_df[res_df['dataset'] == ds]
    for lname in LOSS_NAMES:
        active_f1 = ds_df[ds_df[lname] == 1]['micro_f1'].mean()
        inactive_f1 = ds_df[ds_df[lname] == 0]['micro_f1'].mean()
        diff = active_f1 - inactive_f1
        print(f"{lname:>10}: {diff:+.4f} (Active F1: {active_f1:.4f}, Inactive F1: {inactive_f1:.4f})")

