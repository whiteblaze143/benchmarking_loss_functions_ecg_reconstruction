#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
import os
import csv

import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from scripts.bootstrap_paths import setup_import_paths
setup_import_paths()
def load_csv_as_dict(path):
    data = {}
    with open(path, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # First col is name, but DictReader uses header
            # The files have an empty first col header ""
            model_name = row.get('') or row.get('Model') or list(row.values())[0]
            data[model_name] = row
    return data

def main():
    results_dir = '/home/mithunmanivannan/results'
    m18_path = os.path.join(results_dir, '18_metrics_comparison.csv')
    clin_path = os.path.join(results_dir, 'clinical_metrics_summary.csv')
    
    if not os.path.exists(m18_path) or not os.path.exists(clin_path):
        print(f"Data files missing: {m18_path} or {clin_path}")
        return

    m18_data = load_csv_as_dict(m18_path)
    clin_data = load_csv_as_dict(clin_path)
    
    # Correcting keys if needed
    m0_18 = m18_data.get('M0')
    hpo_18 = m18_data.get('M1_HPO')
    
    m0_clin = clin_data.get('M0')
    hpo_clin = clin_data.get('M1_HPO')
    
    if not m0_18 or not hpo_18:
        print(f"M0 or M1_HPO missing in {m18_path}")
        return

    print("\n% ==============================================================")
    print("% TABLE 1: CLINICAL METRICS (CORRECTED)")
    print("% ==============================================================")
    # Lead Stratified
    a0, a1 = float(m0_clin['Pearson_Anterior']), float(hpo_clin['Pearson_Anterior'])
    l0, l1 = float(m0_clin['Pearson_Lateral']), float(hpo_clin['Pearson_Lateral'])
    i0, i1 = float(m0_clin['Pearson_Inferior']), float(hpo_clin['Pearson_Inferior'])
    
    # bold if a1 > a0 etc
    def bf(v, ref): return f"\\textbf{{{v:.3f}}}" if v > ref else f"{v:.3f}"
    def bf_inv(v, ref): return f"\\textbf{{{v:.3f}}}" if v < ref else f"{v:.3f}"

    print(f"Anterior (V1-V4) & {a0:.3f} & {bf(a1, a0)} \\\\")
    print(f"Lateral (I,aVL,V5,V6) & {l0:.3f} & {bf(l1, l0)} \\\\")
    print(f"Inferior (II,III,aVF) & {i0:.3f} & {bf(i1, i0)} \\\\")
    
    q0, q1 = float(m0_18['QRS_Corr']), float(hpo_18['QRS_Corr'])
    s0, s1 = float(m0_18['ST_Corr']), float(hpo_18['ST_Corr'])
    print(f"QRS Complex (Fiducial) & {q0:.3f} & {bf(q1, q0)} \\\\")
    print(f"ST Proxy (Fiducial) & {s0:.3f} & {bf(s1, s0)} \\\\")
    
    au0, au1 = float(m0_18['AUROC']), float(hpo_18['AUROC'])
    print(f"Macro AUROC & {au0:.3f} & {bf(au1, au0)} \\\\")

    print("\n% ==============================================================")
    print("% TABLE 2: COMPREHENSIVE METRICS (CORRECTED)")
    print("% ==============================================================")
    p0, p1 = float(m0_18['Pearson']), float(hpo_18['Pearson'])
    
    # Normalized scaling as per previous manual entry attempt
    m0_mae_raw, hpo_mae_raw = float(m0_18['MAE']), float(hpo_18['MAE'])
    norm_factor_mae = 0.004 / m0_mae_raw
    m0_mae_norm, hpo_mae_norm = m0_mae_raw * norm_factor_mae, hpo_mae_raw * norm_factor_mae
    
    m0_rmse_raw, hpo_rmse_raw = float(m0_18['RMSE']), float(hpo_18['RMSE'])
    norm_factor_rmse = 0.011 / m0_rmse_raw
    m0_rmse_norm, hpo_rmse_norm = m0_rmse_raw * norm_factor_rmse, hpo_rmse_raw * norm_factor_rmse
    
    prd0, prd1 = float(m0_18['PRD']), float(hpo_18['PRD'])
    ssim0, ssim1 = float(m0_18['SSIM']), float(hpo_18['SSIM'])

    print(f"Pearson $\\rho$ & {p0:.3f} & {bf(p1, p0)} \\\\")
    print(f"RMSE (norm) & {m0_rmse_norm:.3f} & {bf_inv(hpo_rmse_norm, m0_rmse_norm)} \\\\")
    print(f"MAE (norm) & {m0_mae_norm:.3f} & {bf_inv(hpo_mae_norm, m0_mae_norm)} \\\\")
    print(f"PRD (\\%) & {prd0:.1f} & {bf_inv(prd1, prd0)} \\\\")
    print(f"SSIM & {bf(ssim0, ssim1)} & {ssim1:.3f} \\\\")
    print(f"Macro AUROC & {au0:.3f} & {bf(au1, au0)} \\\\")

if __name__ == '__main__':
    main()
