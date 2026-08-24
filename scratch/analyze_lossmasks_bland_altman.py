import sqlite3
import pandas as pd
import numpy as np

conn = sqlite3.connect('results/clinical_biomarkers_multids/clinical_metrics.db')
df = pd.read_sql_query('''
    SELECT DISTINCT model_id, bland_v1_r2, bland_v3_r2, bland_v6_r2, 
           bland_v3_slope, bland_v3_var_ret, bland_v6_var_ret, interlead_r2
    FROM clinical_metrics 
    WHERE bland_v3_r2 IS NOT NULL
''', conn)
conn.close()

def get_mask(mid):
    parts = mid.split('_')
    for p in parts:
        if len(p) == 7 and p.isdigit():
            return p
    return None

df['mask'] = df['model_id'].apply(get_mask)
df = df.dropna(subset=['mask'])

df['corr_loss'] = df['mask'].apply(lambda m: int(m[1]))
df['deriv_loss'] = df['mask'].apply(lambda m: int(m[2]))
df['vcg_loss'] = df['mask'].apply(lambda m: int(m[3]))
df['energy_loss'] = df['mask'].apply(lambda m: int(m[4]))
df['lead_cons_loss'] = df['mask'].apply(lambda m: int(m[5]))
df['mmd_kernel'] = df['mask'].apply(lambda m: int(m[6]))

print(f"=== TOTAL EVALUATED MODELS SO FAR: {len(df)} ===\n")

print("=== MAIN EFFECTS (MEAN METRICS BY LOSS FACTOR) ===")
factors = {
    'corr_loss': 'Correlation / Pearson Loss (Digit 1)',
    'deriv_loss': 'Derivative / Gradient Loss (Digit 2)',
    'vcg_loss': 'Kors 3D VCG Loss (Digit 3)',
    'energy_loss': 'Energy Distance Loss (Digit 4)',
    'lead_cons_loss': 'Kirchhoff Lead Consistency (Digit 5)',
    'mmd_kernel': 'MMD Kernel (Digit 6: 0=None, 1=RBF, 2=Laplacian, 3=IMQ, 4=Temporal)'
}

for col, name in factors.items():
    print(f"\n--- {name} ---")
    grp = df.groupby(col)[['bland_v3_r2', 'bland_v3_slope', 'bland_v3_var_ret', 'bland_v6_var_ret', 'interlead_r2']].agg(['mean', 'count'])
    print(grp.to_string())

print("\n=== TOP 10 WINNING MASKS (Highest Bland-Altman V3 R2) ===")
print(df[['mask', 'model_id', 'bland_v3_r2', 'bland_v3_slope', 'bland_v3_var_ret', 'bland_v6_var_ret', 'interlead_r2']].sort_values(by='bland_v3_r2', ascending=False).head(10).to_string(index=False))

print("\n=== BOTTOM 10 LOSING MASKS (Severe Collapse) ===")
print(df[['mask', 'model_id', 'bland_v3_r2', 'bland_v3_slope', 'bland_v3_var_ret', 'bland_v6_var_ret', 'interlead_r2']].sort_values(by='bland_v3_r2', ascending=True).head(10).to_string(index=False))
