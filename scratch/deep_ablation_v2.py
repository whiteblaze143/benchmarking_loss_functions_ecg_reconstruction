import pandas as pd
import numpy as np

df = pd.read_csv("results/factorial_v4/temporal_mmd_evaluation.csv")
df['mean_error'] = (df['mean_real'] - df['mean_recon']).abs()
df['var_ratio_error'] = np.abs(np.log(df['variance_ratio'].replace(0, np.nan)))

# Add boolean columns for the factorial mask
# [MSE][Pearson][Derivative][VCG][EnergyDifference][Einthoven][MMD_Kernel]
df['model_mask_str'] = df['model_mask'].astype(str).str.zfill(7)
df['has_mse'] = df['model_mask_str'].str[0] == '1'
df['has_pearson'] = df['model_mask_str'].str[1] == '1'
df['has_deriv'] = df['model_mask_str'].str[2] == '1'
df['has_vcg'] = df['model_mask_str'].str[3] == '1'
df['has_ed'] = df['model_mask_str'].str[4] == '1'
df['has_einthoven'] = df['model_mask_str'].str[5] == '1'
df['mmd_kernel'] = df['model_mask_str'].str[6].astype(int)

# Effect of VCG on Leads
print("=== EFFECT OF VCG BY LEAD (MAE) ===")
vcg_effect = df.groupby(['has_vcg', 'lead'])['mean_error'].mean().unstack()
print(vcg_effect)

# Effect of Einthoven on Leads
print("\n=== EFFECT OF EINTHOVEN BY LEAD (MAE) ===")
einthoven_effect = df.groupby(['has_einthoven', 'lead'])['mean_error'].mean().unstack()
print(einthoven_effect)

# Effect of MMD Kernel on Features
print("\n=== EFFECT OF MMD KERNEL BY FEATURE (MAE) ===")
mmd_effect = df.groupby(['mmd_kernel', 'clinical_feature'])['mean_error'].mean().unstack()
print(mmd_effect)

# Effect of ED on Variance Ratio by Feature
print("\n=== EFFECT OF ED ON VARIANCE RATIO BY FEATURE ===")
ed_var = df.groupby(['has_ed', 'clinical_feature'])['var_ratio_error'].mean().unstack()
print(ed_var)

# Effect of Derivative on Features
print("\n=== EFFECT OF DERIVATIVE ON FEATURES (MAE) ===")
deriv_effect = df.groupby(['has_deriv', 'clinical_feature'])['mean_error'].mean().unstack()
print(deriv_effect)

