import pandas as pd
import numpy as np

df = pd.read_csv("results/factorial_v4/temporal_mmd_evaluation.csv")

# We want to see which model_mask performs best.
# Perfect reconstruction means:
# 1. mean_recon close to mean_real -> Mean Absolute Error (MAE) of means = |mean_real - mean_recon|
# 2. variance_ratio close to 1.0 -> absolute log variance ratio = |log(variance_ratio)|
# 3. ba_robust_slope close to 0.0 -> |ba_robust_slope|

df['mean_error'] = (df['mean_real'] - df['mean_recon']).abs()
df['var_ratio_error'] = np.abs(np.log(df['variance_ratio'].replace(0, np.nan)))
df['slope_error'] = df['ba_robust_slope'].abs()

# Group by model mask and feature
summary = df.groupby(['model_mask', 'clinical_feature']).agg({
    'mean_error': 'mean',
    'var_ratio_error': 'mean',
    'slope_error': 'mean',
    'n_beats': 'sum'
}).reset_index()

# Overall model performance across all features
overall = df.groupby('model_mask').agg({
    'mean_error': 'mean',
    'var_ratio_error': 'mean',
    'slope_error': 'mean'
}).reset_index()

print("OVERALL MODEL PERFORMANCE")
print("=========================")
print(overall.to_string(index=False))

print("\n\nPERFORMANCE BY FEATURE")
print("=========================")
# Pivot to show mean error by feature
pivot_mean = summary.pivot(index='model_mask', columns='clinical_feature', values='mean_error')
print(pivot_mean.to_string())

