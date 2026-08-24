import pandas as pd
import numpy as np

df = pd.read_csv("results/factorial_v4/temporal_mmd_evaluation.csv")
df['mean_error'] = (df['mean_real'] - df['mean_recon']).abs()
df['var_ratio_error'] = np.abs(np.log(df['variance_ratio'].replace(0, np.nan)))

# 1. Feature level analysis across top vs worst models
# Let's get top 3 and worst 3 models
summary = df.groupby('model_mask')['mean_error'].mean().sort_values()
top_models = summary.head(3).index.tolist()
worst_models = summary.tail(3).index.tolist()

print("TOP MODELS:", top_models)
print("WORST MODELS:", worst_models)

# 2. Lead-specific performance for top models vs baseline (e.g. 1000000)
# Look at V1 vs V6, Lead II
for m in top_models + worst_models + [1000000]:
    m_df = df[df['model_mask'] == m]
    lead_summary = m_df.groupby('lead')['mean_error'].mean()
    feature_summary = m_df.groupby('clinical_feature')['mean_error'].mean()
    print(f"\nModel {m} Lead Summary:\n", lead_summary.to_dict())
    print(f"Model {m} Feature Summary:\n", feature_summary.to_dict())
