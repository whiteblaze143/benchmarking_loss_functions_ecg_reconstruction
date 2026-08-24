import pandas as pd

df = pd.read_csv("results/sunnybrook_eval/all_models_sunnybrook_clinical.csv")
cols = df.columns
err_cols = [c for c in cols if '_err_' in c]

# Compute absolute errors for each row
df['MAE'] = df[err_cols].abs().mean(axis=1)

# Group by model and compute mean across all ECGs
model_mae = df.groupby('model')['MAE'].mean().sort_values()
print("Aggregate Mean Absolute Errors (Sunnybrook):")
print(model_mae.to_string())

# Also let's check QT interval specifically, as it's often the hardest
if 'ep1_err_global_meanqtint' in err_cols:
    df['QT_MAE'] = df['ep1_err_global_meanqtint'].abs()
    model_qt_mae = df.groupby('model')['QT_MAE'].mean().sort_values()
    print("\nQT Interval MAE (Sunnybrook):")
    print(model_qt_mae.head(10).to_string())
