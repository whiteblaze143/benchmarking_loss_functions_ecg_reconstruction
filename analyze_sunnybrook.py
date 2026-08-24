import pandas as pd

df = pd.read_csv("results/sunnybrook_eval/all_models_sunnybrook_clinical.csv")

# Filter only interesting models for summary
cols = df.columns
metrics = [c for c in cols if 'err' in c or 'recon' in c]

# Print mean absolute errors for each model
print("Mean Absolute Errors (Sunnybrook):")
summary = []
for idx, row in df.iterrows():
    model = row['model']
    # compute average of absolute errors
    err_cols = [c for c in cols if '_err_' in c]
    if len(err_cols) > 0:
        mae = row[err_cols].abs().mean()
        summary.append({'model': model, 'MAE': mae})

summary_df = pd.DataFrame(summary).sort_values('MAE')
print(summary_df.head(20).to_string())
