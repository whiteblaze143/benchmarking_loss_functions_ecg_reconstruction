import pandas as pd
import numpy as np
from pathlib import Path

csv_path = Path("results/sunnybrook_eval/all_models_sunnybrook_clinical.csv")
df = pd.read_csv(csv_path)

out_md = Path("results/sunnybrook_eval/SUNNYBROOK_CLINICAL_ANALYSIS.md")

lines = []
lines.append("# 📊 Sunnybrook Clinical Evaluation & Audit Report")
lines.append("")
lines.append(f"**Total Records Evaluated**: {len(df)}")
lines.append(f"**Unique Models**: {df['model'].nunique()}")
lines.append(f"**Dataset**: Sunnybrook 12-lead ECG Clinical Hold-out ($n=20$ XMLs)")
lines.append("")

# Group by architecture / loss
df["arch"] = df["model"].apply(lambda x: x.split("__")[0])
df["config"] = df["model"].apply(lambda x: x.split("__")[1] if "__" in x else "")

lines.append("## 1. Primary Endpoint Comparison: Reconstruction Fidelity (Recon Open vs Real Open)")
lines.append("Evaluates how accurately reconstructed waveforms reproduce open-source fiducial features compared to ground-truth raw waveforms.")
lines.append("")

features = ["V3_ramp", "V3_tamp", "V3_samp", "V3_st80", "V6_ramp", "V6_tamp", "V6_samp", "V6_st80", "global_meanqrsdur", "global_meanqtint"]

lines.append("| Feature | Unit | Mean Bias | SD | MAE | 95% LoA Lower | 95% LoA Upper | Valid N |")
lines.append("|---|---|---|---|---|---|---|---|")

for feat in features:
    col = f"ep1_err_{feat}"
    if col in df.columns:
        s = df[col].dropna()
        if len(s) > 0:
            bias = s.mean()
            std = s.std()
            mae = s.abs().mean()
            loa_low = bias - 1.96 * std
            loa_high = bias + 1.96 * std
            unit = "ms" if "dur" in feat or "int" in feat else "µV"
            lines.append(f"| `{feat}` | {unit} | {bias:+.2f} | {std:.2f} | {mae:.2f} | {loa_low:+.2f} | {loa_high:+.2f} | {len(s)} |")

lines.append("")
lines.append("## 2. Secondary Endpoint Comparison: Open Extraction vs Philips Vendor XLI Reference")
lines.append("Evaluates baseline agreement between open-source `neurokit2` extraction and proprietary Philips XLI machine measurements on ground-truth signals.")
lines.append("")

lines.append("| Feature | Unit | Mean Bias | SD | MAE | 95% LoA Lower | 95% LoA Upper | Valid N |")
lines.append("|---|---|---|---|---|---|---|---|")

for feat in features:
    col = f"ep2_err_{feat}"
    if col in df.columns:
        s = df[col].dropna()
        if len(s) > 0:
            bias = s.mean()
            std = s.std()
            mae = s.abs().mean()
            loa_low = bias - 1.96 * std
            loa_high = bias + 1.96 * std
            unit = "ms" if "dur" in feat or "int" in feat else "µV"
            lines.append(f"| `{feat}` | {unit} | {bias:+.2f} | {std:.2f} | {mae:.2f} | {loa_low:+.2f} | {loa_high:+.2f} | {len(s)} |")

lines.append("")
lines.append("## 3. Per-Architecture Breakdown (Reconstruction MAE on Precordial Fiducials)")
lines.append("")

arch_df = df.groupby("arch").agg(
    v3_r_mae=("ep1_err_V3_ramp", lambda x: x.abs().mean()),
    v3_t_mae=("ep1_err_V3_tamp", lambda x: x.abs().mean()),
    v6_r_mae=("ep1_err_V6_ramp", lambda x: x.abs().mean()),
    qrs_dur_mae=("ep1_err_global_meanqrsdur", lambda x: x.abs().mean()),
    qt_int_mae=("ep1_err_global_meanqtint", lambda x: x.abs().mean())
).reset_index()

lines.append("| Architecture | V3 R-Amp MAE (µV) | V3 T-Amp MAE (µV) | V6 R-Amp MAE (µV) | QRS Dur MAE (ms) | QT Int MAE (ms) |")
lines.append("|---|---|---|---|---|---|")
for _, r in arch_df.iterrows():
    lines.append(f"| **{r['arch'].upper()}** | {r['v3_r_mae']:.2f} | {r['v3_t_mae']:.2f} | {r['v6_r_mae']:.2f} | {r['qrs_dur_mae']:.2f} | {r['qt_int_mae']:.2f} |")

out_md.write_text("\n".join(lines))
print(f"Saved analysis report to {out_md}")
