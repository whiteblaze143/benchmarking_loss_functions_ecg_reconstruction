import ast
import json
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import wfdb

project_root = Path("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction")

print("==================================================")
print(" 1. LUDB DATASET EDA")
print("==================================================")
ludb_dir = project_root / "data/ludb"
dat_files = sorted(list(set([p.stem for p in ludb_dir.glob("*.dat")])))
print(f"Total LUDB records: {len(dat_files)}")

if len(dat_files) > 0:
    rec_id = dat_files[0]
    rec = wfdb.rdrecord(str(ludb_dir / rec_id))
    print(f"LUDB record {rec_id} signal shape: {rec.p_signal.shape}, fs: {rec.fs}, sig_name: {rec.sig_name}")
    
    # Check all extension annotations available
    exts = [p.suffix[1:] for p in ludb_dir.glob(f"{rec_id}.*") if p.suffix not in ['.dat', '.hea']]
    print(f"Annotation extensions for {rec_id}: {exts}")
    
    # Inspect symbols in lead 'ii' or 'i'
    for ext in ['i', 'ii', 'v1', 'atr', 'ann'][:3]:
        ann_path = ludb_dir / f"{rec_id}.{ext}"
        if ann_path.exists():
            ann = wfdb.rdann(str(ludb_dir / rec_id), extension=ext)
            symbols = np.array(ann.symbol)
            samples = np.array(ann.sample)
            print(f"Lead/Extension '{ext}' symbols count: {len(symbols)}")
            print(f"Unique symbols in '{ext}': {set(symbols)}")
            print(f"Sample symbol-sample pairs (first 15): {list(zip(symbols[:15], samples[:15]))}")

print("\n==================================================")
print(" 2. ISP DELINEATION DATASET EDA")
print("==================================================")
isp_dir = project_root / "data/isp/isp_delineation_dataset"
if not isp_dir.exists():
    isp_dir = project_root / "data/isp_delineation_dataset"

csv_path = isp_dir / "train_isp_delineation_data.csv"
if csv_path.exists():
    df_isp = pd.read_csv(csv_path)
    print(f"ISP dataframe shape: {df_isp.shape}")
    print("ISP columns:", df_isp.columns.tolist())
    print("Sample rows:")
    print(df_isp.head(3))
    
    # Extract unique labels in target column
    all_labels = set()
    for _, row in df_isp.head(100).iterrows():
        try:
            tuples_list = ast.literal_eval(str(row['target']))
            for t in tuples_list:
                if len(t) >= 3:
                    all_labels.add(t[2])
        except Exception as e:
            pass
    print(f"Unique labels in target column (sample 100): {all_labels}")

print("\n==================================================")
print(" 3. ZHEJIANG DATASET EDA")
print("==================================================")
zhe_dir = project_root / "data/zhejiang"
zhe_ecg_dir = zhe_dir / "ecg"
zhe_label_dir = zhe_dir / "label"

ecg_pkls = list(zhe_ecg_dir.glob("*.pkl"))
label_pkls = list(zhe_label_dir.glob("*.pkl"))
print(f"Zhejiang ECG pkl count: {len(ecg_pkls)}, Label pkl count: {len(label_pkls)}")

if len(label_pkls) > 0:
    sample_lbl = label_pkls[0]
    with open(sample_lbl, "rb") as f:
        lbl_data = pickle.load(f)
    lbl_arr = np.asarray(lbl_data)
    print(f"Sample Zhejiang label shape: {lbl_arr.shape}, dtype: {lbl_arr.dtype}")
    print(f"Unique values in Zhejiang label: {np.unique(lbl_arr)}")

    # Check signal files for this record
    rec_id = sample_lbl.stem
    lead_files = list(zhe_ecg_dir.glob(f"{rec_id}_*.pkl"))
    print(f"Leads available for record {rec_id}: {[p.stem for p in lead_files]}")
    if len(lead_files) > 0:
        with open(lead_files[0], "rb") as f:
            sig = pickle.load(f)
        sig_arr = np.asarray(sig)
        print(f"Sample signal shape: {sig_arr.shape}, dtype: {sig_arr.dtype}, min: {sig_arr.min()}, max: {sig_arr.max()}, mean: {sig_arr.mean():.2f}")

print("\nEDA Completed.")
