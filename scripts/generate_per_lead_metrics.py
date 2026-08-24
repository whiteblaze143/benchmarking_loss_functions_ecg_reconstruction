#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from scripts.bootstrap_paths import setup_import_paths
setup_import_paths()
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error


from src.reconstruction.learn_functions.hubert_bridge import HuBERTBridge
from src.reconstruction.learn_functions.fam_ecg import UniversalSpatialFusionAdapter
from src.data.ptbxl_dataset import PTBXLDataset

# Config
CHECKPOINT_PATH = "checkpoints/hpo_hubert/trial_8/best.pt"
DATA_DIR = "data/ptb_xl"
DATABASE_CSV = "data/ptb_xl/ptbxl_database.csv"
LEAD_INDICES = [0, 1, 8] # Mason-Likar (I, II, V3) logic from HPO script
LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Model
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    if not os.path.exists(CHECKPOINT_PATH):
        print("Checkpoint not found!")
        sys.exit(1)
        
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    
    # Init Model
    bridge = HuBERTBridge(
        model_name="Edoardo-BS/hubert-ecg-large", 
        freeze_encoder=True,
        physics_projection=True, 
        use_res_decoder=True, 
        target_len=5000
    ).to(device)
    
    adapter = UniversalSpatialFusionAdapter(dim=bridge.embed_dim).to(device)
    
    bridge.decoder.load_state_dict(checkpoint['decoder_state_dict'])
    adapter.load_state_dict(checkpoint['adapter_state_dict'])
    
    bridge.eval()
    adapter.eval()
    
    # 2. Load Data
    test_ds = PTBXLDataset(DATA_DIR, DATABASE_CSV, split='test', target_fs=500)
    print(f"Test Set Size: {len(test_ds)}")
    
    # 3. Inference Loop
    all_gt = []
    all_pred = []
    
    print("Running inference...")
    with torch.no_grad():
        for i in tqdm(range(len(test_ds))):
            _, gt = test_ds[i] # gt: (12, 5000)
            
            # Prepare Input (Subset leads)
            x_in = gt[LEAD_INDICES, :].unsqueeze(0).to(device) # (1, 3, 5000)
            
            # Forward
            with torch.amp.autocast('cuda'):
                recon = bridge(x_in, lead_indices=LEAD_INDICES, adapter=adapter)
                
            all_gt.append(gt.numpy())
            all_pred.append(recon.squeeze(0).float().cpu().numpy())
            
    # Stack: (N, 12, 5000)
    all_gt = np.array(all_gt)
    all_pred = np.array(all_pred)
    
    # 4. Compute Metrics Per Lead
    results = []
    
    print("\nComputing per-lead metrics...")
    for l_idx, lead_name in enumerate(LEAD_NAMES):
        gt_l = all_gt[:, l_idx, :].flatten()
        pred_l = all_pred[:, l_idx, :].flatten()
        
        mse = mean_squared_error(gt_l, pred_l)
        mae = mean_absolute_error(gt_l, pred_l)
        
        # Pearson per record then average (more robust than flattened)
        corrs = []
        for i in range(len(all_gt)):
            c, _ = pearsonr(all_gt[i, l_idx], all_pred[i, l_idx])
            if not np.isnan(c):
                corrs.append(c)
        mean_corr = np.mean(corrs)
        
        results.append({
            "Lead": lead_name,
            "MSE": mse,
            "MAE": mae,
            "Correlation": mean_corr
        })
        
    df_results = pd.DataFrame(results)
    
    # Save CSV first
    df_results.to_csv("metrics_per_lead.csv", index=False)
    print("Saved metrics to metrics_per_lead.csv")
    
    # 5. Render Table (Markdown/LaTeX friendly)
    try:
        print("\n--- Per-Lead Analysis (WearECG Style) ---")
        print(df_results.to_markdown(index=False, floatfmt=".4f"))
    except ImportError:
        print("Tabulate not installed, skipping markdown print.")
        print(df_results)
    
    # LateX Format print
    print("\nLaTeX Table Body:")
    for _, row in df_results.iterrows():
        print(f"{row['Lead']} & {row['MSE']:.4f} & {row['MAE']:.4f} & {row['Correlation']:.4f} \\\\")
        
    # 6. Save Boxplot of Correlations
    # We need separate list of correlations for boxplot
    lead_corrs = {l: [] for l in LEAD_NAMES}
    for i in range(len(all_gt)):
        for l_idx, lead_name in enumerate(LEAD_NAMES):
            c, _ = pearsonr(all_gt[i, l_idx], all_pred[i, l_idx])
            if not np.isnan(c):
                lead_corrs[lead_name].append(c)
                
    plt.figure(figsize=(12, 6))
    
    # Prepare data for sns boxplot
    plot_data = []
    for l in LEAD_NAMES:
        for val in lead_corrs[l]:
            plot_data.append({"Lead": l, "Correlation": val})
    df_plot = pd.DataFrame(plot_data)
    
    sns.set_theme(style="whitegrid")
    sns.boxplot(x="Lead", y="Correlation", data=df_plot, showfliers=False, palette="vlag")
    plt.title("Reconstruction Fidelity by Lead (Pearson Correlation)")
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig("figures/per_lead_performance.png", dpi=300)
    print("\nSaved figure to figures/per_lead_performance.png")

if __name__ == "__main__":
    main()
