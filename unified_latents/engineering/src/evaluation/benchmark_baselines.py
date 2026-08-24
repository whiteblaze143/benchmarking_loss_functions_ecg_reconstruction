
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add project root and third party paths
_engineering_root = Path(__file__).resolve().parents[2]
if str(_engineering_root) not in sys.path:
    sys.path.insert(0, str(_engineering_root))
_cnvae_path = _engineering_root / "third_party" / "cNVAE_ECG-main" / "conditional"
if _cnvae_path.exists() and str(_cnvae_path) not in sys.path:
    sys.path.append(str(_cnvae_path))

from src.data.multi_source_dataset import MultiSourceECGDataset
from src.training.train_mason import MasonReconstructor
from src.models.legacy_mason import LegacyMasonReconstructor

# Device configuration (CPU preference to avoid OOM)
device = torch.device('cpu') 
print(f"Using device: {device}")

# cNVAE Wrapper (Copied from Notebook)
class CNVAEWrapper(nn.Module):
    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = device
        self.expected_channels = model.num_input_channels 
        self.lead_indices = [0, 1, 6, 7, 8, 9, 10, 11]

    def forward(self, x, metadata):
        if self.expected_channels == 3:
            input_leads = x[:, [0, 1, 7], :]
        else:
            input_leads = x[:, self.lead_indices, :] if self.expected_channels == 8 else x

        batch_size = x.shape[0]
        class_vec = torch.ones(batch_size, 1, device=x.device)
        logits, _, _, _, _ = self.model(input_leads, class_vec)
        dist = self.model.decoder_output(logits)
        recon_small = dist.mean 
        recon = torch.zeros_like(x)
        if recon_small.shape[1] == 3:
            recon[:, 0, :] = recon_small[:, 0, :]
            recon[:, 1, :] = recon_small[:, 1, :]
            recon[:, 7, :] = recon_small[:, 2, :]
            recon[:, 2, :] = recon[:, 1, :] - recon[:, 0, :]
            recon[:, 3, :] = -(recon[:, 0, :] + recon[:, 1, :])/2
            recon[:, 4, :] = recon[:, 0, :] - recon[:, 1, :]/2
            recon[:, 5, :] = recon[:, 1, :] - recon[:, 0, :]/2
        elif recon_small.shape[1] == 12:
             recon = recon_small
        else:
             ch = recon_small.shape[1]
             recon[:, :ch, :] = recon_small
        return recon

class MockArgs:
    pass

def load_checkpoints(mason_path, cnvae_path, device):
    models = {}
    
    # 1. Load Mason (Trying Legacy first based on previous debugs)
    if os.path.exists(mason_path):
        try:
            print("Loading Mason with LegacyMasonReconstructor...")
            state = torch.load(mason_path, map_location=device)
            model_mason = LegacyMasonReconstructor(input_lead_num=3, output_lead_num=12).to(device)
            model_mason.load_state_dict(state['model_state_dict']) 
            model_mason.eval()
            models['Mason'] = model_mason
            print(f"✅ Mason Checkpoint Loaded (Legacy)")
        except Exception as e:
            print(f"Legacy loading failed, trying Standard Mason: {e}")
            try:
                model_mason = MasonReconstructor(in_leads=3, out_leads=12).to(device)
                model_mason.load_state_dict(state['model_state_dict'])    
                model_mason.eval()
                models['Mason'] = model_mason
                print(f"✅ Mason Checkpoint Loaded (Standard)")
            except Exception as e2:
                print(f"❌ Mason Load Failed: {e2}")

    # 2. Check cNVAE
    if os.path.exists(cnvae_path):
         try:
             state = torch.load(cnvae_path, map_location=device)
             from model_conditional_1d import AutoEncoder
             from utils import get_arch_cells
             
             # Mock Args Setup
             mock_args = MockArgs()
             mock_args.data_dir = "data/ptbxl_tensors"
             mock_args.num_input_channels = 8 # Need 8 for init, but 3 for stem patch
             mock_args.num_channels_enc = 32
             mock_args.num_channels_dec = 32
             mock_args.num_latent_scales = 3 
             mock_args.num_groups_per_scale = 5  # Updated
             mock_args.num_latent_per_group = 10 # Updated
             mock_args.ada_groups = False
             mock_args.min_groups_per_scale = 1
             mock_args.num_preprocess_blocks = 2 # Updated
             mock_args.num_preprocess_cells = 3
             mock_args.num_cell_per_cond_enc = 1
             mock_args.num_postprocess_blocks = 2 # Updated
             mock_args.num_postprocess_cells = 3
             mock_args.num_cell_per_cond_dec = 1
             mock_args.input_size = 5000
             mock_args.num_mixture_dec = 10
             mock_args.num_nf = 0
             mock_args.num_x_bits = 8
             mock_args.use_se = False
             mock_args.res_dist = False
             mock_args.focal = False
             
             official_model = AutoEncoder(mock_args, writer=None, arch_instance=get_arch_cells('res_bnswish'), num_classes=1)
             from third_party.cNVAE_ECG.conditional.neural_operations_1d import Conv1D
             official_model.stem = Conv1D(3, 32, 3, padding=1, bias=True).to(device)
             
             if 'state_dict' in state: sd = state['state_dict']
             else: sd = state
             new_sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
             
             official_model.load_state_dict(new_sd, strict=False)
             model_cnvae = CNVAEWrapper(official_model, device).to(device)
             model_cnvae.eval()
             models['cNVAE'] = model_cnvae
             print(f"✅ cNVAE Checkpoint Loaded")
         except Exception as e:
             print(f"⚠️ cNVAE Load Failed: {e}")
             import traceback
             traceback.print_exc()

    return models

def evaluate():
    data_path = os.path.join(project_root, "data/ptbxl_tensors")
    sources = [{"name": "PTB-XL", "path": data_path, "format": "pt"}]
    test_ds = MultiSourceECGDataset(split='val', sources=sources, target_len=5000, normalization='min_max')
    loader = torch.utils.data.DataLoader(test_ds, batch_size=16, shuffle=False) # Small batch for CPU
    print(f"Dataset: {len(test_ds)} samples")

    mason_ckpt = os.path.join(project_root, "checkpoints/mason_baseline/best_mason.pt")
    cnvae_ckpt = os.path.join(project_root, "checkpoints/cnvae_v1/best_cnvae.pt")
    models = load_checkpoints(mason_ckpt, cnvae_ckpt, device)
    
    results = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Benchmarking"):
            x = batch['input'].to(device)
            y = batch['target'].to(device)
            
            # Simple metadata mock if missing
            metadata = {'age_years': torch.zeros(x.size(0)), 'sex_code': torch.zeros(x.size(0))}
            
            for name, model in models.items():
                try:
                    if name == 'cNVAE':
                        pred = model(x, metadata)
                    else:
                        pred = model(x)
                    
                    # Compute simple metrics
                    mse = torch.mean((pred - y)**2).item()
                    rmse = np.sqrt(mse)
                    
                    # Pearson per sample
                    vx = pred - torch.mean(pred, dim=2, keepdim=True)
                    vy = y - torch.mean(y, dim=2, keepdim=True)
                    cov = torch.sum(vx * vy, dim=2)
                    denom = torch.sqrt(torch.sum(vx ** 2, dim=2)) * torch.sqrt(torch.sum(vy ** 2, dim=2)) + 1e-6
                    pearson = torch.mean(cov / denom).item()
                    
                    results.append({'Model': name, 'MSE': mse, 'RMSE': rmse, 'Pearson': pearson})
                except Exception as e:
                    print(f"Error evaluating {name}: {e}")

    df = pd.DataFrame(results)
    if not df.empty:
        print("\n\n=== FINAL BENCHMARK RESULTS ===")
        print(df.groupby('Model').mean())
        df.groupby('Model').mean().to_csv(os.path.join(project_root, "results/baseline_benchmark.csv"))
    else:
        print("No results generated.")

if __name__ == "__main__":
    evaluate()
