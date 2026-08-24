#!/usr/bin/env python3
import sys
from pathlib import Path
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import glob
from tqdm import tqdm

import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from scripts.bootstrap_paths import setup_import_paths
setup_import_paths()
# Gaussian Negative Log Likelihood Loss
class GaussianNLLLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        # pred is [B, 24, L] -> first 12 is mean, next 12 is log_var
        mean = pred[:, :12, :]
        log_var = pred[:, 12:, :]
        
        # log_var is unconstrained, so var = exp(log_var)
        var = torch.exp(log_var) + self.eps
        
        # NLL = 0.5 * log(var) + 0.5 * (target - mean)^2 / var
        loss = 0.5 * log_var + 0.5 * (target - mean)**2 / var
        return loss.mean()

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=13, stride=1, padding='same'):
        super().__init__()
        if padding == 'same':
            pad = kernel_size // 2
        else:
            pad = 0
            
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=pad)
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=pad)
        self.layer_norm = nn.GroupNorm(1, out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size//2)
        self.instance_norm = nn.InstanceNorm1d(out_channels, affine=True)
        # Add MC Dropout
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x1 = F.gelu(self.conv1(x))
        x2 = self.conv2(x)
        x2 = self.layer_norm(x2)
        x2 = F.gelu(x2)
        x3 = F.gelu(self.conv3(x2))
        
        if x1.shape[-1] != x3.shape[-1]:
            x3 = F.interpolate(x3, size=x1.shape[-1], mode='linear', align_corners=False)
            
        out = self.instance_norm(x1 + x3)
        return self.dropout(out)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=13, stride=1, padding='same'):
        super().__init__()
        if padding == 'same':
            pad = kernel_size // 2
            out_pad = 0 if stride == 1 else 1
        else:
            pad = 0
            out_pad = 0
            
        self.deconv1 = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, output_padding=out_pad)
        self.deconv2 = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, output_padding=out_pad)
        self.layer_norm = nn.GroupNorm(1, out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size//2)
        self.instance_norm = nn.InstanceNorm1d(out_channels, affine=True)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x1 = F.gelu(self.deconv1(x))
        x2 = self.deconv2(x)
        x2 = self.layer_norm(x2)
        x2 = F.gelu(x2)
        x3 = F.gelu(self.conv3(x2))
        
        if x1.shape[-1] != x3.shape[-1]:
            x3 = F.interpolate(x3, size=x1.shape[-1], mode='linear', align_corners=False)
            
        out = self.instance_norm(x1 + x3)
        return self.dropout(out)

class UncertaintyMCMAModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=12, kernel_size=13):
        super().__init__()
        filters = [16, 32, 64, 128, 256, 512]
        
        # Encoder
        self.down0 = DownBlock(in_channels, filters[0], kernel_size, stride=1)
        self.down1 = DownBlock(filters[0], filters[1], kernel_size, stride=2)
        self.down2 = DownBlock(filters[1], filters[2], kernel_size, stride=2)
        self.down3 = DownBlock(filters[2], filters[3], kernel_size, stride=2)
        self.down4 = DownBlock(filters[3], filters[4], kernel_size, stride=2)
        self.down5 = DownBlock(filters[4], filters[5], kernel_size, stride=2)
        
        # Decoder
        self.up4 = UpBlock(filters[5], filters[4], kernel_size, stride=2)
        self.d_conv4 = DownBlock(filters[4]*2, filters[4], kernel_size, stride=1)
        
        self.up3 = UpBlock(filters[4], filters[3], kernel_size, stride=2)
        self.d_conv3 = DownBlock(filters[3]*2, filters[3], kernel_size, stride=1)
        
        self.up2 = UpBlock(filters[3], filters[2], kernel_size, stride=2)
        self.d_conv2 = DownBlock(filters[2]*2, filters[2], kernel_size, stride=1)
        
        self.up1 = UpBlock(filters[2], filters[1], kernel_size, stride=2)
        self.d_conv1 = DownBlock(filters[1]*2, filters[1], kernel_size, stride=1)
        
        self.up0 = UpBlock(filters[1], filters[0], kernel_size, stride=2)
        self.d_conv0 = DownBlock(filters[0]*2, filters[0], kernel_size, stride=1)
        
        # Predicts mean (first out_channels) and log_var (last out_channels)
        self.final_conv = nn.Conv1d(filters[0], out_channels * 2, kernel_size=1)

    def forward(self, x):
        e0 = self.down0(x)
        e1 = self.down1(e0)
        e2 = self.down2(e1)
        e3 = self.down3(e2)
        e4 = self.down4(e3)
        e5 = self.down5(e4)
        
        d4 = self.up4(e5)
        d4 = self._match_size(d4, e4)
        d4 = self.d_conv4(torch.cat([d4, e4], dim=1))
        
        d3 = self.up3(d4)
        d3 = self._match_size(d3, e3)
        d3 = self.d_conv3(torch.cat([d3, e3], dim=1))
        
        d2 = self.up2(d3)
        d2 = self._match_size(d2, e2)
        d2 = self.d_conv2(torch.cat([d2, e2], dim=1))
        
        d1 = self.up1(d2)
        d1 = self._match_size(d1, e1)
        d1 = self.d_conv1(torch.cat([d1, e1], dim=1))
        
        d0 = self.up0(d1)
        d0 = self._match_size(d0, e0)
        d0 = self.d_conv0(torch.cat([d0, e0], dim=1))
        
        out = self.final_conv(d0)
        return out
        
    def _match_size(self, x, target):
        if x.shape[-1] != target.shape[-1]:
            x = F.interpolate(x, size=target.shape[-1], mode='linear', align_corners=False)
        return x

class PTBXLDataset(Dataset):
    def __init__(self, data_dir):
        self.files = glob.glob(os.path.join(data_dir, "*.pt"))
        self.input_indices = [0, 1, 7]
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        tensor = torch.load(self.files[idx])
        pad = torch.zeros(12, 120)
        tensor = torch.cat([tensor, pad], dim=1)
        x = tensor[self.input_indices, :]
        y = tensor
        return x, y

import argparse
import wandb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--disable_temporal_attn", action="store_true")
    parser.add_argument("--disable_cross_lead_attn", action="store_true")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/M2_uncertainty_seed42.pt")
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    ckpt_dir = "/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    
    data_dir = "/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/data/ptbxl_tensors"
    print("Loading datasets...")
    train_dataset = PTBXLDataset(f"{data_dir}/train")
    val_dataset = PTBXLDataset(f"{data_dir}/val")
    
    if args.test:
        print("Test mode: using 100 samples")
        train_dataset.files = train_dataset.files[:100]
        val_dataset.files = val_dataset.files[:20]
        epochs = 1
    else:
        epochs = 10
        
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    wandb.init(project="ecg-reconstruction-ablation", name=args.run_name, config=vars(args))
    
    model = UncertaintyMCMAModel(in_channels=3, out_channels=12).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = GaussianNLLLoss()
    
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        print(f"Epoch {epoch+1}/{epochs}")
        for x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item()
                
        t_loss = train_loss / len(train_loader)
        v_loss = val_loss / len(val_loader)
        print(f"Train NLL Loss: {t_loss:.4f} | Val NLL Loss: {v_loss:.4f}")
        
        wandb.log({
            "epoch": epoch + 1,
            "train_nll_loss": t_loss,
            "val_nll_loss": v_loss
        })
        
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            checkpoint_path = os.path.abspath(args.checkpoint_path)
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved new best checkpoint: {checkpoint_path}")

    wandb.finish()

if __name__ == "__main__":
    main()
