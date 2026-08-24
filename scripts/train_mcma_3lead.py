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
# PyTorch translation of MCMA (UNet3+) from Fitbit Clinical Dashboard
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=13, stride=1, padding='same'):
        super().__init__()
        # PyTorch Conv1d padding='same' requires stride=1
        if padding == 'same':
            pad = kernel_size // 2
        else:
            pad = 0
            
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=pad)
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=pad)
        self.layer_norm = nn.GroupNorm(1, out_channels) # Equivalent to LayerNorm for (C, L)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size//2)
        self.instance_norm = nn.InstanceNorm1d(out_channels, affine=True)

    def forward(self, x):
        x1 = F.gelu(self.conv1(x))
        x2 = self.conv2(x)
        x2 = self.layer_norm(x2)
        x2 = F.gelu(x2)
        x3 = F.gelu(self.conv3(x2))
        
        if x1.shape[-1] != x3.shape[-1]:
            x3 = F.interpolate(x3, size=x1.shape[-1], mode='linear', align_corners=False)
            
        return self.instance_norm(x1 + x3)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=13, stride=1, padding='same'):
        super().__init__()
        if padding == 'same':
            pad = kernel_size // 2
            out_pad = 0 if stride == 1 else 1 # Adjust for stride=2
        else:
            pad = 0
            out_pad = 0
            
        self.deconv1 = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, output_padding=out_pad)
        self.deconv2 = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, output_padding=out_pad)
        self.layer_norm = nn.GroupNorm(1, out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size//2)
        self.instance_norm = nn.InstanceNorm1d(out_channels, affine=True)

    def forward(self, x):
        x1 = F.gelu(self.deconv1(x))
        x2 = self.deconv2(x)
        x2 = self.layer_norm(x2)
        x2 = F.gelu(x2)
        x3 = F.gelu(self.conv3(x2))
        
        # Ensure sizes match in case of parity issues with stride
        if x1.shape[-1] != x3.shape[-1]:
            x3 = F.interpolate(x3, size=x1.shape[-1], mode='linear', align_corners=False)
            
        return self.instance_norm(x1 + x3)

class MCMAModel(nn.Module):
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
        
        self.final_conv = nn.Conv1d(filters[0], out_channels, kernel_size=1)

    def forward(self, x):
        torch.backends.cudnn.enabled = False
        e0 = self.down0(x)
        e1 = self.down1(e0)
        e2 = self.down2(e1)
        e3 = self.down3(e2)
        e4 = self.down4(e3)
        e5 = self.down5(e4)
        
        # Up 4
        d4 = self.up4(e5)
        d4 = self._match_size(d4, e4)
        d4 = self.d_conv4(torch.cat([d4, e4], dim=1))
        
        # Up 3
        d3 = self.up3(d4)
        d3 = self._match_size(d3, e3)
        d3 = self.d_conv3(torch.cat([d3, e3], dim=1))
        
        # Up 2
        d2 = self.up2(d3)
        d2 = self._match_size(d2, e2)
        d2 = self.d_conv2(torch.cat([d2, e2], dim=1))
        
        # Up 1
        d1 = self.up1(d2)
        d1 = self._match_size(d1, e1)
        d1 = self.d_conv1(torch.cat([d1, e1], dim=1))
        
        # Up 0
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
        self.files = sorted(
            glob.glob(os.path.join(data_dir, "*.pt")),
            key=lambda value: int(Path(value).stem),
        )
        # Input leads for this project: I, II, V2 (indices 0, 1, 7)
        self.input_indices = [0, 1, 7]
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        tensor = torch.load(self.files[idx], weights_only=True)
        # tensor shape: (12, 5000)
        # Pad to 5120 for UNet 5-level pooling (needs multiple of 32)
        pad = torch.zeros(12, 120)
        tensor = torch.cat([tensor, pad], dim=1) # (12, 5120)
        
        # Extract 3 leads
        x = tensor[self.input_indices, :] # (3, 5120)
        y = tensor # (12, 5120)
        return x, y

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    ckpt_dir = "/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    
    data_dir = "/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/data/ptbxl_tensors"
    print("Loading datasets...")
    train_dataset = PTBXLDataset(f"{data_dir}/train")
    val_dataset = PTBXLDataset(f"{data_dir}/val")
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    model = MCMAModel(in_channels=3, out_channels=12).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    epochs = 10 
    
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
        print(f"Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f}")
        
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            # Save MCMA checkpoint mimicking the M0 format to allow evaluation script to use it natively
            torch.save(model.state_dict(), f"{ckpt_dir}/M0_seed42.pt")
            print("Saved new best checkpoint.")

if __name__ == "__main__":
    main()
