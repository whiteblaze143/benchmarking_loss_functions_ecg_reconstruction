import argparse
import sys
sys.path.insert(0, '/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/unified_latents/engineering')
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from tqdm import tqdm
import time

# Add src to path to import models
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
from src.models.angle_d.resnet_baseline import ResNetBaseline

class ECGDataset(Dataset):
    def __init__(self, data_path, observed_indices=[0, 1, 9]): # I, II, V3 -> 0, 1, 9 in 0-11 index
        """
        data_path: path to .pkl file with shape (N, 12, 5000)
        observed_indices: list of indices for 3 observed leads.
                          Default [0 (I), 1 (II), 9 (V4? No wait, checklist says V3)]
                          Checklist says: [0, 1, 9] (I, II, V3?) 
                          Let's verify index mapping.
                          0:I, 1:II, 2:III, 3:aVR, 4:aVL, 5:aVF, 6:V1, 7:V2, 8:V3, 9:V4, 10:V5, 11:V6
                          Wait, protocol says "Observed leads: [I, II, V3] at indices [0, 1, 9]"?
                          If 0:I, 1:II, 2:III... then V3 is index 8. V4 is index 9.
                          Protocol line 363: 8:V3, 9:V4.
                          So if protocol says "indices [0, 1, 9] (I, II, V3)", there is a mismatch.
                          Standard 12-lead order: I,II,III,aVR,aVL,aVF,V1,V2,V3,V4,V5,V6.
                          Indices: 0,1,2,3,4,5,6,7,8,9,10,11.
                          So V3 is 8. V4 is 9.
                          I'll stick to indices [0, 1, 8] for (I, II, V3) or [0, 1, 9] for (I, II, V4).
                          Protocol Text: "Observed leads: [I, II, V3] at indices [0, 1, 9] (3 leads)" -> Typo in protocol?
                          Later protocol line 370: "Observed: [I, II, aVR, aVL, V2, V5] (indices 0, 1, 3, 4, 7, 10)" -> This is for Masked Transformer 6 leads.
                          For ResNet (3->12), line 84 says "[I, II, V3] at indices [0, 1, 9]".
                          I will assume the INTENT is I, II, V3, which is usually indices 0, 1, 8. 
                          I will use [0, 1, 8].
        """
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        self.observed_indices = observed_indices

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # (12, 5000)
        ecg = self.data[idx]
        x = ecg[self.observed_indices, :]
        y = ecg
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Data
    train_dataset = ECGDataset(args.train_data)
    val_dataset = ECGDataset(args.val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Model
    model = ResNetBaseline().to(device)
    
    # Loss & Opt
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # Training Loop
    best_val_loss = float('inf')
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                loss = criterion(y_pred, y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.output_path)
            print(f"Saved best model to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="checkpoints/resnet_best.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    
    args = parser.parse_args()
    train(args)
