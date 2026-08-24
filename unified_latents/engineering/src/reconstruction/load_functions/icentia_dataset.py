
import os
import glob
import torch
import wfdb
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

class IcentiaDataset(Dataset):
    """
    Dataset for Icentia11k.
    Returns:
        sig: Tensor [1, 150000] (10 minutes @ 250Hz).
             If record is longer, selects random 10-min crop.
             If record is shorter, pads with zeros (should be rare for valid Icentia segments).
    """
    def __init__(self, root_dir: str, target_fs: int = 250, crop_len: int = 150000):
        self.root_dir = root_dir
        self.target_fs = target_fs
        self.crop_len = crop_len
        
        # Find all .hea files recursively
        self.records = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.hea'):
                    full_path = os.path.join(root, file)
                    base_path = os.path.splitext(full_path)[0]
                    # Verify .dat exists
                    if os.path.exists(f"{base_path}.dat"):
                        self.records.append(base_path)
        
        print(f"Found {len(self.records)} Icentia records in {root_dir}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record_path = self.records[idx]
        
        # We need total samples to random crop
        # Reading header is fast
        try:
            header = wfdb.rdheader(record_path)
            total_samples = header.sig_len
            fs = header.fs
            
            if fs != self.target_fs:
                # Should not happen per PRD (locked 250Hz), but safety check
                raise ValueError(f"Record {record_path} has fs={fs}, expected {self.target_fs}")
            
            # Determine start index
            if total_samples > self.crop_len:
                max_start = total_samples - self.crop_len
                start_idx = np.random.randint(0, max_start)
                end_idx = start_idx + self.crop_len
            else:
                start_idx = 0
                end_idx = total_samples # Read all
                
            # Read signal
            # wfdb returns physical units (mV) by default unless fmt is specified differently
            # We confirmed units are mV in Legacy Analysis validation
            record = wfdb.rdrecord(record_path, sampfrom=start_idx, sampto=end_idx, channels=[0]) 
            # channels=[0] strictly for Single-Lead Icentia
            
            sig = record.p_signal.T # [samples, channels] -> [channels, samples]
            
            # Convert to Tensor
            sig_tensor = torch.from_numpy(sig).float() # [1, L]
            sig_tensor = torch.nan_to_num(sig_tensor, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Pad if short
            if sig_tensor.shape[1] < self.crop_len:
                pad_len = self.crop_len - sig_tensor.shape[1]
                sig_tensor = torch.nn.functional.pad(sig_tensor, (0, pad_len))
                
            return sig_tensor

        except Exception as e:
            print(f"Error loading {record_path}: {e}")
            # Return zero tensor fallback
            return torch.zeros(1, self.crop_len)
