#!/usr/bin/env python3
"""
Phase 1 Data Preparation: Download and preprocess PTB-XL dataset

This script:
1. Downloads PTB-XL from PhysioNet (if not present)
2. Preprocesses WFDB files into PyTorch tensors
3. Creates train/val/test splits based on strat_fold

Usage:
    python src/01_prepare_ptbxl_data.py --output_dir ~/data/ptb_xl

Requirements:
    pip install wfdb requests tqdm pandas
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional
import shutil

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def download_ptbxl(output_dir: str, force: bool = False) -> Path:
    """
    Download PTB-XL dataset from PhysioNet.
    
    Note: Full dataset is ~5.5GB. This function downloads and extracts it.
    """
    import urllib.request
    import tarfile
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Check if already downloaded
    records_dir = output_path / "records500"
    if records_dir.exists() and not force:
        logger.info(f"PTB-XL already exists at {records_dir}")
        return output_path
    
    # Download URL
    url = "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiogram-dataset-1.0.3.zip"
    zip_path = output_path / "ptbxl.zip"
    
    if zip_path.exists() and zip_path.stat().st_size > 1000 * 1024 * 1024:
        logger.info(f"Complete ptbxl.zip already exists at {zip_path}, skipping download.")
    else:
        logger.info(f"Downloading PTB-XL from PhysioNet (~5.5GB)...")
        logger.info(f"This may take 20-30 minutes depending on your connection.")
        
        # Download with progress
        def download_progress(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            print(f"\rProgress: {percent}%", end='', flush=True)
        
        try:
            urllib.request.urlretrieve(url, zip_path, download_progress)
            print()  # New line after progress
        except Exception as e:
            logger.error(f"Download failed: {e}")
            logger.info("Please download manually from: https://physionet.org/content/ptb-xl/1.0.3/")
            raise
    
    # Extract
    logger.info("Extracting archive...")
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(output_path)
    
    # Move files from nested directory
    nested = output_path / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
    if not nested.exists():
        # Fallback to the electrocardiogram spelling if needed
        nested = output_path / "ptb-xl-a-large-publicly-available-electrocardiogram-dataset-1.0.3"
    if nested.exists():
        for item in nested.iterdir():
            shutil.move(str(item), str(output_path / item.name))
        nested.rmdir()
    
    # Clean up
    if zip_path.exists():
        zip_path.unlink()
    
    logger.info(f"PTB-XL extracted to {output_path}")
    return output_path


def preprocess_record(
    record_path: str,
    target_fs: int = 500,
    target_len: int = 5000
) -> Optional[torch.Tensor]:
    """
    Load and preprocess a single WFDB record.
    
    Returns:
        Tensor of shape (12, target_len) or None if loading fails
    """
    import wfdb
    
    try:
        # Load record
        record = wfdb.rdsamp(record_path)
        signal = record[0]  # (samples, leads)
        fs = record[1]['fs']
        
        # Transpose to (leads, samples)
        signal = signal.T
        
        # Resample if needed
        if fs != target_fs:
            from scipy import signal as sig
            num_samples = int(signal.shape[1] * target_fs / fs)
            signal = sig.resample(signal, num_samples, axis=1)
        
        # Pad/crop to target length
        if signal.shape[1] < target_len:
            pad = np.zeros((signal.shape[0], target_len - signal.shape[1]))
            signal = np.concatenate([signal, pad], axis=1)
        elif signal.shape[1] > target_len:
            signal = signal[:, :target_len]
        
        # Ensure 12 leads
        if signal.shape[0] < 12:
            pad = np.zeros((12 - signal.shape[0], target_len))
            signal = np.concatenate([signal, pad], axis=0)
        elif signal.shape[0] > 12:
            signal = signal[:12]
        
        # Convert to tensor
        tensor = torch.tensor(signal, dtype=torch.float32)
        
        # Check for NaN/Inf
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            logger.warning(f"NaN/Inf detected in {record_path}")
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        
        return tensor
        
    except Exception as e:
        logger.warning(f"Failed to load {record_path}: {e}")
        return None


def preprocess_ptbxl(
    data_dir: str,
    output_dir: str,
    database_csv: str,
    use_hr: bool = True  # Use 500Hz records (hr) vs 100Hz (lr)
) -> None:
    """
    Preprocess all PTB-XL records into tensor format.
    
    Args:
        data_dir: Directory containing PTB-XL WFDB files
        output_dir: Directory to save processed tensors
        database_csv: Path to ptbxl_database.csv
        use_hr: Whether to use high-resolution (500Hz) records
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load database
    df = pd.read_csv(database_csv)
    logger.info(f"Processing {len(df)} records")
    
    # Create split directories
    for split in ['train', 'val', 'test']:
        (output_path / split).mkdir(exist_ok=True)
    
    # Process each record
    filename_col = 'filename_hr' if use_hr else 'filename_lr'
    
    success_count = 0
    fail_count = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing records"):
        ecg_id = row['ecg_id']
        filename = row[filename_col]
        strat_fold = row['strat_fold']
        
        # Determine split
        if strat_fold == 10:
            split = 'test'
        elif strat_fold == 9:
            split = 'val'
        else:
            split = 'train'
        
        # Load and preprocess
        record_path = data_path / filename
        if not record_path.with_suffix('.dat').exists():
            fail_count += 1
            continue
        
        tensor = preprocess_record(str(record_path))
        
        if tensor is not None:
            # Save tensor
            output_file = output_path / split / f"{ecg_id}.pt"
            torch.save(tensor, output_file)
            success_count += 1
        else:
            fail_count += 1
    
    logger.info(f"Preprocessing complete: {success_count} success, {fail_count} failed")
    
    # Create split metadata
    for split in ['train', 'val', 'test']:
        split_dir = output_path / split
        files = list(split_dir.glob("*.pt"))
        logger.info(f"{split}: {len(files)} files")


def create_manifests(
    data_dir: str,
    database_csv: str,
    output_dir: str
) -> None:
    """
    Create manifest files for fairseq-signals format.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(database_csv)
    
    for split_name, fold_filter in [('train', lambda x: x not in [9, 10]),
                                     ('valid', lambda x: x == 9),
                                     ('test', lambda x: x == 10)]:
        split_df = df[df['strat_fold'].apply(fold_filter)]
        
        # Create manifest
        manifest_path = output_path / f"{split_name}.tsv"
        with open(manifest_path, 'w') as f:
            f.write(f"{data_dir}\n")  # Root directory
            for _, row in split_df.iterrows():
                filename = row['filename_hr']
                # Get number of samples (assume 5000 for 10s @ 500Hz)
                f.write(f"{filename}\t5000\n")
        
        logger.info(f"Created {manifest_path} with {len(split_df)} entries")


def main():
    parser = argparse.ArgumentParser(description="PTB-XL Data Preparation")
    parser.add_argument("--data_dir", type=str, default="~/data/ptb_xl",
                       help="Directory containing/to download PTB-XL WFDB files")
    parser.add_argument("--output_dir", type=str, default="~/data/ptb_xl/tensors",
                       help="Directory to save processed tensors")
    parser.add_argument("--database_csv", type=str, default="~/data/ptbxl_database.csv",
                       help="Path to ptbxl_database.csv")
    parser.add_argument("--download", action="store_true",
                       help="Download PTB-XL from PhysioNet")
    parser.add_argument("--preprocess", action="store_true",
                       help="Preprocess WFDB files to tensors")
    parser.add_argument("--create_manifests", action="store_true",
                       help="Create fairseq-signals manifest files")
    args = parser.parse_args()
    
    # Expand paths
    args.data_dir = os.path.expanduser(args.data_dir)
    args.output_dir = os.path.expanduser(args.output_dir)
    args.database_csv = os.path.expanduser(args.database_csv)
    
    if args.download:
        download_ptbxl(args.data_dir)
    
    if args.preprocess:
        preprocess_ptbxl(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            database_csv=args.database_csv
        )
    
    if args.create_manifests:
        create_manifests(
            data_dir=args.data_dir,
            database_csv=args.database_csv,
            output_dir=os.path.dirname(args.output_dir) + "/manifest"
        )
    
    if not (args.download or args.preprocess or args.create_manifests):
        logger.info("No action specified. Use --download, --preprocess, or --create_manifests")
        logger.info("Example: python src/01_prepare_ptbxl_data.py --download --preprocess")


if __name__ == "__main__":
    main()
