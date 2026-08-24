#!/usr/bin/env python3
"""
Phase 3: Extract ECG-FM Embeddings

Extracts embeddings from ECG-FM backbone for morphology analysis.
Computes embeddings for Original, M0, and M1 signals.

Usage:
    python src/03_extract_embeddings.py --output_dir outputs/embeddings

Outputs:
    - outputs/embeddings/original_embeddings.pt
    - outputs/embeddings/m0_embeddings.pt
    - outputs/embeddings/m1_embeddings.pt
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'fairseq-signals'))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from ecg_fm_classifier import ECGFMClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EmbeddingDataset(Dataset):
    """Dataset for embedding extraction."""
    
    def __init__(
        self,
        data_dir: str,
        database_csv: str,
        split: str = "test",
        signal_type: str = "original",  # original, m0, m1
        target_len: int = 5000
    ):
        self.data_dir = Path(data_dir)
        self.target_len = target_len
        self.signal_type = signal_type
        
        # Load database and filter by split
        df = pd.read_csv(database_csv)
        if split == "test":
            self.df = df[df['strat_fold'] == 10]
        elif split == "val":
            self.df = df[df['strat_fold'] == 9]
        else:
            self.df = df[~df['strat_fold'].isin([9, 10])]
        
        logger.info(f"Loaded {len(self.df)} records for {signal_type} embeddings")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        ecg_id = row['ecg_id']
        
        # Determine path based on signal type
        if self.signal_type == "original":
            tensor_path = self.data_dir / "tensors" / "test" / f"{ecg_id}.pt"
        else:
            tensor_path = self.data_dir / "reconstructed" / self.signal_type / f"{ecg_id}.pt"
        
        # Load signal
        if tensor_path.exists():
            signal = torch.load(tensor_path, weights_only=True)
            if isinstance(signal, dict):
                signal = signal.get('data', signal.get('signal', torch.zeros(12, self.target_len)))
        else:
            signal = torch.zeros(12, self.target_len)
        
        # Ensure correct shape
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)
        if signal.shape[0] > 12:
            signal = signal[:12]
        elif signal.shape[0] < 12:
            pad = torch.zeros(12 - signal.shape[0], signal.shape[1])
            signal = torch.cat([signal, pad], dim=0)
        
        # Pad/crop
        if signal.shape[1] < self.target_len:
            pad = torch.zeros(12, self.target_len - signal.shape[1])
            signal = torch.cat([signal, pad], dim=1)
        elif signal.shape[1] > self.target_len:
            signal = signal[:, :self.target_len]
        
        # Normalize
        mean = signal.mean(dim=1, keepdim=True)
        std = signal.std(dim=1, keepdim=True) + 1e-6
        signal = (signal - mean) / std
        
        return {
            'signal': signal,
            'ecg_id': ecg_id
        }


@torch.no_grad()
def extract_embeddings(
    model: ECGFMClassifier,
    data_loader: DataLoader,
    device: torch.device,
    pool: bool = False  # Return full temporal embeddings
) -> Tuple[np.ndarray, List[int]]:
    """Extract embeddings from ECG-FM backbone."""
    model.eval()
    
    all_embeddings = []
    all_ids = []
    
    for batch in tqdm(data_loader, desc="Extracting embeddings"):
        signals = batch['signal'].to(device)
        ecg_ids = batch['ecg_id']
        
        # Extract embeddings (full temporal for morphology analysis)
        embeddings = model.extract_embeddings(signals, pool=pool)
        
        all_embeddings.append(embeddings.cpu().numpy())
        all_ids.extend([int(eid) for eid in ecg_ids])
    
    return np.concatenate(all_embeddings, axis=0), all_ids


def compute_embedding_distances(
    orig_embeddings: np.ndarray,
    recon_embeddings: np.ndarray,
    pooled: bool = True
) -> Dict[str, np.ndarray]:
    """
    Compute embedding distances between original and reconstructed signals.
    
    Returns:
        Dictionary with global_distance, qrs_distance, st_distance arrays
    """
    n_samples = orig_embeddings.shape[0]
    
    if pooled:
        # For pooled embeddings: simple L2 distance
        global_dist = np.linalg.norm(orig_embeddings - recon_embeddings, axis=1)
        return {
            'global_distance': global_dist,
            'qrs_distance': global_dist,  # Same for pooled
            'st_distance': global_dist
        }
    
    # For temporal embeddings: compute regional distances
    # Assuming 312 time steps for 5000 samples (16x downsampling)
    # QRS: ~80-180ms → steps 5-11 (approx)
    # ST: ~180-400ms → steps 11-25 (approx)
    
    time_steps = orig_embeddings.shape[1]
    qrs_start, qrs_end = int(time_steps * 0.08), int(time_steps * 0.18)
    st_start, st_end = int(time_steps * 0.18), int(time_steps * 0.40)
    
    global_dist = np.zeros(n_samples)
    qrs_dist = np.zeros(n_samples)
    st_dist = np.zeros(n_samples)
    
    for i in range(n_samples):
        # Global distance (time-averaged)
        orig_pooled = orig_embeddings[i].mean(axis=0)
        recon_pooled = recon_embeddings[i].mean(axis=0)
        global_dist[i] = np.linalg.norm(orig_pooled - recon_pooled)
        
        # QRS region distance
        orig_qrs = orig_embeddings[i, qrs_start:qrs_end].mean(axis=0)
        recon_qrs = recon_embeddings[i, qrs_start:qrs_end].mean(axis=0)
        qrs_dist[i] = np.linalg.norm(orig_qrs - recon_qrs)
        
        # ST region distance
        orig_st = orig_embeddings[i, st_start:st_end].mean(axis=0)
        recon_st = recon_embeddings[i, st_start:st_end].mean(axis=0)
        st_dist[i] = np.linalg.norm(orig_st - recon_st)
    
    return {
        'global_distance': global_dist,
        'qrs_distance': qrs_dist,
        'st_distance': st_dist
    }


def main():
    parser = argparse.ArgumentParser(description="Extract ECG-FM Embeddings")
    parser.add_argument("--checkpoint", type=str, 
                       default="checkpoints/mimic_iv_ecg_physionet_pretrained.pt")
    parser.add_argument("--data_dir", type=str, default="~/data/ptb_xl")
    parser.add_argument("--database_csv", type=str, default="~/data/ptbxl_database.csv")
    parser.add_argument("--output_dir", type=str, default="outputs/embeddings")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--pool", action="store_true", help="Pool embeddings (faster, less detailed)")
    args = parser.parse_args()
    
    # Expand paths
    args.checkpoint = os.path.expanduser(args.checkpoint)
    args.data_dir = os.path.expanduser(args.data_dir)
    args.database_csv = os.path.expanduser(args.database_csv)
    args.output_dir = os.path.expanduser(args.output_dir)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    model = ECGFMClassifier(
        checkpoint_path=args.checkpoint,
        num_classes=5,
        freeze_backbone=True
    ).to(device)
    
    # Extract embeddings for each signal type
    results = {}
    
    for signal_type in ['original', 'm0', 'm1_hpo']:
        logger.info(f"\nExtracting {signal_type} embeddings...")
        
        dataset = EmbeddingDataset(
            data_dir=args.data_dir,
            database_csv=args.database_csv,
            split="test",
            signal_type=signal_type
        )
        
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        embeddings, ids = extract_embeddings(model, loader, device, pool=args.pool)
        
        # Save embeddings
        output_file = Path(args.output_dir) / f"{signal_type}_embeddings.pt"
        torch.save({
            'embeddings': embeddings,
            'ecg_ids': ids,
            'pooled': args.pool
        }, output_file)
        
        results[signal_type] = {
            'embeddings': embeddings,
            'ids': ids
        }
        
        logger.info(f"Saved {signal_type} embeddings: shape {embeddings.shape}")
    
    # Compute embedding distances
    logger.info("\nComputing embedding distances...")
    
    distance_results = []
    
    for recon_type in ['m0', 'm1_hpo']:
        if recon_type in results:
            distances = compute_embedding_distances(
                results['original']['embeddings'],
                results[recon_type]['embeddings'],
                pooled=args.pool
            )
            
            for i, ecg_id in enumerate(results['original']['ids']):
                distance_results.append({
                    'ecg_id': ecg_id,
                    'signal_type': recon_type,
                    'global_distance': distances['global_distance'][i],
                    'qrs_distance': distances['qrs_distance'][i],
                    'st_distance': distances['st_distance'][i]
                })
    
    # Save distance results
    if distance_results:
        df = pd.DataFrame(distance_results)
        df.to_csv(Path(args.output_dir) / 'embedding_distances.csv', index=False)
        
        # Print summary
        print("\n" + "=" * 60)
        print("EMBEDDING DISTANCE SUMMARY")
        print("=" * 60)
        
        for recon_type in ['m0', 'm1_hpo']:
            subset = df[df['signal_type'] == recon_type]
            if len(subset) > 0:
                print(f"\n{recon_type}:")
                print(f"  Global distance: {subset['global_distance'].mean():.4f} ± {subset['global_distance'].std():.4f}")
                print(f"  QRS distance:    {subset['qrs_distance'].mean():.4f} ± {subset['qrs_distance'].std():.4f}")
                print(f"  ST distance:     {subset['st_distance'].mean():.4f} ± {subset['st_distance'].std():.4f}")
        
        # Statistical test
        if 'm0' in df['signal_type'].values and 'm1_hpo' in df['signal_type'].values:
            from scipy import stats
            m0_dist = df[df['signal_type'] == 'm0']['global_distance'].values
            m1_dist = df[df['signal_type'] == 'm1_hpo']['global_distance'].values
            
            t_stat, p_val = stats.ttest_rel(m0_dist, m1_dist)
            effect_size = (m0_dist.mean() - m1_dist.mean()) / np.sqrt((m0_dist.std()**2 + m1_dist.std()**2) / 2)
            
            print(f"\nPaired t-test (M0 vs M1):")
            print(f"  t-statistic: {t_stat:.4f}")
            print(f"  p-value: {p_val:.6f}")
            print(f"  Cohen's d: {effect_size:.4f}")
            
            if p_val < 0.05 and m1_dist.mean() < m0_dist.mean():
                print("  ✓ M1 significantly closer to original than M0")
            elif p_val < 0.05:
                print("  ✗ M0 significantly closer to original than M1")
            else:
                print("  - No significant difference")


if __name__ == "__main__":
    main()
