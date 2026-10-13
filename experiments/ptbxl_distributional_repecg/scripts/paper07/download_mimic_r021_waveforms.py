#!/usr/bin/env python3
"""
Download MIMIC-III-Ext-PPG Waveforms for Prespecified R021 Benchmark.
Downloads .hea and .dat files for Fold 8 (Test), Fold 7 (Validation), and Folds 0-6 (Train).
Saves files directly into /data/mithunmanivannan/datasets/mimic3_ext_ppg/.
"""

import os
import sys
import time
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from requests.auth import HTTPBasicAuth
import polars as pl

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("download_mimic_r021")

MIMIC_DIR = Path("/data/mithunmanivannan/datasets/mimic3_ext_ppg")
PAIRS_PARQUET = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/mimic3_ext_ppg_safe_audit/safe_transition_pairs.parquet")
BASE_URL = "https://physionet.org/files/mimic-iii-ext-ppg/1.1.0"

USER = os.environ.get("PHYSIONET_USER", "")
PASS = os.environ.get("PHYSIONET_PASS", "")
HEADERS = {"User-Agent": "Wget/1.21.3"}

def get_required_file_keys(df: pl.DataFrame) -> set:
    keys = set()
    for row in df.iter_rows(named=True):
        # support
        p_sup = f"p{row['subject_id']:06d}"
        f_sup = f"{p_sup[:3]}/{p_sup}/{row['record_id']}_{row['event_id']}_0"
        keys.add(f_sup)
        # target
        p_tgt = f"p{row['subject_id']:06d}"
        f_tgt = f"{p_tgt[:3]}/{p_tgt}/{row['target_record_id']}_{row['target_event_id']}_0"
        keys.add(f_tgt)
    return keys

def download_file(rel_path: str, ext: str, session: requests.Session) -> bool:
    target_path = MIMIC_DIR / f"{rel_path}.{ext}"
    if target_path.exists() and target_path.stat().st_size > 0:
        return True

    target_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_URL}/{rel_path}.{ext}"

    for attempt in range(3):
        try:
            r = session.get(url, auth=HTTPBasicAuth(USER, PASS), headers=HEADERS, timeout=20)
            if r.status_code == 200:
                with open(target_path, "wb") as f:
                    f.write(r.content)
                return True
            elif r.status_code == 404:
                logger.warning(f"File not found (404): {url}")
                return False
            else:
                time.sleep(1.0)
        except Exception as e:
            time.sleep(1.0)
    return False

def main():
    logger.info("=" * 70)
    logger.info("Downloading Real PhysioNet MIMIC-III-Ext-PPG WFDB Waveforms")
    logger.info("=" * 70)

    df_pairs = pl.read_parquet(PAIRS_PARQUET)
    logger.info(f"Loaded {len(df_pairs):,} total safe transition pairs.")

    # 1. Filter cohorts
    test_pos = df_pairs.filter((pl.col("target_strat_fold") == 8) & (pl.col("transition_cell") == "AF->SR"))
    test_neg = df_pairs.filter((pl.col("target_strat_fold") == 8) & (pl.col("transition_cell") == "AF->AF")).sample(n=len(test_pos)*5, seed=42)
    test_pairs = pl.concat([test_pos, test_neg])

    val_pos = df_pairs.filter((pl.col("target_strat_fold") == 7) & (pl.col("transition_cell") == "AF->SR"))
    val_neg = df_pairs.filter((pl.col("target_strat_fold") == 7) & (pl.col("transition_cell") == "AF->AF")).sample(n=len(val_pos)*5, seed=42)
    val_pairs = pl.concat([val_pos, val_neg])

    train_pos = df_pairs.filter((pl.col("target_strat_fold").is_in(list(range(7)))) & (pl.col("transition_cell") == "AF->SR"))
    train_neg = df_pairs.filter((pl.col("target_strat_fold").is_in(list(range(7)))) & (pl.col("transition_cell") == "AF->AF")).sample(n=len(train_pos)*5, seed=42)
    train_pairs = pl.concat([train_pos, train_neg])

    test_keys = get_required_file_keys(test_pairs)
    val_keys = get_required_file_keys(val_pairs)
    train_keys = get_required_file_keys(train_pairs)

    all_keys = test_keys.union(val_keys).union(train_keys)
    logger.info(f"Identified {len(all_keys):,} unique 30-s segments to download:")
    logger.info(f"  Fold 8 Sealed Test: {len(test_keys):,} unique segments")
    logger.info(f"  Fold 7 Validation:  {len(val_keys):,} unique segments")
    logger.info(f"  Folds 0-6 Training: {len(train_keys):,} unique segments")

    # Priority download: Test + Val first, then Train
    priority_keys = list(test_keys.union(val_keys))
    logger.info(f"\nPhase 1: Downloading Test + Val Segments ({len(priority_keys)} segments, 2 files each)...")

    tasks = []
    for k in priority_keys:
        tasks.append((k, "hea"))
        tasks.append((k, "dat"))

    session = requests.Session()
    success = 0
    total = len(tasks)
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(download_file, k, ext, session): (k, ext) for k, ext in tasks}
        for i, future in enumerate(as_completed(futures), start=1):
            if future.result():
                success += 1
            if i % 200 == 0 or i == total:
                elapsed = time.time() - start_time
                logger.info(f"Progress (Test + Val): {i}/{total} files downloaded ({success} ok) [{i/elapsed:.1f} files/sec]")

    logger.info(f"Phase 1 Complete! Successfully downloaded {success}/{total} test+val files in {time.time()-start_time:.1f}s.")

    # Phase 2: Training set download
    remaining_keys = list(train_keys.difference(test_keys.union(val_keys)))
    logger.info(f"\nPhase 2: Downloading Training Segments ({len(remaining_keys)} segments, 2 files each)...")

    train_tasks = []
    for k in remaining_keys:
        train_tasks.append((k, "hea"))
        train_tasks.append((k, "dat"))

    train_success = 0
    total_train = len(train_tasks)
    start_train = time.time()

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(download_file, k, ext, session): (k, ext) for k, ext in train_tasks}
        for i, future in enumerate(as_completed(futures), start=1):
            if future.result():
                train_success += 1
            if i % 1000 == 0 or i == total_train:
                elapsed = time.time() - start_train
                logger.info(f"Progress (Train): {i}/{total_train} files downloaded ({train_success} ok) [{i/elapsed:.1f} files/sec]")

    logger.info(f"Phase 2 Complete! Successfully downloaded {train_success}/{total_train} training files in {time.time()-start_train:.1f}s.")
    logger.info("ALL MIMIC R021 WAVEFORMS MATERIALIZED LOCALLY ON DISK!")

if __name__ == "__main__":
    main()
