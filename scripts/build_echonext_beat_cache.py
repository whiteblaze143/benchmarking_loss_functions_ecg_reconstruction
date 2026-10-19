#!/usr/bin/env python3
"""Build Pre-extracted Beat Cache for EchoNext (Train and Val).

Extracts authentic phase-normalized beats for all EchoNext waveforms
and writes memory-mapped binary files (.dat + _index.npy) to:
    /data/mithunmanivannan/echonext/cache_beats/echonext_{split}_beats.dat
    /data/mithunmanivannan/echonext/cache_beats/echonext_{split}_beats_index.npy

Safety features:
- Runs at low scheduling priority (nice=19) so it does not compete with active GPU training.
- Uses 3 worker processes (leaving remaining cores for active training and OS).
- Writes to temporary files and atomically renames only after rigorous verification.
- Constant memory consumption (< 150 MB RAM) by streaming chunked writes to disk.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
import time
from typing import Tuple, List

import numpy as np
import torch
import multiprocessing as mp

DEV_ROOT = Path("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction_geometry_dev")
BENCHMARK_DIR = DEV_ROOT / "experiments/echonext_operator_benchmark"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from phase_kme_representation import EchoNextPhaseKMEExtractor, DEFAULT_FIT_PATH, CANONICAL_8_LEADS

DATA_DIR = Path("/home/mithunmanivannan/data/echonext")
CACHE_DIR = Path("/data/mithunmanivannan/echonext/cache_beats")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("build_beat_cache")


_worker_extractor: Optional[EchoNextPhaseKMEExtractor] = None


def worker_init(fit_path_str: str):
    global _worker_extractor
    try:
        os.nice(19)
    except Exception:
        pass
    _worker_extractor = EchoNextPhaseKMEExtractor(fit_path=fit_path_str, device="cpu")


def extract_single_record(args: Tuple[int, np.ndarray]) -> Tuple[int, np.ndarray]:
    global _worker_extractor
    idx, wf_8 = args
    assert _worker_extractor is not None
    beats = _worker_extractor.extract_beats(wf_8)
    return idx, beats


def build_cache_for_split(
    split: str,
    waveforms_path: Path,
    num_workers: int = 3,
    chunk_size: int = 200,
    dry_run_limit: int | None = None,
) -> None:
    logger.info("==================================================")
    logger.info("Building beat cache for split: %s", split)
    logger.info("Source waveforms: %s", waveforms_path)
    logger.info("Cache directory: %s", CACHE_DIR)
    logger.info("Workers: %d (nice=19)", num_workers)
    logger.info("==================================================")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    target_dat = CACHE_DIR / f"echonext_{split}_beats.dat"
    target_idx = CACHE_DIR / f"echonext_{split}_beats_index.npy"

    temp_dat = CACHE_DIR / f"temp_echonext_{split}_beats.dat"
    temp_idx = CACHE_DIR / f"temp_echonext_{split}_beats_index.npy"

    if temp_dat.exists():
        temp_dat.unlink()
    if temp_idx.exists():
        temp_idx.unlink()

    waveforms = np.load(waveforms_path, mmap_mode="r")
    total_records = len(waveforms) if dry_run_limit is None else min(dry_run_limit, len(waveforms))
    logger.info("Total records to extract: %d", total_records)

    index_array = np.zeros((total_records, 2), dtype=np.int64)
    total_beats_accum = 0

    t0 = time.time()
    last_log_time = t0

    # Initialize extractor once for reference
    ref_extractor = EchoNextPhaseKMEExtractor(fit_path=DEFAULT_FIT_PATH, device="cpu")

    # Use multiprocessing pool with low priority
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=num_workers, initializer=worker_init, initargs=(str(DEFAULT_FIT_PATH),)) as pool:
        with open(temp_dat, "wb") as f_out:
            for start_idx in range(0, total_records, chunk_size):
                end_idx = min(start_idx + chunk_size, total_records)
                
                tasks = []
                for i in range(start_idx, end_idx):
                    wf_8 = np.array(waveforms[i, 0][:, CANONICAL_8_LEADS], copy=True)
                    tasks.append((i, wf_8))

                results = pool.map(extract_single_record, tasks)

                # Ensure results are in sequential order
                results.sort(key=lambda x: x[0])

                for i, beats in results:
                    n_beats = len(beats)
                    assert n_beats > 0, f"Record {i} produced 0 beats"
                    assert beats.shape[1:] == (256, 8), f"Record {i} shape mismatch: {beats.shape}"
                    
                    index_array[i, 0] = total_beats_accum
                    index_array[i, 1] = n_beats
                    total_beats_accum += n_beats

                    f_out.write(beats.astype(np.float32).tobytes())

                now = time.time()
                if now - last_log_time >= 15.0 or end_idx == total_records:
                    elapsed = now - t0
                    rate = end_idx / max(elapsed, 1e-4)
                    remaining = (total_records - end_idx) / max(rate, 1e-4)
                    logger.info(
                        "Split [%s] Progress: %d / %d records (%.1f%%) | %.1f rec/s | Beats: %d | Elapsed: %.1fm | ETA: %.1fm",
                        split, end_idx, total_records, 100.0 * end_idx / total_records,
                        rate, total_beats_accum, elapsed / 60.0, remaining / 60.0,
                    )
                    last_log_time = now

    f_out.close()
    np.save(temp_idx, index_array)
    logger.info("Extraction finished in %.1f minutes. Total beats: %d. Verifying cache...", (time.time() - t0) / 60.0, total_beats_accum)

    # Verification Gate
    expected_bytes = total_beats_accum * 256 * 8 * 4
    actual_bytes = temp_dat.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(f"Cache file size mismatch! Expected {expected_bytes} bytes, got {actual_bytes}")
    logger.info("Byte count verified: %d bytes (%.2f GB)", actual_bytes, actual_bytes / (1024**3))

    # Spot check 5 random records
    mmap_beats = np.memmap(temp_dat, dtype=np.float32, mode="r", shape=(total_beats_accum, 256, 8))
    check_indices = [0, total_records // 4, total_records // 2, 3 * total_records // 4, total_records - 1]
    for chk_i in check_indices:
        b_start, b_count = index_array[chk_i]
        cached_b = np.array(mmap_beats[b_start : b_start + b_count], copy=True)
        raw_sig_8 = waveforms[chk_i, 0][:, CANONICAL_8_LEADS]
        direct_b = ref_extractor.extract_beats(raw_sig_8)
        if not np.allclose(cached_b, direct_b, atol=1e-6):
            raise ValueError(f"Spot check verification failed at record index {chk_i}")
    logger.info("Spot check verification PASSED cleanly across indices: %s", check_indices)

    # Atomic Rename
    temp_dat.replace(target_dat)
    temp_idx.replace(target_idx)
    logger.info("Successfully published beat cache for split [%s] to %s and %s", split, target_dat, target_idx)


def main():
    parser = argparse.ArgumentParser(description="Build EchoNext Beat Cache")
    parser.add_argument("--split", choices=["train", "val", "all"], default="all")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--dry-run-limit", type=int, default=None)
    args = parser.parse_args()

    try:
        os.nice(19)
    except Exception as e:
        logger.warning("Could not set nice level: %s", e)

    splits = ["val", "train"] if args.split == "all" else [args.split]

    for sp in splits:
        wf_path = DATA_DIR / f"EchoNext_{sp}_waveforms.npy"
        build_cache_for_split(
            split=sp,
            waveforms_path=wf_path,
            num_workers=args.workers,
            dry_run_limit=args.dry_run_limit,
        )

    logger.info("ALL BEAT CACHES SUCCESSFULLY BUILT AND VERIFIED!")


if __name__ == "__main__":
    main()
