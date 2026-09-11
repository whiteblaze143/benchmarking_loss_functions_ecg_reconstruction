"""Download Panobench dataset from Hugging Face to NFS /data directory.

Target: /data/mithunmanivannan/panobench
Source: whynotJunger/Panobench (Hugging Face dataset)
Size: ~4.3 GB, 4,470 ten-second recordings with 48 viewpoints.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from huggingface_hub import snapshot_download

TARGET_DIR = Path("/data/mithunmanivannan/panobench")
REPO_ID = "whynotJunger/Panobench"


def main():
    print(f"=== Starting Panobench Download from Hugging Face ({REPO_ID}) ===")
    print(f"Target NFS Directory: {TARGET_DIR}")
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    download_path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(TARGET_DIR),
        max_workers=8,
    )
    elapsed = time.time() - start_time
    print(f"\nDownload completed in {elapsed:.1f}s.")

    # Audit downloaded contents
    total_files = 0
    total_bytes = 0
    subdirs = set()
    for root, dirs, files in os.walk(TARGET_DIR):
        for f in files:
            p = Path(root) / f
            total_files += 1
            total_bytes += p.stat().st_size
            rel_dir = p.parent.relative_to(TARGET_DIR)
            if str(rel_dir) != ".":
                subdirs.add(str(rel_dir))

    total_gb = total_bytes / (1024 ** 3)
    print(f"Total Downloaded Files: {total_files}")
    print(f"Total Size: {total_gb:.2f} GB")
    print(f"Subdirectories: {sorted(list(subdirs))[:10]}")
    print(f"Panobench ready at: {TARGET_DIR}")


if __name__ == "__main__":
    main()
