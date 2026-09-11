#!/usr/bin/env python3
"""
Stage A: Environment and Provenance Audit for HEEDB v5.0
Produces:
  00_environment/environment.json
  00_environment/pip_freeze.txt
  00_environment/git_commit.txt
  00_environment/storage_snapshot.txt
  00_environment/source_file_checksums.csv
"""
import os
import sys
import json
import hashlib
import datetime
import subprocess
import platform
from concurrent.futures import ThreadPoolExecutor

AUDIT_DIR = "/data/mithunmanivannan/heedb_audit"
ENV_DIR = os.path.join(AUDIT_DIR, "00_environment")
LOCAL_DATA_DIR = "/data/mithunmanivannan"

os.makedirs(ENV_DIR, exist_ok=True)

def run_cmd(cmd):
    try:
        res = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return res.stdout.strip()
    except Exception as e:
        return f"ERROR: {e}"

def compute_file_stats(filepath, max_workers=16, chunk_size=4 * 1024 * 1024):
    size = os.path.getsize(filepath)
    mtime = datetime.datetime.fromtimestamp(os.path.getmtime(filepath), datetime.timezone.utc).isoformat()
    if size == 0:
        return {
            "filepath": filepath,
            "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "size_bytes": 0,
            "mtime_utc": mtime,
            "row_count": 0
        }

    fd = os.open(filepath, os.O_RDONLY)
    h = hashlib.sha256()
    line_count = 0
    n_chunks = (size + chunk_size - 1) // chunk_size

    def read_chunk(idx):
        offset = idx * chunk_size
        length = min(chunk_size, size - offset)
        return idx, os.pread(fd, length, offset)

    window_size = max_workers * 2
    futures = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for i in range(min(window_size, n_chunks)):
            futures[i] = pool.submit(read_chunk, i)

        for i in range(n_chunks):
            next_idx = i + window_size
            if next_idx < n_chunks:
                futures[next_idx] = pool.submit(read_chunk, next_idx)

            _, chunk = futures.pop(i).result()
            h.update(chunk)
            if filepath.endswith('.csv'):
                line_count += chunk.count(b'\n')

    os.close(fd)
    return {
        "filepath": filepath,
        "sha256": h.hexdigest(),
        "size_bytes": size,
        "mtime_utc": mtime,
        "row_count": line_count - 1 if filepath.endswith('.csv') and line_count > 0 else line_count
    }

def main():
    now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    
    # 1. Git commit
    git_commit = run_cmd("git rev-parse HEAD")
    git_status = run_cmd("git status --short")
    with open(os.path.join(ENV_DIR, "git_commit.txt"), "w") as f:
        f.write(f"Commit: {git_commit}\nStatus:\n{git_status}\n")

    # 2. Pip freeze
    pip_freeze = run_cmd(f"{sys.executable} -m pip freeze")
    with open(os.path.join(ENV_DIR, "pip_freeze.txt"), "w") as f:
        f.write(pip_freeze + "\n")

    # 3. Storage snapshot
    storage_snapshot = run_cmd("df -h /data /")
    mounts = run_cmd("mount | grep -E '/data|ext4|nfs'")
    with open(os.path.join(ENV_DIR, "storage_snapshot.txt"), "w") as f:
        f.write(f"--- Disk Free ---\n{storage_snapshot}\n\n--- Mounts ---\n{mounts}\n")

    # 4. Check package versions
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import polars as pl
    import duckdb
    import wfdb

    env_info = {
        "heedb_version": "5.0",
        "bdsp_url": "https://bdsp.io/content/heedb/5.0/",
        "s3_access_points": {
            "credentialed": "s3://bdsp-credentialed-ac-psbrsg8wcmky4w5tbtn3b31yh4otause1b-s3alias",
            "projects": "s3://bdsp-credentialed-pr-pakus5b5ruieu8mruai3jgxtcu56suse1b-s3alias"
        },
        "audit_timestamp_utc": now_utc,
        "hostname": platform.node(),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python_version": sys.version,
        "python_executable": sys.executable,
        "package_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
            "polars": pl.__version__,
            "duckdb": duckdb.__version__,
            "wfdb": wfdb.__version__
        },
        "git_commit": git_commit,
        "storage_mount": "/data (NFS: rsnfsprd.carleton.ca:/UtkarshDang)"
    }

    # AWS caller identity check
    try:
        import boto3
        sts = boto3.client('sts')
        ident = sts.get_caller_identity()
        env_info["aws_identity"] = {
            "arn": ident.get("Arn"),
            "account": ident.get("Account"),
            "user_id": ident.get("UserId"),
            "status": "VALID_CREDENTIALS"
        }
    except Exception as e:
        env_info["aws_identity"] = {
            "status": "NO_CREDENTIALS_OR_ROTATION_REQUIRED",
            "error": str(e)
        }

    with open(os.path.join(ENV_DIR, "environment.json"), "w") as f:
        json.dump(env_info, f, indent=2)

    # 5. Compute checksums of local source files
    source_files = [
        "/data/mithunmanivannan/heedb_metadata/EUH/metadata.csv",
        "/data/mithunmanivannan/heedb_metadata/EUH/diagnoses_acquisition.csv",
        "/data/mithunmanivannan/heedb_metadata/EUH/diagnoses_dictionary.csv",
        "/data/mithunmanivannan/heedb_metadata/MGH/metadata.csv",
        "/data/mithunmanivannan/heedb_metadata/MGH/diagnoses_acquisition.csv",
        "/data/mithunmanivannan/heedb_metadata/MGH/diagnoses_dictionary.csv",
        "/data/mithunmanivannan/papers/Nature_Scientific_Data_2026_HEEDB.pdf",
        "/data/mithunmanivannan/papers/Nature_Machine_Intelligence_2026_CSFM.pdf"
    ]

    print("Computing SHA256 hashes and row counts for local source files via pipelined prefetch...")
    checksum_records = []
    for fp in source_files:
        if os.path.exists(fp):
            print(f"Hashing: {fp} (size: {os.path.getsize(fp) / (1024*1024):.2f} MB)")
            stats = compute_file_stats(fp)
            print(f"  Done: SHA256={stats['sha256'][:12]}..., Rows={stats['row_count']}")
            checksum_records.append(stats)
        else:
            print(f"Missing file: {fp}")

    # Write source_file_checksums.csv
    import csv
    with open(os.path.join(ENV_DIR, "source_file_checksums.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "sha256", "size_bytes", "mtime_utc", "row_count"])
        writer.writeheader()
        writer.writerows(checksum_records)

    print("\nStage A completed successfully.")
    print(f"Checksum records saved to {os.path.join(ENV_DIR, 'source_file_checksums.csv')}")

if __name__ == "__main__":
    main()
