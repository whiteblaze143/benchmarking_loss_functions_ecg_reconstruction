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
import shutil
import platform

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

def compute_file_stats(filepath):
    h = hashlib.sha256()
    size = os.path.getsize(filepath)
    mtime = datetime.datetime.fromtimestamp(os.path.getmtime(filepath), datetime.timezone.utc).isoformat()
    
    # Check if text file for line counting
    line_count = 0
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(1024 * 1024 * 8)
            if not chunk:
                break
            h.update(chunk)
            if filepath.endswith('.csv'):
                line_count += chunk.count(b'\n')
                
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

    print("Computing SHA256 hashes and row counts for local source files...")
    checksum_records = []
    for fp in source_files:
        if os.path.exists(fp):
            print(f"Hashing: {fp}")
            stats = compute_file_stats(fp)
            checksum_records.append(stats)
        else:
            print(f"Missing file: {fp}")

    # Write source_file_checksums.csv
    import csv
    with open(os.path.join(ENV_DIR, "source_file_checksums.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "sha256", "size_bytes", "mtime_utc", "row_count"])
        writer.writeheader()
        writer.writerows(checksum_records)

    print("Stage A completed successfully.")
    print(f"Checksum records saved to {os.path.join(ENV_DIR, 'source_file_checksums.csv')}")

if __name__ == "__main__":
    main()
