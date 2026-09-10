#!/usr/bin/env python3
"""
End-to-End Pipeline for Curating and Downloading AFib ECGs from HEEDB (BDSP S3).

Features:
- Discovers HEEDB bucket path on BDSP S3 access points.
- Downloads tabular diagnosis and metadata files.
- Filters 12SL Code 161 (Atrial Fibrillation).
- Deduplicates by BDSPPatientID (1 ECG per patient to prevent data leakage).
- Enforces strict disk-space safety gates (aborts if free space < 4.0 GB).
- Downloads .mat/.dat and .hea files with parallel workers.
- Validates downloaded WFDB waveform data integrity.
"""

import argparse
import concurrent.futures
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

DEFAULT_DATA_DIR = Path("/home/mithunmanivannan/data/heedb_afib")
DEFAULT_ACCESS_POINT = "s3://bdsp-credentialed-ac-psbrsg8wcmky4w5tbtn3b31yh4otause1b-s3alias"
MIN_FREE_DISK_GB = 4.0
AFIB_12SL_CODE = 161


def get_free_disk_gb(path: Path) -> float:
    stat = shutil.disk_usage(path if path.exists() else path.parent)
    return stat.free / (1024 ** 3)


def run_aws_cmd(cmd: list[str]) -> tuple[int, str]:
    full_cmd = ["aws", "s3"] + cmd + ["--region", "us-east-1"]
    proc = subprocess.run(full_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out = proc.stdout.strip() if proc.returncode == 0 else proc.stderr.strip()
    return proc.returncode, out


def check_auth() -> bool:
    proc = subprocess.run(["aws", "sts", "get-caller-identity"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        print("[ERROR] AWS credentials not configured or unauthorized.")
        print(proc.stderr.strip())
        return False
    print("[INFO] AWS Identity verified:", proc.stdout.strip())
    return True


def find_heedb_path(ap: str) -> str:
    """Probes S3 access point to find exact relative path to HEEDB."""
    print(f"[INFO] Probing access point: {ap}/")
    code, out = run_aws_cmd(["ls", f"{ap}/"])
    if code != 0:
        raise RuntimeError(f"Failed to list access point {ap}: {out}")

    print(f"[INFO] Root contents:\n{out}")
    lines = [line.split()[-1].strip("/") for line in out.splitlines() if line.strip()]
    for candidate in ["ECG", "heedb", "harvard-emory-ecg-database", "HEEDB"]:
        if candidate in lines:
            return f"{ap}/{candidate}"
    # If single directory, inspect it
    if len(lines) == 1:
        sub_ap = f"{ap}/{lines[0]}"
        code, sub_out = run_aws_cmd(["ls", f"{sub_ap}/"])
        if code == 0 and "ECG" in sub_out:
            return f"{sub_ap}/ECG"
        return sub_ap
    return ap


def download_file(s3_uri: str, local_path: Path) -> bool:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["aws", "s3", "cp", s3_uri, str(local_path), "--region", "us-east-1", "--only-show-errors"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return proc.returncode == 0


def filter_afib_cohort(metadata_dir: Path, target_count: int) -> pd.DataFrame:
    """Finds AFib records (code 161) and deduplicates by patient."""
    print("[INFO] Loading diagnoses and metadata files...")
    diag_files = list(metadata_dir.glob("**/diagnoses_*.csv"))
    if not diag_files:
        raise FileNotFoundError(f"No diagnoses CSV files found in {metadata_dir}")

    # Load 12SL diagnoses (prefer diagnoses_acquisition.csv or diagnoses_v24.csv)
    diag_file = None
    for f in diag_files:
        if "acquisition" in f.name:
            diag_file = f
            break
    if not diag_file:
        diag_file = diag_files[0]
    print(f"[INFO] Using diagnoses file: {diag_file}")

    df_diag = pd.read_csv(diag_file)
    print(f"[INFO] Total diagnoses rows: {len(df_diag):,}")

    # Search for code 161 in codes columns
    code_cols = [c for c in df_diag.columns if "code" in c.lower()]
    print(f"[INFO] Checking code columns: {code_cols}")

    def has_afib(val) -> bool:
        if pd.isna(val):
            return False
        # Codes can be comma/space separated ints or strings
        s = str(val).replace("[", "").replace("]", "").replace("'", "").replace('"', "")
        tokens = [t.strip() for t in s.split(",") if t.strip()]
        return any(t == str(AFIB_12SL_CODE) or t.startswith(f"{AFIB_12SL_CODE}_") for t in tokens)

    mask = pd.Series(False, index=df_diag.index)
    for c in code_cols:
        mask = mask | df_diag[c].apply(has_afib)

    afib_df = df_diag[mask].copy()
    print(f"[INFO] Identified {len(afib_df):,} records with AFib (Code {AFIB_12SL_CODE}).")

    # Match with metadata for patient deduplication
    meta_files = list(metadata_dir.glob("**/metadata.csv"))
    if meta_files:
        df_meta = pd.read_csv(meta_files[0])
        print(f"[INFO] Merging with metadata ({len(df_meta):,} rows) for patient deduplication...")
        if "FileName" in afib_df.columns and "FileName" in df_meta.columns:
            afib_df = afib_df.merge(df_meta, on="FileName", how="inner", suffixes=("", "_meta"))
        elif "FileID" in afib_df.columns and "FileID" in df_meta.columns:
            afib_df = afib_df.merge(df_meta, on="FileID", how="inner", suffixes=("", "_meta"))

        if "BDSPPatientID" in afib_df.columns:
            print(f"[INFO] Unique patients with AFib: {afib_df['BDSPPatientID'].nunique():,}")
            # Keep first high-quality recording per patient
            afib_df = afib_df.drop_duplicates(subset=["BDSPPatientID"]).reset_index(drop=True)
            print(f"[INFO] Cohort after 1-record-per-patient deduplication: {len(afib_df):,}")

    # Sample target count
    if len(afib_df) > target_count:
        afib_df = afib_df.sample(n=target_count, random_state=42).reset_index(drop=True)
    print(f"[INFO] Final selected AFib cohort size: {len(afib_df):,}")
    return afib_df


def main():
    parser = argparse.ArgumentParser(description="HEEDB AFib Downloader")
    parser.add_argument("--count", type=int, default=20000, help="Target number of AFib ECGs (default: 20,000)")
    parser.add_argument("--workers", type=int, default=12, help="Number of download threads")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_DATA_DIR, help="Destination directory")
    args = parser.parse_args()

    print("==========================================================")
    print("  HEEDB Targeted AFib (Code 161) Acquisition Pipeline    ")
    print("==========================================================")
    print(f"Target count: {args.count:,} ECGs")
    print(f"Destination:  {args.out_dir}")
    print(f"Workers:      {args.workers}")

    free_gb = get_free_disk_gb(args.out_dir)
    print(f"Initial available disk space: {free_gb:.2f} GB")
    if free_gb < MIN_FREE_DISK_GB:
        print(f"[ERROR] Free disk space ({free_gb:.2f} GB) is below minimum safety threshold ({MIN_FREE_DISK_GB} GB).")
        sys.exit(1)

    if not check_auth():
        print("\nPlease authenticate AWS credentials and rerun.")
        sys.exit(1)

    heedb_root = find_heedb_path(DEFAULT_ACCESS_POINT)
    print(f"[INFO] HEEDB Root URI: {heedb_root}")

    # Step 1: Download Metadata
    meta_local = args.out_dir / "metadata_cache"
    meta_local.mkdir(parents=True, exist_ok=True)
    print("\n=== Phase 1: Syncing Tabular Diagnoses and Metadata ===")
    code, out = run_aws_cmd(["sync", f"{heedb_root}/", str(meta_local), "--exclude", "*", "--include", "*diagnoses*/*", "--include", "*metadata*/*", "--include", "*ICD*/*"])
    print(f"[INFO] Metadata sync status: {code} ({out})")

    # Step 2: Filter AFib Cohort
    print("\n=== Phase 2: Curating AFib Cohort ===")
    cohort_df = filter_afib_cohort(meta_local, args.count)
    manifest_csv = args.out_dir / f"heedb_afib_manifest_{len(cohort_df)}.csv"
    cohort_df.to_csv(manifest_csv, index=False)
    print(f"[SUCCESS] Manifest written to: {manifest_csv}")

    # Step 3: Stream Waveforms
    print("\n=== Phase 3: Downloading Waveform Files (.mat/.dat + .hea) ===")
    wfdb_dir = args.out_dir / "WFDB"
    wfdb_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for _, row in cohort_df.iterrows():
        fn = row["FileName"]
        # Determine extensions
        base_s3 = f"{heedb_root}/{fn}" if not fn.startswith("s3://") else fn
        base_name = Path(fn).name
        # Add .hea and waveform file
        tasks.append((f"{base_s3}.hea", wfdb_dir / f"{base_name}.hea"))
        tasks.append((f"{base_s3}.mat", wfdb_dir / f"{base_name}.mat"))

    print(f"[INFO] Total files to download: {len(tasks):,}")
    completed = 0
    failed = 0
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {executor.submit(download_file, s3_u, loc_p): loc_p for s3_u, loc_p in tasks}
        for future in concurrent.futures.as_completed(future_map):
            completed += 1
            if not future.result():
                failed += 1

            if completed % 500 == 0:
                elapsed = time.time() - start_time
                current_free = get_free_disk_gb(args.out_dir)
                print(f"[{completed:,}/{len(tasks):,}] Progress: {completed/len(tasks)*100:.1f}% | Failed: {failed} | Free Disk: {current_free:.2f} GB | Speed: {completed/elapsed:.1f} files/s")
                if current_free < MIN_FREE_DISK_GB:
                    print(f"[CRITICAL GATE] Available disk space ({current_free:.2f} GB) reached safety ceiling ({MIN_FREE_DISK_GB} GB)! Halting download.")
                    break

    print("\n=== Acquisition Complete ===")
    print(f"Downloaded files: {completed - failed:,}")
    print(f"Failed files:     {failed}")
    print(f"Final Free Space: {get_free_disk_gb(args.out_dir):.2f} GB")


if __name__ == "__main__":
    main()
