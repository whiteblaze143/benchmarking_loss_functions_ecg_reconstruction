#!/usr/bin/env bash
# Sync tabular diagnoses and metadata from BDSP S3 to the 500GB NFS share (/data/mithunmanivannan/heedb_metadata).
# STRICT SAFETY GATE: Only tabular CSV/TSV/metadata files are synced. ZERO waveform files (.mat/.dat/.hea) will be pulled.
set -euo pipefail

DEST_DIR="/data/mithunmanivannan/heedb_metadata"
ACCESS_POINT="s3://bdsp-credentialed-ac-psbrsg8wcmky4w5tbtn3b31yh4otause1b-s3alias"
MIN_FREE_GB=20.0

mkdir -p "${DEST_DIR}"

# Check available disk space
FREE_GB=$(df -BG "${DEST_DIR}" | awk 'NR==2 {gsub("G","",$4); print $4}')
echo "Available space on /data: ${FREE_GB} GB (Minimum required: ${MIN_FREE_GB} GB)"
if (( FREE_GB < MIN_FREE_GB )); then
  echo "[CRITICAL ERROR] Insufficient disk space on /data."
  exit 1
fi

echo "=========================================================================="
echo "  Syncing HEEDB Tabular Metadata & Diagnoses to NFS"
echo "  Source:      ${ACCESS_POINT}/"
echo "  Destination: ${DEST_DIR}"
echo "  Filters:     Include diagnoses/metadata/ICD CSVs only; Exclude waveforms"
echo "=========================================================================="

aws s3 sync "${ACCESS_POINT}/" "${DEST_DIR}" \
  --region us-east-1 \
  --exclude "*" \
  --include "*diagnoses*" \
  --include "*diagnoses*/*" \
  --include "*metadata*" \
  --include "*metadata*/*" \
  --include "*ICD*" \
  --include "*ICD*/*" \
  --include "*.csv" \
  --include "*.tsv" \
  --include "*.parquet" \
  --exclude "*.mat" \
  --exclude "*.dat" \
  --exclude "*.hea" \
  --exclude "*.wfdb"

echo "[SUCCESS] Metadata sync completed successfully."
