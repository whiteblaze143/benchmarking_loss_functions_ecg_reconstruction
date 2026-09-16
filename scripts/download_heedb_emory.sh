#!/usr/bin/env bash
set -e

DEST_DIR="/data/mithunmanivannan/heedb_emory"
S3_URI="s3://bdsp-credentialed-ac-psbrsg8wcmky4w5tbtn3b31yh4otause1b-s3alias/ECG/I0006"

: "${AWS_ACCESS_KEY_ID:?AWS_ACCESS_KEY_ID must be set in the environment}"
: "${AWS_SECRET_ACCESS_KEY:?AWS_SECRET_ACCESS_KEY must be set in the environment}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"

mkdir -p "$DEST_DIR"

echo "================================================================================"
echo "  HEEDB EMORY (I0006) WAVEFORM & METADATA ACQUISITION PIPELINE"
echo "  Target NFS Directory: $DEST_DIR"
echo "  Available NFS Space:  $(df -h "$DEST_DIR" | awk 'NR==2 {print $4}')"
echo "  Source S3 URI:        $S3_URI"
echo "================================================================================"

echo "[INFO] Starting aws s3 sync for Emory ECGs (~124 GB)..."
start_time=$(date +%s)

aws s3 sync "$S3_URI/" "$DEST_DIR/" \
    --region "$AWS_DEFAULT_REGION" \
    --only-show-errors

end_time=$(date +%s)
elapsed=$((end_time - start_time))

echo "================================================================================"
echo "  HEEDB EMORY DOWNLOAD COMPLETE in ${elapsed}s!"
echo "  Final Directory Size: $(du -sh "$DEST_DIR" | cut -f1)"
echo "  Remaining NFS Space:  $(df -h "$DEST_DIR" | awk 'NR==2 {print $4}')"
echo "================================================================================"
