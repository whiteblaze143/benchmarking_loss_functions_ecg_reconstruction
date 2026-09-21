#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR=/data/mithunmanivannan/echonext
mkdir -p "$TARGET_DIR"

LOG="$TARGET_DIR/download.log"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting EchoNext download from PhysioNet to $TARGET_DIR" | tee -a "$LOG"

wget -r -N -c -np \
    --user=manivanm \
    --password='Mitch@143' \
    -nd \
    -P "$TARGET_DIR" \
    https://physionet.org/files/echonext/1.1.1/ 2>&1 | tee -a "$LOG"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] EchoNext download complete!" | tee -a "$LOG"
