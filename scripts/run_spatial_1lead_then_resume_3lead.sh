#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source "$HOME/.venv/bin/activate"
python3 scripts/run_spatial_1lead_queue.py
exec python3 scripts/run_3arch_queue.py
