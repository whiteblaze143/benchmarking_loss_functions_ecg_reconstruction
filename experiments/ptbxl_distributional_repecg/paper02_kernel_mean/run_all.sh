#!/usr/bin/env bash
set -euo pipefail

paper_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python=/home/mithunmanivannan/.venv/bin/python
export PYTHONPATH="$paper_dir/../src${PYTHONPATH:+:$PYTHONPATH}"

"$python" "$paper_dir/prepare.py"
"$python" "$paper_dir/build_representation.py"
"$python" "$paper_dir/robustness.py"
"$python" "$paper_dir/train.py"
"$python" "$paper_dir/evaluate.py"
"$python" "$paper_dir/ablations.py"
"$python" "$paper_dir/figures.py"
