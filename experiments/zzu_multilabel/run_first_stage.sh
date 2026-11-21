#!/usr/bin/env bash
set -euo pipefail
cd /home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction
hrs_cuda_libs=$(find /home/mithunmanivannan/.venv/lib/python3.12/site-packages/nvidia -maxdepth 2 -type d -name lib -printf '%p:')
export LD_LIBRARY_PATH="/data/mithunmanivannan/codex_artifacts/hrs2027_af_recurrence/runtime_cudnn93/nvidia/cudnn/lib:$hrs_cuda_libs"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
study_out=/data/mithunmanivannan/codex_artifacts/zzu_multilabel_20261120
test -f "$study_out/cache_receipt.json"
/home/mithunmanivannan/.venv/bin/python experiments/zzu_multilabel/study.py train --family raw --config full12 > "$study_out/raw_full12.log" 2>&1
/home/mithunmanivannan/.venv/bin/python experiments/zzu_multilabel/study.py train --family raw --config II > "$study_out/raw_II.log" 2>&1
/home/mithunmanivannan/.venv/bin/python experiments/zzu_multilabel/study.py represent > "$study_out/represent.log" 2>&1
/home/mithunmanivannan/.venv/bin/python experiments/zzu_multilabel/study.py train --family set --config full12 > "$study_out/set_full12.log" 2>&1
/home/mithunmanivannan/.venv/bin/python experiments/zzu_multilabel/study.py train --family set --config II > "$study_out/set_II.log" 2>&1
/home/mithunmanivannan/.venv/bin/python experiments/zzu_multilabel/analyze.py > "$study_out/analysis.log" 2>&1
