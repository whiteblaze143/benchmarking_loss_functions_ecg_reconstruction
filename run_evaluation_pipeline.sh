#!/bin/bash
# 1-Lead Factorial Evaluation Pipeline

echo "Running MultiDS Evaluation Daemon on CPU..."
# Export OMP limits as requested to avoid OOMing the background training queue
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export CUDA_VISIBLE_DEVICES=""  # Force CPU execution

# Run the MultiDS Evaluation
~/.venv/bin/python3 scripts/evaluate_clinical_biomarkers_multids.py

echo "MultiDS Evaluation Completed."

echo "Running Linear Probing against EchoNext targets..."
# Run the Linear Probes for the downstream clinical targets
~/.venv/bin/python3 scripts/evaluate_linear_probe.py

echo "Evaluation Pipeline Completed!"
