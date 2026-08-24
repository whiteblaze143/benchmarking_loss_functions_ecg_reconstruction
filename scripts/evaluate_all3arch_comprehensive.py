#!/usr/bin/env python3
"""
Comprehensive 3-Architecture Continuous Evaluation Watcher.
Monitors completed checkpoints across UNet, MS-VAE, and ECG-AIM and executes:
1. Signal Quality & Missing Lead Metrics (12 leads vs GT)
2. External Wave Delineation (LUDB, ISP, Zhejiang)
3. Multi-Dataset Clinical Biomarkers (PTB-XL, EchoNext, LUDB, ISP, Zhejiang, Sunnybrook)
Enforces <= 2 CPU threads to protect CPU resources.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import json
import logging
import sys
import time
import subprocess
import torch
from pathlib import Path

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_SCRIPT = ROOT / "scripts" / "generate_full_model_registry.py"
CLINICAL_EVAL_SCRIPT = ROOT / "scripts" / "evaluate_clinical_biomarkers_multids.py"

def main():
    print("Starting Comprehensive 3-Architecture Continuous Evaluator Watcher...")
    while True:
        try:
            # 1. Update model registry with all completed checkpoints
            subprocess.run([sys.executable, str(REGISTRY_SCRIPT)], check=True)
            
            # 2. Run multi-dataset clinical biomarker & signal quality evaluator
            subprocess.run([sys.executable, str(CLINICAL_EVAL_SCRIPT)], check=True)
            
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Evaluation pass complete. Sleeping 120s...")
        except Exception as e:
            print(f"Evaluation watcher exception: {e}")
            
        time.sleep(120)

if __name__ == "__main__":
    main()
