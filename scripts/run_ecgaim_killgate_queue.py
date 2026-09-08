#!/usr/bin/env python3
"""Sequential Experiment Queue for ECG-AIM Complexity / Necessity Kill-Gate.

Executes a comprehensive 15-cell ablation matrix covering:
  - Phase K1: Training Machinery & Loss Kill-Gate (Masking, MTL Head, L1 vs Composite vs Adaptive)
  - Phase K2: Physics & Deterministic Limb Derivation (Einthoven / Goldberger Algebraic Basis)
  - Phase K3: Structural Capacity Pruning (Depth 8->4, Decoder 4->2, Width 768->512, Minimal Backbone)
  - Phase K4: Representation & Normalization (Z-Score Standardization)

After each run completes, automatically triggers paired bootstrap evaluation against anchor
(conv15e_A0_raw_s42_l0) across all held-out validation records (Fold 9, N=2,183).
"""

from __future__ import annotations
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = sys.executable or "/home/mithunmanivannan/.venv/bin/python3"
TRAINER = _ROOT / "scripts/train_1lead_wavelet_ssl_mtl.py"
EVALUATOR = _ROOT / "scripts/evaluate_killgate_bootstrap.py"
OUTPUT_ROOT = _ROOT / "refine-logs/killgate/runs"
ANCHOR_DIR = _ROOT / "refine-logs/convergence_10e/runs/conv15e_A0_raw_s42_l0"
LEDGER_FILE = _ROOT / "refine-logs/killgate/summary_ledger.json"

# Common hyperparameters strictly inherited from anchor conv15e_A0_raw_s42_l0
BASE_ARGS = [
    "--observed-leads", "0",
    "--epochs", "15",
    "--batch-size", "32",
    "--lr", "0.0001",
    "--max-lr", "0.0005",
    "--weight-decay", "0.0001",
    "--pct-start", "0.2",
    "--delineation-every", "2",
    "--delineation-dir", "data/rdb_wavelet_delineation_cache",
    "--no-use-wavelet-branch",
    "--ssl-mode", "none",
    "--width", "768",
    "--encoder-depth", "8",
    "--decoder-depth", "4",
    "--heads", "12",
    "--patch-size", "25",
    "--factorial-mask", "1110000",
    "--seed", "42",
    "--checkpoint-policy", "best",
    "--rolling-resume",
]

EXPERIMENTS = [
    # -------------------------------------------------------------------------
    # Phase K1: Training Machinery & Loss Kill-Gate
    # -------------------------------------------------------------------------
    {
        "id": "K1_nold",
        "name": "conv15e_K1_nold_s42_l0",
        "desc": "Kill-Gate 1: Remove whole-lead source dropout regime",
        "extra_args": [
            "--artificial-mask-mode", "no_lead_dropout",
            "--seg-ce-weight", "1.0",
            "--dice-weight", "0.5",
        ],
    },
    {
        "id": "K2_noart",
        "name": "conv15e_K2_noart_s42_l0",
        "desc": "Kill-Gate 2: Remove all artificial masking (M_art = 0)",
        "extra_args": [
            "--artificial-mask-mode", "none",
            "--seg-ce-weight", "1.0",
            "--dice-weight", "0.5",
        ],
    },
    {
        "id": "K3_nodel",
        "name": "conv15e_K3_nodel_s42_l0",
        "desc": "Kill-Gate 3: Remove delineation head & auxiliary losses (pure reconstruction)",
        "extra_args": [
            "--no-delineation-head",
            "--seg-ce-weight", "0.0",
            "--dice-weight", "0.0",
        ],
    },
    {
        "id": "K4_l1",
        "name": "conv15e_K4_l1_s42_l0",
        "desc": "Kill-Gate 4: Pure L1 reconstruction objective",
        "extra_args": [
            "--reconstruction-loss-type", "l1",
            "--seg-ce-weight", "1.0",
            "--dice-weight", "0.5",
        ],
    },
    {
        "id": "K5_adaptive",
        "name": "conv15e_K5_adaptive_s42_l0",
        "desc": "Kill-Gate 5: Composite loss with adaptive homoscedastic uncertainty weighting",
        "extra_args": [
            "--reconstruction-loss-type", "adaptive_composite",
            "--seg-ce-weight", "1.0",
            "--dice-weight", "0.5",
        ],
    },
    # -------------------------------------------------------------------------
    # Phase K2: Physics & Deterministic Limb Derivation
    # -------------------------------------------------------------------------
    {
        "id": "B1_hardbasis",
        "name": "conv15e_B1_hardbasis_s42_l0",
        "desc": "Kill-Gate 6 (Physics): Deterministic Einthoven limb lead derivation",
        "extra_args": [
            "--deterministic-limb-derivation",
            "--seg-ce-weight", "1.0",
            "--dice-weight", "0.5",
        ],
    },
    {
        "id": "B2_hardbasis_nodel",
        "name": "conv15e_B2_hardbasis_nodel_s42_l0",
        "desc": "Kill-Gate 7: Deterministic Einthoven + Pure Reconstruction (No MTL Head)",
        "extra_args": [
            "--deterministic-limb-derivation",
            "--no-delineation-head",
            "--seg-ce-weight", "0.0",
            "--dice-weight", "0.0",
        ],
    },
    {
        "id": "B3_hardbasis_adaptive",
        "name": "conv15e_B3_hardbasis_adaptive_s42_l0",
        "desc": "Kill-Gate 8: Deterministic Einthoven + Adaptive Composite Loss",
        "extra_args": [
            "--deterministic-limb-derivation",
            "--reconstruction-loss-type", "adaptive_composite",
            "--seg-ce-weight", "1.0",
            "--dice-weight", "0.5",
        ],
    },
    {
        "id": "B4_hardbasis_l1",
        "name": "conv15e_B4_hardbasis_l1_s42_l0",
        "desc": "Kill-Gate 9: Deterministic Einthoven + Pure L1 Objective",
        "extra_args": [
            "--deterministic-limb-derivation",
            "--reconstruction-loss-type", "l1",
            "--seg-ce-weight", "1.0",
            "--dice-weight", "0.5",
        ],
    },
    # -------------------------------------------------------------------------
    # Phase K3: Structural Capacity Pruning
    # -------------------------------------------------------------------------
    {
        "id": "C1_enc4",
        "name": "conv15e_C1_enc4_s42_l0",
        "desc": "Kill-Gate 10 (Capacity): Halved encoder depth (E=4 instead of 8)",
        "extra_args": [
            "--encoder-depth", "4",
            "--seg-ce-weight", "1.0",
            "--dice-weight", "0.5",
        ],
    },
    {
        "id": "C2_dec2",
        "name": "conv15e_C2_dec2_s42_l0",
        "desc": "Kill-Gate 11 (Capacity): Halved decoder depth (D=2 instead of 4)",
        "extra_args": [
            "--decoder-depth", "2",
            "--seg-ce-weight", "1.0",
            "--dice-weight", "0.5",
        ],
    },
    {
        "id": "C3_width512",
        "name": "conv15e_C3_width512_s42_l0",
        "desc": "Kill-Gate 12 (Capacity): Pruned hidden width (W=512, H=8)",
        "extra_args": [
            "--width", "512",
            "--heads", "8",
            "--seg-ce-weight", "1.0",
            "--dice-weight", "0.5",
        ],
    },
    {
        "id": "C4_minimal",
        "name": "conv15e_C4_minimal_s42_l0",
        "desc": "Kill-Gate 13 (Capacity): Minimal architecture (E=4, D=2, W=512, H=8)",
        "extra_args": [
            "--encoder-depth", "4",
            "--decoder-depth", "2",
            "--width", "512",
            "--heads", "8",
            "--seg-ce-weight", "1.0",
            "--dice-weight", "0.5",
        ],
    },
    # -------------------------------------------------------------------------
    # Phase K4: Representation & Normalization
    # -------------------------------------------------------------------------
    {
        "id": "Z1_zscore",
        "name": "conv15e_Z1_zscore_s42_l0",
        "desc": "Kill-Gate 14 (Norm): Per-sample Z-score standardization",
        "extra_args": [
            "--zscore-norm",
            "--seg-ce-weight", "1.0",
            "--dice-weight", "0.5",
        ],
    },
    {
        "id": "Z2_hardbasis_zscore",
        "name": "conv15e_Z2_hardbasis_zscore_s42_l0",
        "desc": "Kill-Gate 15 (Combined): Deterministic Einthoven + Z-score standardization",
        "extra_args": [
            "--deterministic-limb-derivation",
            "--zscore-norm",
            "--seg-ce-weight", "1.0",
            "--dice-weight", "0.5",
        ],
    },
]


def log(msg: str):
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def update_ledger(exp_id: str, run_name: str, status: str, eval_data: dict = None):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    ledger = {}
    if LEDGER_FILE.is_file():
        try:
            ledger = json.loads(LEDGER_FILE.read_text())
        except Exception:
            pass
    ledger[exp_id] = {
        "run_name": run_name,
        "status": status,
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "eval": eval_data,
    }
    LEDGER_FILE.write_text(json.dumps(ledger, indent=2) + "\n")


def run_experiment(exp: dict) -> bool:
    run_dir = OUTPUT_ROOT / exp["name"]
    success_file = run_dir / "_SUCCESS.json"
    eval_file = run_dir / "killgate_eval_val.json"

    log("=" * 70)
    log(f"STARTING {exp['id']}: {exp['name']}")
    log(f"Description: {exp['desc']}")

    if success_file.is_file():
        log(f"Run {exp['name']} already completed (_SUCCESS.json exists). Skipping training.")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            PYTHON_BIN, str(TRAINER),
            "--run-name", exp["name"],
            "--output-dir", str(run_dir),
            *BASE_ARGS,
            *exp["extra_args"],
        ]
        log(f"Executing: {' '.join(cmd)}")
        train_log = run_dir / "train.log"
        with open(train_log, "w") as f_out:
            p = subprocess.Popen(cmd, cwd=str(_ROOT), stdout=f_out, stderr=subprocess.STDOUT)
            ret = p.wait()

        if ret != 0 or not success_file.is_file():
            log(f"ERROR: {exp['name']} failed with returncode {ret}. Check {train_log}")
            update_ledger(exp["id"], exp["name"], f"FAILED (code {ret})")
            return False

        log(f"Training completed successfully for {exp['name']}!")

    # Evaluate paired bootstrap
    log(f"Running paired bootstrap evaluation against anchor for {exp['name']}...")
    eval_cmd = [
        PYTHON_BIN, str(EVALUATOR),
        "--candidate-dir", str(run_dir),
        "--anchor-dir", str(ANCHOR_DIR),
        "--split", "val",
        "--n-boot", "10000",
        "--batch-size", "64",
    ]
    eval_proc = subprocess.run(eval_cmd, cwd=str(_ROOT), capture_output=True, text=True)
    if eval_proc.returncode != 0:
        log(f"Evaluation error: {eval_proc.stderr}")
    else:
        log("Evaluation output:\n" + eval_proc.stdout)

    eval_data = None
    if eval_file.is_file():
        try:
            eval_data = json.loads(eval_file.read_text())
        except Exception:
            pass

    status = "COMPLETED"
    if eval_data:
        verdict = "PASS" if eval_data.get("gates", {}).get("pass_overall") else "FAIL"
        status = f"COMPLETED ({verdict})"

    update_ledger(exp["id"], exp["name"], status, eval_data)
    return True


def main():
    log("Starting ECG-AIM Kill-Gate Experiment Queue...")
    log(f"Queue contains {len(EXPERIMENTS)} experiments:")
    for e in EXPERIMENTS:
        log(f"  - {e['id']}: {e['name']} ({e['desc']})")

    for exp in EXPERIMENTS:
        ok = run_experiment(exp)
        if not ok:
            log(f"Halting queue due to failure in {exp['id']}.")
            sys.exit(1)

    log("=" * 70)
    log("ALL KILL-GATE EXPERIMENTS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
