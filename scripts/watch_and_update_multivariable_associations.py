#!/usr/bin/env python3
"""Daemon watcher that continuously monitors the 1-Lead evaluation queue,
detects whenever a new model finishes evaluation in clinical_metrics.db,
and automatically runs the multivariable clinical association analysis.
"""

import logging
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = _ROOT / "results/clinical_biomarkers_multids/clinical_metrics.db"
SCRIPT_PATH = _ROOT / "scripts/analyze_multivariable_clinical_associations.py"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [DAEMON] %(message)s",
    datefmt="%H:%M:%S"
)


def get_evaluated_models():
    if not DB_PATH.exists():
        return set()
    try:
        with sqlite3.connect(DB_PATH, timeout=30) as con:
            cur = con.cursor()
            cur.execute("SELECT DISTINCT model_id FROM clinical_metrics WHERE evaluation_version='1lead_clinical_v1'")
            return {r[0] for r in cur.fetchall()}
    except Exception:
        return set()


def main():
    logging.info("Starting Multivariable Clinical Association Daemon Watcher...")
    last_models = set()

    while True:
        current_models = get_evaluated_models()
        new_models = current_models - last_models

        if new_models:
            logging.info("Detected %d new model(s): %s", len(new_models), list(new_models))
            logging.info("Triggering multivariable association, MMRM, & 28-endpoint atlas analysis...")
            cmd1 = [sys.executable, str(SCRIPT_PATH)]
            res1 = subprocess.run(cmd1, cwd=str(_ROOT), capture_output=True, text=True)
            cmd2 = [sys.executable, str(_ROOT / "scripts/run_clinical_mmrm_analysis.py")]
            res2 = subprocess.run(cmd2, cwd=str(_ROOT), capture_output=True, text=True)
            cmd3 = [sys.executable, str(_ROOT / "scripts/compute_deep_clinical_endpoints.py")]
            res3 = subprocess.run(cmd3, cwd=str(_ROOT), capture_output=True, text=True)
            if res1.returncode == 0 and res2.returncode == 0 and res3.returncode == 0:
                logging.info("✓ Multivariable, MMRM, & 28-Endpoint Atlas successfully updated! Total models now: %d", len(current_models))
            else:
                logging.error("Failed in background update:\n%s\n%s\n%s", res1.stderr, res2.stderr, res3.stderr)
            last_models = current_models

        time.sleep(30)


if __name__ == "__main__":
    main()
