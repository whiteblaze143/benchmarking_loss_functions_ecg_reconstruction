#!/usr/bin/env python3
"""Master pipeline runner for AF-DENSITY V5.
Orchestrates Phase A through Phase N in sequence:
  1. Build canonical timelines & run inventory
  2. Build standardized observation windows
  3. Compute reference AF density & validation (Gate 1)
  4. Experiment E01: Temporal aggregation sensitivity
  5. Experiment E01: Continuity identifiability & bounds
  6. Extract ventricular rates & physiological endpoints
  7. Experiment E02: Common support feasibility (Gate 2)
  8. Experiment E02: Conditional physiology & matching
  9. Experiment E02: Within-patient sensitivity
 10. Experiment E03: Temporal localization of rapid AF
 11. Experiment E03: Conditional randomization tests
 12. Phase G: Episode-order decomposition
 13. Phase H: SHDB clinical & pharmacological context
 14. Phase I: Noise robustness testing
 15. Phase K: Generate publication figures 1-9
 16. Phase M & N: Build programmatic master report & deliverables
"""

import sys
import subprocess
import time
from pathlib import Path

root_dir = Path(__file__).resolve().parents[1]
python_bin = sys.executable

SCRIPTS = [
    ("01_build_canonical_timelines.py", "Phase A01: Canonical Timelines"),
    ("02_build_observation_windows.py", "Phase A02: Observation Windows"),
    ("03_compute_reference_density.py", "Phase B: Density Computation & Gate 1"),
    ("04_temporal_aggregation.py", "Phase C02: E01 Aggregation Sensitivity"),
    ("05_density_identifiability.py", "Phase C03/C04: E01 Continuity Identifiability"),
    ("06_extract_ventricular_rates.py", "Phase D: Ventricular Rates Extraction"),
    ("07_common_support_feasibility.py", "Phase E03: Gate 2 Common Support"),
    ("08_conditional_physiology.py", "Phase E04-E06: E02 Conditional Physiology"),
    ("09_within_patient_analysis.py", "Phase E07: Within-Patient Sensitivity"),
    ("10_rapid_af_localization.py", "Phase F02-F03: E03 Localization"),
    ("11_conditional_randomization.py", "Phase F04: E03 Randomization Tests"),
    ("12_episode_order_decomposition.py", "Phase G: Episode-Order Decomposition"),
    ("13_shdb_clinical_context.py", "Phase H: SHDB Clinical Context"),
    ("14_noise_validation.py", "Phase I: Noise Robustness"),
    ("15_generate_figures.py", "Phase K: Figures 1-9"),
    ("16_build_master_report.py", "Phase M & N: Master Report")
]

def main():
    print("=" * 80)
    print("AF-DENSITY V5: MASTER PIPELINE ORCHESTRATION")
    print("=" * 80)

    total_start = time.time()
    for script_name, desc in SCRIPTS:
        script_path = root_dir / "scripts" / script_name
        print(f"\n>>> Running {desc} ({script_name})...")
        t0 = time.time()
        res = subprocess.run([python_bin, str(script_path)], cwd=root_dir)
        elapsed = time.time() - t0
        if res.returncode != 0:
            print(f"FAILED: {script_name} exited with code {res.returncode}")
            sys.exit(res.returncode)
        print(f"Completed {script_name} in {elapsed:.1f}s")

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 80)
    print(f"ALL PHASES COMPLETED SUCCESSFULLY IN {total_elapsed/60.0:.1f} MINUTES.")
    print("=" * 80)

if __name__ == "__main__":
    main()
