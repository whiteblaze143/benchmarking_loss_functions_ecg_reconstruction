#!/usr/bin/env python3
"""Generate all isolated factorial-v4 poster candidate figures."""

from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
SCRIPTS = [
    "fig1_loss_landscape.py",
    "fig2_component_effects.py",
    "fig3_morphology_utility.py",
    "fig4_robustness_external.py",
    "fig5_smartwatch_domain_gap.py",
]


def main() -> None:
    for script in SCRIPTS:
        subprocess.run([sys.executable, str(HERE / script)], check=True)
    print(f"Generated {len(SCRIPTS)} poster figures")


if __name__ == "__main__":
    main()
