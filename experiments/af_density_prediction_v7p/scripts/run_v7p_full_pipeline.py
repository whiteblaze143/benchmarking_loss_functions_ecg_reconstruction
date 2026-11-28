"""Master Pipeline Runner for V7-P.

Orchestrates:
1. Feature extraction & causal embedding caching (cache_all_embeddings.py)
2. Complete 13-Stage Information-Ablation Matrix evaluation (run_13stage_ablations.py)
3. SetOperator Downstream Heads audit (Linear + Discrete hazard)
4. Mandatory Statistical Contrasts (B=500 patient bootstrap)
"""

import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SRC_DIR = PROJECT_DIR / "src"

for p in [SRC_DIR, PROJECT_DIR, SCRIPT_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import cache_all_embeddings
import run_13stage_ablations


def main():
    t_start = time.time()
    print("=" * 80)
    print("V7-P MASTER EXECUTION SUITE: GPU FOUNDATION EXTRACTION & 13-STAGE ABLATIONS")
    print("=" * 80)

    # Step 1: Cache All Embeddings
    print("\n[Phase 1] Caching Foundation Model and SetOperator Embeddings (10s, 30s, 60s)...")
    cache_all_embeddings.main()

    # Step 2: Run Complete 13-Stage Ablation Matrix
    print("\n[Phase 2] Running 13-Stage Information-Ablation Matrix & Contrasts...")
    run_13stage_ablations.main()

    print("\n" + "=" * 80)
    print(f"V7-P PIPELINE COMPLETE IN {time.time() - t_start:.1f}s!")
    print("=" * 80)


if __name__ == "__main__":
    main()
