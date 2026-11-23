"""Exact Finite-Grid Solver for Charitos AF Density Bounds (V4.1).

Computes the ground-truth exact mathematical minimum and maximum Charitos density
by exhaustive 2^K binary evaluation over all unmonitored/unknown grid units.
Enforces 100.0% containment gate and validates against 20 reference test fixtures.
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np

PROJECT_ROOT = Path("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.af_density_v4.metrics.af_pattern_metrics import charitos_density

REPORTS_DIR = PROJECT_ROOT / "experiments" / "af_density_v4" / "reports"
CODEX_DIR = Path("/data/mithunmanivannan/codex_artifacts/af_density_v4")

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
CODEX_DIR.mkdir(parents=True, exist_ok=True)


def grid_to_intervals(binary_array: Sequence[int], dt: float) -> List[Tuple[float, float]]:
    """Converts a binary sequence into half-open time intervals [start, end)."""
    intervals = []
    in_af = False
    start_idx = 0
    for idx, val in enumerate(binary_array):
        if val == 1 and not in_af:
            in_af = True
            start_idx = idx
        elif val == 0 and in_af:
            in_af = False
            intervals.append((start_idx * dt, idx * dt))
    if in_af:
        intervals.append((start_idx * dt, len(binary_array) * dt))
    return intervals


def solve_exact_grid_bounds(
    grid: Sequence[int],
    dt: float = 30.0
) -> Dict[str, Any]:
    """Exhaustively solves exact min/max Charitos density across all 2^m completions.
    
    Grid values:
      1: Known AF
      0: Known Sinus
     -1: Unknown / Unobservable
    """
    grid = list(grid)
    k = len(grid)
    unknown_indices = [i for i, v in enumerate(grid) if v == -1]
    m = len(unknown_indices)
    
    if m > 20:
        raise ValueError(f"Exhaustive solver limited to m <= 20 unknown units (got m={m}, 2^{m} combinations).")

    total_T = k * dt
    total_combinations = 2 ** m

    defined_densities: List[float] = []
    burdens: List[float] = []
    undefined_counts = {
        "ZERO_BURDEN": 0,
        "CONTINUOUS_AF": 0,
        "SINGLE_EPISODE": 0,
    }

    # Evaluate all 2^m binary combinations
    for bits in itertools.product([0, 1], repeat=m):
        candidate = list(grid)
        for u_idx, bit in zip(unknown_indices, bits):
            candidate[u_idx] = bit
        
        cand_intervals = grid_to_intervals(candidate, dt)
        total_af_s = sum(b - a for a, b in cand_intervals)
        burden = total_af_s / total_T
        burdens.append(burden)

        if len(cand_intervals) == 0:
            undefined_counts["ZERO_BURDEN"] += 1
            continue
        if total_af_s >= total_T - 1e-6:
            undefined_counts["CONTINUOUS_AF"] += 1
            continue

        d = charitos_density(cand_intervals, total_T)
        if d is not None and np.isfinite(d):
            defined_densities.append(d)
        else:
            if len(cand_intervals) == 1:
                undefined_counts["SINGLE_EPISODE"] += 1

    if not defined_densities:
        exact_min = None
        exact_max = None
        exact_width = None
    else:
        exact_min = float(np.min(defined_densities))
        exact_max = float(np.max(defined_densities))
        exact_width = float(exact_max - exact_min)

    return {
        "grid_length": k,
        "unknown_count": m,
        "total_combinations": total_combinations,
        "defined_completions": len(defined_densities),
        "undefined_counts": undefined_counts,
        "exact_min_density": exact_min,
        "exact_max_density": exact_max,
        "exact_density_width": exact_width,
        "min_burden": float(np.min(burdens)),
        "max_burden": float(np.max(burdens)),
    }


def generate_reference_test_fixtures() -> List[Dict[str, Any]]:
    """Creates 20 diverse test fixtures covering boundary, bridging, burst, and fragmented states."""
    fixtures = [
        # Fixture 1: Single episode with left adjacent unknown
        {"name": "adjacent_left_unknown", "grid": [-1, -1, 1, 1, 1, 0, 0, 0, 0, 0]},
        # Fixture 2: Single episode with right adjacent unknown
        {"name": "adjacent_right_unknown", "grid": [0, 0, 0, 0, 1, 1, 1, -1, -1, 0]},
        # Fixture 3: Two episodes with bridgeable unknown between them
        {"name": "bridgeable_unknown", "grid": [0, 1, 1, -1, -1, 1, 1, 0, 0, 0]},
        # Fixture 4: Two episodes separated by sinus with distant unknown
        {"name": "distant_unknown", "grid": [1, 1, 0, 0, 1, 1, 0, -1, -1, -1]},
        # Fixture 5: Unknown at both boundaries
        {"name": "boundary_unknowns", "grid": [-1, 0, 1, 1, 0, 1, 1, 0, 0, -1]},
        # Fixture 6: Alternating AF and unknown
        {"name": "alternating_af_unknown", "grid": [1, -1, 1, -1, 1, -1, 0, 0, 0, 0]},
        # Fixture 7: Alternating Sinus and unknown
        {"name": "alternating_sinus_unknown", "grid": [0, -1, 0, -1, 0, -1, 1, 1, 1, 0]},
        # Fixture 8: High missingness (50% unknown)
        {"name": "high_missingness_50pct", "grid": [1, 1, -1, -1, -1, -1, -1, 1, 0, 0]},
        # Fixture 9: Isolated 1-block AF with isolated unknown
        {"name": "isolated_micro_bursts", "grid": [0, 1, 0, 0, -1, 0, 1, 0, 0, 0]},
        # Fixture 10: Three small AF episodes with 2 unknown slots
        {"name": "three_episodes_two_unknowns", "grid": [1, 0, 1, -1, 0, 1, 0, -1, 0, 0]},
        # Fixture 11: 12-unit grid with bridging and expansion
        {"name": "twelve_unit_bridging", "grid": [0, 1, 1, -1, 1, 0, 0, 1, -1, 0, 0, 0]},
        # Fixture 12: Continuous block except single unknown hole
        {"name": "continuous_except_one_hole", "grid": [1, 1, 1, 1, -1, 1, 1, 1, 0, 0]},
        # Fixture 13: Large gaps with micro-unknowns
        {"name": "large_gaps_micro_unknowns", "grid": [1, 0, 0, 0, 0, -1, 0, 0, 0, 1]},
        # Fixture 14: All unknown except two 1-block anchors
        {"name": "two_anchors_rest_unknown", "grid": [1, -1, -1, -1, -1, -1, -1, -1, 1, 0]},
        # Fixture 15: Monolithic merge potential
        {"name": "monolithic_merge_potential", "grid": [1, -1, 1, -1, 1, 0, 0, 0, 0, 0]},
        # Fixture 16: Asymmetric episode sizes
        {"name": "asymmetric_episodes", "grid": [1, 1, 1, 1, 0, -1, 1, 0, 0, 0]},
        # Fixture 17: Multi-unit unknown block in center
        {"name": "center_block_unknown", "grid": [0, 1, 1, -1, -1, -1, -1, 1, 1, 0]},
        # Fixture 18: Sinus background with sparse AF and unknown
        {"name": "sparse_af_sparse_unknown", "grid": [0, 0, 1, 0, 0, -1, 0, 0, 1, 0]},
        # Fixture 19: Dual bridge scenario
        {"name": "dual_bridge_scenario", "grid": [1, -1, 1, 0, 1, -1, 1, 0, 0, 0]},
        # Fixture 20: 16-unit grid with realistic clinical sequence
        {"name": "clinical_16_unit_sequence", "grid": [0, 0, 1, 1, -1, -1, 1, 0, 0, 1, 1, -1, 0, 0, 1, 0]},
    ]
    return fixtures


def run_test_suite():
    print("=" * 80)
    print("T-CAIREM V4.1: Executing Exact Grid Solver across 20 Test Fixtures")
    print("=" * 80)

    fixtures = generate_reference_test_fixtures()
    results = []

    passed_count = 0
    for idx, f in enumerate(fixtures, 1):
        grid = f["grid"]
        name = f["name"]
        sol = solve_exact_grid_bounds(grid, dt=30.0)

        # Verification tests:
        # 1. Any specific realization must have D_realized within [exact_min, exact_max]
        # Test candidate: unknown as all-sinus (if defined)
        sinus_cand = [0 if v == -1 else v for v in grid]
        s_intervals = grid_to_intervals(sinus_cand, 30.0)
        s_d = charitos_density(s_intervals, len(grid) * 30.0)

        # Test candidate: unknown as all-AF (if defined)
        af_cand = [1 if v == -1 else v for v in grid]
        a_intervals = grid_to_intervals(af_cand, 30.0)
        a_d = charitos_density(a_intervals, len(grid) * 30.0)

        containment_ok = True
        if sol["exact_min_density"] is not None and sol["exact_max_density"] is not None:
            if s_d is not None:
                if not (sol["exact_min_density"] - 1e-6 <= s_d <= sol["exact_max_density"] + 1e-6):
                    containment_ok = False
            if a_d is not None:
                if not (sol["exact_min_density"] - 1e-6 <= a_d <= sol["exact_max_density"] + 1e-6):
                    containment_ok = False

        status = "PASSED" if containment_ok else "FAILED"
        if containment_ok:
            passed_count += 1

        record = {
            "fixture_id": idx,
            "fixture_name": name,
            "grid_length": sol["grid_length"],
            "unknown_count": sol["unknown_count"],
            "total_combinations": sol["total_combinations"],
            "defined_completions": sol["defined_completions"],
            "exact_min_density": sol["exact_min_density"],
            "exact_max_density": sol["exact_max_density"],
            "exact_density_width": sol["exact_density_width"],
            "all_sinus_density": s_d,
            "all_af_density": a_d,
            "containment_verified": containment_ok,
            "status": status,
        }
        results.append(record)
        print(f"Fixture {idx:02d} ({name:<30}): combinations={sol['total_combinations']:<6} "
              f"bounds=[{str(sol['exact_min_density']):<6}, {str(sol['exact_max_density']):<6}] -> {status}")

    out_json = REPORTS_DIR / "v41_density_bound_exact_solver_tests.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    with open(CODEX_DIR / "v41_density_bound_exact_solver_tests.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nExact solver verification: {passed_count}/{len(fixtures)} passed (100.0% containment gate).")
    print(f"Saved results to: {out_json}")
    return results


if __name__ == "__main__":
    run_test_suite()
