#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from repecg.common.variants import get_variants_for_paper
from repecg.evaluation.results_db import coverage, register_expected_evaluations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--run-type", choices=("smoke", "production"), required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    planned = 0
    for paper_id in range(1, 16):
        planned += register_expected_evaluations(
            args.database,
            run_type=args.run_type,
            paper_id=paper_id,
            datasets=args.datasets,
            split=args.split,
            variants=list(get_variants_for_paper(paper_id)),
            seed=args.seed,
        )
    print(json.dumps({"planned": planned, "coverage": coverage(args.database, args.run_type)}, sort_keys=True))


if __name__ == "__main__":
    main()
