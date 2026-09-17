from __future__ import annotations

import argparse
import json
from pathlib import Path

from repecg.evaluation.grid import aggregate_grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-id", type=int, required=True)
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(aggregate_grid(args.paper_id, args.cells, args.output, args.seed), sort_keys=True))


if __name__ == "__main__":
    main()
