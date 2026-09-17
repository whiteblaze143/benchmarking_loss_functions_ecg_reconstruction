from __future__ import annotations

import argparse
import json
from pathlib import Path

from repecg.evaluation.grid import evaluate_development_variant


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-id", type=int, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=4096)
    args = parser.parse_args()
    payload = evaluate_development_variant(
        args.paper_id,
        args.variant,
        args.representations,
        args.training,
        args.output,
        args.batch,
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
