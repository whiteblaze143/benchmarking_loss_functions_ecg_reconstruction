#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from repecg.evaluation.results_db import ingest_paper


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--paper-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--run-type", choices=("smoke", "production"), required=True)
    args = parser.parse_args()
    result = ingest_paper(
        args.database, args.paper_id, args.output, args.dataset, args.split, args.run_type
    )
    print(json.dumps({"paper_id": args.paper_id, **result}, sort_keys=True))


if __name__ == "__main__":
    main()
