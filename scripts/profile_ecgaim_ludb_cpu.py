#!/usr/bin/env python3
"""Benchmark ECG-AIM LUDB CPU inference batch sizes on one loaded checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_comprehensive_registry import load_adapter  # noqa: E402
from scripts.evaluate_ecgaim_ludb_daemon import OBSERVED, load_ludb  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="factorial_ecg_aim_1000001_s42")
    parser.add_argument("--mask", default="1000001")
    parser.add_argument("--records", type=int, default=48)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--batches", default="2,4,8,12,4")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    records, _ = load_ludb(ROOT / "data/ludb", args.records)
    target = torch.from_numpy(np.stack([record["signal"] for record in records]))
    adapter = load_adapter(
        {
            "id": args.model_id,
            "kind": "alitok",
            "checkpoint": f"checkpoints/factorial_ecg_aim_{args.mask}_s42.pt",
            "observed_leads": list(OBSERVED),
        },
        torch.device("cpu"),
    )
    with torch.inference_mode():
        adapter.reconstruct(target[:1])
    rows = []
    for batch_size in [int(value) for value in args.batches.split(",")]:
        started = time.perf_counter()
        with torch.inference_mode():
            for offset in range(0, len(target), batch_size):
                output = adapter.reconstruct(target[offset : offset + batch_size])
                del output
        seconds = time.perf_counter() - started
        row = {
            "batch_size": batch_size,
            "records": len(target),
            "seconds": seconds,
            "records_per_second": len(target) / seconds,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    payload = {
        "model_id": args.model_id,
        "threads": args.threads,
        "cpu_only": True,
        "runs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, args.output)


if __name__ == "__main__":
    main()
