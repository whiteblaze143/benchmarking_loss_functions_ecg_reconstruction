#!/usr/bin/env python3
"""Wait for safe host resources, then exec the resumable RDB calibration phase."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def available_memory_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024**2
    raise RuntimeError("MemAvailable is absent from /proc/meminfo")


def event(name: str, **values: object) -> None:
    print(json.dumps({"event": name, "at": dt.datetime.now(dt.timezone.utc).isoformat(), **values}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-load-1m", type=float, default=3.0)
    parser.add_argument("--min-available-memory-gib", type=float, default=7.0)
    parser.add_argument("--min-free-disk-gib", type=float, default=8.0)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--torch-threads", type=int, default=6)
    parser.add_argument("--degraded-max-load-1m", type=float, default=8.0)
    parser.add_argument("--degraded-min-available-memory-gib", type=float, default=4.75)
    parser.add_argument("--degraded-torch-threads", type=int, default=1)
    parser.add_argument("--results-db", type=Path, default=ROOT / "results/onelead_rdb_semiseg_blinded/compact.sqlite")
    args = parser.parse_args()
    while True:
        load = os.getloadavg()[0]
        memory = available_memory_gib()
        disk = shutil.disk_usage(ROOT).free / 1024**3
        ideal = load <= args.max_load_1m and memory >= args.min_available_memory_gib
        degraded = load <= args.degraded_max_load_1m and memory >= args.degraded_min_available_memory_gib
        if disk >= args.min_free_disk_gib and (ideal or degraded):
            selected_threads = args.torch_threads if ideal else args.degraded_torch_threads
            selected_mode = "ideal" if ideal else "degraded_one_core"
            break
        event("resource_wait", load_1m=load, available_memory_gib=round(memory, 3), free_disk_gib=round(disk, 3),
              required={"ideal": {"max_load_1m": args.max_load_1m, "min_available_memory_gib": args.min_available_memory_gib, "threads": args.torch_threads},
                        "degraded": {"max_load_1m": args.degraded_max_load_1m, "min_available_memory_gib": args.degraded_min_available_memory_gib, "threads": args.degraded_torch_threads},
                        "min_free_disk_gib": args.min_free_disk_gib})
        time.sleep(args.poll_seconds)
    event("resource_gate_passed", mode=selected_mode, load_1m=os.getloadavg()[0], available_memory_gib=available_memory_gib(), torch_threads=selected_threads)
    command = [
        sys.executable, str(ROOT / "scripts/evaluate_onelead_rdb_semiseg_blinded.py"),
        "--phase", "calibration", "--torch-threads", str(selected_threads),
        "--results-db", str(args.results_db),
    ]
    os.chdir(ROOT)
    subprocess.run(command, check=True)
    event("calibration_complete_starting_screened_population")
    command[command.index("calibration")] = "screened-all"
    screened = subprocess.run(command)
    if screened.returncode == 0:
        event("screened_population_complete")
        return
    event("screening_gate_unavailable_falling_back_to_full_population", returncode=screened.returncode)
    command[command.index("screened-all")] = "full-all"
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
