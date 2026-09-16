#!/usr/bin/env python3
"""Wait for GPU 0 to become idle, then run the frozen N003a follow-up queue."""
from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
PYTHON = Path("/home/mithunmanivannan/.venv/bin/python")
STATE = ROOT / "results/n003a_followup_queue/queue_state.json"
LOG_DIR = ROOT / "results/n003a_followup_queue/logs"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(payload: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, STATE)


def gpu_memory_used_mib() -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
        check=True, capture_output=True, text=True,
    )
    return int(result.stdout.strip())


def jobs() -> list[dict]:
    dense_output = ROOT / "results/n003a_matched_dense_panobench"
    return [
        {
            "name": "n003a_dense_panobench",
            "output": dense_output,
            "completion": dense_output / "dense_summary.json",
            "command": [str(PYTHON), "-u", str(ROOT / "scripts/eval_n003a_matched_dense_panobench.py")],
        },
        *[
            {
                "name": f"n003a_stage1_fixed_norm_seed{seed}",
                "output": ROOT / f"results/n003a_stage1_fixed_norm_seed{seed}",
                "completion": ROOT / f"results/n003a_stage1_fixed_norm_seed{seed}/model_final.pt",
                "command": [
                    str(PYTHON), "-u", str(ROOT / "scripts/train_n003_stage1_clinical.py"),
                    "--seed", str(seed), "--epochs", "100", "--batch-size", "64", "--workers", "8",
                    "--output", str(ROOT / f"results/n003a_stage1_fixed_norm_seed{seed}"),
                ],
            }
            for seed in (42, 2026)
        ],
    ]


def main() -> None:
    planned = jobs()
    for job in planned:
        if job["output"].exists():
            raise FileExistsError(f"refusing to overwrite queued output: {job['output']}")
    state = {
        "created_at": now(), "status": "waiting_for_gpu", "gpu": 0,
        "idle_threshold_mib": 500, "jobs": [
            {"name": job["name"], "command": job["command"], "status": "pending"}
            for job in planned
        ],
    }
    atomic_json(state)
    consecutive_idle = 0
    while consecutive_idle < 2:
        used = gpu_memory_used_mib()
        consecutive_idle = consecutive_idle + 1 if used < 500 else 0
        state.update(last_gpu_memory_used_mib=used, last_checked_at=now())
        atomic_json(state)
        print(f"waiting_for_gpu memory_used_mib={used} idle_checks={consecutive_idle}/2", flush=True)
        if consecutive_idle < 2:
            time.sleep(30)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES="0", PYTHONPATH=f"{ROOT / 'src'}:{ROOT / 'author_code/nefnet_v2'}")
    state["status"] = "running"
    atomic_json(state)
    for index, job in enumerate(planned):
        entry = state["jobs"][index]
        entry.update(status="running", started_at=now())
        atomic_json(state)
        log_path = LOG_DIR / f"{job['name']}.log"
        print(f"starting {job['name']}", flush=True)
        with log_path.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(job["command"], cwd=REPO, env=environment,
                                       stdout=stream, stderr=subprocess.STDOUT)
        entry.update(status="complete" if completed.returncode == 0 else "failed",
                     exit_code=completed.returncode, finished_at=now(), log=str(log_path))
        atomic_json(state)
        if completed.returncode != 0:
            state.update(status="failed", failed_job=job["name"], finished_at=now())
            atomic_json(state)
            raise SystemExit(completed.returncode)
        if not job["completion"].exists():
            state.update(status="failed", failed_job=job["name"],
                         failure_reason=f"missing completion artifact {job['completion']}", finished_at=now())
            atomic_json(state)
            raise RuntimeError(state["failure_reason"])
    state.update(status="complete", finished_at=now())
    atomic_json(state)
    print("N003a follow-up queue complete", flush=True)


if __name__ == "__main__":
    main()
