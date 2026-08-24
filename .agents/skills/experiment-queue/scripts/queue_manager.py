#!/usr/bin/env python3
"""queue_manager.py — ARIS experiment-queue scheduler.

Runs on the SSH remote host (or locally for Modal/Vast.ai future support).
Reads a manifest, launches jobs across free GPUs via `screen`, retries on OOM,
cleans stale screens, and writes state continuously to disk.

Usage (on remote):
    nohup python3 queue_manager.py \\
        --manifest manifest.json \\
        --state queue_state.json \\
        --log-dir ./logs \\
        > queue_mgr.log 2>&1 &

(Pass --log-dir, NOT --log: --log is declared but unused; per-job log
files in --log-dir drive OOM detection and stale-screen cleanup.)

The manifest.json is either produced manually or by `build_manifest.py`.

State file format (queue_state.json):
{
  "meta": {"project": "...", "started": "ISO8601", "host": "..."},
  "phases": [{"name": "...", "depends_on": [...], "status": "..."}],
  "jobs": [
    {
      "id": "s200_N64_n50K",
      "phase": "distill",
      "status": "running",  # pending|running|completed|failed_oom|failed_transient|stuck
      "gpu": 3,
      "screen_name": "EQ_s200_N64_n50K",
      "pid": 12345,
      "attempts": 1,
      "started": "...",
      "completed": null,
      "expected_output": "figures/distill_sw_N64_n50K_...json",
      "error": null
    }, ...
  ]
}
"""

import argparse
import glob
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


OOM_RE = re.compile(r"(CUDA out of memory|torch\.OutOfMemoryError)")
TRANSIENT_CUDA_RE = re.compile(
    r"(Invalid access of peer GPU memory over nvlink or a hardware error"
    r"|CUDA error: unknown error"
    r"|CUDA error: unspecified launch failure"
    r"|CUDA driver error: initialization error)"
)
DEFAULT_GPU_FREE_THRESHOLD_MIB = 500
POLL_INTERVAL_SEC = int(os.environ.get("EXPERIMENT_QUEUE_POLL_INTERVAL_SEC", "60"))


def resolve_conda_hook(manifest_hook=None):
    """Resolve conda hook command via (1) manifest, (2) env var, (3) auto-detect, (4) PATH.

    manifest_hook: value of `conda_hook` field in manifest (full hook command, e.g.
        `eval "$(/custom/path/conda shell.bash hook)"`), or a bare conda binary path
        which will be wrapped automatically.
    """
    def wrap(path_or_cmd):
        if path_or_cmd.startswith("eval"):
            return path_or_cmd
        return f'eval "$({path_or_cmd} shell.bash hook)"'

    # 1. Manifest override
    if manifest_hook:
        return wrap(manifest_hook)
    # 2. Env var override
    env_hook = os.environ.get("ARIS_CONDA_HOOK")
    if env_hook:
        return wrap(env_hook)
    # 3. Auto-detect common install paths
    for p in (
        os.path.expanduser("~/anaconda3/bin/conda"),
        os.path.expanduser("~/miniconda3/bin/conda"),
        os.path.expanduser("~/miniforge3/bin/conda"),
        "/opt/anaconda3/bin/conda",
        "/opt/miniconda3/bin/conda",
        "/opt/miniforge3/bin/conda",
        "/usr/local/anaconda3/bin/conda",
        "/opt/homebrew/anaconda3/bin/conda",
    ):
        if os.path.exists(p):
            return wrap(p)
    # 4. Fall back to PATH
    out, rc = run("command -v conda 2>/dev/null")
    if rc == 0 and out.strip():
        return wrap(out.strip())
    # 5. Last resort
    return 'eval "$(conda shell.bash hook)"'


def now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_timestamp(value):
    """Parse queue timestamps as UTC, including legacy naive values."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def run(cmd, check=False, capture=True):
    """Run shell command, return (stdout, returncode)."""
    r = subprocess.run(cmd, shell=True, capture_output=capture, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}\n{r.stderr}")
    return r.stdout, r.returncode


def gpu_memory_free():
    """Return list of free MiB per GPU index."""
    out, rc = run("nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits")
    if rc != 0:
        return []
    return [int(x.strip()) for x in out.strip().split("\n") if x.strip()]


def free_gpus(allowed, threshold_mib=DEFAULT_GPU_FREE_THRESHOLD_MIB):
    """Return list of GPU indices with memory.free > threshold."""
    free_mem = gpu_memory_free()
    return [i for i in allowed if i < len(free_mem) and free_mem[i] > threshold_mib]


def disk_space_guard(manifest, state, cwd):
    """Clean old disposable data when disk is low and gate new launches.

    The cleanup command is opt-in and manifest-controlled.  It is run while
    jobs are active as well as between jobs, so a long training job cannot
    silently consume the last bytes before the scheduler notices.
    """
    minimum_gib = float(manifest.get("minimum_free_disk_gib", 0))
    if minimum_gib <= 0:
        return True

    free_gib = shutil.disk_usage(cwd).free / 2**30
    guard = state.setdefault("meta", {}).setdefault("disk_guard", {})
    guard["checked"] = now()
    guard["free_gib"] = round(free_gib, 3)
    guard["minimum_free_gib"] = minimum_gib

    cleanup_cmd = manifest.get("low_disk_cleanup_cmd")
    interval = int(manifest.get("low_disk_cleanup_interval_sec", 300))
    last_cleanup = guard.get("last_cleanup")
    cleanup_due = True
    if last_cleanup:
        try:
            cleanup_due = (
                datetime.now(timezone.utc) - parse_timestamp(last_cleanup)
            ).total_seconds() >= interval
        except (AttributeError, TypeError, ValueError):
            cleanup_due = True

    if free_gib < minimum_gib and cleanup_cmd and cleanup_due:
        _, rc = run(f"cd {shlex.quote(cwd)} && {cleanup_cmd}")
        guard["last_cleanup"] = now()
        guard["last_cleanup_exitcode"] = rc
        free_gib = shutil.disk_usage(cwd).free / 2**30
        guard["free_gib_after_cleanup"] = round(free_gib, 3)

    healthy = free_gib >= minimum_gib
    guard["status"] = "healthy" if healthy else "launches_paused_low_disk"
    return healthy


def screen_exists(name):
    out, _ = run("screen -ls")
    return f".{name}\t" in out


def kill_screen(name):
    run(f"screen -S {name} -X quit", check=False)


def detect_oom_in_log(log_path):
    if not log_path or not Path(log_path).exists():
        return False


def detect_transient_cuda_in_log(log_path):
    """Detect narrowly scoped CUDA/driver faults that are safe to retry.

    This intentionally excludes device-side assertions, illegal memory access,
    and generic nonzero exits because those can indicate deterministic model or
    data bugs. Only errors known to arise from transient GPU/driver state are
    requeued automatically.
    """
    if not log_path or not Path(log_path).exists():
        return False
    try:
        out, _ = run(f"tail -c 20000 {shlex.quote(log_path)}")
        return bool(TRANSIENT_CUDA_RE.search(out))
    except Exception:
        return False
    try:
        # Check tail of log for OOM marker
        out, _ = run(f"tail -c 10000 {shlex.quote(log_path)}")
        return bool(OOM_RE.search(out))
    except Exception:
        return False


def matching_outputs(path_pattern, cwd):
    """Return existing output paths for a manifest path or glob."""
    if not path_pattern:
        return []
    full = os.path.join(cwd, path_pattern) if not os.path.isabs(path_pattern) else path_pattern
    return [Path(path) for path in glob.glob(full) if Path(path).exists()]


def output_exists(path_pattern, cwd):
    """Check if an output file exists (pattern supports shell glob)."""
    return bool(matching_outputs(path_pattern, cwd))


def output_is_fresh(path_pattern, cwd, started):
    """Check that at least one matching output was written by this attempt."""
    if not started:
        return False
    try:
        started_ts = parse_timestamp(started).timestamp()
    except (AttributeError, TypeError, ValueError):
        return False
    return any(path.stat().st_mtime >= started_ts for path in matching_outputs(path_pattern, cwd))


def load_state(state_file, manifest):
    """Load state from disk or initialize from manifest."""
    if Path(state_file).exists():
        with open(state_file) as f:
            return json.load(f)
    # Initialize from manifest
    state = {
        "meta": {
            "project": manifest.get("project", "unknown"),
            "started": now(),
            "manifest_path": str(manifest.get("_path", "")),
        },
        "phases": [
            {"name": p.get("name", f"phase_{i}"),
             "depends_on": p.get("depends_on", []),
             "status": "pending"}
            for i, p in enumerate(manifest.get("phases", []))
        ],
        "jobs": [],
    }
    return state


def save_state(state, state_file):
    tmp = state_file + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.rename(tmp, state_file)


def phase_ready(phase_name, state):
    """Check if all depends_on phases are completed."""
    for p in state["phases"]:
        if p["name"] == phase_name:
            if not p["depends_on"]:
                return True
            for dep in p["depends_on"]:
                dep_phase = next((x for x in state["phases"] if x["name"] == dep), None)
                if not dep_phase or dep_phase["status"] != "completed":
                    return False
            return True
    return False


def phase_complete(phase_name, state):
    phase_jobs = [j for j in state["jobs"] if j.get("phase") == phase_name]
    if not phase_jobs:
        return False
    # A stuck job is a visible terminal failure, not proof that the phase's
    # scientific artifacts exist. Keep dependent phases blocked.
    return all(j["status"] == "completed" for j in phase_jobs)


def assign_jobs_to_phases(manifest, state):
    """Synchronize manifest phases and jobs into persisted state; idempotent."""
    for phase in manifest.get("phases", []):
        phase_name = phase.get("name")
        state_phase = next(
            (item for item in state["phases"] if item["name"] == phase_name),
            None,
        )
        if state_phase is None:
            state["phases"].append({
                "name": phase_name,
                "depends_on": phase.get("depends_on", []),
                "status": "pending",
            })
        else:
            # The manifest remains authoritative when a combined/resumed queue
            # adds or repairs dependency definitions after state creation.
            state_phase["depends_on"] = phase.get("depends_on", [])

        for job in phase.get("jobs", []):
            existing = next((j for j in state["jobs"] if j["id"] == job["id"]), None)
            if existing:
                # The manifest is the source of truth for executable job
                # definitions. This makes safe command fixes take effect after
                # a manager restart without discarding completed job history.
                existing["cmd"] = job["cmd"]
                existing["expected_output"] = job.get("expected_output")
                if "label" in job:
                    existing["label"] = job["label"]
            else:
                state["jobs"].append({
                    "id": job["id"],
                    "phase": phase_name,
                    "cmd": job["cmd"],
                    "expected_output": job.get("expected_output"),
                    "status": "pending",
                    "gpu": None,
                    "screen_name": None,
                    "pid": None,
                    "attempts": 0,
                    "started": None,
                    "completed": None,
                    "error": None,
                })


def launch_job(job, gpu, conda_env, cwd, log_dir, conda_hook):
    """Launch job in a detached screen, return (screen_name, pid)."""
    screen_name = f"EQ_{job['id']}"
    if screen_exists(screen_name):
        # Shouldn't happen; clean up
        kill_screen(screen_name)
        time.sleep(2)
    log_file = os.path.join(log_dir, f"{job['id']}.log")
    exitcode_file = log_file + ".exitcode"
    # A sentinel from an earlier attempt must never certify the new attempt.
    try:
        Path(exitcode_file).unlink()
    except FileNotFoundError:
        pass
    cmd = job["cmd"]
    # Substitute GPU placeholder if present
    cmd_with_gpu = cmd.replace("${GPU}", str(gpu))
    if conda_env.lower() == "none":
        full = (
            f'cd {shlex.quote(cwd)} && '
            f'CUDA_VISIBLE_DEVICES={gpu} {cmd_with_gpu} 2>&1 | tee {shlex.quote(log_file)}; echo ${{PIPESTATUS[0]}} > {shlex.quote(log_file)}.exitcode'
        )
    else:
        full = (
            f'cd {shlex.quote(cwd)} && '
            f'{conda_hook} && '
            f'conda activate {conda_env} && '
            f'CUDA_VISIBLE_DEVICES={gpu} {cmd_with_gpu} 2>&1 | tee {shlex.quote(log_file)}; echo ${{PIPESTATUS[0]}} > {shlex.quote(log_file)}.exitcode'
        )
    screen_cmd = f'screen -dmS {screen_name} bash -c {shlex.quote(full)}'
    run(screen_cmd)
    time.sleep(2)
    # Resolve the actual Python child for this job. Matching only
    # CUDA_VISIBLE_DEVICES returns a screen/bash wrapper and, when stacking was
    # accidentally enabled, assigned the same stale PID to several jobs.
    process_table, _ = run("ps -eo pid=,args=")
    python_pid = None
    for line in process_table.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) != 2 or job["id"] not in fields[1]:
            continue
        executable = os.path.basename(fields[1].split(None, 1)[0])
        if executable.startswith("python"):
            python_pid = int(fields[0])
            break
    return screen_name, python_pid


def job_status_check(job, log_dir, cwd):
    """Return new status for a running job."""
    screen_name = job["screen_name"]
    log_file = os.path.join(log_dir, f"{job['id']}.log")
    exitcode_file = log_file + ".exitcode"

    # 1. OOM detected → failed_oom
    if detect_oom_in_log(log_file):
        return "failed_oom", "CUDA OOM detected"

    # A small allowlist of GPU/driver faults is retryable. Detect it before the
    # screen check because the shell wrapper can remain briefly after Python
    # aborts and writes its diagnostic.
    if detect_transient_cuda_in_log(log_file):
        return "failed_transient", "Transient CUDA hardware/driver fault detected"

    # 2. A live screen means the attempt is still running. Many jobs write a
    # checkpoint or report before their process has actually finished.
    if screen_name and screen_exists(screen_name):
        if job.get("pid"):
            _, rc = run(f"kill -0 {job['pid']} 2>/dev/null")
            if rc == 0:
                return "running", None
        # The recorded PID may be a wrapper or become stale after restart.
        return "running", None

    # 3. Once the screen exits, the per-attempt exit code is authoritative.
    if not screen_name or not screen_exists(screen_name):
        if Path(exitcode_file).exists():
            try:
                exitcode = int(Path(exitcode_file).read_text().strip())
            except (OSError, ValueError):
                return "failed_other", f"Invalid exit-code sentinel: {exitcode_file}"
            if exitcode == 0:
                expected = job.get("expected_output")
                if expected and not output_is_fresh(expected, cwd, job.get("started")):
                    return (
                        "failed_other",
                        "Command exited successfully but the expected output "
                        f"was not refreshed by this attempt: {expected}",
                    )
                return "completed", None
            return "failed_other", f"Process exited with code {exitcode}"

        # Legacy attempts may predate exit-code sentinels. Only accept an
        # expected output if its timestamp proves this attempt created it.
        expected = job.get("expected_output")
        if expected and output_is_fresh(expected, cwd, job.get("started")):
            return "completed", None
        return "failed_other", "Screen exited without a fresh expected output or exitcode"

    # Default: running
    return "running", None


def pending_jobs_in_active_phases(state, manifest):
    active_phases = []
    for phase in manifest.get("phases", []):
        phase_name = phase.get("name")
        if phase_ready(phase_name, state) and not phase_complete(phase_name, state):
            active_phases.append(phase_name)
    return [
        j for j in state["jobs"]
        if j["status"] == "pending" and j["phase"] in active_phases
    ]


def step(manifest, state, state_file, log_dir):
    """Run one scheduler step: poll, launch, update state."""
    cwd = manifest.get("cwd", ".")
    conda_env = manifest.get("conda", "base")
    conda_hook = resolve_conda_hook(manifest.get("conda_hook"))
    allowed_gpus = manifest.get("gpus", list(range(8)))
    max_parallel = manifest.get("max_parallel", len(allowed_gpus))
    gpu_free_threshold = manifest.get("gpu_free_threshold_mib",
                                       DEFAULT_GPU_FREE_THRESHOLD_MIB)
    oom_delay = manifest.get("oom_retry", {}).get("delay", 120)
    max_oom_attempts = manifest.get("oom_retry", {}).get("max_attempts", 3)
    transient_delay = manifest.get("transient_retry", {}).get("delay", 120)
    max_transient_attempts = manifest.get("transient_retry", {}).get(
        "max_attempts", 2
    )

    # 1. Check disk even while a job is active. If cleanup cannot restore the
    # reserve, current jobs remain visible but no additional job is launched.
    disk_healthy = disk_space_guard(manifest, state, cwd)

    # 2. Check running jobs
    for job in state["jobs"]:
        if job["status"] != "running":
            continue
        new_status, err = job_status_check(job, log_dir, cwd)
        if new_status == "completed":
            job["status"] = "completed"
            job["completed"] = now()
            # Clean up screen
            if job["screen_name"]:
                kill_screen(job["screen_name"])
        elif new_status == "failed_oom":
            job["status"] = "failed_oom"
            job["error"] = err
            job["completed"] = now()
            if job["screen_name"]:
                kill_screen(job["screen_name"])
        elif new_status == "failed_transient":
            job["status"] = "failed_transient"
            job["error"] = err
            job["completed"] = now()
            if job["screen_name"]:
                kill_screen(job["screen_name"])
        elif new_status == "failed_other":
            # Non-OOM failures require operator inspection and are terminal for
            # this scheduler run. Parking them as stuck keeps the failure
            # visible while allowing all_done() to terminate deterministically.
            job["status"] = "stuck"
            job["error"] = err
            job["completed"] = now()
            if job["screen_name"]:
                kill_screen(job["screen_name"])

    # 3. Retry OOM jobs that have waited long enough
    current_time = time.time()
    for job in state["jobs"]:
        if job["status"] != "failed_oom":
            continue
        if job["attempts"] >= max_oom_attempts:
            job["status"] = "stuck"
            continue
        # Wait oom_delay after failure before retry
        if job["completed"]:
            last = parse_timestamp(job["completed"])
            elapsed = (datetime.now(timezone.utc) - last).total_seconds()
            if elapsed >= oom_delay:
                job["status"] = "pending"  # Requeue

    # 4. Retry only the allowlisted transient GPU/driver failures. A repeated
    # fault is parked as stuck so the queue cannot spin indefinitely.
    for job in state["jobs"]:
        if job["status"] != "failed_transient":
            continue
        if job["attempts"] >= max_transient_attempts:
            job["status"] = "stuck"
            continue
        if job["completed"]:
            last = parse_timestamp(job["completed"])
            elapsed = (datetime.now(timezone.utc) - last).total_seconds()
            if elapsed >= transient_delay:
                job["status"] = "pending"

    # 5. Launch new jobs up to max_parallel
    running = [j for j in state["jobs"] if j["status"] == "running"]
    pending = pending_jobs_in_active_phases(state, manifest)
    free = free_gpus(allowed_gpus, gpu_free_threshold)
    # One scheduler job per physical GPU. Memory headroom alone is not a safe
    # stacking signal: four nominally fitting jobs caused severe context
    # thrashing, host-memory pressure, exit 137, and repeated CUDA OOMs.
    taken = {j["gpu"] for j in running if j.get("gpu") is not None}
    free = [g for g in free if g not in taken]

    slots = (
        min(max_parallel - len(running), len(pending), len(free))
        if disk_healthy and len(free) > 0 else 0
    )
    for i in range(slots):
        job = pending[i]
        gpu = free[i]
        # Record the attempt boundary before spawning. Fast jobs can create
        # their expected output during launch_job's screen/PID discovery wait;
        # recording this afterward incorrectly labels that fresh output stale.
        job["started"] = now()
        job["attempts"] += 1
        job["error"] = None
        job["completed"] = None
        job["gpu"] = None
        job["screen_name"] = None
        job["pid"] = None
        save_state(state, state_file)
        screen_name, pid = launch_job(job, gpu, conda_env, cwd, log_dir, conda_hook)
        job["status"] = "running"
        job["gpu"] = gpu
        job["screen_name"] = screen_name
        job["pid"] = pid

    # 6. Update phase status
    for phase in state["phases"]:
        if phase_complete(phase["name"], state):
            phase["status"] = "completed"
        elif any(j["status"] == "running"
                 for j in state["jobs"] if j.get("phase") == phase["name"]):
            phase["status"] = "running"

    save_state(state, state_file)


def all_done(state):
    return all(j["status"] in ("completed", "stuck") for j in state["jobs"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--log", default=None, help="Human-readable log file")
    ap.add_argument("--log-dir", default=None,
                    help="Per-job log directory (default: cwd)")
    ap.add_argument("--poll", type=int, default=POLL_INTERVAL_SEC)
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)
    manifest["_path"] = args.manifest

    log_dir = args.log_dir or manifest.get("cwd", ".")
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    state = load_state(args.state, manifest)
    assign_jobs_to_phases(manifest, state)
    save_state(state, args.state)

    print(f"[{now()}] Queue manager started with {len(state['jobs'])} jobs")
    sys.stdout.flush()

    while not all_done(state):
        try:
            step(manifest, state, args.state, log_dir)
        except Exception as e:
            print(f"[{now()}] Step error: {e}")
            sys.stdout.flush()
        time.sleep(args.poll)

    print(f"[{now()}] All jobs done")


if __name__ == "__main__":
    main()
