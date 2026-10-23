#!/usr/bin/env python3
"""Sentinel Interceptor: Cleanly halt supervisor after A3_no_q completion to trigger A0 Long handoff.

Watches active A3_no_q training (PID 221960) and subsequent dev-bank evaluation.
The moment A3_no_q is fully evaluated and marked COMPLETE in ablation_queue_state.json,
sends SIGINT to supervisor PID 4184400 so it exits cleanly, releasing ABLAT_LOCK.
This triggers the waiting a0_auto_handoff daemon (PID 84475) to execute the single-experiment
launch of A0_native_init_no_earlystop_50ep_s42 without allowing the supervisor to run B1.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import signal
import sys
import time

STATE_FILE = Path("/data/mithunmanivannan/codex_artifacts/echonext_geometry_setoperator_v1/supervisor_state/ablation_queue_state.json")
A3_RESULT_FILE = Path("/data/mithunmanivannan/codex_artifacts/echonext_geometry_setoperator_v1/checkpoints/ablations/A3_no_q/result.json")
LOG_FILE = Path("/data/mithunmanivannan/codex_artifacts/echonext_geometry_setoperator_v1/supervisor_state/intercept_a3.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a"),
    ],
)
logger = logging.getLogger("intercept_a3")


def is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main():
    parser = argparse.ArgumentParser(description="Intercept supervisor after A3_no_q")
    parser.add_argument("--a3-pid", type=int, default=221960, help="PID of A3 training worker")
    parser.add_argument("--supervisor-pid", type=int, default=4184400, help="PID of ablation supervisor")
    parser.add_argument("--poll-interval", type=int, default=10, help="Poll interval in seconds")
    args = parser.parse_args()

    logger.info("================================================================================")
    logger.info("SENTINEL INTERCEPTOR INITIALIZED")
    logger.info("Monitoring A3 Worker PID:       %d", args.a3_pid)
    logger.info("Monitoring Supervisor PID:      %d", args.supervisor_pid)
    logger.info("Target A3 Result:               %s", A3_RESULT_FILE)
    logger.info("================================================================================")

    # Phase 1: Wait for A3 training worker to finish
    t0 = time.time()
    last_log = 0
    while is_pid_alive(args.a3_pid):
        now = time.time()
        if now - last_log >= 300:
            last_log = now
            elapsed = (now - t0) / 60.0
            logger.info("Heartbeat: A3 Worker PID %d still active (watchdog elapsed: %.1f min)...", args.a3_pid, elapsed)
        time.sleep(args.poll_interval)

    logger.info(">>> A3 training worker PID %d has finished! Awaiting dev-bank evaluation...", args.a3_pid)

    # Phase 2: Wait for evaluation to finish and state to be saved as COMPLETE
    t_eval_start = time.time()
    while True:
        if A3_RESULT_FILE.exists() and A3_RESULT_FILE.stat().st_size > 0:
            try:
                if STATE_FILE.exists():
                    with open(STATE_FILE, "r") as f:
                        state = json.load(f)
                    a3_status = state.get("experiments", {}).get("A3_no_q", {}).get("status")
                    if a3_status == "COMPLETE":
                        logger.info(">>> A3_no_q is verified COMPLETE in queue state!")
                        break
            except Exception as e:
                logger.warning("Error reading state file: %s", e)

        if not is_pid_alive(args.supervisor_pid):
            logger.warning("Supervisor PID %d already terminated before interception. Exiting cleanly.", args.supervisor_pid)
            return

        now = time.time()
        if now - last_log >= 60:
            last_log = now
            logger.info("Awaiting A3 evaluation completion (elapsed: %.1f min)...", (now - t_eval_start) / 60.0)

        time.sleep(args.poll_interval)

    # Settle file writes
    time.sleep(3)

    # Phase 3: Terminate supervisor cleanly to prevent B1 launch
    logger.info("Sending SIGINT to supervisor PID %d to halt queue cleanly...", args.supervisor_pid)
    try:
        os.kill(args.supervisor_pid, signal.SIGINT)
    except OSError as e:
        logger.warning("Could not send SIGINT: %s. Trying SIGTERM...", e)
        try:
            os.kill(args.supervisor_pid, signal.SIGTERM)
        except OSError as e2:
            logger.warning("Could not send SIGTERM: %s", e2)

    # Phase 4: Verify supervisor termination
    for _ in range(15):
        if not is_pid_alive(args.supervisor_pid):
            logger.info(">>> Supervisor PID %d has terminated cleanly!", args.supervisor_pid)
            break
        time.sleep(1)
    else:
        logger.warning("Supervisor did not terminate after 15s. Sending SIGKILL...")
        try:
            os.kill(args.supervisor_pid, signal.SIGKILL)
        except OSError:
            pass

    logger.info("Interceptor finished successfully. a0_auto_handoff daemon will now launch A0-Long.")


if __name__ == "__main__":
    main()
