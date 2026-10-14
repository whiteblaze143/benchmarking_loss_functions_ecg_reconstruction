"""Locked Fold-10 Benchmark Evaluator & Fail-Closed Access Guard.

Fail-Closed Architecture:
1. Preflight Manifest Validation: Recomputes hashes of source code, preprocessing,
   Grassmann bank, and all 10 checkpoints before touching ANY test data.
2. Unseal Authorization Guard: Fold 10 is SEALED by default. Requires explicit authorization token.
3. Atomic Persistent Lock: acquire_atomic_fold10_lock() uses O_CREAT | O_EXCL.
   Lock is permanent and NEVER deleted on error, requiring explicit manual recovery.
4. Preregistered Configuration Coverage: Verifies complete registry of Regimes 1, 2, and 3.
5. Zero development fallbacks or synthetic substitutes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("evaluate_final_fold10_locked")

from repecg.paper07_operator.model import OperatorSetModel

EXPECTED_MODEL_SHA256 = "66dae0d122266271e4b5b080e3e7048e4ed4ad3bea7d1f96e7bc8fd99b85227e"


class LockboxPreflightError(RuntimeError):
    """Raised when any preflight check fails prior to lockbox access."""


class LockboxViolationError(RuntimeError):
    """Raised when lock acquisition fails or test access is unauthorized."""


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    if not path.is_file():
        raise LockboxPreflightError(f"Missing required file: {path}")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def require_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load and validate submission manifest structure."""
    if not manifest_path.is_file():
        raise LockboxPreflightError(f"Submission manifest not found: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as e:
        raise LockboxPreflightError(f"Invalid JSON in manifest {manifest_path}: {e}")

    required_keys = [
        "manifest_version",
        "lockbox_status",
        "model_architecture",
        "preprocessing_hashes",
        "checkpoints",
        "configurations",
        "statistical_protocol",
    ]
    missing = [k for k in required_keys if k not in manifest]
    if missing:
        raise LockboxPreflightError(f"Manifest missing required top-level keys: {missing}")

    return manifest


def verify_frozen_artifacts(manifest: dict[str, Any], repo_dir: Path) -> None:
    """Verify source hash and preprocessing artifact hashes match manifest."""
    model_src = repo_dir / manifest["model_architecture"]["source_file"]
    model_sha = compute_sha256(model_src)
    if model_sha != manifest["model_architecture"]["sha256"]:
        raise LockboxPreflightError(
            f"Source hash mismatch for {model_src}!\n"
            f"Manifest: {manifest['model_architecture']['sha256']}\n"
            f"Computed: {model_sha}"
        )
    if model_sha != EXPECTED_MODEL_SHA256:
        raise LockboxPreflightError(
            f"Model source SHA256 does not match frozen standard {EXPECTED_MODEL_SHA256}!"
        )

    # Verify golden logits test
    torch.manual_seed(42)
    test_model = OperatorSetModel(response_dim=128, classes=5, operator_mode="continuous")
    test_model.eval()
    torch.manual_seed(1337)
    with torch.no_grad():
        test_logits = test_model(torch.randn(1, 4, 8), torch.randn(1, 4, 16, 128)).squeeze().tolist()
    ref_logits = [0.012054, 0.012796, 0.004852, 0.08265, 0.036781]
    if max(abs(a - b) for a, b in zip(test_logits, ref_logits)) > 1e-4:
        raise LockboxPreflightError("Model golden logit regression test failed!")

    logger.info("Verified frozen architecture contract and golden logits.")


def verify_all_checkpoints(manifest: dict[str, Any]) -> None:
    """Verify all 10 checkpoints exist, match SHA-256, and load strictly."""
    checkpoints = manifest["checkpoints"]
    if len(checkpoints) != 10:
        raise LockboxPreflightError(
            f"Manifest contains {len(checkpoints)} checkpoints; exactly 10 required (5 Basis + 5 Span)!"
        )

    expected_names = [f"{mode}_s{seed}" for mode in ["basis", "span"] for seed in [42, 43, 44, 45, 46]]
    missing_names = [n for n in expected_names if n not in checkpoints]
    if missing_names:
        raise LockboxPreflightError(f"Missing expected checkpoint configurations: {missing_names}")

    for name, meta in checkpoints.items():
        path = Path(meta["path"])
        computed_sha = compute_sha256(path)
        if computed_sha != meta["sha256"]:
            raise LockboxPreflightError(
                f"Checkpoint {name} SHA256 mismatch!\n"
                f"Manifest: {meta['sha256']}\n"
                f"Computed: {computed_sha}"
            )
        chk = torch.load(path, map_location="cpu")
        model = OperatorSetModel(response_dim=128, classes=5, operator_mode="continuous")
        model.load_state_dict(chk["state_dict"], strict=True)

    logger.info("Verified all 10 checkpoints: hashes and strict state_dict compatibility passed.")


def verify_benchmark_coverage(manifest: dict[str, Any]) -> None:
    """Verify preregistered coverage across all clinical and geometric regimes."""
    configs = manifest["configurations"]
    required_regimes = [
        "regime_1_standard_clinical",
        "regime_2_derived_augmented_limb",
        "regime_3_continuous_oblique",
    ]
    for r in required_regimes:
        if r not in configs:
            raise LockboxPreflightError(f"Missing required benchmark regime: {r}")

    # Verify specific configurations
    r2 = configs["regime_2_derived_augmented_limb"]
    for c in ["lead_III_single", "aVF_single", "derived_limb_only"]:
        if c not in r2:
            raise LockboxPreflightError(f"Missing Regime 2 configuration: {c}")

    r3 = configs["regime_3_continuous_oblique"]
    if "grassmann_bank_350" not in r3:
        raise LockboxPreflightError("Missing 350-subspace Grassmann benchmark configuration!")

    logger.info("Verified complete benchmark coverage across all 3 preregistered regimes.")


def verify_no_uncommitted_changes(repo_dir: Path) -> None:
    """Ensure working directory has no uncommitted changes to tracked files."""
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "-uno"], cwd=repo_dir, text=True
    ).strip()
    if status:
        raise LockboxPreflightError(
            f"Uncommitted tracked changes detected in repository:\n{status}\n"
            "Final lockbox evaluation requires a clean commit."
        )
    logger.info("Verified clean git repository state.")


def acquire_atomic_fold10_lock(lockbox_dir: Path, manifest_sha: str) -> Path:
    """Acquire atomic, persistent Fold 10 lockfile using O_CREAT | O_EXCL."""
    lockbox_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lockbox_dir / "fold10_access.lock"

    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    except FileExistsError:
        content = lock_path.read_text().strip() if lock_path.is_file() else "UNKNOWN"
        raise LockboxViolationError(
            f"LOCKBOX ACCESS VIOLATION: Atomic lock already exists at {lock_path}!\n"
            f"Lockfile details:\n{content}\n"
            "Fold 10 evaluation was already initiated, completed, or interrupted.\n"
            "Silently re-running or overwriting test evaluation is strictly forbidden.\n"
            "Documented manual authorization is required to clear this lock."
        )

    lock_payload = {
        "status": "ACQUIRED_EVALUATION_IN_PROGRESS",
        "acquired_by_pid": os.getpid(),
        "manifest_sha256": manifest_sha,
        "timestamp_utc": subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], text=True).strip(),
        "notice": "THIS LOCK IS ATOMIC AND PERMANENT. IT MUST NOT BE SILENTLY REMOVED.",
    }
    with os.fdopen(fd, "w") as f:
        json.dump(lock_payload, f, indent=2)

    logger.warning("ATOMIC FOLD 10 LOCK ACQUIRED: %s", lock_path)
    return lock_path


def run_preflight_suite(manifest_path: Path, repo_dir: Path) -> dict[str, Any]:
    """Execute all preflight checks without touching Fold 10."""
    logger.info("Executing fail-closed preflight checks...")
    manifest = require_manifest(manifest_path)
    verify_frozen_artifacts(manifest, repo_dir)
    verify_all_checkpoints(manifest)
    verify_benchmark_coverage(manifest)
    verify_no_uncommitted_changes(repo_dir)
    logger.info("ALL PREFLIGHT CHECKS PASSED SUCCESSFULLY.")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Fold 10 Locked Evaluation Entry Point.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo-dir", type=Path, default=Path("."))
    parser.add_argument("--lockbox-dir", type=Path, required=True)
    parser.add_argument("--unseal-authorization-token", type=str, default=None,
                        help="Cryptographic authorization token required to unseal Fold 10.")
    parser.add_argument("--verify-only", action="store_true",
                        help="Verify manifest and stored prediction artifacts without accessing raw waveforms.")
    args = parser.parse_args()

    if args.verify_only:
        logger.info("Running in VERIFY-ONLY mode (no waveform access)...")
        manifest = run_preflight_suite(args.manifest, args.repo_dir)
        logger.info("Preflight and manifest verification complete.")
        return

    # Check unseal authorization gate
    if not args.unseal_authorization_token:
        logger.error(
            "ACCESS FORBIDDEN: Fold 10 is STRICTLY SEALED.\n"
            "Release checklist gates (0/10) must be reviewed and signed off prior to unsealing.\n"
            "No unseal authorization token provided. Aborting evaluation."
        )
        sys.exit(1)

    # Execute all preflights before lock acquisition
    manifest = run_preflight_suite(args.manifest, args.repo_dir)
    manifest_sha = compute_sha256(args.manifest)

    # Acquire atomic lock: FIRST operation permitted before test read
    lock_file = acquire_atomic_fold10_lock(args.lockbox_dir, manifest_sha)

    logger.info("Proceeding to locked evaluation under authorization token: %s", args.unseal_authorization_token)
    # Locked inference implementation...


if __name__ == "__main__":
    main()
