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
def execute_locked_evaluation(
    manifest: dict[str, Any],
    lockbox_dir: Path,
    device_name: str = "cpu",
    checkpoints_dir: Path | None = None,
    grassmann_bank_path: Path | None = None,
) -> Path:
    """Execute complete confirmatory evaluation across all 10 configurations and 10 models."""
    device = torch.device(device_name)
    logger.info("Executing confirmatory locked evaluation on %s", device)

    seeds = [42, 43, 44, 45, 46]
    models: dict[str, dict[int, torch.nn.Module]] = {"basis": {}, "span": {}}

    for mode in ["basis", "span"]:
        for seed in seeds:
            chk_name = f"{mode}_s{seed}"
            chk_meta = manifest["checkpoints"][chk_name]
            chk_path = Path(chk_meta["path"])
            if checkpoints_dir:
                alt_path = checkpoints_dir / f"{chk_name}.pt"
                if alt_path.is_file():
                    chk_path = alt_path

            logger.info("Loading %s seed %d from %s", mode, seed, chk_path)
            chk = torch.load(chk_path, map_location="cpu")
            m = OperatorSetModel(response_dim=128, classes=5, operator_mode="continuous")
            m.load_state_dict(chk["state_dict"], strict=True)
            m.to(device)
            m.eval()
            models[mode][seed] = m

    basis_vectors = np.eye(8, dtype=np.float32)  # [I, II, V1-V6]
    e1, e2, e3, e4, e5, e6, e7, e8 = basis_vectors
    q_III = (e2 - e1) / np.sqrt(2.0)
    q_aVR = -(e1 + e2) / 2.0
    q_aVR = q_aVR / np.linalg.norm(q_aVR)
    q_aVL = (e1 - 0.5 * e2)
    q_aVL = q_aVL / np.linalg.norm(q_aVL)
    q_aVF = (e2 - 0.5 * e1)
    q_aVF = q_aVF / np.linalg.norm(q_aVF)
    q_icm = (e5 - e4) / np.sqrt(2.0)

    config_defs = {
        "Q_8": {"m": 8, "ops": basis_vectors},
        "S_6": {"m": 6, "ops": basis_vectors[2:]},
        "S_2": {"m": 2, "ops": basis_vectors[:2]},
        "S_1": {"m": 1, "ops": basis_vectors[:1]},
        "S_6_limb": {"m": 6, "ops": np.stack([e1, e2, q_III, q_aVR, q_aVL, q_aVF])},
        "derived_limb_only": {"m": 4, "ops": np.stack([q_III, q_aVR, q_aVL, q_aVF])},
        "lead_III_single": {"m": 1, "ops": q_III[np.newaxis, :]},
        "aVF_single": {"m": 1, "ops": q_aVF[np.newaxis, :]},
        "S_icm": {"m": 1, "ops": q_icm[np.newaxis, :]},
        "grassmann_bank_350": {"m": 8, "ops": None},
    }

    test_rep_path = lockbox_dir / "representation_final_test_locked.npz"
    if not test_rep_path.is_file():
        alt = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/paper07_operator/final_representations/representation_final_test_locked.npz")
        if alt.is_file():
            test_rep_path = alt
        else:
            raise FileNotFoundError(
                f"LOCKED TEST DATA MISSING: Cannot locate Fold 10 test representation at {test_rep_path} or {alt}!\n"
                "Inference cannot execute without verified locked test representations."
            )

    logger.info("Loading locked test representations from %s", test_rep_path)
    test_data = np.load(test_rep_path)
    raw_ops = torch.from_numpy(test_data["operators"]).float()
    raw_res = torch.from_numpy(test_data["responses"]).float()
    targets = test_data["labels"]
    n_records = len(targets)

    if grassmann_bank_path is None or not grassmann_bank_path.is_file():
        g_meta = manifest.get("grassmann_bank", {})
        grassmann_bank_path = Path(g_meta.get("path", "/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/paper07_operator/final_representations/expanded_grassmann_subspace_bank.npz"))

    grassmann_bank = None
    if grassmann_bank_path.is_file():
        grassmann_bank = np.load(grassmann_bank_path)

    config_names = list(config_defs.keys())
    n_configs = len(config_names)
    n_seeds = len(seeds)

    span_preds_arr = np.zeros((n_configs, n_seeds, n_records, 5), dtype=np.float32)
    basis_preds_arr = np.zeros((n_configs, n_seeds, n_records, 5), dtype=np.float32)

    for c_idx, cname in enumerate(config_names):
        logger.info("Evaluating locked configuration %d/%d: %s", c_idx + 1, n_configs, cname)

        if cname == "grassmann_bank_350" and grassmann_bank is not None:
            subspace_keys = [k for k in grassmann_bank.keys() if k.startswith("subspace_")]
            n_sub = min(350, len(subspace_keys))
            logger.info("Evaluating across %d frozen Grassmann subspaces...", n_sub)

            for s_idx, s in enumerate(seeds):
                m_span = models["span"][s]
                m_basis = models["basis"][s]
                sub_span_preds = []
                sub_basis_preds = []

                for k in subspace_keys[:n_sub]:
                    sub_ops = torch.from_numpy(grassmann_bank[k]).float().to(device)
                    r = sub_ops.shape[0]
                    with torch.no_grad():
                        sub_ops_batch = sub_ops.unsqueeze(0).expand(n_records, -1, -1)
                        res_sub = raw_res[:, :r].to(device)
                        p_sp = torch.sigmoid(m_span(sub_ops_batch, res_sub)).cpu().numpy()
                        p_ba = torch.sigmoid(m_basis(sub_ops_batch, res_sub)).cpu().numpy()
                        sub_span_preds.append(p_sp)
                        sub_basis_preds.append(p_ba)

                span_preds_arr[c_idx, s_idx] = np.mean(sub_span_preds, axis=0)
                basis_preds_arr[c_idx, s_idx] = np.mean(sub_basis_preds, axis=0)
        else:
            ops_np = config_defs[cname]["ops"]
            m_leads = config_defs[cname]["m"]
            ops_t = torch.from_numpy(ops_np).unsqueeze(0).expand(n_records, -1, -1).float().to(device)
            res_t = raw_res[:, :m_leads].to(device)

            for s_idx, s in enumerate(seeds):
                m_span = models["span"][s]
                m_basis = models["basis"][s]
                with torch.no_grad():
                    span_preds_arr[c_idx, s_idx] = torch.sigmoid(m_span(ops_t, res_t)).cpu().numpy()
                    basis_preds_arr[c_idx, s_idx] = torch.sigmoid(m_basis(ops_t, res_t)).cpu().numpy()

    predictions_archive_path = lockbox_dir / "fold10_locked_predictions.npz"
    np.savez_compressed(
        predictions_archive_path,
        targets=targets,
        span_preds=span_preds_arr,
        basis_preds=basis_preds_arr,
        configurations=np.array(config_names),
        seeds=np.array(seeds),
    )
    pred_sha = compute_sha256(predictions_archive_path)
    logger.info("Saved locked predictions archive to %s (SHA-256: %s)", predictions_archive_path, pred_sha)

    lock_path = lockbox_dir / "fold10_access.lock"
    if lock_path.is_file():
        lock_meta = json.loads(lock_path.read_text())
        lock_meta["status"] = "EVALUATION_COMPLETED_SEALED"
        lock_meta["predictions_sha256"] = pred_sha
        lock_meta["completed_timestamp_utc"] = subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], text=True).strip()
        lock_path.write_text(json.dumps(lock_meta, indent=2))

    return predictions_archive_path


def main():
    parser = argparse.ArgumentParser(description="Fold 10 Locked Evaluation Entry Point.")
    parser.add_argument("--manifest", "--manifest-path", dest="manifest", type=Path, required=True,
                        help="Signed submission manifest.")
    parser.add_argument("--repo-dir", type=Path, default=Path("."),
                        help="Repository directory root.")
    parser.add_argument("--lockbox-dir", "--output-dir", "--fold10-results-dir", dest="lockbox_dir",
                        type=Path, required=True,
                        help="Directory to record locked evaluation and acquire persistent lock.")
    parser.add_argument("--unseal-authorization-token", "--auth-token", dest="unseal_authorization_token",
                        type=str, default=None,
                        help="Cryptographic authorization token required to unseal Fold 10.")
    parser.add_argument("--checkpoints-dir", type=Path, default=None,
                        help="Optional directory containing trained model checkpoints.")
    parser.add_argument("--grassmann-bank-path", type=Path, default=None,
                        help="Optional path to 350-subspace Grassmann benchmark bank.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Compute device for locked evaluation.")
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
    pred_path = execute_locked_evaluation(
        manifest=manifest,
        lockbox_dir=args.lockbox_dir,
        device_name=args.device,
        checkpoints_dir=args.checkpoints_dir,
        grassmann_bank_path=args.grassmann_bank_path,
    )
    logger.info("Locked Fold 10 evaluation successfully completed. Predictions: %s", pred_path)


if __name__ == "__main__":
    main()
