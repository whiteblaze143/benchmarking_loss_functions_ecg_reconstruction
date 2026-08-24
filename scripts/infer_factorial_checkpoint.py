#!/usr/bin/env python3
"""Run an exact archived factorial checkpoint on a real project ECG tensor."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.checkpoint_store import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DB,
    connect,
    load_checkpoint_with_identity,
    prune_cache,
    store_lock,
)
from scripts.train_mcma_3lead import MCMAModel


INPUT_LEADS = (0, 1, 7)  # I, II, V2, matching factorial training.
DEFAULT_COMPATIBILITY_AUDIT = (
    ROOT / "results/checkpoint_store/compatibility_audit.json"
)
DEFAULT_TRAINING_CONTRACT = ROOT / "refine-logs/factorial_training_contract.json"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("model_id", nargs="?", help="Queue id, e.g. f_1000002_s42")
    result.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="Real ECG tensor: [12,time], [3,time], or [batch,channels,time]",
    )
    result.add_argument("--output", type=Path, help="Optional reconstructed .pt file")
    result.add_argument(
        "--report",
        type=Path,
        help="Optional machine-readable JSON inference/identity report",
    )
    result.add_argument("--db", type=Path, default=DEFAULT_DB)
    result.add_argument(
        "--compatibility-audit",
        type=Path,
        default=DEFAULT_COMPATIBILITY_AUDIT,
    )
    result.add_argument(
        "--training-contract", type=Path, default=DEFAULT_TRAINING_CONTRACT
    )
    result.add_argument(
        "--list-ready",
        action="store_true",
        help="Print current release-compatible model ids, one per line, and exit",
    )
    result.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    result.add_argument(
        "--max-cache-gib",
        type=float,
        default=0.0,
        help=(
            "Prune verified cache copies to this bound in a finally block; "
            "default 0 leaves no materialized checkpoint behind"
        ),
    )
    result.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="CPU is the safe default while training occupies the GPU",
    )
    return result


def compatible_identities(audit_path: Path, contract_path: Path) -> dict[str, dict]:
    """Return only identities bound to the current contract and approved source."""
    audit = json.loads(audit_path.read_text())
    contract = json.loads(contract_path.read_text())
    audit_contract = audit.get("contract") or {}
    if audit_contract.get("contract_id") != contract.get("contract_id"):
        raise RuntimeError("Compatibility audit is not bound to the current contract")
    if audit_contract.get("approved_source_bundle_sha256") != contract.get(
        "approved_source_bundle_sha256"
    ):
        raise RuntimeError("Compatibility audit has the wrong approved source bundle")
    ready = {}
    for row in audit.get("models", []):
        if not row.get("compatible"):
            continue
        if row.get("source_bundle_sha256") != contract.get(
            "approved_source_bundle_sha256"
        ):
            continue
        model_id = row.get("model_id")
        digest = row.get("checkpoint_sha256")
        if not model_id or not digest:
            raise RuntimeError("Compatible audit row lacks model id or checkpoint digest")
        if model_id in ready:
            raise RuntimeError(f"Duplicate compatible audit identity: {model_id}")
        ready[model_id] = row
    return ready


def prepare_input(path: Path) -> tuple[torch.Tensor, int]:
    signal = torch.load(path, map_location="cpu", weights_only=True).float()
    if signal.ndim == 2:
        signal = signal.unsqueeze(0)
    if signal.ndim != 3:
        raise ValueError(f"Expected 2-D or 3-D ECG tensor, got {tuple(signal.shape)}")
    if signal.shape[1] == 12:
        signal = signal[:, INPUT_LEADS, :]
    elif signal.shape[1] != 3:
        raise ValueError(
            f"Expected 12 full leads or 3 model inputs, got {signal.shape[1]} channels"
        )
    original_length = signal.shape[-1]
    padded_length = ((original_length + 31) // 32) * 32
    if padded_length != original_length:
        signal = torch.nn.functional.pad(signal, (0, padded_length - original_length))
    return signal, original_length


def atomic_torch_save(value: object, destination: Path) -> None:
    """Durably publish a torch artifact without exposing a partial final file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = parser().parse_args()
    ready = compatible_identities(args.compatibility_audit, args.training_contract)
    if args.list_ready:
        if args.model_id is not None or args.input is not None:
            raise ValueError("--list-ready does not accept model_id or input")
        print("\n".join(sorted(ready)))
        return
    if args.model_id is None or args.input is None:
        raise ValueError("model_id and input are required unless --list-ready is used")
    if args.model_id not in ready:
        raise RuntimeError(
            f"{args.model_id} is not release-compatible in {args.compatibility_audit}"
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    if args.max_cache_gib < 0:
        raise ValueError("--max-cache-gib must be nonnegative")
    # Reject an invalid input before fetching a remote checkpoint. This keeps a
    # malformed invocation from consuming network, cache, and verification I/O.
    signal, original_length = prepare_input(args.input)
    result = None
    pruned_files = 0
    try:
        payload, catalog_identity = load_checkpoint_with_identity(
            args.model_id,
            db_path=args.db,
            cache_dir=args.cache_dir,
            map_location="cpu",
            weights_only=False,
        )
        if catalog_identity["sha256"] != ready[args.model_id]["checkpoint_sha256"]:
            raise RuntimeError(
                f"{args.model_id} catalog digest does not match compatibility audit"
            )
        state = payload.get("model_state_dict", payload)
        model = MCMAModel(in_channels=3, out_channels=12)
        model.load_state_dict(state, strict=True)
        model = model.to(args.device).eval()
        with torch.inference_mode():
            reconstruction = model(signal.to(args.device))[..., :original_length].cpu()
        if not torch.isfinite(reconstruction).all():
            raise RuntimeError("Inference produced non-finite output")
        if args.output:
            atomic_torch_save(reconstruction, args.output)
        provenance = payload.get("provenance") or {}
        architecture = provenance.get("architecture_provenance") or {}
        contract = provenance.get("training_contract") or {}
        provenance_summary = {
            "seed": provenance.get("seed"),
            "factorial_mask": provenance.get("factorial_mask"),
            "split_inventory_name_size_sha256": provenance.get(
                "split_inventory_name_size_sha256",
                provenance.get("split_inventory_sha256"),
            ),
            "split_content_roots": provenance.get("split_content_roots"),
            "preprocessing": provenance.get("preprocessing"),
            "source_bundle_sha256": architecture.get("source_bundle_sha256"),
            "source_file_sha256": architecture.get("source_file_sha256"),
            "exact_sources_embedded": bool(
                architecture.get("source_file_contents_base64")
            ),
            "training_contract_id": contract.get("contract_id"),
            "state_precision_policy": contract.get("state_precision"),
            "checkpoint_selector": provenance.get("checkpoint_selector"),
        }
        result = {
            "model_id": args.model_id,
            "checkpoint_sha256": catalog_identity["sha256"],
            "checkpoint_size_bytes": int(catalog_identity["size_bytes"]),
            "input": str(args.input.resolve()),
            "input_shape": list(signal.shape),
            "output": str(args.output.resolve()) if args.output else None,
            "output_shape": list(reconstruction.shape),
            "finite": True,
            "mean": float(reconstruction.mean()),
            "std": float(reconstruction.std()),
            "checkpoint_provenance": provenance_summary,
        }
    finally:
        with store_lock(args.db):
            connection = connect(args.db)
            try:
                pruned_files = prune_cache(
                    connection, args.cache_dir, args.max_cache_gib
                )
            finally:
                connection.close()
    result["cache_pruned_files"] = pruned_files
    result["cache_limit_gib"] = args.max_cache_gib
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_name(
            f".{args.report.name}.{os.getpid()}.tmp"
        )
        try:
            temporary.write_text(json.dumps(result, indent=2) + "\n")
            os.replace(temporary, args.report)
        finally:
            temporary.unlink(missing_ok=True)
    print(
        json.dumps(result, indent=2)
    )


if __name__ == "__main__":
    main()
