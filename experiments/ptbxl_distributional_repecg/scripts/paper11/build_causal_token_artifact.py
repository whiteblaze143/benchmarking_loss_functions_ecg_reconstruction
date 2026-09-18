from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from repecg.common import load_config
from repecg.common.ptbxl import PTBXLStore
from repecg.paper11_predictive_state import CausalTokenConfig, build_morphology_tokens, eligible_next_token_examples


_STORE: PTBXLStore | None = None
_MEAN: np.ndarray | None = None
_STD: np.ndarray | None = None
_OUTPUT: Path | None = None
_PREFIX_LENGTH: int | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _initialize_store(config_path: str) -> None:
    global _STORE
    _STORE = PTBXLStore(load_config(config_path))


def _raw_moments(ecg_id: int) -> tuple[int, np.ndarray, np.ndarray]:
    assert _STORE is not None
    values = _STORE.read_record(ecg_id).signal_mv
    return len(values), values.sum(axis=0, dtype=np.float64), np.square(values).sum(axis=0, dtype=np.float64)


def fit_scaler(args: argparse.Namespace) -> None:
    store = PTBXLStore(load_config(args.config))
    frame = store.metadata[store.metadata.strat_fold.isin(range(1, 7))]
    ids = [int(value) for value in frame.index]
    if args.limit is not None:
        ids = ids[: args.limit]
    count = 0
    total = np.zeros(12, dtype=np.float64)
    total_square = np.zeros(12, dtype=np.float64)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_initialize_store, initargs=(str(args.config),)) as executor:
        for sequence, (cell_count, cell_sum, cell_square) in enumerate(executor.map(_raw_moments, ids, chunksize=8), start=1):
            count += cell_count
            total += cell_sum
            total_square += cell_square
            if sequence == 1 or sequence % 500 == 0 or sequence == len(ids):
                print(json.dumps({"record": sequence, "records": len(ids), "samples": count}), flush=True)
    mean = total / count
    variance = np.maximum(total_square / count - np.square(mean), 0.0)
    payload = {
        "kind": "paper11_raw_12lead_scaler",
        "source_folds": [1, 2, 3, 4, 5, 6],
        "records": len(ids),
        "samples": count,
        "mean": mean.tolist(),
        "std": np.sqrt(variance).tolist(),
        "config_sha256": _sha256(args.config),
        "scientific_status": "full_folds_1_6" if args.limit is None else "non_scientific_limited_smoke",
    }
    args.scaler.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.scaler, payload)


def _initialize_build(config_path: str, scaler_path: str, output: str, prefix_length: int) -> None:
    global _STORE, _MEAN, _STD, _OUTPUT, _PREFIX_LENGTH
    _STORE = PTBXLStore(load_config(config_path))
    scaler = json.loads(Path(scaler_path).read_text())
    if scaler.get("source_folds") != [1, 2, 3, 4, 5, 6]:
        raise ValueError("Paper 11 scaler must be fit on folds 1-6 only")
    _MEAN = np.asarray(scaler["mean"], dtype=np.float64)
    _STD = np.asarray(scaler["std"], dtype=np.float64)
    _OUTPUT = Path(output)
    _PREFIX_LENGTH = prefix_length


def _token_identity(token: object) -> tuple[object, ...]:
    return (
        token.anchor_sample,
        token.confirmation_sample,
        token.token_start,
        token.token_stop,
        token.availability_sample,
        _array_hash(token.values),
    )


def _suffix_gate(signal: np.ndarray, tokens: tuple[object, ...], ecg_id: int) -> bool:
    if not tokens:
        return True
    cutoff = tokens[len(tokens) // 2].availability_sample
    altered = signal.copy()
    mode = ecg_id % 3
    if mode == 0:
        altered[cutoff + 1 :] = 0.0
    elif mode == 1:
        altered[cutoff + 1 :] *= -1.0
    else:
        generator = np.random.default_rng(ecg_id)
        altered[cutoff + 1 :] = generator.normal(size=altered[cutoff + 1 :].shape)
    rebuilt = build_morphology_tokens(altered, _MEAN, _STD)
    left = tuple(_token_identity(token) for token in tokens if token.availability_sample <= cutoff)
    right = tuple(_token_identity(token) for token in rebuilt if token.availability_sample <= cutoff)
    return left == right


def _target_gate(signal: np.ndarray, tokens: tuple[object, ...], examples: tuple[dict[str, object], ...]) -> bool:
    eligible = [row for row in examples if row["eligible"]]
    if not eligible:
        return True
    row = eligible[0]
    target = tokens[int(row["target_token_id"])]
    altered = signal.copy()
    altered[target.token_start : target.token_stop] = 17.0
    rebuilt = build_morphology_tokens(altered, _MEAN, _STD)
    for token_id in row["prefix_token_ids"]:
        index = int(token_id)
        if index >= len(rebuilt) or _token_identity(tokens[index]) != _token_identity(rebuilt[index]):
            return False
    return True


def _build_record(ecg_id: int) -> dict[str, object]:
    assert _STORE is not None and _MEAN is not None and _STD is not None and _OUTPUT is not None and _PREFIX_LENGTH is not None
    record = _STORE.read_record(ecg_id)
    tokens = build_morphology_tokens(record.signal_mv, _MEAN, _STD)
    examples = eligible_next_token_examples(tokens, _PREFIX_LENGTH)
    suffix_pass = _suffix_gate(record.signal_mv, tokens, ecg_id)
    target_pass = _target_gate(record.signal_mv, tokens, examples)
    if not suffix_pass or not target_pass:
        raise RuntimeError(f"causality intervention failed for ecg_id={ecg_id}")
    values = np.stack([token.values for token in tokens]) if tokens else np.empty((0, 250, 12), dtype=np.float32)
    anchors = np.asarray([token.anchor_sample for token in tokens], dtype=np.int64)
    confirmations = np.asarray([token.confirmation_sample for token in tokens], dtype=np.int64)
    starts = np.asarray([token.token_start for token in tokens], dtype=np.int64)
    stops = np.asarray([token.token_stop for token in tokens], dtype=np.int64)
    available = np.asarray([token.availability_sample for token in tokens], dtype=np.int64)
    destination = _OUTPUT / "records" / f"{ecg_id}.npz"
    temporary = destination.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        values=values,
        anchors=anchors,
        confirmations=confirmations,
        starts=starts,
        stops=stops,
        availability=available,
        labels=record.labels.astype(np.float32),
        patient_id=np.int64(record.patient_id),
        ecg_id=np.int64(record.ecg_id),
        fold=np.int64(record.fold),
    )
    os.replace(temporary, destination)
    rows = []
    for row in examples:
        item = dict(row)
        item.update({"patient_id": record.patient_id, "ecg_id": ecg_id, "fold": record.fold})
        rows.append(item)
    return {
        "ecg_id": ecg_id,
        "patient_id": record.patient_id,
        "fold": record.fold,
        "tokens": len(tokens),
        "examples": rows,
        "eligible_examples": sum(int(row["eligible"]) for row in examples),
        "suffix_intervention_pass": suffix_pass,
        "target_intervention_pass": target_pass,
        "token_sha256": _array_hash(values),
        "artifact": f"records/{ecg_id}.npz",
    }


def build_artifact(args: argparse.Namespace) -> None:
    store = PTBXLStore(load_config(args.config))
    scaler = json.loads(args.scaler.read_text())
    if scaler.get("scientific_status") != "full_folds_1_6" and args.limit is None:
        raise ValueError("full artifact requires the full folds-1-6 scaler")
    frame = store.metadata[store.metadata.strat_fold.isin(range(1, 8))]
    ids = [int(value) for value in frame.index]
    if args.limit is not None:
        ids = ids[: args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "records").mkdir(exist_ok=True)
    state = {
        "kind": "paper11_causal_token_v1",
        "source_folds": [1, 2, 3, 4, 5, 6, 7],
        "fold8_status": "unread",
        "prefix_length": args.prefix_length,
        "records_requested": len(ids),
        "config_sha256": _sha256(args.config),
        "scaler_sha256": _sha256(args.scaler),
        "detector_hash": CausalTokenConfig().sha256(),
        "limit": args.limit,
    }
    _atomic_json(args.output / "state.json", state)
    progress = args.output / "progress.jsonl"
    completed: dict[int, dict[str, object]] = {}
    if progress.exists():
        for line in progress.read_text().splitlines():
            row = json.loads(line)
            if (args.output / row["artifact"]).is_file():
                completed[int(row["ecg_id"])] = row
    pending = [ecg_id for ecg_id in ids if ecg_id not in completed]
    initializer = (str(args.config), str(args.scaler), str(args.output), args.prefix_length)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_initialize_build, initargs=initializer) as executor:
        for sequence, row in enumerate(executor.map(_build_record, pending, chunksize=1), start=1):
            completed[int(row["ecg_id"])] = row
            with progress.open("a") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            if sequence == 1 or sequence % 100 == 0 or sequence == len(pending):
                print(json.dumps({"completed_now": sequence, "pending_at_start": len(pending), "total_completed": len(completed)}), flush=True)
    rows = [completed[ecg_id] for ecg_id in ids]
    record_frame = pd.DataFrame([{key: value for key, value in row.items() if key != "examples"} for row in rows]).sort_values("ecg_id")
    example_frame = pd.DataFrame([item for row in rows for item in row["examples"]]).sort_values(["ecg_id", "target_token_id"])
    record_frame.to_csv(args.output / "record_qc.csv", index=False)
    example_frame.to_csv(args.output / "example_manifest.csv", index=False)
    patients_by_fold = {str(fold): set(frame.loc[frame.strat_fold == fold, "patient_id"]) for fold in range(1, 8)}
    overlap = any(patients_by_fold[str(left)] & patients_by_fold[str(right)] for left in range(1, 8) for right in range(left + 1, 8))
    if overlap:
        raise RuntimeError("patient overlap across folds 1-7")
    summary = {
        "kind": "paper11_causal_token_v1_g0",
        "scientific_status": "full_g0" if args.limit is None else "non_scientific_limited_smoke",
        "records": len(record_frame),
        "patients": int(frame.loc[ids, "patient_id"].nunique()),
        "tokens": int(record_frame.tokens.sum()),
        "candidate_examples": int(len(example_frame)),
        "eligible_examples": int(example_frame.eligible.sum()) if len(example_frame) else 0,
        "suffix_intervention_all_pass": bool(record_frame.suffix_intervention_pass.all()),
        "target_intervention_all_pass": bool(record_frame.target_intervention_pass.all()),
        "patient_fold_disjoint": True,
        "fold8_status": "unread",
        "detector_hash": state["detector_hash"],
        "scaler_sha256": state["scaler_sha256"],
    }
    _atomic_json(args.output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper 11 causal-token scaler and G0 artifact builder")
    subparsers = parser.add_subparsers(dest="command", required=True)
    scaler = subparsers.add_parser("fit-scaler")
    scaler.add_argument("--config", type=Path, required=True)
    scaler.add_argument("--scaler", type=Path, required=True)
    scaler.add_argument("--workers", type=int, default=8)
    scaler.add_argument("--limit", type=int)
    build = subparsers.add_parser("build")
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--scaler", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--workers", type=int, default=8)
    build.add_argument("--prefix-length", type=int, default=4)
    build.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.workers <= 0 or (args.limit is not None and args.limit <= 0):
        raise ValueError("workers and limit must be positive")
    fit_scaler(args) if args.command == "fit-scaler" else build_artifact(args)


if __name__ == "__main__":
    main()
