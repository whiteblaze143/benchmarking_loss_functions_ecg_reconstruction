#!/usr/bin/env python3
"""Milestone R028: Complete Final Folds 1-8 Dataset & Representation Pipeline.

Implements the locked experimental protocol for CVPR promotion:
1. Loads final_scaler.json (folds 1-8).
2. Fits response_fit_final.npz (voltage scale, whitening, IMQ Nyström landmarks)
   strictly using physical & response atoms from folds 1-8 in memory.
3. Materializes consolidated training datasets on GPU:
   - representation_final_train_basis_only.npz [17418, 8, 16, 128]
   - representation_final_train_operator_span.npz [17418, 16, 16, 128]
4. Extracts Fold 9 (final_select, 2,183 records) and computes validation representations:
   - representation_final_select_waveforms.npz [2183, 16, 16, 128]
5. Pre-hashes and freezes the expanded Grassmann subspace bank:
   - expanded_grassmann_subspace_bank.npz (50 subspaces per rank r in 1..7 = 350 subspaces)
Fold 10 remains strictly locked and untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

# Ensure repecg is in sys.path
SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from repecg.common import load_config
from repecg.common.beats import detect_rpeaks
from repecg.common.kernels import NystromMap, WhiteningTransform, audit_nystrom, biased_mmd2
from repecg.common.phase import phase_normalize
from repecg.common.preprocess import (
    LeadScaler,
    LeadScalerAccumulator,
    bandpass_ecg,
    independent_basis,
)
from repecg.common.ptbxl import PTBXLStore
from repecg.paper07_operator import (
    normalize_operator,
    record_training_operators,
    response_atoms,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_final_r028")

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs/common.yaml"
CODEX_DIR = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg")
OUT_DIR = CODEX_DIR / "paper07_operator/final_representations"
PHASE_CACHE_DIR = CODEX_DIR / "paper02_kernel_mean"
DEV_TRAIN_DIR = PHASE_CACHE_DIR / "development_train"
DEV_SELECT_DIR = PHASE_CACHE_DIR / "development_select"
DEV_SCALER_PATH = PHASE_CACHE_DIR / "development_scaler.json"

_STORE: PTBXLStore | None = None
_SCALER: LeadScaler | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _initialize_worker(config_path: str, scaler_path: str | None) -> None:
    global _STORE, _SCALER
    _STORE = PTBXLStore(load_config(config_path))
    if scaler_path is not None:
        payload = json.loads(Path(scaler_path).read_text())
        _SCALER = LeadScaler(
            mean=np.asarray(payload["mean"], dtype=np.float64),
            std=np.asarray(payload["std"], dtype=np.float64),
        )


def _filtered_basis(ecg_id: int) -> Tuple[Any, np.ndarray]:
    assert _STORE is not None
    record = _STORE.read_record(ecg_id)
    return record, independent_basis(bandpass_ecg(record.signal_mv))


def _prepare_record(ecg_id: int) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    assert _SCALER is not None
    record, basis = _filtered_basis(ecg_id)
    detection = detect_rpeaks(basis)
    beats = phase_normalize(_SCALER.transform(basis), detection.valid_intervals)
    minimum = int(_STORE.config.raw["beats"]["min_full_cycles"])
    eligible = len(beats) >= minimum
    row = {
        "ecg_id": record.ecg_id,
        "patient_id": record.patient_id,
        "fold": record.fold,
        "detector_lead": detection.lead,
        "rpeaks": len(detection.rpeaks),
        "valid_cycles": len(beats),
        "eligible": eligible,
        "reason": "" if eligible else f"fewer_than_{minimum}_valid_cycles",
        "artifact": f"records/{record.ecg_id}.npz",
    }
    return row, beats.astype(np.float32), record.labels.astype(np.float32)


def get_heldout_test_operators() -> np.ndarray:
    """Define operators that MUST NEVER be seen during training."""
    e = np.eye(8, dtype=np.float64)
    q_III = normalize_operator(e[1] - e[0])
    q_aVR = normalize_operator(-e[0] - e[1])
    q_aVL = normalize_operator(e[0] - 0.5 * e[1])
    q_aVF = normalize_operator(e[1] - 0.5 * e[0])
    q_V3_V2 = normalize_operator(e[4] - e[3])
    q_V1_V6 = normalize_operator(e[2] - e[7])
    q_I_V4 = normalize_operator(e[0] + e[5])
    q_II_V5 = normalize_operator(e[1] - e[6])
    return np.stack([q_III, q_aVR, q_aVL, q_aVF, q_V3_V2, q_V1_V6, q_I_V4, q_II_V5])


def is_safe_operator(q: np.ndarray, heldout: np.ndarray, min_dist: float = 0.08) -> bool:
    q_norm = normalize_operator(q)
    for h in heldout:
        if np.linalg.norm(q_norm - h) < min_dist or np.linalg.norm(q_norm + h) < min_dist:
            return False
    return True


def sample_training_operators(rng: np.random.Generator, heldout: np.ndarray, n_span: int = 8) -> np.ndarray:
    span_ops = []
    attempts = 0
    while len(span_ops) < n_span:
        attempts += 1
        if attempts > 2000:
            raise RuntimeError("Exceeded max attempts to sample safe operators")
        mode = len(span_ops) % 2
        if mode == 0:
            alpha = rng.normal(size=8)
        else:
            k = rng.choice([2, 3])
            active = rng.choice(8, size=k, replace=False)
            alpha = np.zeros(8)
            alpha[active] = rng.normal(size=k)
        if np.linalg.norm(alpha) < 1e-6:
            continue
        q = normalize_operator(alpha)
        if is_safe_operator(q, heldout):
            span_ops.append(q)
    return np.stack(span_ops, axis=0)


class FastResponseComputer:
    def __init__(self, fit_path: Path, device: torch.device):
        data = np.load(fit_path)
        self.v_scale = float(data["voltage_scale"])
        self.mean = data["physical_mean"].astype(np.float32)
        self.std = data["physical_std"].astype(np.float32)

        self.landmarks = torch.as_tensor(data["landmarks"], device=device, dtype=torch.float32)
        self.inv_root = torch.as_tensor(data["inverse_root"], device=device, dtype=torch.float32)
        self.c2 = float(data["c2"])
        self.w_mean = torch.as_tensor(data["whitening_mean"], device=device, dtype=torch.float32)
        self.w_comp = torch.as_tensor(data["whitening_components"], device=device, dtype=torch.float32)
        self.w_scale = torch.as_tensor(data["whitening_scales"], device=device, dtype=torch.float32)
        self.device = device

    def compute_record_responses(self, beats_mv: np.ndarray, q_bank: np.ndarray) -> np.ndarray:
        n_ops = len(q_bank)
        atoms = [response_atoms(beats_mv.astype(np.float64), q, self.v_scale) for q in q_bank]
        atoms = np.stack(atoms, axis=0)  # (K, beat, 256, 2)
        cells = atoms.reshape(n_ops, len(beats_mv), 16, 16, 2).transpose(0, 2, 1, 3, 4).reshape(n_ops, 16, -1, 2)
        t_cells = torch.as_tensor(cells, device=self.device, dtype=torch.float32)
        white = (t_cells - self.w_mean) @ self.w_comp * self.w_scale
        d2 = torch.cdist(white, self.landmarks.expand(n_ops, 16, -1, -1)).square() + self.c2
        feat = torch.rsqrt(d2) @ self.inv_root
        return feat.mean(dim=2).cpu().numpy().astype(np.float32)  # (K, 16, 128)


def step1_fit_response_mapping(scaler_path: Path, resp_fit_path: Path, seed: int = 42) -> Dict[str, Any]:
    logger.info("=== Step 1: Fitting Response Mapping & IMQ Landmarks on Folds 1-8 ===")
    if resp_fit_path.exists():
        logger.info(f"Using existing response fit at {resp_fit_path}")
        return json.loads((resp_fit_path.parent / "response_fit_manifest.json").read_text())

    scaler_payload = json.loads(scaler_path.read_text())
    physical_mean = np.asarray(scaler_payload["mean"], dtype=np.float64)
    physical_std = np.asarray(scaler_payload["std"], dtype=np.float64)

    dev_scaler = json.loads(DEV_SCALER_PATH.read_text())
    m_dev = np.asarray(dev_scaler["mean"], dtype=np.float64)
    s_dev = np.asarray(dev_scaler["std"], dtype=np.float64)

    qc_train = pd.read_csv(DEV_TRAIN_DIR / "qc.csv").query("eligible").reset_index(drop=True)
    qc_select = pd.read_csv(DEV_SELECT_DIR / "qc.csv").query("eligible").reset_index(drop=True)

    items_1_8 = []
    for _, r in qc_train.iterrows():
        items_1_8.append((DEV_TRAIN_DIR / r["artifact"], int(r["ecg_id"])))
    for _, r in qc_select.iterrows():
        items_1_8.append((DEV_SELECT_DIR / r["artifact"], int(r["ecg_id"])))

    keyed = []
    for idx, (path, ecg_id) in enumerate(items_1_8):
        token = f"paper07-final-res:{seed}:{ecg_id}".encode()
        keyed.append((hashlib.sha256(token).digest(), idx))
    order = [idx for _, idx in sorted(keyed)]

    target_count = 500_000
    n_sample_records = 250
    sample_items = [items_1_8[idx] for idx in order[:n_sample_records]]

    logger.info(f"Loading {n_sample_records} records in parallel (16 threads) for reservoir...")
    t0_load = time.time()

    def _load_sample(item):
        p, ecg_id = item
        with np.load(p) as d:
            b_dev = d["beats"].astype(np.float64)
        return b_dev * s_dev + m_dev, ecg_id

    with ThreadPoolExecutor(max_workers=16) as ex:
        sample_beats_phys = list(ex.map(_load_sample, sample_items))

    logger.info(f"Loaded {len(sample_beats_phys)} records in {time.time() - t0_load:.1f}s.")

    # 1. Physical reservoir for voltage scale
    chunks = [b.reshape(-1, 8) for b, _ in sample_beats_phys]
    physical = np.concatenate(chunks)[:target_count]
    voltage_scale = float(np.std(physical))
    logger.info(f"Folds 1-8 voltage scale: {voltage_scale:.6f} mV")

    # 2. Response reservoir
    logger.info("Sampling 500k response atoms from in-memory beats...")
    resp_chunks = []
    observed_resp = 0
    for b_phys, ecg_id in sample_beats_phys:
        for q in record_training_operators(ecg_id, seed):
            atoms = response_atoms(b_phys, q, voltage_scale)
            vals = atoms.reshape(-1, atoms.shape[-1])
            take = min(len(vals), target_count - observed_resp)
            resp_chunks.append(vals[:take])
            observed_resp += take
            if observed_resp == target_count:
                break
        if observed_resp == target_count:
            break
    response_res = np.concatenate(resp_chunks)

    # 3. Whitening Transform
    logger.info("Fitting Whitening Transform...")
    whitening = WhiteningTransform.fit(response_res)
    whitened = whitening.transform(response_res)

    # 4. IMQ Nyström Mapping
    logger.info("Fitting IMQ Nyström Mapping...")
    mapping = None
    audits = {}

    audit_cells = []
    for b_phys, ecg_id in sample_beats_phys[:32]:
        q = record_training_operators(ecg_id, seed)[0]
        atoms = response_atoms(b_phys, q, voltage_scale)
        cells = atoms.reshape(len(atoms), 16, 16, 2).transpose(1, 0, 2, 3)
        for c in cells:
            audit_cells.append(c.reshape(-1, 2))

    for landmarks in (128, 256):
        m = NystromMap.fit(whitened, landmarks=landmarks, seed=seed)
        rng = np.random.default_rng(seed)
        w_cells = [whitening.transform(c) for c in audit_cells]
        pairs = 1000
        exact = np.empty(pairs)
        approx = np.empty(pairs)
        for p in range(pairs):
            l, r = rng.choice(len(w_cells), size=2, replace=False)
            exact[p] = biased_mmd2(w_cells[l], w_cells[r])
            approx[p] = np.square(m.mean(w_cells[l]) - m.mean(w_cells[r])).sum()
        res = audit_nystrom(exact, approx)
        audits[str(landmarks)] = res
        logger.info(f"Landmarks {landmarks}: Spearman={res['spearman']:.4f}, RelError={res['median_relative_error']:.4f}, Passed={res['passed']}")
        if res["passed"]:
            mapping = m
            break

    if mapping is None:
        raise RuntimeError(f"Nyström audit failed: {audits}")

    resp_fit_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        resp_fit_path,
        physical_mean=physical_mean,
        physical_std=physical_std,
        voltage_scale=np.asarray(voltage_scale),
        whitening_mean=whitening.mean,
        whitening_components=whitening.components,
        whitening_scales=whitening.scales,
        landmarks=mapping.landmarks,
        inverse_root=mapping.inverse_root,
        c2=np.asarray(mapping.c2),
    )
    logger.info(f"Saved response_fit_final.npz to {resp_fit_path}")

    summary = {
        "source_split": "final_train (Folds 1-8)",
        "voltage_scale": voltage_scale,
        "n_landmarks": len(mapping.landmarks),
        "audits": audits,
        "scaler_sha256": _sha256(scaler_path),
        "fit_path": str(resp_fit_path),
    }
    (resp_fit_path.parent / "response_fit_manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def step2_materialize_final_train_datasets(
    resp_fit_path: Path,
    out_dir: Path,
    device: torch.device,
) -> None:
    logger.info("=== Step 2: Materializing Final Train Datasets on GPU (Folds 1-8) ===")
    p_basis = out_dir / "representation_final_train_basis_only.npz"
    p_span = out_dir / "representation_final_train_operator_span.npz"
    if p_basis.exists() and p_span.exists():
        logger.info("Final train datasets already exist.")
        return

    dev_scaler = json.loads(DEV_SCALER_PATH.read_text())
    s_dev = np.asarray(dev_scaler["std"], dtype=np.float32)
    m_dev = np.asarray(dev_scaler["mean"], dtype=np.float32)

    qc_train = pd.read_csv(DEV_TRAIN_DIR / "qc.csv").query("eligible")
    qc_select = pd.read_csv(DEV_SELECT_DIR / "qc.csv").query("eligible")

    records_meta = []
    for _, r in qc_train.iterrows():
        records_meta.append((DEV_TRAIN_DIR / r["artifact"], int(r["ecg_id"]), int(r["patient_id"])))
    for _, r in qc_select.iterrows():
        records_meta.append((DEV_SELECT_DIR / r["artifact"], int(r["ecg_id"]), int(r["patient_id"])))

    n_records = len(records_meta)
    logger.info(f"Total eligible final_train records across Folds 1-8: {n_records}")

    heldout = get_heldout_test_operators()
    computer = FastResponseComputer(resp_fit_path, device)

    # Pre-allocate output arrays
    basis_ops_all = np.empty((n_records, 8, 8), dtype=np.float32)
    basis_resps_all = np.empty((n_records, 8, 16, 128), dtype=np.float32)
    span_ops_all = np.empty((n_records, 16, 8), dtype=np.float32)
    span_resps_all = np.empty((n_records, 16, 16, 128), dtype=np.float32)

    labels_all = np.empty((n_records, 5), dtype=np.float32)
    ecg_ids_all = np.array([m[1] for m in records_meta], dtype=np.int64)
    patient_ids_all = np.array([m[2] for m in records_meta], dtype=np.int64)

    basis_8 = np.eye(8, dtype=np.float32)
    chunk_size = 500
    t0_start = time.time()

    for chunk_start in range(0, n_records, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_records)
        chunk_items = records_meta[chunk_start:chunk_end]

        def _load_item(item):
            p, eid, pid = item
            with np.load(p) as d:
                b = d["beats"].astype(np.float32) * s_dev + m_dev
                lbl = d["labels"].astype(np.float32)
            return b, lbl

        with ThreadPoolExecutor(max_workers=16) as ex:
            chunk_data = list(ex.map(_load_item, chunk_items))

        for idx_in_chunk, (beats, lbl) in enumerate(chunk_data):
            rec_idx = chunk_start + idx_in_chunk
            labels_all[rec_idx] = lbl

            rng = np.random.default_rng(np.random.SeedSequence([42, int(ecg_ids_all[rec_idx])]))
            span_8 = sample_training_operators(rng, heldout, n_span=8).astype(np.float32)
            bank_16 = np.concatenate([basis_8, span_8], axis=0)

            resps_16 = computer.compute_record_responses(beats, bank_16)

            basis_ops_all[rec_idx] = basis_8
            basis_resps_all[rec_idx] = resps_16[:8]

            span_ops_all[rec_idx] = bank_16
            span_resps_all[rec_idx] = resps_16

        elapsed = time.time() - t0_start
        rate = chunk_end / elapsed
        logger.info(f"Processed {chunk_end}/{n_records} final_train records ({rate:.1f} rec/s, elapsed: {elapsed:.1f}s)")

    logger.info(f"Saving basis-only dataset to {p_basis}...")
    np.savez_compressed(
        p_basis,
        operators=basis_ops_all,
        responses=basis_resps_all,
        labels=labels_all,
        ecg_ids=ecg_ids_all,
        patient_ids=patient_ids_all,
    )

    logger.info(f"Saving operator-span dataset to {p_span}...")
    np.savez_compressed(
        p_span,
        operators=span_ops_all,
        responses=span_resps_all,
        labels=labels_all,
        ecg_ids=ecg_ids_all,
        patient_ids=patient_ids_all,
    )
    logger.info("Final train datasets saved successfully.")


def step3_build_final_select_waveforms(
    scaler_path: Path,
    resp_fit_path: Path,
    out_dir: Path,
    device: torch.device,
    workers: int = 8,
) -> None:
    logger.info("=== Step 3: Extracting Fold 9 (final_select) & Computing Validation Waveforms ===")
    p_val = out_dir / "representation_final_select_waveforms.npz"
    if p_val.exists():
        logger.info(f"final_select waveforms already exist at {p_val}")
        return

    config = load_config(CONFIG_PATH)
    store = PTBXLStore(config)
    ids = [int(v) for v in store.split_frame("final_select").index]

    logger.info(f"Processing {len(ids)} Fold 9 records with {workers} workers...")
    t0 = time.time()
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize_worker,
        initargs=(str(CONFIG_PATH), str(scaler_path)),
    ) as executor:
        results = list(executor.map(_prepare_record, ids, chunksize=8))

    eligible_data = [item for item in results if item[0]["eligible"]]
    n_eligible = len(eligible_data)
    logger.info(f"Fold 9 complete in {time.time() - t0:.1f}s. Eligible: {n_eligible}/{len(ids)}")

    computer = FastResponseComputer(resp_fit_path, device)

    # 16 standard validation evaluation configurations
    e = np.eye(8, dtype=np.float32)
    q_bank = np.stack([
        e[0],                                       # 0: I
        e[1],                                       # 1: II
        e[2],                                       # 2: V1
        e[3],                                       # 3: V2
        e[4],                                       # 4: V3
        e[5],                                       # 5: V4
        e[6],                                       # 6: V5
        e[7],                                       # 7: V6
        normalize_operator(e[1] - e[0]),             # 8: III
        normalize_operator(-e[0] - e[1]),            # 9: aVR
        normalize_operator(e[0] - 0.5 * e[1]),       # 10: aVL
        normalize_operator(e[1] - 0.5 * e[0]),       # 11: aVF
        normalize_operator(e[4] - e[3]),             # 12: V3 - V2 (S_icm)
        normalize_operator(e[2] - e[7]),             # 13: V1 - V6
        normalize_operator(e[0] + e[5]),             # 14: I + V4
        normalize_operator(e[1] - e[6]),             # 15: II - V5
    ]).astype(np.float32)

    val_resps_all = np.empty((n_eligible, 16, 16, 128), dtype=np.float32)
    labels_all = np.empty((n_eligible, 5), dtype=np.float32)
    ecg_ids_all = np.array([item[0]["ecg_id"] for item in eligible_data], dtype=np.int64)
    patient_ids_all = np.array([item[0]["patient_id"] for item in eligible_data], dtype=np.int64)

    t0_resp = time.time()
    for idx, (row, beats, lbl) in enumerate(eligible_data):
        labels_all[idx] = lbl
        beats_mv = beats * computer.std + computer.mean
        resps = computer.compute_record_responses(beats_mv, q_bank)
        val_resps_all[idx] = resps
        if (idx + 1) % 500 == 0:
            logger.info(f"Computed responses for {idx + 1}/{n_eligible} Fold 9 records...")

    logger.info(f"Saving final_select waveforms to {p_val} (took {time.time() - t0_resp:.1f}s)...")
    np.savez_compressed(
        p_val,
        operators=np.tile(q_bank[None, :, :], (n_eligible, 1, 1)),
        responses=val_resps_all,
        labels=labels_all,
        ecg_ids=ecg_ids_all,
        patient_ids=patient_ids_all,
    )


def step4_freeze_grassmann_bank(out_dir: Path, n_per_rank: int = 50, seed: int = 42) -> Path:
    logger.info(f"=== Step 4: Pre-hashing & Freezing Expanded Grassmann Subspace Bank ({n_per_rank} per rank) ===")
    p_bank = out_dir / "expanded_grassmann_subspace_bank.npz"
    if p_bank.exists():
        logger.info(f"Grassmann bank already exists at {p_bank}")
        return p_bank

    rng = np.random.default_rng(seed)
    bank_data = {}
    for r in range(1, 8):
        subspaces = []
        for _ in range(n_per_rank):
            G = rng.normal(size=(8, r))
            Q, _ = np.linalg.qr(G)
            subspaces.append(Q.T.astype(np.float64))  # [r, 8]
        bank_data[f"rank_{r}"] = np.stack(subspaces, axis=0)  # [n_per_rank, r, 8]

    np.savez_compressed(p_bank, **bank_data)
    manifest = {
        "dataset": "expanded_grassmann_subspace_bank",
        "n_subspaces_per_rank": n_per_rank,
        "ranks": list(range(1, 8)),
        "total_subspaces": n_per_rank * 7,
        "seed": seed,
        "sha256": _sha256(p_bank),
    }
    (out_dir / "expanded_grassmann_bank_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    logger.info(f"Saved and locked Grassmann bank to {p_bank} (SHA-256: {manifest['sha256']})")
    return p_bank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    logger.info(f"Running pipeline on device: {device}")

    scaler_path = OUT_DIR / "final_scaler.json"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing {scaler_path}. Run fit_final_scaler first.")

    resp_fit_path = OUT_DIR / "response_fit_final.npz"

    # Step 1: Fit response mapping & Nystrom landmarks on Folds 1-8
    step1_fit_response_mapping(scaler_path, resp_fit_path, seed=args.seed)

    # Step 2: Materialize final_train datasets on GPU
    step2_materialize_final_train_datasets(resp_fit_path, OUT_DIR, device)

    # Step 3: Extract Fold 9 & compute final_select waveforms
    step3_build_final_select_waveforms(scaler_path, resp_fit_path, OUT_DIR, device, workers=args.workers)

    # Step 4: Pre-hash and freeze expanded Grassmann bank
    step4_freeze_grassmann_bank(OUT_DIR, n_per_rank=50, seed=args.seed)

    logger.info("=== MILESTONE R028 DATASET & REPRESENTATION PIPELINE COMPLETE! ===")


if __name__ == "__main__":
    main()
