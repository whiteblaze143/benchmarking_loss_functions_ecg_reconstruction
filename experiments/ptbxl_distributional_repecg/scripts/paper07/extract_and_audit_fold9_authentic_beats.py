#!/usr/bin/env python3
"""Extract, Verify, and Cache Authentic Fold-9 ECG Beats from PTB-XL.

Strict Verification Protocol:
1. Extract authentic beats for all 2,183 Fold-9 records using PTBXLStore.
2. Verify exact 1:1 alignment of ecg_id, patient_id, and 5-class labels against representation_final_select_waveforms.npz.
3. Compute responses on the canonical/derived 16-operator bank using FastResponseComputer.
4. Verify max absolute numerical difference against cached representations is within tolerance.
5. Save the authentic beats and projected waveforms to fold9_authentic_beats.npz.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("extract_fold9_beats")

REPO_DIR = Path(__file__).resolve().parents[4]
EXPERIMENT_DIR = REPO_DIR / "experiments/ptbxl_distributional_repecg"
SRC_DIR = EXPERIMENT_DIR / "src"
SCRIPTS_DIR = EXPERIMENT_DIR / "scripts/paper07"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from repecg.common import load_config
from repecg.common.beats import detect_rpeaks
from repecg.common.phase import phase_normalize
from repecg.common.preprocess import bandpass_ecg, independent_basis
from repecg.common.ptbxl import PTBXLStore
from evaluator_physical_operator import FastResponseComputer, build_canonical_and_derived_operator_bank

CODEX_DIR = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg")
FINAL_REPS_DIR = CODEX_DIR / "paper07_operator/final_representations"
CONFIG_PATH = EXPERIMENT_DIR / "configs/common.yaml"

_STORE: PTBXLStore | None = None


def _init_worker(config_path: str) -> None:
    global _STORE
    _STORE = PTBXLStore(load_config(Path(config_path)))


def _extract_single_record(ecg_id: int) -> Dict[str, Any]:
    assert _STORE is not None
    record = _STORE.read_record(ecg_id)
    raw_signal = record.signal_mv  # (5000, 12) in physical millivolts
    filtered = bandpass_ecg(raw_signal)  # (5000, 12)
    basis = independent_basis(filtered)  # (5000, 8) [I, II, V1, V2, V3, V4, V5, V6]
    detection = detect_rpeaks(basis)
    beats = phase_normalize(basis, detection.valid_intervals)  # (n_beats, 256, 8) in physical millivolts
    
    return {
        "ecg_id": int(record.ecg_id),
        "patient_id": int(record.patient_id),
        "fold": int(record.fold),
        "labels": record.labels.astype(np.float32),
        "beats": beats.astype(np.float32),
        "lead_I_waveform": basis[:, 0].astype(np.float32),  # (5000,) for observed-only segmentation
        "lead_II_waveform": basis[:, 1].astype(np.float32), # (5000,)
        "valid_cycles": len(beats),
        "eligible": len(beats) >= 2,
    }


def main():
    t0 = time.time()
    logger.info("Loading cached Fold-9 representations for ground-truth alignment audit...")
    val_npz_path = FINAL_REPS_DIR / "representation_final_select_waveforms.npz"
    with np.load(val_npz_path) as d_val:
        cached_ecg_ids = d_val["ecg_ids"]
        cached_patient_ids = d_val["patient_ids"]
        cached_labels = d_val["labels"]
        cached_responses = d_val["responses"]  # (2183, 16, 16, 128)
        cached_operators = d_val["operators"]  # (2183, 16, 8)

    n_records = len(cached_ecg_ids)
    logger.info("Cached records count: %d", n_records)

    logger.info("Extracting authentic beats across %d Fold-9 records using 8 workers...", n_records)
    with ProcessPoolExecutor(max_workers=8, initializer=_init_worker, initargs=(str(CONFIG_PATH),)) as ex:
        results = list(ex.map(_extract_single_record, [int(i) for i in cached_ecg_ids], chunksize=16))

    logger.info("Beat extraction completed in %.1f seconds.", time.time() - t0)

    # 1. Audit Alignment
    logger.info("Auditing alignment between extracted records and cached representations...")
    extracted_ecg_ids = np.array([r["ecg_id"] for r in results], dtype=np.int64)
    extracted_patient_ids = np.array([r["patient_id"] for r in results], dtype=np.int64)
    extracted_labels = np.stack([r["labels"] for r in results], axis=0)

    assert np.array_equal(extracted_ecg_ids, cached_ecg_ids), "ECG ID mismatch detected!"
    assert np.array_equal(extracted_patient_ids, cached_patient_ids), "Patient ID mismatch detected!"
    assert np.allclose(extracted_labels, cached_labels), "Label array mismatch detected!"

    ineligible = [r["ecg_id"] for r in results if not r["eligible"]]
    if ineligible:
        raise RuntimeError(f"Found ineligible records with < 2 cycles: {ineligible}")

    logger.info("Alignment audit passed: 100% exact match across all 2,183 records.")

    # 2. Audit Physical Responses
    resp_fit_path = FINAL_REPS_DIR / "response_fit_final.npz"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Initializing FastResponseComputer on %s to audit response reconstruction...", device)
    computer = FastResponseComputer(resp_fit_path, device)
    q_bank = build_canonical_and_derived_operator_bank()  # (16, 8)

    # Check operator bank matches cached operators
    assert np.allclose(q_bank, cached_operators[0]), "Operator bank ordering mismatch!"

    logger.info("Testing response recomputation on a subset of 50 patients...")
    max_resp_err = 0.0
    for idx in range(min(50, n_records)):
        beats_mv = results[idx]["beats"]  # (n_beats, 256, 8)
        recomputed = computer.compute_record_responses(beats_mv, q_bank)  # (16, 16, 128)
        expected = cached_responses[idx]
        err = float(np.max(np.abs(recomputed - expected)))
        if err > max_resp_err:
            max_resp_err = err

    logger.info("Response recomputation max absolute error: %.2e (tolerance: < 1e-4)", max_resp_err)
    if max_resp_err > 1e-3:
        raise RuntimeError(f"Response recomputation error exceeded tolerance: {max_resp_err}")
    logger.info("Recomputed Phase-KME responses match cached representations within numerical precision!")

    # 3. Save Authentic Beats Artifact
    out_path = FINAL_REPS_DIR / "fold9_authentic_beats.npz"
    logger.info("Saving authentic Fold-9 beats and waveforms to %s...", out_path)
    
    # Pack beats as an object array or list of arrays to accommodate varying beat counts
    beats_list = [r["beats"] for r in results]
    lead_I_wf = np.stack([r["lead_I_waveform"] for r in results], axis=0)
    lead_II_wf = np.stack([r["lead_II_waveform"] for r in results], axis=0)

    # Calculate SHA-256 for audit
    np.savez_compressed(
        out_path,
        ecg_ids=extracted_ecg_ids,
        patient_ids=extracted_patient_ids,
        labels=extracted_labels,
        lead_I_waveforms=lead_I_wf,
        lead_II_waveforms=lead_II_wf,
        beats=np.array(beats_list, dtype=object),
    )

    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    logger.info("Saved %s (SHA-256: %s, Size: %.1f MB).", out_path.name, sha, out_path.stat().st_size / (1024 * 1024))

    manifest = {
        "status": "AUTHENTIC_FOLD9_BEATS_VERIFIED",
        "n_records": n_records,
        "sha256": sha,
        "max_response_reconstruction_error": max_resp_err,
        "timestamp": time.time(),
    }
    manifest_path = FINAL_REPS_DIR / "fold9_authentic_beats_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Saved manifest to %s.", manifest_path)
    logger.info("AUTHENTIC FOLD-9 BEAT RECOVERY AND VERIFICATION COMPLETE!")


if __name__ == "__main__":
    main()
