#!/usr/bin/env python3
"""
Fit Primary Canonical Nyström RKHS Rhythm Landmarks (M=32) strictly on PTB-XL Development Patients (Folds 1–7).
Strictly PTB-XL-only: No MIMIC data is used during landmark fitting, coordinate standardization, or bandwidth selection.
"""

import sys
import hashlib
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import wfdb

# Path setup
sys.path.insert(0, str(Path("experiments/ptbxl_distributional_repecg/src").resolve()))
from repecg.common.beats import detect_rpeaks
from repecg.paper07_operator.rhythm import extract_rr_atoms, RhythmResponseMap

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fit_rhythm_ptbxl")

OUTPUT_PATH = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/paper07_operator/development_representations/rhythm_fit.npz")
PTBXL_DIR = Path("data/ptb_xl")


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def extract_lead_ii_peaks(sig_ii: np.ndarray, fs: int = 500) -> np.ndarray:
    """Detect R-peaks using canonical detector on Lead II."""
    if len(sig_ii) < 500:
        return np.empty(0, dtype=np.int64)
    basis = np.zeros((len(sig_ii), 8), dtype=np.float64)
    basis[:, 1] = sig_ii
    try:
        detection = detect_rpeaks(basis, fs=fs, primary_lead="II")
        return detection.rpeaks
    except Exception:
        return np.empty(0, dtype=np.int64)


def collect_ptbxl_dev_atoms(max_records: int = 2500, seed: int = 42) -> list:
    logger.info("Collecting rhythm atoms strictly from PTB-XL development patients (strat_fold in 1..7)...")
    meta_path = PTBXL_DIR / "ptbxl_database.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"PTB-XL metadata not found at {meta_path}!")

    df = pd.read_csv(meta_path)
    dev_df = df[df["strat_fold"].isin(range(1, 8))]
    logger.info(f"Total PTB-XL development records available: {len(dev_df):,}")

    # Stratified balance across rhythm classes in development folds
    af_df = dev_df[dev_df["scp_codes"].str.contains("AFIB", na=False)]
    sr_df = dev_df[dev_df["scp_codes"].str.contains("SR", na=False)]
    other_df = dev_df[~dev_df.index.isin(af_df.index) & ~dev_df.index.isin(sr_df.index)]

    logger.info(f"  Development AFIB: {len(af_df):,}")
    logger.info(f"  Development SR:   {len(sr_df):,}")
    logger.info(f"  Development Other:{len(other_df):,}")

    # Sample balanced set
    rng = np.random.default_rng(seed)
    n_af = len(af_df)
    n_sr = min(len(sr_df), 1200)
    n_oth = min(len(other_df), 800)

    selected = pd.concat([
        af_df,
        sr_df.sample(n=n_sr, random_state=seed),
        other_df.sample(n=n_oth, random_state=seed),
    ])
    logger.info(f"Selected {len(selected):,} development records for landmark fitting.")

    atoms_list = []
    processed = 0
    for _, row in selected.iterrows():
        rec_path = PTBXL_DIR / row["filename_hr"]
        if not rec_path.with_suffix(".hea").exists():
            continue
        try:
            rec = wfdb.rdrecord(str(rec_path))
            if "II" not in rec.sig_name:
                continue
            idx = rec.sig_name.index("II")
            sig = rec.p_signal[:, idx]
            sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
            peaks = extract_lead_ii_peaks(sig, fs=500)
            atoms = extract_rr_atoms(peaks, fs=500)
            if len(atoms) >= 3:
                atoms_list.append(atoms)
                processed += 1
            if processed % 500 == 0 and processed > 0:
                logger.info(f"  Extracted atoms from {processed}/{len(selected)} records...")
        except Exception:
            continue

    total_atoms = sum(len(a) for a in atoms_list)
    logger.info(f"Successfully collected {len(atoms_list):,} atom sets ({total_atoms:,} interval atoms).")
    return atoms_list


def main():
    logger.info("=== Fitting Primary Canonical Rhythm Response Map (PTB-XL Development Only) ===")
    atoms_list = collect_ptbxl_dev_atoms(max_records=2500, seed=42)

    total_intervals = sum(len(a) for a in atoms_list)
    if total_intervals < 1000:
        raise RuntimeError("Insufficient training interval atoms extracted from PTB-XL development set!")

    # Fit standardized RhythmResponseMap with M=32 landmarks
    logger.info("Fitting standardized RhythmResponseMap (M=32, deterministic median bandwidth)...")
    rhythm_map = RhythmResponseMap()
    rhythm_map.fit(atoms_list, n_landmarks=32, seed=42)

    logger.info(f"Landmarks shape: {rhythm_map.landmarks.shape}")
    logger.info(f"Coordinate Mean u: [log_rr={rhythm_map.mean_u[0]:.4f}, delta_log_rr={rhythm_map.mean_u[1]:.4f}]")
    logger.info(f"Coordinate Std  u: [log_rr={rhythm_map.std_u[0]:.4f}, delta_log_rr={rhythm_map.std_u[1]:.4f}]")
    logger.info(f"RBF Gamma (deterministic median distance rule): {rhythm_map.gamma:.4f}")
    logger.info(f"Whitening matrix shape: {rhythm_map.whitening.shape} (cond={np.linalg.cond(rhythm_map.whitening):.2f})")

    # Save canonical artifact
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rhythm_map.save(OUTPUT_PATH)
    logger.info(f"Saved primary canonical rhythm_fit.npz to {OUTPUT_PATH}")

    # Checksum and verification
    sha256 = compute_sha256(OUTPUT_PATH)
    logger.info(f"Canonical rhythm_fit.npz SHA256: {sha256}")

    # Verify reload self-consistency
    reloaded = RhythmResponseMap.load(OUTPUT_PATH)
    assert reloaded.dim == 32
    np.testing.assert_allclose(rhythm_map.landmarks, reloaded.landmarks)
    np.testing.assert_allclose(rhythm_map.whitening, reloaded.whitening)
    np.testing.assert_allclose(rhythm_map.mean_u, reloaded.mean_u)
    np.testing.assert_allclose(rhythm_map.std_u, reloaded.std_u)
    logger.info("Reload and numerical self-consistency verification: PASS!")


if __name__ == "__main__":
    main()
