#!/usr/bin/env python3
"""Script 14: Phase I - Authentic and Synthetic Measurement Robustness.
Evaluates QRS detection and ventricular rate estimation robustness across:
  - Noise Holter Database authentic artifacts
  - NSTDB calibrated signal-to-noise ratios (24dB down to -6dB)
Generates:
  - results/noise_robustness.csv
"""

import os
import sys
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import wfdb
import wfdb.processing

root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir))

def main():
    results_dir = root_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE I: AUTHENTIC AND SYNTHETIC MEASUREMENT ROBUSTNESS")
    print("=" * 80)

    # 1. NSTDB stress testing
    nstdb_dir = "/data/mithunmanivannan/datasets/nstdb/mit-bih-noise-stress-test-database-1.0.0"
    records = []

    if os.path.exists(nstdb_dir):
        # Clean record 118 and 119 with calibrated noise levels: 24dB, 18dB, 12dB, 6dB, 0dB, -6dB
        snr_levels = ["24", "18", "12", "06", "00", "_6"]
        for snr in snr_levels:
            rec_name = f"118e{snr}"
            rec_path = os.path.join(nstdb_dir, rec_name)
            if not os.path.exists(f"{rec_path}.hea"):
                continue

            try:
                header = wfdb.rdheader(rec_path)
                fs = float(header.fs)
                # Read 5 minutes of ECG
                n_samp = int(min(header.sig_len, fs * 300.0))
                rec = wfdb.rdrecord(rec_path, sampto=n_samp)
                ref_ann = wfdb.rdann(rec_path, 'atr', sampto=n_samp)
                ref_beats = np.array(ref_ann.sample)

                # Detector
                det_beats = wfdb.processing.xqrs_detect(rec.p_signal[:, 0], fs=fs, verbose=False)

                tol = int(round(0.050 * fs))
                matched = sum(1 for rb in ref_beats if np.any(np.abs(det_beats - rb) <= tol))
                se = matched / len(ref_beats) if len(ref_beats) > 0 else 1.0
                ppv = matched / len(det_beats) if len(det_beats) > 0 else 1.0

                ref_rate = (len(ref_beats) / 300.0) * 60.0
                det_rate = (len(det_beats) / 300.0) * 60.0
                rate_err = abs(det_rate - ref_rate)

                records.append({
                    "test_type": "NSTDB_SYNTHETIC_NOISE",
                    "condition": f"SNR_{snr.replace('_', '-')}dB",
                    "n_ref_beats": len(ref_beats),
                    "n_det_beats": len(det_beats),
                    "sensitivity": se,
                    "positive_predictive_value": ppv,
                    "ref_rate_bpm": ref_rate,
                    "det_rate_bpm": det_rate,
                    "rate_error_bpm": rate_err,
                    "rate_available_pct": 100.0 if len(det_beats) > 0 else 0.0
                })
            except Exception as e:
                print(f"Error testing {rec_name}: {e}")

    # 2. Authentic Noise Holter Database
    nh_dir = "/data/mithunmanivannan/datasets/noise_holter_database"
    if os.path.exists(nh_dir):
        nh_hea = sorted(glob.glob(os.path.join(nh_dir, "*.hea")))
        for f in nh_hea[:10]:
            r_base = os.path.splitext(f)[0]
            r_id = os.path.basename(r_base)
            try:
                hdr = wfdb.rdheader(r_base)
                fs = float(hdr.fs)
                n_samp = int(min(hdr.sig_len, fs * 120.0))
                rec = wfdb.rdrecord(r_base, sampto=n_samp)
                det = wfdb.processing.xqrs_detect(rec.p_signal[:, 0], fs=fs, verbose=False)
                rate = (len(det) / (n_samp / fs)) * 60.0

                records.append({
                    "test_type": "AUTHENTIC_NOISE_HOLTER",
                    "condition": f"Record_{r_id}",
                    "n_ref_beats": np.nan,
                    "n_det_beats": len(det),
                    "sensitivity": np.nan,
                    "positive_predictive_value": np.nan,
                    "ref_rate_bpm": np.nan,
                    "det_rate_bpm": rate,
                    "rate_error_bpm": np.nan,
                    "rate_available_pct": 100.0 if len(det) > 10 else 0.0
                })
            except Exception:
                pass

    df_noise = pd.DataFrame(records)
    out_csv = results_dir / "noise_robustness.csv"
    df_noise.to_csv(out_csv, index=False)
    print(f"Saved noise robustness results to {out_csv}")
    print(df_noise.to_string(index=False))

if __name__ == "__main__":
    main()
