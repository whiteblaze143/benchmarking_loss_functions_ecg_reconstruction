"""Technical RR availability from source detector; never uses .atr/.qrs inputs.

This is not admission of a complete clinical inference pipeline. Published
filtering/standalone-head operating points remain separate admission gates.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb


def detect_observed(signal, fs, binary, library):
    # Exact source PECG warm-up policy repeats five seconds of observed signal;
    # detections from the repeated prefix are discarded, never made into RRs.
    warmup = min(len(signal), int(5 * fs))
    padded = np.concatenate([signal[:warmup], signal])
    environment = dict(os.environ, LD_LIBRARY_PATH=str(library))
    with tempfile.TemporaryDirectory(prefix='holter_epltd_') as temp:
        # Source executable dispatches only recognized header lead names.
        # This is a transport tag, not an anatomical attribution: neither the
        # model nor the manifest receives this temporary V1 identifier.
        wfdb.wrsamp('transport', fs=int(fs), units=['mV'], sig_name=['V1'],
                    p_signal=padded[:, None], fmt=['16'], write_dir=temp)
        result = subprocess.run([str(binary), '-r', 'transport'], cwd=temp, env=environment,
                                check=True, capture_output=True, text=True, timeout=20)
        peaks = wfdb.rdann(str(Path(temp) / 'transport'), 'epltd0').sample - warmup
    return peaks[(peaks >= 0) & (peaks < len(signal))].astype(np.int64)


def process(payload):
    row, archive_path, binary, library = payload
    try:
        with zipfile.ZipFile(archive_path) as archive:
            header = archive.read(row['header_archive_member']).decode().splitlines()
            fields = header[1].split()
            if fields[1] != '16':
                raise ValueError('Unsupported waveform format')
            gain_text = fields[2]
            match = re.fullmatch(r'([\d.eE+-]+)(?:\((-?[\d.eE+-]+)\))?/(\w+)', gain_text)
            if match is None or match.group(3) != 'mV':
                raise ValueError('Unverified physical voltage units')
            gain = float(match.group(1))
            baseline = float(match.group(2)) if match.group(2) else float(fields[4])
            signal_name = str(Path(row['header_archive_member']).with_suffix('.dat'))
            raw = np.frombuffer(archive.read(signal_name), dtype='<i2').reshape(-1, row['channel_count'])
            signal = (raw[:, 0].astype(np.float64) - baseline) / gain
        if not np.isfinite(signal).all():
            raise ValueError('Nonfinite physical input')
        peaks = detect_observed(signal, row['sampling_frequency'], binary, library)
        intervals = np.diff(peaks) / row['sampling_frequency']
        status = 'available' if len(intervals) >= 60 else 'insufficient_observed_RR'
        return {'strip_id': row['strip_id'], 'selected_observed_channel': json.loads(row['channel_identity'])[0],
                'observed_peaks': len(peaks), 'observed_RR_intervals': len(intervals),
                'ArNet2_60RR_blocks': len(intervals) // 60, 'observed_RR_status': status,
                'detector_error': '', 'observed_peak_samples': json.dumps(peaks.tolist()),
                'oracle_annotations_read': False}
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        return {'strip_id': row['strip_id'], 'observed_RR_status': 'extractor_failed',
                'detector_error': repr(error), 'oracle_annotations_read': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    root = args.output
    integrity = json.loads((root / 'manifests/release_integrity.json').read_text())
    binary = root / 'author_extractor/epltd_all'
    library = root / 'author_extractor/runtime/lib'
    frame = pd.read_parquet(root / 'manifests/segment_manifest.parquet')
    payloads = [(row, integrity['archive'], binary, library) for row in frame.to_dict('records')]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(process, payloads, chunksize=16))
    observed = pd.DataFrame(rows)
    observed.to_parquet(root / 'cohorts/observed_RR_technical_availability.parquet', index=False)
    merged = frame.merge(observed, on='strip_id', validate='one_to_one')
    eligibility = []
    for row in merged.to_dict('records'):
        base_reason = row['exclusion_reason']
        for model in ['RawECGNet_30s', 'ArNet2_60RR']:
            reason = base_reason
            if not reason:
                if model == 'RawECGNet_30s' and row['duration_seconds'] < 30:
                    reason = 'insufficient_native_duration'
                elif model == 'ArNet2_60RR' and row['observed_RR_status'] != 'available':
                    reason = row['observed_RR_status']
            eligibility.append({'strip_id': row['strip_id'], 'recording_prefix': row['recording_prefix'],
                                'model_candidate': model, 'cohort': row['primary_clinical_cohort'],
                                'technical_input_available': not bool(reason), 'exclusion_reason': reason,
                                'scientific_model_admission': 'pending_separate_semantic_gate',
                                'selected_observed_channel': row.get('selected_observed_channel'),
                                'source_annotation_read_for_model_input': False})
    pd.DataFrame(eligibility).to_csv(root / 'cohorts/eligibility_by_model.csv', index=False)
    report = {'detector': 'PECG2.0.5 bundled epltd_all',
              'binary_sha256': hashlib.file_digest(binary.open('rb'), 'sha256').hexdigest(),
              'wfdb_library_sha256': hashlib.file_digest((library / 'libwfdb.so.10.7').open('rb'), 'sha256').hexdigest(),
              'physical_strips_attempted': len(frame), 'status_counts': observed.observed_RR_status.value_counts().to_dict(),
              'input_policy': 'First genuinely observed channel. Source five-second observed warm-up, discarded prefix detections. No external filter added for this utility-level technical audit. No annotation peaks supplied.',
              'limitation': 'Technical availability only. It does not certify agreement with the publication preprocessing or admit the score/decision wrapper.'}
    (root / 'reports/observed_RR_extractor_receipt.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
