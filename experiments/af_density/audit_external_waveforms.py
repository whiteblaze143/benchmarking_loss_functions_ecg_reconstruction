"""Publisher hashes and full digital-signal audit for format-16 Holter files."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import wfdb

ROOTS = {'shdb': Path('/data/mithunmanivannan/datasets/shdb_af'),
         'ltaf': Path('/data/mithunmanivannan/datasets/ltafdb_1.0.0_reference')}


def main(dataset):
    root = ROOTS[dataset]
    manifest = (root/'SHA256SUMS.txt' if dataset == 'ltaf' else
        Path('/data/mithunmanivannan/datasets/iridia_af_8405941_v1.0.1/source_docs/shdb_1.0.1_SHA256SUMS.txt'))
    expected = {name.lstrip('*'): digest for digest, name in
                (line.split(maxsplit=1) for line in manifest.read_text().splitlines())}
    headers = sorted(root.glob('*.hea'));assert len(headers) == (128 if dataset == 'shdb' else 84)
    result = []
    for header in headers:
        assert hashlib.sha256(header.read_bytes()).hexdigest() == expected[header.name]
        h = wfdb.rdheader(str(header.with_suffix('')))
        assert h.n_sig == 2 and h.fmt == ['16', '16'] and len(set(h.file_name)) == 1
        assert all(not x for x in h.byte_offset) and all(not x for x in h.skew)
        assert h.units == ['mV', 'mV'] and np.isfinite(h.adc_gain).all() and min(h.adc_gain) > 0
        path = root/h.file_name[0]
        assert path.stat().st_size == h.sig_len*4
        digest = hashlib.sha256();sums = np.zeros(2, np.int64);missing = np.zeros(2, np.int64)
        low = np.full(2, np.inf);high = np.full(2, -np.inf);count = 0;initial = None
        with path.open('rb') as stream:
            while chunk := stream.read(4*1024*1024):
                digest.update(chunk);x = np.frombuffer(chunk, dtype='<i2').reshape(-1, 2)
                if initial is None: initial = x[0].tolist()
                sums += x.sum(axis=0, dtype=np.int64);missing += (x == -32768).sum(axis=0)
                for c in range(2):
                    good = x[:, c] != -32768
                    if good.any():
                        physical = (x[good, c].astype(np.float64)-h.baseline[c])/h.adc_gain[c]
                        assert np.isfinite(physical).all()
                        low[c] = min(low[c], physical.min());high[c] = max(high[c], physical.max())
                count += len(x)
        assert digest.hexdigest() == expected[path.name] and count == h.sig_len
        checksum_matches = [int(s % 65536) == int(v % 65536) for s, v in zip(sums, h.checksum)]
        row = {'record': header.stem, 'samples': count, 'sampling_rate': h.fs,
               'signal_sha256': digest.hexdigest(), 'header_checksum_matches': checksum_matches,
               'initial_value_matches': initial == h.init_value,
               'encoded_missing_samples': missing.tolist(),
               'physical_min_mV': [float(v) if np.isfinite(v) else None for v in low],
               'physical_max_mV': [float(v) if np.isfinite(v) else None for v in high]}
        result.append(row);print(json.dumps(row), flush=True)
    receipt = {'complete': True, 'dataset': dataset, 'records': len(result), 'publisher_hashes_match': True,
               'all_header_checksums_match': all(all(r['header_checksum_matches']) for r in result),
               'all_initial_values_match': all(r['initial_value_matches'] for r in result), 'files': result,
               'limitations': 'Numeric and release integrity only, not clinical signal quality. Encoded missing samples remain missing; no model inference, channel geometry inference or label modification.'}
    (root/'waveform_integrity_audit.json').write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('dataset', choices=ROOTS)
    main(parser.parse_args().dataset)
