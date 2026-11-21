"""Freeze common annotation-derived RR-start evaluation clocks, never features."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import wfdb
from wfdb.io.annotation import is_qrs
from metrics import discrete_pattern_from_runs

W = Path('/data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1')
BASE = Path('/data/mithunmanivannan/datasets')


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def runs(x):
    return list(zip(np.flatnonzero(x & ~np.r_[False, x[:-1]]).tolist(),
                    (np.flatnonzero(x & ~np.r_[x[1:], False])+1).tolist()))


def main(dataset):
    folder = W/('external_'+dataset+'_v1')
    contract = json.loads((folder/'input_contract.json').read_text())
    for name, digest in contract['input_hashes'].items(): assert sha(folder/name) == digest
    root = BASE/('shdb_af' if dataset == 'shdb' else 'ltafdb_1.0.0_reference')
    publisher = root/'SHA256SUMS.txt' if dataset == 'ltaf' else BASE/'iridia_af_8405941_v1.0.1/source_docs/shdb_1.0.1_SHA256SUMS.txt'
    expected = {name.lstrip('*'): digest for digest, name in (line.split(maxsplit=1) for line in publisher.read_text().splitlines())}
    references = list(csv.DictReader((folder/'reference_intervals.csv').open()))
    output = folder/'beat_clock';output.mkdir(exist_ok=False);receipts = []
    for row in csv.DictReader((folder/'file_manifest.csv').open()):
        if row['input_contract_eligible'] != 'True': continue
        rid = row['record_id'];ext = 'qrs' if dataset == 'shdb' else 'atr';source = root/(rid+'.'+ext)
        assert sha(source) == expected[source.name]
        annotation = wfdb.rdann(str(root/rid), ext, return_label_elements=['label_store'])
        codes = np.asarray(annotation.label_store)
        assert ((codes >= 0) & (codes < len(is_qrs))).all()
        q = annotation.sample[np.asarray(is_qrs)[codes]]
        assert len(q) >= 2 and (np.diff(q) >= 0).all() and q[0] >= 0 and q[-1] < int(row['source_samples'])
        duplicates = int((np.diff(q) == 0).sum());q = q[np.r_[True, np.diff(q) > 0]]
        clock = q[:-1][q[:-1] >= int(row['initial_unknown_samples'])]
        assert len(clock) > 0 and (np.diff(clock) > 0).all()
        reference = np.zeros(len(clock), bool);episode_observations = []
        for span in references:
            if span['file_index'] != row['file_index'] or span['rhythm'] != '(AFIB': continue
            hit = (clock >= int(span['start_sample'])) & (clock < int(span['end_sample_exclusive']))
            reference |= hit;episode_observations.append(int(hit.sum()))
        path = output/f"file_{int(row['file_index']):03d}.npz"
        np.savez_compressed(path, native_RR_start_samples=clock, reference_af=reference)
        receipt = {'record_id': rid, 'file_index': int(row['file_index']), 'clock_observations': len(clock),
            'exact_duplicate_QRS_events_removed': duplicates, 'annotation_sha256': sha(source),
            'clock_sha256': sha(path), 'reference_pattern_metrics': discrete_pattern_from_runs(runs(reference), len(clock)),
            'source_AF_spans_without_clock_observations': sum(n == 0 for n in episode_observations)}
        receipts.append(receipt);print(rid, len(clock), 'evaluation clock committed', flush=True)
    result = {'complete': True, 'dataset': dataset, 'records': len(receipts), 'files': receipts,
        'input_contract_sha256': sha(folder/'input_contract.json'), 'source_sha256': sha(__file__),
        'clock_policy': 'Use source annotations classified as QRS by pinned installed WFDB is_qrs table; remove/log exact duplicate QRS timestamps, reject decreases. Use RR-start sample q[:-1], retain positive intervals of all lengths; restrict to known annotation suffix. Both reference AF and every future model prediction are projected onto these identical native coordinates.',
        'QRS_label_table': list(is_qrs),
        'limitations': 'Evaluation clock only, never an observed-RR classifier feature or R-peak source for Phase-KME. Beat-indexed metrics do not replace elapsed time; no claim of new clinical beat adjudication. Last QRS has no complete following annotated RR interval and is not an RR-start observation.',
        'diagnostic_predictions_generated': False}
    assert len(receipts) == contract['eligible_records']
    (output/'receipt.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('dataset', choices=['shdb', 'ltaf'])
    main(parser.parse_args().dataset)
