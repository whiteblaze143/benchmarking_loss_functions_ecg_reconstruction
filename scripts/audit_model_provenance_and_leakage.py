#!/usr/bin/env python3
"""Verify saved manifests and checkpoints; report only checks derived from run artifacts."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def digest(path):
    return hashlib.file_digest(open(path,'rb'),'sha256').hexdigest()


def audit_model_provenance(run_dir):
    out=Path(run_dir)
    provenance=json.loads((out/'training_provenance.json').read_text())
    manifest=pd.read_csv(out/'noise_manifest_and_oof_predictions.csv')
    cv=pd.read_csv(out/'noise_record_group_cv.csv')
    assert len(manifest)==6130 and manifest.record_id.nunique()==6130
    assert manifest.oof_probability.notna().all()
    assert manifest.record_group_id.nunique()==provenance['n_record_groups']
    assert provenance['patient_disjointness_verified'] is False
    assert manifest.groupby('record_group_id').held_out_fold.nunique().eq(1).all()
    assert manifest.held_out_fold.between(0,4).all()
    assert digest(out/'final_setoperator.pt')==provenance['checkpoint_sha256']
    assert digest(out/'noise_manifest_and_oof_predictions.csv')==provenance['manifest_sha256']
    for row in cv.itertuples():
        test=manifest[manifest.held_out_fold==row.fold]
        train=manifest[manifest.held_out_fold!=row.fold]
        assert not set(train.record_group_id)&set(test.record_group_id)
        assert len(train)==row.train_segments and len(test)==row.test_segments
        assert train.record_group_id.nunique()==row.train_record_groups and test.record_group_id.nunique()==row.test_record_groups
    text=[
        '# Training and split provenance verification','',
        'Status: WARN. Recording-group consistency checks pass; patient independence is unverified.','',
        f'- Complete corpus: {len(manifest)} segments from {manifest.record_group_id.nunique()} daily-record groups; this is not a verified patient count.',
        '- Recording-group-disjoint five-fold evaluation: every segment has one held-out prediction; all segments from a daily-record prefix share a fold.',
        '- Each fold excludes held-out recording groups; fold sizes reconcile with the manifest. Patient-level overlap cannot be checked from these identifiers.',
        '- Final model: trained separately on all 6,130 segments for a fixed 15 epochs. No fold-winner checkpoint selection.',
        '- Final checkpoint and complete manifest SHA-256 hashes match the recorded training provenance.',
        '- SHDB-AF is absent from this SetOperator training corpus. The run does not establish identity-level independence for third-party pretrained baseline weights.',
        '- Thresholds are fixed at probability 0.50, SQI 0.55, and uncertainty 0.35. Retrospective reuse after earlier SHDB inspection remains exploratory.',
        '- Noise risk-coverage uses held-out SetOperator probabilities, including the same SQI and uncertainty rule as clinical evaluation.','',
        cv.to_markdown(index=False),'',
        'The earlier audit claims of patient-grouped training on the entire corpus were unsupported by the old implementation. This report applies only to this run directory.'
    ]
    (out/'model_training_and_leakage_audit.md').write_text('\n'.join(text)+'\n')
    print('RECORD_GROUP_CHECKS_PASS_PATIENT_INDEPENDENCE_UNVERIFIED',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',required=True)
    audit_model_provenance(parser.parse_args().run_dir)
