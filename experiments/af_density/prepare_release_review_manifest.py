"""Snapshot evidence for independent review; cannot authorize or release inference."""
import datetime
import hashlib
import json
from pathlib import Path

W = Path('/data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1')
R = Path('/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction')


def record(p):
    assert p.is_file(), p
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def main():
    assert not (W/'evaluation_protocol_v1/heldout_release.json').exists()
    specifications = {
        'raw': ('raw_mask_resnet_v1', 'predict_raw_waveform.py', 'inference_validation_v1'),
        'rr': ('observed_rr_hgb_v1', 'predict_observed_rr_v2.py', 'inference_validation_v2'),
        'math_v2_categorical': ('set_models_math_sdp_v2/categorical', 'predict_set_math_waveform_v1.py', 'inference_validation_v1'),
        'math_v2_q_ablated': ('set_models_math_sdp_v2/q_ablated', 'predict_set_math_waveform_v1.py', 'inference_validation_v1'),
        'math_v2_hybrid': ('set_models_math_sdp_v2/hybrid', 'predict_set_math_waveform_v1.py', 'inference_validation_v1'),
    }
    models = {}
    for name, (folder, predictor, parity_folder) in specifications.items():
        root = W/folder;lock = json.loads((root/'evaluation_lock.json').read_text())
        parity = json.loads((root/parity_folder/'receipt.json').read_text())
        assert lock['complete_train_validation'] and not lock['test_inference_performed']
        assert parity['complete'] and parity['validation_bitwise_replay'] and not parity['test_inference_performed']
        assert parity['windows'] == 166047
        models[name] = {'model_lock': record(root/'evaluation_lock.json'),
                        'predictor': record(R/'experiments/af_density'/predictor),
                        'validation_parity_receipt': record(root/parity_folder/'receipt.json'),
                        'threshold': lock['threshold']}
    files = ['evaluation_protocol_v1/protocol.json', 'evaluation_protocol_v1/freeze.json',
             'evaluation_protocol_v1/decision_hierarchy_proposal_v1.json', 'input_contract.json',
             'set_models_math_sdp_v2/protocol.json',
             'pre_release_corrective_v1/release_readiness_v14.json',
             'pre_release_corrective_v1/validation_coverage_corrected_family_v1/summary.json',
             'pre_release_corrective_v1/paired_math_v2_categorical_hybrid_q_ablated_validation_v1/independent_aggregation_check.json']
    result = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'kind': 'DRAFT_REVIEW_EVIDENCE_NOT_RELEASE', 'released': False,
              'authorize_test_inference': False, 'reviewer_approval': None, 'test_inference_performed': False,
              'models': models, 'contracts_and_evidence': [record(W/f) for f in files],
              'reader_gate_findings': [
                  'Corrected SetOperator reader requires release, model and source hashes, input contract, approval receipt, and model-specific production parity receipt; receipt is a local file, not authenticated reviewer identity.',
                  'Raw reader checks release, protocol and model lock, and validation source parity, but does not itself require the independent approval receipt.',
                  'RR v2 reader checks release, protocol and model lock, but does not itself require the independent approval receipt or validation parity receipt at test entry.',
                  'There is currently no heldout_release.json. The present hold is effective for the supplied readers. A source-bound draft alone does not fix unequal enforcement, establish independent approval or grant authorization.'
              ],
              'remaining': ['Independent scientific and release-custody review.',
                            'Resolve baseline versus Set reader gate differences with versioned readers and fresh validation parity if changed.',
                            'Resolve missingness and claim hierarchy without using held-out outcomes.',
                            'Keep historical test exposure and absence of an approved clinical preservation margin explicit.'],
              'source_sha256': record(Path(__file__))['sha256']}
    out = W/'pre_release_corrective_v1/release_review_evidence_draft_v1.json'
    assert not out.exists();out.write_text(json.dumps(result, indent=2)+'\n');print(out)


if __name__ == '__main__': main()
