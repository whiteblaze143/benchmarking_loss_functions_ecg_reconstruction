"""Reuse frozen descriptive summaries with explicit corrected-family identities."""
import argparse
import hashlib
import json
from pathlib import Path
import report_validation_calibration_v2 as calibration
import summarize_validation_timelines as timeline

W = Path('/data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1')


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(mode):
    label = 'math_v2_'+mode
    root = W/'set_models_math_sdp_v2'/mode
    replay = json.loads((root/'independent_validation_v1/receipt.json').read_text())
    assert replay['all_pass'] and not replay['test_access']
    # Add a new explicit identity; never overwrite a v1 model mapping/result.
    assert label not in calibration.MODELS
    calibration.MODELS[label] = 'set_models_math_sdp_v2/'+mode
    calibration.main(label)
    timeline.main(label)
    receipt = {'complete': True, 'model': label, 'architecture_mode': mode,
        'source_sha256': sha(__file__), 'test_inference': False,
        'family_protocol_sha256': sha(W/'set_models_math_sdp_v2/protocol.json'),
        'calibration_source_sha256': sha(Path(calibration.__file__)),
        'timeline_summary_source_sha256': sha(Path(timeline.__file__)),
        'outputs': {name: sha(W/'evaluation_protocol_v1'/name) for name in
                    [f'validation_calibration_{label}_v2.json', f'validation_summary_{label}.json']},
        'scope': 'Descriptive validation summaries only; no model/decoder selection, calibration fitting, held-out inference or mixing of v1/v2 models.'}
    path = root/'descriptive_summary_receipt.json';assert not path.exists()
    path.write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--mode', choices=['categorical', 'q_ablated', 'hybrid'], required=True)
    main(parser.parse_args().mode)
