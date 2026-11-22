"""Verify versioned baseline release guards with synthetic files only."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import predict_raw_waveform_v3 as raw
import predict_observed_rr_v3 as rr


class ReleaseTests(unittest.TestCase):
    def exercise(self, reader, name, version):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);model_root = root/'model';model_root.mkdir()
            def write(path, value):
                path = root/path;path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value));return path
            write('evaluation_protocol_v1/protocol.json', {})
            write('input_contract.json', {})
            write('model/evaluation_lock.json', {})
            with patch.object(reader, 'W', root), patch.object(reader, 'R', model_root):
                with self.assertRaises(FileNotFoundError): reader.require_test_release(name)
                locks = {name: reader.sha(model_root/'evaluation_lock.json')}
                sources = {name: reader.sha(reader.__file__)}
                approval = write('approval.json', {'authorize_test_inference': True, 'scope': 'locked_internal_benchmark',
                    'reviewed_model_lock_hashes': locks, 'reviewed_predictor_source_hashes': sources,
                    'reviewed_input_contract_sha256': reader.sha(root/'input_contract.json')})
                parity = write(f'model/inference_validation_{version}/receipt.json', {'complete': True,
                    'validation_bitwise_replay': True, 'source_sha256': reader.sha(reader.__file__),
                    'model_evaluation_lock_sha256': locks[name]})
                release = {'released': True, 'authorized_models': [name],
                    'evaluation_protocol_sha256': reader.sha(root/'evaluation_protocol_v1/protocol.json'),
                    'input_contract_sha256': reader.sha(root/'input_contract.json'),
                    'evaluation_lock_hashes': locks, 'predictor_source_hashes': sources,
                    'independent_approval_receipt': {'path': str(approval), 'sha256': reader.sha(approval)},
                    'validation_parity_receipt_hashes': {name: reader.sha(parity)}}
                write('evaluation_protocol_v1/heldout_release.json', release)
                reader.require_test_release(name)
                self.assertFalse((model_root/f'inference_test_{version}').exists())
                approval.write_text('{}')
                with self.assertRaises(AssertionError): reader.require_test_release(name)
                write('approval.json', {'authorize_test_inference': True, 'scope': 'locked_internal_benchmark',
                    'reviewed_model_lock_hashes': locks, 'reviewed_predictor_source_hashes': sources,
                    'reviewed_input_contract_sha256': reader.sha(root/'input_contract.json')})
                parity.write_text('{}')
                with self.assertRaises(AssertionError): reader.require_test_release(name)
                parity.write_text(json.dumps({'complete': True, 'validation_bitwise_replay': False,
                    'source_sha256': reader.sha(reader.__file__), 'model_evaluation_lock_sha256': locks[name]}))
                release['validation_parity_receipt_hashes'][name] = reader.sha(parity)
                write('evaluation_protocol_v1/heldout_release.json', release)
                with self.assertRaises(AssertionError): reader.require_test_release(name)

    def test_raw_guard(self): self.exercise(raw, 'raw', 'v3')
    def test_rr_guard(self): self.exercise(rr, 'rr', 'v3')


if __name__ == '__main__': unittest.main()
