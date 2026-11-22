import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import predict_set_math_waveform_v1 as reader


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.mock_root = patch.object(reader, 'W', self.root);self.mock_root.start();self.addCleanup(self.mock_root.stop)
        self.key = 'math_v2_categorical'
        for name in ['evaluation_protocol_v1/protocol.json', 'input_contract.json',
                     'set_models_math_sdp_v2/protocol.json', 'set_models_math_sdp_v2/categorical/evaluation_lock.json']:
            self.write(name, {})

    def write(self, name, value):
        p = self.root/name;p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value));return p

    def release(self):
        lock = reader.sha(self.root/'set_models_math_sdp_v2/categorical/evaluation_lock.json')
        models = {self.key: lock};sources = {self.key: reader.sha(reader.__file__)}
        approval = self.write('approval.json', {'authorize_test_inference': True,
            'scope': 'locked_internal_benchmark', 'reviewed_model_lock_hashes': models,
            'reviewed_predictor_source_hashes': sources,
            'reviewed_input_contract_sha256': reader.sha(self.root/'input_contract.json')})
        parity = self.write('set_models_math_sdp_v2/categorical/inference_validation_v1/receipt.json',
            {'complete': True, 'validation_bitwise_replay': True, 'source_sha256': reader.sha(reader.__file__),
             'model_evaluation_lock_sha256': lock})
        r = {'released': True, 'authorized_models': [self.key],
            'evaluation_protocol_sha256': reader.sha(self.root/'evaluation_protocol_v1/protocol.json'),
            'input_contract_sha256': reader.sha(self.root/'input_contract.json'),
            'corrected_family_protocol_sha256': reader.sha(self.root/'set_models_math_sdp_v2/protocol.json'),
            'evaluation_lock_hashes': models, 'predictor_source_hashes': sources,
            'independent_approval_receipt': {'path': str(approval), 'sha256': reader.sha(approval)},
            'validation_parity_receipt_hashes': {self.key: reader.sha(parity)}}
        self.write('evaluation_protocol_v1/heldout_release.json', r);return r

    def test_absent_release_blocks_main_before_inputs_or_output_creation(self):
        with self.assertRaises(FileNotFoundError): reader.main('categorical', 'test')
        self.assertFalse((self.root/'set_models_math_sdp_v2/categorical/inference_test_v1').exists())

    def test_matching_source_model_approval_and_parity_pass_gate_only(self):
        self.release();reader.require_test_release('categorical')
        self.assertFalse((self.root/'set_models_math_sdp_v2/categorical/inference_test_v1').exists())

    def test_changed_prediction_source_binding_is_rejected(self):
        r = self.release();r['predictor_source_hashes'][self.key] = 'wrong'
        self.write('evaluation_protocol_v1/heldout_release.json', r)
        with self.assertRaises(AssertionError): reader.require_test_release('categorical')

    def test_changed_checkpoint_or_input_contract_is_rejected(self):
        self.release();self.write('set_models_math_sdp_v2/categorical/evaluation_lock.json', {'changed': True})
        with self.assertRaises(AssertionError): reader.require_test_release('categorical')
        self.release();self.write('input_contract.json', {'changed': True})
        with self.assertRaises(AssertionError): reader.require_test_release('categorical')

    def test_modified_approval_and_failed_parity_are_rejected(self):
        self.release();self.write('approval.json', {'authorize_test_inference': False})
        with self.assertRaises(AssertionError): reader.require_test_release('categorical')
        r = self.release();p = self.root/'set_models_math_sdp_v2/categorical/inference_validation_v1/receipt.json'
        data = json.loads(p.read_text());data['validation_bitwise_replay'] = False;p.write_text(json.dumps(data))
        r['validation_parity_receipt_hashes'][self.key] = reader.sha(p)
        self.write('evaluation_protocol_v1/heldout_release.json', r)
        with self.assertRaises(AssertionError): reader.require_test_release('categorical')


if __name__ == '__main__': unittest.main()
