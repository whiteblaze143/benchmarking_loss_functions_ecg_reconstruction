"""Stage 0 Integrity Tests: Continuous SetOperator, Tensor Shapes, Masking, Permutation Invariance, Indeterminate Handling, and Finite Gradients."""
import json
import os
import sys
import unittest
from pathlib import Path
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend

# Ensure import paths
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ptbxl_distributional_repecg' / 'src'))

from set_waveform_model import ContinuousAFSetOperator, AFSetOperator
from repecg.paper07_operator.model import OperatorSetModel


class TestStage0Integrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Configure cuDNN / deterministic setup
        if torch.cuda.is_available():
            if torch.backends.cudnn.version() != 90300:
                # Use PyTorch native CUDA engines if custom cuDNN 9.3 runtime is not preloaded in current shell
                torch.backends.cudnn.enabled = False
            else:
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.B = 4
        self.response = torch.randn(self.B, 2, 16, 128, device=self.device)
        self.available = torch.tensor([[True, True], [True, False], [False, True], [True, True]], dtype=torch.bool, device=self.device)
        self.valid = torch.ones(self.B, 2, dtype=torch.bool, device=self.device)
        self.nominal_operators = torch.zeros(self.B, 2, 8, device=self.device)
        self.nominal_operators[:, 0, 0] = 1.0  # Lead I: [1, 0, 0, 0, 0, 0, 0, 0]
        self.nominal_operators[:, 1, 1] = 1.0  # Lead II: [0, 1, 0, 0, 0, 0, 0, 0]

    def test_nominal_operator_geometry(self):
        """Verify nominal Lead I and Lead II coordinates in the 8-lead clinical basis [I, II, V1, V2, V3, V4, V5, V6]."""
        model = ContinuousAFSetOperator(operator_dim=8, operator_mode='continuous').to(self.device)
        self.assertEqual(model.operator_dim, 8)
        self.assertEqual(model.nominal_operators.shape, (2, 8))
        # Lead I basis vector e_1 = [1, 0, 0, 0, 0, 0, 0, 0]
        expected_lead_i = torch.zeros(8, device=self.device); expected_lead_i[0] = 1.0
        # Lead II basis vector e_2 = [0, 1, 0, 0, 0, 0, 0, 0]
        expected_lead_ii = torch.zeros(8, device=self.device); expected_lead_ii[1] = 1.0
        self.assertTrue(torch.equal(model.nominal_operators[0], expected_lead_i))
        self.assertTrue(torch.equal(model.nominal_operators[1], expected_lead_ii))

    def test_response_tensor_shapes_and_masking(self):
        """Verify response tensor shapes (B, 2, 16, 128) and key padding masking (~active)."""
        model = ContinuousAFSetOperator(operator_dim=8, operator_mode='continuous').to(self.device).eval()
        
        # Test shape compliance
        with torch.no_grad():
            latent = model.encode(self.response, self.available, self.valid)
            logits = model(self.response, self.available, self.valid)
        self.assertEqual(latent.shape, (self.B, 256))
        self.assertEqual(logits.shape, (self.B,))

        # Test invalid tensor shape rejection
        invalid_shapes = [
            torch.randn(self.B, 1, 16, 128, device=self.device),
            torch.randn(self.B, 2, 8, 128, device=self.device),
            torch.randn(self.B, 2, 16, 64, device=self.device),
        ]
        for bad_shape in invalid_shapes:
            with self.assertRaises(ValueError):
                model.encode(bad_shape, self.available, self.valid)

        # Test that inactive channels are completely masked and values do not contaminate output
        valid_part = self.valid.clone()
        valid_part[0, 1] = False  # Make channel 1 inactive for record 0
        active = self.available & valid_part
        perturbed_response = torch.where(active[:, :, None, None], self.response, torch.full_like(self.response, 999.0))
        with torch.no_grad():
            out_orig = model(self.response, self.available, valid_part)
            out_pert = model(perturbed_response, self.available, valid_part)
        torch.testing.assert_close(out_orig, out_pert, rtol=0, atol=1e-6)

    def test_permutation_invariance(self):
        """Verify permutation invariance: permuting channel 0 and 1 with operators/masks leaves pooled latent identical."""
        model = ContinuousAFSetOperator(operator_dim=8, operator_mode='continuous').to(self.device).eval()
        with torch.no_grad():
            # Original forward
            latent_orig = model.encode(self.response, self.available, self.valid, operators=self.nominal_operators)
            logits_orig = model(self.response, self.available, self.valid, operators=self.nominal_operators)

            # Permuted forward (channel 0 <-> channel 1)
            resp_perm = self.response[:, [1, 0]]
            avail_perm = self.available[:, [1, 0]]
            valid_perm = self.valid[:, [1, 0]]
            ops_perm = self.nominal_operators[:, [1, 0]]

            latent_perm = model.encode(resp_perm, avail_perm, valid_perm, operators=ops_perm)
            logits_perm = model(resp_perm, avail_perm, valid_perm, operators=ops_perm)

        # Invariance of pooled latent and output logits
        torch.testing.assert_close(latent_orig, latent_perm, rtol=0, atol=1e-6)
        torch.testing.assert_close(logits_orig, logits_perm, rtol=0, atol=1e-6)

    def test_all_absent_channel_behavior(self):
        """Verify that all-absent channel behavior raises an explicit indeterminate exception without fabricating beats."""
        model = ContinuousAFSetOperator(operator_dim=8, operator_mode='continuous').to(self.device).eval()
        
        # When all channels are unavailable or invalid
        all_absent_available = torch.zeros(self.B, 2, dtype=torch.bool, device=self.device)
        with self.assertRaisesRegex(ValueError, 'indeterminate'):
            model(self.response, all_absent_available, self.valid)

        all_invalid = torch.zeros(self.B, 2, dtype=torch.bool, device=self.device)
        with self.assertRaisesRegex(ValueError, 'indeterminate'):
            model(self.response, self.available, all_invalid)

    def test_finite_gradients_fp32_and_bf16(self):
        """Verify that gradients are strictly finite across all active parameters in both FP32 and BF16."""
        target = torch.tensor([1.0, 0.0, 1.0, 0.0], device=self.device)
        precisions = ['fp32']
        if torch.cuda.is_available():
            precisions.append('bf16')

        for prec in precisions:
            torch.manual_seed(42)
            model = ContinuousAFSetOperator(operator_dim=8, operator_mode='continuous').to(self.device).train()
            
            if prec == 'fp32':
                logits = model(self.response, self.available, self.valid, operators=self.nominal_operators)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
                loss.backward()
            else:
                with sdpa_kernel(SDPBackend.MATH), torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(self.response, self.available, self.valid, operators=self.nominal_operators)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), target)
                loss.backward()

            # Verify active parameters have non-empty, finite gradients
            grad_checked = 0
            for name, param in model.named_parameters():
                if 'decoder' in name:
                    # Training-only reconstruction decoder is not in the AF classification forward path
                    continue
                self.assertIsNotNone(param.grad, f'Parameter {name} had None grad in {prec}')
                self.assertTrue(torch.isfinite(param.grad).all(), f'Parameter {name} had non-finite grad in {prec}')
                self.assertGreater(float(param.grad.abs().sum()), 0.0, f'Parameter {name} had zero grad in {prec}')
                grad_checked += 1

            self.assertGreaterEqual(grad_checked, 40, f'Expected at least 40 active parameters in {prec}')


def run_and_save_receipt():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestStage0Integrity)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    receipt = {
        'suite': 'TestStage0Integrity',
        'tests_run': result.testsRun,
        'errors': len(result.errors),
        'failures': len(result.failures),
        'passed': result.wasSuccessful(),
        'cuda_available': torch.cuda.is_available(),
        'cuda_device': torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu',
        'tested_capabilities': {
            'response_tensor_shape': '(B, 2, 16, 128)',
            'masking': '~active key_padding_masking with unobserved channel isolation',
            'permutation_invariance': 'Exact pooled latent invariance under channel swapping (atol <= 1e-6)',
            'all_absent_behavior': 'Raises ValueError(indeterminate) and refuses beat fabrication',
            'fp32_gradients_finite': True,
            'bf16_math_sdp_gradients_finite': True if torch.cuda.is_available() else 'skipped_cpu',
            'nominal_lead_geometry': {
                'clinical_basis': '[I, II, V1, V2, V3, V4, V5, V6]',
                'nominal_lead_i': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                'nominal_lead_ii': [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            }
        }
    }
    
    out_dir_codex = Path('/data/mithunmanivannan/codex_artifacts/af_density_continuous_setoperator')
    out_dir_local = Path(__file__).resolve().parent
    out_dir_codex.mkdir(parents=True, exist_ok=True)
    
    receipt_str = json.dumps(receipt, indent=2) + '\n'
    (out_dir_codex / 'test_stage0_integrity_receipt.json').write_text(receipt_str)
    (out_dir_local / 'test_stage0_integrity_receipt.json').write_text(receipt_str)
    print('Integrity receipt written to codex artifacts and experiments/af_density.')
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_and_save_receipt()
    sys.exit(0 if success else 1)
