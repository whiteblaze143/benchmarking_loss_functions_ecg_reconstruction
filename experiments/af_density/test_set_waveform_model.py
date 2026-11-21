import unittest
import torch
from set_waveform_model import AFSetOperator


class SetWaveformTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.response = torch.randn(4, 2, 16, 128)
        self.available = torch.tensor([[1, 1], [1, 0], [0, 1], [1, 1]], dtype=torch.bool)
        self.valid = torch.ones(4, 2, dtype=torch.bool)
        self.ids = torch.tensor([[0, 1]]*4)

    def test_original_architecture_and_binary_head_identity(self):
        model = AFSetOperator('categorical').eval()
        with torch.no_grad():
            actual = model(self.response, self.available, self.valid, self.ids)
            clean = torch.where(self.available[:, :, None, None], self.response, 0.)
            expected = model.original(None, clean, mask=~self.available, operator_ids=self.ids).squeeze(-1)
            latent = model.encode(self.response, self.available, self.valid, self.ids)
        self.assertTrue(torch.equal(actual, expected));self.assertEqual(latent.shape, (4, 256))

    def test_joint_measurement_permutation_invariance(self):
        for mode in ['categorical', 'q_ablated']:
            model = AFSetOperator(mode).eval()
            with torch.no_grad():
                original = model(self.response, self.available, self.valid, self.ids)
                permuted = model(self.response.flip(1), self.available.flip(1), self.valid.flip(1), self.ids.flip(1))
            torch.testing.assert_close(original, permuted, rtol=0, atol=1e-6)

    def test_unobserved_and_failed_response_isolation(self):
        model = AFSetOperator('categorical').eval();valid = self.valid.clone();valid[0, 1] = False
        active = self.available & valid
        changed = torch.where(active[:, :, None, None], self.response, torch.full_like(self.response, float('inf')))
        with torch.no_grad():
            original = model(self.response, self.available, valid, self.ids)
            altered = model(changed, self.available, valid, self.ids)
        self.assertTrue(torch.equal(original, altered))

    def test_q_ablation_ignores_id_and_unknown_embedding_is_zero(self):
        model = AFSetOperator('q_ablated').eval()
        with torch.no_grad():
            a = model(self.response, self.available, self.valid, self.ids)
            b = model(self.response, self.available, self.valid, torch.full_like(self.ids, -1))
        self.assertTrue(torch.equal(a, b))
        categorical = AFSetOperator('categorical')
        embedding = categorical.original.encode_operator(None, torch.full_like(self.ids, -1))
        self.assertTrue(torch.equal(embedding, torch.zeros_like(embedding)))

    def test_active_gradient_path_without_auxiliary_decoder(self):
        for mode in ['categorical', 'q_ablated']:
            model = AFSetOperator(mode)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                model(self.response, self.available, self.valid, self.ids), torch.tensor([0., 1., 0., 1.]))
            loss.backward()
            for name in ['response.input.weight', 'set_encoder.layers.0.linear1.weight', 'pool.in_proj_weight', 'head.weight']:
                gradient = dict(model.original.named_parameters())[name].grad
                self.assertIsNotNone(gradient);self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(float(gradient.abs().sum()), 0)
            self.assertTrue(all(p.grad is None for p in model.original.decoder.parameters()))

    def test_no_geometry_or_all_channel_failure_fallback(self):
        for mode in ['continuous', 'projective']:
            with self.assertRaises(ValueError): AFSetOperator(mode)
        model = AFSetOperator('categorical')
        with self.assertRaisesRegex(ValueError, 'indeterminate'):
            model(self.response, self.available, torch.zeros_like(self.valid), self.ids)


if __name__ == '__main__':
    unittest.main()
