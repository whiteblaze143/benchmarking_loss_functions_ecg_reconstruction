import unittest
import numpy as np
import torch
from hybrid_set_model import AFSetRR, fit_rr_scaling, rr_side_input


class HybridSetTests(unittest.TestCase):
    def test_training_constants_and_unobserved_rr_isolation(self):
        rr = np.ones((8, 2, 16), np.float32);valid = np.ones((8, 2), bool)
        scale = fit_rr_scaling(rr, valid)
        self.assertTrue(scale['constant_feature'].all())
        np.testing.assert_array_equal(scale['sd'], np.ones((2, 16)))
        x = torch.tensor(rr);availability = torch.tensor([[True, False]]*8)
        mean, sd = torch.tensor(scale['mean'], dtype=torch.float32), torch.tensor(scale['sd'], dtype=torch.float32)
        a, _ = rr_side_input(x, torch.tensor(valid), availability, mean, sd)
        x[:, 1] = torch.inf
        b, _ = rr_side_input(x, torch.tensor(valid), availability, mean, sd)
        self.assertTrue(torch.equal(a, b))

    def test_phase_only_rr_only_and_both_have_explicit_paths(self):
        torch.manual_seed(42);model = AFSetRR()
        response = torch.randn(3, 2, 16, 128)
        available = torch.ones(3, 2, dtype=torch.bool)
        phase_valid = torch.tensor([[1, 1], [0, 0], [1, 1]], dtype=torch.bool)
        rr_present = torch.tensor([False, True, True])
        rr = torch.zeros(3, 36);rr[1:] = torch.randn(2, 36)
        ids = torch.tensor([[0, 1]]*3)
        result = model(response, available, phase_valid, ids, rr, rr_present, available.float())
        self.assertEqual(result.shape, (3,));result.sum().backward()
        self.assertIsNotNone(model.phase.original.response.input.weight.grad)
        self.assertIsNotNone(model.head.weight.grad)
        self.assertTrue(all(p.grad is None for p in model.phase.original.head.parameters()))
        self.assertTrue(all(p.grad is None for p in model.phase.original.decoder.parameters()))

    def test_no_double_modality_failure_prediction(self):
        model = AFSetRR();available = torch.ones(1, 2, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, 'indeterminate'):
            model(torch.full((1, 2, 16, 128), torch.nan), available, torch.zeros_like(available),
                  torch.tensor([[0, 1]]), torch.zeros(1, 36), torch.zeros(1, dtype=torch.bool), torch.zeros(1, 2))


if __name__ == '__main__':
    unittest.main()
