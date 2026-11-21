import unittest
import numpy as np
from torch_phase_response import TorchPhaseResponse
from phase_representation import phase_response
from repecg.common.kernels import WhiteningTransform, NystromMap, imq_kernel


class TorchPhaseResponseTests(unittest.TestCase):
    def test_float64_torch_matches_original_numpy_map(self):
        rng = np.random.default_rng(93)
        white = WhiteningTransform.fit(rng.normal(size=(512, 2)))
        anchors = rng.normal(size=(128, 2));gram = imq_kernel(anchors, anchors)+np.eye(128)*1e-6
        values, vectors = np.linalg.eigh(gram)
        mapping = NystromMap(anchors, (vectors*values**-.5)@vectors.T, 1.)
        transform = TorchPhaseResponse(white, mapping, device='cpu')
        for count in [32, 64, 128]:
            atoms = rng.normal(size=(16, count, 2)).astype(np.float32)
            expected = phase_response(atoms, white, mapping);actual = transform(atoms)
            self.assertEqual(actual.shape, (16, 128));self.assertEqual(actual.dtype, np.float32)
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)

    def test_invalid_atoms_rejected(self):
        white = WhiteningTransform(np.zeros(2), np.eye(2), np.ones(2))
        mapping = NystromMap(np.zeros((128, 2)), np.eye(128), 1.)
        transform = TorchPhaseResponse(white, mapping, device='cpu')
        with self.assertRaises(ValueError): transform(np.zeros((16, 16, 2)))
        with self.assertRaises(ValueError): transform(np.full((16, 32, 2), np.nan))


if __name__ == '__main__':
    unittest.main()
