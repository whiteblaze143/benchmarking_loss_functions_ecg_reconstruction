import unittest
import numpy as np
from fit_phase_kernel import linear_mean_response
from phase_representation import phase_response
from repecg.common.kernels import WhiteningTransform, NystromMap, imq_kernel


class PhaseKernelTests(unittest.TestCase):
    def test_mean_order_equivalence_with_nontrivial_inverse_root(self):
        rng = np.random.default_rng(518)
        white = WhiteningTransform.fit(rng.normal(size=(1000, 2)))
        anchors = rng.normal(size=(128, 2))
        gram = imq_kernel(anchors, anchors)+np.eye(128)*1e-6
        values, vectors = np.linalg.eigh(gram)
        inverse = (vectors*values**-.5)@vectors.T
        mapping = NystromMap(anchors, inverse, 1.)
        for count in [32, 80, 512]:
            atoms = rng.normal(size=(16, count, 2)).astype(np.float32)
            reference = phase_response(atoms, white, mapping)
            optimized = linear_mean_response(atoms, white, mapping)
            np.testing.assert_allclose(optimized, reference, rtol=0, atol=1e-7)

    def test_nonfinite_atoms_fail(self):
        white = WhiteningTransform(np.zeros(2), np.eye(2), np.ones(2))
        mapping = NystromMap(np.zeros((128, 2)), np.eye(128), 1.)
        with self.assertRaises(ValueError):
            linear_mean_response(np.full((16, 32, 2), np.nan), white, mapping)


if __name__ == '__main__':
    unittest.main()
