import unittest
import numpy as np
from batched_phase_response import batched_phase_response
from torch_phase_response import TorchPhaseResponse
from phase_representation import phase_response
from repecg.common.kernels import WhiteningTransform, NystromMap, imq_kernel


class BatchedPhaseTests(unittest.TestCase):
    def test_variable_beat_padding_contributes_no_feature_mass(self):
        rng = np.random.default_rng(62)
        white = WhiteningTransform.fit(rng.normal(size=(512, 2)))
        anchors = rng.normal(size=(128, 2));gram = imq_kernel(anchors, anchors)+np.eye(128)*1e-6
        values, vectors = np.linalg.eigh(gram)
        mapping = NystromMap(anchors, (vectors*values**-.5)@vectors.T, 1.)
        transform = TorchPhaseResponse(white, mapping, device='cpu')
        atoms = [rng.normal(size=(16, n, 2)).astype(np.float32) for n in [32, 64, 128]]
        actual = batched_phase_response(transform, atoms)
        for i, a in enumerate(atoms):
            np.testing.assert_allclose(actual[i], phase_response(a, white, mapping), rtol=0, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
