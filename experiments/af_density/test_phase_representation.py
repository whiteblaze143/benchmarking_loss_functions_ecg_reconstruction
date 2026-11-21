import unittest
import numpy as np
from scipy.interpolate import PchipInterpolator
from phase_representation import phase_atoms, phase_response
from repecg.common.kernels import WhiteningTransform, NystromMap, imq_kernel


class PhaseRepresentationTests(unittest.TestCase):
    def test_original_phase_samples_and_grouping(self):
        x = np.sin(np.arange(6000)*.03).astype(np.float32)
        peaks = np.array([10, 70, 471, 671, 1071, 1120])
        atoms = phase_atoms(x, peaks)
        # Accepted intervals are [10,70), [471,671), [671,1071).
        manual = []
        for a, b in [(10, 70), (471, 671), (671, 1071)]:
            y = PchipInterpolator(np.arange(b-a)/(b-a), x[a:b])(
                np.arange(256)/256).astype(np.float32)
            manual.append(np.column_stack((y, np.gradient(y))))
        manual = np.stack(manual)
        for cell in range(16):
            np.testing.assert_array_equal(atoms[cell], manual[:, cell*16:(cell+1)*16].reshape(-1, 2))
        self.assertEqual(atoms.shape, (16, 48, 2))

    def test_polarity_and_no_unit_conversion(self):
        x = np.linspace(-2, 2, 6000, dtype=np.float32)
        peaks = np.array([100, 300, 500])
        atoms = phase_atoms(x, peaks)
        np.testing.assert_array_equal(phase_atoms(-x, peaks), -atoms)
        self.assertEqual(atoms[0, 0, 0], x[100])

    def test_no_representation_for_insufficient_intervals(self):
        x = np.ones(6000, dtype=np.float32)
        for peaks in [[], [1], [1, 201], [1, 502, 1003]]:
            self.assertIsNone(phase_atoms(x, np.array(peaks, dtype=np.int32)))

    def test_invalid_input_fails(self):
        x = np.ones(6000, dtype=np.float32)
        for peaks in [[1., 201., 401.], [1, 201, 201], [-1, 201, 401], [1, 201, 6000]]:
            with self.assertRaises(ValueError):
                phase_atoms(x, peaks)
        with self.assertRaises(ValueError):
            phase_atoms(np.zeros((6000, 2)), [1, 201, 401])

    def test_response_matches_explicit_kernel_feature_average(self):
        rng = np.random.default_rng(123)
        atoms = rng.normal(size=(16, 32, 2)).astype(np.float32)
        white = WhiteningTransform.fit(rng.normal(size=(1000, 2)))
        # Fixed synthetic transform fixture, not data-fitted model landmarks.
        mapping = NystromMap(rng.normal(size=(128, 2)), np.eye(128), 1.0)
        response = phase_response(atoms, white, mapping)
        expected = np.stack([imq_kernel(white.transform(cell), mapping.landmarks).mean(axis=0)
                             for cell in atoms]).astype(np.float32)
        np.testing.assert_array_equal(response, expected)
        self.assertEqual(response.dtype, np.float32)


if __name__ == '__main__':
    unittest.main()
