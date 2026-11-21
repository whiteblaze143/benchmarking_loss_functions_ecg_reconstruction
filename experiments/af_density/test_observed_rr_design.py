import unittest
import numpy as np
from observed_rr_design import rr_design


class ObservedRRDesignTests(unittest.TestCase):
    def test_unobserved_channel_cannot_change_design(self):
        rr = np.ones((4, 2, 16), np.float32);valid = np.ones((4, 2), bool)
        first, usable = rr_design(rr, valid, [True, False])
        rr[:, 1] = np.inf
        second, _ = rr_design(rr, valid, [True, False])
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (4, 36))
        self.assertTrue(usable.all())

    def test_failed_requested_channel_remains_indeterminate(self):
        rr = np.ones((2, 2, 16), np.float32);valid = np.array([[False, True], [False, False]])
        x, usable = rr_design(rr, valid, [True, False])
        self.assertFalse(usable.any());self.assertTrue(np.isnan(x[:, :32]).all())
        x, usable = rr_design(rr, valid, [True, True])
        np.testing.assert_array_equal(usable, [True, False])
        np.testing.assert_array_equal(x[:, -2:], [[0, 1], [0, 0]])

    def test_float64_cache_rejected(self):
        with self.assertRaises(ValueError):
            rr_design(np.ones((2, 2, 16), np.float64), np.ones((2, 2), bool), [True, True])


if __name__ == '__main__':
    unittest.main()
