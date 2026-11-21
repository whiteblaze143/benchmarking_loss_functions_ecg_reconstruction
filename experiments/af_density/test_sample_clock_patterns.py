import unittest
import numpy as np
from sample_clock_patterns import evaluate_sample_clock, compress_sample_clock
from metrics import discrete_pattern_reference


class SampleClockTests(unittest.TestCase):
    def test_half_open_assignment_boundaries_and_observation_endpoint(self):
        r = evaluate_sample_clock([(4, 8)], [0, 1, 0], [0, 4, 8], [4, 8, 12], 1, [0, 2, 4, 6, 8, 10, 12])
        self.assertEqual(r['clock_observations'], 6)
        self.assertEqual(r['reference'], discrete_pattern_reference([0, 0, 1, 1, 0, 0]))
        self.assertEqual(r['absolute_errors'], {'density': 0., 'aggregation': 0.})

    def test_compression_removes_gap_clock_points_and_transforms_coordinates(self):
        np.testing.assert_array_equal(compress_sample_clock([0, 3, 6, 9, 12], [(0, 4, 0), (8, 14, 4)]), [0, 3, 5, 8])
        r = evaluate_sample_clock([(0, 3)], [0, -1, 1], [0, 4, 8], [4, 8, 12], 1, [1, 9])
        self.assertEqual(r['status'], 'indeterminate_unknown_physical_time')
        self.assertIsNone(r['predicted'])


if __name__ == '__main__': unittest.main()
