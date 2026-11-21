import unittest
import numpy as np
from beat_clock_evaluation import released_rr_starts, evaluate_shared_beat_clock
from metrics import discrete_pattern_reference


class SharedClockTests(unittest.TestCase):
    def test_zero_intervals_do_not_shift_elapsed_clock(self):
        clock, excluded = released_rr_starts([1000, 0, 500, 1000], 200)
        np.testing.assert_array_equal(clock, [0, 200, 300])
        self.assertEqual(excluded, 1)
        with self.assertRaises(ValueError): released_rr_starts([1000, -1], 200)

    def test_shared_projection_matches_literal_published_equations(self):
        result = evaluate_shared_beat_clock([(200, 400), (600, 700)], [0, 1, 0, 1],
            [0, 200, 400, 600], [200, 400, 600, 800], 200, [1000]*8)
        self.assertEqual(result['reference'], discrete_pattern_reference([0, 1, 0, 1]))
        self.assertEqual(result['absolute_errors'], {'density': 0., 'aggregation': 0.})

    def test_unknown_time_without_clock_point_still_indeterminate(self):
        result = evaluate_shared_beat_clock([(200, 400)], [0, -1, 1],
            [0, 100, 150], [100, 150, 800], 200, [1000]*8)
        self.assertEqual(result['status'], 'indeterminate_unknown_physical_time')
        self.assertIsNone(result['predicted'])


if __name__ == '__main__': unittest.main()
