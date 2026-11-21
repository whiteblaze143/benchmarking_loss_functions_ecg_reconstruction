import unittest
import numpy as np
from temporal_baselines import threshold_states, filter_and_merge, timeline_intervals


class TemporalBaselineTests(unittest.TestCase):
    def test_threshold_ties_and_unknown(self):
        np.testing.assert_array_equal(threshold_states([0, .49, .5, 1, np.nan], .5), [0, 0, 1, 1, -1])
        for values in [[np.inf], [-.1], [1.1], []]:
            with self.assertRaises(ValueError):
                threshold_states(values, .5)

    def test_median_edges_and_isolated_errors(self):
        x = [0, 0, 1, 0, 0, 1, 1, 1]
        y = filter_and_merge(x, [30]*8, median_width=3, max_gap_seconds=0)
        np.testing.assert_array_equal(y, [0, 0, 0, 0, 0, 1, 1, 1])
        np.testing.assert_array_equal(filter_and_merge([1], [2], median_width=5, max_gap_seconds=0), [1])

    def test_gap_merging_uses_actual_elapsed_time(self):
        x = [0, 1, 0, 0, 1, 0]
        durations = [30, 30, 30, 1, 30, 30]
        np.testing.assert_array_equal(filter_and_merge(x, durations, median_width=1, max_gap_seconds=31),
                                      [0, 1, 1, 1, 1, 0])
        np.testing.assert_array_equal(filter_and_merge(x, durations, median_width=1, max_gap_seconds=30), x)

    def test_unknown_never_bridged_by_filter_or_merge(self):
        x = [1, 0, -1, 0, 1]
        np.testing.assert_array_equal(filter_and_merge(x, [30]*5, median_width=1, max_gap_seconds=1000), x)
        y = filter_and_merge([0, 0, -1, 1, 1], [30]*5, median_width=9, max_gap_seconds=1000)
        np.testing.assert_array_equal(y, [0, 0, -1, 1, 1])
        np.testing.assert_array_equal(filter_and_merge([-1, -1], [1, 1], median_width=3, max_gap_seconds=60), [-1, -1])

    def test_timeline_tail_and_original_clock(self):
        r = timeline_intervals([1, 1, -1, 0, 1], [6000, 12000, 18000, 24000, 30000],
                               [12000, 18000, 24000, 30000, 30040], 200)
        self.assertEqual(r['af'], [(30, 90), (150, 150.2)])
        self.assertEqual(r['unknown'], [(90, 120)])
        self.assertEqual(r['start_seconds'], 30)
        self.assertEqual(r['end_seconds'], 150.2)

    def test_invalid_state_duration_and_clock_fail(self):
        for x, duration in [([2], [30]), ([1.], [30]), ([1], [0]), ([1], [np.nan]), ([1], [])]:
            with self.assertRaises(ValueError):
                filter_and_merge(x, duration, median_width=3, max_gap_seconds=0)
        for width in [0, 2, True]:
            with self.assertRaises(ValueError):
                filter_and_merge([1], [30], median_width=width, max_gap_seconds=0)
        for starts, ends in [([0, 20], [10, 30]), ([0, 9], [10, 20]), ([0., 10.], [10., 20.])]:
            with self.assertRaises(ValueError):
                timeline_intervals([0, 1], starts, ends, 200)


if __name__ == '__main__':
    unittest.main()
