import unittest
import numpy as np
from elapsed_gap_strategies import gap_adjusted_timeline


class ElapsedGapTests(unittest.TestCase):
    def test_context_midpoint_uses_elapsed_samples_and_splits_assignment(self):
        result = gap_adjusted_timeline([(10, 30)], [0, -1, -1, 1], [0, 10, 20, 21], [10, 20, 21, 31], 1, 'context')
        self.assertEqual(result['assigned_start'].tolist(), [0, 10, 15, 21])
        self.assertEqual(result['assigned_end'].tolist(), [10, 15, 21, 31])
        self.assertEqual(result['states'].tolist(), [0, 0, 1, 1])
        self.assertEqual(result['imputed_samples'], 11)

    def test_observed_only_changes_reference_and_joins_AF_across_removed_time(self):
        result = gap_adjusted_timeline([(0, 10), (20, 30)], [1, -1, 1], [0, 10, 20], [10, 20, 30], 1, 'observed_only')
        self.assertEqual(result['reference_samples'], [(0, 20)])
        self.assertEqual(result['assigned_end'].tolist(), [10, 20])
        self.assertEqual(result['clock'], 'compressed_observed_samples')
        self.assertEqual(result['removed_samples'], 10)

    def test_terminal_context_and_all_missing_are_indeterminate(self):
        for strategy, expected in [('context', 'indeterminate_missing_flanks'), ('observed_only', 'indeterminate_no_observed_time')]:
            result = gap_adjusted_timeline([], [-1], [0], [7], 1, strategy)
            self.assertEqual(result['status'], expected);self.assertIsNone(result['states'])
        result = gap_adjusted_timeline([], [-1], [0], [7], 1, 'nonaf')
        np.testing.assert_array_equal(result['states'], [0]);self.assertEqual(result['assigned_end'][0], 7)


if __name__ == '__main__': unittest.main()
