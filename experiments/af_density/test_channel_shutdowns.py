import unittest
import numpy as np
from channel_shutdowns import shutdown_schedule, select_available_scores


class ShutdownTests(unittest.TestCase):
    def test_schedule_deterministic_disjoint_and_nearest_declared_target(self):
        for n in [1, 11, 2880]:
            for fraction in [.01, .05, .10]:
                a, receipt = shutdown_schedule(n, dataset='fixture', record='1', segment=0, fraction=fraction, block_seconds=300)
                b, again = shutdown_schedule(n, dataset='fixture', record='1', segment=0, fraction=fraction, block_seconds=300)
                np.testing.assert_array_equal(a, b);self.assertEqual(receipt, again)
                self.assertGreater(a.sum(), 0)
                self.assertEqual(a.sum(), sum(b-a for a, b in receipt['selected_slots']))
                self.assertLessEqual(abs(a.sum()-round(n*fraction)), 10)

    def test_subset_selection_retains_model_failure_and_both_missing(self):
        scores = np.array([[.1, .2, .3], [.4, .5, .6], [.7, .8, .9], [1., .2, .3], [np.nan, .8, .9]])
        availability = np.array([[1, 0], [0, 1], [1, 1], [0, 0], [1, 0]], bool)
        np.testing.assert_array_equal(select_available_scores(scores, availability), [.1, .5, .9, np.nan, np.nan])


if __name__ == '__main__': unittest.main()
