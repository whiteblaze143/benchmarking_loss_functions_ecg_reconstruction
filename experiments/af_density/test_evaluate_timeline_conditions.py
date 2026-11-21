import unittest
import numpy as np
from evaluate_timeline_conditions import evaluate_conditions


class TimelineConditionTests(unittest.TestCase):
    def test_all_nine_conditions_share_reference_and_unknown_denominator(self):
        scores = np.array([[.01, .01, .01], [.99, np.nan, .99], [.01, .01, .01]])
        rows = evaluate_conditions(scores, [(6000, 12000)], [0, 6000, 12000], [6000, 12000, 18000], 200,
            threshold=.5, training_target_prior=.3,
            duration_model={'duration_pmf': np.array([[.2, .3, .5], [.2, .3, .5]]), 'initial_occupancy': [.7, .3]},
            released_rr_milliseconds=[1000]*90)
        self.assertEqual(len(rows), 9)
        for row in rows:
            result = row['results']
            self.assertEqual(result['reference_af_seconds'], 30)
            self.assertEqual(result['observation_seconds'], 90)
            if row['pattern'] == 1:
                self.assertEqual(result['unknown_reference_af_seconds'], 30)
                self.assertIsNone(result['burden_absolute_error'])
                self.assertIsNone(result['published_shared_RR_clock']['predicted'])
            else:
                self.assertEqual(result['unknown_seconds'], 0)
                self.assertIsNotNone(result['published_shared_RR_clock']['predicted'])


if __name__ == '__main__': unittest.main()
