import unittest
import numpy as np
from verify_set_math_validation_v2 import reselect_threshold_independently
from matched_selection import select_validation_threshold


class IndependentThresholdTests(unittest.TestCase):
    def fixture(self):
        return (np.array([[.8, .7, np.nan], [.1, .2, .3], [.2, .1, .4], [.9, .8, .7]], np.float32),
                np.array([10, 0, 0, 15], np.int64), np.array([10, 20, 5, 15], np.int64),
                np.array(['a', 'a', 'b', 'b']))

    def test_unknown_keeps_denominator_and_candidate_ties_choose_point5(self):
        threshold, values = reselect_threshold_independently(*self.fixture())
        self.assertEqual(threshold, .5)
        self.assertAlmostEqual(values[49], 11/12)

    def test_all99_candidates_match_original_policy_on_partial_assignments(self):
        p, af, duration, patients = self.fixture()
        threshold, values = reselect_threshold_independently(p, af, duration, patients)
        expected = select_validation_threshold(p.reshape(-1), np.repeat(af, 3),
            np.repeat(duration, 3), np.repeat(patients, 3), allowed_validation_patients=['a', 'b'])
        self.assertEqual(threshold, expected['selected']['threshold'])
        np.testing.assert_array_equal(values, [r['balanced_accuracy'] for r in expected['candidates']])

    def test_all_unknown_does_not_drop_reference_classes(self):
        p, af, duration, patients = self.fixture();p[:] = np.nan
        threshold, values = reselect_threshold_independently(p, af, duration, patients)
        self.assertEqual(threshold, .5)
        self.assertEqual(values, [0.]*99)

    def test_missing_reference_class_is_rejected(self):
        p, af, duration, patients = self.fixture();af[:] = 0
        with self.assertRaises(ValueError): reselect_threshold_independently(p, af, duration, patients)


if __name__ == '__main__': unittest.main()
