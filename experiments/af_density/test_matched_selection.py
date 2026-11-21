import unittest
import numpy as np
from matched_selection import patient_sample_weights, elapsed_balanced_accuracy, select_validation_threshold


class MatchedSelectionTests(unittest.TestCase):
    def test_patient_totals_equal_despite_duration_and_row_imbalance(self):
        patients = ['a', 'a', 'a', 'b']
        w = patient_sample_weights(patients, [30, 30, 1, 1000], [True, True, False, True])
        np.testing.assert_allclose(w, [.5, .5, 0, 1])
        with self.assertRaisesRegex(ValueError, 'No eligible'):
            patient_sample_weights(patients, [30, 30, 1, 1000], [True, True, True, False])

    def test_unknown_retains_both_denominators(self):
        r = elapsed_balanced_accuracy([np.nan, np.nan], [30, 0], [30, 30], ['a', 'a'], .5)
        self.assertEqual(r['balanced_accuracy'], 0)
        self.assertEqual(r['unknown_assigned_samples'], 60)

    def test_actual_af_assignment_differs_from_binary_window_target(self):
        r = elapsed_balanced_accuracy([.8, .2], [29, 1], [30, 30], ['a', 'a'], .5)
        self.assertAlmostEqual(r['patient_mean_time_sensitivity'], 29/30)
        self.assertAlmostEqual(r['patient_mean_time_specificity'], 29/30)

    def test_patient_mean_not_pooled_time(self):
        r = elapsed_balanced_accuracy([1, 0, 0, 1], [1000, 0, 1, 0], [1000, 1000, 1, 1], ['a', 'a', 'b', 'b'], .5)
        self.assertEqual(r['balanced_accuracy'], .5)
        self.assertEqual(r['positive_patient_denominator'], 2)

    def test_validation_patient_contract_and_tie_rule(self):
        result = select_validation_threshold([.9, .1], [30, 0], [30, 30], ['v', 'v'], allowed_validation_patients=['v'])
        self.assertEqual(result['selected']['threshold'], .5)
        self.assertEqual(len(result['candidates']), 99)
        with self.assertRaises(ValueError):
            select_validation_threshold([.9, .1], [30, 0], [30, 30], ['test', 'test'], allowed_validation_patients=['v'])


if __name__ == '__main__':
    unittest.main()
