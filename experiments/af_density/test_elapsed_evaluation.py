import unittest
import numpy as np
from elapsed_evaluation import equal_bin_amounts, evaluate_elapsed_segment


class ElapsedEvaluationTests(unittest.TestCase):
    def test_equal_bins_cover_fractional_tail_without_padding(self):
        fractions = equal_bin_amounts([(0., .4), (1.2, 1.6)], 1.6, 1.)
        np.testing.assert_allclose(fractions, [.5, .5])
        self.assertAlmostEqual(float(fractions.sum())*.8, .8)

    def test_exact_identity_preserves_distinct_edge_conventions(self):
        result = evaluate_elapsed_segment([(0, 600)], [1, 1], [0, 300], [300, 600], 10)
        self.assertEqual(result['status'], 'complete')
        self.assertIsNone(result['reference_patterns']['density_continuous'])
        self.assertEqual(result['reference_patterns']['density_author_binned_30s_max'], 1)
        self.assertIsNone(result['pattern_absolute_errors']['density_continuous'])
        self.assertEqual(result['pattern_absolute_errors']['density_author_binned_30s_max'], 0)
        self.assertEqual(result['uncensored_boundary_count'], 0)

    def test_unknown_not_a_true_negative_or_false_complete_timeline(self):
        r = evaluate_elapsed_segment([(0, 300)], [-1, -1], [0, 300], [300, 600], 10)
        self.assertEqual(r['status'], 'indeterminate_unknown_time')
        self.assertEqual(r['unknown_seconds'], 60)
        self.assertEqual(r['all_reference_time_sensitivity'], 0)
        self.assertEqual(r['all_reference_time_specificity'], 0)
        self.assertIsNone(r['known_time_sensitivity'])
        self.assertIsNone(r['known_time_specificity'])
        self.assertIsNone(r['predicted_patterns'])
        self.assertIsNone(r['episode_errors'])

    def test_sample_clock_prefix_and_uncensored_boundaries(self):
        r = evaluate_elapsed_segment([(50, 150), (190, 310)], [1, 0, 1], [100, 200, 300], [200, 300, 320], 10)
        self.assertEqual(r['observation_seconds'], 22)
        self.assertEqual(r['reference_af_seconds'], 17)
        self.assertEqual(r['episode_errors']['merged_predicted_episodes'], 1)
        self.assertEqual(r['episode_errors']['split_reference_episodes'], 1)

    def test_touching_and_invalid_reference_rejected(self):
        for ref in [[(0, 10), (10, 20)], [(5, 4)], [(0., 10.)]]:
            with self.assertRaises(ValueError):
                evaluate_elapsed_segment(ref, [0], [0], [300], 10)


if __name__ == '__main__':
    unittest.main()
