import itertools
import unittest
from episode_metrics import episode_errors


def runs(bits):
    result = []
    start = None
    for i, value in enumerate([*bits, 0]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            result.append((start, i))
            start = None
    return result


class EpisodeTests(unittest.TestCase):
    def test_identity_and_boundary_sign(self):
        r = episode_errors([(2, 5)], [(1, 6)], 10)
        self.assertEqual(r['episode_sensitivity'], 1)
        self.assertEqual(r['boundary_pairs'][0]['onset_error_seconds'], -1)
        self.assertEqual(r['boundary_pairs'][0]['offset_error_seconds'], 1)
        self.assertEqual(r['boundary_absolute_error_seconds'], 1)
        self.assertEqual(episode_errors([(2, 5)], [(2, 5)], 10)['burden_absolute_error'], 0)

    def test_split_and_merge_remain_visible(self):
        split = episode_errors([(1, 9)], [(1, 3), (5, 9)], 10)
        self.assertEqual(split['split_rate'], 1)
        self.assertEqual(split['episode_sensitivity'], 1)
        self.assertEqual(split['episode_ppv'], 1)
        self.assertEqual(split['boundary_pair_count'], 0)
        self.assertIsNone(split['boundary_absolute_error_seconds'])
        merge = episode_errors([(1, 3), (5, 9)], [(1, 9)], 10)
        self.assertEqual(merge['merge_rate'], 1)
        self.assertEqual(merge['false_positive_seconds'], 2)

    def test_touching_is_not_detection(self):
        r = episode_errors([(1, 3)], [(3, 5)], 10)
        self.assertEqual(r['episode_sensitivity'], 0)
        self.assertEqual(r['episode_ppv'], 0)
        self.assertEqual(r['missed_reference_episodes'], 1)
        self.assertEqual(r['false_predicted_episodes'], 1)

    def test_empty_denominators_and_full_af(self):
        r = episode_errors([], [], 10)
        self.assertIsNone(r['episode_sensitivity'])
        self.assertIsNone(r['episode_ppv'])
        self.assertIsNone(r['split_rate'])
        self.assertIsNone(r['merge_rate'])
        self.assertEqual(r['burden_absolute_error'], 0)
        self.assertEqual(episode_errors([(0, 10)], [], 10)['missed_reference_episodes'], 1)

    def test_invalid_intervals_fail(self):
        for intervals in [[(3, 2)], [(0, 11)], [(1, 3), (3, 5)],
                          [(4, 5), (1, 2)], [(1, float('nan'))]]:
            with self.assertRaises(ValueError):
                episode_errors(intervals, [], 10)
        for duration in [0, -1, float('inf')]:
            with self.assertRaises(ValueError):
                episode_errors([], [], duration)

    def test_exhaustive_short_timelines_against_all_pairs(self):
        sequences = list(itertools.product([0, 1], repeat=6))
        for a in sequences:
            for b in sequences:
                ref, pred = runs(a), runs(b)
                r = episode_errors(ref, pred, 6)
                # Independent all-pairs graph and per-bin confusion reference.
                edges = [(i, j) for i, (s, e) in enumerate(ref)
                         for j, (u, v) in enumerate(pred) if max(s, u) < min(e, v)]
                self.assertEqual(r['reference_overlap_degrees'],
                                 [sum(i == k for i, j in edges) for k in range(len(ref))])
                self.assertEqual(r['predicted_overlap_degrees'],
                                 [sum(j == k for i, j in edges) for k in range(len(pred))])
                self.assertEqual(r['true_positive_seconds'], sum(x and y for x, y in zip(a, b)))
                self.assertEqual(r['false_negative_seconds'], sum(x and not y for x, y in zip(a, b)))
                self.assertEqual(r['false_positive_seconds'], sum(y and not x for x, y in zip(a, b)))


if __name__ == '__main__':
    unittest.main()
