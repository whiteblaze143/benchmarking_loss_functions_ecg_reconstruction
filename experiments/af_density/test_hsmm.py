import itertools
import unittest
import numpy as np
from hsmm import hsmm_viterbi


def exhaustive_score(states, emission, pmf, prior, left, right):
    changes = [0]+[i for i in range(1, len(states)) if states[i] != states[i-1]]+[len(states)]
    score = np.log(prior[states[0]]) + sum(emission[i, s] for i, s in enumerate(states))
    for a, b in zip(changes[:-1], changes[1:]):
        state, d = states[a], b-a
        if d > pmf.shape[1]:
            return -np.inf
        # Direct finite sums, independent of decoder cumulative arrays.
        if a == 0 and left:
            mean = sum((k+1)*p for k, p in enumerate(pmf[state]))
            if b == len(states) and right:
                probability = sum((k-d+2)*p for k, p in enumerate(pmf[state]) if k+1 >= d)/mean
            else:
                probability = sum(pmf[state, d-1:])/mean
        elif b == len(states) and right:
            probability = sum(pmf[state, d-1:])
        else:
            probability = pmf[state, d-1]
        if probability == 0:
            return -np.inf
        score += np.log(probability)
    return score


class HSMMTests(unittest.TestCase):
    def test_exhaustive_paths_all_censoring_combinations(self):
        rng = np.random.default_rng(811)
        for n in range(1, 8):
            emission = np.log(rng.uniform(.01, .99, size=(n, 2)))
            pmf = rng.uniform(.1, 1, size=(2, 8))
            pmf /= pmf.sum(axis=1, keepdims=True)
            prior = np.array([.7, .3])
            for left, right in itertools.product([False, True], repeat=2):
                result = hsmm_viterbi(emission, pmf, prior, left_censored=left, right_censored=right)
                best = max(exhaustive_score(s, emission, pmf, prior, left, right)
                           for s in itertools.product([0, 1], repeat=n))
                self.assertAlmostEqual(result['log_score'], best, places=12)
                self.assertAlmostEqual(exhaustive_score(result['states'], emission, pmf, prior, left, right), best, places=12)

    def test_impossible_duration_support_fails(self):
        with self.assertRaisesRegex(ValueError, 'No feasible'):
            hsmm_viterbi(np.zeros((1, 2)), [[0, 1], [0, 1]], [.5, .5], left_censored=False, right_censored=False)
        result = hsmm_viterbi(np.zeros((1, 2)), [[0, 1], [0, 1]], [.5, .5], left_censored=True, right_censored=True)
        self.assertEqual(len(result['states']), 1)

    def test_unknown_emission_and_invalid_laws_fail(self):
        with self.assertRaises(ValueError):
            hsmm_viterbi([[np.nan, 0]], [[1], [1]], [.5, .5], left_censored=True, right_censored=True)
        for law in [[[.2], [1]], [[-1, 2], [.5, .5]]]:
            with self.assertRaises(ValueError):
                hsmm_viterbi([[0, 0]], law, [.5, .5], left_censored=True, right_censored=True)


if __name__ == '__main__':
    unittest.main()
