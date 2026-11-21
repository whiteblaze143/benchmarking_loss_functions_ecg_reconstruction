import unittest
import numpy as np
from duration_fit import duration_log_laws, fit_duration_law


class DurationFitTests(unittest.TestCase):
    def test_censoring_laws_against_direct_finite_sums(self):
        for mu, sigma in [(0., .1), (2., .6), (7., 3.)]:
            laws = duration_log_laws(mu, sigma, 64)
            p, survival, residual, residual_survival = [np.exp(x) for x in laws]
            self.assertAlmostEqual(p.sum(), 1)
            np.testing.assert_allclose(survival, [p[i:].sum() for i in range(64)], atol=1e-14)
            mean = np.dot(p, np.arange(1, 65))
            np.testing.assert_allclose(residual, survival/mean, atol=1e-14)
            np.testing.assert_allclose(residual_survival, [residual[i:].sum() for i in range(64)], atol=1e-14)

    def test_uncensored_parameter_recovery(self):
        rng = np.random.default_rng(61)
        lengths = np.clip(np.floor(rng.lognormal(2., .6, 5000)+.5), 1, 128).astype(int)
        result = fit_duration_law(lengths, np.zeros(5000, bool), np.zeros(5000, bool),
                                  np.ones(5000), support=128)
        self.assertAlmostEqual(result['mu_log_steps'], 2., delta=.04)
        self.assertAlmostEqual(result['sigma_log_steps'], .6, delta=.04)

    def test_all_censored_or_invalid_inputs_fail(self):
        with self.assertRaises(ValueError):
            fit_duration_law([4], [True], [True], [1], support=64)
        with self.assertRaises(ValueError):
            fit_duration_law([4], [False], [False], [0], support=64)


if __name__ == '__main__':
    unittest.main()
