import unittest
import numpy as np
from decode_timeline import posterior_log_emissions, decode_segment
from hsmm import hsmm_viterbi


class TimelineDecodeTests(unittest.TestCase):
    def model(self):
        return {'duration_pmf': np.array([[.2, .3, .5], [.5, .3, .2]]), 'initial_occupancy': [.7, .3]}

    def config(self):
        return {'mode': 'D2', 'posterior_prior': .3, 'emission_strength': 1., 'probability_floor': 1e-6, 'nominal_input_seconds': 30.}

    def test_training_prior_and_tail_emission_weight(self):
        emissions, clipped = posterior_log_emissions([.3, .8, .8], posterior_prior=.3, emission_strength=1.,
            probability_floor=1e-6, assigned_fraction=[1., 1., .1])
        np.testing.assert_array_equal(emissions[0], [0., 0.])
        np.testing.assert_allclose(emissions[2], emissions[1]*.1)
        self.assertEqual(clipped, 0)

    def test_missing_window_splits_hsmm_without_compressing_clock(self):
        p = [.1, .9, np.nan, .8, .2]
        result = decode_segment(p, [6000, 12000, 18000, 24000, 30000], [12000, 18000, 24000, 30000, 36000], 200,
                                self.config(), duration_model=self.model())
        self.assertEqual(result['unknown_assigned_samples'], 6000)
        self.assertEqual(result['states'][2], -1)
        self.assertEqual(result['timeline']['unknown'], [(90, 120)])
        self.assertEqual(len(result['hsmm_available_blocks']), 2)
        for a, b in [(0, 2), (3, 5)]:
            emissions, _ = posterior_log_emissions(p[a:b], posterior_prior=.3, emission_strength=1., probability_floor=1e-6, assigned_fraction=[1, 1])
            expected = hsmm_viterbi(emissions, self.model()['duration_pmf'], self.model()['initial_occupancy'], left_censored=True, right_censored=True)
            np.testing.assert_array_equal(result['states'][a:b], expected['states'])

    def test_all_unknown_stays_unknown_in_all_decoders(self):
        configs = [{'mode': 'D0', 'threshold': .5}, {'mode': 'D1', 'threshold': .5, 'median_width': 3, 'max_gap_seconds': 60}, self.config()]
        for config in configs:
            result = decode_segment([np.nan, np.nan], [0, 6000], [6000, 12000], 200, config,
                                    duration_model=self.model() if config['mode'] == 'D2' else None)
            np.testing.assert_array_equal(result['states'], [-1, -1])

    def test_invalid_extra_config_and_discontinuous_clock_fail(self):
        with self.assertRaises(ValueError):
            decode_segment([.5], [0], [6000], 200, {'mode': 'D0', 'threshold': .5, 'silent_impute': True})
        with self.assertRaises(ValueError):
            decode_segment([.5, .5], [0, 7000], [6000, 13000], 200, {'mode': 'D0', 'threshold': .5})


if __name__ == '__main__':
    unittest.main()
