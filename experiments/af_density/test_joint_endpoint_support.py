import unittest
from assess_joint_endpoint_support import assess_support, METRICS


class JointSupportTests(unittest.TestCase):
    def report(self):
        rows = []
        for baseline in ['raw', 'rr']:
            for metric in METRICS:
                # Exercise reversed orientation: baseline minus categorical.
                ci = [.1, .2] if metric.endswith('_error') else [-.2, -.1]
                rows.append({'model': baseline, 'comparator': 'categorical', 'metric': metric,
                    'declared_patients': 3, 'complete_pair_patients': 3,
                    'complete_pair_ids': ['a', 'b', 'c'], 'full_population_complete': True,
                    'conditional_percentile_95ci': ci})
        return {'patients': ['a', 'b', 'c'], 'models': ['categorical', 'raw', 'rr'], 'comparisons': rows}

    def test_all_required_favorable_but_never_authorizes_clinical_or_test_claim(self):
        result = assess_support(self.report())
        self.assertTrue(result['density_statistical_support'])
        self.assertTrue(result['joint_computational_improvement_support'])
        self.assertFalse(result['clinical_preservation_claim_authorized'])
        self.assertFalse(result['test_inference_authorized'])

    def test_one_weaker_baseline_cannot_be_selected(self):
        r = self.report();r['comparisons'][4]['conditional_percentile_95ci'] = [-.1, .1]
        self.assertFalse(assess_support(r)['density_statistical_support'])

    def test_no_preservation_from_null_or_zero_boundary(self):
        r = self.report();r['comparisons'][2]['conditional_percentile_95ci'] = [-.2, 0]
        result = assess_support(r)
        self.assertTrue(result['density_statistical_support'])
        self.assertFalse(result['joint_computational_improvement_support'])

    def test_missing_patient_blocks_population_claim(self):
        r = self.report();r['comparisons'][0]['complete_pair_patients'] = 2
        r['comparisons'][0]['complete_pair_ids'] = ['a', 'b']
        r['comparisons'][0]['full_population_complete'] = False
        self.assertFalse(assess_support(r)['density_statistical_support'])

    def test_undefined_interval_cannot_pass(self):
        r = self.report();r['comparisons'][0]['conditional_percentile_95ci'] = None
        self.assertFalse(assess_support(r)['density_statistical_support'])

    def test_missing_duplicate_or_invalid_pair_rejected(self):
        r = self.report();r['comparisons'].append(r['comparisons'][0])
        with self.assertRaises(ValueError): assess_support(r)
        r = self.report();r['comparisons'][0]['conditional_percentile_95ci'] = [float('nan'), .2]
        with self.assertRaises(ValueError): assess_support(r)


if __name__ == '__main__': unittest.main()
