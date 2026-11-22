import unittest
from paired_endpoint_comparison import compare_endpoints, METRICS


class EndpointTests(unittest.TestCase):
    def rows(self):
        return [{'patient': p, 'segment': s, 'pattern': a,
                 'models': {'raw': {m: .3 for m in METRICS},
                            'set': {m: .1 for m in METRICS}}}
                for p, segments in [('a', [0, 1]), ('b', [2])]
                for s in segments for a in range(3)]

    def run_rows(self, rows):
        return compare_endpoints(rows,
            expected_keys=[(r['patient'], r['segment'], r['pattern']) for r in self.rows()])

    def test_equal_patient_weight_despite_different_record_counts(self):
        rows = self.rows()
        for r in rows:
            if r['patient'] == 'b': r['models']['set']['burden_error'] = .3
        result = self.run_rows(rows)
        burden = next(x for x in result['comparisons'] if x['metric'] == 'burden_error')
        self.assertAlmostEqual(burden['conditional_mean_difference'], -.1)
        self.assertEqual(burden['complete_pair_patients'], 2)
        self.assertFalse(burden['preservation_claim_authorized'])

    def test_failure_retains_roster_and_only_invalidates_affected_endpoint(self):
        rows = self.rows();rows[0]['models']['set']['episode_ppv'] = None
        result = self.run_rows(rows)
        ppv = next(x for x in result['comparisons'] if x['metric'] == 'episode_ppv')
        self.assertEqual(ppv['declared_patients'], 2)
        self.assertEqual(ppv['complete_pair_ids'], ['b'])
        self.assertFalse(ppv['full_population_complete'])
        self.assertGreater(ppv['undefined_bootstrap_replicates'], 0)
        self.assertIsNone(ppv['conditional_percentile_95ci'])
        self.assertEqual(result['comparisons'][0]['complete_pair_patients'], 2)

    def test_all_undefined_has_no_interval(self):
        rows = self.rows()
        for r in rows: r['models']['set']['episode_ppv'] = None
        ppv = next(x for x in self.run_rows(rows)['comparisons'] if x['metric'] == 'episode_ppv')
        self.assertIsNone(ppv['conditional_percentile_95ci'])
        self.assertEqual(ppv['undefined_bootstrap_replicates'], 2000)

    def test_bad_population_or_nonfinite_endpoint_rejected(self):
        rows = self.rows()
        with self.assertRaises(ValueError): self.run_rows(rows[:-1])
        rows[0]['models']['raw']['density_error'] = float('nan')
        with self.assertRaises(ValueError): self.run_rows(rows)


if __name__ == '__main__': unittest.main()
