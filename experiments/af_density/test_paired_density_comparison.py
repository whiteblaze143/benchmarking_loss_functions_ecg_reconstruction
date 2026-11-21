import unittest
from paired_density_comparison import compare_density_errors


class PairedDensityTests(unittest.TestCase):
    def rows(self):
        return [{'patient': p, 'segment': s, 'pattern': v, 'set_error': .1, 'baseline_error': .3}
                for p, segments in [('a', [0, 1]), ('b', [2])] for s in segments for v in range(3)]

    def run_rows(self, rows):
        return compare_density_errors(rows, expected_keys=[(r['patient'], r['segment'], r['pattern']) for r in rows])

    def test_patient_equal_and_exact_constant_paired_difference(self):
        rows = self.rows()
        for row in rows:
            if row['patient'] == 'b': row['set_error'] = .2
        result = self.run_rows(rows)
        self.assertAlmostEqual(result['conditional_mean_error_difference'], -.15)
        self.assertTrue(result['full_cohort_superiority_supported'])

    def test_one_missing_condition_blocks_full_cohort_claim(self):
        rows = self.rows();rows[0]['set_error'] = None
        result = self.run_rows(rows)
        self.assertEqual(result['complete_paired_patients'], 1)
        self.assertFalse(result['full_cohort_superiority_supported'])
        self.assertEqual(result['indeterminate_patients'][0]['set_missing_conditions'], 1)
        self.assertLessEqual(result['full_cohort_mean_difference_bounds'][0], -.2)

    def test_duplicate_or_missing_condition_rejected(self):
        rows = self.rows()
        with self.assertRaises(ValueError): self.run_rows(rows+rows[:1])
        with self.assertRaises(ValueError): self.run_rows(rows[1:])


if __name__ == '__main__': unittest.main()
