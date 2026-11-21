import tempfile
import unittest
from pathlib import Path
import numpy as np
from generate_phase_features import make_arrays, commit, verified_commit


class PhaseFeatureCacheTests(unittest.TestCase):
    def failed_row(self):
        return {'window_index': np.array([0], np.int32), 'phase_valid': np.zeros((1, 2), bool),
                'accepted_intervals': np.zeros((1, 2), np.uint16), 'represented_fraction': np.zeros((1, 2), np.float32)}

    def test_total_failure_stays_in_denominator_without_transform_call(self):
        def forbidden(atoms):
            raise AssertionError('Failed representation must not be encoded')
        arrays, errors = make_arrays(np.full((6000, 2), np.nan, np.float32), [0], self.failed_row(), forbidden, None, None)
        self.assertEqual(arrays['response'].shape, (1, 2, 16, 128))
        self.assertTrue(np.isnan(arrays['response']).all())
        self.assertEqual(errors, [None, None])

    def test_modified_commit_is_not_silently_reused(self):
        arrays = self.failed_row();arrays['response'] = np.full((1, 2, 16, 128), np.nan, np.float32)
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory)/'file'
            receipt = commit(prefix, arrays, {'source': 'fixture'})
            self.assertEqual(verified_commit(prefix, {'source': 'fixture'}), receipt)
            with prefix.with_suffix('.npz').open('ab') as f: f.write(b'changed')
            with self.assertRaises(AssertionError): verified_commit(prefix, {'source': 'fixture'})

    def test_orphan_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory)/'file';prefix.with_suffix('.npz').write_bytes(b'orphan')
            with self.assertRaisesRegex(AssertionError, 'Orphan'):
                verified_commit(prefix, {'source': 'fixture'})


if __name__ == '__main__':
    unittest.main()
