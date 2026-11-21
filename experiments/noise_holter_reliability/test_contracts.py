import json
import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from build_cohorts import clinical_group


ROOT = Path(os.environ.get('HOLTER_RELIABILITY_ARTIFACTS', '/data/mithunmanivannan/codex_artifacts/noise_holter_reliability_20261120'))


class ClinicalContractTests(unittest.TestCase):
    def test_af_with_ectopy_is_not_negative(self):
        self.assertEqual(clinical_group('N1', {'A. Fib.', 'VPB'}), 'N1_AF')

    def test_flutter_is_separate_from_ectopy(self):
        self.assertEqual(clinical_group('N1', {'A. Flutter', 'SVPB'}), 'N1_AFL_sensitivity')

    def test_unknown_is_not_normal_sinus(self):
        self.assertEqual(clinical_group('Clean', set()), 'excluded_no_primary_reference')

    def test_severe_noise_never_becomes_rhythm_negative(self):
        self.assertEqual(clinical_group('N2', {'A. Fib.'}), 'N2_safety')
        self.assertEqual(clinical_group('N2', set()), 'N2_safety')

    def test_duplicate_conflicts_excluded_and_geometry_unknown(self):
        frame = pd.read_parquet(ROOT / 'manifests/segment_manifest.parquet')
        self.assertEqual(len(frame), 6130)
        conflict = frame[frame.duplicate_annotation_conflict]
        self.assertEqual(len(conflict), 13)
        self.assertFalse(conflict.eligible_10s.any())
        self.assertFalse(conflict.eligible_30s.any())
        self.assertTrue((frame.patient_id_if_verified == 'unknown').all())
        self.assertTrue((frame.channel_geometry == 'unknown').all())
        self.assertFalse(frame[(frame.quality_class == 'N1') & frame.af_present & (frame.duration_seconds < 30)].eligible_30s.any())

    def test_execution_does_not_imply_scientific_admission(self):
        data = json.loads((ROOT / 'reports/window_export_semantic_failure.json').read_text())
        self.assertEqual(set(data), {'RawECGNet', 'ArNet2'})
        for model in data.values():
            self.assertLess(model['maximum_error_export_vs_sigmoid_of_head'], 1e-6)
            native = np.asarray(model['native_head_probability'])
            exported = np.asarray(model['window_export_score'])
            np.testing.assert_allclose(exported, 1 / (1 + np.exp(-native)), atol=1e-6)
            self.assertLess(model['source_fixed_threshold'], .5)
            self.assertTrue((exported > model['source_fixed_threshold']).all())
            self.assertEqual(model['scientific_admission'], 'rejected_as_shipped_window_export')
        self.assertNotEqual(data['RawECGNet']['native_head_calls_at_source_threshold_DIAGNOSTIC_ONLY'],
                            data['RawECGNet']['window_export_calls_at_source_threshold'])


if __name__ == '__main__':
    unittest.main()
