"""Development-only analytical illustration; no patient outcomes are read."""
import hashlib
import json
from pathlib import Path
from scipy.stats import binom, beta

W = Path('/data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1')


def main():
    result = {'test_outcomes_accessed': False, 'data_access': 'No cohort inputs or outcomes read',
              'scope': ('Illustrative exact sign-test calculations on independent, untied patient pairs; '
                        'not the proposed percentile-bootstrap decision rule, actual study power, '
                        'episode-level sensitivity, or evidence for a noninferiority margin.'),
              'sign_test': {}}
    for n in (4, 5, 11, 13):
        critical = next((k for k in range(n + 1) if binom.sf(k - 1, n, .5) <= .05), None)
        result['sign_test'][str(n)] = {
            'one_sided_alpha': .05,
            'minimum_favorable_pairs_for_rejection': critical,
            'exact_p_if_all_favorable': float(binom.sf(n-1, n, .5)),
            'rejection_probability_given_favorable_pair_probability': {
                str(p): 0.0 if critical is None else float(binom.sf(critical-1, n, p))
                for p in (.65, .75, .8, .9)},
        }
    result['zero_failures_illustration'] = {
        'n_patients': 13, 'one_sided_lower_95_percent_bound': float(beta.ppf(.05, 13, 1)),
        'two_sided_lower_95_percent_bound': float(beta.ppf(.025, 13, 1)),
        'estimand': 'A hypothetical independent binary patient-level success probability; not AF episode sensitivity or preservation.',
    }
    result['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out = W/'pre_release_corrective_v1/precision_feasibility_fixed_n_v1.json'
    assert not out.exists()
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
