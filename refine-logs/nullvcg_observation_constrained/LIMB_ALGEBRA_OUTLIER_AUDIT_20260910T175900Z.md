# Fold-9 limb-algebra outlier audit

The full geometry-specificity oracle stopped because the source-data integrity rule failed. A second read-only scan localized all gross violations and verified them against the original 500 Hz PTB-XL WFDB records.

## Findings

- Only two of 2,183 fold-9 ECGs have residuals above 0.01 mV.
- ECG 5930 (`records500/05000/05930_hr`) contains a source excursion at sample 4871: I = −0.070, II = 0.095, aVL = 2.242, aVF = 2.490 mV. The algebraic expectations are aVL = −0.1175 and aVF = 0.1300 mV, giving ≈2.36 mV violations.
- ECG 15867 (`records500/15000/15867_hr`) contains a source excursion at sample 1159: I = −0.107, II = −0.035, III = 0.839 mV. The expected III is 0.072 mV, giving a 0.767 mV violation.
- The tensor copies and original WFDB records agree; this is source-record behavior, not tensor conversion or evaluator error.
- Across all 43,660,000 validation samples per lead, counts above 0.002 mV were: III 21, aVR 2, aVL 13, aVF 15. Counts above 0.01 mV were: III 6, aVR 0, aVL 7, aVF 8; all >0.01 mV samples belong to the two ECGs above.

## Decision

The frozen 0.002 mV integrity ceiling is exceeded, so `GEOMETRY_SPECIFICITY_GATE=FAIL` remains authoritative and neural training is stopped. A future protocol may preregister either robust clipping or exclusion of the two source-corrupt records, but this run cannot make that change retrospectively.

