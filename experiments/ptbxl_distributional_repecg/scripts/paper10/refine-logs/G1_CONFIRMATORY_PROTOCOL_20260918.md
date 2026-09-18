# Paper 10 G1 Confirmatory Signal-Survival Protocol

**Frozen before reading fold 7.** Development-train folds 1–6 were used only to set these thresholds. Fold 7 is the confirmatory selection split; folds 8–10 remain unread.

This gate establishes preprocessing survival and usable coverage, not preservation of diagnostic information. The latter requires the frozen clinical comparison.

## Primary bank

Clean, gain 0.8, gain 1.2, 500-to-250-to-500 resampling, and three deterministic 20 dB noise realizations.

Every primary environment must satisfy:

1. overall common-support coverage at least 0.99;
2. coverage at least 0.98 in every reported diagnosis subgroup;
3. mean valid-cycle count at least 95% of the clean mean;
4. mean Phase-KME drift at least 0.02, proving that preprocessing did not erase the intervention;
5. finite stage metrics and exact same-record provenance.

For 20 dB noise, mean measured post-filter SNR must be between 24 and 29 dB. This verifies survival after the frozen filter; it is not required to remain numerically equal to raw SNR.

## Boundary bank

The three 10 dB noise realizations remain boundary-only regardless of this result. They are reported but cannot enter primary training based on G1 alone.

## Decision

G1 passes only if every primary criterion passes on fold 7 without threshold revision. Failure removes the offending transform from the primary bank or stops Paper 10 if no meaningful bank remains.
