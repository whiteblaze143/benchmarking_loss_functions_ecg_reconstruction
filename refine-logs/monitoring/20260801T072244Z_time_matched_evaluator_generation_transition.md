# Time-matched evaluator generation transition

Timestamp: 2026-08-01T07:22:44Z

- The fourth order-paired artifact (`f_1011014_s42`) published atomically before
  the old evaluator watcher was stopped. A partially started next evaluation
  did not publish an artifact.
- Evaluator generation `a04bc7dec1cd85857df4ce3b6ea9b6fb8fb7e1e4f19e72f5f92def3eaafceec5`
  replaces detection-order truncation with monotonic one-to-one event matching.
  The algorithm maximizes match cardinality and then minimizes total absolute
  detector-time error within a prespecified 100 ms tolerance.
- Extracted P/Q/R/S/T and QT events now carry sample indices. QT uses the
  detected QRS onset as its temporal anchor.
- Every accepted row must close both identities:
  `paired + unmatched_real = real_detected` and
  `paired + unmatched_reconstructed = reconstructed_detected`. Median and p95
  absolute timing errors must be finite and p95 cannot exceed 100 ms.
- Seven focused matcher/cache/summary tests pass, including a counterexample
  where naive first-in-order pairing chooses the wrong event, explicit missed
  and extra events, unsorted-event rejection, and accounting-tamper rejection.
- Full focused suite: 32 passed with only NeuroKit's upstream `scipy.misc`
  deprecation warning.
- Fail-closed transition state: 0/11 accepted, 4 older artifacts excluded for
  evaluator-SHA mismatch. The schema-v2 target cache is rebuilding on isolated
  CPU core 7; old artifacts remain forensic only.
- Remaining methodological work: tolerance sensitivity, detector-failure
  stratification, per-record paired endpoints, and seed-aware inference.

