# Experiment Plan

**Problem**: view-dependent ECG measurements can destabilize temporal recurrence inference.

**Method thesis**: a frozen GeoVT query operator sampled at eight standard views supplies a canonical, interpretable attribute geometry to unchanged repSpat.

## Claim map

| Claim | Minimum convincing evidence | Blocks |
|---|---|---|
| C1: cross-view geometry stability | R2 improves paired-view distance-matrix agreement over R1 without worse phase discrimination | B1, B2 |
| C2: recurrence utility | Patient-level R2 recurrence endpoint is no worse than R0 and positive controls remain separated | B3 |
| Anti-claim: gain is only dimensionality or smoothing | R2 beats R1; R3 identifies compression effect; selectivity does not collapse | B2, B3 |

## Frozen systems

- R0: Kors VCG, 3-D.
- R1: raw independent standard ECG, 8-D.
- R2: frozen Nef-Net completed panorama, 8-D; primary.
- R3: fixed Kors projection of R2, 3-D.
- Same records, temporal coordinates, CAHC candidates, IMQ MMD, whole-block permutations, global BH, seeds, and patient aggregation for all arms.

## Experiment blocks

### B0: training integrity

- Train only on `/data/mithunmanivannan/panobench/train`; never select on `test`.
- Use the final epoch, not a test-selected checkpoint.
- Verify exact resume, finite gradients, deterministic sample generation, source checksum, and held-out reconstruction metrics.
- Gate: no split contamination or incomplete checkpoint state.

### B1: canonicalization

- Represent the same held-out PanoBench event from multiple prespecified three-view subsets.
- Primary metric: correlation of vectorized pairwise temporal distance matrices.
- Secondary: nearest-neighbor overlap, pointwise error, CAHC partition ARI.
- Gate: R2 must improve the paired-view geometry metric over R1 with patient/record paired uncertainty intervals.

### B2: invariance-selectivity

- Invariance: within-event geometry agreement across view subsets.
- Selectivity: prespecified distinct-phase positive controls and reconstruction fidelity.
- Gate: no conclusion from lower MMD alone; improvement is rejected if phase discrimination materially degrades.

### B3: patient-level temporal repSpat

- Run R0-R3 through one frozen repSpat protocol.
- Primary endpoint: patient-level recurrence rejection quantity already defined by the temporal-repSpat audit.
- Report effect estimates and patient bootstrap intervals; BH non-rejection is not called equivalence.
- No cluster-admission purity threshold is used.

## Run order

| Milestone | Decision |
|---|---|
| M0 source/data/interface smoke | proceed only if copy, dataset, forward, backward, and resume contracts pass |
| M1 one full seed | stop if reconstruction is non-finite or degenerate |
| M2 held-out geometry | stop if invariance improves only by selectivity collapse |
| M3 R0-R3 patient analysis | go/no-go on the two claims |
| M4 seeds 42 and 2026 | run only after M3 passes; quantify training stochasticity |

No empirical success thresholds will be chosen after observing outcomes. Directional hypotheses, paired effect sizes, confidence intervals, and full distributions are reported; any formal non-inferiority margin requires an independently justified clinical or measurement basis before execution.
