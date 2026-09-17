# PTB-XL Distributional repECG Program

This directory is an isolated implementation of the eight-paper program in the
two frozen user specifications.  It does not consume learned artifacts from the
existing temporal repStat, Q-VCG, N003a, or Nef-Net experiments.

The program is intentionally gated.  All eight branches will be independently
runnable, but GPU-scale training starts only after the shared PTB-XL data,
label, preprocessing, phase, MMD, and Nyström contracts pass their tests.

## Authoritative specification

The later specification, described as the version to hand to an implementation
agent, is authoritative when the two source documents disagree.  See
`docs/SPEC_RECONCILIATION.md` for every known resolution.

## Program structure

```text
configs/                         frozen common configuration
docs/                            proposal, decisions, and execution evidence
refine-logs/                     research-refine-pipeline outputs
src/repecg/common/               shared non-learned harness
src/repecg/paper01_recurrence/   distributional recurrence operator
src/repecg/paper02_kernel_mean/  kernel-mean ECG
src/repecg/paper03_signature/    path-signature ECG
src/repecg/paper04_hankel/       local Hankel/DMD ECG
src/repecg/paper05_koopman/      recurrent-state Koopman ECG
src/repecg/paper06_conditional/  conditional residual repStat
src/repecg/paper07_operator/     continuous measurement-operator ECG
src/repecg/paper08_tokens/       statistical ECG tokenization
tests/                           invariant and mechanism tests
outputs/                         ignored runtime artifacts
```

## Execution gate

1. Validate official folds, patient disjointness, labels, lead order, and QC.
2. Validate exact MMD and Nyström approximation on development data only.
3. Train shared raw baselines and Paper 2 on folds 1--7; select on fold 8.
4. Run each branch's mechanism-destroying falsification before full seeds.
5. Freeze configurations; retrain on folds 1--8 and select on fold 9.
6. Open fold 10 once, only after the freeze manifest exists.

No fold-10 access is permitted during implementation or development.
