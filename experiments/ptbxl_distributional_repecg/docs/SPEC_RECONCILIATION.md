# Frozen Specification Reconciliation

**Date:** 2026-09-16

## Source precedence

The two supplied documents describe the same fifteen-paper program but are not
identical.  The second document explicitly calls itself the version to hand to
an implementation agent and is therefore the normative contract.  The first
document is retained as design history and supplies sensitivities where useful.

## Resolved conflicts

| Topic | Earlier document | Implementation document | Frozen decision |
|---|---|---|---|
| Eligible record | at least 3 complete beats | at least 2 valid full R-to-R cycles | Primary: at least 2 cycles; report a stricter at-least-3 sensitivity |
| Final seeds | 42, 43, 44 | 42, 43, 44, 45, 46 | Five seeds |
| Development firewall | folds 1--8 train, 9 validation | 1--7 development train, 8 representation selection; then 1--8 final train, 9 selection | Use the stricter two-stage firewall |
| IMQ scale | train-pair median | whiten descriptors; source-faithful primary c-squared=1; sensitivity c in {0.5,1,2,median} | Whitened c-squared=1 primary plus all specified sensitivities |
| Nyström audit | 500 pairs, rho at least 0.95 | 1,000 pairs, rho at least 0.95 and median relative error below 10 percent | Use 1,000 pairs and both gates |
| Recurrence normalization | row-normalized transition operator | symmetric degree normalization | Symmetric normalization for the primary; row normalization is a named sensitivity only |
| Paper 1 representation | approximately 13 spectral features primary | direct, spectral, and learned 2-D encoder | Run all three; preregister direct operator versus matched mean recurrence as primary |
| Signature path | explicit time channel | no time channel in primary; endpoints, mean, and LogSig3 | No time channel primary; time-augmented path is a sensitivity |
| Hankel cellization | 16 cells of length 16, delay 4 | 8 cells of length 32, delay 6 | 8 cells, length 32, delay 6 |
| Koopman coordinates | per-record diffusion coordinates | global 32-anchor coordinates shared by all patients | Global aligned anchor coordinates; per-record bases are prohibited |
| Token initialization | 100k cells, K0=256 | 200k cells, K0=512 | 200k/K0=512, capped at 8 cells per ECG |
| Token merging | point estimate below delta | upper 95 percent bound below delta | Confidence-bound complete-linkage equivalence |

## Non-negotiable invariants

- PTB-XL `records500` is the sole primary training source. Evaluation spans all 9 available external dataset adapters.
- The primary task is five-superclass multilabel diagnosis.
- No clinical metadata enters a model.
- Physical-mV and train-standardized copies remain distinct.
- The independent basis is `[I, II, V1, V2, V3, V4, V5, V6]`.
- Learned objects are fit on the applicable training folds only.
- Fold 10 remains inaccessible until a freeze manifest is written.
- Each paper starts from PTB-XL and cannot import another paper's learned output.
- Every proposed mechanism has a matched control and a mechanism-destroying
  falsification.
- Statistical non-rejection is never interpreted as equivalence.

## Compute interpretation

"Full throughput" means maximizing utilization for a run that has passed its
gate.  It does not mean launching all branches or all seeds before their
mechanisms are validated.  This is consistent with the supplied program, which
requires the shared harness first and explicit kill criteria for each paper.
