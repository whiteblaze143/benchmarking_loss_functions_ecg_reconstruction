# Claim, Control, and Gate Matrix

These are development decision gates, frozen before any fold-8 result. They do
not convert fold 8 into a test set and cannot be changed after fold 10 opens.

Common notation:

- `Delta_task`: proposed minus primary comparator macro AUROC.
- `Delta_stability`: proposed minus comparator representation stability.
- `Delta_mechanism`: the single scalar named in the table, oriented so positive
  means the proposed information was specifically destroyed or improved over
  its matched mechanism control.
- Confidence intervals are 95-percent patient-clustered bootstrap intervals.
- Task ROPE is plus/minus 0.005 macro AUROC. A stability improvement is
  material at 0.05 on a zero-to-one similarity scale.

A branch passes its development gate when its mechanism interval is strictly
positive and either (a) the lower task interval exceeds 0.005, or (b) the lower
stability interval exceeds 0.05 while the lower task interval exceeds -0.005.
It fails when the upper mechanism interval is at most zero, or when the upper
task interval is below -0.005 without a lower stability interval above 0.05.
All other outcomes are terminal `INCONCLUSIVE` development outcomes. No repeat,
new seed, or gate revision is allowed for the primary development decision.
The branch remains implemented and its negative/inconclusive result is retained,
but it does not advance to the five-seed claim run.

| Paper | Primary claim | Primary comparator | Destroyer / sham | Scalar primary mechanism statistic | Safeguards / secondary endpoints | Failure interpretation |
|---|---|---|---|---|---|---|
| 1 | Distributional distant recurrence adds information or stability beyond mean recurrence | Direct mean-recurrence vector with identical probe | Content-only phase permutation / joint cyclic reindexing | original-to-sham normalized Frobenius similarity minus original-to-destroyer similarity | task AUROC drop | recurrence gain is explainable by means or phase indexing |
| 2 | Nonlinear kernel means retain useful higher-order cell structure | mean plus covariance with identical phase CNN | exactly moment-matched Gaussian / atom-order permutation | original-to-sham KME cosine minus original-to-destroyer cosine | IMQ-vs-linear separation and task drop | KME behaves as a moment representation |
| 3 | Local path order is used | unordered waveform KME with identical phase CNN | endpoint-preserving interior permutation / fixed monotone warp | original-to-sham embedding cosine minus original-to-destroyer cosine | task AUROC drop | claimed path geometry is unused |
| 4 | Local phase-conditioned dynamics matter | raw local waveform KME | endpoint-preserving temporal permutation / fixed monotone warp | original-to-sham Hankel-descriptor cosine minus original-to-destroyer cosine | raw-KME change and task AUROC drop | improvement is morphology or corruption sensitivity, not dynamics |
| 5 | Transition dynamics add beyond state occupancy | occupancy-only vector with identical probe | chronological shuffle / identity order | destroyer minus sham normalized one-step prediction error | rollout error and task AUROC drop | state composition, not transitions, carries the signal |
| 6 | Conditional residual structure adds beyond both marginals | concatenated macro and residual marginals | residual-within-cell permutation / joint-pair permutation | original-to-sham recurrence similarity minus original-to-destroyer similarity | task AUROC drop | conditional pairing is irrelevant |
| 7 | Continuous operator coordinates improve unseen-operator behavior over lead IDs | categorical `<UNK_q>` set encoder matched in width/depth | continuous `q` / frozen unseen `<UNK_q>` | categorical-auxiliary minus continuous-auxiliary mean response MSE on the frozen unseen bank | BCE-only pair diagnosis AUROC; sign consistency; interpolation curvature | model is disguised channel identification |
| 8 | UCB-equivalence tokens outperform size-matched arbitrary merging | random complete-linkage vocabulary with identical size/transformer | random merge order / UCB merge order | patient-equal difference in `1-JSD(p_odd,p_even)/log(2)` | `<UNK>`, perplexity, phase NMI, task AUROC | performance comes from vocabulary size or transformer capacity |

Branch-specific hard gates remain binding: Nyström fidelity must pass; Paper
5 must meet its transition-count rule; Paper 7 may claim only within-span
operators; Paper 8 fails a general-purpose claim if locked-test `<UNK>` exceeds
20 percent or phase nearly determines tokens. "Nearly determines" is fixed as
normalized mutual information `NMI(token, phase) >= 0.90`.

Every scalar representation statistic is computed per record, averaged within
patient, and then patient-equal. Cosine and normalized Frobenius similarity are
one for two zero vectors/matrices and zero when exactly one argument is zero;
these cases are counted and reported. Bootstrap intervals resample patients.
