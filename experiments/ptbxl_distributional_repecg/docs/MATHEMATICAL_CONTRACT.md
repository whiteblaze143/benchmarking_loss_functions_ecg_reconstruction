# Mathematical and Tensor Contract

**Status:** frozen for implementation; no fold-10 data inspected.

## Indices and canonical tensors

For record `r`, let the filtered physical signal be

\[
X_r^{mV}\in\mathbb R^{5000\times12}
\]

in canonical order `[I, II, III, aVR, aVL, aVF, V1, ..., V6]`. The independent
basis is `B_r_mV = X_r_mV[:, [I, II, V1, ..., V6]]`, shape `(5000, 8)`.
`B_r_std` applies a mean and standard deviation fitted over all time samples of
the allowed training records, separately for each lead. Physical and
standardized arrays are different fields and are never overwritten.

After R detection, valid full cycles are PCHIP-resampled to

\[
Z_r[b,s,l]\in\mathbb R^{B_r\times256\times8}.
\]

Here `b` is chronological beat index, `s` is normalized phase sample, and `l`
is independent-lead index. A 16-cell view is
`Z16[r,b,g,t,l] = Z[r,b,16*g+t,l]`, shape `(B_r,16,16,8)`. Paper 4 alone uses
`Z8`, shape `(B_r,8,32,8)`.

## Empirical distributions

The term "cell distribution" has one of the following exact meanings:

| Use | One atom | Empirical distribution |
|---|---|---|
| Papers 1--2 | standardized 8-vector `Z16[b,g,t,:]` | all `16*B_r` atoms at fixed `g` |
| Paper 3 | one beat-cell path descriptor | the `B_r` descriptors at fixed `g` |
| Paper 4 | one beat-cell Hankel/DMD descriptor | the `B_r` descriptors at fixed `g` |
| Paper 5 | one beat-cell kernel mean of its 16 standardized samples | chronological sequence of `16*B_r` means |
| Paper 6 | paired physical macrostate and residual sample `(z_i,r_i)` | all sample pairs at fixed phase cell |
| Paper 7 | standardized two-vector `[x_q(s), dx_q(s)/ds]` | all phase samples and beats at fixed `(q,g)` |
| Paper 8 | one beat-cell kernel mean of its 16 standardized samples | candidate token occurrences |

No record, beat, or beat-cell normalization is permitted.

## Descriptor whitening

Every descriptor space has an independent whitening transform. A deterministic
reservoir from allowed training records is used to fit mean `m` and covariance
`C`. With eigendecomposition `C = V diag(e) V.T`,

\[
W(x)=\operatorname{diag}(\max(e,10^{-8})^{-1/2})V^T(x-m).
\]

The transform is fitted on training atoms only and serialized with record IDs,
folds, seed, sample count, and a SHA-256 input/config digest. Whitening applies
to kernel descriptors, never to the retained physical-mV tensor.

## Exact IMQ and Nyström coordinates

Write `x_tilde = W(x)`. The primary exact audit kernel is defined only in
whitened space:

\[
\kappa(\tilde x,\tilde y)=(1+\|\tilde x-\tilde y\|_2^2)^{-1/2}.
\]

and the biased empirical MMD estimator. Kernel-scale sensitivities replace the
leading one by `c^2` for `c in {0.5,1,2,training-pair median}`.

MiniBatchKMeans chooses `M=128` landmarks `A` from a deterministic reservoir of
at most 500,000 whitened training atoms. Landmarks therefore already live in
whitened space. Define

\[
G=\kappa(A,A)+\epsilon I,\qquad
\epsilon=10^{-6}\operatorname{tr}(K_{AA})/M,
\]

and compute the symmetric inverse square root with eigenvalues clamped below
`1e-10`. Coordinates and a kernel mean are

\[
\phi(x)=\kappa(W(x),A)G^{-1/2},\qquad
\mu_P=|P|^{-1}\sum_{x\in P}\phi(x).
\]

Fitting uses float64; stored landmarks/transforms and inference use float32.
For exact values `d_i` and approximations `a_i` on the 1,000 audit pairs, define
`floor = max(1e-8, 0.001 * median({d_i: d_i > 0}))` and relative error
`abs(a_i-d_i)/max(d_i,floor)`. The approximation must pass both locked
development gates: Spearman rho at least 0.95 and median relative error below
0.10. Otherwise the same procedure is repeated with `M=256`. Failure at 256
stops the branch.

## Eligibility and missingness

The common classification cohort requires at least two valid full cycles.
Every input metadata row is retained with `eligible`, `reason`, detected beat
count, and requested denominator. No eligible-only result may be reported
without the full reconciliation table.

| Endpoint | Additional requirement |
|---|---|
| Papers 1, 2, 6, 7, 8 primary representation | none beyond two cycles |
| Odd/even representation stability | at least four cycles, giving at least two per half |
| Paper 3 path distribution | at least two descriptors per phase cell |
| Paper 4 local operator distribution | at least two descriptors per phase cell |
| Paper 5 Koopman | at least three cycles, so `16*B-1 >= 32` transitions |

An insufficient endpoint is `ineligible`, never a zero vector or imputed
metric. Primary classification denominators and mechanism-endpoint denominators
are reported separately. The fixed-window all-record sensitivity is frozen now
as an unconditional secondary analysis; fold-10 exclusion cannot activate a
new analysis.

## Paper 1 recurrence operator

For the 16 phase-cell means, `d_gh = ||mu_g-mu_h||^2`. A single positive
`tau` is the median nonzero, non-neighbor `d_gh` over allowed training records.
Set `A_gh=exp(-d_gh/tau)` when cyclic distance exceeds one and zero otherwise.
Let `v_g=max(sum_h A_gh,1e-8)` and

\[
R=\operatorname{diag}(v^{-1/2})A\operatorname{diag}(v^{-1/2}).
\]

The primary vector is the 120 strict-upper-triangle values in lexicographic
`(g,h)` order. Spectral features and the 2-D CNN are secondary. A sham applies
the same cyclic relabeling to both phase identities and exclusion mask; the
destroyer permutes cell content while retaining the original exclusion mask.

## Paper 2 moment-preserving destroyer

For each `(record, phase cell)`, the destroyer draws `n=16*B` standard-normal
row vectors using a frozen seed. Let `E_c` be the centered draw, `S_E` its sample
covariance, and `S_X` the original cell covariance. With eigenspaces restricted
to eigenvalues at least `1e-8` times the largest eigenvalue, construct
`Y = E_c @ S_E^{-1/2} @ S_X^{1/2} + mean_X`. A final float64 recenter/recolor
step is required until the realized maximum mean and covariance errors are both
below `1e-8`. Thus the realized destroyed cell, not merely its expectation,
matches the original first two moments. The sham is a frozen permutation of the
original atom rows, which preserves the complete empirical distribution as well
as the exact moments.

## Paper 3 path descriptor

The primary path is the 8-channel standardized cell without a time channel or
base-point augmentation. `iisignature` preparation uses dimension 8 and depth
3 with its canonical log-signature ordering. The descriptor concatenates the
8-vector start, end, and mean with the log-signature, then applies the
train-only whitening followed by PCA to exactly 64 components (or all available
components if descriptor rank is below 64). Order destruction permutes the 14 interior
samples while preserving endpoints and the full sample marginal; a sham
applies `f_a(s)=s+a*sin(2*pi*s)/(2*pi)` for frozen `a` in `{-0.1,+0.1}` and
PCHIP-resamples back to 16 samples. Since `f'_a(s)>0`, this is monotone.

## Paper 4 local Hankel/DMD descriptor

For `cell` shape `(32,8)` and delay six, create 27 delay states of length 48.
Use the first 26 as columns of `X` and the next 26 as columns of `Y`. SVD rank
is `min(8, count(s_i/s_0 >= 1e-6))`. The reduced ridge-free DMD follows the
specified pseudoinverse with singular values clamped at `1e-8`. Eigenvalues are
sorted by decreasing magnitude, padded with zeros to eight, and encoded as
magnitude and wrapped angle. The descriptor additionally contains eight
normalized singular values, effective rank, 95-percent energy rank, top-1 and
top-3 energy, reconstruction residual, and spectral radius. Non-finite output
is an explicit failure. The destroyer permutes interior time indices while
fixing endpoints; the sham is the same monotone warp budget used for Paper 3.

## Paper 5 regularized Koopman estimator

Training-only MiniBatchKMeans fits 32 anchors to beat-cell kernel means. RBF
soft assignment with training-median anchor distance produces a shared
coordinate `z_i` whose entries sum to one. Chronological transitions include
cell 16 to cell 1 only for consecutive detected cycles. With `lambda=1e-2`,

\[
K=Z_+Z_-^T(Z_-Z_-^T+\lambda I)^{-1}.
\]

This is called a regularized Koopman observable estimator. Records failing the
transition-count gate are ineligible. The primary comparison is occupancy
versus `[occupancy, K descriptors]`; chronological permutation is paired with
an identity-order sham and preserves occupancy exactly.

## Paper 6 conditional residual estimator

Let `x_bar` be the train-only physical-lead mean and let the centered
train-only covariance supply the top three eigenvectors `U3`. For physical
sample `x`, define `x_c=x-x_bar`, `z=U3.T@x_c`, and `r=x_c-U3@z`. Sixteen train-only
K-means macrostate anchors yield stable softmax weights `pi_m(z)`. A separately
fitted residual Nyström map gives the weighted conditional mean

\[
\mu^r_{gm}=\frac{\sum_{i\in g}\pi_m(z_i)\phi_r(r_i)}
{\sum_{i\in g}\pi_m(z_i)+10^{-8}}.
\]

For cell `g`, set `N_g` to its atom count,
`p_gm = sum_{i in g} pi_m(z_i)/N_g`, and effective mass
`ess_gm=(sum pi_m)^2/(sum pi_m^2+1e-12)`. For pair `(g,h)`, macrostate `m` is
active only when both effective masses are at least two. Let
`a_ghm=sqrt(p_gm*p_hm)` on active states and zero otherwise, then

\[
w_{ghm}=a_{ghm}/\sum_j a_{ghj},\qquad
d^{cond}_{gh}=\sum_m w_{ghm}\|\mu^r_{gm}-\mu^r_{hm}\|_2^2.
\]

If the weight denominator is below `1e-8`, the record is `ineligible` for this
endpoint. The complete distance feeds Paper 1's train-fitted affinity map. This
is a soft-stratified conditional kernel mean, not a general conditional mean
embedding. The destroyer permutes residuals within record and phase cell; the
sham jointly permutes `(z,r)` pairs.

## Paper 7 acquisition interface

The model never receives the eight-lead basis tensor. For each record it sees a
set of `m` observed pairs `(q_j,F(q_j))`, with `m` sampled uniformly from one to
six. Diagnosis uses only that set. A target `q*` and `F(q*)` are held out from
the context for the secondary reconstruction loss. Operators are generated
from the physical basis before applying one shared train-derived voltage scale.

The orientation law is the paired identity

\[
(q,x_q)\equiv(-q,-x_q).
\]

During primary training, four of eight canonical basis operators are sampled
without replacement and four sparse operators are generated by a seeded
Gaussian draw, top-three absolute-value truncation, and unit normalization.
The context size is sampled uniformly from one through six from these eight.
A frozen seen-validation bank contains all canonical basis operators and 100
sparse operators from seed 1701. Frozen unseen banks contain the normalized
derived-limb coefficients, 100 dense Gaussian unit vectors from seed 1702, and
the stated I-to-V2 interpolation family; any exact overlap with a seen vector up
to sign and tolerance `1e-8` is removed.

The primary model is trained with diagnosis BCE only. An explicitly secondary
auxiliary model adds reconstruction weight 0.1 and paired-orientation
consistency weight 0.05. Orientation consistency is the symmetric KL divergence
between diagnostic predictions from sign-paired contexts; target-response MSE
uses the explicitly computed `F(-q)`, because nonlinear kernel coordinates are
not assumed to negate. Held-out operators test interpolation within the span of
observed leads, never new electrode locations.

## Paper 8 equivalence vocabulary

Patient IDs are deterministically partitioned inside the allowed training folds
into construction (80%) and equivalence-calibration (20%) subsets. K-means
over-segmentation (`K0=512`) is fit on at most 200,000 construction beat-cells,
at most eight per ECG. The calibration subset alone supplies 100
patient-blocked within-candidate splits and
`delta = quantile_0.95(D_within)`.

Freeze the candidate-pair family before uncertainty estimation. For every
patient-clustered bootstrap replicate, compute all pair distances and the
maximum centered deviation `T_b=max_pair(D_b-D_hat)`. Let `c_0.95` be the 95th
percentile of `T_b`; the simultaneous familywise upper bound is
`U_pair=D_hat_pair+c_0.95`. Complete-linkage merging is permitted only when
every simultaneous candidate-pair upper bound in the proposed token is below
`delta`.
The deterministic merge order is increasing maximum upper bound, then
lexicographic candidate ID. This is an effect-size equivalence construction;
no failure-to-reject p-value enters merging. The separate source-faithful audit
uses beat blocks, 4,999 final permutations, and BH q=0.05.

New occurrences map to the nearest final prototype and become `<UNK>` beyond
the training 95th percentile of within-token distance. The mechanism-use
destroyer performs size-matched complete-linkage merges ordered by a frozen
random permutation rather than UCB, retaining final vocabulary size and model
capacity.

## Statistical aggregation

Predictions are record-level. Metrics are computed over records, while every
bootstrap samples patients and includes all of each sampled patient's records.
Final primary predictions are the arithmetic mean of five fixed-seed
probabilities; seed-level metrics and SD are also reported. Each paper has one
locked primary comparison. If any umbrella claim is made, Holm correction is
applied across the eight primary comparisons; otherwise each paper is reported
independently and BH is limited to declared secondary families.
