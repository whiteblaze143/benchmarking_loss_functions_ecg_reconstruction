# Literature basis for the ECG admissible Morlet bank

## Implementation boundary

The built-in `ecg_admissible_morlet` bank is a reproducible engineering
implementation of the corrected complex Morlet wavelet. It is **not** an exact
reproduction of the unpublished "custom ECG-inspired mother wavelet" mentioned
in the motivating CinC abstract.

The mother wavelet is:

`psi(t) = pi^(-1/4) [exp(i*w0*t) - exp(-w0^2/2)] exp(-t^2/2)`.

The correction term makes the wavelet zero-mean. The implementation evaluates
the corresponding two-Gaussian frequency response on the discrete FFT grid,
keeps positive frequencies to produce analytic coefficients, and normalizes
each filter to equal L2 energy. Center pseudo-frequencies are logarithmically
spaced over the prespecified ECG band, 0.5--45 Hz. The default `w0`/cycle count
is 6; 4, 6, and 8 remain explicit sweep controls.

## Sources

1. R. Buessow, "An Algorithm for the Continuous Morlet Wavelet Transform,"
   *Mechanical Systems and Signal Processing* (2007), arXiv:0706.0099,
   https://arxiv.org/abs/0706.0099. Supports FFT-domain Morlet CWT computation.
2. S. Barmase, S. Das, and S. Mukhopadhyay, "Wavelet Transform-Based Analysis
   of QRS complex in ECG Signals," arXiv:1311.6460,
   https://arxiv.org/abs/1311.6460. Demonstrates Morlet-CWT sensitivity to ECG
   QRS morphology under noise and baseline drift.
3. A. M. S. Aguiar-Conraria and M. J. Soares, "The Continuous Wavelet
   Transform: A Primer" (2011). Gives the admissibility correction term and
   discusses the conventional `w0=6` scale/frequency interpretation.
4. J. M. Lilly, "Introduction to redundancy rules: the continuous wavelet
   transform comes of age," *Philosophical Transactions A* 376 (2018),
   doi:10.1098/rsta.2017.0258. Discusses analytic Morlet CWT, admissibility, and
   the scale-to-characteristic-frequency relationship.

## Claim discipline

- Allowed: "literature-grounded corrected analytic Morlet bank" and
  "abstract-inspired wavelet SSL experiment."
- Not allowed: "exact physiology-informed wavelet reproduction" or replication
  of the motivating abstract's reported delineation sensitivity.

## Potse/Stoks repolarization candidate

The physiology family uses a second, distinct analysis bank for SSL view B.
View A remains standard Morlet magnitude; view B is the phase (or a named
control representation) of a 0.9-second analytic UEG template bank.  The real
templates implement the simple Potse model in the polarity convention

`UEG(t) proportional to mean_surface_TMP(t) - TMP_local(t)`,

which is the inverted/scaled form validated by Orini et al. and produces the
expected early-positive / late-negative morphology with the positive-plateau
logistic parameterization in the generator.  The neighboring TMP average
varies recovery-time dispersion and skew; local recovery offset and transition
width are also varied.  A Hilbert transform supplies the quadrature component.

The Wyatt derivative mixtures 0.10, 0.25, and 0.50 are explicit ablations, not
the default bank.  Stoks et al. report that maximum T-wave upslope, regardless
of polarity, tracks repolarization more accurately than the alternative rule
in their contact and ECGI experiments (DOI 10.3389/fphys.2023.1158003).

This supports the labels "Potse/Stoks-model-derived repolarization candidate"
and "physiology-derived second SSL view."  It does not establish that a UEG
kernel is the exact unpublished CinC wavelet, nor that an intracardiac UEG
model transfers optimally to body-surface ECG without empirical validation.
