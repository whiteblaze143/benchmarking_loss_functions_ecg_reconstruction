# Paper 03: Phase-Cell Signature Path

## The Core Question
Cardiac rhythms often suffer from extreme time-warping. A patient might have a wildly varying heart rate, arrhythmias, or ectopic beats. How do we deal with the fact that different patients traverse the physiological cardiac cycle at completely different speeds?

## The Mathematical Framework: Rough Path Theory
We derive our solution from **Rough Path Theory**, a branch of stochastic analysis. The "signature" of a path is an infinite series of iterated integrals that extracts global geometric properties of the trajectory (like the total algebraic area enclosed by the path in the RKHS). 

Crucially, path signatures are **completely invariant to time re-parameterization**. This means the signature of a path is identical regardless of how fast or slow the path is traversed.

## The Architecture
1. We mathematically lift the discrete sequence of phase-cell KMEs into a continuous geometric path tracing through the RKHS.
2. We use the `iisignature` mathematical library to compute the truncated signature features (up to a certain depth, e.g., depth 3 or 4).
3. These highly complex geometric features are mathematically extracted *without any learned neural network parameters*.
4. The deterministic signature vector is flattened and passed into a simple linear classification probe.

## The Biological Context
Because path signatures inherently ignore time-warping, they focus purely on the structural, geometric "shape" of the cardiac cycle's trajectory in the electrical RKHS space. 
If a diseased ischemic heart traces a distinctly different structural shape through electrical space compared to a healthy heart, the signature will flawlessly identify it, regardless of whether the patient's heart rate was 60 BPM or 120 BPM.

## Rigorous Falsification & Controls
- **Matched Control**: Raw temporal pooling of the KME vectors.
- **Falsification (The Kill Test)**: We intentionally synthetically time-warp the signal (stretching and compressing random parts of the heartbeat) in the test set. Because signatures are mathematically invariant to time-warping, this manipulation should *not* degrade the signature model's performance. However, it must severely degrade the standard control models.
