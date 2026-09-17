# Paper 3: Phase-Cell Signature Path

## The Core Question
Cardiac rhythms often suffer from extreme time-warping (e.g., wildly varying heart rates, arrhythmias). How do we deal with the fact that different patients traverse the cardiac cycle at completely different speeds?

## The Math
Derived from **Rough Path Theory**. The signature of a path extracts geometric properties (like the total area enclosed by the trajectory) that are completely invariant to time re-parameterization (how fast the path is traversed).

## The Architecture
We lift the sequence of phase-cell KMEs into a continuous geometric path. We use the `iisignature` library to compute the truncated signature features. These features are mathematically extracted without any learned neural network parameters, flattened, and then passed into a simple linear classification probe.

## The Mechanism (Biological Context)
Because path signatures inherently ignore time-warping, they focus purely on the structural, geometric shape of the cardiac cycle's trajectory in the RKHS. If an ischemic heart traces a different "shape" through electrical space than a healthy heart, the signature will catch it, regardless of whether the patient's heart rate was 60 BPM or 120 BPM.

## The Falsification & Control
- **Matched Control**: Raw KME pooling.
- **Falsification**: Time-warping the signal. Because signatures are invariant to time-warping, adding synthetic severe time-warps to the test set should not degrade the signature model's performance, but it *will* degrade standard models.
