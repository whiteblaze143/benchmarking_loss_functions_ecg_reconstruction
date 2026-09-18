# Review Summary: Paper 10

**Problem:** test acquisition robustness without confounding it with hospital membership.

**Rounds:** 1 / 5

**Final Verdict:** RETHINK

## Problem Anchor

Test whether a representation retains diagnosis while becoming stable to a known, label-preserving acquisition transform of the same ECG.

## Resolution Log

| Round | Finding | Resolution | Remaining Risk |
|---|---|---|---|
| 1 | Existing variants all optimize BCE and have no environments | Reframed around paired controlled transforms and objective-execution gates | no intervention artifact or implementation yet |
| 2 | Balanced transformations test stability but create no shortcut; normalized matching can hide environment in latent norms | Added G0–G8, four synthetic worlds, `erm_aug`, pair/aux ablations, norm probes, and survival/admissibility audit | no implementation or results yet |

## Literature-informed Guardrails

IRMv1 can fail to recover the desired invariance and may be fragile under finite environments; class-conditional alignment alone is not sufficient for domain generalization. The paired comparison is motivated by same-object matching rather than generic distribution alignment. Paper 10 therefore treats IRMv1/CORAL as comparators and makes no unobserved-hospital generalization claim. [Kamath et al.](https://proceedings.mlr.press/v130/kamath21a.html), [Mahajan et al.](https://proceedings.mlr.press/v139/mahajan21b.html), [Deep CORAL](https://mlanthology.org/eccv/2016/sun2016eccv-deep/)

## Review Limitation

The skill's external reviewer backend was unavailable in this workspace. This is an implementation and literature audit, not an independent model-family review.
