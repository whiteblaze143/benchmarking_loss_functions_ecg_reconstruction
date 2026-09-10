# Observation-Constrained NullVCG

This directory isolates the new fixed-geometry Lead-I to 12-lead approach.

It must not import or reuse the earlier LVCG/ECG-AIM wrapper, beat-boundary,
pretraining, embedding, or control implementations under
`refine-logs/lvcg_ecgaim` and their associated scripts/modules.

The first executable phases are geometry validation and the validation-only
oracle kill gate. Neural training is prohibited until that gate passes.
