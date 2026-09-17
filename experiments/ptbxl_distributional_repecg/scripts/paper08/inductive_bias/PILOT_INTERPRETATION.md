# Paper 08 equivalence-vocabulary pilot

This is an engineering-only 256-record, `K0=16`, 20-bootstrap pipeline check.
It is not a development result and cannot be promoted.

- reproducibility margin `delta(q95)`: `0.109289`
- minimum simultaneous candidate-pair UCB: `0.132566`
- unsupported base codes: `1 / 16`
- certified merges among the 15 supported codes: `0`
- validation unknown rate: `0.0881348`
- token-phase NMI: `0.277404`
- patient-equal odd/even agreement: `0.823738`

The important outcome is the absence of forced merges: even the closest pair's
upper bound exceeded the reproducibility margin. The full patient-supported
run may reduce uncertainty, but this pilot supplies no evidence that it will.
