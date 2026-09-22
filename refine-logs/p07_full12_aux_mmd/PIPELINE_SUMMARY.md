# Pipeline Summary

**Problem:** current SetOperator auxiliary reconstruction does not predict complete 12-lead ECGs.

**Final Method Thesis:** reuse the continuous encoder, align full and sparse real-view latents with IMQ MMD², and reconstruct only real train-standardized full 12-lead ECGs during training.

**Final Verdict:** READY_FOR_IMPLEMENTATION (not an external scientific verdict)

## First Runs

1. Build and verify the fold-1–7/8 waveform target artifact.
2. Run objective-gradient and decoder-shape tests.
3. Train `continuous_full12_aux_mmd` after the current Braid wave releases the GPU.

## Main Risk

The phase-set input may not retain enough rhythm information for low full-waveform MSE. This is measured and reported; it does not justify substituting a lower-dimensional or synthetic target.
