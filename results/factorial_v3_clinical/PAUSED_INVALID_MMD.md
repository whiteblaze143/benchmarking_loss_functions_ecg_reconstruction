# EchoNext Queue Pause: Invalid Legacy MMD

Timestamp: 2026-07-25T02:47:00Z

The `factorial_v3_clinical_queue` was stopped after atomically completing
10/48 EchoNext cells. The preserved partial ends at
`unet__e1c0m0d1__s42`.

The legacy `mmd_loss` used a fixed-width Gaussian kernel over flattened
45,000--60,000-dimensional ECG records. At `[8,9,5000]` its loss was the
constant diagonal value 0.25 and its measured gradient L1 norm was
approximately `9.27e-22`. Two matched U-Net MMD-toggle checkpoint pairs were
tensor-identical. Continuing external evaluation would therefore waste GPU
time and could not support MMD factorial claims.

The corrected adaptive multi-bandwidth implementation measured gradient L1
approximately `2.39e-1` at `[8,9,5000]`. The v4 queue retrains the entire
controlled grid before this clinical evaluation resumes.
