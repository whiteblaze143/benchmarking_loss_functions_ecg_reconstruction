# Foundation-Model Comparison Tracker

| Run ID | Purpose | Model | Split / configuration | Priority | Status |
|---|---|---|---|---|---|
| FM-00 | Archive and strict-load admission | ECGFounder 12L + 1L | Artifact-only | MUST | BLOCKED by P07 repair |
| FM-01 | Frozen-head evaluation | ECGFounder 1L | PTB-XL Fold 8, Lead I | MUST | TODO after FM-00 |
| FM-02 | Paired inference | ECGFounder 1L vs GraphECG vs SetOperator | Fold 8, Lead I | MUST | TODO after FM-01 |
| FM-03 | Full-context ceiling | ECGFounder 12L | PTB-XL Fold 8, full 12 lead | NICE | TODO after FM-00 |
| FM-04 | Foundation-model latency | ECGFounder / GraphECG / SetOperator | Predeclared native-preprocess boundary | NICE | TODO after clinical cells |

Excluded: HuBERT-ECG (no checkpoint), EchoNext Minimodel (task-specific predictor), ICM and zero-filled/made-up ECGFounder subset inputs.
