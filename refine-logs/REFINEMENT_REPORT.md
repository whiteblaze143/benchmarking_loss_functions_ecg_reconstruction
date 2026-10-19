# Refinement Report: SetOperator 168-Hour Architecture Campaign

## 1. Evolution of the Architecture Thesis
- **Initial Concept**: Explore arbitrary neural network modifications (attention depth, loss functions, token dimensions).
- **V3 Refined Thesis**: Focus strictly on the core scientific claim of continuous measurement operator generalization. Decompose the architecture into 5 distinct mechanistic hypotheses:
  1. Operator representation (signed vs projective vs relational).
  2. Cardiac phase morphology (local convolution vs global vs absolute phase anchoring).
  3. Set aggregation (cross-token interaction vs independent encoding).
  4. Clinical modality attribution (ECG vs covariates).
  5. Factorial interaction.

## 2. Decision Log
- **Decision 1: Preserve Active Production Runs**: Baseline SO-Span seed 42 (`train_so_span_s42`) and the queued G1 run (`g1_launch_controller`) must run to completion without modification or disruption.
- **Decision 2: Sequential Queue Execution**: All ablations will run strictly one at a time on the A100.
- **Decision 3: Dedicated Watchdog Supervisor**: The supervisor runs in detached tmux session `paper07_ablation_week` and checks `pipeline_complete_summary.json` before initiating the ablation queue.
- **Decision 4: Hard 168-Hour Cap**: The supervisor dynamically tracks elapsed GPU-hours and enforces a hard stop at 168 hours.
