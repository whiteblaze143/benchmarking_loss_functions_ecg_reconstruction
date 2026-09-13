# Experiment Tracker

| Run ID | Purpose | Variant | Split | Status | Notes |
|---|---|---|---|---|---|
| T000 | unit/data contract | adapter | train record 1 | PASS | 5 passed, AnnData test skipped because dependency absent |
| T001 | author GeoVT forward | random frozen weights | synthetic 4608 | PASS | output `(1,1,4608)`, finite |
| T002 | backward interface | from-scratch one step | train record 1 | PASS | finite L1 and parameters |
| N001 | full training | seed 123 | PanoBench train | RUNNING | resumed after epoch 3 at batch 512; final epoch checkpoint; no test selection |
| G001 | held-out reconstruction | frozen N001 | PanoBench test | BLOCKED | waits for N001 |
| G002 | cross-view geometry | R1 vs R2 | PanoBench test | BLOCKED | waits for N001 |
| G003 | temporal recurrence | R0-R3 | patient cohorts | BLOCKED | waits for G002 gate |
