# Experiment Tracker

| Run ID | Purpose | Variant | Split | Status | Notes |
|---|---|---|---|---|---|
| T000 | unit/data contract | adapter | train record 1 | PASS | 5 passed, AnnData test skipped because dependency absent |
| T001 | author GeoVT forward | random frozen weights | synthetic 4608 | PASS | output `(1,1,4608)`, finite |
| T002 | backward interface | from-scratch one step | train record 1 | PASS | finite L1 and parameters |
| P001 | provenance diagnostic | hybrid lr 0.1, batch 512 | PanoBench train | INVALID-PILOT | stopped after epoch 28; flat L1 around 0.486; preserved, never a candidate |
| S001 | 10-epoch sanity gate | GeoVT default lineage, seed 123 | PanoBench train | TODO | require clear training-L1 decrease before N001 |
| N001 | full training | GeoVT default lineage, seed 123 | PanoBench train | BLOCKED | waits for S001; 200 epochs, batch 32, lr 1e-3 |
| G001 | held-out reconstruction | frozen N001 | PanoBench test | BLOCKED | waits for N001 |
| G002 | cross-view geometry | R1 vs R2 | PanoBench test | BLOCKED | waits for N001 |
| G003 | temporal recurrence | R0-R3 | patient cohorts | BLOCKED | waits for G002 gate |
