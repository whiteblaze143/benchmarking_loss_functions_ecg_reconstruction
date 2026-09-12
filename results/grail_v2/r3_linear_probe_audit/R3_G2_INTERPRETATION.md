# R3-G2 Reconciliation and Interpretation

- Rows: 117 unique target-concept cells
- Duplicate keys: 0
- Missing target cells: 0
- All probes converged: yes
- Fold 8 remained evaluation-only.

## Decision table

| target    | tier   |   n_concepts |   residual_auroc_mean |   incremental_delta_mean |   residual_retention_mean |   incremental_positive_n | interpretation_gate                                         |
|:----------|:-------|-------------:|----------------------:|-------------------------:|--------------------------:|-------------------------:|:------------------------------------------------------------|
| model_001 | anchor |           25 |                0.6516 |                  -0.0010 |                    0.3998 |                       14 | complementary but redundant accessibility                   |
| model_001 | P1     |            8 |                0.7324 |                   0.0086 |                    0.5817 |                        6 | incrementally useful complementary structure                |
| model_001 | P2     |            4 |                0.7480 |                  -0.0039 |                    0.5350 |                        2 | complementary but redundant accessibility                   |
| model_001 | P3     |            2 |                0.7086 |                   0.0075 |                    0.4480 |                        2 | incrementally useful complementary structure                |
| model_101 | anchor |           25 |                0.7431 |                   0.0220 |                    0.5886 |                       21 | incrementally useful complementary structure                |
| model_101 | P1     |            8 |                0.7906 |                   0.0189 |                    0.6825 |                        7 | incrementally useful complementary structure                |
| model_101 | P2     |            4 |                0.7777 |                  -0.0054 |                    0.6096 |                        1 | predictive residual with negative incremental accessibility |
| model_101 | P3     |            2 |                0.6783 |                   0.0113 |                    0.3931 |                        2 | incrementally useful complementary structure                |
| model_m   | anchor |           25 |                0.7284 |                   0.0094 |                    0.5748 |                       19 | incrementally useful complementary structure                |
| model_m   | P1     |            8 |                0.7638 |                   0.0114 |                    0.6232 |                        7 | incrementally useful complementary structure                |
| model_m   | P2     |            4 |                0.8241 |                  -0.0195 |                    0.7811 |                        2 | predictive residual with negative incremental accessibility |
| model_m   | P3     |            2 |                0.7533 |                   0.0081 |                    0.5683 |                        2 | incrementally useful complementary structure                |

The residual is described only as information not linearly recoverable from B1 under the fitted training-derived map; it is not evidence of new physiological information.
