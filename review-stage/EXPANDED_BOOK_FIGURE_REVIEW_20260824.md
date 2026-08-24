# Expanded Book Figure Review — 2026-08-24

## Scope

Manually reviewed the five data-driven figure families added in Chapters 18–22 after recomputing static equivalents from the same live data frames. Chapter 23 is tabular and has no figure.

| Chapter | Figure question | Manual finding | Decision |
|---|---|---|---|
| 18 | Which seed-42 architecture grids are complete? | 160/155/123 bars and 160 reference line are legible; the missingness gradient is unambiguous. | Retain. |
| 19 | What does the complete U-Net validation landscape show? | The Pearson–MSE trade-off is visible; kernel colors overlap substantially. This supports screening heterogeneity, not a kernel winner. | Retain with validation-only warning. |
| 20 | How far does clinical evaluation lag checkpoint availability? | 160/37/0 clinical bars clearly expose differential coverage. | Retain; this is the key anti-misinterpretation figure. |
| 21 | What one-lead checkpoints exist by architecture and observed lead? | Lead I/II counts are balanced; long architecture labels remain readable with angled labels. | Retain as inventory, not performance. |
| 22 | Do completed screens expose a reconstruction–delineation trade-off? | Two broad performance bands are visible and the top point is not isolated enough to justify a winner claim. | Retain as exploratory; hover metadata carries configuration identity. |

## Checks

- Axes, units/metric names, titles, category labels, and denominators were legible.
- No figure treats an absent clinical row as zero performance.
- No figure pools four-mask, seven-mask, three-lead, and one-lead results.
- No validation figure is described as test evidence.
- Interactive Plotly figures have equivalent source tables printed or available in the same chapter.
