# Invalid Pilot Ledger

## Paper 2 pre-exact-control grid (2026-09-16)

The first shared-stream development grid was stopped after kernel epoch 5 and
is not eligible for model selection or scientific interpretation. Its inputs
violated the frozen Paper 2 control contract in two ways:

1. the Gaussian draw was transformed by the target covariance square root but
   was not whitened by its own realized sample covariance, so its moments
   matched only in expectation;
2. the `moments` control contained mean and standard deviation (16 values per
   phase cell), not mean and the full non-duplicated sample covariance (44
   values per phase cell).

The partial log and input arrays are retained under
`outputs/paper02_kernel_mean/invalid_pilot_pre_exact_controls/`. No cell
completed, and no checkpoint or prediction file from this pilot was promoted.

The replacement implementation performs float64 realized moment matching,
fails above `1e-8` mean or covariance error, uses the 44-dimensional full
moment control, and adds the required whitened linear-kernel mean safeguard.
