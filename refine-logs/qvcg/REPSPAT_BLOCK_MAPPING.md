# repSpat block-count mapping for Q-VCG M3

The original repSpat construction uses its selected CAHC spatial-neighborhood
parameter `m` and forms approximately `ceil(n_g / m)` K-means attribute blocks
inside cluster `g`.

The Q-VCG spatial stage does not select the same scalar `m`: it constructs a
24³ occupancy lattice with fixed 26-neighbor voxel connectivity, then requests 64
spatially constrained agglomerative clusters. Consequently, there is no learned
repSpat `m` to transfer exactly.

For this adaptation, M3 explicitly sets `m = 10` microstates per attribute block
and forms `ceil(n_g / 10)` K-means blocks from `(s, rho, kappa)` within each Q-VCG
domain. This preserves the original block-count functional rule, while the value
10 is an adaptation parameter rather than an exact replication of the source
spatial-neighborhood selection. It is exposed as `--cahc-neighborhood-size` and
recorded in `VCG_MOTIF_SUMMARY.json`.
