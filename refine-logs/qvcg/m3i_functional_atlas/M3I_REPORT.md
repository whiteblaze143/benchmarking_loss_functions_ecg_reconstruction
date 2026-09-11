# M3I Functional Atlas Audit

## Verdict

The categorical connected-component quotient remains failed. The frozen primary
M3 graph nevertheless contains substantial label-free relational structure that
is not reducible to smooth physical proximity alone.

No community, spectral-cluster, or replacement motif labels were created.

## Frozen Scientific Objects

- 64 Q-VCG spatial domains.
- 2,016 observed biased IMQ MMD² values.
- 953 global-BH non-rejection edges and 1,063 rejected pairs.
- Frozen inputs were checksum-verified before analysis.

## Graph Topology

- Edge density: 0.4727; mean degree: 29.78.
- One connected component; diameter 2; mean shortest path 1.527.
- Every rejected pair is connected by a two-edge non-rejection path, explaining
  why connected-component closure destroys the direct pairwise contradictions.
- Clique number: 22, versus 64 nodes in the component.
- 44 maximal cliques; average local clustering coefficient: 0.851.

## Physical Versus Functional Geometry

- Physical centroid distance and MMD² are positively associated
  (Spearman rho = 0.603, p = 7.73e-200), so local dynamics are partly spatially
  organized.
- Nevertheless, 330 of the 504 top-quartile physical-distance pairs are BH
  non-rejected. These are candidate statistically supported nonlocal
  correspondences, not equivalence classes.
- Among 540 pairs whose occupied voxels touch under 26-neighbor adjacency, 414
  are BH rejected. Physical contact therefore does not imply similar local
  dynamics distributions.

Raw MMD magnitude must not substitute for the permutation decision: median
observed MMD² is 0.0961 among non-rejections and 0.0711 among rejections. The
pair-specific permutation null depends on domain sizes and block structure.

## Functional Embedding

Classical MDS was applied to the frozen squared Hilbert distance matrix
(`MMD²`), equivalently using `sqrt(MMD²)` as distances. The Gram spectrum has no
material negative eigenmass, supporting a Euclidean embedding interpretation.
The coordinates are an atlas visualization only and were not clustered.

## Claim Boundary

Supported for further audit: Q-VCG physical space contains numerous physically
distant domain pairs whose local-dynamics distributions are not distinguished by
the primary repSpat test.

Not supported: the 64 domains form one motif; non-rejection proves equivalence;
or the current graph defines a categorical tokenizer.

M3R must finish before attributing this topology to repSpat rather than the
adapted block construction and Monte Carlo settings.

