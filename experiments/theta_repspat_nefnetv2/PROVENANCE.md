# Source provenance

- Nef-Net v2 source: copied byte-for-byte from `external/NEFNET-v2-main` on 2026-09-13 into `author_code/nefnet_v2`.
- Copy verification: `diff -qr` returned no differences immediately after copying.
- Aggregate SHA-256 of the sorted 67-file source checksum list: `ad8baf335a16ea7f363e52cf91b3e0c86f641cda4a8c234721b532633460aaad`.
- Author paper SHA-256: `b8ebbb5b309e47511cd9387a84dc038231cfb1bc960498e1ce21d035ddc06b01`.
- repSpat is not copied again. The frozen dependency is the existing `external/repspat-main` tree.
- Aggregate SHA-256 of its sorted 29-file checksum list: `ff66a6e82431bc1f19592dfe24e38312845bd2eb3fee0b30120c537eb0cdeb9e`.
- Repository snapshot containing both external trees: `e3a5969554d253de3908b29128b70356681d1101`.

The author Nef-Net tree is kept unchanged. All experiment-specific code lives in `src/`, `scripts/`, `configs/`, and `tests/`.
