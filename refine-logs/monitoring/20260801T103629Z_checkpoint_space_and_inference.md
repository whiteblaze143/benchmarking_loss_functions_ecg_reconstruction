# Checkpoint space and inference audit

**Timestamp:** 2026-08-01T10:36:29Z

- Queue: 20 completed, 1 running (`f_1000103_s42`), 459 pending, 0 stuck.
- Disk reserve after content-preserving deduplication: approximately 13 GiB free; queue disk guard healthy with a 5 GiB launch floor.
- Catalog: 20 inference-addressable models (820,219,136 exact remote bytes), 126 historical/error generations (7,656,540,798 remote logical bytes), and one 41,010,928-byte evaluator cache materialization currently local.
- Inference cohort audit: 20/20 current compatible models strictly loaded and executed on `/home/mithunmanivannan/data/ptb_xl/tensors/test/100.pt`; every output was finite with shape `[1, 12, 5000]`, and retained audit-cache bytes were zero.
- Fresh single-model witness: `f_1000102_s42`, checkpoint SHA-256 `322e7a169d8443bb94f25834839dace2d217946e12f905634f6ead5ec3cf6115`; finite output; dedicated cache pruned to zero.
- Space recovery: four byte-identical 311,496,307-byte result paths with SHA-256 `f72bb9c460af676e99b12e9841a0a426e8410d69bc6972b7374d443eda5f0cec` now share one filesystem inode. All paths and bytes remain available; approximately 0.87 GiB was recovered.
- Git caution: 3,433,682,498 compressed bytes are loose and unreachable from refs/reflogs, but they were not pruned because dangling Git objects may contain recoverable user history.
- Review independence: the requested Claude bridge is unavailable; this is a local integrity audit, not an independent scored review.
