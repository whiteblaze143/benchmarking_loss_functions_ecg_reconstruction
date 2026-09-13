# Source provenance

- Nef-Net v2 source: copied byte-for-byte from `external/NEFNET-v2-main` on 2026-09-13 into `author_code/nefnet_v2`.
- Copy verification: `diff -qr` returned no differences immediately after copying.
- Aggregate SHA-256 of the sorted 67-file source checksum list: `ad8baf335a16ea7f363e52cf91b3e0c86f641cda4a8c234721b532633460aaad`.
- Author paper SHA-256: `b8ebbb5b309e47511cd9387a84dc038231cfb1bc960498e1ce21d035ddc06b01`.
- repSpat is not copied again. The frozen dependency is the existing `external/repspat-main` tree.
- Aggregate SHA-256 of its sorted 29-file checksum list: `ff66a6e82431bc1f19592dfe24e38312845bd2eb3fee0b30120c537eb0cdeb9e`.
- Repository snapshot containing both external trees: `e3a5969554d253de3908b29128b70356681d1101`.

The author Nef-Net tree is kept unchanged. All experiment-specific code lives in `src/`, `scripts/`, `configs/`, and `tests/`.

---

## N001 — PILOT_HYBRID_CONFIG (archived, not a candidate)

- Run: `results/pilot_hybrid_lr0.1_batch512_seed123/`
- Architecture: `nefnet_plus.layer` (GeoVT branch) ✓
- Optimizer: SGD lr=0.1, B=512 — sourced from `nef_net.yml` (wrong lineage for this architecture)
- Result: L1 plateau ~0.486 through epoch 28. Archived diagnostic only.

---

## N002 — panobench_fixed_triplet_randview3_geovt_seed123

- Run: `results/panobench_fixed_triplet_randview3_geovt_seed123/`
- Checkpoint SHA-256: to be filled after G001 passes
- Architecture: `nefnet_plus.layer` — PASS
- Optimizer: AdamW lr=1e-3 wd=0.01, MultiStep [50,100,150] γ=0.5, B=32, 200 epochs — PASS
- Training L1: 0.068 (epoch 0) → 0.004 (epoch 199) — PASS
- **Scientific contract** (verified against source):
  - `super_mode='optimization'`, `lead_num=3`: model is a **fixed three-slot GeoVT**, not Any-Pairs.
    The `else` branch in `nefnet_plus.layer.forward` uses dedicated `W_encoder_list[i]` modules
    for slots beyond 2. Runtime lead count is fixed at construction, not dynamic.
  - Third input: **random from PanoBench indices 2–43**, sampled per record per epoch via
    `rng.integers(2, 44)` in `ReleasedPanoBench.__getitem__`. This is NOT a fixed view-28
    (upstream optimization-mode PanoBench policy) and NOT clinical V3 (PTB-XL index 4,
    angle (95°, 15°)). The dedicated third slot was trained across randomly varying view
    identities — a mixed contract.
  - Normalization: global min-max across all 44 PanoBench channels before selecting 3.
    This is author-faithful (same as upstream PTB-XL preprocessing) but introduces mild
    leakage: unobserved views' extrema influence the scaling of the 3 observed inputs.
  - **ANY_PAIRS = NO. THIRD_INPUT = RANDOM_PANO_VIEW, NOT V3.**
- **CUDA/cuDNN note**: All convolutions failed with `ptrDesc->finalize()` during this run.
  Disabling cuDNN (`torch.backends.cudnn.enabled = False`) resolved the failure.
  The GPU reported 33 volatile uncorrected ECC errors at the time.
  **Cause is unconfirmed.** NVIDIA drivers guarantee backward-compatibility of older CUDA
  runtimes with newer drivers, so the PyTorch 2.6+cu124 / driver CUDA 13.0 combination
  is not definitively the cause. The ECC errors are the more likely explanation.
  Do not assert "driver update caused the failure" without machine-administrator confirmation.
  Before publication-grade use of N002 weights: reproduce forward passes from this checkpoint
  on a clean GPU or CPU and verify numerical outputs.
- **Status**: ARCHITECTURE=PASS, OPTIMIZATION=PASS, CONVERGENCE=PASS,
  FIXED_TRIPLET_PANO=PASS. NOT the final frozen Θ-repSpat representation.
  Pending: G001 held-out test evaluation.

---

## G001 — Held-out PanoBench test evaluation of N002

- Script: `scripts/eval_g001.py`
- Test split: `/data/mithunmanivannan/panobench/test/` (1030 records, never seen in training)
- Eval seed: 42 (distinct from training seed 123)
- Samples per record: 5 random (third-view, query) pairs
- Output: `results/g001_eval/` — pending completion
