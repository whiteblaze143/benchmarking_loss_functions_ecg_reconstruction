# Theta-repSpat with Nef-Net v2

This isolated experiment evaluates a single claim: whether a frozen Nef-Net v2 query operator can turn `{I, II, V3}` into a canonical eight-view ECG representation whose induced temporal geometry is more stable across observed-view subsets without erasing physiological distinctions.

The primary representation is `[I, II, predicted V1, predicted V2, V3, predicted V4, predicted V5, predicted V6]`. Observed leads are copied exactly. The Nef-Net architecture and repSpat inference procedure are not modified.

## Status

- Author Nef-Net source copied and verified.
- Existing author repSpat source reused by recorded path and checksum.
- Canonical panorama extraction and temporal AnnData adapter implemented.
- Unit/interface tests implemented.
- From-scratch PanoBench trainer implemented against the downloaded release at `/data/mithunmanivannan/panobench`.
- Seed-123 training is running at batch 512, using approximately 39.9/41.0 GiB and 90-100% GPU utilization during sampled compute intervals.

## Train from scratch

```bash
PYTHONPATH=src:/home/mithunmanivannan/.venv/lib/python3.12/site-packages \
python3 scripts/train_panobench.py --output results/train_seed123
```

The frozen training contract uses the official GeoVT architecture (`network.nefnet_plus.layer`) with the internally consistent `default.py` lineage: L1 reconstruction loss, AdamW with weight decay 0.01, learning rate `1e-3`, batch 32, 200 epochs, milestones `[50,100,150]`, gamma 0.5, and seed 123. This is an externally specified leakage-free training protocol, not a claim of exact reproduction of the author's solver. Any-Pairs sampling retains I and II as anchors. Each downloaded 250 Hz record undergoes 2x Fourier resampling and a seeded valid 4608-sample crop.

The discarded hybrid pilot (`lr=0.1`, batch 512) is preserved under `results/pilot_hybrid_lr0.1_batch512_seed123` and is not a candidate model.

## Extract panoramas

With dependencies and a raw author `state_dict` available:

```bash
PYTHONPATH=src:/home/mithunmanivannan/.venv/lib/python3.12/site-packages \
python3 scripts/extract_panorama.py \
  --input /absolute/path/eight_lead_records.npy \
  --checkpoint results/train_seed123/model_epoch_149.pt \
  --output results/panoramas.npz
```

The input order must be `I, II, V1, V2, V3, V4, V5, V6`. The script deliberately rejects missing weights, wrapped/ambiguous checkpoints, incorrect shapes, unavailable requested CUDA, and incompatible time lengths.
