import time
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
from scripts.evaluate_1lead_rdb_semiseg_blinded import *

print("Loading records...")
records, dataset_sha = load_cached_rdb_split(Path("data/rdb_wavelet_delineation_cache"), "test")
print("Total records:", len(records))

delineator = build_delineator(DEFAULT_CHECKPOINT, state="model_ema").to("cpu")
preprocessor = SignalPreprocessor()

print("Evaluating __original__ to fill cache...")
raw_signals = np.stack([r["signal"] for r in records], axis=0) # [N, 12, 5000]
predictions = predict_all(delineator, preprocessor, raw_signals, tuple(range(12)), 128)
score_relative_boundaries(records, predictions, tuple(range(12)), is_original=True)
print("Finished original")

model_id = "wavelet_ssl_1110000_s42_l0"
ckpt_path = Path("results/onelead_checkpoint_store/wavelet_ssl_1110000_s42_l0/checkpoint.pth")
obs_lead = 0
adapter = load_onelead_reconstruction_adapter(ckpt_path, obs_lead)

print("Forward passing model...")
reconstructed_list = []
for start_idx in range(0, len(raw_signals), 32):
    batch = torch.from_numpy(raw_signals[start_idx : start_idx + 32]).float()
    recon_batch_out = adapter(batch)
    reconstructed_list.append(recon_batch_out)

reconstructed = np.concatenate(reconstructed_list, axis=0)
missing_leads = tuple(i for i in range(12) if i != obs_lead)
print(f"Reconstructed shape: {reconstructed.shape}, predicting all...")

started = time.perf_counter()
model_preds = predict_all(delineator, preprocessor, reconstructed, missing_leads, 128)
print("Predict all took:", time.perf_counter() - started)

started = time.perf_counter()
rows = score_relative_boundaries(records, model_preds, missing_leads, is_original=False)
print("Score relative boundaries took:", time.perf_counter() - started)
