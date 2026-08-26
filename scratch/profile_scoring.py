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

print("Predicting all...")
started = time.perf_counter()
signals = np.stack([r["signal"] for r in records], axis=0) # [N, 12, 5000]
all_leads = tuple(range(12))
predictions = predict_all(delineator, preprocessor, signals, all_leads, 128)
print("Predict all took:", time.perf_counter() - started)

print("Scoring relative boundaries (is_original=True)...")
started = time.perf_counter()
rows = score_relative_boundaries(records, predictions, all_leads, is_original=True)
print("Scoring original took:", time.perf_counter() - started)

print("Scoring relative boundaries (is_original=False)...")
started = time.perf_counter()
rows = score_relative_boundaries(records, predictions, all_leads, is_original=False)
print("Scoring models took:", time.perf_counter() - started)
