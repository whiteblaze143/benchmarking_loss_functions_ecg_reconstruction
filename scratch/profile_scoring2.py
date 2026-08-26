import time
import os
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
from scripts.evaluate_1lead_rdb_semiseg_blinded import *

print("Loading records...")
records, dataset_sha = load_cached_rdb_split(Path("data/rdb_wavelet_delineation_cache"), "test")

delineator = build_delineator(DEFAULT_CHECKPOINT, state="model_ema").to("cpu")
preprocessor = SignalPreprocessor()

# Let's generate a random noise signal to simulate a bad model
signals = np.random.randn(20, 12, 5000).astype(np.float32)
all_leads = tuple(range(12))

print("Predicting on noise...")
started = time.perf_counter()
predictions = predict_all(delineator, preprocessor, signals, all_leads, 32)
print("Predict all took:", time.perf_counter() - started)

lengths = []
for k, v in predictions.items():
    for b, arr in v.items():
        lengths.append(len(arr))
print("Max events per lead/boundary:", max(lengths))
print("Mean events per lead/boundary:", sum(lengths)/len(lengths))
