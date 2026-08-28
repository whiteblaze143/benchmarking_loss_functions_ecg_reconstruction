import json, hashlib
from pathlib import Path
from scripts.wavelet_ssl_queue import current_input_fingerprints, command_options, resolve_path, sha256_file

ROOT = Path(".").resolve()
with open("refine-logs/wavelet_ssl_1110000/full/manifest.json") as f:
    base_m = json.load(f)

# Copy base metadata
print("Base manifest metadata keys:")
for k, v in base_m.items():
    if k != "jobs":
        print(f"  {k}: {v}")
