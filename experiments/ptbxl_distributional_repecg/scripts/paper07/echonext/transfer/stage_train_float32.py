"""Dedicated automated script to stage EchoNext val and train waveforms in float32 on local NVMe SSD.

1. Waits for EchoNext_val_waveforms.npy copy to reach 1,110,240,128 bytes.
2. Converts local EchoNext_val_waveforms.npy to float32 (reclaiming 555 MB).
3. Streams EchoNext_train_waveforms.npy from NFS in 15.36 MB chunks (64 ECGs),
   converts to float32 on the fly, and writes to local SSD.
4. Validates final array shapes, dtypes, and free disk space.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import time
import numpy as np

DEST_DIR = Path("/home/mithunmanivannan/data/echonext")
SRC_DIR = Path("/data/mithunmanivannan/echonext")

VAL_RAW = DEST_DIR / "EchoNext_val_waveforms.npy"
VAL_EXPECTED_RAW_BYTES = 1110240128
VAL_SHAPE = (4626, 1, 2500, 12)

TRAIN_SRC = SRC_DIR / "EchoNext_train_waveforms.npy"
TRAIN_DST = DEST_DIR / "EchoNext_train_waveforms.npy"
TRAIN_SHAPE = (72475, 1, 2500, 12)


def get_avail_gb() -> float:
    s = os.statvfs("/")
    return s.f_bavail * s.f_frsize / (1024 ** 3)


def wait_and_convert_val():
    print("[VAL] Waiting for EchoNext_val_waveforms.npy copy to complete...")
    sys.stdout.flush()

    while True:
        if VAL_RAW.exists():
            sz = VAL_RAW.stat().st_size
            if sz == VAL_EXPECTED_RAW_BYTES:
                print(f"[VAL] Raw validation file complete ({sz} bytes).")
                break
            else:
                pct = 100.0 * sz / VAL_EXPECTED_RAW_BYTES
                print(f"  ... {sz / (1024**2):.1f} / {VAL_EXPECTED_RAW_BYTES / (1024**2):.1f} MB ({pct:.1f}%)")
                sys.stdout.flush()
        time.sleep(10)

    # Convert val to float32 locally
    print("[VAL] Converting local validation waveforms to float32...")
    sys.stdout.flush()
    t0 = time.time()
    arr = np.load(VAL_RAW).astype(np.float32)
    tmp_val = VAL_RAW.with_name("EchoNext_val_waveforms_f32.tmp.npy")
    np.save(tmp_val, arr)
    tmp_val.rename(VAL_RAW)
    print(f"[VAL] ✓ Converted to float32 in {time.time()-t0:.2f}s (new size: {VAL_RAW.stat().st_size / (1024**2):.1f} MB)")
    print(f"[DISK] Available space: {get_avail_gb():.2f} GB")
    sys.stdout.flush()


def stream_train_to_float32():
    if TRAIN_DST.exists():
        try:
            m = np.load(TRAIN_DST, mmap_mode="r")
            if m.shape == TRAIN_SHAPE and m.dtype == np.float32:
                print(f"[TRAIN] ✓ EchoNext_train_waveforms.npy already staged as float32 ({TRAIN_DST.stat().st_size / (1024**3):.2f} GB).")
                sys.stdout.flush()
                return
        except Exception:
            pass

    dst_tmp = TRAIN_DST.with_name("EchoNext_train_waveforms.npy.tmp")
    if dst_tmp.exists():
        dst_tmp.unlink()

    n_samples, c, length, leads = TRAIN_SHAPE
    bytes_per_sample_in = c * length * leads * 8   # float64 = 240,000 bytes
    bytes_per_sample_out = c * length * leads * 4  # float32 = 120,000 bytes
    total_out_bytes = n_samples * bytes_per_sample_out  # 8.697 GB

    print(f"\n[TRAIN] Streaming {n_samples} training records from NFS and converting to float32 ({total_out_bytes / (1024**3):.2f} GB)...")
    sys.stdout.flush()
    t0 = time.time()

    with open(dst_tmp, "wb") as f_out:
        np.lib.format.write_array_header_1_0(
            f_out,
            {"descr": "<f4", "fortran_order": False, "shape": TRAIN_SHAPE}
        )

        with open(TRAIN_SRC, "rb") as f_in:
            f_in.seek(128)  # Skip 128-byte header
            chunk_samples = 64  # 64 records = 15.36 MB read (optimal for NFS)
            samples_copied = 0
            last_log = t0

            while samples_copied < n_samples:
                n_curr = min(chunk_samples, n_samples - samples_copied)
                raw_bytes = f_in.read(n_curr * bytes_per_sample_in)
                if not raw_bytes:
                    break
                arr_f32 = np.frombuffer(raw_bytes, dtype="<f8").astype(np.float32).reshape(n_curr, c, length, leads)
                f_out.write(arr_f32.tobytes())
                samples_copied += n_curr

                now = time.time()
                if now - last_log >= 10.0 or samples_copied == n_samples:
                    pct = 100.0 * samples_copied / n_samples
                    gb_done = (samples_copied * bytes_per_sample_out) / (1024**3)
                    gb_tot = total_out_bytes / (1024**3)
                    mb_s = (samples_copied * bytes_per_sample_in / (1024 * 1024)) / (now - t0 + 1e-6)
                    eta_sec = (total_out_bytes - samples_copied * bytes_per_sample_out) / ((samples_copied * bytes_per_sample_out) / (now - t0 + 1e-6))
                    print(f"  -> {gb_done:.2f} / {gb_tot:.2f} GB ({pct:.1f}%) | {mb_s:.1f} MB/s | Elapsed: {now - t0:.1f}s | ETA: {eta_sec/60:.1f}m")
                    sys.stdout.flush()
                    last_log = now

            f_out.flush()
            os.fsync(f_out.fileno())

    dst_tmp.rename(TRAIN_DST)
    total_time = time.time() - t0
    final_mb_s = (n_samples * bytes_per_sample_in / (1024 * 1024)) / (total_time + 1e-6)
    print(f"[TRAIN] ✓ Training waveforms successfully staged as float32 in {total_time/60:.2f}m ({final_mb_s:.1f} MB/s)")
    sys.stdout.flush()


def verify_all():
    print("\n[VERIFY] Final validation of local dataset on NVMe SSD:")
    sys.stdout.flush()
    for name, shape in [
        ("EchoNext_test_waveforms.npy", (5442, 1, 2500, 12)),
        ("EchoNext_val_waveforms.npy", VAL_SHAPE),
        ("EchoNext_train_waveforms.npy", TRAIN_SHAPE),
    ]:
        p = DEST_DIR / name
        arr = np.load(p, mmap_mode="r")
        assert arr.shape == shape, f"Shape mismatch for {name}: {arr.shape} != {shape}"
        assert arr.dtype == np.float32, f"dtype mismatch for {name}: {arr.dtype}"
        print(f"  ✓ {name:32s} shape={arr.shape} dtype={arr.dtype} ({p.stat().st_size / (1024**3):.2f} GB)")

    print(f"\n[DISK] Final Available Space on /: {get_avail_gb():.2f} GB")
    print("🎉 ECHONEXT DATASET IS 100% STAGED & VERIFIED ON LOCAL NVMe SSD!")


def main():
    wait_and_convert_val()
    stream_train_to_float32()
    verify_all()


if __name__ == "__main__":
    main()
