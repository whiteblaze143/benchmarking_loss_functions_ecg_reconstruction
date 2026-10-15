"""Sequential block-aligned 16MB streaming converter for EchoNext datasets.

Reads continuously from byte 0 with zero unaligned seeks.
Converts float64 to float32 on the fly and writes valid npy files to local NVMe SSD.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import time
import numpy as np

DEST_DIR = Path("/home/mithunmanivannan/data/echonext")
SRC_DIR = Path("/data/mithunmanivannan/echonext")

TARGETS = [
    {
        "name": "EchoNext_val_waveforms.npy",
        "shape": (4626, 1, 2500, 12),
    },
    {
        "name": "EchoNext_train_waveforms.npy",
        "shape": (72475, 1, 2500, 12),
    },
]


def get_avail_gb() -> float:
    s = os.statvfs("/")
    return s.f_bavail * s.f_frsize / (1024 ** 3)


def stream_dataset(target: dict):
    name = target["name"]
    shape = target["shape"]
    src = SRC_DIR / name
    dst = DEST_DIR / name

    if dst.exists():
        try:
            m = np.load(dst, mmap_mode="r")
            if m.shape == shape and m.dtype == np.float32:
                print(f"[SKIP] {name} already staged as float32 ({dst.stat().st_size / (1024**3):.2f} GB).")
                sys.stdout.flush()
                return
        except Exception:
            pass

    dst_tmp = dst.with_name(name + ".tmp")
    if dst_tmp.exists():
        dst_tmp.unlink()

    n_samples, c, length, leads = shape
    bytes_per_sample_in = c * length * leads * 8   # 240,000 bytes
    bytes_per_sample_out = c * length * leads * 4  # 120,000 bytes
    total_out_bytes = n_samples * bytes_per_sample_out
    total_in_bytes = n_samples * bytes_per_sample_in

    print(f"\n[STREAM] Streaming {name} ({total_out_bytes / (1024**3):.2f} GB float32) sequentially via 16MB-aligned reads...")
    sys.stdout.flush()
    t0 = time.time()

    read_block = 16 * 1024 * 1024  # 16 MB block-aligned
    samples_written = 0
    leftover = b""
    first_read = True
    last_log = t0

    with open(dst_tmp, "wb") as f_out:
        np.lib.format.write_array_header_1_0(
            f_out,
            {"descr": "<f4", "fortran_order": False, "shape": shape}
        )

        with open(src, "rb") as f_in:
            while True:
                chunk = f_in.read(read_block)
                if not chunk:
                    break
                if first_read:
                    chunk = chunk[128:]  # strip 128-byte header from first 16MB block
                    first_read = False

                buf = leftover + chunk
                n_avail = len(buf) // bytes_per_sample_in
                if n_avail > 0:
                    usable = n_avail * bytes_per_sample_in
                    raw_data = buf[:usable]
                    leftover = buf[usable:]

                    f32 = np.frombuffer(raw_data, dtype="<f8").astype(np.float32).reshape(n_avail, c, length, leads)
                    f_out.write(f32.tobytes())
                    samples_written += n_avail
                else:
                    leftover = buf

                now = time.time()
                if now - last_log >= 5.0 or samples_written == n_samples:
                    pct = 100.0 * samples_written / n_samples
                    gb_done = (samples_written * bytes_per_sample_out) / (1024**3)
                    gb_tot = total_out_bytes / (1024**3)
                    mb_s = (samples_written * bytes_per_sample_in / (1024 * 1024)) / (now - t0 + 1e-6)
                    eta_s = (n_samples - samples_written) * (now - t0) / (samples_written + 1e-6)
                    print(f"  -> {gb_done:.2f} / {gb_tot:.2f} GB ({pct:.1f}%) | {mb_s:.1f} MB/s | Elapsed: {now - t0:.1f}s | ETA: {eta_s:.1f}s")
                    sys.stdout.flush()
                    last_log = now

            if len(leftover) >= bytes_per_sample_in:
                n_rem = len(leftover) // bytes_per_sample_in
                f32 = np.frombuffer(leftover[: n_rem * bytes_per_sample_in], dtype="<f8").astype(np.float32).reshape(n_rem, c, length, leads)
                f_out.write(f32.tobytes())
                samples_written += n_rem

        f_out.flush()
        os.fsync(f_out.fileno())

    dst_tmp.rename(dst)
    dur = time.time() - t0
    rate = (total_in_bytes / (1024 * 1024)) / (dur + 1e-6)
    print(f"✓ {name} successfully staged in {dur:.1f}s ({rate:.1f} MB/s).")
    print(f"Available disk space on /: {get_avail_gb():.2f} GB\n")
    sys.stdout.flush()


def verify_all():
    print("\n[VERIFICATION] Auditing all local EchoNext files in", DEST_DIR)
    sys.stdout.flush()
    all_ok = True

    checks = [
        ("EchoNext_test_tabular_features.npy", 304880),
        ("EchoNext_val_tabular_features.npy", 259184),
        ("EchoNext_train_tabular_features.npy", 4058728),
        ("echonext_metadata_100k.csv", 17493749),
    ]
    for fname, exp_sz in checks:
        p = DEST_DIR / fname
        if p.exists() and p.stat().st_size == exp_sz:
            print(f"  ✓ {fname:36s} ({exp_sz} bytes, OK)")
        else:
            print(f"  ❌ {fname:36s} missing or size mismatch!")
            all_ok = False

    wf_checks = [
        ("EchoNext_test_waveforms.npy", (5442, 1, 2500, 12)),
        ("EchoNext_val_waveforms.npy", (4626, 1, 2500, 12)),
        ("EchoNext_train_waveforms.npy", (72475, 1, 2500, 12)),
    ]
    for fname, exp_shape in wf_checks:
        p = DEST_DIR / fname
        try:
            arr = np.load(p, mmap_mode="r")
            if arr.shape == exp_shape and arr.dtype == np.float32:
                print(f"  ✓ {fname:36s} shape={arr.shape} dtype={arr.dtype} ({p.stat().st_size / (1024**3):.2f} GB, OK)")
            else:
                print(f"  ❌ {fname:36s} shape/dtype mismatch: {arr.shape}, {arr.dtype}")
                all_ok = False
        except Exception as e:
            print(f"  ❌ Error reading {fname}: {e}")
            all_ok = False

    print(f"\n[DISK] Final Available Space on /: {get_avail_gb():.2f} GB")
    if all_ok:
        print("🎉 ALL ECHONEXT DATASETS ARE 100% STAGED & VERIFIED ON LOCAL NVMe SSD!")
    else:
        print("❌ Dataset verification failed.")
        sys.exit(1)


def main():
    print(f"[DISK] Initial available space on /: {get_avail_gb():.2f} GB")
    sys.stdout.flush()

    for target in TARGETS:
        stream_dataset(target)

    verify_all()


if __name__ == "__main__":
    main()
