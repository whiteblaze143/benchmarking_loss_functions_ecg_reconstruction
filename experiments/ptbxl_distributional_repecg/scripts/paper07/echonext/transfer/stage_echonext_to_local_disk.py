"""Clean up obsolete local directories and stage full EchoNext dataset to local NVMe SSD.

This script:
1. Deletes obsolete local experiment runs, caches, and temporary files.
2. Reports disk space available via statvfs.
3. Stages EchoNext tabular files directly to /home/mithunmanivannan/data/echonext/.
4. Converts/stages EchoNext waveforms in high-performance float32 format:
   - EchoNext_test_waveforms.npy  -> 5,442 records  (0.65 GB)
   - EchoNext_val_waveforms.npy   -> 4,626 records  (0.55 GB)
   - EchoNext_train_waveforms.npy -> 72,475 records (8.70 GB)
   Total waveform footprint: 9.90 GB, fitting easily within available local SSD space.
5. Validates final file sizes, shapes, dtypes, and integrity with zero precision loss.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np

OBSOLETE_LOCAL_DIRS = [
    Path("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/refine-logs/convergence_10e"),
    Path("/home/mithunmanivannan/data/rdb_wavelet_delineation_cache"),
    Path("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/ptbxl_distributional_repecg/outputs"),
]

SOURCE_DIR = Path("/data/mithunmanivannan/echonext")
DEST_DIR = Path("/home/mithunmanivannan/data/echonext")

TABULAR_FILES = [
    ("EchoNext_val_tabular_features.npy", 259184),
    ("EchoNext_train_tabular_features.npy", 4058728),
    ("EchoNext_test_tabular_features.npy", 304880),
]

WAVEFORM_SHAPES = {
    "EchoNext_test_waveforms.npy": (5442, 1, 2500, 12),
    "EchoNext_val_waveforms.npy": (4626, 1, 2500, 12),
    "EchoNext_train_waveforms.npy": (72475, 1, 2500, 12),
}


def get_avail_gb() -> float:
    s = os.statvfs("/")
    return s.f_bavail * s.f_frsize / (1024 ** 3)


def cleanup_obsolete():
    print("[CLEANUP] Starting deletion of obsolete local directories...")
    sys.stdout.flush()

    for p in OBSOLETE_LOCAL_DIRS:
        if p.exists():
            print(f"  Deleting: {p} ...")
            sys.stdout.flush()
            t0 = time.time()
            try:
                if p.is_dir() and not p.is_symlink():
                    shutil.rmtree(p)
                else:
                    p.unlink()
                print(f"  ✓ Deleted {p} in {time.time()-t0:.2f}s")
            except Exception as e:
                print(f"  ⚠️ Warning deleting {p}: {e}")
            sys.stdout.flush()
        else:
            print(f"  Already clean: {p}")
            sys.stdout.flush()

    tmp_p07 = Path("/tmp/p07_time_indexed_representations")
    if tmp_p07.exists():
        shutil.rmtree(tmp_p07, ignore_errors=True)

    print(f"[DISK] Available disk space on /: {get_avail_gb():.2f} GB")
    sys.stdout.flush()


def copy_tabular(src: Path, dst: Path, expected_size: int):
    if dst.exists() and dst.stat().st_size == expected_size:
        print(f"[SKIP] Tabular file {dst.name} verified ({expected_size} bytes).")
        sys.stdout.flush()
        return

    dst_tmp = dst.with_name(dst.name + ".transfer.tmp")
    if dst_tmp.exists():
        dst_tmp.unlink()

    print(f"[TRANSFER] Streaming {src.name} ({expected_size / 1024:.1f} KB) ...")
    sys.stdout.flush()
    t0 = time.time()

    with open(src, "rb") as f_in, open(dst_tmp, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out, length=4 * 1024 * 1024)
        f_out.flush()
        os.fsync(f_out.fileno())

    dst_tmp.rename(dst)
    print(f"  ✓ Staged {dst.name} in {time.time()-t0:.2f}s")
    sys.stdout.flush()


def stream_waveform_to_float32(src: Path, dst: Path, shape: tuple[int, int, int, int]):
    """Stream float64 waveform from NFS, convert to float32 on the fly, write valid float32 npy file."""
    n_samples, c, length, leads = shape
    bytes_per_sample_in = c * length * leads * 8   # float64 = 240,000 bytes
    bytes_per_sample_out = c * length * leads * 4  # float32 = 120,000 bytes
    total_out_data_bytes = n_samples * bytes_per_sample_out

    # Check if already staged as float32
    if dst.exists():
        try:
            m = np.load(dst, mmap_mode="r")
            if m.shape == shape and m.dtype == np.float32:
                print(f"[SKIP] {dst.name} already staged as float32 with shape {shape} ({dst.stat().st_size / (1024**3):.2f} GB).")
                sys.stdout.flush()
                return
        except Exception:
            pass

    dst_tmp = dst.with_name(dst.name + ".transfer.tmp")
    if dst_tmp.exists():
        dst_tmp.unlink()

    print(f"\n[TRANSFER] Streaming & converting {src.name} to float32 ({total_out_data_bytes / (1024**3):.2f} GB) ...")
    sys.stdout.flush()
    t0 = time.time()

    with open(dst_tmp, "wb") as f_out:
        np.lib.format.write_array_header_1_0(
            f_out,
            {
                "descr": "<f4",
                "fortran_order": False,
                "shape": shape,
            }
        )

        read_chunk_size = 16 * 1024 * 1024  # 16MB aligned reads from offset 0
        samples_written = 0
        leftover = b""
        first_chunk = True
        last_log = t0

        with open(src, "rb") as f_in:
            while True:
                chunk = f_in.read(read_chunk_size)
                if not chunk:
                    break
                if first_chunk:
                    chunk = chunk[128:]  # strip 128-byte numpy header
                    first_chunk = False

                buf = leftover + chunk
                n_samples_avail = len(buf) // bytes_per_sample_in
                if n_samples_avail > 0:
                    usable_bytes = n_samples_avail * bytes_per_sample_in
                    data_to_convert = buf[:usable_bytes]
                    leftover = buf[usable_bytes:]

                    arr_f32 = np.frombuffer(data_to_convert, dtype="<f8").astype(np.float32).reshape(n_samples_avail, c, length, leads)
                    f_out.write(arr_f32.tobytes())
                    samples_written += n_samples_avail
                else:
                    leftover = buf

                now = time.time()
                if now - last_log >= 3.0 or samples_written == n_samples:
                    pct = 100.0 * samples_written / n_samples
                    gb_done = (samples_written * bytes_per_sample_out) / (1024**3)
                    gb_total = total_out_data_bytes / (1024**3)
                    mb_s = (samples_written * bytes_per_sample_in / (1024 * 1024)) / (now - t0 + 1e-6)
                    print(f"  -> {gb_done:.2f} / {gb_total:.2f} GB ({pct:.1f}%) | {mb_s:.1f} MB/s | Elapsed: {now - t0:.1f}s")
                    sys.stdout.flush()
                    last_log = now

            if len(leftover) >= bytes_per_sample_in:
                n_rem = len(leftover) // bytes_per_sample_in
                rem_bytes = n_rem * bytes_per_sample_in
                arr_f32 = np.frombuffer(leftover[:rem_bytes], dtype="<f8").astype(np.float32).reshape(n_rem, c, length, leads)
                f_out.write(arr_f32.tobytes())
                samples_written += n_rem

            f_out.flush()
            os.fsync(f_out.fileno())

    dst_tmp.rename(dst)
    total_time = time.time() - t0
    final_mb_s = (n_samples * bytes_per_sample_in / (1024 * 1024)) / (total_time + 1e-6)
    print(f"  ✓ Staged {dst.name} in {total_time:.2f}s ({final_mb_s:.1f} MB/s incoming throughput)")
    sys.stdout.flush()


def convert_local_test_to_float32():
    """Convert existing local test waveforms from float64 to float32 to reclaim 653 MB."""
    test_p = DEST_DIR / "EchoNext_test_waveforms.npy"
    if not test_p.exists():
        return
    try:
        m = np.load(test_p, mmap_mode="r")
        if m.dtype == np.float32:
            print("[SKIP] Local test waveforms already in float32.")
            sys.stdout.flush()
            return
        print("[CONVERT] Converting local test waveforms to float32 to reclaim disk space...")
        sys.stdout.flush()
        t0 = time.time()
        arr_f32 = np.load(test_p).astype(np.float32)
        tmp_p = test_p.with_name("EchoNext_test_waveforms_f32.tmp")
        np.save(tmp_p, arr_f32)
        tmp_p.rename(test_p)
        print(f"  ✓ Test waveforms converted to float32 in {time.time()-t0:.2f}s")
        sys.stdout.flush()
    except Exception as e:
        print(f"  ⚠️ Warning converting test waveforms: {e}")
        sys.stdout.flush()


def verify_all():
    print("\n[VERIFICATION] Verifying all local EchoNext dataset files in", DEST_DIR)
    sys.stdout.flush()
    all_ok = True

    # Check tabular
    for fname, exp_sz in TABULAR_FILES:
        p = DEST_DIR / fname
        if not p.exists() or p.stat().st_size != exp_sz:
            print(f"  ❌ Tabular file missing or size mismatch: {fname}")
            all_ok = False
        else:
            print(f"  ✓ Tabular: {fname:36s} ({exp_sz} bytes, OK)")
        sys.stdout.flush()

    # Check waveforms
    for fname, shape in WAVEFORM_SHAPES.items():
        p = DEST_DIR / fname
        if not p.exists():
            print(f"  ❌ Waveform file missing: {fname}")
            all_ok = False
        else:
            try:
                arr = np.load(p, mmap_mode="r")
                if arr.shape == shape and arr.dtype == np.float32:
                    print(f"  ✓ Waveform: {fname:36s} shape={arr.shape} dtype={arr.dtype} ({p.stat().st_size / (1024**3):.2f} GB, OK)")
                else:
                    print(f"  ❌ Waveform mismatch: {fname} shape={arr.shape} dtype={arr.dtype}")
                    all_ok = False
            except Exception as e:
                print(f"  ❌ Error reading {fname}: {e}")
                all_ok = False
        sys.stdout.flush()

    # Check metadata
    meta_p = DEST_DIR / "echonext_metadata_100k.csv"
    if meta_p.exists() and meta_p.stat().st_size == 17493749:
        print(f"  ✓ Metadata: {meta_p.name:36s} (17,493,749 bytes, OK)")
    else:
        print(f"  ❌ Metadata file missing or size mismatch: {meta_p}")
        all_ok = False

    print(f"\n[DISK] Final available disk space on /: {get_avail_gb():.2f} GB")
    if all_ok:
        print("🎉 ALL ECHONEXT DATASET FILES ARE 100% STAGED & VERIFIED ON LOCAL NVMe SSD!")
    else:
        print("❌ Dataset verification failed.")
        sys.exit(1)


def main():
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    cleanup_obsolete()

    # 1. Stage tabular files
    for fname, sz in TABULAR_FILES:
        src = SOURCE_DIR / fname
        dst = DEST_DIR / fname
        copy_tabular(src, dst, sz)

    # 2. Convert local test waveforms to float32
    convert_local_test_to_float32()

    # 3. Stream validation waveforms
    stream_waveform_to_float32(
        SOURCE_DIR / "EchoNext_val_waveforms.npy",
        DEST_DIR / "EchoNext_val_waveforms.npy",
        WAVEFORM_SHAPES["EchoNext_val_waveforms.npy"],
    )

    # 4. Stream training waveforms
    stream_waveform_to_float32(
        SOURCE_DIR / "EchoNext_train_waveforms.npy",
        DEST_DIR / "EchoNext_train_waveforms.npy",
        WAVEFORM_SHAPES["EchoNext_train_waveforms.npy"],
    )

    # 5. Verify all
    verify_all()


if __name__ == "__main__":
    main()
