#!/usr/bin/env python3
"""Download one quality-screened synchronized II/V5/PLETH window per VitalDB case.

The VitalDB bulk files contain many unrelated tracks and are large.  This
acquisition script uses the public FHIR API to retain only the three waveform
tracks needed for the registered ECG-II + PPG -> ECG-V5 falsification test.
It is resumable: a valid per-case ``.npz`` is never downloaded twice.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import io
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np


BASE_URL = "https://api.vitaldb.net"
TRACKS = ("SNUADC/ECG_II", "SNUADC/ECG_V5", "SNUADC/PLETH")
SAMPLE_RATE_HZ = 500
WINDOW_SECONDS = 10


def _request(url: str, retries: int = 4) -> bytes:
    for attempt in range(retries):
        try:
            with urlopen(url, timeout=90) as response:
                payload = response.read()
                # The public ``/trks`` endpoint serves a gzip payload without a
                # Content-Encoding header. Detect the file signature rather
                # than relying on transport metadata.
                return gzip.decompress(payload) if payload[:2] == b"\x1f\x8b" else payload
        except HTTPError as error:
            if error.code == 404:
                raise ValueError(f"source observation unavailable: {url}") from error
            if attempt + 1 == retries:
                raise RuntimeError(f"request failed after {retries} attempts: {url}") from error
            time.sleep(2**attempt)
        except (URLError, TimeoutError) as error:
            if attempt + 1 == retries:
                raise RuntimeError(f"request failed after {retries} attempts: {url}") from error
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def _json(url: str) -> dict[str, object]:
    return json.loads(_request(url))


def _track_cases(output: Path) -> dict[int, dict[str, str]]:
    cache = output / "source_tracks.csv"
    if not cache.exists():
        temporary = cache.with_suffix(".tmp")
        temporary.write_bytes(_request(f"{BASE_URL}/trks"))
        os.replace(temporary, cache)
    grouped: dict[int, dict[str, str]] = defaultdict(dict)
    raw = cache.read_bytes()
    text = (gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw).decode("utf-8")
    with io.StringIO(text, newline="") as handle:
        for row in csv.DictReader(handle):
            if row["tname"] in TRACKS:
                grouped[int(row["caseid"])][row["tname"]] = row["tid"]
    return {case_id: tracks for case_id, tracks in grouped.items() if set(tracks) == set(TRACKS)}


def _subject_id(case_id: int) -> str:
    encounter = _json(f"{BASE_URL}/fhir/Encounter/{case_id}")
    reference = str(dict(encounter.get("subject", {})).get("reference", ""))
    if not reference.startswith("Patient/"):
        raise ValueError(f"VitalDB case {case_id} lacks a patient reference")
    return reference.split("/", 1)[1]


def _track_period(case_id: int, track: str) -> tuple[datetime, datetime]:
    summary_query = urlencode({"encounter": case_id, "code": track, "_summary": "true"})
    summary = _json(f"{BASE_URL}/fhir/Observation?{summary_query}")
    summary_entries = list(summary.get("entry", []))
    if len(summary_entries) != 1:
        raise ValueError(f"case {case_id} has {len(summary_entries)} summaries for {track}")
    period = dict(dict(summary_entries[0])["resource"].get("effectivePeriod", {}))
    start = datetime.fromisoformat(str(period["start"]).replace("Z", "+00:00"))
    end = datetime.fromisoformat(str(period["end"]).replace("Z", "+00:00"))
    return start, end


def _track_values(case_id: int, track: str, slice_start: datetime) -> np.ndarray:
    slice_end = slice_start + timedelta(seconds=60)
    def stamp(value: datetime) -> str:
        return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    query = urlencode(
        {
            "encounter": case_id,
            "code": track,
            "date": [f"ge{stamp(slice_start)}", f"le{stamp(slice_end)}"],
        },
        doseq=True,
    )
    bundle = _json(f"{BASE_URL}/fhir/Observation?{query}")
    entries = list(bundle.get("entry", []))
    # The API includes the next one-minute segment when the requested end
    # equals its boundary.  The first response is the registered first-minute
    # slice; we subsequently choose a complete 10-second subwindow from it.
    if not entries:
        raise ValueError(f"case {case_id} has no FHIR observation for {track}")
    sampled = dict(dict(entries[0])["resource"].get("valueSampledData", {}))
    if float(sampled.get("period", 0.0)) != 1000.0 / SAMPLE_RATE_HZ:
        raise ValueError(f"case {case_id} {track} is not {SAMPLE_RATE_HZ} Hz")
    tokens = str(sampled.get("data", "")).split()
    values = np.fromiter(
        (np.nan if token == "E" else float(token) for token in tokens), dtype=np.float32, count=len(tokens),
    )
    expected = 60 * SAMPLE_RATE_HZ
    if values.shape != (expected,):
        raise ValueError(f"case {case_id} {track} has {len(values)}, expected {expected} samples")
    return values


def _best_window(values: np.ndarray) -> tuple[int, np.ndarray] | None:
    width = WINDOW_SECONDS * SAMPLE_RATE_HZ
    candidates = []
    for start in range(0, values.shape[-1] - width + 1, width):
        window = values[:, start:start + width]
        finite_fraction = float(np.isfinite(window).mean())
        variation = float(np.nanmin(np.nanstd(window, axis=1)))
        if finite_fraction == 1.0 and variation > 1e-6:
            candidates.append((variation, start, window))
    if not candidates:
        return None
    _, start, window = max(candidates, key=lambda item: item[0])
    return start, window


def _valid(path: Path) -> bool:
    try:
        with np.load(path, allow_pickle=False) as item:
            return item["waveforms"].shape == (3, WINDOW_SECONDS * SAMPLE_RATE_HZ) and np.isfinite(item["waveforms"]).all()
    except (OSError, ValueError, KeyError):
        return False


def _download_case(case_id: int, output: Path) -> dict[str, object]:
    destination = output / "records" / f"case_{case_id:04d}.npz"
    if destination.exists() and _valid(destination):
        return {"case_id": case_id, "status": "existing"}
    periods = [_track_period(case_id, track) for track in TRACKS]
    shared_start = max(period[0] for period in periods)
    shared_end = min(period[1] for period in periods)
    if shared_end - shared_start < timedelta(seconds=60):
        return {"case_id": case_id, "status": "ineligible_no_shared_minute"}
    latest_start = shared_end - timedelta(seconds=60)
    selected = None
    # Fixed fractional locations keep acquisition deterministic while avoiding
    # the frequent all-missing monitor warm-up at the beginning of a case.
    for fraction in np.linspace(0.0, 1.0, 9):
        slice_start = shared_start + (latest_start - shared_start) * float(fraction)
        values = np.stack([_track_values(case_id, track, slice_start) for track in TRACKS])
        selected = _best_window(values)
        if selected is not None:
            break
    if selected is None:
        return {"case_id": case_id, "status": "ineligible_no_complete_variable_window"}
    start, window = selected
    subject_id = _subject_id(case_id)
    temporary = destination.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        waveforms=window.astype(np.float32),
        case_id=np.asarray(case_id, dtype=np.int64),
        subject_id=np.asarray(subject_id),
        start_seconds=np.asarray((slice_start - shared_start).total_seconds() + start / SAMPLE_RATE_HZ, dtype=np.float64),
        sample_rate_hz=np.asarray(SAMPLE_RATE_HZ, dtype=np.int64),
        tracks=np.asarray(TRACKS),
    )
    os.replace(temporary, destination)
    return {
        "case_id": case_id,
        "subject_id": subject_id,
        "start_seconds": (slice_start - shared_start).total_seconds() + start / SAMPLE_RATE_HZ,
        "status": "downloaded",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=0, help="0 means all complete II/V5/PLETH cases")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.max_cases < 0 or args.workers < 1:
        raise ValueError("max-cases must be non-negative and workers positive")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "records").mkdir(exist_ok=True)
    cases = sorted(_track_cases(args.output))
    if args.max_cases:
        cases = cases[:args.max_cases]
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_download_case, case_id, args.output): case_id for case_id in cases}
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            case_id = futures[future]
            try:
                result = future.result()
            except ValueError as error:
                result = {"case_id": case_id, "status": "ineligible_data_contract", "error": str(error)}
            except Exception as error:  # preserve an explicit failure ledger and continue acquisition
                result = {"case_id": case_id, "status": "failed", "error": str(error)}
            results.append(result)
            if completed % 25 == 0 or completed == len(cases):
                print(json.dumps({"completed": completed, "total": len(cases), "last": result}), flush=True)
    results.sort(key=lambda row: int(row["case_id"]))
    (args.output / "acquisition_ledger.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    complete = [row for row in results if row["status"] in {"downloaded", "existing"}]
    failed = [row for row in results if row["status"] == "failed"]
    ineligible = [row for row in results if str(row["status"]).startswith("ineligible_")]
    manifest = {
        "schema_version": "vitaldb_ii_v5_pleth_windows_v1",
        "status": "complete" if not failed else "incomplete",
        "source": "VitalDB public FHIR API",
        "tracks": list(TRACKS),
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "window_seconds": WINDOW_SECONDS,
        "candidate_cases": len(cases),
        "materialized_cases": len(complete),
        "ineligible_cases": len(ineligible),
        "failed_cases": len(failed),
        "subject_disjoint_split_required": True,
        "source_tracks_sha256": _sha256(args.output / "source_tracks.csv"),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
