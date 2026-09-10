#!/usr/bin/env python3
"""SAP Eval Queue — GPU+CPU fully saturated & memory-safe.

Architecture:
  1 GPU worker  → evaluate_sap_v2.py --mode gpu-infer (serial, GPU)
  2 CPU workers → evaluate_sap_v2.py --mode cpu-aggregate (parallel CPU)

As soon as GPU finishes saving raw numpy tensors for a model, it enqueues
to CPU workers and immediately moves to the next model.

Usage:
  python scripts/run_sap_eval_queue.py [--cpu-workers 2] [--device cuda:0] [--batch-size 32]
"""
from __future__ import annotations
import argparse, json, logging, os, queue, signal, sqlite3, subprocess, sys, time, threading
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = sys.executable
EVAL_SCRIPT = _ROOT / "scripts/evaluate_sap_v2.py"
DB_PATH = _ROOT / "results/sap_benchmark_v2/sap_metrics.db"
RAW_BASE = Path("/tmp/sap_eval_raw")
LOG_DIR = _ROOT / "results/sap_benchmark_v2/logs"
CSV_PATH = _ROOT / "results/sap_benchmark_v2/sap_benchmark_v2.csv"
MASTER_ROSTER = _ROOT / "results/clinical_biomarkers_multids/FINAL_CLINICAL_BENCHMARK_METRICS_MASTER.csv"
ONELEAD_CATALOG = _ROOT / "results/onelead_checkpoint_store/catalog.sqlite"
ONELEAD_STORE = _ROOT / "scripts/onelead_checkpoint_store.py"
CONV10_ARCHIVE = _ROOT / "refine-logs/convergence_10e/archive/conv10e_best_pts.tar.gz"
CONV10_CACHE = _ROOT / "checkpoints/conv10e_cache"

# ============================================================
# Model Registry — comprehensive multi-suite candidate finder
# ============================================================

def _local_checkpoints():
    """Index checkpoint identities without depending on directory ordering."""
    paths = list(_ROOT.glob("refine-logs/**/best.pt"))
    paths += list((_ROOT / "checkpoints").glob("*.pt"))
    paths += list((_ROOT / "checkpoints/cache").glob("*.pt"))
    paths += list((_ROOT / "checkpoints/onelead_cache").glob("*.pt"))
    indexed = {}
    for path in paths:
        mid = path.parent.name if path.name == "best.pt" else path.stem
        indexed.setdefault(mid, path.resolve())
    return indexed


def _archived_onelead_ids():
    if not ONELEAD_CATALOG.exists():
        return set()
    with sqlite3.connect(ONELEAD_CATALOG) as con:
        return {
            row[0] for row in con.execute(
                "SELECT model_id FROM checkpoints WHERE status IN ('remote_verified','cached')"
            )
        }


def _candidates(include_unavailable=False):
    """Resolve the canonical 188-row master roster to local or archived weights."""
    if not MASTER_ROSTER.exists():
        raise FileNotFoundError(f"Canonical model roster is missing: {MASTER_ROSTER}")
    import pandas as pd
    roster = pd.read_csv(MASTER_ROSTER, usecols=["model_id", "study_track"])
    if roster["model_id"].duplicated().any():
        dupes = roster.loc[roster["model_id"].duplicated(), "model_id"].tolist()
        raise RuntimeError(f"Canonical roster contains duplicate model IDs: {dupes}")

    local = _local_checkpoints()
    archived = _archived_onelead_ids()
    cands = []
    for row in roster.itertuples(index=False):
        mid = str(row.model_id)
        if mid in {"reference", "__original_l0__"}:
            cand = {"model_id": mid, "path": Path("identity"), "group": str(row.study_track),
                    "source": "identity", "available": True, "is_ephemeral": False}
        elif mid in local:
            cand = {"model_id": mid, "path": local[mid], "group": str(row.study_track),
                    "source": "local", "available": True, "is_ephemeral": False}
        elif mid in archived:
            cand = {"model_id": mid, "path": _ROOT / f"checkpoints/onelead_cache/{mid}.pt",
                    "group": str(row.study_track), "source": "onelead_archive",
                    "available": True, "is_ephemeral": True}
        elif mid.startswith("conv10e_") and CONV10_ARCHIVE.is_file():
            member = f"refine-logs/convergence_10e/runs/{mid}/best.pt"
            cand = {"model_id": mid, "path": CONV10_CACHE / f"{mid}.pt",
                    "group": str(row.study_track), "source": "conv10e_tar_archive",
                    "archive_member": member, "available": True, "is_ephemeral": True}
        else:
            cand = {"model_id": mid, "path": None, "group": str(row.study_track),
                    "source": "unavailable", "available": False, "is_ephemeral": False}
        if include_unavailable or cand["available"]:
            cands.append(cand)
    return cands


def materialize(cand):
    if not cand["is_ephemeral"]:
        return cand["path"]
    if cand["source"] == "conv10e_tar_archive":
        destination = Path(cand["path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(f".pt.{os.getpid()}.part")
        try:
            with temporary.open("wb") as output:
                result = subprocess.run(
                    ["tar", "-xOzf", str(CONV10_ARCHIVE), cand["archive_member"]],
                    cwd=_ROOT, stdout=output, stderr=subprocess.PIPE,
                )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.decode(errors="replace").strip())
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise RuntimeError(f"archive member is empty: {cand['archive_member']}")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination
    result = subprocess.run(
        [PYTHON_BIN, str(ONELEAD_STORE), "materialize", cand["model_id"]],
        cwd=_ROOT, capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    path = Path(result.stdout.strip())
    if not path.is_absolute():
        path = _ROOT / path
    if not path.is_file():
        raise RuntimeError(f"materializer returned a missing file: {path}")
    return path


def evict(cand, path):
    if cand["is_ephemeral"] and path and path.is_file():
        path.unlink()

def already_done(model_id: str) -> bool:
    if not DB_PATH.exists(): return False
    try:
        with sqlite3.connect(DB_PATH, timeout=10) as con:
            r = con.execute(
                "SELECT COUNT(*) FROM sap_model_metrics WHERE model_id=? AND evaluation_version='sap_v2'",
                (model_id,)
            ).fetchone()
            return r[0] > 50  # at least 50 metrics written → consider done
    except Exception:
        return False

def raw_ready(model_id: str) -> bool:
    return (RAW_BASE / model_id / "_DONE").exists()

# ============================================================
# CSV Export
# ============================================================

def export_csv():
    """Pivot sap_model_metrics into a wide CSV (one row per model)."""
    if not DB_PATH.exists(): return
    import pandas as pd
    try:
        with sqlite3.connect(DB_PATH, timeout=30) as con:
            df = pd.read_sql("SELECT model_id, metric_name, point_estimate FROM sap_model_metrics WHERE evaluation_version='sap_v2'", con)
        if df.empty: return
        wide = df.pivot_table(index="model_id", columns="metric_name", values="point_estimate", aggfunc="first").reset_index()
        wide.to_csv(CSV_PATH, index=False)
        log.info("CSV exported → %s (%d models, %d metrics)", CSV_PATH, len(wide), wide.shape[1] - 1)
    except Exception as e:
        log.warning("CSV export failed: %s", e)

# ============================================================
# Worker subprocess helpers
# ============================================================

def run_proc(cmd, log_file, timeout=None):
    with open(log_file, "w") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
    proc.wait(timeout=timeout)
    return proc.returncode

# ============================================================
# Main orchestrator
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-workers", type=int, default=2, help="parallel CPU aggregation workers (default: 2)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true", default=True,
                    help="skip models already in DB (default: True)")
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--preflight", action="store_true", help="report roster/checkpoint coverage and exit")
    args = ap.parse_args()

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RAW_BASE.mkdir(parents=True, exist_ok=True)

    full_roster = _candidates(include_unavailable=True)
    cands = [c for c in full_roster if c["available"]]
    unavailable = [c for c in full_roster if not c["available"]]
    archived = [c for c in cands if c["is_ephemeral"]]
    log.info("Canonical roster: %d models | checkpoint-ready: %d | archived/on-demand: %d | unavailable: %d",
             len(full_roster), len(cands), len(archived), len(unavailable))
    if unavailable:
        missing_path = LOG_DIR / "unavailable_models.json"
        missing_path.write_text(json.dumps(unavailable, indent=2, default=str) + "\n")
        log.warning("%d roster models have no recoverable checkpoint; details: %s",
                    len(unavailable), missing_path)
    if args.preflight:
        return

    if args.resume:
        cands = [c for c in cands if not already_done(c["model_id"])]
        log.info("After resume filter: %d models to evaluate", len(cands))

    if not cands:
        log.info("All models already evaluated. Exporting CSV...")
        export_csv(); return

    cpu_q: queue.Queue = queue.Queue()
    stop_evt = threading.Event()

    def cpu_worker(worker_id: int):
        while not stop_evt.is_set():
            try:
                cand = cpu_q.get(timeout=3)
            except queue.Empty:
                continue
            mid = cand["model_id"]
            raw_dir = RAW_BASE / mid
            if already_done(mid):
                log.info("[CPU#%d] %s already done in DB, skipping.", worker_id, mid)
                cpu_q.task_done()
                continue
            log.info("[CPU#%d] Aggregating %s ...", worker_id, mid)
            log_file = LOG_DIR / f"{mid}_cpu.log"
            cmd = [
                PYTHON_BIN, str(EVAL_SCRIPT),
                "--model-id", mid,
                "--checkpoint-path", str(cand["path"]),
                "--mode", "cpu-aggregate",
                "--raw-dir", str(raw_dir),
                "--db-path", str(DB_PATH),
            ] + (["--smoke"] if args.smoke else [])
            rc = run_proc(cmd, log_file, timeout=1800)
            if rc == 0:
                log.info("[CPU#%d] ✓ %s aggregated successfully.", worker_id, mid)
                import shutil; shutil.rmtree(raw_dir, ignore_errors=True)
                export_csv()
            else:
                log.error("[CPU#%d] ✗ %s aggregation failed (rc=%d). Log: %s", worker_id, mid, rc, log_file)
            cpu_q.task_done()

    threads = []
    for wid in range(args.cpu_workers):
        t = threading.Thread(target=cpu_worker, args=(wid,), daemon=True)
        t.start(); threads.append(t)

    # Pre-enqueue models whose raw arrays are already computed
    pre_done = [c for c in cands if raw_ready(c["model_id"]) and not already_done(c["model_id"])]
    for c in pre_done:
        log.info("Pre-queuing %s (raw arrays exist)", c["model_id"])
        cpu_q.put(c)

    # GPU inference loop
    gpu_pending = [c for c in cands if not raw_ready(c["model_id"])]
    total = len(cands)
    n_gpu_done = 0

    log.info("Starting GPU inference for %d models with %d CPU workers...", len(gpu_pending), args.cpu_workers)

    for i, cand in enumerate(gpu_pending):
        mid = cand["model_id"]
        raw_dir = RAW_BASE / mid
        # Clean any partial raw dir
        if not (raw_dir / "_DONE").exists():
            import shutil; shutil.rmtree(raw_dir, ignore_errors=True)
        log.info("[GPU] (%d/%d) Running inference on %s ...", i+1, len(gpu_pending), mid)
        log_file = LOG_DIR / f"{mid}_gpu.log"
        checkpoint_path = None
        try:
            checkpoint_path = materialize(cand)
            if cand["is_ephemeral"]:
                log.info("[GPU] Materialized archived checkpoint for %s → %s", mid, checkpoint_path)
            cmd = [
                PYTHON_BIN, str(EVAL_SCRIPT),
                "--model-id", mid,
                "--checkpoint-path", str(checkpoint_path),
                "--mode", "gpu-infer",
                "--raw-dir", str(raw_dir),
                "--db-path", str(DB_PATH),
                "--device", args.device,
                "--batch-size", str(args.batch_size),
            ] + (["--smoke"] if args.smoke else [])
            rc = run_proc(cmd, log_file, timeout=1800)
            if rc != 0:
                log.warning("[GPU] First attempt failed for %s (rc=%d), retrying once...", mid, rc)
                time.sleep(3)
                rc = run_proc(cmd, log_file, timeout=1800)
        except Exception as exc:
            rc = 1
            log.exception("[GPU] Could not prepare %s: %s", mid, exc)
        finally:
            evict(cand, checkpoint_path)

        if rc == 0:
            log.info("[GPU] ✓ %s inference finished.", mid)
            cpu_q.put(cand)
            n_gpu_done += 1
        else:
            log.error("[GPU] ✗ %s inference failed permanently (rc=%d). Log: %s", mid, rc, log_file)

    log.info("[GPU] All GPU inference jobs finished. Waiting for CPU workers to complete...")
    cpu_q.join()
    stop_evt.set()
    for t in threads: t.join(timeout=10)

    log.info("All evaluations completed! Exporting final CSV...")
    export_csv()
    ndone = sum(1 for c in full_roster if already_done(c["model_id"]))
    log.info("Final status: %d/%d canonical models evaluated in %s", ndone, len(full_roster), DB_PATH)

if __name__ == "__main__":
    main()
