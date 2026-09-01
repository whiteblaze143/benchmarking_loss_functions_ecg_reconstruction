#!/usr/bin/env python3
"""CPU-only, resumable external RDB evaluation for convergence models.
"""

from __future__ import annotations
import argparse
import datetime as dt
import gc
import json
import math
import os
import signal
import sqlite3
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Iterable

os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.evaluate_ecgaim_ludb_semiseg_blinded import (
    DEFAULT_CHECKPOINT,
    DEFAULT_SELECTION,
    SignalPreprocessor,
    build_delineator,
    selected_checkpoint,
)

# Re-use evaluation functions from the blinded script
from scripts.evaluate_onelead_rdb_semiseg_blinded import (
    load_test_index, lead_groups, fresh_accumulator, score_record, complete,
    summarize,
    finite_mean, quantile, bootstrap_upper, bootstrap_quantile_upper,
)
from unified_latents.engineering.utils.common import mask_unobserved_leads
from unified_latents.engineering.utils.regimes import make_lead_indices

DEFAULT_TEST_CACHE = ROOT / "data/rdb_wavelet_delineation_cache"
DEFAULT_DB = ROOT / "results/convergence_rdb_semiseg_v1/compact.sqlite"
STOP_REQUESTED = False


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def request_stop(signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(json.dumps({"event": "stop_requested", "signal": signum}), flush=True)


def connect_results(path: Path) -> sqlite3.Connection:
    """Open or create the convergence evaluation result DB."""
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=60)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript("""
        CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value_json TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS evaluations(
            model_id TEXT, stage TEXT NOT NULL, status TEXT NOT NULL,
            started_at TEXT, completed_at TEXT,
            primary_mean_micro_f1_20ms REAL,
            primary_signal_pearson_p05 REAL,
            details_json TEXT,
            PRIMARY KEY (model_id, stage)
        );
    """)
    return con


def initialize(con: sqlite3.Connection, protocol: dict[str, Any]) -> None:
    with con:
        con.execute("INSERT OR IGNORE INTO metadata(key, value_json) VALUES (?, ?)",
                    ("protocol", json.dumps(protocol)))
        con.execute("INSERT OR IGNORE INTO metadata(key, value_json) VALUES (?, ?)",
                    ("created_at", json.dumps(utc_now())))

def load_convergence_model(path: Path, observed_lead: int) -> tuple[torch.nn.Module, int]:
    """Load a convergence model checkpoint (ecg_aim_wavelet_mtl_v1 format).
    Returns (model, observed_lead_index).
    """
    import argparse as _ap
    from unified_latents.engineering.experimental.wavelet_ssl_ecg_aim import build_wavelet_ecg_aim
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = _ap.Namespace(**payload["config"])
    model = build_wavelet_ecg_aim(
        target_len=5000,
        patch_size=cfg.patch_size, width=cfg.width,
        encoder_depth=cfg.encoder_depth, decoder_depth=cfg.decoder_depth, heads=cfg.heads,
        random_mask_ratio=cfg.random_mask_ratio, temporal_mask_ratio=cfg.temporal_mask_ratio,
        consistency_weight=cfg.consistency_weight,
        lead_conditioning_mode=cfg.lead_conditioning_mode,
        use_relative_geometry=cfg.use_relative_geometry,
        use_spatial_film=cfg.use_spatial_film,
        spatial_gain_init=cfg.spatial_gain_init,
        geometry_control=cfg.geometry_control,
        use_wavelet_branch=cfg.use_wavelet_branch,
        wavelet_bank=cfg.wavelet_bank, custom_wavelet_asset=cfg.custom_wavelet_asset,
        view_a_bank=cfg.view_a_bank, view_b_bank=cfg.view_b_bank,
        view_a_custom_wavelet_asset=cfg.view_a_custom_wavelet_asset,
        view_b_custom_wavelet_asset=cfg.view_b_custom_wavelet_asset,
        n_scales=cfg.n_scales, min_freq_hz=cfg.min_freq_hz, max_freq_hz=cfg.max_freq_hz,
        morlet_cycles=cfg.morlet_cycles, view_a=cfg.view_a, view_b=cfg.view_b,
        wavelet_encoder=cfg.wavelet_encoder, wavelet_dim=cfg.wavelet_dim,
        wavelet_depth=cfg.wavelet_depth, wavelet_heads=cfg.wavelet_heads,
        wavelet_conv_hidden=cfg.wavelet_conv_hidden, wavelet_fusion=cfg.wavelet_fusion,
        fusion_heads=cfg.fusion_heads, inference_view=cfg.inference_view,
        ssl_mode=cfg.ssl_mode,
        ssl_projector_hidden=cfg.ssl_projector_hidden, ssl_projector_dim=cfg.ssl_projector_dim,
        ssl_predictor_hidden=cfg.ssl_predictor_hidden, byol_tau=cfg.byol_tau,
        use_delineation_head=not cfg.no_delineation_head,
        delineation_hidden=cfg.delineation_hidden, delineation_kernel=cfg.delineation_kernel,
        predict_fiducials=not cfg.no_fiducial_head,
        mask_type_mode=cfg.mask_type_mode,
    )
    state = {k.removeprefix("_orig_mod."): v for k, v in payload["model_state_dict"].items()}
    # Convert half -> float if the checkpoint was saved in fp16 compact form
    state = {k: v.float() if v.is_floating_point() else v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model.float().eval(), observed_lead

def extract_and_evaluate(args, connection, records, delineator, preprocessor):
    import time
    
    # Extract 10-epoch models once at startup
    archive_path = ROOT / "refine-logs/convergence_10e/archive/conv10e_best_pts.tar.gz"
    if archive_path.exists():
        with tempfile.TemporaryDirectory() as tmpdir:
            print(f"[{utc_now()}] Extracting 10e models to {tmpdir}...", flush=True)
            with tarfile.open(archive_path, "r:gz") as tar:
                for member in tar.getmembers():
                    if STOP_REQUESTED: break
                    if member.name.endswith("best.pt"):
                        tar.extract(member, path=tmpdir)
                        extracted_path = Path(tmpdir) / member.name
                        run_name = Path(member.name).parent.name
                        model_id = run_name.replace("conv10e_", "conv10e_")
                        lead = 1 if "_l1" in model_id else 0
                        identity = {
                            "model_id": model_id,
                            "architecture": "alitok",
                            "observed_leads_json": json.dumps([lead]),
                            "local_path": str(extracted_path)
                        }
                        if not complete(connection, model_id, args.phase):
                            run_model(connection, identity, args.phase, records, delineator, preprocessor, args)
                        extracted_path.unlink()
    
    # Daemon loop for 15-epoch checkpoints
    runs_dir = ROOT / "refine-logs/convergence_10e/runs"
    while not STOP_REQUESTED:
        print(f"[{utc_now()}] Scanning for newly completed 15e models...", flush=True)
        if runs_dir.exists():
            for run_path in runs_dir.iterdir():
                if STOP_REQUESTED: break
                if not run_path.is_dir(): continue
                best_pt = run_path / "best.pt"
                if not best_pt.exists(): continue
                
                model_id = run_path.name
                lead = 1 if "_l1" in model_id else 0
                identity = {
                    "model_id": model_id,
                    "architecture": "alitok",
                    "observed_leads_json": json.dumps([lead]),
                    "local_path": str(best_pt)
                }
                if not complete(connection, model_id, args.phase):
                    run_model(connection, identity, args.phase, records, delineator, preprocessor, args)
                    
        # Sleep for 5 minutes before checking again
        if not STOP_REQUESTED:
            for _ in range(60):
                if STOP_REQUESTED: break
                time.sleep(5)

def reconstruct_record(model: torch.nn.Module, observed_lead: int, target: torch.Tensor) -> torch.Tensor:
    """Run wavelet ECG-AIM reconstruction, returns (B, 12, 5000) float tensor."""
    obs_list = [observed_lead]  # mask_unobserved_leads / make_lead_indices need list[int]
    target = target.float()
    masked = mask_unobserved_leads(target, obs_list).contiguous()
    indices = make_lead_indices(obs_list, target.shape[0], torch.device("cpu"))
    result = model(
        masked, y_full=target, lead_indices=indices,
        compute_delineation=False, compute_ssl=False
    )["y_pred"].float()
    result = result[:, :12, :5000]
    result[:, observed_lead, :] = target[:, observed_lead, :]  # restore observed
    return result

def predict_masks(model, preprocessor, waveforms, batch_size):
    prepared = [preprocessor(waveform) for waveform in waveforms]
    masks = []
    for offset in range(0, len(prepared), batch_size):
        batch = torch.from_numpy(np.stack(prepared[offset:offset + batch_size]))
        with torch.inference_mode():
            logits = model(batch)["seg_logits"]
        masks.extend(np.repeat(item.astype(np.int8), 2)[:5000] for item in logits.argmax(1).numpy())
    return masks

def run_model(connection: sqlite3.Connection, identity: dict[str, Any], stage: str, records: list[dict[str, Any]], delineator: torch.nn.Module, preprocessor: SignalPreprocessor, args: argparse.Namespace) -> None:
    print(f"[{utc_now()}] Evaluating {identity['model_id']} stage={stage} on {len(records)} RDB records...", flush=True)
    with connection:
        connection.execute("INSERT OR REPLACE INTO evaluations(model_id, stage, status, started_at) VALUES (?, ?, 'running', ?)", (identity["model_id"], stage, utc_now()))
    
    try:
        observed_lead: int = json.loads(identity["observed_leads_json"])[0]
        model, obs = load_convergence_model(Path(identity["local_path"]), observed_lead)
        acc = fresh_accumulator(lead_groups(obs))
        
        for offset in range(0, len(records), args.reconstruction_batch_size):
            if STOP_REQUESTED:
                raise InterruptedError("stop requested")
            batch_records = records[offset:offset + args.reconstruction_batch_size]
            # RDB cache files are saved with torch.save; keys are 'waveform' and 'segmentation'
            payloads = [torch.load(r["path"], map_location="cpu", weights_only=True) for r in batch_records]
            batch_target = torch.stack([p["waveform"].float() for p in payloads])
            
            with torch.inference_mode():
                recons = reconstruct_record(model, obs, batch_target)
            
            missing = list(lead_groups(obs)["all_missing"])
            waveforms = [recons[ri, lead].numpy() for ri in range(len(payloads)) for lead in missing]
            masks = predict_masks(delineator, preprocessor, waveforms, args.delineation_batch_size)
            cursor = 0
            for ri, payload in enumerate(payloads):
                predictions = {}
                for lead in missing:
                    predictions[lead] = masks[cursor]; cursor += 1
                score_record(acc, batch_target[ri].numpy(), recons[ri].numpy(), payload["segmentation"].numpy(), predictions)
            del payloads, batch_target, recons, waveforms, masks
                
            gc.collect()
            
        del model
        gc.collect()
        
        boundary_rows, region_rows, signal_rows, headline = summarize(acc)
        f1_20ms = headline.get("primary_mean_micro_f1_20ms") or 0.0
        pearson_p05 = headline.get("primary_signal_pearson_p05")
        
        with connection:
            connection.execute(
                "UPDATE evaluations SET status='complete', completed_at=?, primary_mean_micro_f1_20ms=?, primary_signal_pearson_p05=? WHERE model_id=? AND stage=?",
                (utc_now(), f1_20ms, pearson_p05, identity["model_id"], stage)
            )
        print(f"[{utc_now()}] DONE {identity['model_id']} RDB boundary F1@20ms = {f1_20ms:.4f}, signal p05 = {pearson_p05}", flush=True)
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        with connection:
            connection.execute(
                "UPDATE evaluations SET status='failed', completed_at=?, details_json=? WHERE model_id=? AND stage=?",
                (utc_now(), json.dumps({"error": str(e)}), identity["model_id"], stage)
            )

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "full"), default="full")
    parser.add_argument("--test-cache", type=Path, default=DEFAULT_TEST_CACHE)
    parser.add_argument("--results-db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--delineator-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--selection-summary", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--state", choices=("model", "model_ema"), default="model_ema")
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--reconstruction-batch-size", type=int, default=4)
    parser.add_argument("--delineation-batch-size", type=int, default=32)
    args = parser.parse_args()
    
    signal.signal(signal.SIGINT, request_stop); signal.signal(signal.SIGTERM, request_stop)
    torch.set_num_threads(args.torch_threads); torch.set_num_interop_threads(1)
    
    selected_checkpoint(args.delineator_checkpoint, args.selection_summary, args.state)
    full_records, pilot_records, dataset_sha, dataset_audit = load_test_index(args.test_cache, 6)
    records = pilot_records[:2] if args.phase == "smoke" else full_records
    
    connection = connect_results(args.results_db)
    initialize(connection, {"name": "convergence_rdb_eval"})
    
    delineator = build_delineator(args.delineator_checkpoint, args.state)
    preprocessor = SignalPreprocessor()
    
    extract_and_evaluate(args, connection, records, delineator, preprocessor)
    connection.close()
    
if __name__ == "__main__":
    main()
