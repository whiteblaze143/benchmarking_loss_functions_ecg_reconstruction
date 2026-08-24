"""Verified checkpoint → encoder activation → UMAP utilities for the book.

The catalog remains the source of checkpoint identity. No embeddings are
persisted unless a caller explicitly saves the returned DataFrame.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from live_results import ROOT


DEFAULT_LAYERS = {"unet": "down5", "msvae": "encoder", "ecg_aim": "encoder"}
OBSERVED_LEADS = [0, 1, 7]


def catalog_row(model_id: str, db_path: Path | None = None) -> dict:
    db_path = db_path or ROOT / "results/checkpoint_store/catalog.sqlite"
    with sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True) as con:
        con.row_factory = sqlite3.Row
        row = con.execute("select * from checkpoints where model_id=?", (model_id,)).fetchone()
    if row is None:
        raise KeyError(f"Checkpoint {model_id!r} is not registered in {db_path}")
    result = dict(row)
    metadata = json.loads(result.get("metadata_json") or "{}")
    result["architecture"] = result.get("architecture") or metadata.get("family")
    return result


def _build_model(architecture: str):
    if architecture == "unet":
        from scripts.train_mcma_3lead import MCMAModel
        return MCMAModel(in_channels=3, out_channels=12)
    if architecture == "msvae":
        from unified_latents.engineering.experimental.Multi_Scale_VAE import WearECGVAE
        return WearECGVAE(latent_channels=4, target_len=5000, beta_kl=1e-4, missing_lead_weight=1.0)
    if architecture == "ecg_aim":
        from unified_latents.engineering.experimental.alitok_vae_exp import build_alitok_vae_1d
        return build_alitok_vae_1d(architecture="ecg_aim_v1", target_len=5000,
                                   patch_size=25, encoder_depth=8, decoder_depth=4)
    raise ValueError(f"Unsupported three-lead architecture: {architecture}")


def load_verified_model(model_id: str, device="cpu", db_path: Path | None = None):
    """Materialize through the verified store and strictly load its state."""
    from scripts.checkpoint_store import load_checkpoint_with_identity
    from scripts.evaluate_comprehensive_registry import normalize_compiled_state_dict

    db_path = db_path or ROOT / "results/checkpoint_store/catalog.sqlite"
    row = catalog_row(model_id, db_path)
    payload, identity = load_checkpoint_with_identity(
        model_id=model_id, db_path=db_path, map_location="cpu", weights_only=False
    )
    if identity["sha256"] != row["sha256"]:
        raise RuntimeError("Materialized checkpoint digest differs from the catalog")
    state = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    state = normalize_compiled_state_dict(state)
    model = _build_model(row["architecture"])
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), row, identity


def layer_names(model, contains="encoder") -> list[str]:
    """Useful in a notebook before choosing a hook layer."""
    return [name for name, _ in model.named_modules() if name and contains.lower() in name.lower()]


def _tensor_from_output(output):
    if torch.is_tensor(output): return output
    if isinstance(output, (list, tuple)):
        return next((x for x in output if torch.is_tensor(x)), None)
    if isinstance(output, dict):
        return next((x for x in output.values() if torch.is_tensor(x)), None)
    return None


def _pool_activation(value: torch.Tensor, architecture: str) -> np.ndarray:
    """Return mean/std/max per feature channel, one row per record."""
    x = value.detach().float()
    if x.ndim == 2:
        return x.cpu().numpy()
    # Transformer encoder is batch × tokens × width; CNN encoders are batch × channels × time.
    feature_axis = x.ndim - 1 if architecture == "ecg_aim" else 1
    x = x.movedim(feature_axis, 1).flatten(2)
    pooled = torch.cat([x.mean(-1), x.std(-1), x.amax(-1)], dim=1)
    return pooled.cpu().numpy()


def _forward(model, architecture: str, y: torch.Tensor, device: str):
    y = y.to(device)
    if architecture == "unet":
        return model(F.pad(y[:, OBSERVED_LEADS], (0, 120)))
    from unified_latents.engineering.utils.common import mask_unobserved_leads
    from unified_latents.engineering.utils.regimes import make_lead_indices
    masked = mask_unobserved_leads(y, OBSERVED_LEADS)
    lead_indices = make_lead_indices(OBSERVED_LEADS, len(y), torch.device(device))
    return model(masked, y_full=y, lead_indices=lead_indices, mode="stage1")


def checkpoint_umap(model_id: str, data_dir: str | Path, *, layer: str | None = None,
                    max_records=240, batch_size=8, device="cpu", seed=42,
                    db_path: Path | None = None):
    """Extract true checkpoint activations and return (embedding, provenance)."""
    # Resolve before importing model code; legacy bootstrap modules may change cwd.
    data_dir = Path(data_dir).resolve()
    model, row, identity = load_verified_model(model_id, device, db_path)
    architecture = row["architecture"]
    layer = layer or DEFAULT_LAYERS[architecture]
    modules = dict(model.named_modules())
    if layer not in modules:
        candidates = [n for n in modules if "encoder" in n or "down" in n]
        raise KeyError(f"Layer {layer!r} absent. Candidate layers: {candidates[:80]}")

    captured = []
    def hook(_module, _inputs, output):
        tensor = _tensor_from_output(output)
        if tensor is None: raise TypeError(f"Layer {layer} did not emit a tensor")
        captured.append(_pool_activation(tensor, architecture))
    handle = modules[layer].register_forward_hook(hook)

    all_files = sorted(data_dir.glob("*.pt"))
    if max_records and len(all_files) > max_records:
        # Cover the complete deterministic inventory instead of taking an
        # alphabetic prefix that may contain only one rhythm class.
        indices = np.linspace(0, len(all_files) - 1, max_records, dtype=int)
        files = [all_files[i] for i in indices]
    else:
        files = all_files
    if len(files) < 5: raise ValueError(f"Need at least five .pt records in {data_dir}")
    metadata = []
    try:
        with torch.inference_mode():
            for start in range(0, len(files), batch_size):
                batch_files = files[start:start + batch_size]
                examples = [torch.load(p, map_location="cpu", weights_only=False) for p in batch_files]
                waves = [e.get("waveform") if isinstance(e, dict) else e for e in examples]
                y = torch.stack([w.float() for w in waves])
                _forward(model, architecture, y, device)
                for p, e in zip(batch_files, examples):
                    metadata.append({
                        "record_id": str(e.get("record_id", p.stem)) if isinstance(e, dict) else p.stem,
                        "patient_id": str(e.get("patient_id", "unknown")) if isinstance(e, dict) else "unknown",
                        "label": str(e.get("canonical_rhythm", "unknown")) if isinstance(e, dict) else "unknown",
                        "released_rhythm": str(e.get("released_rhythm", "unknown")) if isinstance(e, dict) else "unknown",
                        "source_dataset": str(e.get("source_dataset", "unknown")) if isinstance(e, dict) else "unknown",
                        "split": str(e.get("split", "unknown")) if isinstance(e, dict) else "unknown",
                    })
    finally:
        handle.remove()
        del model
        if device == "cuda": torch.cuda.empty_cache()

    features = np.concatenate(captured, axis=0)
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    import umap
    x = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(features))
    n_neighbors = min(15, len(x) - 1)
    xy = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=.15,
                   metric="cosine", random_state=seed).fit_transform(x)
    result = pd.DataFrame(metadata)
    result[["umap_1", "umap_2"]] = xy
    from sklearn.manifold import trustworthiness
    from sklearn.metrics import pairwise_distances, silhouette_score
    distances = pairwise_distances(x, metric="cosine")
    np.fill_diagonal(distances, np.inf)
    nearest = distances.argmin(axis=1)
    result["nearest_record_id"] = result.iloc[nearest].record_id.to_numpy()
    result["nearest_label"] = result.iloc[nearest].label.to_numpy()
    result["nearest_cosine_distance"] = distances[np.arange(len(result)), nearest]
    trust_k = min(10, max(1, (len(result) - 1) // 2))
    label_counts = result.label.value_counts()
    eligible = result.label.isin(label_counts[label_counts >= 2].index)
    silhouette = None
    if eligible.sum() >= 4 and result.loc[eligible, "label"].nunique() >= 2:
        silhouette = float(silhouette_score(x[eligible], result.loc[eligible, "label"], metric="cosine"))
    provenance = {
        "model_id": model_id, "architecture": architecture, "checkpoint_sha256": identity["sha256"],
        "layer": layer, "pooling": "per-feature mean+std+max", "data_dir": str(data_dir),
        "records": len(result), "feature_dimensions": int(features.shape[1]), "device": device,
        "umap": {"n_neighbors": n_neighbors, "min_dist": .15, "metric": "cosine", "seed": seed},
        "diagnostics": {
            "trustworthiness_k": trust_k,
            "trustworthiness": float(trustworthiness(x, xy, n_neighbors=trust_k, metric="cosine")),
            "rhythm_silhouette_cosine": silhouette,
            "nearest_neighbor_label_agreement": float((result.label == result.nearest_label).mean()),
        },
    }
    return result, provenance
