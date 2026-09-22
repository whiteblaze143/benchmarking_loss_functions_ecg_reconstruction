#!/usr/bin/env python3
"""Train P07 full-12-lead auxiliary reconstruction from random lead subsets."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from repecg.common.metrics import multilabel_metrics
from repecg.common.models import biased_imq_mmd2
from repecg.paper07_operator import Full12LeadWaveformDecoder, OperatorSetModel


TARGET_SCHEMA = "paper07_full12_waveform_targets_v1"
CONTEXT_SCHEMA = "paper07_frozen_full12_context_responses_v1"
PATIENT_CONTEXT_SCHEMA = "paper07_ptbxl_patient_context_v1"
VARIANTS = {
    "continuous_full12_aux_mmd": 1.0,
    "continuous_full12_aux_no_mmd": 0.0,
    "lead1_unconditioned": 1.0,
    "lead1_age_sex_conditioned": 1.0,
}
RECONSTRUCTION_WEIGHT = 0.1


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_representations(path: Path, split: str) -> dict[str, np.ndarray]:
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("schema_version") != CONTEXT_SCHEMA or manifest.get("status") != "complete":
        raise ValueError("requires complete frozen full-12-lead context responses")
    responses = np.load(path / f"{split}_responses.npy", mmap_mode="r")
    labels = np.load(path / f"{split}_labels.npy", allow_pickle=False)
    ecg_ids = np.load(path / f"{split}_ecg_ids.npy", allow_pickle=False)
    patient_ids = np.load(path / f"{split}_patient_ids.npy", allow_pickle=False)
    operators = np.load(path / "operators.npy", allow_pickle=False)
    if operators.shape != (12, 8) or not np.isfinite(operators).all():
        raise ValueError("full-12 context operator bank must have shape [12,8]")
    if responses.shape != (len(labels), 12, 16, 128) or responses.dtype != np.float32:
        raise ValueError(f"invalid {split} full-12 response shape or dtype")
    if labels.shape != (len(responses), 5) or ecg_ids.shape != (len(responses),) or patient_ids.shape != (len(responses),):
        raise ValueError(f"invalid {split} response metadata alignment")
    if not np.isfinite(responses).all():
        raise ValueError(f"non-finite {split} full-12 responses")
    return {"operators": operators, "responses": responses, "labels": labels, "ecg_ids": ecg_ids, "patient_ids": patient_ids}


def _load_targets(target_root: Path, split: str, expected: dict[str, np.ndarray]) -> np.memmap:
    manifest = json.loads((target_root / "manifest.json").read_text())
    if manifest.get("schema_version") != TARGET_SCHEMA or manifest.get("status") != "complete":
        raise ValueError("requires a complete full-12-lead target artifact")
    waveforms = np.load(target_root / f"{split}_waveforms.npy", mmap_mode="r")
    ecg_ids = np.load(target_root / f"{split}_ecg_ids.npy", allow_pickle=False)
    patient_ids = np.load(target_root / f"{split}_patient_ids.npy", allow_pickle=False)
    if waveforms.shape != (len(expected["labels"]), 12, 5000) or waveforms.dtype != np.float16:
        raise ValueError(f"invalid {split} waveform target shape or dtype")
    if not np.array_equal(ecg_ids, expected["ecg_ids"]) or not np.array_equal(patient_ids, expected["patient_ids"]):
        raise ValueError(f"{split} waveform target IDs do not match frozen full-12 contexts")
    return waveforms


def _load_patient_context(path: Path, split: str, expected: dict[str, np.ndarray]) -> np.ndarray:
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("schema_version") != PATIENT_CONTEXT_SCHEMA or manifest.get("status") != "complete":
        raise ValueError("requires complete PTB-XL patient context")
    values = np.load(path / f"{split}_context.npy", allow_pickle=False)
    ecg_ids = np.load(path / f"{split}_ecg_ids.npy", allow_pickle=False)
    patient_ids = np.load(path / f"{split}_patient_ids.npy", allow_pickle=False)
    if values.shape != (len(expected["labels"]), 3) or not np.isfinite(values).all():
        raise ValueError(f"invalid {split} patient context")
    if not np.array_equal(ecg_ids, expected["ecg_ids"]) or not np.array_equal(patient_ids, expected["patient_ids"]):
        raise ValueError(f"{split} patient context IDs do not match frozen representations")
    return values.astype(np.float32, copy=False)


def _sample_subsets(
    batch: int, generator: torch.Generator, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random proper subsets: order, padding mask, and original-order held-out mask."""
    sizes = torch.randint(1, 12, (batch,), generator=generator, device=device)
    order = torch.rand((batch, 12), generator=generator, device=device).argsort(dim=1)
    padded_context = torch.arange(12, device=device).expand(batch, -1) >= sizes.unsqueeze(1)
    observed = torch.zeros((batch, 12), device=device, dtype=torch.bool)
    observed.scatter_(1, order, ~padded_context)
    return order, padded_context, ~observed


def _gather(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return values[torch.arange(len(values), device=values.device).unsqueeze(1), indices]


def heldout_waveform_mse(
    reconstruction: torch.Tensor, target: torch.Tensor, heldout_leads: torch.Tensor
) -> torch.Tensor:
    if reconstruction.shape != target.shape or reconstruction.ndim != 3 or heldout_leads.shape != reconstruction.shape[:2]:
        raise ValueError("reconstruction, target, and held-out lead mask must align")
    selected = heldout_leads.unsqueeze(-1).to(reconstruction.dtype)
    denominator = selected.sum() * reconstruction.shape[-1]
    if denominator.item() <= 0:
        raise ValueError("each context must withhold at least one lead")
    return ((reconstruction - target).square() * selected).sum() / denominator


def _save_checkpoint(
    output: Path, variant: str, model: OperatorSetModel, decoder: Full12LeadWaveformDecoder,
    optimizer: torch.optim.Optimizer, epoch: int, val_metrics: dict[str, object],
    history: list[dict[str, float]], target_manifest: Path, context_manifest: Path,
    patient_context_manifest: Path | None, conditioning: str,
) -> None:
    payload = {
        "schema_version": "paper07_full12_aux_mmd_checkpoint_v2",
        "variant": variant,
        "state_dict": copy.deepcopy(model.state_dict()),
        "decoder_state_dict": copy.deepcopy(decoder.state_dict()),
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        "epoch": epoch,
        "val_metrics": val_metrics,
        "loss_contract": {
            "diagnosis_bce": 1.0,
            "view_imq_mmd2": VARIANTS[variant],
            "view_mmd_c2": 1.0,
            "view_normalization": "LayerNorm(elementwise_affine=False)",
            "full12_waveform_heldout_mse": RECONSTRUCTION_WEIGHT,
            "conditioning": conditioning,
        },
        "target_manifest_sha256": _sha256(target_manifest),
        "context_manifest_sha256": _sha256(context_manifest),
        "patient_context_manifest_sha256": _sha256(patient_context_manifest) if patient_context_manifest else None,
        "history": history,
    }
    temporary = output / f"{variant}_best.tmp.pt"
    torch.save(payload, temporary)
    os.replace(temporary, output / f"{variant}_best.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representations", type=Path, required=True, help="Frozen 12-lead context-response artifact")
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--patient-context", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.batch < 2 or args.max_epochs < 1 or args.patience < 1:
        raise ValueError("batch must be at least 2; epochs and patience must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite populated training output: {args.output}")
    torch.backends.cudnn.enabled = False
    _seed(args.seed)
    lead1_mode = args.variant.startswith("lead1_")
    if lead1_mode != (args.patient_context is not None):
        raise ValueError("--patient-context is required exactly for the Lead-I variants")
    train = _load_representations(args.representations, "train")
    validation = _load_representations(args.representations, "validation")
    train_targets = _load_targets(args.targets, "train", train)
    validation_targets = _load_targets(args.targets, "validation", validation)
    del validation_targets  # Selection is diagnosis-only; targets were admission-validated above.
    train_context = _load_patient_context(args.patient_context, "train", train) if lead1_mode else None
    validation_context = _load_patient_context(args.patient_context, "validation", validation) if lead1_mode else None
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda")
    model = OperatorSetModel(
        response_dim=128, classes=5, operator_mode="continuous", patient_context_dim=3 if lead1_mode else 0,
    ).to(device)
    for parameter in model.decoder.parameters():
        parameter.requires_grad_(False)
    decoder = Full12LeadWaveformDecoder().to(device)
    view_normalizer = nn.LayerNorm(256, elementwise_affine=False).to(device)
    prevalence = torch.as_tensor(train["labels"].mean(axis=0), device=device)
    pos_weight = ((1.0 - prevalence) / prevalence.clamp_min(1e-8)).clamp_max(10.0)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    trainable = [p for p in list(model.parameters()) + list(decoder.parameters()) if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=3e-4, weight_decay=1e-4)
    operator_bank = torch.as_tensor(train["operators"], device=device)
    train_labels = torch.as_tensor(train["labels"], device=device)
    train_context_tensor = torch.as_tensor(train_context, device=device) if train_context is not None else None
    best_score, stale = -np.inf, 0
    history: list[dict[str, float]] = []
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        decoder.train()
        permutation = torch.randperm(len(train_labels), device=device)
        sums = {"loss": 0.0, "bce": 0.0, "mmd": 0.0, "reconstruction": 0.0}
        batches = 0
        for start in range(0, len(permutation), args.batch):
            index = permutation[start:start + args.batch]
            if len(index) < 2:
                continue
            generator = torch.Generator(device=device)
            generator.manual_seed(args.seed * 1_000_000 + epoch * 100_000 + start)
            if lead1_mode:
                subset_index = torch.zeros((len(index), 1), dtype=torch.long, device=device)
                subset_mask = None
                heldout_leads = torch.ones((len(index), 12), dtype=torch.bool, device=device)
                heldout_leads[:, 0] = False
            else:
                subset_index, subset_mask, heldout_leads = _sample_subsets(len(index), generator, device)
            operators = operator_bank.expand(len(index), -1, -1)
            responses = torch.as_tensor(np.asarray(train["responses"][index.cpu().numpy()]), device=device)
            subset_operators, subset_responses = _gather(operators, subset_index), _gather(responses, subset_index)
            target = torch.as_tensor(np.asarray(train_targets[index.cpu().numpy()]), device=device, dtype=torch.float32)
            patient_context = None
            if train_context_tensor is not None:
                patient_context = train_context_tensor[index]
                if args.variant == "lead1_unconditioned":
                    patient_context = torch.zeros_like(patient_context)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h_subset = model.encode_context(subset_operators, subset_responses, subset_mask, patient_context=patient_context)
                h_full = model.encode_context(operators, responses, patient_context=patient_context)
                loss_bce = bce(model.head(h_subset), train_labels[index])
                loss_reconstruction = heldout_waveform_mse(decoder(h_subset), target, heldout_leads)
            with torch.autocast("cuda", enabled=False):
                loss_mmd = biased_imq_mmd2(view_normalizer(h_subset.float()), view_normalizer(h_full.float()), c2=1.0)
                loss = loss_bce.float() + VARIANTS[args.variant] * loss_mmd + RECONSTRUCTION_WEIGHT * loss_reconstruction.float()
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}, batch {start}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            for name, value in (("loss", loss), ("bce", loss_bce), ("mmd", loss_mmd), ("reconstruction", loss_reconstruction)):
                sums[name] += float(value.detach())
            batches += 1
        if not batches:
            raise RuntimeError("no complete training batches")
        model.eval()
        probabilities = []
        with torch.inference_mode():
            for start in range(0, len(validation["labels"]), 256):
                stop = min(start + 256, len(validation["labels"]))
                responses = torch.as_tensor(np.asarray(validation["responses"][start:stop]), device=device)
                patient_context = None
                if validation_context is not None:
                    patient_context = torch.as_tensor(validation_context[start:stop], device=device)
                    if args.variant == "lead1_unconditioned":
                        patient_context = torch.zeros_like(patient_context)
                if lead1_mode:
                    operators = operator_bank[:1].expand(stop - start, -1, -1)
                    responses = responses[:, :1]
                else:
                    operators = operator_bank.expand(stop - start, -1, -1)
                probabilities.append(torch.sigmoid(model(operators, responses, patient_context=patient_context)).cpu().numpy())
        metrics = multilabel_metrics(validation["labels"], np.concatenate(probabilities))
        score = float(metrics["macro_auroc"])
        epoch_row = {"epoch": float(epoch), "val_macro_auroc": score, **{key: value / batches for key, value in sums.items()}}
        history.append(epoch_row)
        if score > best_score + 1e-6:
            best_score, stale, marker = score, 0, " [BEST]"
            _save_checkpoint(
                args.output, args.variant, model, decoder, optimizer, epoch, metrics, history,
                args.targets / "manifest.json", args.representations / "manifest.json",
                args.patient_context / "manifest.json" if args.patient_context else None,
                "lead_I_only_with_age_sex_context" if lead1_mode else "random_uniform_subset_size_1_through_11_of_standard_12_leads",
            )
        else:
            stale, marker = stale + 1, ""
        print(json.dumps({**epoch_row, "best_val_macro_auroc": best_score, "stale_epochs": stale}) + marker, flush=True)
        if stale >= args.patience:
            break
    summary = {
        "schema_version": "paper07_full12_aux_mmd_training_v2", "variant": args.variant,
        "best_val_macro_auroc": best_score, "epochs_run": len(history),
        "selection_metric": "validation_macro_auroc",
        "target_manifest_sha256": _sha256(args.targets / "manifest.json"),
        "context_manifest_sha256": _sha256(args.representations / "manifest.json"), "history": history,
        "patient_context_manifest_sha256": _sha256(args.patient_context / "manifest.json") if args.patient_context else None,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", **{key: summary[key] for key in ("variant", "best_val_macro_auroc", "epochs_run")} }))


if __name__ == "__main__":
    main()
