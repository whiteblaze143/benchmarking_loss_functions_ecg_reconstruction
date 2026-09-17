#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from repecg.common.variants import ExperimentVariant
from repecg.paper08_tokens import PhaseTokenTransformer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    with np.load(args.representations / "representation_val.npz") as item:
        values = np.asarray(item["kernel"][: args.records], dtype=np.float32)
    x = torch.from_numpy(values)
    reversed_x = x.flip(1)
    permutation = torch.tensor([7, 3, 12, 0, 15, 2, 9, 5, 1, 14, 6, 11, 4, 10, 8, 13])

    dynamic = PhaseTokenTransformer(x.shape[-1], mode="global").eval()
    static = PhaseTokenTransformer(x.shape[-1], mode="static").eval()
    agnostic = PhaseTokenTransformer(x.shape[-1], mode="phase_agnostic").eval()
    cnn = PhaseTokenTransformer(
        x.shape[-1], variant=ExperimentVariant(mechanism="cnn_matched")
    ).eval()
    with torch.inference_mode():
        dynamic_logits, dynamic_attention = dynamic(x, return_attention=True)
        dynamic_reversed = dynamic(reversed_x)
        _, dynamic_attention_reversed = dynamic(reversed_x, return_attention=True)
        _, static_attention = static(x, return_attention=True)
        _, static_attention_reversed = static(reversed_x, return_attention=True)
        agnostic_error = float((agnostic(x) - agnostic(x[:, permutation])).abs().max())
        cnn_logits = cnn(x)

    dynamic_attention_shift = float(
        torch.linalg.vector_norm(dynamic_attention[0] - dynamic_attention_reversed[0])
        / np.sqrt(dynamic_attention[0].numel())
    )
    static_attention_shift = float(
        (static_attention[0] - static_attention_reversed[0]).abs().max()
    )
    dynamic_phase_reversal_shift = float(
        torch.linalg.vector_norm(dynamic_logits - dynamic_reversed) / np.sqrt(dynamic_logits.numel())
    )
    dynamic_count = sum(p.numel() for p in dynamic.parameters() if p.requires_grad)
    static_count = sum(p.numel() for p in static.parameters() if p.requires_grad)
    cnn_count = sum(p.numel() for p in cnn.parameters() if p.requires_grad)
    checks = {
        "finite_input": bool(np.isfinite(values).all()),
        "dynamic_attention_record_dependent": dynamic_attention_shift > 1e-6,
        "static_attention_input_independent": static_attention_shift < 1e-7,
        "phase_agnostic_permutation_invariant": agnostic_error < 1e-5,
        "cnn_is_distinct_and_finite": bool(torch.isfinite(cnn_logits).all() and cnn.cnn is not None),
        "static_capacity_within_5pct": abs(static_count - dynamic_count) / dynamic_count < 0.05,
        "cnn_capacity_within_15pct": abs(cnn_count - dynamic_count) / dynamic_count < 0.15,
    }
    payload = {
        "kind": "paper08_quick_inductive_bias_audit",
        "scientific_status": "engineering_only_not_a_result",
        "passed": all(checks.values()),
        "records": len(values),
        "checks": checks,
        "measurements": {
            "dynamic_attention_rms_shift_under_phase_reversal": dynamic_attention_shift,
            "static_attention_max_shift_under_phase_reversal": static_attention_shift,
            "dynamic_logit_rms_shift_under_phase_reversal": dynamic_phase_reversal_shift,
            "phase_agnostic_max_permutation_error": agnostic_error,
            "trainable_parameters": {
                "dynamic": dynamic_count, "static": static_count, "cnn": cnn_count,
            },
        },
        "equivalence_vocabulary": {
            "ready": False,
            "reason": "UCB equivalence distance and fitted vocabulary artifact remain unresolved",
        },
        "source": {
            "manifest_sha256": _sha256(args.representations / "manifest.json"),
            "representation": "kernel",
            "split": "development_selection_fold_8",
        },
        "seed": args.seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
