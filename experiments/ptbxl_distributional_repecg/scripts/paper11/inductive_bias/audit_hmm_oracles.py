"""Diagnostic only: explain why a four-label HMM can favor an eight-state neural bottleneck."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from repecg.paper11_predictive_state import (
    hmm_filter_beliefs,
    hmm_next_mean_from_belief,
    hmm_next_mean_from_state,
    make_hmm_sequences,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper 11 frozen-HMM oracle diagnostic")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sequence, latent = make_hmm_sequences(2048, 20, args.seed + 1, device, states=4)
    target = sequence[:, 4:].reshape(-1, sequence.shape[-1])
    copy = sequence[:, 3:-1].reshape_as(target)
    true_state = hmm_next_mean_from_state(latent, states=4)[:, 3:].reshape_as(target)
    belief = hmm_next_mean_from_belief(hmm_filter_beliefs(sequence, states=4), states=4)[:, 3:].reshape_as(target)
    payload = {
        "kind": "paper11_hmm_oracle_diagnostic",
        "seed": args.seed,
        "device": str(device),
        "true_state_oracle_mse": float(F.mse_loss(true_state, target)),
        "filtering_belief_oracle_mse": float(F.mse_loss(belief, target)),
        "beat_copy_mse": float(F.mse_loss(copy, target)),
        "interpretation": "Diagnostic only. It neither changes nor rescues the failed categorical G1 gate.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
