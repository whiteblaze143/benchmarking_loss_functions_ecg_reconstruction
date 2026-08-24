"""Measure representative ECG-AIM training throughput and peak CUDA memory."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from unified_latents.engineering.experimental.alitok_vae_exp import build_alitok_vae_1d


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--decoder-depth", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("profile_output/ecg_aim_gpu.json"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    device = torch.device("cuda")
    results: list[dict[str, float | int | str]] = []

    for batch_size in args.batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            model = build_alitok_vae_1d(
                architecture="ecg_aim_v1",
                target_len=5000,
                patch_size=25,
                encoder_width=args.width,
                decoder_width=args.width,
                encoder_depth=args.encoder_depth,
                decoder_depth=args.decoder_depth,
                encoder_heads=8,
                decoder_heads=8,
                clustering_vq=False,
            ).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95))
            signal = torch.randn(batch_size, 12, 5000, device=device) * 0.15
            observed = torch.tensor([0, 1, 6], device=device).expand(batch_size, -1)
            masked = torch.zeros_like(signal)
            masked[:, [0, 1, 6]] = signal[:, [0, 1, 6]]

            elapsed = 0.0
            for step in range(args.warmup + args.steps):
                optimizer.zero_grad(set_to_none=True)
                if step == args.warmup:
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(masked, target=signal, lead_indices=observed)["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            peak = torch.cuda.max_memory_allocated() / (1024**3)
            results.append(
                {
                    "batch_size": batch_size,
                    "status": "ok",
                    "ecg_per_second": batch_size * args.steps / elapsed,
                    "step_seconds": elapsed / args.steps,
                    "peak_allocated_gib": peak,
                }
            )
            del model, optimizer, signal, observed, masked, loss
        except torch.cuda.OutOfMemoryError:
            results.append({"batch_size": batch_size, "status": "oom"})
        finally:
            torch.cuda.empty_cache()

    payload = {
        "gpu": torch.cuda.get_device_name(0),
        "architecture": "ecg_aim_v1",
        "width": args.width,
        "encoder_depth": args.encoder_depth,
        "decoder_depth": args.decoder_depth,
        "precision": "bfloat16 autocast; fp32 parameters/optimizer",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
