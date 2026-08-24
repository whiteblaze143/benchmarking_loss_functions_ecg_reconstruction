#!/usr/bin/env python3
"""
Build a physiology-inspired ventricular-repolarization mother-wavelet bank
from the simple unipolar electrogram (UEG) model discussed by Potse et al.
and summarized/validated by Stoks et al. (2023).

Core physiological model
------------------------
UEG(t) ∝ mean_surface_TMP(t) - TMP_local(t)

The mother-wavelet template is constructed as the difference between:
  1) a local transmembrane repolarization trajectory, and
  2) an ensemble-average repolarization trajectory from neighboring tissue
     with a distribution of recovery times.

The sign follows the inverted/scaled Potse/Orini formulation under the
positive-plateau logistic convention used here.  It is what makes relatively
early local recovery positive and relatively late recovery negative.

This produces positive, biphasic, or negative T-like UEG morphology depending
on whether local recovery is early, intermediate, or late relative to the
surrounding myocardium.

The real UEG template is converted to an analytic complex wavelet via a Hilbert
transform so that its PHASE can be used as the second CWT view in the
physiology-informed BYOL framework.

IMPORTANT
---------
This is a mechanistically derived reproduction candidate, NOT a claim that it
is the exact unpublished CinC 2026 mother-wavelet implementation. The abstract
does not disclose its exact equation/parameters.

Output .pt schema is compatible with FixedComplexKernelBank:
{
    "real": [F,K],
    "imag": [F,K],
    "metadata": {...}
}

Suggested use:
python build_repolarization_ueg_wavelet.py \
  --output repolarization_ueg_wavelets.pt \
  --sample-rate 500 \
  --kernel-ms 900 \
  --n-kernels 32
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


def logistic_repolarization(
    t_ms: torch.Tensor,
    rt_ms: float,
    width_ms: float,
    plateau: float = 1.0,
) -> torch.Tensor:
    """Simple normalized TMP repolarization trajectory.

    Returns approximately:
      plateau before local RT
      0 after local RT

    The negative dV/dt peak occurs close to rt_ms.
    """
    width_ms = max(float(width_ms), 1e-3)
    return float(plateau) / (1.0 + torch.exp((t_ms - float(rt_ms)) / width_ms))


def average_tmp(
    t_ms: torch.Tensor,
    mean_rt_ms: float,
    dispersion_ms: float,
    width_ms: float,
    *,
    n_neighbors: int = 81,
    skew: float = 0.0,
) -> torch.Tensor:
    """Approximate mean myocardial TMP using distributed recovery times.

    Recovery times are sampled deterministically over a normal-like support.
    `skew` warps the quantile locations to generate asymmetric repolarization.
    """
    if n_neighbors < 3:
        raise ValueError("n_neighbors must be >= 3")
    q = torch.linspace(-3.0, 3.0, n_neighbors, dtype=t_ms.dtype, device=t_ms.device)

    # Smooth deterministic skew warp while retaining ordered recovery times.
    if abs(skew) > 1e-12:
        q = q + float(skew) * (q.square() - q.square().mean()) / 3.0

    rt = float(mean_rt_ms) + float(dispersion_ms) * q
    tmp = torch.stack(
        [logistic_repolarization(t_ms, float(r), width_ms) for r in rt],
        dim=0,
    )

    # Gaussian quadrature-like weights emphasize central myocardium.
    w = torch.exp(-0.5 * q.square())
    w = w / w.sum()
    return (tmp * w[:, None]).sum(dim=0)


def tukey_window(n: int, alpha: float = 0.35) -> torch.Tensor:
    """Torch-only Tukey window."""
    if n <= 1:
        return torch.ones(n)
    if alpha <= 0:
        return torch.ones(n)
    if alpha >= 1:
        return torch.hann_window(n, periodic=False)
    x = torch.linspace(0.0, 1.0, n)
    w = torch.ones(n)
    first = x < alpha / 2
    last = x >= (1 - alpha / 2)
    w[first] = 0.5 * (1 + torch.cos(math.pi * (2 * x[first] / alpha - 1)))
    w[last] = 0.5 * (1 + torch.cos(math.pi * (2 * x[last] / alpha - 2 / alpha + 1)))
    return w


def analytic_signal(real: torch.Tensor) -> torch.Tensor:
    """Return analytic signal using FFT Hilbert construction."""
    n = real.numel()
    X = torch.fft.fft(real)
    h = torch.zeros(n, dtype=real.dtype, device=real.device)
    if n % 2 == 0:
        h[0] = 1.0
        h[n // 2] = 1.0
        h[1:n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1:(n + 1) // 2] = 2.0
    return torch.fft.ifft(X * h)


def make_ueg_mother(
    *,
    sample_rate_hz: float,
    kernel_ms: float,
    local_offset_ms: float,
    mean_rt_ms: float,
    dispersion_ms: float,
    repol_width_ms: float,
    skew: float = 0.0,
    n_neighbors: int = 81,
    taper_alpha: float = 0.35,
    derivative_mix: float = 0.0,
) -> torch.Tensor:
    """Construct one complex physiology-inspired UEG mother wavelet.

    local_offset_ms < 0  -> local tissue repolarizes earlier -> positive T-like UEG
    local_offset_ms ~ 0  -> intermediate/biphasic
    local_offset_ms > 0  -> late local repolarization -> negative T-like UEG

    derivative_mix optionally adds a normalized first derivative component.
    This can sharpen Wyatt-relevant maximum-upslope structure while preserving
    the local-minus-average physiological template.
    """
    n = int(round(float(sample_rate_hz) * float(kernel_ms) / 1000.0))
    if n < 32:
        raise ValueError("kernel too short")

    # Center the global mean repolarization within the kernel.
    t_ms = torch.arange(n, dtype=torch.float32) * (1000.0 / float(sample_rate_hz))
    center = kernel_ms / 2.0
    shift = center - float(mean_rt_ms)
    t_phys = t_ms - shift

    local_rt = float(mean_rt_ms) + float(local_offset_ms)
    local = logistic_repolarization(t_phys, local_rt, repol_width_ms)
    global_mean = average_tmp(
        t_phys,
        mean_rt_ms,
        dispersion_ms,
        repol_width_ms,
        n_neighbors=n_neighbors,
        skew=skew,
    )

    # Potse/Orini simple UEG model, including its polarity inversion:
    # UEG = -alpha * (local TMP - location-independent mean TMP).
    # The positive scale alpha is immaterial after unit-energy normalization.
    ueg = global_mean - local

    # Optional Wyatt-sensitive sharpening: steepest T-wave upslope is the
    # physiologically supported RT marker. Keep this as a tunable ablation,
    # not a hidden default.
    if derivative_mix != 0.0:
        dt = 1000.0 / float(sample_rate_hz)
        deriv = torch.zeros_like(ueg)
        deriv[1:-1] = (ueg[2:] - ueg[:-2]) / (2 * dt)
        deriv[0] = (ueg[1] - ueg[0]) / dt
        deriv[-1] = (ueg[-1] - ueg[-2]) / dt
        deriv = deriv / deriv.abs().max().clamp_min(1e-8)
        base = ueg / ueg.abs().max().clamp_min(1e-8)
        ueg = base + float(derivative_mix) * deriv

    # Wavelet admissibility/stability:
    # - taper
    # - exact zero DC
    # - L2 normalization
    ueg = ueg * tukey_window(n, taper_alpha).to(ueg)
    ueg = ueg - ueg.mean()
    ueg = ueg / torch.sqrt(ueg.square().sum().clamp_min(1e-12))

    analytic = analytic_signal(ueg)
    analytic = analytic - analytic.mean()
    analytic = analytic / torch.sqrt(
        (analytic.real.square() + analytic.imag.square()).sum().clamp_min(1e-12)
    )
    return analytic


def build_bank(
    *,
    sample_rate_hz: float = 500.0,
    kernel_ms: float = 900.0,
    n_kernels: int = 32,
    mean_rt_ms: float = 320.0,
    offset_min_ms: float = -120.0,
    offset_max_ms: float = 120.0,
    dispersion_min_ms: float = 20.0,
    dispersion_max_ms: float = 90.0,
    repol_width_min_ms: float = 8.0,
    repol_width_max_ms: float = 35.0,
    skew_min: float = -0.35,
    skew_max: float = 0.35,
    derivative_mix: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """Create a bank spanning early/intermediate/late repolarization morphology.

    The sweep intentionally changes physiology, not just arbitrary scale:
      - local-vs-global RT offset
      - spatial RT dispersion
      - cellular repolarization transition width
      - mild distribution skew

    This gives the wavelet encoder multiple morphology-sensitive basis
    functions instead of simple dilations of one arbitrary waveform.
    """
    if n_kernels < 3:
        raise ValueError("n_kernels must be >=3")

    u = torch.linspace(0.0, 1.0, n_kernels)
    offsets = torch.linspace(offset_min_ms, offset_max_ms, n_kernels)

    # Orthogonal-ish slow sweeps so neighboring kernels do not vary every
    # parameter in lockstep.
    dispersions = dispersion_min_ms + (dispersion_max_ms - dispersion_min_ms) * (
        0.5 - 0.5 * torch.cos(2 * math.pi * u)
    )
    widths = repol_width_min_ms + (repol_width_max_ms - repol_width_min_ms) * (
        0.5 + 0.5 * torch.sin(2 * math.pi * u - math.pi / 2)
    )
    skews = torch.linspace(skew_min, skew_max, n_kernels)

    kernels = []
    rows = []
    for i in range(n_kernels):
        z = make_ueg_mother(
            sample_rate_hz=sample_rate_hz,
            kernel_ms=kernel_ms,
            local_offset_ms=float(offsets[i]),
            mean_rt_ms=mean_rt_ms,
            dispersion_ms=float(dispersions[i]),
            repol_width_ms=float(widths[i]),
            skew=float(skews[i]),
            derivative_mix=derivative_mix,
        )
        kernels.append(z)
        rows.append({
            "index": i,
            "local_offset_ms": float(offsets[i]),
            "dispersion_ms": float(dispersions[i]),
            "repol_width_ms": float(widths[i]),
            "skew": float(skews[i]),
        })

    bank = torch.stack(kernels, dim=0)

    metadata = {
        "name": "potse_stoks_repolarization_ueg_wavelet_v1",
        "status": "mechanistically_derived_not_exact_cinc2026_reproduction",
        "physiological_equation": "UEG(t) proportional to mean_surface_TMP(t) - TMP_local(t)",
        "polarity_convention": "inverted Potse/Orini form: early local RT positive, late local RT negative",
        "repolarization_marker": "Wyatt maximum T-wave upslope",
        "sample_rate_hz": float(sample_rate_hz),
        "kernel_ms": float(kernel_ms),
        "kernel_samples": int(bank.shape[-1]),
        "n_kernels": int(n_kernels),
        "mean_rt_ms": float(mean_rt_ms),
        "offset_range_ms": [float(offset_min_ms), float(offset_max_ms)],
        "dispersion_range_ms": [float(dispersion_min_ms), float(dispersion_max_ms)],
        "repol_width_range_ms": [float(repol_width_min_ms), float(repol_width_max_ms)],
        "skew_range": [float(skew_min), float(skew_max)],
        "derivative_mix": float(derivative_mix),
        "analytic_conversion": "FFT Hilbert transform",
        "normalization": "Tukey taper + zero mean + unit L2",
        "kernel_parameters": rows,
    }
    return bank, metadata


def validate_bank(bank: torch.Tensor, sample_rate_hz: float) -> dict:
    """Numerical sanity checks relevant to wavelet admissibility and RT shape."""
    real = bank.real
    imag = bank.imag
    dc = real.mean(-1).abs()
    energy = (real.square() + imag.square()).sum(-1)
    finite = torch.isfinite(real).all() and torch.isfinite(imag).all()

    # Peak positive derivative timing in each real template.
    dt_ms = 1000.0 / sample_rate_hz
    deriv = torch.diff(real, dim=-1) / dt_ms
    max_up_idx = deriv.argmax(-1)
    max_up_ms = max_up_idx.float() * dt_ms

    return {
        "finite": bool(finite),
        "max_abs_real_dc": float(dc.max()),
        "min_complex_energy": float(energy.min()),
        "max_complex_energy": float(energy.max()),
        "max_upslope_time_ms_min": float(max_up_ms.min()),
        "max_upslope_time_ms_max": float(max_up_ms.max()),
    }


def save_asset(path: Path, bank: torch.Tensor, metadata: dict) -> str:
    payload = {
        "real": bank.real.float().cpu(),
        "imag": bank.imag.float().cpu(),
        "metadata": metadata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--sample-rate", type=float, default=500.0)
    ap.add_argument("--kernel-ms", type=float, default=900.0)
    ap.add_argument("--n-kernels", type=int, default=32)
    ap.add_argument("--mean-rt-ms", type=float, default=320.0)
    ap.add_argument("--offset-min-ms", type=float, default=-120.0)
    ap.add_argument("--offset-max-ms", type=float, default=120.0)
    ap.add_argument("--dispersion-min-ms", type=float, default=20.0)
    ap.add_argument("--dispersion-max-ms", type=float, default=90.0)
    ap.add_argument("--repol-width-min-ms", type=float, default=8.0)
    ap.add_argument("--repol-width-max-ms", type=float, default=35.0)
    ap.add_argument("--skew-min", type=float, default=-0.35)
    ap.add_argument("--skew-max", type=float, default=0.35)
    ap.add_argument(
        "--derivative-mix",
        type=float,
        default=0.0,
        help="Optional Wyatt-upslope sharpening; keep 0 for the pure Potse/Stoks template."
    )
    ap.add_argument("--metadata-json", default=None)
    args = ap.parse_args()

    bank, metadata = build_bank(
        sample_rate_hz=args.sample_rate,
        kernel_ms=args.kernel_ms,
        n_kernels=args.n_kernels,
        mean_rt_ms=args.mean_rt_ms,
        offset_min_ms=args.offset_min_ms,
        offset_max_ms=args.offset_max_ms,
        dispersion_min_ms=args.dispersion_min_ms,
        dispersion_max_ms=args.dispersion_max_ms,
        repol_width_min_ms=args.repol_width_min_ms,
        repol_width_max_ms=args.repol_width_max_ms,
        skew_min=args.skew_min,
        skew_max=args.skew_max,
        derivative_mix=args.derivative_mix,
    )
    checks = validate_bank(bank, args.sample_rate)
    metadata["validation"] = checks

    out = Path(args.output)
    final_sha = save_asset(out, bank, metadata)
    print(json.dumps({
        "output": str(out.resolve()),
        "shape": list(bank.shape),
        "final_sha256": final_sha,
        "validation": checks,
        "warning": "Mechanistically derived candidate; not exact CinC 2026 wavelet unless author implementation confirms it."
    }, indent=2))

    if args.metadata_json:
        metadata_sidecar = {**metadata, "asset_sha256": final_sha}
        Path(args.metadata_json).write_text(json.dumps(metadata_sidecar, indent=2))


if __name__ == "__main__":
    main()
