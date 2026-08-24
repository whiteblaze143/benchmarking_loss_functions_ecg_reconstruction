import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from unified_latents.engineering.experimental.wavelet_ssl_ecg_aim import (
    ECGAdmissibleMorletFilterBank,
    ECGDelineationHead,
    FixedComplexKernelBank,
    WaveletViewBuilder,
)
from scripts.train_1lead_wavelet_ssl_mtl import broad_cells


def test_principal_wavelet_and_complementarity_cells_are_matched():
    cells = {cell["name"]: cell for cell in broad_cells()}
    parent = cells["P1_A0_morlet_mag_phase_noSSL"]
    morlet = cells["R0_morlet_mag_morlet_phase"]
    ueg = cells["R1_morlet_mag_ueg_phase"]
    e1 = cells["C1_E1_morlet_mag_morlet_phase"]

    assert parent["wavelet_bank"] == morlet["wavelet_bank"] == "morlet"
    assert parent["view_a"] == morlet["view_a"] == "magnitude"
    assert parent["view_b"] == morlet["view_b"] == "phase"
    assert parent["ssl_mode"] == "none" and morlet["ssl_mode"] == "both"

    permitted_ueg = {"name", "view_a_bank", "view_b_bank", "view_b_custom_wavelet_asset"}
    assert {key for key in morlet | ueg if morlet.get(key) != ueg.get(key)} == permitted_ueg

    permitted_e1 = {"name", "lead_conditioning_mode", "use_relative_geometry", "use_spatial_film"}
    assert {key for key in morlet | e1 if morlet.get(key) != e1.get(key)} == permitted_e1
from scripts.build_repolarization_ueg_wavelet import build_bank, validate_bank


def test_admissible_morlet_has_zero_dc_and_equal_filter_energy():
    bank = ECGAdmissibleMorletFilterBank(
        sample_rate_hz=500, target_len=5000, n_scales=12,
        min_freq_hz=0.5, max_freq_hz=45, cycles=6,
    )
    response = bank.frequency_response()
    assert torch.equal(response[:, 0], torch.zeros(12))
    assert torch.allclose(response.square().sum(-1), torch.ones(12), atol=1e-6)


def test_admissible_morlet_peaks_near_requested_frequencies():
    bank = ECGAdmissibleMorletFilterBank(
        sample_rate_hz=500, target_len=5000, n_scales=8,
        min_freq_hz=1, max_freq_hz=40, cycles=6,
    )
    response = bank.frequency_response()
    bins = torch.fft.fftfreq(5000, d=1 / 500)
    peaks = bins[response.argmax(-1)]
    assert torch.all((peaks - bank.center_frequencies_hz).abs() <= 0.11)


def test_admissible_morlet_rejects_constant_signal_and_is_deterministic():
    bank = ECGAdmissibleMorletFilterBank(target_len=500, n_scales=6)
    x = torch.ones(2, 500)
    first = bank(x)
    second = bank(x)
    assert torch.equal(first, second)
    assert first.abs().max().item() < 1e-6


def test_delineation_head_accepts_transposed_layout():
    head = ECGDelineationHead(width=24, patch_size=5, hidden=8, kernel=3)
    seg, fid = head(torch.randn(2, 3, 20, 24))
    assert seg.shape == (2, 3, 4, 100)
    assert fid.shape == (2, 3, 6, 100)


def test_potse_stoks_bank_polarity_admissibility_and_views(tmp_path):
    bank, metadata = build_bank(n_kernels=8)
    checks = validate_bank(bank, 500)
    assert checks["finite"]
    assert checks["max_abs_real_dc"] < 1e-7
    assert checks["min_complex_energy"] > 0.999
    # Under the generator's positive-plateau convention, early recovery is
    # positive and late recovery is negative after the Potse/Orini inversion.
    assert bank[0].real.max() > bank[0].real.min().abs()
    assert bank[-1].real.min().abs() > bank[-1].real.max()
    asset = tmp_path / "ueg.pt"
    torch.save({"real": bank.real, "imag": bank.imag, "metadata": metadata}, asset)
    builder = WaveletViewBuilder(FixedComplexKernelBank(asset, target_len=5000))
    x = torch.randn(2, 5000)
    assert builder(x, "phase_sin").shape == (2, 1, 8, 200)
