"""Authentic local Phase-KME adapter; no fitting, detection or synthetic beats."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                       'ptbxl_distributional_repecg' / 'src'))
from repecg.common.phase import phase_normalize


def phase_atoms(signal_mv, peak_samples):
    """Return (16 cells, accepted beats * 16 phase samples, voltage/derivative).

    Signal is one actual 30-second channel, already expressed in mV. Derivative
    is per phase-sample index, as in the original representation, not mV/second.
    Only successive observed peaks 300--2000 ms apart enter the representation.
    Fewer than two accepted intervals returns None, never manufactured beats.
    """
    x = np.asarray(signal_mv, dtype=np.float32)
    peaks = np.asarray(peak_samples)
    if x.shape != (6000,) or not np.isfinite(x).all():
        raise ValueError('Expected finite 6000-sample observed channel in mV')
    if peaks.ndim != 1 or peaks.dtype.kind not in 'iu':
        raise ValueError('Expected integer one-dimensional local peaks')
    peaks = peaks.astype(np.int64)
    if len(peaks) and (peaks[0] < 0 or peaks[-1] >= 6000 or
                       not (np.diff(peaks) > 0).all()):
        raise ValueError('Invalid local peak coordinates')
    lengths = np.diff(peaks)
    accepted = (lengths >= 60) & (lengths <= 400)
    if accepted.sum() < 2:
        return None
    intervals = np.column_stack((peaks[:-1], peaks[1:]))[accepted]
    beats = phase_normalize(x[:, None], intervals, samples=256).astype(np.float32)
    voltage = beats[:, :, 0]
    atoms = np.stack((voltage, np.gradient(voltage, axis=-1)), axis=-1)
    return atoms.reshape(len(beats), 16, 16, 2).transpose(1, 0, 2, 3).reshape(16, -1, 2)


def phase_response(atoms, whitening, mapping):
    """Apply already fitted authentic transforms; return float32 (16,128)."""
    values = np.asarray(atoms)
    if (values.ndim != 3 or values.shape[0] != 16 or values.shape[2] != 2 or
            values.shape[1] < 32 or values.shape[1] % 16 or
            not np.isfinite(values).all()):
        raise ValueError('Expected finite phase atoms from at least two beats')
    if mapping.landmarks.shape != (128, 2):
        raise ValueError('Authentic 128-landmark two-coordinate map required')
    response = np.stack([mapping.mean(whitening.transform(cell)) for cell in values])
    if response.shape != (16, 128) or not np.isfinite(response).all():
        raise ValueError('Invalid Phase-KME response')
    return response.astype(np.float32)
