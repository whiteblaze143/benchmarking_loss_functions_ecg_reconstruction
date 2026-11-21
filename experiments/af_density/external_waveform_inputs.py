"""External 30-second inputs with evaluation assignments on native sample time."""
import math
import numpy as np
from scipy.signal import resample_poly


def native_window_layout(source_samples, known_start, sampling_rate):
    """Keep a real 30s input and assign the overlapping tail exactly once.

    Initial unannotated time stays outside the known-reference segment, with its
    original offset retained. A suffix shorter than 30s cannot support this model
    input contract and must remain an explicitly unrepresented record.
    """
    if any(not isinstance(v, (int, np.integer)) for v in [source_samples, known_start, sampling_rate]):
        raise ValueError('Native sample coordinates and sampling rate must be integers')
    if sampling_rate not in [128, 200] or not 0 <= known_start < source_samples:
        raise ValueError('Unsupported source clock or invalid known-reference start')
    width = 30*sampling_rate
    if source_samples-known_start < width:
        raise ValueError('Known suffix cannot provide an actual 30-second input')
    inputs = np.arange(known_start, source_samples-width+1, width, dtype=np.int64)
    starts = inputs.copy();ends = inputs+width
    if ends[-1] < source_samples:
        inputs = np.r_[inputs, source_samples-width]
        starts = np.r_[starts, ends[-1]];ends = np.r_[ends, source_samples]
    assert (ends-starts).sum() == source_samples-known_start
    assert np.array_equal(starts[1:], ends[:-1])
    return inputs, starts, ends


def model_window(voltage_mV, sampling_rate):
    """Native mV -> float32 6000x2; no fitted external normalization.

    LTAFDB uses fixed scipy polyphase 25/16, Kaiser beta5, constant-zero filter
    boundary extension within each actual 30s input. This is a resampling edge
    convention, not added observed time. SHDB's released200Hz values are copied.
    A channel with any nonfinite source value remains entirely unavailable;
    neither interpolation nor replacement manufactures an observed channel.
    """
    source = np.asarray(voltage_mV, dtype=np.float32)
    if sampling_rate not in [128, 200] or source.shape != (30*sampling_rate, 2):
        raise ValueError('Expected an actual 30-second two-channel native waveform')
    available = np.isfinite(source).all(axis=0)
    output = np.full((6000, 2), np.nan, dtype=np.float32)
    divisor = math.gcd(200, sampling_rate)
    for c in np.flatnonzero(available):
        values = source[:, c] if sampling_rate == 200 else resample_poly(source[:, c], 200//divisor,
            sampling_rate//divisor, window=('kaiser', 5.), padtype='constant', cval=0.)
        assert values.shape == (6000,) and np.isfinite(values).all()
        output[:, c] = values
    return output, available
