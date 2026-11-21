"""Bounded CPU preparation of original local Phase-KME atoms."""
from collections import deque
import numpy as np
from phase_representation import phase_atoms


def prepare_chunk(task):
    first, waveforms, peaks, valid, accepted = task
    atoms = [];positions = []
    for j in range(len(waveforms)):
        for channel in [0, 1]:
            if not valid[j, channel]: continue
            a = phase_atoms(waveforms[j, :, channel], peaks[j][channel])
            assert a is not None and a.shape[1] == int(accepted[j, channel])*16
            atoms.append(a);positions.append((first+j, channel))
    return atoms, positions


def prepared_chunks(voltage, starts, observed, pool):
    def tasks():
        for first in range(0, len(starts), 8):
            stop = min(first+8, len(starts));peaks = []
            for j in range(first, stop):
                row = []
                for channel in [0, 1]:
                    if not observed['phase_valid'][j, channel]:
                        row.append(np.empty(0, np.int16));continue
                    offsets = observed[f'peak_offsets_{channel}']
                    row.append(observed[f'peak_samples_{channel}'][offsets[j]:offsets[j+1]])
                peaks.append(row)
            x = np.ascontiguousarray(voltage[np.asarray(starts[first:stop])[:, None]+np.arange(6000)])
            yield first, x, peaks, observed['phase_valid'][first:stop], observed['accepted_intervals'][first:stop]
    if pool is None:
        for task in tasks(): yield prepare_chunk(task)
        return
    iterator = iter(tasks());pending = deque()
    # Four production workers, at most eight input/result chunks in flight.
    for _ in range(8):
        task = next(iterator, None)
        if task is None: break
        pending.append(pool.submit(prepare_chunk, task))
    while pending:
        result = pending.popleft().result()
        task = next(iterator, None)
        if task is not None: pending.append(pool.submit(prepare_chunk, task))
        yield result
