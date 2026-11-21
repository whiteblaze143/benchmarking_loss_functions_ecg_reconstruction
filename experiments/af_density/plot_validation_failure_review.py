"""One reproducible, explicitly selected validation missing-RR case figure."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np

W = Path('/data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1')
E = W/'evaluation_protocol_v1'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    rr_receipt = json.loads((E/'validation_rr/receipt.json').read_text())
    population = json.loads((E/'population.json').read_text())
    eligible = {r['file_index'] for r in population['segments'] if r['partition'] == 'validation' and r['primary_eligible']}
    selected = None
    for entry in rr_receipt['outputs']:
        path = E/'validation_rr'/entry['file'];assert sha(path) == entry['sha256']
        data = json.loads(path.read_text())
        if data['file_index'] not in eligible: continue
        for condition in data['conditions']:
            if condition['decoder'] == 'D2' and condition['unknown_windows']:
                selected = path, data, condition;break
        if selected: break
    assert selected is not None
    rr_path, rr, condition = selected;fi = rr['file_index'];pattern = condition['pattern']
    raw_path = E/'validation_raw'/rr_path.name;raw = json.loads(raw_path.read_text())
    raw_receipt = json.loads((E/'validation_raw/receipt.json').read_text())
    assert sha(raw_path) == next(r['sha256'] for r in raw_receipt['outputs'] if r['file'] == raw_path.name)
    raw_condition = next(c for c in raw['conditions'] if c['decoder'] == 'D2' and c['pattern'] == pattern)
    manifest = list(csv.DictReader((W/'file_manifest.csv').open()));row = manifest[fi]
    assert row['partition'] == 'validation' and raw['window_index'] == rr['window_index']
    with np.load(W/'windows.npz') as z:
        ids = np.asarray(rr['window_index']);a = z['assigned_start'][ids];b = z['assigned_end'][ids]
        local = int(np.flatnonzero(np.asarray(condition['states']) == -1)[0]);wi = int(ids[local]);input_start = int(z['input_start'][wi])
    base = int(a[0]);duration = (int(b[-1])-base)/200
    reference = [(max(base, int(r['start_sample']))-base, min(int(b[-1]), int(r['end_sample_exclusive']))-base)
        for r in csv.DictReader((W/'reference_intervals.csv').open()) if int(r['file_index']) == fi and int(r['start_sample']) < b[-1] and int(r['end_sample_exclusive']) > base]
    with h5py.File(row['h5_path']) as h: voltage = h['ecg'][input_start:input_start+6000].astype(np.float32)
    with np.load(W/'observed_features'/f'file_{fi:03d}.npz') as z:
        index = int(np.flatnonzero(z['window_index'] == wi)[0]);peaks = []
        for c in range(2):
            offsets = z[f'peak_offsets_{c}'];peaks.append(z[f'peak_samples_{c}'][offsets[index]:offsets[index+1]].astype(int))
        status = z['detector_status'][index].tolist();rr_valid = z['rr_valid'][index].tolist()
    output = E/'failure_review_v3';output.mkdir(exist_ok=False)
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 10, 'axes.spines.top': False,
        'axes.spines.right': False, 'savefig.dpi': 300})
    fig, axes = plt.subplots(4, 1, figsize=(8, 9), gridspec_kw={'height_ratios': [1, 1, 1.2, 1.2]})
    observed = [[True, False], [False, True], [True, True]][pattern]
    labels = ['Source reference', 'Raw ResNet + HSMM', 'Observed RR + HSMM']
    rail_states = [None, np.asarray(raw_condition['states']), np.asarray(condition['states'])]
    for ax, scale in zip(axes[:2], [3600., 60.]):
        for y, states in zip([2, 1, 0], rail_states):
            ax.broken_barh([(0, duration/scale)], (y-.24, .48), facecolors='white', edgecolors='.65', linewidth=.7)
            if states is None:
                spans = [(lo/200, hi/200, 1) for lo, hi in reference]
            else:
                spans = [((int(lo)-base)/200, (int(hi)-base)/200, int(state)) for lo, hi, state in zip(a, b, states) if state != 0]
            for lo, hi, state in spans:
                ax.broken_barh([(lo/scale, (hi-lo)/scale)], (y-.24, .48),
                    facecolors='#0072B2' if state == 1 else '#E69F00', edgecolors='none' if state == 1 else '.2',
                    hatch=None if state == 1 else '////', linewidth=.4)
        ax.set_yticks([2, 1, 0], labels);ax.set_ylim(-.5, 2.5)
    axes[0].set_xlim(0, duration/3600);axes[0].set_xlabel('Hours since post-calibration observation start')
    center = (input_start+3000-base)/200;lo, hi = max(0, center-180), min(duration, center+180)
    axes[1].set_xlim(lo/60, hi/60);axes[1].set_xlabel('Minutes since post-calibration observation start (detail)')
    axes[1].add_patch(Rectangle(((input_start-base)/200/60, -.4), .5, 2.8, fill=False, edgecolor='black', linewidth=1.1))
    axes[0].legend(handles=[Patch(facecolor='#0072B2', label='AF'), Patch(facecolor='white', edgecolor='.65', label='Non-AF'),
        Patch(facecolor='#E69F00', edgecolor='.2', hatch='////', label='Unknown; not non-AF')], loc='lower center', bbox_to_anchor=(.5, 1.08), ncol=3, frameon=False)
    t = np.arange(6000)/200
    for c, ax in enumerate(axes[2:]):
        ax.plot(t, voltage[:, c], color='black' if observed[c] else '.5', linewidth=.6)
        if observed[c] and len(peaks[c]): ax.scatter(peaks[c]/200, voltage[peaks[c], c], marker='v', color='#D55E00', s=12, zorder=3)
        availability_label = 'Observed by models' if observed[c] else 'Unavailable to models'
        ax.set_xlim(0, 30);ax.set_ylabel(f'Stored channel {c+1}\n{availability_label}\nVoltage (mV)', fontsize=9)
        ax.set_xlabel('Seconds within actual 30-second model input')
    fig.subplots_adjust(left=.26, right=.98, top=.93, bottom=.07, hspace=.65)
    fig.savefig(output/'validation_failure_review.pdf', bbox_inches='tight')
    fig.savefig(output/'validation_failure_review.png', bbox_inches='tight');plt.close(fig)
    observed_description = '; '.join(f'stored channel {c+1} ({len(peaks[c])} detected QRS peaks)' for c in range(2) if observed[c])
    caption = ('One daily segment from one validation patient in the frozen informative multi-episode population, not a cohort estimate. Selection: first committed primary-eligible validation daily file/availability condition with an HSMM RR timeline containing unknown windows, then its first unknown window. '
        'Upper panels show the same post-calibration source clock for reference, raw ResNet and observed-RR HGB, with the same fixed HSMM policy. Blue denotes AF, white non-AF, and hatched orange unknown. The outlined 30-second input is shown below at original 200 Hz in mV; triangles mark detected QRS peaks only on channels available to the models. '
        f'Available to both models: {observed_description}. '
        'RR validity requires at least 4 intervals of 200--4000 ms and at least 2 originally adjacent valid pairs; these requirements were not met. Unknown intervals are too narrow to resolve on the full-day axis; the detail panel shows their orange hatching. '
        'Source labels are not new clinical adjudication. No clinical failure mechanism or SetOperator advantage is asserted. ')
    (output/'caption.md').write_text(caption+'\n')
    (output/'latex_includes.tex').write_text('\\begin{figure}[t]\n\\centering\n\\includegraphics[width=0.95\\textwidth]{validation_failure_review.pdf}\n\\caption{'+caption.replace('200Hz', '200 Hz')+'}\n\\end{figure}\n')
    shutil.copyfile(__file__, output/Path(__file__).name)
    (output/'case.json').write_text(json.dumps({'partition': 'validation', 'patient_id': row['patient_id'], 'file_index': fi,
        'window_index': wi, 'pattern': pattern, 'primary_eligible': True, 'population_sha256': sha(E/'population.json'), 'observed_channels': observed, 'detector_status_by_channel': status,
        'RR_valid_by_channel': rr_valid, 'peak_counts': [len(p) for p in peaks], 'raw_archive_sha256': sha(raw_path),
        'RR_archive_sha256': sha(rr_path), 'source_waveform_sha256': row['h5_sha256'], 'script_sha256': sha(__file__),
        'raw_density': raw_condition['results']['predicted_patterns']['density_author_binned_30s_max'],
        'RR_density': None, 'reference_density': condition['results']['reference_patterns']['density_author_binned_30s_max']}, indent=2)+'\n')
    print(output)


if __name__ == '__main__': main()
