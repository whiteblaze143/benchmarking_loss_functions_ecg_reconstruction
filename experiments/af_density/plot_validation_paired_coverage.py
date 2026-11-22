"""Visualize strict patient-level validation errors without imputing missing values."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Patch, Rectangle
import numpy as np

W = Path('/data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1')
MODELS = [('Raw ECG', 'raw'), ('Observed RR', 'rr'), ('Set: channel ID', 'math_v2_categorical'),
          ('Set: no channel ID', 'math_v2_q_ablated'), ('Hybrid', 'math_v2_hybrid')]
METRICS = [('density_author_binned_absolute_error', 'A  AF density absolute error'),
           ('burden_absolute_error', 'B  AF burden absolute error')]


def main():
    out = W/'evaluation_protocol_v1/validation_paired_coverage_figure_v1'
    out.mkdir(exist_ok=False)
    docs = [json.loads((W/f'evaluation_protocol_v1/validation_summary_{key}.json').read_text()) for _, key in MODELS]
    patients = sorted(docs[0]['decoders']['D2']['patient_values'][METRICS[0][0]])
    assert len(patients) == 11 and all(doc['informative_patients'] == 11 for doc in docs)
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), layout='constrained', sharey=True)
    fig.get_layout_engine().set(rect=(0, .14, 1, .86))
    cmap = plt.get_cmap('YlOrRd').copy();cmap.set_bad('#d9d9d9')
    norm = colors.Normalize(vmin=0, vmax=1)
    for ax, (metric, title) in zip(axes, METRICS):
        values = np.empty((len(patients), len(docs)))
        for col, doc in enumerate(docs):
            m = doc['decoders']['D2']['patient_values'][metric]
            assert sorted(m) == patients
            values[:, col] = [np.nan if m[p] is None else m[p] for p in patients]
        ax.imshow(np.ma.masked_invalid(values), aspect='auto', cmap=cmap, norm=norm)
        for i in range(len(patients)):
            for j in range(len(docs)):
                v = values[i, j]
                if np.isnan(v):
                    ax.add_patch(Rectangle((j-.5, i-.5), 1, 1, facecolor='none', edgecolor='#888', hatch='///', linewidth=.25))
                    ax.text(j, i, '—', ha='center', va='center', color='#444', fontsize=10)
                else:
                    ax.text(j, i, f'{v:.2f}', ha='center', va='center', color='white' if v > .56 else 'black', fontsize=10)
        counts = np.isfinite(values).sum(axis=0)
        ax.set_xticks(range(len(docs)), [f'{name}\n{n}/11' for (name, _), n in zip(MODELS, counts)], fontsize=9)
        ax.set_yticks(range(len(patients)), [p.replace('patient_', 'P') for p in patients], fontsize=9)
        ax.set_title(title, loc='left', fontweight='bold', fontsize=12)
        ax.tick_params(length=0)
        ax.set_xticks(np.arange(-.5, len(docs), 1), minor=True);ax.set_yticks(np.arange(-.5, len(patients), 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=1.5);ax.tick_params(which='minor', bottom=False, left=False)
    axes[0].set_ylabel('Pseudonymous validation patient')
    fig.suptitle('AF-density development results: patient-level errors and missing coverage', fontsize=15, fontweight='bold')
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, label='Absolute error (0 = exact; 1 = largest)', shrink=.75, location='right')
    fig.legend(handles=[Patch(facecolor='#d9d9d9', edgecolor='#888', hatch='///', label='Indeterminate: at least one required condition missing')],
               loc='outside lower center' if matplotlib.__version__ >= '3.7' else 'lower center', bbox_to_anchor=(.5, .075), frameon=False, fontsize=10)
    fig.text(.5, .035, 'Validation only · D2 duration-aware decoder · 15 segments × 3 channel patterns · patient means · no imputation',
             ha='center', fontsize=9)
    for ext in ('png', 'pdf'): fig.savefig(out/f'paired_patient_errors.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    inputs = [W/f'evaluation_protocol_v1/validation_summary_{key}.json' for _, key in MODELS]
    receipt = {'partition': 'validation', 'test_access': False, 'patients': len(patients),
               'segments': 15, 'channel_conditions_per_segment': 3, 'decoder': 'D2',
               'indeterminate_policy': 'Hatched and excluded from model-specific complete-case means; no imputation.',
               'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'input_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
               'scope': 'Development-only descriptive visualization; not a paired confidence interval or a test of clinical superiority.'}
    (out/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(out)


if __name__ == '__main__': main()
