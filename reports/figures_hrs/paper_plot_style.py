"""Shared HRS Publication Style & Clinical Visualization Utilities.

Designed for Heart Rhythm Society (HRS) clinical readability:
- Clean sans-serif typography with Type-42 editable vector PDF fonts
- Okabe-Ito colorblind-safe clinical palette
- Zero interior plot titles (panel letters A, B, C, D only)
- Minimal, non-overlapping legends and open top/right spines
- Calibrated 25 mm/s, 10 mm/mV clinical ECG grid renderer aligned with ecg-plot (v0.3.0)
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.signal import butter, filtfilt

try:
    import ecg_plot
    ECG_STYLE = ecg_plot.Style()
except Exception:
    class _FallbackStyle:
        line_color = '#172331'
        major_color = '#e6b8b8'
        minor_color = '#f8ebeb'
    ECG_STYLE = _FallbackStyle()

# Print constants for Heart Rhythm / HRS journals
DPI = 300
FONT_SIZE = 9.5
SINGLE_COL_W = 3.5   # inches
DOUBLE_COL_W = 7.2   # inches
WIDE_PANEL_W = 10.5  # inches (comfortable aspect, scales cleanly to 7.2 in)

OUT_DIR = Path('reports/figures_hrs')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Okabe-Ito colorblind-safe clinical palette
PALETTE = {
    'navy': '#0072B2',       # IRIDIA-AF / Primary Model / Pre-onset
    'vermilion': '#D55E00',  # LTAFDB / Post-termination / Sustained AF
    'teal': '#009E73',       # SHDB-AF / Clean ECG / Net Benefit
    'amber': '#E69F00',      # Patient-averaged / N1 Mild Noise
    'purple': '#CC79A7',     # Non-linear GAM / N2 Severe Noise
    'sky': '#56B4E9',        # Light accent / secondary curve
    'slate': '#4D4D4D',      # Baseline / Null / Treat None
    'light_gray': '#BDBDBD', # Subtle reference lines
}

COHORT_COLORS = {
    'IRIDIA-AF': PALETTE['navy'],
    'LTAFDB': PALETTE['vermilion'],
    'SHDB-AF': PALETTE['teal'],
    'Noise-Holter': PALETTE['amber'],
}

COHORT_LABELS = {
    'IRIDIA-AF': 'IRIDIA-AF',
    'LTAFDB': 'LTAFDB',
    'SHDB-AF': 'SHDB-AF',
}


def apply_hrs_style():
    """Apply clean, low-clutter HRS clinical publication rcParams."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size': FONT_SIZE,
        'axes.labelsize': FONT_SIZE,
        'axes.titlesize': FONT_SIZE,
        'xtick.labelsize': FONT_SIZE - 1.0,
        'ytick.labelsize': FONT_SIZE - 1.0,
        'legend.fontsize': FONT_SIZE - 1.2,
        'figure.dpi': DPI,
        'savefig.dpi': DPI,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.06,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.8,
        'axes.edgecolor': '#222222',
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'axes.grid': False,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
    })


def add_panel_label(ax, label, x=-0.10, y=1.06):
    """Place crisp bold panel tag (A, B, C, D) outside data area."""
    ax.text(
        x, y, label, transform=ax.transAxes,
        fontsize=FONT_SIZE + 2.0, fontweight='bold',
        va='bottom', ha='left', color='#111111'
    )


def add_subtle_ygrid(ax):
    """Add faint horizontal guideline only when aiding value lookup."""
    ax.yaxis.grid(True, linestyle='-', linewidth=0.45, color='#EAEAEA', zorder=0)
    ax.set_axisbelow(True)


def save_hrs_fig(fig, name, also_copy_v7=False):
    """Save editable vector PDF and 300 DPI PNG."""
    pdf_path = OUT_DIR / f'{name}.pdf'
    png_path = OUT_DIR / f'{name}.png'
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', pad_inches=0.06)
    fig.savefig(png_path, format='png', dpi=DPI, bbox_inches='tight', pad_inches=0.06)
    if also_copy_v7:
        v7_dir = Path('reports/figures_v7')
        v7_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(v7_dir / f'{name}.pdf', format='pdf', bbox_inches='tight', pad_inches=0.06)
        fig.savefig(v7_dir / f'{name}.png', format='png', dpi=DPI, bbox_inches='tight', pad_inches=0.06)
    plt.close(fig)
    print(f'Saved: {pdf_path} & {png_path}')


def highpass_center_ecg(sig, fs=200.0, cutoff=0.5):
    """Remove DC baseline wander only for clean visual display on calibrated 10 mm/mV paper."""
    nyq = 0.5 * fs
    b, a = butter(2, cutoff / nyq, btype='high')
    centered = filtfilt(b, a, np.asarray(sig, dtype=float) - np.median(sig))
    return centered


def draw_clinical_ecg_strip(ax, sig_mv, fs=200.0, t_start=0.0, label='', y_lim=(-1.4, 1.6),
                            show_cal_pulse=True, highlight_span=None, highlight_label=None):
    """Render an authentic ECG strip on calibrated 25 mm/s, 10 mm/mV millimeter grid.

    Uses exact ecg-plot (v0.3.0) clinical paper palette:
    - Minor grid: 0.04 s (1 mm) x 0.1 mV (1 mm)
    - Major grid: 0.20 s (5 mm) x 0.5 mV (5 mm)
    - Standard 1.0 mV x 200 ms calibration pulse on left edge
    """
    n = len(sig_mv)
    dur = n / fs
    t = t_start + np.arange(n) / fs

    # Background and calibrated clinical grid
    ax.set_facecolor('white')
    x_min = t_start - (0.45 if show_cal_pulse else 0.0)
    x_max = t_start + dur
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_lim[0], y_lim[1])

    # Physical 25 mm/s and 10 mm/mV aspect ratio: 1 sec = 25 mm, 1 mV = 10 mm => 0.4 s per 1 mV
    ax.set_aspect(0.40)

    # Minor and major millimeter gridlines matching ecg_plot.Style
    ax.xaxis.set_major_locator(MultipleLocator(0.20))
    ax.xaxis.set_minor_locator(MultipleLocator(0.04))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    ax.grid(which='minor', color=ECG_STYLE.minor_color, linewidth=0.35, zorder=0)
    ax.grid(which='major', color=ECG_STYLE.major_color, linewidth=0.65, zorder=1)

    if highlight_span is not None:
        ax.axvspan(highlight_span[0], highlight_span[1], color='#FDEBD0', alpha=0.55, zorder=1.5)
        if highlight_label:
            ax.text(
                0.5 * (highlight_span[0] + highlight_span[1]), y_lim[1] - 0.14,
                highlight_label, fontsize=FONT_SIZE - 1.5, fontweight='bold',
                color=PALETTE['vermilion'], ha='center', va='top', zorder=4
            )

    if show_cal_pulse:
        cal_x = [t_start - 0.40, t_start - 0.30, t_start - 0.30, t_start - 0.10, t_start - 0.10, t_start - 0.02]
        cal_y = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
        ax.plot(cal_x, cal_y, color=ECG_STYLE.line_color, linewidth=1.0, zorder=3)

    ax.plot(t, sig_mv, color=ECG_STYLE.line_color, linewidth=0.9, zorder=3)

    if label:
        ax.text(
            0.015, 0.90, label, transform=ax.transAxes,
            fontsize=FONT_SIZE - 1.0, fontweight='bold', color='#111111',
            va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.82),
            zorder=5
        )

    ax.text(
        0.99, 0.06, '25 mm/s, 10 mm/mV', transform=ax.transAxes,
        fontsize=FONT_SIZE - 2.2, color='#555555', ha='right', va='bottom',
        bbox=dict(boxstyle='square,pad=0.12', facecolor='white', edgecolor='none', alpha=0.75),
        zorder=5
    )

    for sp in ['top', 'right', 'left']:
        ax.spines[sp].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks(np.arange(np.ceil(t_start), np.floor(t_start + dur) + 0.1, 1.0))
    ax.tick_params(axis='x', length=2.5, labelsize=FONT_SIZE - 1.2, colors='#333333')
