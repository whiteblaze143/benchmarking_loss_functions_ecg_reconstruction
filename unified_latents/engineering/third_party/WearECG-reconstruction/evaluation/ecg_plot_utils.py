# ecg_plot_utils.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from math import ceil


lead_index = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

def _ax_plot(ax, x, y, secs=10, lwidth=0.5, amplitude_ecg=1.8, time_ticks=0.2):
    ax.set_xticks(np.arange(0, secs + 1, time_ticks))
    ax.set_yticks(np.arange(-ceil(amplitude_ecg), ceil(amplitude_ecg) + 1, 1.0))
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.set_ylim(-amplitude_ecg, amplitude_ecg)
    ax.set_xlim(0, secs)
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='red')
    ax.grid(which='minor', linestyle='-', linewidth='0.5', color=(1, 0.7, 0.7))
    ax.plot(x, y, linewidth=lwidth)


def plot(
    ecg,
    sample_rate=500,
    title='ECG 12-lead',
    lead_index=lead_index,
    lead_order=None,
    style=None,
    columns=2,
    row_height=6,
    show_lead_name=True,
    show_grid=True,
    show_separate_line=True,
):
    if not lead_order:
        lead_order = list(range(0, len(ecg)))

    secs = len(ecg[0]) / sample_rate
    leads = len(lead_order)
    rows = int(ceil(leads / columns))
    fig, ax = plt.subplots(figsize=(secs * columns, rows * row_height / 5))
    fig.subplots_adjust(hspace=0, wspace=0, left=0, right=1, bottom=0, top=0.95)
    fig.suptitle(title)

    x_min = 0
    x_max = columns * secs
    y_min = row_height / 4 - (rows / 2) * row_height
    y_max = row_height / 4

    color_major = (1, 0, 0)
    color_minor = (1, 0.7, 0.7)
    color_line = (0, 0, 0)

    if show_grid:
        ax.set_xticks(np.arange(x_min, x_max, 0.2))
        ax.set_yticks(np.arange(y_min, y_max, 0.5))
        ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.grid(which='major', linestyle='-', linewidth=0.5, color=color_major)
        ax.grid(which='minor', linestyle='-', linewidth=0.5, color=color_minor)

    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min, x_max)

    for c in range(columns):
        for i in range(rows):
            idx = c * rows + i
            if idx >= leads:
                continue
            t_lead = lead_order[idx]
            y_offset = -(row_height / 2) * i
            x_offset = secs * c

            if show_separate_line:
                ax.plot([x_offset, x_offset], [y_offset - 0.3, y_offset + 0.3],
                        linewidth=0.5, color=color_line)

            if show_lead_name:
                ax.text(x_offset + 0.07, y_offset - 0.5, lead_index[t_lead], fontsize=9)

            step = 1.0 / sample_rate
            ax.plot(
                np.arange(0, len(ecg[t_lead]) * step, step) + x_offset,
                ecg[t_lead] + y_offset,
                linewidth=0.5,
                color=color_line
            )
