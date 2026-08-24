import os
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator
from math import ceil

def _ax_plot(ax, x, y, secs=10, lwidth=0.5, amplitude_ecg=1.8, time_ticks=0.2):
    ax.set_xticks(np.arange(0, secs+0.01, time_ticks))
    ax.set_yticks(np.arange(-ceil(amplitude_ecg), ceil(amplitude_ecg)+0.01, 1.0))
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.set_ylim(-amplitude_ecg, amplitude_ecg)
    ax.set_xlim(0, secs)
    ax.grid(which='major', linestyle='-', linewidth=0.5, color='red')
    ax.grid(which='minor', linestyle='-', linewidth=0.5, color=(1, 0.7, 0.7))
    ax.plot(x, y, linewidth=lwidth, color='black')

def plot_ecg_comparison(
    ecg_data,
    index,
    save_dir,
    sample_rate=500,
    amplitude_range=1.8,
    line_width=0.6,
    alpha=0.7,
    time_tick=0.2,
    mode="fake",  # 或者 "real"
):
    assert mode in ["real", "fake"], "mode must be 'real' or 'fake'"

    standard_leads = ["I", "II", "III", "aVR", "aVL", "aVF",
                      "V1", "V2", "V3", "V4", "V5", "V6"]

    # 取出当前样本的12导联数据，shape: (N, 12) → (12, N)
    ecg = ecg_data[index].T  # shape: (12, N)
    ecg_length = ecg.shape[1]
    secs = ecg_length / sample_rate
    leads = 12
    row_height = 3  # 每个导联的垂直间隔，从6减小到3

    plt.rcParams.update({'font.size': 8})
    fig, ax = plt.subplots(figsize=(secs * 1.2, leads * row_height / 5))
    fig.subplots_adjust(
        hspace=0,
        wspace=0,
        left=0.04,
        right=0.98,
        bottom=0.06,
        top=0.95
    )
    title_type = "FAKE" if mode == "fake" else "REAL"
    fig.suptitle(f"ECG {title_type} - Sample {index}", fontsize=16)

    x_min = 0
    x_max = secs
    y_min = row_height/4 - (leads/2)*row_height
    y_max = row_height/4

    # 画红色网格
    ax.set_xticks(np.arange(x_min, x_max, 0.2))
    ax.set_yticks(np.arange(y_min, y_max, 0.5))
    x_ticks = ax.get_xticks()
    if x_ticks[-1] < int(x_ticks[-1]) + 1:
        x_ticks = np.append(x_ticks, int(x_ticks[-1]) + 1)
    ax.set_xticks(x_ticks)
    x_labels = [f'{int(x)}' if x.is_integer() else '' for x in x_ticks]
    ax.set_xticklabels(x_labels)
    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.grid(which='major', linestyle='-', linewidth=0.5, color='red')
    ax.grid(which='minor', linestyle='-', linewidth=0.5, color=(1, 0.7, 0.7))
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min, x_max)

    # 画12导联，每个导联上下错开
    step = 1.0 / sample_rate
    for i in range(leads):
        y_offset = -(row_height/2) * ceil(i % leads)
        if i < len(standard_leads):
            ax.text(x_min + 0.07, y_offset - 0.5, standard_leads[i], fontsize=9)
        ax.plot(
            np.arange(0, ecg.shape[1]*step, step),
            ecg[i] + y_offset,
            linewidth=line_width,
            color='black'
        )

    subfolder = "fake" if mode == "fake" else "real"
    fig_save_dir = os.path.join(save_dir, "generated_figs_ecgstyle2", subfolder)
    os.makedirs(fig_save_dir, exist_ok=True)
    save_path = os.path.join(fig_save_dir, f"{index}_{mode}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    data_folder = "results/vae_100/MIMIC"
    
    # 加载真实和生成数据
    ori_data = np.load(os.path.join(data_folder, "samples", "ground_truth_5000_test.npy"))
    samples  = np.load(os.path.join(data_folder, "samples", "overall_fake_data.npy"))

    print(f"original data scale: {np.min(ori_data)} {np.max(ori_data)}")
    print(f"samples data scale: {np.min(samples)} {np.max(samples)}")

    for index in range(8):
        plot_ecg_comparison(
            ecg_data=ori_data,
            index=index,
            save_dir=data_folder,
            mode="real"
        )
        plot_ecg_comparison(
            ecg_data=samples,
            index=index,
            save_dir=data_folder,
            mode="fake"
        )
        print(f"Sample {index} plots saved.")
