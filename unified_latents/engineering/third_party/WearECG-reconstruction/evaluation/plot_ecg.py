import os
from matplotlib import pyplot as plt
import numpy as np


def plot_channel_distributions(data, bins, fig_save_dir, range_min=-200, range_max=200):
    """
    绘制所有通道的数据分布，每个通道一个子图，并标注最小值、最大值和四分位数所在区间。

    参数:
    - data: np.ndarray，形状为 (nums, lens, channels) 的数据集。
    - bins: int，分组的数量（步长为 range / bins）。
    - fig_save_dir: str，保存图片的目录。
    - range_min: float，统计区间的最小值。
    - range_max: float，统计区间的最大值。
    """
    num_channels = data.shape[-1]  # 获取通道数
    bin_edges = np.linspace(range_min, range_max, bins + 1)  # 定义区间边界

    # 创建画布，按通道数动态调整子图布局
    fig, axes = plt.subplots(
        num_channels, 1, figsize=(12, 4 * num_channels), constrained_layout=True
    )

    # 如果只有一个通道，确保 axes 是列表
    if num_channels == 1:
        axes = [axes]

    for i in range(num_channels):
        channel_data = data[:, :, i].flatten()  # 展平该通道数据
        histogram, _ = np.histogram(channel_data, bins=bin_edges)  # 统计分布

        # 找到最小值、最大值和四分位数
        min_value = channel_data.min()
        max_value = channel_data.max()
        q1 = np.percentile(channel_data, 25)  # 第 1 四分位数
        median = np.percentile(channel_data, 50)  # 中位数
        q3 = np.percentile(channel_data, 75)  # 第 3 四分位数

        # 找到这些值对应的区间索引
        min_index = np.digitize(min_value, bin_edges) - 1
        max_index = np.digitize(max_value, bin_edges) - 1
        q1_index = np.digitize(q1, bin_edges) - 1
        median_index = np.digitize(median, bin_edges) - 1
        q3_index = np.digitize(q3, bin_edges) - 1

        # 绘制直方图到对应子图
        axes[i].bar(
            bin_edges[:-1],
            histogram,
            width=(range_max - range_min) / bins,
            edgecolor="black",
            align="edge",
        )
        axes[i].set_title(f"Channel {i + 1} Value Distribution")
        axes[i].set_xlabel("Value Range")
        axes[i].set_ylabel("Frequency")
        axes[i].grid(True)

        # 标注最小值、最大值和四分位数区间
        axes[i].axvline(
            x=bin_edges[min_index],
            color="red",
            linestyle="--",
            label=f"Min: {min_value:.2f}",
        )
        axes[i].axvline(
            x=bin_edges[max_index],
            color="blue",
            linestyle="--",
            label=f"Max: {max_value:.2f}",
        )
        axes[i].axvline(
            x=bin_edges[q1_index], color="orange", linestyle="--", label=f"Q1: {q1:.2f}"
        )
        axes[i].axvline(
            x=bin_edges[median_index],
            color="green",
            linestyle="--",
            label=f"Median: {median:.2f}",
        )
        axes[i].axvline(
            x=bin_edges[q3_index], color="purple", linestyle="--", label=f"Q3: {q3:.2f}"
        )
        axes[i].legend()

    plt.suptitle("Value Distributions for All Channels", fontsize=16)
    plt.savefig(
        os.path.join(fig_save_dir, f"channel_distributions.png"), dpi=400
    )  # 设置 DPI


def plot_ecg_comparison(ori_data, samples, index, save_dir, alpha=0.7):
    # 标准导联名称顺序
    standard_leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    
    # 创建画板
    fig, axes = plt.subplots(
        nrows=12, ncols=1, figsize=(10, 15), constrained_layout=True
    )
    fig.suptitle(f"ECG Comparison for Sample Index {index}", fontsize=16)

    # 遍历每个导联进行绘制
    for i in range(ori_data.shape[2]):  # 遍历 channel
        ax = axes[i]
        # ax.plot(
        #     ori_data[index, :, i],
        #     color="blue",
        #     label="Unknown Data",
        #     alpha=0.5,  # 设置透明度
        # )
        ax.plot(
            samples[index, :, i],
            color="blue",
            label="Unknown Data",
            alpha=0.5,  # 设置透明度
        )
        ax.set_title(f"Lead {standard_leads[i]}")  # 使用标准导联名称
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
        # ax.set_ylim(0, 1)  # 设置纵轴范围为 [-1, 1]
        ax.legend(loc="upper right")
        ax.grid(True)

    # 保存图形
    fig_save_dir = os.path.join(save_dir, "generated_figs")
    os.makedirs(fig_save_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_save_dir, f"{index}.png"), dpi=250)  # 设置 DPI

from ecg_plot_utils import plot  # 假设你把上面的画图代码保存到了 ecg_plot_utils.py 文件中
import os
import matplotlib.pyplot as plt

def plot_ecg_comparison_redgrid(data, index, save_dir, mode="fake", sample_rate=500):
    """
    用红色网格专业风格绘制单个样本 ECG 图。
    - data: ECG ndarray，shape = (num, length, 12)
    - index: 第几个样本
    - save_dir: 根目录，会自动分到 generated_figs/{mode}/ 目录
    - mode: 'real' or 'fake'
    """
    # 转换为 shape = (12, length)
    ecg = data[index].T

    # 设置保存目录
    fig_save_dir = os.path.join(save_dir, "generated_figs", mode)
    os.makedirs(fig_save_dir, exist_ok=True)

    # 开始绘图
    plt.figure()
    plot(
        ecg=ecg,
        sample_rate=sample_rate,
        title=f"{mode.upper()} ECG Sample {index}",
        style=None,               # 默认就是红色
        columns=2,                # 分两列排列
        row_height=6,             # 每个导联高度
        show_lead_name=True,
        show_grid=True,
        show_separate_line=True,
    )
    
    # 保存图像
    plt.savefig(os.path.join(fig_save_dir, f"{index}.png"), dpi=300)
    plt.close()

if __name__ == "__main__":

    data_folder = "results/vae_100/MIMIC"
    ori_data = np.load(
        os.path.join(data_folder, "samples", "ground_truth_5000_test.npy")
    )

    # plot_channel_distributions(
    #     ori_data, bins=400, fig_save_dir=data_folder, range_min=-200, range_max=200
    # )

    # samples = np.load(
    #     os.path.join(data_folder, "samples", "gound_truth_1000_train.npy")
    # )
    samples = np.load(os.path.join(data_folder, "samples", "overall_fake_data.npy"))

    print(f"original data scale: {np.min(ori_data)} {np.max(ori_data)}")
    print(f"samples data scale: {np.min(samples)} {np.max(samples)}")

    # for index in range(32):
    #     plot_ecg_comparison(
    #         ori_data,
    #         samples,
    #         index=index,
    #         save_dir=data_folder,
    #     )
    #     print(f"samples {index} plot finished")
    for index in range(32):
        plot_ecg_comparison_redgrid(ori_data, index=index, save_dir=data_folder, mode="real")
        plot_ecg_comparison_redgrid(samples, index=index, save_dir=data_folder, mode="fake")
        print(f"Sample {index} plots saved.")
