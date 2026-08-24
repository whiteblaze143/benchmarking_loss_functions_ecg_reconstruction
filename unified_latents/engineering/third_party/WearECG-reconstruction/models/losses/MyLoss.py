from torch import nn
import torch
from scipy.signal import find_peaks
import torch.nn.functional as F


def focal_mse_loss(y_true, y_pred, gamma=3.0, reduction="none"):
    threshold = torch.mean(y_true, dim=1, keepdim=True)
    diff = torch.abs(y_true - y_pred)

    focal_weight = torch.where(
        diff > threshold,
        1 + gamma * (diff - threshold),  # 线性增大 离得越远权重越大
        torch.ones_like(diff),
    )

    weighted_mse = focal_weight * diff**2

    if reduction == "none":
        return weighted_mse
    elif reduction == "mean":
        return torch.mean(weighted_mse)
    elif reduction == "sum":
        return torch.sum(weighted_mse)
    else:
        raise ValueError("Reduction must be 'none', 'mean', or 'sum'")


def rpeak_amplitude_loss(
    pred_wave: torch.Tensor,
    target_wave: torch.Tensor,
    height: float = 2.0,
    penalty_scale: float = 10.0,
    reduction: str = "none",
) -> torch.Tensor:
    """
    基于真实波形的 R 峰位置, 对多通道 ECG 幅值进行惩罚的示例。
    输入形状: (batch_size, time, channel).

    参数:
    -------
    pred_wave : [B, T, C]  预测波形
    target_wave : [B, T, C]  真实波形
    height : float
        find_peaks 的 height 阈值, 用于从 target_wave[i, :, c] 里检测 R 峰
    penalty_scale : float
        对 R 峰处幅值误差的放大系数, 值越大则对 R 峰的幅值差异惩罚越重

    返回:
    -------
    torch.Tensor
        一个标量损失值 (不可对 R 峰位置反传, 但可与其他损失相加).
    """

    B, T, C = pred_wave.shape
    device = pred_wave.device
    loss_per_sample = []

    # 在 no_grad 块内执行 find_peaks 以避免 R 峰位置带来梯度开销
    with torch.no_grad():
        # 遍历批次内每个样本
        for i in range(B):
            # 收集该样本在所有通道上的 R 峰损失
            channel_losses = []

            # 逐通道处理
            for c in range(C):
                # 提取预测波形 & 真实波形的 numpy 数组
                pred_signal = pred_wave[i, :, c].detach().cpu().numpy()
                tgt_signal = target_wave[i, :, c].detach().cpu().numpy()

                # 在真实信号该通道中检测 R 峰位置
                tgt_peaks, _ = find_peaks(tgt_signal, height=height)

                if len(tgt_peaks) > 0:
                    # 取 R 峰处的幅值
                    pred_r_values = pred_signal[tgt_peaks]
                    tgt_r_values = tgt_signal[tgt_peaks]

                    # 简单使用 MSE
                    amplitude_diff = (pred_r_values - tgt_r_values) ** 2
                    dist_loss = amplitude_diff.mean() * penalty_scale
                else:
                    # 若该通道未检测到 R 峰, 简单给个常数惩罚
                    dist_loss = penalty_scale * 1.0

                channel_losses.append(dist_loss)
                sample_loss = sum(channel_losses) / len(channel_losses)

            loss_per_sample.append(sample_loss)

        final_loss_val = sum(loss_per_sample) / len(loss_per_sample)

    return torch.tensor(final_loss_val, requires_grad=True).to(device)


def r_aware_mse_loss(
    pred_wave: torch.Tensor,
    target_wave: torch.Tensor,
    height: float = 1.0,
    penalty_scale: float = 5.0,
    alpha: float = 1.0,
    reduction: str = "none",
) -> torch.Tensor:
    mse_criterion = nn.MSELoss(reduction=reduction)
    global_mse = mse_criterion(pred_wave, target_wave)

    rpeak_loss = rpeak_amplitude_loss(
        pred_wave, target_wave, height=height, penalty_scale=penalty_scale
    )

    total_loss = global_mse + alpha * rpeak_loss
    return total_loss


def smooth_mal_mse_loss(
    ecg_rec,
    ecg_ref,
    window_size = 10,
    reduction="mean",
):
    abs_diff = torch.abs(ecg_ref - ecg_rec)
    abs_diff = abs_diff.permute(0, 2, 1)
    smoothed_abs_diff = F.avg_pool1d(
        abs_diff,
        kernel_size=window_size,
        stride=window_size // 2,
        padding=0,
    )
    smoothed_abs_diff = smoothed_abs_diff.permute(0, 2, 1)
    smooth_mal_loss =  torch.mean(torch.max(smoothed_abs_diff, dim=1)[0])
    # mal_loss = torch.max(torch.abs(ecg_ref - ecg_rec), dim=1)[0]
    mse_loss = F.mse_loss(ecg_rec, ecg_ref, reduction=reduction)
    return smooth_mal_loss + mse_loss


def mal_mse_loss(
    ecg_rec,
    ecg_ref,
    reduction="mean",
):
    mal_loss = torch.mean(torch.max(torch.abs(ecg_ref - ecg_rec), dim=1)[0])
    mse_loss = F.mse_loss(ecg_rec, ecg_ref, reduction=reduction)
    return mal_loss + mse_loss


def DynamicSparsityLoss(binary_output,min_sparsity=0.1, max_sparsity=0.9):
    target_sparsity = (
        torch.rand(1).item() * (max_sparsity - min_sparsity)
        + min_sparsity
    )
    target_sparsity = torch.tensor(target_sparsity, device=binary_output.device)

    current_sparsity = torch.mean(binary_output)
    sparsity_loss = torch.abs(current_sparsity - target_sparsity)
    return sparsity_loss

