## Necessary Packages
import os
import scipy.stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import torch
import torch.nn.functional as F
from torcheval.metrics.image.fid import FrechetInceptionDistance
from scipy.linalg import sqrtm
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from wfdb import processing
import ot
import random
import warnings

def display_scores(results):
    mean = np.mean(results)
    sigma = scipy.stats.sem(results)
    sigma = sigma * scipy.stats.t.ppf((1 + 0.95) / 2.0, 5 - 1)
    #  sigma = 1.96*(np.std(results)/np.sqrt(len(results)))
    print("Final Score: ", f"{mean} \xb1 {sigma}")


def train_test_divide(data_x, data_x_hat, data_t, data_t_hat, train_rate=0.8):
    """Divide train and test data for both original and synthetic data.

    Args:
      - data_x: original data
      - data_x_hat: generated data
      - data_t: original time
      - data_t_hat: generated time
      - train_rate: ratio of training data from the original data
    """
    # Divide train/test index (original data)
    no = len(data_x)
    idx = np.random.permutation(no)
    train_idx = idx[: int(no * train_rate)]
    test_idx = idx[int(no * train_rate) :]

    train_x = [data_x[i] for i in train_idx]
    test_x = [data_x[i] for i in test_idx]
    train_t = [data_t[i] for i in train_idx]
    test_t = [data_t[i] for i in test_idx]

    # Divide train/test index (synthetic data)
    no = len(data_x_hat)
    idx = np.random.permutation(no)
    train_idx = idx[: int(no * train_rate)]
    test_idx = idx[int(no * train_rate) :]

    train_x_hat = [data_x_hat[i] for i in train_idx]
    test_x_hat = [data_x_hat[i] for i in test_idx]
    train_t_hat = [data_t_hat[i] for i in train_idx]
    test_t_hat = [data_t_hat[i] for i in test_idx]

    return (
        train_x,
        train_x_hat,
        test_x,
        test_x_hat,
        train_t,
        train_t_hat,
        test_t,
        test_t_hat,
    )


def extract_time(data):
    """Returns Maximum sequence length and each sequence length.

    Args:
      - data: original data

    Returns:
      - time: extracted time information
      - max_seq_len: maximum sequence length
    """
    time = list()
    max_seq_len = 0
    for i in range(len(data)):
        max_seq_len = max(max_seq_len, len(data[i][:, 0]))
        time.append(len(data[i][:, 0]))

    return time, max_seq_len


def visualization(ori_data, generated_data, analysis, compare=1000, save_dir=None):
    """Using PCA or tSNE for generated and original data visualization.

    Args:
      - ori_data: original data
      - generated_data: generated synthetic data
      - analysis: tsne or pca or kernel
    """
    # Analysis sample size (for faster computation)
    anal_sample_no = min([compare, ori_data.shape[0]])
    idx = np.random.permutation(ori_data.shape[0])[:anal_sample_no]

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # Data preprocessing
    # ori_data = np.asarray(ori_data)
    # generated_data = np.asarray(generated_data)

    ori_data = ori_data[idx]
    generated_data = generated_data[idx]

    no, seq_len, dim = ori_data.shape

    for i in range(anal_sample_no):
        if i == 0:
            prep_data = np.reshape(np.mean(ori_data[0, :, :], 1), [1, seq_len])
            prep_data_hat = np.reshape(
                np.mean(generated_data[0, :, :], 1), [1, seq_len]
            )
        else:
            prep_data = np.concatenate(
                (prep_data, np.reshape(np.mean(ori_data[i, :, :], 1), [1, seq_len]))
            )
            prep_data_hat = np.concatenate(
                (
                    prep_data_hat,
                    np.reshape(np.mean(generated_data[i, :, :], 1), [1, seq_len]),
                )
            )

    # Visualization parameter
    colors = ["red" for i in range(anal_sample_no)] + [
        "blue" for i in range(anal_sample_no)
    ]

    if analysis == "pca":
        # PCA Analysis
        pca = PCA(n_components=2)
        pca.fit(prep_data)
        pca_results = pca.transform(prep_data)
        pca_hat_results = pca.transform(prep_data_hat)

        # Plotting
        f, axes = plt.subplots(1, 3, figsize=(18, 6))  # 创建1行3列的子图

        # 原始数据分布
        axes[0].scatter(
            pca_results[:, 0],
            pca_results[:, 1],
            c=colors[:anal_sample_no],
            alpha=0.6,
            label="Original",
        )
        axes[0].set_title("Original Data")
        axes[0].set_xlabel("x-pca")
        axes[0].set_ylabel("y-pca")
        axes[0].legend()

        # 生成数据分布
        axes[1].scatter(
            pca_hat_results[:, 0],
            pca_hat_results[:, 1],
            c=colors[anal_sample_no:],
            alpha=0.6,
            label="Synthetic",
        )
        axes[1].set_title("Synthetic Data")
        axes[1].set_xlabel("x-pca")
        axes[1].set_ylabel("y-pca")
        axes[1].legend()

        # 叠加分布
        axes[2].scatter(
            pca_results[:, 0],
            pca_results[:, 1],
            c=colors[:anal_sample_no],
            alpha=0.2,
            label="Original",
        )
        axes[2].scatter(
            pca_hat_results[:, 0],
            pca_hat_results[:, 1],
            c=colors[anal_sample_no:],
            alpha=0.2,
            label="Synthetic",
        )
        axes[2].set_title("Overlay")
        axes[2].set_xlabel("x-pca")
        axes[2].set_ylabel("y-pca")
        axes[2].legend()

        # 保存或展示
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, "pca_comparison.png"), dpi=400)
        else:
            plt.show()

    elif analysis == "tsne":

        # Do t-SNE Analysis together
        prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)

        # TSNE analysis
        tsne = TSNE(n_components=2, verbose=1, perplexity=20, n_iter=2000, init="pca")
        tsne_results = tsne.fit_transform(prep_data_final)

        # Plotting
        f, axes = plt.subplots(1, 3, figsize=(18, 6))  # 创建1行3列的子图

        # 原始数据分布
        axes[0].scatter(
            tsne_results[:anal_sample_no, 0],
            tsne_results[:anal_sample_no, 1],
            c=colors[:anal_sample_no],
            alpha=0.6,
            label="Original",
        )
        axes[0].set_title("Original Data")
        axes[0].set_xlabel("x-tsne")
        axes[0].set_ylabel("y-tsne")
        axes[0].legend()

        # 生成数据分布
        axes[1].scatter(
            tsne_results[anal_sample_no:, 0],
            tsne_results[anal_sample_no:, 1],
            c=colors[anal_sample_no:],
            alpha=0.6,
            label="Synthetic",
        )
        axes[1].set_title("Synthetic Data")
        axes[1].set_xlabel("x-tsne")
        axes[1].set_ylabel("y-tsne")
        axes[1].legend()

        # 叠加分布
        axes[2].scatter(
            tsne_results[:anal_sample_no, 0],
            tsne_results[:anal_sample_no, 1],
            c=colors[:anal_sample_no],
            alpha=0.2,
            label="Original",
        )
        axes[2].scatter(
            tsne_results[anal_sample_no:, 0],
            tsne_results[anal_sample_no:, 1],
            c=colors[anal_sample_no:],
            alpha=0.2,
            label="Synthetic",
        )
        axes[2].set_title("Overlay")
        axes[2].set_xlabel("x-tsne")
        axes[2].set_ylabel("y-tsne")
        axes[2].legend()

        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, "tsne_comparison.png"), dpi=400)
        else:
            plt.show()

    elif analysis == "kernel":

        # Visualization parameter
        # colors = ["red" for i in range(anal_sample_no)] + ["blue" for i in range(anal_sample_no)]

        f, ax = plt.subplots(1)
        # 绘制原始数据的 KDE 曲线
        sns.kdeplot(prep_data, linewidth=5, label="Original", color="red")

        # 绘制合成数据的 KDE 曲线
        sns.kdeplot(
            prep_data_hat, linewidth=5, linestyle="--", label="Synthetic", color="blue"
        )
        # Plot formatting

        # plt.legend(prop={'size': 22})
        plt.legend()
        plt.xlabel("Data Value")
        plt.ylabel("Data Density Estimate")
        # plt.rcParams['pdf.fonttype'] = 42

        # plt.savefig(str(args.save_dir)+"/"+args.model1+"_histo.png", dpi=100,bbox_inches='tight')
        # plt.ylim((0, 12))

        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, "kernel.png"), dpi=400)
        else:
            plt.show()


def calculate_mse(gt_data, generated_data, decimal_places=5):
    gt_data, generated_data = torch.from_numpy(gt_data), torch.from_numpy(
        generated_data
    )
    mse_per_channel = F.mse_loss(generated_data, gt_data, reduction="none").mean(
        dim=(0, 1)
    )
    overall_mse = mse_per_channel.mean()

    # Format values to specified decimal places
    mse_per_channel_formatted = [f"{val:.{decimal_places}f}" for val in mse_per_channel]
    overall_mse_formatted = f"{overall_mse:.{decimal_places}f}"

    return mse_per_channel_formatted, overall_mse_formatted


def calculate_mae(gt_data, generated_data, decimal_places=5):
    gt_data, generated_data = torch.from_numpy(gt_data), torch.from_numpy(
        generated_data
    )

    mae_per_channel = F.l1_loss(generated_data, gt_data, reduction="none").mean(
        dim=(0, 1)
    )

    overall_mae = mae_per_channel.mean()

    mae_per_channel_formatted = [f"{val:.{decimal_places}f}" for val in mae_per_channel]
    overall_mae_formatted = f"{overall_mae:.{decimal_places}f}"

    return mae_per_channel_formatted, overall_mae_formatted


def calculate_statistics(data):
    """
    Calculate the mean and covariance of the input data.

    Args:
        data (numpy.ndarray): Input data of shape (num_samples, num_features).

    Returns:
        Tuple[Tensor, Tensor]: Mean vector and covariance matrix as PyTorch tensors.
    """
    data = torch.from_numpy(data)
    num_data = data.shape[0]
    data_sum = torch.sum(data, dim=0)
    data_cov_sum = torch.matmul(data.T, data)

    # Compute the mean activations for each distribution
    mean = (data_sum / num_data).unsqueeze(0)

    # Compute the covariance matrices for each distribution
    cov_num = data_cov_sum - num_data * torch.matmul(mean.T, mean)
    cov = cov_num / (num_data - 1)

    return mean, cov


def calculate_frechet_distance(
    mu1,
    sigma1,
    mu2,
    sigma2,
):
    """
    Calculate the Frechet Distance between two multivariate Gaussian distributions.

    Args:
        mu1 (Tensor): The mean of the first distribution.
        sigma1 (Tensor): The covariance matrix of the first distribution.
        mu2 (Tensor): The mean of the second distribution.
        sigma2 (Tensor): The covariance matrix of the second distribution.

    Returns:
        tensor: The Frechet Distance between the two distributions.
    """

    # Compute the squared distance between the means
    mean_diff = mu1 - mu2
    mean_diff_squared = mean_diff.square().sum(dim=-1)

    # Calculate the sum of the traces of both covariance matrices
    trace_sum = sigma1.trace() + sigma2.trace()

    # Compute the eigenvalues of the matrix product of the real and fake covariance matrices
    sigma_mm = torch.matmul(sigma1, sigma2)
    eigenvals = torch.linalg.eigvals(sigma_mm)

    # Take the square root of each eigenvalue and take its sum
    sqrt_eigenvals_sum = eigenvals.sqrt().real.sum(dim=-1)

    # Calculate the FID using the squared distance between the means,
    # the sum of the traces of the covariance matrices, and the sum of the square roots of the eigenvalues
    fid = mean_diff_squared + trace_sum - 2 * sqrt_eigenvals_sum

    return fid


def calculate_FID_score(M1: torch.Tensor, M2: torch.Tensor):
    M1, M2 = M1.numpy(), M2.numpy()
    mu1, sigma1 = M1.mean(axis=0), np.cov(M1, rowvar=False)
    mu2, sigma2 = M2.mean(axis=0), np.cov(M2, rowvar=False)

    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def compute_representations_in_batches(model, data, batch_size=128, device="cuda"):
    dataset = TensorDataset(torch.from_numpy(data).float().transpose(1, 2))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    representations = []
    with torch.no_grad():
        for batch in dataloader:
            batch_data = batch[0].to(device)
            _, batch_rep = model(batch_data)
            representations.append(batch_rep.cpu())

    return torch.cat(representations, dim=0)


def detect_hr_average(ecg):
    fs = 100
    heart_rates = []

    for lead in range(11):
        xqrs = processing.XQRS(sig=ecg[:, lead], fs=fs)
        xqrs.detect(verbose=False)
        qrs_inds = xqrs.qrs_inds

        if len(qrs_inds) > 1:
            rr_intervals = np.diff(qrs_inds) / fs
            heart_rate = 60 / np.mean(rr_intervals)
            heart_rates.append(heart_rate)

    valid_hr = [hr for hr in heart_rates if hr is not None]
    average_hr = np.mean(valid_hr) if valid_hr else None

    return average_hr


def detect_hr(ecg):
    fs = 100
    for lead in range(11):
        xqrs = processing.XQRS(sig=ecg[:, lead], fs=fs)
        xqrs.detect(verbose=False)
        qrs_inds = xqrs.qrs_inds
        if len(qrs_inds) > 1:
            rr_intervals = np.diff(qrs_inds) / fs
            heart_rate = 60 / np.mean(rr_intervals)
            break

    return heart_rate


def calculate_hr_score(fake_ecgs, real_ecgs, metric="MAE"):
    real_hr_list = []
    fake_hr_list = []

    print("Processing real ECGs...")
    for ecg in tqdm(real_ecgs, desc="Real ECGs", unit="sample"):    
        hr = detect_hr_average(ecg)
        real_hr_list.append(hr)

    print("Processing fake ECGs...")
    for ecg in tqdm(fake_ecgs, desc="Fake ECGs", unit="sample"):
        hr = detect_hr_average(ecg)
        fake_hr_list.append(hr)

    real_hr_list = np.array(real_hr_list)
    fake_hr_list = np.array(fake_hr_list)

    if metric == "MSE":
        hr_score = np.mean((real_hr_list - fake_hr_list) ** 2)
    elif metric == "MAE":
        hr_score = np.mean(np.abs(real_hr_list - fake_hr_list))
    else:
        raise ValueError("Metric must be 'MSE' or 'MAE'")

    return hr_score


def calculate_EMD_L2(target_ecg, generated_ecg):
    N = generated_ecg.shape[0]
    source = generated_ecg.reshape((N, -1))
    target = target_ecg.reshape((N, -1))
    M = ot.dist(source, target)
    return ot.emd2(a=[], b=[], M=M)


def calculate_CosDist(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # x_norm, y_norm = np.linalg.norm(x, axis=(1, 2)), np.linalg.norm(y, axis=(1, 2))
    # return (x * y).sum(axis=(1, 2)) / x_norm / y_norm
    x_norm, y_norm = np.linalg.norm(x, axis=1), np.linalg.norm(y, axis=1)
    return ((x * y).sum(axis=1) / x_norm / y_norm).mean(axis=1)


def seed_everything(seed, cudnn_deterministic=True):
    """
    Function that sets seed for pseudo-random number generators in:
    pytorch, numpy, python.random
    
    Args:
        seed: the integer value seed for global random state
    """
    if seed is not None:
        print(f"Global seed set to {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False

    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

if __name__ == "__main__":
    pass
