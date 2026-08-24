import os
import pickle
import torch
import numpy as np
import pandas as pd

from scipy import io
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from torch.utils.data import Dataset
from models.interpretable_diffusion.model_utils import (
    normalize_to_neg_five_to_five,
    normalize_to_neg_one_to_one,
    unnormalize_to_zero_to_one,
)
from utils.masking_utils import noise_mask


class PTBXLDataset(Dataset):

    def __init__(
        self,
        name,
        data_root,
        window=1000,
        step=1,
        save2npy=True,
        seed=42,
        period="train",
        auto_norm_type="neg_one_to_one",
        condition_type="uncond",
        output_dir="./results/PTBXL/",
        predict_length=None,
        missing_ratio=None,
        synthesis_channels=None,
        style="separate",
        distribution="geometric",
        mean_mask_length=3,
    ):
        super(PTBXLDataset, self).__init__()
        assert period in ["train", "test"], "period must be train or test."
        if period == "train":
            assert ~(predict_length is not None or missing_ratio is not None), ""

        self.seed = seed
        self.save2npy = save2npy
        self.dir = os.path.join(output_dir, "samples")
        os.makedirs(self.dir, exist_ok=True)
        self.save2npy = save2npy
        self.auto_norm_type = auto_norm_type
        self.auto_norm = False if auto_norm_type == "None" else True
        print(f"Normalization Type: {self.auto_norm_type}")

        self.predict_length, self.missing_ratio, self.synthesis_channels = (
            predict_length,
            missing_ratio,
            synthesis_channels,
        )
        self.style, self.distribution, self.mean_mask_length = (
            style,
            distribution,
            mean_mask_length,
        )
        self.window, self.step, self.period, self.condition_type = (
            window,
            step,
            period,
            condition_type,
        )

        self.minmax_scaler_path = os.path.join(self.dir, f"train_minmax_scaler.pkl")

        self.samples = self.read_data(data_root)
        # self.samples = self.samples[:32]  # first 100 sample for fast sampling
        self.num_sample, self.lens, self.channels = self.samples.shape
        self.sample_num_total = max((self.num_sample - self.window) // self.step + 1, 0)

        print(
            f"{self.period.capitalize()} Data Scale Before Norm: [{np.min(self.samples)}, {np.max(self.samples)}]"
        )
        print(
            f"{self.period.capitalize()} Data Scale Before Norm: [{np.min(self.samples)}, {np.max(self.samples)}]"
        )

        if self.auto_norm:
            self.normalize()

        if self.save2npy:
            self.save_data(self.samples, self.period)

        self.masking = self.make_masking(period)

    def make_masking(self, period):
        if period == "test":
            if self.missing_ratio is not None:
                masking = self.__imputation_mask_data__(self.seed)
            elif self.predict_length is not None:
                masks = np.ones(self.samples.shape)
                masks[:, -self.predict_length :, :] = 0
                masking = masks.astype(bool)
            elif self.synthesis_channels is not None:
                masks = np.ones(self.samples.shape)
                masks[:, :, self.synthesis_channels] = 0
                masking = masks.astype(bool)
                if self.save2npy:
                    np.save(os.path.join(self.dir, f"masking_{self.window}.npy"), masks)
            else:
                raise NotImplementedError()

            return masking

    def save_data(self, data, period):
        if self.auto_norm:
            np.save(
                os.path.join(
                    self.dir,
                    f"norm_truth_{self.window}_{period}.npy",
                ),
                data,
            )
        else:
            np.save(
                os.path.join(
                    self.dir,
                    f"ground_truth_{self.window}_{period}.npy",
                ),
                data,
            )

    def normalize(self):
        if self.period == "train":
            self.samples = self.samples.reshape(-1, self.channels)
            self.minmax_scaler = MinMaxScaler()
            self.minmax_scaler = self.minmax_scaler.fit(self.samples)

            with open(self.minmax_scaler_path, "wb") as f:
                pickle.dump(self.minmax_scaler, f)

            self.samples = self.minmax_normalize(self.samples)

        elif self.period == "test":
            self.samples = self.samples.reshape(-1, self.channels)
            self.minmax_scaler = MinMaxScaler()
            if os.path.exists(self.minmax_scaler_path):
                with open(self.minmax_scaler_path, "rb") as f:
                    self.minmax_scaler = pickle.load(f)
            else:
                raise FileNotFoundError(
                    f"Scaler file not found at {self.minmax_scaler_path}. Please ensure the scaler is saved during training."
                )
            self.samples = self.minmax_normalize(self.samples)

        print(
            f"{self.period.capitalize()} Data Scale After Norm: [{np.min(self.samples)}, {np.max(self.samples)}]"
        )
        print(
            f"{self.period.capitalize()} Data Scale After Norm: [{np.min(self.samples)}, {np.max(self.samples)}]"
        )

    def robust_normalize(self, sq):
        d = sq.reshape(-1, self.channels)
        d = self.robust_scaler.transform(d)
        return d.reshape(-1, self.window, self.channels)

    def minmax_normalize_per_channel(self, d, channel):
        d = d.reshape(-1, 1)
        d = self.minmax_scalers[channel].transform(d)

        if self.auto_norm_type == "neg_five_to_five":
            d = normalize_to_neg_five_to_five(d)
        elif self.auto_norm_type == "neg_one_to_one":
            d = normalize_to_neg_one_to_one(d)
        elif self.auto_norm_type == "zero_to_one":
            pass

        return d.reshape(-1, self.lens)

    def minmax_normalize(self, sq):
        d = sq.reshape(-1, self.channels)
        d = self.minmax_scaler.transform(d)

        if self.auto_norm_type == "neg_five_to_five":
            d = normalize_to_neg_five_to_five(d)
        elif self.auto_norm_type == "neg_one_to_one":
            d = normalize_to_neg_one_to_one(d)
        elif self.auto_norm_type == "zero_to_one":
            pass

        return d.reshape(-1, self.window, self.channels)

    def unnormalize(self, sq):
        d = self.__unnormalize(sq.reshape(-1, self.channels))
        return d.reshape(-1, self.window, self.channels)

    def __normalize(self, rawdata):
        data = self.minmax_scaler.transform(rawdata)
        if self.auto_norm:
            if self.auto_norm_type == "neg_five_to_five":
                d = normalize_to_neg_five_to_five(d)
            elif self.auto_norm_type == "neg_one_to_one":
                d = normalize_to_neg_one_to_one(d)
            elif self.auto_norm_type == "zero_to_one":
                pass
        return data

    def __unnormalize(self, data):
        if self.auto_norm:
            if self.auto_norm_type == "neg_five_to_five":
                d = normalize_to_neg_five_to_five(d)
            elif self.auto_norm_type == "neg_one_to_one":
                d = normalize_to_neg_one_to_one(d)
        x = data
        return self.minmax_scaler.inverse_transform(x)

    @staticmethod
    def divide(data, ratio, seed=2023):
        size = data.shape[0]
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        regular_train_num = int(np.ceil(size * ratio))
        # id_rdm = np.random.permutation(size)
        id_rdm = np.arange(size)
        regular_train_id = id_rdm[:regular_train_num]
        irregular_train_id = id_rdm[regular_train_num:]

        regular_data = data[regular_train_id, :]
        irregular_data = data[irregular_train_id, :]

        # Restore RNG.
        np.random.set_state(st0)
        return regular_data, irregular_data

    def read_data(self, filepath):
        """Reads .npy files"""

        if self.period == "train":
            samples = np.load(os.path.join(filepath, "train_data.npy"))
        elif self.period == "test":
            val_data = np.load(os.path.join(filepath, "val_data.npy"))
            test_data = np.load(os.path.join(filepath, "test_data.npy"))
            samples = np.concatenate(val_data, test_data, axis=0)

        return samples

    def __imputation_mask_data__(self, seed=2023):
        masks = np.ones_like(self.samples)
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        for idx in range(self.samples.shape[0]):
            x = self.samples[idx, :, :]  # (seq_length, feat_dim) array
            mask = noise_mask(
                x,
                self.missing_ratio,
                self.mean_mask_length,
                self.style,
                self.distribution,
            )  # (seq_length, feat_dim) boolean array
            masks[idx, :, :] = mask

        if self.save2npy:
            np.save(os.path.join(self.dir, f"masking_{self.window}.npy"), masks)

        # Restore RNG.
        np.random.set_state(st0)
        return masks.astype(bool)

    def __getitem__(self, ind):

        if self.period == "train":
            if self.condition_type == "uncond":
                x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
                return torch.from_numpy(x).float()

            elif self.condition_type == "cond":
                x = self.samples[ind, :, :].copy()  # (seq_length, feat_dim) array
                t = self.samples[ind, :, :].copy()
                x[:, self.synthesis_channels] = 0  # single lead mask
                return torch.from_numpy(x).float(), torch.from_numpy(t).float()

        elif self.period == "test" or self.period == "val":
            x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
            m = self.masking[ind, :, :]  # (seq_length, feat_dim) boolean array
            return torch.from_numpy(x).float(), torch.from_numpy(m)

    def __len__(self):
        return self.num_sample


# 初始化 scalers
# self.robust_scalers = [RobustScaler() for _ in range(self.channels)]
# self.minmax_scalers = [
#     MinMaxScaler(feature_range=(-5, 5)) for _ in range(self.channels)
# ]

# 逐通道对训练集进行 MinMaxScaler 归一化
# for channel in range(self.channels):
#     channel_data = self.train_data[:, :, channel].copy()
#     channel_data = channel_data.reshape(-1, 1)

#     self.minmax_scalers[channel].fit(channel_data)

#     self.train_data[:, :, channel] = self.minmax_normalize_per_channel(
#         self.train_data[:, :, channel], channel
#     )
#     self.val_data[:, :, channel] = self.minmax_normalize_per_channel(
#         self.val_data[:, :, channel], channel
#     )
#     self.test_data[:, :, channel] = self.minmax_normalize_per_channel(
#         self.test_data[:, :, channel], channel
#     )
