import json
import os
import pickle
import re
import torch
import numpy as np
import pandas as pd

from scipy import io
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from torch.utils.data import Dataset
from models.interpretable_diffusion.model_utils import (
    normalize_to_neg_five_to_five,
    normalize_to_neg_one_to_one,
    unnormalize_to_zero_to_one,
)
from utils.masking_utils import noise_mask


class ECGDataset(Dataset):

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
        use_text=False,
    ):
        super(ECGDataset, self).__init__()
        assert period in ["train", "test"], "period must be train or test."
        if period == "train":
            assert ~(predict_length is not None or missing_ratio is not None), ""

        self.name = name
        self.seed = seed
        self.save2npy = save2npy
        self.use_text = use_text
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

        self.zscore_scaler_path = os.path.join(
            self.dir, f"train_zscore_scaler.pkl")

        # if self.use_text:
        #     self.samples, self.reports = self.read_data(data_root)
        # else:
        #     self.samples = self.read_data(data_root)
        self.samples = self.read_data(data_root)


        # self.samples = self.samples[:512]  # first 100 sample for fast sampling
        # self.test_reports = "Sinus tachycardia. Poor R wave progression - probable normal variant. Low QRS voltages in precordial leads. Borderline ECG."
        # self.reports[:32] = [self.test_reports] * 32

        self.num_sample, self.lens, self.channels = self.samples.shape
        self.sample_num_total = max(
            (self.num_sample - self.window) // self.step + 1, 0)

        self.masking = self.make_masking(period)

        print(f"{self.period.capitalize()} Data Shape: {self.samples.shape}")
        print(
            f"{self.period.capitalize()} Data Scale Before Norm: [{np.min(self.samples)}, {np.max(self.samples)}]"
        )

        if self.auto_norm:
            self.normalize()

        if self.save2npy:
            self.save_data(self.samples, self.period)

        if self.use_text:
            pass
            # with open(
            #     "utils/data_utils/prompt_template.txt", "r", encoding="utf-8"
            # ) as file:
            #     self.prompt_template = file.read()

            # Enhance reports with ckepe_prompt explanations
            # self.reports = [self.__enhance_report(report) for report in self.reports]

            # Add prompt template to reports
            # self.reports = [
            #     re.sub(
            #         r"\[REPORT CONTENT\]",
            #         f"[REPORT CONTENT]\n{report}",
            #         self.prompt_template,
            #         count=1,
            #     )
            #     for report in self.reports
            # ]

            # print(self.reports[3])

    def __enhance_report(self, report):
        """
        Enhance report by adding explanations for medical terms.
        This version checks whether the matched text has an 's' at the end and
        appends the explanation accordingly without forcing an extra 's'.

        No using now
        """
        for term, explanation in self.ckepe_prompt.items():
            pattern = r"\b({})((?<!\S)s)?\b".format(re.escape(term))

            def replacement_func(match):
                base_term = match.group(1)  # 匹配到的原始term
                plural_suffix = match.group(2) or ""  # 可能匹配到的's'
                return f"{base_term}{plural_suffix} ({explanation})"

            report = re.sub(pattern, replacement_func, report)

        return report

    def make_masking(self, period):
        """
        if self.missing_ratio is not None:
            masks = self.__imputation_mask_data__(self.seed)
        elif self.predict_length is not None:
            masks = np.ones(self.samples.shape)
            masks[:, -self.predict_length:, :] = 0
        elif self.synthesis_channels is not None:
            masks = np.ones(self.samples.shape)
            masks[:, :, self.synthesis_channels] = 0
            if self.save2npy:
                np.save(os.path.join(self.dir, f"masks_{period}.npy"), masks)
        else:
            raise NotImplementedError()

        return masks"""
        if self.synthesis_channels is None:
        # 默认保留 II, V1, V5
            keep_channels = [1, 6, 10]
            self.synthesis_channels = [i for i in range(12) if i not in keep_channels]

        masks = np.ones(self.samples.shape)
        masks[:, :, self.synthesis_channels] = 0
        if self.save2npy:
            np.save(os.path.join(self.dir, f"masks_{period}.npy"), masks)

        return masks

    

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
            self.scaler = StandardScaler()
            self.scaler = self.scaler.fit(self.samples)
            with open(self.zscore_scaler_path, "wb") as f:
                pickle.dump(self.scaler, f)
            self.samples = self.zscore_normalize(self.samples)
        elif self.period == "test":
            self.samples = self.samples.reshape(-1, self.channels)
            if os.path.exists(self.zscore_scaler_path):
                with open(self.zscore_scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
            else:
                raise FileNotFoundError(
                    f"Z-score scaler file not found at {self.zscore_scaler_path}. Please ensure the scaler is saved during training."
                )
            self.samples = self.zscore_normalize(self.samples)
        print(
            f"{self.period.capitalize()} Data Scale After Z-score Norm: [{np.min(self.samples)}, {np.max(self.samples)}]"
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

    def zscore_normalize(self, sq):
        d = sq.reshape(-1, self.channels)
        d = self.scaler.transform(d)
        return d.reshape(-1, self.window, self.channels)

    def unnormalize(self, sq):
        d = self.__unnormalize(sq.reshape(-1, self.channels))
        return d.reshape(-1, self.window, self.channels)

    def __unnormalize(self, data):
        # 反归一化: z * std + mean
        if hasattr(self, 'scaler') and self.scaler is not None:
            return self.scaler.inverse_transform(data)
        else:
            raise ValueError("Scaler not found. Cannot unnormalize data.")

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
        # Load Dataset
        if self.period == "train":
            if self.name == "PTBXL":
                samples = np.load(os.path.join(filepath, "train_data.npy"))
                self.meta_info = None  # PTBXL 没有 meta 信息
            elif self.name == "MIMIC" or self.name == "MIMIC_subset":
                # 加载所有训练数据文件
                all_samples = []
                all_meta = []
                
                # 找到所有训练数据文件
                train_files = [f for f in os.listdir(filepath) if f.startswith("train_data_") and f.endswith(".npy")]
                train_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))  # 按批次号排序
                
                for data_file in train_files:
                    # 加载信号数据
                    samples = np.load(os.path.join(filepath, data_file))
                    all_samples.append(samples)
                    
                    # 加载对应的 meta 信息
                    meta_file = data_file.replace("_data_", "_meta_").replace(".npy", ".json")
                    with open(os.path.join(filepath, meta_file), 'r') as f:
                        meta = json.load(f)
                    all_meta.extend(meta)
                
                samples = np.concatenate(all_samples, axis=0)
                self.meta_info = all_meta
                
                reports = None

        elif self.period == "test":
            if self.name == "PTBXL":
                test_data = np.load(os.path.join(filepath, "test_data.npy"))
                samples = test_data
                self.meta_info = None

            elif self.name == "MIMIC" or self.name == "MIMIC_subset":
                # 加载测试数据
                all_samples = []
                all_meta = []
                
                # 找到所有测试数据文件
                test_files = [f for f in os.listdir(filepath) if f.startswith("test_data_") and f.endswith(".npy")]
                test_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
                
                for data_file in test_files:
                    # 加载信号数据
                    samples = np.load(os.path.join(filepath, data_file))
                    all_samples.append(samples)
                    
                    # 加载对应的 meta 信息
                    meta_file = data_file.replace("_data_", "_meta_").replace(".npy", ".json")
                    with open(os.path.join(filepath, meta_file), 'r') as f:
                        meta = json.load(f)
                    all_meta.extend(meta)
                
                samples = np.concatenate(all_samples, axis=0)
                self.meta_info = all_meta

        elif self.period == "val":
            if self.name == "PTBXL":
                val_data = np.load(os.path.join(filepath, "val_data.npy"))
                samples = val_data
                self.meta_info = None

        if self.use_text:
            return samples, reports
        else:
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
            np.save(os.path.join(
                self.dir, f"masking_{self.window}.npy"), masks)

        # Restore RNG.
        np.random.set_state(st0)
        return masks.astype(bool)

    def get_meta_info(self, idx):
        """获取指定索引的meta信息"""
        if self.meta_info is None:
            return None
        return self.meta_info[idx]
    
    def get_all_subject_ids(self):
        """获取所有subject_id列表"""
        if self.meta_info is None:
            return None
        return [meta.get('subject_id', None) for meta in self.meta_info]
    
    def get_all_study_ids(self):
        """获取所有study_id列表"""
        if self.meta_info is None:
            return None
        return [meta.get('study_id', None) for meta in self.meta_info]

    def __getitem__(self, ind):

        if self.period == "train":
            if self.condition_type == "uncond":
                x = self.samples[ind, :, :]  # (seq_length, feat_dim) array

                if self.use_text:
                    report = self.reports[ind]
                    return (
                        torch.from_numpy(x).float(),
                        torch.from_numpy(report).float(),
                    )
                else:
                    return torch.from_numpy(x).float()

            elif self.condition_type == "cond":
                # (seq_length, feat_dim) array
                x = self.samples[ind, :, :].copy()
                t = self.samples[ind, :, :].copy()
                m = self.masking[ind, :, :].copy()

                # 12-lead training, do not use single lead mask
                # x[:, self.synthesis_channels] = 0  # single lead mask
                if self.use_text:
                    report = self.reports[ind]
                    return (
                        (torch.from_numpy(x).float(),
                         torch.from_numpy(report).float()),
                        torch.from_numpy(t).float(),
                        torch.from_numpy(m).float(),
                    )
                else:
                    return (
                        torch.from_numpy(x).float(),
                        torch.from_numpy(t).float(),
                        torch.from_numpy(m).float(),
                    )

        elif self.period == "test" or self.period == "val":
            x = self.samples[ind, :, :].copy()  # (seq_length, feat_dim) array
            # (seq_length, feat_dim) boolean array
            m = self.masking[ind, :, :].copy()

            if self.use_text:
                report = self.reports[ind]
                return (torch.from_numpy(x).float(), report), torch.from_numpy(
                    m
                ).float()
            else:
                return torch.from_numpy(x).float(), torch.from_numpy(m).float()

    def __len__(self):
        return self.num_sample
