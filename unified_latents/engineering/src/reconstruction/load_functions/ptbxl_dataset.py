import os
import pandas as pd
import numpy as np
import torch
import wfdb
from torch.utils.data import Dataset
from scipy.signal import resample

# Mason 12-lead: normalization and constants from single source (third_party/ecg_reconstruction)
from src.reconstruction.util_functions.mason_12lead import (
    normalize_mason,
    MASON_INPUT_LIMB_V3,
)

class PTBXLDataset(Dataset):
    """
    PTB-XL Dataset for 12-lead reconstruction.

    - Default output units: physical millivolts (`mV`). When `use_mason_scaling=True`
      the dataset returns Mason-normalized signals in [0, 1].
    - Mason lead-order is preserved (I, II, III, aVR, aVL, aVF, V1..V6).
    - Attributes:
        - output_units: string, either `'mV'` or `'mason'` describing units returned by __getitem__.
    - Parameters:
        - target_fs: desired sampling rate (default 500 Hz)
        - use_mason_scaling: when True returns Mason-normalized signals in [0,1]
    """
    def __init__(
        self,
        root_dir,
        csv_file,
        split='train',
        target_fs=500,
        use_mason_scaling=False,
        input_lead_indices=None,
    ):
        import logging
        self.LOGGER = logging.getLogger(__name__)

        self.root_dir = root_dir
        self.df = pd.read_csv(csv_file)
        self.target_fs = target_fs
        self.use_mason_scaling = use_mason_scaling
        self.input_lead_indices = list(MASON_INPUT_LIMB_V3 if input_lead_indices is None else input_lead_indices)
        # Public attribute indicating units returned by __getitem__: 'mV' or 'mason'
        self.output_units = 'mason' if self.use_mason_scaling else 'mV'
        
        # Load SCP statement groups for strict paper alignment
        scp_statements_path = os.path.join(os.path.dirname(csv_file), "scp_statements.csv")
        self.scp_statements = pd.read_csv(scp_statements_path, index_col=0) if os.path.exists(scp_statements_path) else None
        
        # Mason-Style Split loading (Absolute Fidelity)
        map_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data", "Feature_map", "Dataset")
        split_map_file = os.path.join(map_path, f"{split}_map.pkl")
        
        if os.path.exists(split_map_file):
            import pickle
            with open(split_map_file, 'rb') as f:
                target_ids = [int(i) for i in pickle.load(f)]
            self.df = self.df[self.df.ecg_id.isin(target_ids)]
            print(f"Loaded {len(self.df)} records from Mason-style {split} split map.")
        else:
            # Fallback to strat_fold
            if split == 'train':
                self.df = self.df[self.df.strat_fold <= 8]
            elif split == 'val':
                self.df = self.df[self.df.strat_fold == 9]
            elif split == 'test':
                self.df = self.df[self.df.strat_fold == 10]
        
        # Strict Mason Data Rejection (Phase 44 Compliance)
        # 1. Reject Age < 18
        # 2. Reject "poor data quality" keywords in reports
        # 3. Waveform flatness is checked in __getitem__ (as per Mason's process_dataset.py)
        
        poor_quality_keywords = [
            'poor data quality',
            'low technical quality',
            'questionable change',
            'no ecg analysis possible',
            'lead reversal',
            'undetermined rhythm precludes rhythm comparison'
        ]
        
        # Apply filters
        mask = (self.df.age >= 18)
        
        def is_clean_report(report):
            if not isinstance(report, str): return True
            report_lower = report.lower()
            return not any(kw in report_lower for kw in poor_quality_keywords)
            
        mask = mask & self.df.report.apply(is_clean_report)
        initial_count = len(self.df)
        self.df = self.df[mask].reset_index(drop=True)
        print(f"Mason-Style Filter: Rejected {initial_count - len(self.df)} records (Age < 18 or Poor Quality Report)")

        # Restore missing assignments
        self.records = self.df.filename_hr.values # Update records list
        self.scp_codes = self.df.scp_codes.values
        self.ages = self.df.age.values
        self.sexes = self.df.sex.values

        # Diagnostic Superclasses for Balanced Sampling
        self.superclass_map = {
            'NORM': 'NORM', 'MI': 'MI', 'STTC': 'STTC', 'CD': 'CD', 'HYP': 'HYP'
        }
        self.labels = self._extract_superclasses()
        
        # Strict MIDT-ECG (Huang et al.) Label Maps
        self.diag_list = self.scp_statements[self.scp_statements.diagnostic == 1.0].index.tolist() if self.scp_statements is not None else []
        self.form_list = self.scp_statements[self.scp_statements.form == 1.0].index.tolist() if self.scp_statements is not None else []
        self.rhythm_list = self.scp_statements[self.scp_statements.rhythm == 1.0].index.tolist() if self.scp_statements is not None else []

    def _get_disentangled_labels(self, idx):
        """[STRICT MIDT-ECG] Extracts multi-modal structured attributes."""
        import ast
        scp_dict = ast.literal_eval(str(self.scp_codes[idx])) # Ensure string
        
        # 1. Age Bins [12, 17, 34, 54, 74] -> 6 Bins
        age = self.ages[idx]
        if np.isnan(age): age_bin = 0
        elif age < 12: age_bin = 0
        elif age < 17: age_bin = 1
        elif age < 34: age_bin = 2
        elif age < 54: age_bin = 3
        elif age < 74: age_bin = 4
        else: age_bin = 5
        
        # 2. Gender (0/1)
        sex = int(self.sexes[idx]) if not np.isnan(self.sexes[idx]) else 0
        
        # 3. Clinical Group Multi-Hot
        def get_multi_hot(codes, full_list):
            out = np.zeros(len(full_list))
            for i, c in enumerate(full_list):
                if c in codes: out[i] = 1.0
            return out

        diag_vec = get_multi_hot(scp_dict, self.diag_list)
        form_vec = get_multi_hot(scp_dict, self.form_list)
        rhythm_vec = get_multi_hot(scp_dict, self.rhythm_list)
        
        return {
            'age_bin': age_bin,
            'sex': sex,
            'diag': torch.from_numpy(diag_vec).float(),
            'form': torch.from_numpy(form_vec).float(),
            'rhythm': torch.from_numpy(rhythm_vec).float()
        }

    def _extract_superclasses(self):
        import ast
        labels = []
        for code_str in self.scp_codes:
            codes = ast.literal_eval(code_str)
            assigned = 'NORM'
            for pref in ['MI', 'STTC', 'CD', 'HYP']:
                if pref in codes:
                    assigned = pref
                    break
            labels.append(assigned)
        return labels

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rel_path = self.records[idx]
        abs_path = os.path.join(self.root_dir, rel_path)

        # Load WFDB (500Hz) -- PTB-XL records are already in mV
        data, header = wfdb.rdsamp(abs_path)  # [samples, 12]
        sampling_rate = header['fs']

        # Resample to target_fs (default 500 Hz)
        if sampling_rate != self.target_fs:
            num_samples = int(len(data) * self.target_fs / sampling_rate)
            data = resample(data, num_samples, axis=0)

        # Raw target in mV (standardize order first)
        target_mv = torch.from_numpy(data.T).float()
        
        # Standardize to Mason/HuBERT order: I, II, III, aVR, aVL, aVF, V1-V6
        perm = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        target_mv = target_mv[perm, :]

        # Input view for the caller. Historically this was Mason I/III/V3, but
        # the engineering cNVAE trainer now passes the observed-lead basis
        # explicitly so the data contract matches the training regime.
        x_raw = target_mv[self.input_lead_indices, :].clone()

        # STRICT MASON CLEANING (Phase 21 Stability)
        # Identify "flat" leads (amplitude < 0.01 mV in 1st or 4th quarter)
        # If any lead in the target is flat, recursively pick another sample.
        # This matches the Mason Team's rejection criteria in process_dataset.py.
        T = target_mv.shape[1]
        mask1 = torch.max(torch.abs(target_mv[:, :T//4]), dim=1)[0] < 0.01
        mask2 = torch.max(torch.abs(target_mv[:, 3*T//4:]), dim=1)[0] < 0.01
        nan_mask = torch.isnan(target_mv).any()
        if mask1.any() or mask2.any() or nan_mask:
            # Recursive call with a different index to find a "clean" sample
            return self.__getitem__((idx + 1) % len(self))

        # 4. Strictly Structured Disentangled Labels (Huang et al. Parity)
        disentangled = self._get_disentangled_labels(idx)

        # Optionally return Mason-normalized signals in [0,1] when requested by callers
        if self.use_mason_scaling:
            try:
                # normalize_mason maps mV -> [0,1]
                target_norm = normalize_mason(target_mv)
                x_norm = normalize_mason(x_raw)
                return x_norm, target_norm, disentangled
            except Exception as exc:
                self.LOGGER.exception("PTBXLDataset: failed to apply Mason normalization")
                raise RuntimeError(
                    "PTBXLDataset Mason normalization failed; refusing to fall back to raw mV "
                    f"for idx={idx}, path={abs_path}"
                ) from exc

        # Default return: raw mV for both input and target. Foundation Models handle their own preprocessing in forward.
        return x_raw, target_mv, disentangled
