
import pandas as pd
import torch
import ast
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import logging
from src.reconstruction.load_functions.multi_source_dataset import MultiSourceECGDataset

LOGGER = logging.getLogger(__name__)

# PTB-XL Superclasses
# NORM: Normal ECG
# MI: Myocardial Infarction
# STTC: ST/T Change
# CD: Conduction Disturbance
# HYP: Hypertrophy
AGGREGATION_CLASS_MAP = {
    "NORM": "NORM",
    "MI": "MI",
    "STTC": "STTC",
    "CD": "CD",
    "HYP": "HYP"
}

class DiagnosticECGDataset(MultiSourceECGDataset):
    """
    Extensions of MultiSourceECGDataset for classification.
    Returns (Signal, Label).
    Target Labels: [NORM, MI, STTC, CD, HYP] (Multi-hot)
    """
    def __init__(
        self,
        ptbxl_csv_path: str,
        sources: List[Dict],
        input_leads: List[str] = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"], # Use all 12 by default for training classifier
        target_fs: float = 500.0,
        target_len: int = 5000,
        split: str = "train",
        val_ratio: float = 0.1,
        seed: int = 42
    ):
        # Initialize parent
        # We pass target_leads same as input_leads to extract the full 12-lead tensor
        self.ptbxl_df = pd.read_csv(ptbxl_csv_path)
        
        # Pre-process labels
        self.ptbxl_df['scp_codes'] = self.ptbxl_df['scp_codes'].apply(ast.literal_eval)
        
        # Mapping from benchmarks (Strodthoff et al.)
        self.diagnostic_class_map = {} # Code -> Superclass
        # Hardcoding common ones for robustness if file missing
        # MI
        for c in ['AMI', 'IMI', 'LMI', 'PMI', 'ASMI', 'ILMI', 'ALMI', 'INJAS', 'INJAL', 'INJIL', 'INJLA']: self.diagnostic_class_map[c] = 'MI'
        # NORM
        self.diagnostic_class_map['NORM'] = 'NORM'
        # CD (Conduction Disturbance)
        for c in ['LAFB', 'IRBBB', '1AVB', 'IVCD', 'CRBBB', 'CLBBB', 'LPFB', 'WPW', '2AVB', '3AVB']: self.diagnostic_class_map[c] = 'CD'
        # STTC (ST/T Change)
        for c in ['STTC', 'NST_', 'ISC_', 'SEHYP', 'ISCA', 'ISCI', 'ISCL', 'ISCIL']: self.diagnostic_class_map[c] = 'STTC'
        # HYP (Hypertrophy)
        for c in ['LVH', 'LAO/LAE', 'RVH', 'RAO/RAE']: self.diagnostic_class_map[c] = 'HYP'
        
        self.classes = ["NORM", "MI", "STTC", "CD", "HYP"]
        
        self.stem_to_label = {}
        for idx, row in self.ptbxl_df.iterrows():
            # Use ecg_id as string for stable key (e.g. 1 -> "1")
            stem = str(row['ecg_id'])
            
            # Extract label
            codes = row['scp_codes']
            label_vec = np.zeros(5, dtype=np.float32)
            
            for code, likelihood in codes.items():
                if likelihood >= 100.0: 
                    superclass = self.diagnostic_class_map.get(code)
                    if superclass:
                        idx_cls = self.classes.index(superclass)
                        label_vec[idx_cls] = 1.0
                        
            self.stem_to_label[stem] = label_vec

        # Initialize parent
        super().__init__(
            sources=sources,
            target_fs=target_fs,
            target_len=target_len,
            input_leads=input_leads,
            target_leads=input_leads, # Redundant but safe
            split=split,
            val_ratio=val_ratio,
            seed=seed
        )
        
            
    def __getitem__(self, idx):
        # Get signal from parent
        data_dict = super().__getitem__(idx)
        
        # Get label
        # Filename from parent: data_dict['file']
        # Wait, parent 'file' logic:
        # "file": rec.get("record_name", Path(rec["path"]).stem)
        fname = data_dict['file']
        
        label = self.stem_to_label.get(fname)
        if label is None:
            # Maybe it's a raw filename without _hr?
            # Or maybe this record is not in our csv (e.g. Georgia dataset).
            # For Georgia, we don't have labels loaded here.
            # Return zeros or ignore? 
            # For Training Diagnostic Classifier, we ONLY use PTB-XL records that have labels.
            # We should filter this in __init__.
            label = np.zeros(5, dtype=np.float32)

        return {
            "signal": data_dict["input"], # Standard 12-lead (if input_leads=All)
            "label": torch.tensor(label, dtype=torch.float32),
            "file": fname
        }

    def _filter_split(self):
        super()._filter_split()
        # Additional Step: Remove records that don't have labels (e.g. Georgia or missing map)
        # Accessing self.ptbxl_df here might be needed.
        # Ideally, we iterate and check self.stem_to_label.
        
        # We can't do this easily before __init__ finishes because stems are computed later?
        # No, stems are in self.stem_to_label.
        
        # Let's filter self.records
        valid_records = []
        for r in self.records:
            stem = Path(r["path"]).stem
            if stem in self.stem_to_label:
                valid_records.append(r)
                
        self.records = valid_records
        LOGGER.info(f"Filtered Diagnostic Dataset: {len(self.records)} records with labels.")

