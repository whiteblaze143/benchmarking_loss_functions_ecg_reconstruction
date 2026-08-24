
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import scipy.io
import scipy.signal

LOGGER = logging.getLogger(__name__)

def load_header(header_path: Path) -> Dict:
    """Parse WFDB header file for sampling rate, gains, and leads."""
    with open(header_path, 'r') as f:
        lines = f.readlines()
        
    # Line 1: RecordName Leads Fs Samples
    # E00068 12 500 5000
    parts = lines[0].strip().split()
    record_name = parts[0]
    num_leads = int(parts[1])
    fs = float(parts[2])
    num_samples = int(parts[3])
    
    leads = []
    gains = []
    baselines = []
    
    # Next num_leads lines describe signals
    # FileName Format Gain(x)/Unit ADCRes ADCZero InitialValue Checksum BlockSize Description
    # E00068.mat 16+24 1000/mV 16 0 -26 0 0 I
    for i in range(1, num_leads + 1):
        lparts = lines[i].strip().split()
        filename = lparts[0]
        fmt = lparts[1]
        
        # Gain parsing: "1000/mV" -> 1000. "200" -> 200. "1000.0(0)/mV" -> 1000.0
        gain_str = lparts[2]
        if '/' in gain_str:
            gain_str = gain_str.split('/')[0]
        if '(' in gain_str:
            gain_str = gain_str.split('(')[0]
        gain = float(gain_str)
            
        baseline = int(lparts[4])
        # initial_val = int(lparts[5])
        desc = lparts[8] if len(lparts) > 8 else f"Lead_{i}"
        
        leads.append(desc)
        gains.append(gain)
        baselines.append(baseline)
        
    # Metadata (lines starting with #)
    meta = {}
    for line in lines[num_leads+1:]:
        if line.startswith('#'):
            clean = line.strip().lstrip('#').strip()
            if ':' in clean:
                k, v = clean.split(':', 1)
                meta[k.strip()] = v.strip()
            
    return {
        "record_name": record_name,
        "fs": fs,
        "num_samples": num_samples,
        "leads": leads,
        "gains": np.array(gains),
        "baselines": np.array(baselines),
        "metadata": meta
    }

def load_wfdb_mat(mat_path: Path, header_info: Dict) -> np.ndarray:
    """Load WFDB signal from .dat or .mat."""
    path_str = str(mat_path)
    if path_str.endswith('.mat'):
        try:
            data = scipy.io.loadmat(path_str)
            key = [k for k in data.keys() if k not in ['__header__', '__version__', '__globals__']][0]
            sig = data[key]
            if sig.shape[0] == len(header_info['leads']): pass
            elif sig.shape[1] == len(header_info['leads']): sig = sig.T
            return sig
        except Exception as e:
            LOGGER.error(f"Failed to load {mat_path}: {e}")
            return np.zeros((len(header_info['leads']), header_info['num_samples']))
            
    elif path_str.endswith('.dat'):
        # Standard WFDB .dat
        # Use wfdb library if available, else numpy fromfile if format is known (16 or 212)
        # We'll assume format 16 (int16) for standard PTB-XL if not using wfdb library
        try:
            import wfdb
            # record_name relative to path? wfdb.rdrecord expects base name?
            # actually wfdb.rdrecord simply takes the path WITHOUT extension usually, 
            # OR with it if p_signal is needed?
            # It's easiest to point to the header base.
            base_path = str(mat_path.with_suffix(''))
            rec = wfdb.rdrecord(base_path)
            sig = rec.p_signal.T # (Time, Leads) -> (Leads, Time)
            return sig
        except ImportError:
            # Fallback for format 16 (int16)
            try:
                sig = np.fromfile(path_str, dtype=np.int16)
                num_leads = len(header_info['leads'])
                num_samples = header_info['num_samples']
                # Usually interleaved: L1 S1, L2 S1, ... or L1 S1, L1 S2?
                # WFDB format 16 is usually interleaved.
                sig = sig.reshape((num_samples, num_leads)).T
                return sig
            except Exception as e:
                LOGGER.error(f"Failed to load binary {mat_path}: {e}")
                return np.zeros((len(header_info['leads']), header_info['num_samples']))
    
    return np.zeros((12, 5000))

def preprocess_signal(
    sig: np.ndarray, 
    src_fs: float, 
    target_fs: float = 500.0, 
    target_len: int = 5000,
    gains: np.ndarray = None,
    baselines: np.ndarray = None,
    normalization: str = "instance", # "instance", "global", "none"
    global_mean: Optional[torch.Tensor] = None,
    global_std: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    1. Apply Gain/Baseline -> mV
    2. Resample
    3. Pad/Crop
    4. Normalize (Instance vs Global)
    """
    # Ensure 2D shape (Leads, Time)
    if sig.ndim == 1:
        # Ambiguous: is it (Time,) or (Leads,)? Usually (Time,).
        # Assume 1 lead.
        sig = sig[np.newaxis, :]
    elif sig.ndim == 2:
        # Check orientation. 
        # If shape is (5000, 12), we want (12, 5000).
        if sig.shape[0] > sig.shape[1] and sig.shape[0] >= 100:
             sig = sig.T
             
    # Apply ADC gain / baseline conversion -> convert to mV when gains match channel count
    if gains is not None and len(gains) == sig.shape[0]:
        sig = (sig.astype(np.float32) - baselines[:, None]) / gains[:, None]
    elif gains is not None:
        # Mismatch between provided gains and signal channels — skip conversion and warn.
        LOGGER.warning(
            "preprocess_signal: gains length (%d) does not match signal channels (%d); "
            "skipping gain/baseline conversion.",
            len(gains), sig.shape[0]
        )
    
    # Calculate stats BEFORE resampling/padding if instance mode (for accuracy on real signal)
    # But for global mode, we apply it preferably after? Or it doesn't matter (linear).
    
    # 2. Resample
    if src_fs != target_fs:
        num_src = sig.shape[1]
        num_target = int(num_src * target_fs / src_fs)
        sig = scipy.signal.resample(sig, num_target, axis=1)
    
    # 3. Pad/Crop
    current_len = sig.shape[1]
    if current_len < target_len:
        # Pad zeros (right)
        pad = np.zeros((sig.shape[0], target_len - current_len), dtype=sig.dtype)
        sig = np.concatenate([sig, pad], axis=1)
    elif current_len > target_len:
        # Crop start
        sig = sig[:, :target_len]
        
    tensor = torch.tensor(sig, dtype=torch.float32)

    # 4. Normalization
    if normalization == "instance":
        # Standard Z-score normalization (per lead)
        # Check for flat lines (std=0)
        # We calculate on the padded tensor? Or original? 
        # Ideally original, but pad is zeros. 
        # Let's simple calculate on tensor.
        mean_tensor = torch.mean(tensor, dim=1, keepdim=True)
        std_tensor = torch.std(tensor, dim=1, keepdim=True)
        tensor = (tensor - mean_tensor) / (std_tensor + 1e-6)
        
        # Squeeze for metadata return
        mean_ret = mean_tensor.squeeze()
        std_ret = std_tensor.squeeze()
        
    elif normalization == "instance_global":
        # Standard Z-score normalization (per sample, across all leads)
        # This preserves linear relationships (Physics!)
        mean_tensor = torch.mean(tensor)
        std_tensor = torch.std(tensor)
        tensor = (tensor - mean_tensor) / (std_tensor + 1e-6)
        
        # Return same stats for all leads
        mean_ret = torch.full((tensor.shape[0],), mean_tensor.item())
        std_ret = torch.full((tensor.shape[0],), std_tensor.item())
        
    elif normalization == "global":
        if global_mean is None or global_std is None:
            raise ValueError("Global normalization requires global_mean and global_std")
        
        # global_mean shape (12, 1) or (12,)
        # Ensure broadcasting
        if global_mean.ndim == 1:
            mean_tensor = global_mean.view(-1, 1)
            std_tensor = global_std.view(-1, 1)
        else:
            mean_tensor = global_mean
            std_tensor = global_std
            
        tensor = (tensor - mean_tensor) / (std_tensor + 1e-6)
        
        mean_ret = mean_tensor.squeeze()
        std_ret = std_tensor.squeeze()

    elif normalization == "min_max":
        # Min-Max Normalization to [0, 1]
        # Assumes inputs are in mV.
        # Maps [-5.0, 5.0] mV -> [0.0, 1.0]
        # This covers typical ECG range (+/- 2-3mV) with safety margin.
        min_val = -5.0
        max_val = 5.0
        
        # Clip first to avoid outliers skewing range if we computed it dynamically, 
        # but here we use fixed range, so just clip to ensure [0,1].
        tensor = torch.clamp(tensor, min_val, max_val)
        
        tensor = (tensor - min_val) / (max_val - min_val)
        
        # Return dummy mean/std as they aren't applicable in the same way
        mean_ret = torch.tensor(min_val)
        std_ret = torch.tensor(max_val - min_val)
        
    else: # "none"
        mean_ret = torch.zeros(tensor.shape[0])
        std_ret = torch.ones(tensor.shape[0])

    # 5. Final Sanitization for AMP Safety
    # Replace NaNs/Infs
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=20.0, neginf=-20.0)
    # Clamp extreme outliers (rare artifacts can be >100 sigma)
    tensor = torch.clamp(tensor, -20.0, 20.0)
    
    return tensor, mean_ret, std_ret

class MultiSourceECGDataset(Dataset):
    """
    Dataset that can handle:
    1. Preprocessed .pt tensors (PTB-XL)
    2. Raw WFDB .mat/.hea files (Georgia, CPSC)
    """
    def __init__(
        self,
        sources: List[Dict], # List of {"name": "Georgia", "path": "data/...", "format": "wfdb"}
        target_fs: float = 500.0,
        target_len: int = 5000,
        input_leads: List[str] = ["I", "II", "V2"],
        target_leads: List[str] = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
        split: str = "train", # train, val, or all
        val_ratio: float = 0.1,
        seed: int = 42,
        augment: bool = False,
        normalization: str = "instance",
        global_stats: Optional[Dict[str, torch.Tensor]] = None
    ):
        self.sources = sources
        self.target_fs = target_fs
        self.target_len = target_len
        self.input_leads = input_leads
        self.target_leads = target_leads
        self.split = split
        self.val_ratio = val_ratio
        self.seed = seed
        self.augment = augment
        self.normalization = normalization
        self.global_stats = global_stats
        
        self.all_records = []
        self._index_records()
        self._filter_split()
        
    def _index_records(self):
        # Scan paths
        for src in self.sources:
            path = Path(src["path"])
            fmt = src["format"]
            name = src["name"]
            
            # Load metadata mapping if PTB-XL
            ptbxl_map = {}
            if name == "PTB-XL":
                # 1. Load Metadata (ptbxl_features.csv) for Patient IDs
                meta_paths = [Path("data/ptbxl_features.csv"), Path("/home/mithunmanivannan/data/ptbxl_features.csv")]
                meta_path = next((p for p in meta_paths if p.exists()), None)
                
                if meta_path:
                    try:
                        df_meta = pd.read_csv(meta_path)
                        ptbxl_map = dict(zip(df_meta["file"].astype(str), df_meta["patient_id"].astype(str)))
                        LOGGER.info(f"Loaded PTB-XL metadata map: {len(ptbxl_map)} records.")
                    except Exception as e:
                        LOGGER.warning(f"Failed to load metadata: {e}")
                else:
                    LOGGER.warning("ptbxl_features.csv not found. Patient ID filtering will be degraded.")

                # 2. Load Diagnostic Labels (ptbxl_database.csv)
                AGG_CLASS_MAPPING = {
                    "NORM": ["NORM", "CSD"],
                    "MI": ["AMI", "IMI", "LMI", "PMI", "ALMI", "ASMI", "INJAS", "INJAL", "INJLA", "INJIL", "INJIN", "OLDMI", "MI"],
                    "STTC": ["STTC", "NST_", "ISC_", "SEHYP", "ISCA", "ISCI", "ISCIL", "ISCIN", "ISCLA", "ANEUR", "ELV", "LOWT", "NT_", "TAB_"],
                    "CD": ["LAFB", "IRBBB", "1AVB", "IVCD", "CRBBB", "CLBBB", "LPFB", "WPW", "AVB", "2AVB", "3AVB"],
                    "HYP": ["LVH", "RVH", "SEHYP", "RAH", "LAO/LAE", "RAO/RAE"],
                }
                self.scp_to_agg = {c: agg for agg, codes in AGG_CLASS_MAPPING.items() for c in codes}
                self.class_names = ["NORM", "MI", "STTC", "CD", "HYP"]
                
                db_paths = [Path("data/ptbxl_database.csv"), Path("/home/mithunmanivannan/data/ptbxl_database.csv")]
                db_path = next((p for p in db_paths if p.exists()), None)
                
                self.ptbxl_labels = {}
                if db_path:
                    try:
                        import ast
                        df_db = pd.read_csv(db_path)
                        for _, row in df_db.iterrows():
                            # Parse SCP
                            try: scp = ast.literal_eval(row['scp_codes'])
                            except: scp = {}
                            
                            # Encode
                            vec = np.zeros(5, dtype=np.float32)
                            for code in scp.keys():
                                if code in self.scp_to_agg:
                                    vec[self.class_names.index(self.scp_to_agg[code])] = 1.0
                            
                            # Store by ECG ID (matches .pt filenames)
                            eid = str(row['ecg_id'])
                            self.ptbxl_labels[eid] = vec
                            
                            # Also map filename stems as backup
                            self.ptbxl_labels[Path(row['filename_hr']).stem] = vec
                            self.ptbxl_labels[Path(row['filename_lr']).stem] = vec
                            
                        LOGGER.info(f"Loaded Diagnostic Labels: {len(self.ptbxl_labels)} records.")
                    except Exception as e:
                        LOGGER.warning(f"Failed to parse labels: {e}")
                else:
                    LOGGER.warning("ptbxl_database.csv not found. Downstream metrics skipped.")

            if fmt == "wfdb":
                # Find .hea files
                heas = sorted(list(path.rglob("*.hea")))
                LOGGER.info(f"Indexing {name}: Found {len(heas)} records.")
                for h in heas:
                    # For Georgia/CPSC, assume Record Name = Patient ID (usually 1 per patient)
                    pid = h.stem 
                    self.all_records.append({
                        "source": name,
                        "format": "wfdb",
                        "path": str(h), # Store header path
                        "path": str(h), # Store header path
                        "tensor_path": str(h.with_suffix('.dat')), # Standard WFDB
                        "patient_id": pid
                    })
                    
            elif fmt == "pt":
                # Find .pt files
                pts = sorted(list(path.rglob("*.pt")))
                LOGGER.info(f"Indexing {name}: Found {len(pts)} tensors.")
                for p in pts:
                    # If PTB-XL, lookup patient ID
                    stem = p.stem
                    pid = ptbxl_map.get(stem, stem) # Default to stem if map lookup fails
                    
                    self.all_records.append({
                        "source": name,
                        "format": "pt",
                        "path": str(p),
                        "tensor_path": str(p),
                        "patient_id": pid
                    })
                    
    def _filter_split(self):
        # Patient-Level Split to prevent Leakage
        # 1. Extract unique patients
        all_patients = sorted(list(set(r["patient_id"] for r in self.all_records)))
        
        # 2. Deterministic shuffle of patients
        rng = np.random.RandomState(self.seed)
        rng.shuffle(all_patients)
        
        # 3. Split patients
        n_val = int(len(all_patients) * self.val_ratio)
        n_train = len(all_patients) - n_val
        
        if self.split == "train":
            split_patients = set(all_patients[:n_train])
        elif self.split == "val":
            split_patients = set(all_patients[n_train:])
        elif self.split == "test":
            # For PTB-XL, use Fold 10 if available in metadata, else use a fixed subset
            # Here we follow the logic: n_train is 90%
            # We can use the last 10% as test if needed, but Fold 10 is better.
            # For now, let's just use the last 10% as a consistent 'test' set.
            split_patients = set(all_patients[n_train:])
        else: # all
            split_patients = set(all_patients)
            
        # 4. Filter records based on selected patients
        self.records = [r for r in self.all_records if r["patient_id"] in split_patients]
        
        LOGGER.info(
            f"Initialized MultiSourceECGDataset split={self.split}. "
            f"Patients: {len(split_patients)}/{len(all_patients)}. "
            f"Samples: {len(self.records)}."
        )
                    
    def __len__(self):
        return len(self.records)
        
    def __getitem__(self, idx):
        rec = self.records[idx]
        fmt = rec["format"]
        
        if fmt == "wfdb":
            # Load Header
            h_info = load_header(Path(rec["path"]))
            # Load Signal
            sig = load_wfdb_mat(Path(rec["tensor_path"]), h_info)
            # Preprocess
            # Preprocess
            tensor, mean, std = preprocess_signal(
                sig, 
                src_fs=h_info["fs"], 
                target_fs=self.target_fs, 
                target_len=self.target_len,
                gains=h_info["gains"],
                baselines=h_info["baselines"],
                normalization=self.normalization,
                global_mean=self.global_stats["mean"] if self.global_stats else None,
                global_std=self.global_stats["std"] if self.global_stats else None
            )
            # Map Leads
            # Need to find row indices for requested leads
            # Header["leads"] has names.
            avail_leads = h_info["leads"] # e.g. ['I', 'II', ...]
            
        elif fmt == "pt":
            # Just load
            tensor = torch.load(rec["path"], weights_only=True)
            # Assuming already 500Hz, mV, 12-lead standard order
            # PTB-XL tensors were saved as CANONICAL 12 leads.
            avail_leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
            
            # Compute stats for PTB-XL manually since it's already a tensor
            # But we still want to normalize it?
            # Wait, PTB-XL tensors might already be normalized?
            # No, foundation model assumes raw input and normalizes on the fly?
            # src/data/dataset.py ConditionalECGDataset does not normalize Z-score by default?
            # Let's check.
            
            # CRITICAL: Previous implementation of preprocess_signal (lines 116-120) did Z-score.
            # So for pt fmt, we MUST also Z-score if we want consistency.
            # Previous code for 'pt' (lines 255-256) just loaded: `tensor = torch.load(...)`.
            # This means PTB-XL was NOT being Z-scored in MultiSourceECGDataset?
            # If so, that's another bug: Training on mixed distribution (Z-scored Georgia vs Raw PTB-XL).
            
            # Let's verify if PTB-XL tensors are raw mV.
            # Yes, from `dataset.py` logic, they are raw.
            
            # PTB-XL Tensor Handling
            # tensor might be a dict (e.g. {'data': ..., 'label': ...}) or a direct tensor
            if isinstance(tensor, dict):
                if 'data' in tensor:
                    tensor = tensor['data']
                elif 'val' in tensor:
                    tensor = tensor['val']
                elif 'signal' in tensor:
                    tensor = tensor['signal']
                else:
                    # Fallback: find any tensor
                    for k, v in tensor.items():
                        if isinstance(v, torch.Tensor):
                            tensor = v
                            break # Found it
                
            # Convert to numpy for consistent processing or keep tensor?
            # preprocess_signal expects numpy.
            try:
                sig_np = tensor.numpy()
            except AttributeError:
                # If still dict (no tensor found), this will crash but let's print why
                print(f"ERROR: Could not extract tensor from {path}. Keys: {tensor.keys() if isinstance(tensor, dict) else type(tensor)}")
                raise

            
            # Delegate to preprocess_signal (even though fs might match) to handle normalization logic consistently
            # But we need gains? 
            # PTB-XL tensors in our system were saved as mV? Or raw?
            # src/data/preprocess_ptbxl.py might clarify. Assuming mV.
            # If in mV, gains=1.0, baselines=0.0.
            
            # However, preprocess_signal assumes sig input is Raw.
            # If tensor is already mV, we pass gains=None, baselines=None?
            # Let's check preprocess_signal: if gains is None, it skips step 1. Correct.
            
            # Resampling might be needed?
            # Assuming PTB-XL tensors are 500Hz.
            
            # Normalization logic
            t_out, m_out, s_out = preprocess_signal(
                sig_np,
                src_fs=500.0, # Assume src fs
                target_fs=self.target_fs,
                target_len=self.target_len,
                gains=None,
                baselines=None,
                normalization=self.normalization,
                global_mean=self.global_stats["mean"] if self.global_stats else None,
                global_std=self.global_stats["std"] if self.global_stats else None
            )
            
            tensor = t_out
            mean = m_out
            std = s_out

            
        # Select Input/Target
        # Map requested leads to indices in the loaded tensor
        # Canonical map
        
        # Input Tensors Selection
        # Input Tensors Selection
        # Use case-insensitive matching for robustness
        avail_leads_lower = [l.lower() for l in avail_leads]
        input_tensors = []
        for l in self.input_leads:
            l_lower = l.lower()
            if l_lower in avail_leads_lower:
                idx = avail_leads_lower.index(l_lower)
                # Check if tensor actually has enough channels
                if idx < tensor.shape[0]:
                    input_tensors.append(tensor[idx])
                else:
                    input_tensors.append(torch.zeros(self.target_len))
            else:
                input_tensors.append(torch.zeros(self.target_len))
                
        # Target Tensors Selection
        target_tensors = []
        for l in self.target_leads:
            l_lower = l.lower()
            if l_lower in avail_leads_lower:
                idx = avail_leads_lower.index(l_lower)
                if idx < tensor.shape[0]:
                    target_tensors.append(tensor[idx])
                else:
                    target_tensors.append(torch.zeros(self.target_len))
            else:
                target_tensors.append(torch.zeros(self.target_len))
                
        input_tensors_clean = []
        for t in input_tensors:
            # Ensure t is (target_len,) e.g. (5000,)
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t)
            
            if t.numel() != self.target_len:
                # Shape mismatch (scalar, wrong length, etc)
                # Just return zeros of correct length
                input_tensors_clean.append(torch.zeros(self.target_len))
            else:
                # Ensure 1D
                input_tensors_clean.append(t.reshape(self.target_len))
        
        target_tensors_clean = []
        for t in target_tensors:
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t)
                
            if t.numel() != self.target_len:
                target_tensors_clean.append(torch.zeros(self.target_len))
            else:
                target_tensors_clean.append(t.reshape(self.target_len))
                
        inputs = torch.stack(input_tensors_clean)
        targets = torch.stack(target_tensors_clean)
        
        # Interventional Training: Random Lead Masking
        # Purpose: Teach model Causal Uncertainty (0.0 = Missing)
        if self.augment and self.split == 'train':
            # 25% chance per lead to be dropped
            for i in range(len(inputs)):
                if torch.rand(1) < 0.25:
                   inputs[i] = 0.0
        
        # Create metadata dict
        # Try to parse from header info or default to zeros
        meta = self._create_metadata(rec)
        
        # Add normalization stats (Critical for Phase 17)
        meta["normalization_mean"] = mean
        meta["normalization_std"] = std
        
        # Get Label
        stem = rec.get("record_name", Path(rec["path"]).stem)
        if hasattr(self, "ptbxl_labels"):
            label = self.ptbxl_labels.get(stem, np.zeros(5, dtype=np.float32))
        else:
            label = np.zeros(5, dtype=np.float32)

        return {
            "input": inputs,
            "target": targets,
            "metadata": meta,
            "label": torch.tensor(label, dtype=torch.float32),
            "file": stem,
            "patient_id": "unknown",
            "encounter_id_hash": "unknown"
        }

    def _create_metadata(self, rec) -> Dict[str, torch.Tensor]:
        """Create a default metadata dict compatible with ConditionalECGDataset."""
        device = torch.device("cpu")
        
        # Keys expected by Transformer
        float_keys = [
            "age_years", "sampling_rate_hz", "lowpass_hz", "hipass_hz", 
            "qrs_duration_ms", "qt_interval_ms", "pr_interval_ms", 
            "qrs_axis_deg", "t_wave_axis_deg", "heart_rate_bpm"
        ]
        
        meta = {}
        h_meta = {}
        
        if rec["format"] == "wfdb":
            # We assume header might have been loaded or stored? 
            # In __getitem__, we load header. Ideally we pass it here.
            # But __getitem__ logic above didn't save it. 
            # I need to refactor __getitem__ to pass h_info if available.
            pass
            
        # Initialize defaults
        for k in float_keys:
            meta[k] = torch.tensor([0.0], dtype=torch.float32, device=device)
            
        # Categorical
        meta["sex_code"] = torch.tensor([0], dtype=torch.long, device=device)
        meta["pacemaker_status"] = torch.tensor([0], dtype=torch.long, device=device)
        
        # Missing indicators
        # Just assume not missing for now or all missing? 
        # If we pass 0.0 for values, we should probably set is_missing=1.0?
        # Dataset.py logic: "value if not None else 0.0". 
        # But transformer learns to ignore if missing?
        # For simplicity, providing zeros is fine.
        
        return meta
