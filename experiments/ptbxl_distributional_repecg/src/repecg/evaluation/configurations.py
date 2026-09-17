from dataclasses import dataclass
import numpy as np
from typing import Tuple, Optional

@dataclass(frozen=True)
class ECGConfiguration:
    name: str
    observed_leads: Tuple[str, ...] = ()
    operators: Optional[np.ndarray] = None
    missing_mode: str = "native"

LEAD_NAMES = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")

STANDARD_CONFIGS = {
    "12lead": ECGConfiguration("12lead", observed_leads=LEAD_NAMES),
    "8independent": ECGConfiguration("8independent", observed_leads=("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")),
    "6limb": ECGConfiguration("6limb", observed_leads=("I", "II", "III", "aVR", "aVL", "aVF")),
    "3lead": ECGConfiguration("3lead", observed_leads=("I", "II", "III")),
    "2lead": ECGConfiguration("2lead", observed_leads=("I", "II")),
    "1lead_I": ECGConfiguration("1lead_I", observed_leads=("I",)),
    "1lead_II": ECGConfiguration("1lead_II", observed_leads=("II",)),
    "ICM_V3minusV2": ECGConfiguration("ICM_V3minusV2", observed_leads=("ICM",)),  # Special handling for ICM
}
