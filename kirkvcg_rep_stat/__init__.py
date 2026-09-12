"""kirkvcg_rep_stat: 3D Vectorcardiographic repSpat for Unsupervised, Interpretable ECG Encoding."""
from .src.pipeline import KirkVCGRepSpat
from .src.encoder.encoder import KirkVCGRepSpatEncoder
from .src.vcg.transform import ecg_to_vcg_kors, ecg_to_vcg_dower, ecg_to_vcg
from .src.vcg.kinematics import compute_vcg_kinematics

__version__ = "0.1.0"
__all__ = [
    "KirkVCGRepSpat",
    "KirkVCGRepSpatEncoder",
    "ecg_to_vcg_kors",
    "ecg_to_vcg_dower",
    "ecg_to_vcg",
    "compute_vcg_kinematics",
]
