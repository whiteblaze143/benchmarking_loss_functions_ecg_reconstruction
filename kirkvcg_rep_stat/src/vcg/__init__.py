"""VCG transformation and trajectory differential geometry module."""
from .transform import (
    ecg_to_vcg_kors,
    ecg_to_vcg_dower,
    ecg_to_vcg,
    KORS_MATRIX,
    DOWER_MATRIX,
)
from .kinematics import (
    compute_vcg_kinematics,
    compute_vcg_loop_planarity,
    compute_spatial_qrst_angle,
)

__all__ = [
    "ecg_to_vcg_kors",
    "ecg_to_vcg_dower",
    "ecg_to_vcg",
    "compute_vcg_kinematics",
    "compute_vcg_loop_planarity",
    "compute_spatial_qrst_angle",
    "KORS_MATRIX",
    "DOWER_MATRIX",
]
