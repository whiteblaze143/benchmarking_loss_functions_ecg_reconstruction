"""Geometry-Aware SetOperator Package (NEF-Net Inspired Extension)."""

from repecg.geometry_setoperator.geometry_features import (
    vech_sqrt2,
    gamma,
    compute_operator_gram,
)
from repecg.geometry_setoperator.geometry_attention import (
    GeometryMultiheadAttention,
    GeometryTransformerEncoderLayer,
    GeometryTransformerEncoder,
)
from repecg.geometry_setoperator.geometry_setoperator import (
    GeometryOperatorSetModel,
    EchoNextGeometrySetOperator,
)

__all__ = [
    "vech_sqrt2",
    "gamma",
    "compute_operator_gram",
    "GeometryMultiheadAttention",
    "GeometryTransformerEncoderLayer",
    "GeometryTransformerEncoder",
    "GeometryOperatorSetModel",
    "EchoNextGeometrySetOperator",
]
