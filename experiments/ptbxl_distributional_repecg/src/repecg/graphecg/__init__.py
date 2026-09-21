"""
GraphECG integration package for PTB-XL benchmark.
"""
from graphECG_author_code.graph import ECGGraphBuilder, ALL_ELECTRODES, LEAD_DEFINITIONS, LEAD_ORDER
from graphECG_author_code.model import GraphECG

__all__ = ["ECGGraphBuilder", "GraphECG", "ALL_ELECTRODES", "LEAD_DEFINITIONS", "LEAD_ORDER"]
