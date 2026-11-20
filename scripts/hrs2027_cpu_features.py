"""CPU-only workers keep signal processing parallel without loading GPU models."""
from scripts.hrs2027_protocol import extract_features

_anchors = None


def initialize(anchors):
    global _anchors
    _anchors = anchors


def prepare_window(payload):
    ecg,fs=payload
    return extract_features(ecg,fs,_anchors,return_cells=True)
