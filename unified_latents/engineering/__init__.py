"""Organized ECG engineering models, trainers, evaluators, and utilities."""

from __future__ import annotations


def __getattr__(name: str):
    if name == "UL_ConditionalBridge":
        from .models.ul_ecg import UL_ConditionalBridge

        return UL_ConditionalBridge
    raise AttributeError(name)


__all__ = ["UL_ConditionalBridge"]
